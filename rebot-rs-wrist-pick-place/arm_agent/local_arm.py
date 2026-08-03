"""In-process RS-arm backend used by ``arm_safe``.

This owns the CAN connection directly from the ``rebot_rs`` interpreter.  It uses
the existing reBotArm controller and Pinocchio IK, but deliberately does not start
FastAPI or require ``rebot_arm_service`` to be running.
"""
from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = ROOT / "reBotArm_control_py_fresh"
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import (
    get_end_effector_frame_id, joint_to_pose, load_robot_model, pad_q_for_model, pos_rot_to_se3,
)
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik_with_retry

# Gripper limits come from arm_agent.gripper_model, which holds the angles MEASURED
# on this gripper (close 22.92 deg, open 141.13 deg) in radians. Commands target
# those angles directly; there is no percentage-of-travel abstraction.
from . import gripper_model as _gm  # noqa: E402

GRIPPER_CLOSE_RAD = _gm.CLOSE_RAD
GRIPPER_OPEN_RAD = _gm.OPEN_RAD
# Back-compat aliases for callers that still import the old names.
OPEN_RAD = GRIPPER_OPEN_RAD
CLOSE_RAD = GRIPPER_CLOSE_RAD
# Treat "already at the close limit" as done rather than re-driving into the stop.
GRIPPER_ALREADY_CLOSED_RAD = float(os.getenv("REBOT_GRIPPER_CLOSED_EPS_RAD", "") or 0.05)
# Jaw moves are RAMPED rather than stepped. Handing the MIT loop a target far from
# the current position makes it apply kp * (large error) all at once, which snaps
# the jaws and slams a burst of saturated commands onto the CAN bus. Walking the
# target across in small increments gives the same end state, smoothly.
def _bounded_timing_env(name: str, default: float, low: float, high: float) -> float:
    try:
        requested = float(os.getenv(name, "") or default)
    except ValueError:
        requested = default
    return min(high, max(low, requested))


# The fast grasp profile may shorten these ramps modestly, but hard floors keep
# the jaws from receiving a large position error or a sparse sequence of targets.
GRIPPER_MOVE_S = _bounded_timing_env(
    "REBOT_GRIPPER_MOVE_S", 1.5, 0.8, 3.0
)
GRIPPER_STEP_S = _bounded_timing_env(
    "REBOT_GRIPPER_STEP_S", 0.05, 0.025, 0.10
)
def _joint_vlim() -> float:
    """Per-joint velocity ceiling (rad/s). Read from arm_safe so the joint ceiling
    and the cartesian pace are tuned together; imported lazily because arm_safe
    pulls this module in itself."""
    try:
        from . import arm_safe
        return float(arm_safe.JOINT_VLIM)
    except Exception:  # noqa: BLE001
        return float(os.getenv("REBOT_JOINT_VLIM", "") or 0.25)
# The bundled numerical IK can plateau just outside its combined SE(3) residual
# threshold even when the positional solution is excellent.  This acceptance
# envelope is deliberately tighter than arm_safe's 6 mm request check.
# Near-converged numerical IK remains tighter than arm_safe's independent 6 mm
# execution envelope. Hardware continuation paths have plateaued at 3.5 mm with
# only 0.0012 rad orientation error near the extended grasp boundary; 4 mm admits
# that numerically stable solution while retaining 2 mm beneath the outer gate.
LOCAL_IK_POS_TOL_M = float(
    os.getenv("REBOT_LOCAL_IK_POS_TOL_M", "") or 0.004
)
LOCAL_IK_RPY_TOL_RAD = 0.01
PATH_IK_MAX_JOINT_STEP_RAD = float(
    os.getenv("REBOT_IK_PATH_MAX_JOINT_STEP_RAD", "") or 0.25
)
HOME_POS_TOL_RAD = 0.03
HOME_VEL_TOL_RAD_S = 0.10
HOME_SETTLE_TIMEOUT_S = 2.0


class LocalArm:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._arm: RebotArm | None = None
        self._controller: RebotArmEndPose | None = None
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame = get_end_effector_frame_id(self._model)
        self._ik = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)
        self._joint_lo = np.asarray(self._model.lowerPositionLimit[:6], dtype=np.float64)
        self._joint_hi = np.asarray(self._model.upperPositionLimit[:6], dtype=np.float64)

    @property
    def connected(self) -> bool:
        return self._arm is not None and self._controller is not None

    @staticmethod
    def _verify_can_bitrate(can_iface: str) -> int:
        """Refuse an interface configured differently from this RS station."""
        expected = int(os.getenv("REBOT_CAN_BITRATE", "1000000"))
        try:
            result = subprocess.run(
                ["ip", "-details", "link", "show", can_iface],
                check=True, capture_output=True, text=True, timeout=2.0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot verify CAN bitrate for {can_iface!r}: {exc}"
            ) from exc
        match = re.search(r"\bbitrate\s+(\d+)\b", result.stdout)
        if match is None:
            raise RuntimeError(
                f"CAN interface {can_iface!r} does not report a bitrate"
            )
        actual = int(match.group(1))
        if actual != expected:
            raise RuntimeError(
                f"CAN interface {can_iface!r} is configured at {actual} bit/s; "
                f"this RS station requires {expected} bit/s. Refusing to send "
                "motor traffic at the wrong bitrate"
            )
        return actual

    @staticmethod
    def _verified_feedback(
        arm: RebotArm,
        *,
        attempts: int = 10,
        interval_s: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Require a real, fault-free reply from every configured motor.

        ``RebotArm.get_state`` substitutes zero for a missing motor state. Those
        zeros are useful for display but must never seed or authorize motion.
        Inspect the underlying per-motor reply objects before accepting a state.
        """
        last_missing = list(arm.joint_names)
        last_faults: list[str] = []
        state = (np.zeros(len(last_missing)),) * 3
        for _ in range(max(1, attempts)):
            state = arm.get_state()
            last_missing = []
            last_faults = []
            for name in arm.joint_names:
                motor_state = arm._motor_map[name].get_state()
                if motor_state is None:
                    last_missing.append(name)
                elif int(motor_state.status_code) != 0:
                    last_faults.append(
                        f"{name}=fault:{int(motor_state.status_code)}"
                    )
            arrays = tuple(np.asarray(values, dtype=np.float64)
                           for values in state)
            if not last_missing and not last_faults and all(
                np.all(np.isfinite(values)) for values in arrays
            ):
                return arrays  # type: ignore[return-value]
            time.sleep(max(0.0, interval_s))
        detail = []
        if last_missing:
            detail.append("no feedback from " + ", ".join(last_missing))
        if last_faults:
            detail.append("motor faults " + ", ".join(last_faults))
        if not detail:
            detail.append("non-finite motor state")
        raise RuntimeError(
            "motor feedback unavailable; refusing to enable or move ("
            + "; ".join(detail) + ")"
        )

    @staticmethod
    def _enable_feedback_wake(arm: RebotArm) -> None:
        """Wake RS feedback without starting a controller or sending a target."""
        arm_group = arm.groups.get("arm")
        if arm_group is None or not arm_group.enable():
            raise RuntimeError(
                "motor enable acknowledgement failed while waking feedback; "
                "refusing to start position control"
            )

    def connect(self) -> None:
        with self._lock:
            if self.connected:
                return
            can_iface = os.getenv("REBOT_CAN_IFACE", "can0")
            operstate = Path(f"/sys/class/net/{can_iface}/operstate")
            try:
                is_up = operstate.read_text().strip().lower() == "up"
            except OSError as exc:
                raise RuntimeError(f"CAN interface {can_iface!r} is unavailable") from exc
            if not is_up:
                raise RuntimeError(
                    f"CAN interface {can_iface!r} is down; bring it up before connecting the arm"
                )
            self._verify_can_bitrate(can_iface)
            cfg = ROOT / "reBotArm_control_py_fresh" / "config" / "rebotarm_rs.yaml"
            arm = RebotArm(str(cfg))
            # Read the physical pose before enabling the position controller.  A
            # fresh controller otherwise starts with a zero joint target, which is
            # unsafe for a diagnostic connection when the arm is parked elsewhere.
            arm.connect()
            enabled_for_feedback = False
            try:
                try:
                    # Some RS drives answer while disabled, so prefer the fully
                    # passive read first.
                    q, _, _ = self._verified_feedback(
                        arm, attempts=3, interval_s=0.05
                    )
                except RuntimeError:
                    # RobStride feedback is not reliable while disabled. Enabling
                    # sends no pose target; it only wakes the drives so their
                    # measured state can be read. Position control is not started
                    # until that measured pose has been latched below.
                    self._enable_feedback_wake(arm)
                    enabled_for_feedback = True
                    q, _, _ = self._verified_feedback(arm)
            except BaseException:
                # No pose target or control loop has started. If the feedback wake
                # succeeded partially, make one bounded disable attempt before
                # closing the transport.
                if enabled_for_feedback:
                    try:
                        arm.disable_all()
                    except BaseException:
                        pass
                for ctrl in arm._ctrl_map.values():
                    try:
                        ctrl.shutdown()
                    except BaseException:
                        pass
                    try:
                        ctrl.close()
                    except BaseException:
                        pass
                raise
            controller = RebotArmEndPose(arm)
            controller._q_target = np.asarray(q[:6], dtype=np.float64).copy()
            if len(q) > 6:
                controller._gripper_target = float(q[6])
            try:
                controller.start()
            except BaseException:
                try:
                    arm.stop_control_loop()
                except BaseException:
                    pass
                for ctrl in arm._ctrl_map.values():
                    try:
                        ctrl.shutdown()
                    except BaseException:
                        pass
                    try:
                        ctrl.close()
                    except BaseException:
                        pass
                raise
            # Keep local visual-servo moves deliberately slow even if the hardware
            # YAML has higher per-joint limits. Sourced from arm_safe so the joint
            # ceiling and the cartesian pace are tuned from one place.
            controller._vlim_override = np.full(6, _joint_vlim(), dtype=np.float64)
            self._arm, self._controller = arm, controller

    def freeze(self) -> dict[str, Any]:
        """Cancel trajectory generation and hold the measured joint pose.

        This is the first response to every software failure. It does not disable
        any motor and therefore does not intentionally release the arm to gravity.
        """
        with self._lock:
            arm, controller = self._require()
            controller._stop_send.set()
            sender = controller._send_thread
            if sender is not None and sender.is_alive():
                sender.join(timeout=2.0)
            q, _, _ = arm.get_state()
            held = np.asarray(q[:6], dtype=np.float64).copy()
            controller._q_target = held
            controller._qd_target = np.zeros_like(held)
            controller._moving = False
        deadline = time.monotonic() + 1.0
        state = self.state()
        velocity = [abs(float(v)) for v in state.get("joint_vel_rad_s", [])]
        while max(velocity, default=0.0) > 0.10 and time.monotonic() < deadline:
            time.sleep(0.05)
            state = self.state()
            velocity = [abs(float(v)) for v in state.get("joint_vel_rad_s", [])]
        stopped = max(velocity, default=0.0) <= 0.10
        return {
            "ok": stopped,
            "reason": "measured pose latched; motor control remains enabled"
                      if stopped else "pose target latched but joint velocity did not settle",
            "max_joint_velocity_rad_s": max(velocity, default=0.0),
            "state": state,
        }

    def disconnect(self, *, require_home: bool = True) -> dict[str, Any]:
        """Park before disabling. Refuse an elevated/unverified disconnect.

        The old implementation unconditionally stopped the control loop and called
        ``disable_all``. On a planning failure at the scan pose that removed
        holding torque and the arm fell. Disabling is now reachable only after a
        verified home state.
        """
        with self._lock:
            if not self.connected:
                return {"ok": True, "reason": "already disconnected", "homed": True}

        parked: dict[str, Any] | None = None
        if require_home:
            try:
                frozen = self.freeze()
                parked = self.home() if frozen.get("ok") else {
                    "ok": False,
                    "reason": f"freeze_not_confirmed: {frozen.get('reason')}",
                }
            except BaseException as exc:  # keep control alive; do not disable
                parked = {"ok": False, "reason": f"park_exception {exc}"}
            if not parked.get("ok"):
                message = (
                    "CRITICAL: home was not verified; refusing motor disable. "
                    "The controller is still holding the last measured pose. "
                    f"Reason: {parked.get('reason')}"
                )
                print(message, file=sys.stderr, flush=True)
                return {"ok": False, "reason": message, "homed": False,
                        "holding": True, "home": parked}

        with self._lock:
            arm, controller = self._arm, self._controller
            self._arm = self._controller = None
        try:
            # RebotArm.disconnect owns orderly loop stop, motor disable, and CAN
            # teardown. It is safe here because home was independently verified.
            assert arm is not None
            arm.disconnect()
        except BaseException as exc:
            return {"ok": False, "reason": f"disconnect_error {exc}",
                    "homed": bool(parked and parked.get("ok"))}
        return {"ok": True, "reason": "home verified, then motors disabled",
                "homed": bool(parked and parked.get("ok")) if require_home else None}

    def _require(self) -> tuple[RebotArm, RebotArmEndPose]:
        if self._arm is None or self._controller is None:
            raise RuntimeError("RS arm is not connected")
        return self._arm, self._controller

    def state(self) -> dict[str, Any]:
        with self._lock:
            arm, _ = self._require()
            q, vel, torque = arm.get_state()
            q6 = np.asarray(q[:6], dtype=np.float64)
            pos, rpy = joint_to_pose(q6)
            return {"connected": True, "joints_rad": q6.tolist(),
                    "joint_torque": np.asarray(torque[:6], dtype=float).tolist(),
                    "joint_vel_rad_s": np.asarray(vel[:6], dtype=float).tolist(),
                    "gripper_position_rad": float(q[6]) if len(q) > 6 else None,
                    "end_effector": {"position_m": [float(v) for v in pos],
                                     "rpy_rad": [float(v) for v in rpy]}}

    def solve_ik(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> dict[str, Any]:
        with self._lock:
            if self.connected:
                q_seed, _, _ = self._require()[0].get_state()
                q_seed = np.asarray(q_seed[:6], dtype=np.float64)
            else:
                q_seed = np.zeros(6, dtype=np.float64)
            q_seed = pad_q_for_model(self._model, q_seed, controlled_joints=6)
            target = pos_rot_to_se3(np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw)
            # Never use the helper's random fallback on live hardware.  A random
            # seed can find a mathematically valid but physically distant elbow
            # branch; local motion must either converge from the current joints or
            # be refused by the caller.
            result = solve_ik_with_retry(self._model, self._data, self._end_frame, target, q_seed,
                                         self._ik, max_retries=0)
            target_q = np.asarray(result.q[:6], dtype=np.float64)
            if not np.all(np.isfinite(target_q)):
                raise RuntimeError("IK produced non-finite joint values")
            if np.any(target_q < self._joint_lo) or np.any(target_q > self._joint_hi):
                raise RuntimeError("IK solution violates a URDF joint hard limit")
            pos, rpy = joint_to_pose(result.q)
            pos_error = float(np.linalg.norm(np.asarray(pos) - np.array([x, y, z])))
            rpy_error = float(np.max(np.abs(np.asarray(rpy) - np.array([roll, pitch, yaw]))))
            if not result.success and not (
                pos_error <= LOCAL_IK_POS_TOL_M and rpy_error <= LOCAL_IK_RPY_TOL_RAD
            ):
                raise RuntimeError(
                    f"IK failed (error={result.error:.3e}, position={pos_error*1000:.1f}mm, "
                    f"orientation={rpy_error:.4f}rad)"
                )
            return {"joints_rad": target_q.tolist(),
                    "end_effector": {"position_m": [float(v) for v in pos],
                                     "rpy_rad": [float(v) for v in rpy]},
                    "near_solution": not result.success,
                    "position_error_m": pos_error,
                    "orientation_error_rad": rpy_error}

    def solve_ik_path(
        self,
        poses: list[tuple[float, float, float, float, float, float]],
    ) -> list[dict[str, Any]]:
        """Deterministically solve a Cartesian path without moving hardware.

        Each waypoint is seeded from the preceding waypoint's solution. This
        models the continuous path that the controller will execute and avoids
        both the false negatives of solving every distant waypoint from the
        initial joints and the unsafe branch jumps of random-seed IK retries.
        """
        with self._lock:
            arm, _ = self._require()
            q_live, _, _ = arm.get_state()
            q_seed = pad_q_for_model(
                self._model,
                np.asarray(q_live[:6], dtype=np.float64),
                controlled_joints=6,
            )
            previous_q = np.asarray(q_live[:6], dtype=np.float64)
            solved: list[dict[str, Any]] = []
            for index, (x, y, z, roll, pitch, yaw) in enumerate(poses, 1):
                target = pos_rot_to_se3(
                    np.array([x, y, z], dtype=np.float64),
                    roll=roll, pitch=pitch, yaw=yaw,
                )
                result = solve_ik_with_retry(
                    self._model, self._data, self._end_frame, target,
                    q_seed.copy(), self._ik, max_retries=0,
                )
                target_q = np.asarray(result.q[:6], dtype=np.float64)
                if not np.all(np.isfinite(target_q)):
                    raise RuntimeError(
                        f"path waypoint {index} produced non-finite joints"
                    )
                if np.any(target_q < self._joint_lo) or \
                        np.any(target_q > self._joint_hi):
                    raise RuntimeError(
                        f"path waypoint {index} violates a URDF joint hard limit"
                    )
                joint_step = float(np.max(np.abs(target_q - previous_q)))
                if joint_step > PATH_IK_MAX_JOINT_STEP_RAD:
                    raise RuntimeError(
                        f"path waypoint {index} changes a joint by "
                        f"{joint_step:.3f}rad, beyond the "
                        f"{PATH_IK_MAX_JOINT_STEP_RAD:.3f}rad continuity cap"
                    )
                pos, rpy = joint_to_pose(result.q)
                pos_error = float(np.linalg.norm(
                    np.asarray(pos) - np.array([x, y, z])
                ))
                rpy_error = float(np.max(np.abs(
                    np.asarray(rpy) - np.array([roll, pitch, yaw])
                )))
                if not result.success and not (
                    pos_error <= LOCAL_IK_POS_TOL_M
                    and rpy_error <= LOCAL_IK_RPY_TOL_RAD
                ):
                    raise RuntimeError(
                        f"path waypoint {index} IK failed "
                        f"(error={result.error:.3e}, "
                        f"position={pos_error * 1000:.1f}mm, "
                        f"orientation={rpy_error:.4f}rad)"
                    )
                solved.append({
                    "joints_rad": target_q.tolist(),
                    "end_effector": {
                        "position_m": [float(v) for v in pos],
                        "rpy_rad": [float(v) for v in rpy],
                    },
                    "near_solution": not result.success,
                    "position_error_m": pos_error,
                    "orientation_error_rad": rpy_error,
                })
                q_seed = pad_q_for_model(
                    self._model, target_q, controlled_joints=6
                )
                previous_q = target_q
            return solved

    def move_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                  duration: float) -> dict[str, Any]:
        solution = self.solve_ik(x, y, z, roll, pitch, yaw)
        with self._lock:
            arm, controller = self._require()
            start, _, _ = arm.get_state()
            start = np.asarray(start[:6], dtype=np.float64)
            target = np.asarray(solution["joints_rad"], dtype=np.float64)
            steps = max(12, int(max(0.8, duration) * 12))
            for i in range(1, steps + 1):
                # Quintic minimum-jerk time scaling starts and ends each short
                # Cartesian segment with zero target velocity and acceleration.
                # This is materially smoother than the previous linear ramp,
                # especially when the faster profile crosses many waypoints.
                t = i / steps
                blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
                controller._q_target = (start + (target - start) * blend).copy()
                time.sleep(duration / steps)
        deadline = time.monotonic() + max(3.0, duration)
        while time.monotonic() < deadline:
            state = self.state()
            if max(abs(a - b) for a, b in zip(state["joints_rad"], target)) < 0.02:
                break
            time.sleep(0.1)
        return self.state()

    def move_pose_path(
        self,
        poses: list[tuple[float, float, float, float, float, float]],
        duration: float,
        send_freq: float = 50.0,
    ) -> dict[str, Any]:
        """Execute a fully solved Cartesian chain as one minimum-jerk path.

        ``solve_ik_path`` validates every pose, joint hard limit, and inter-pose
        joint continuity before this method sends the first target.  The global
        minimum-jerk clock then advances through the dense joint-space polyline
        without stopping at each Cartesian waypoint.
        """
        if not poses:
            return self.state()
        solutions = self.solve_ik_path(poses)
        targets = [
            np.asarray(item["joints_rad"], dtype=np.float64)
            for item in solutions
        ]
        with self._lock:
            arm, controller = self._require()
            q_live, _, _ = arm.get_state()
            points = [np.asarray(q_live[:6], dtype=np.float64), *targets]
            segment_lengths = np.asarray([
                float(np.max(np.abs(b - a)))
                for a, b in zip(points[:-1], points[1:])
            ], dtype=np.float64)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            path_length = float(cumulative[-1])
            if path_length < 1e-8:
                controller._q_target[:] = targets[-1]
                return self.state()

            # The peak derivative of 10s^3-15s^4+6s^5 is 1.875.  This lower
            # bound keeps the commanded joint path within the configured velocity
            # ceiling even at the midpoint of the minimum-jerk profile.
            velocity_duration = 1.875 * path_length / max(1e-6, _joint_vlim())
            total_s = max(float(duration), velocity_duration, 0.25)
            count = max(2, int(np.ceil(total_s * max(20.0, send_freq))))
            interval = total_s / count

            for i in range(1, count + 1):
                t = i / count
                blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
                distance = blend * path_length
                segment = min(
                    len(segment_lengths) - 1,
                    int(np.searchsorted(cumulative, distance, side="right") - 1),
                )
                local_length = float(segment_lengths[segment])
                local_fraction = (
                    1.0 if local_length < 1e-10 else
                    float(np.clip(
                        (distance - cumulative[segment]) / local_length,
                        0.0, 1.0,
                    ))
                )
                controller._q_target[:] = (
                    points[segment]
                    + (points[segment + 1] - points[segment]) * local_fraction
                )
                time.sleep(interval)
            controller._q_target[:] = targets[-1]

        deadline = time.monotonic() + 3.0
        state = self.state()
        while time.monotonic() < deadline:
            joint_error = max(
                abs(a - b)
                for a, b in zip(state["joints_rad"], targets[-1])
            )
            max_velocity = max(
                (abs(float(v)) for v in state.get("joint_vel_rad_s", [])),
                default=float("inf"),
            )
            if joint_error < 0.02 and max_velocity < 0.10:
                break
            time.sleep(0.05)
            state = self.state()
        return state

    def gripper(self, action: str, target_rad: float | None = None,
                limit_rads: tuple[float, float] | None = None,
                kp: float | None = None, kd: float | None = None,
                stop_on_contact: bool | None = None) -> dict[str, Any]:
        """Open/close the jaws, optionally to an explicit angle.

        ``target_rad`` lets a caller open only as wide as the object needs (see
        arm_agent.gripper_model). It is advisory: the clamp below still applies.

        ``limit_rads`` is a per-call (close, open) pair in RADIANS, so one skill can
        be more conservative than the measured limits without imposing that on every
        other user of this backend. Omitted => the measured limits.
        """
        if action not in {"open", "close"}:
            raise ValueError("gripper action must be open or close")
        with self._lock:
            arm, controller = self._require()
            close_rad, open_rad = _gm.limits(*(reversed(limit_rads) if limit_rads else (None, None)))
            if limit_rads:
                close_rad, open_rad = limit_rads
            requested = (float(target_rad) if target_rad is not None
                         else _gm.target_for(action, (close_rad, open_rad)))
            # Full open is negative in this B601-RS gripper frame; closing moves
            # toward +rad.  This hard clamp is applied immediately before every
            # motor target assignment, so neither end can ever be over-driven --
            # closing stops at the 5% travel mark rather than at the mechanical
            # stop, which the motor reports as a fault if pushed into.
            target = _gm.clamp_rad(requested, (close_rad, open_rad))

            # Re-commanding a close onto jaws that are already shut drives the motor
            # against its stop and faults it. If we are already at least this closed,
            # there is nothing to do.
            q_now, _, _ = arm.get_state()
            current = float(q_now[6]) if len(q_now) > 6 else None
            # There is no validated gripper force/contact signal. Never infer
            # contact from encoder stall: a stall may be the object, a mechanism
            # fault, or the hard stop. Callers must provide a depth-sized position.
            if stop_on_contact:
                return {
                    "ok": False,
                    "action": action,
                    "reason": "contact-aware closing is disabled; command an explicit "
                              "depth-sized target instead",
                    "connected": True,
                }

            # A reading well beyond the closed stop is not a closed gripper: the
            # motor has stalled into the stop or lost its zero reference. Say so
            # instead of reporting "already closed", which hides the fault AND
            # blocks the close command that might otherwise recover it.
            if current is not None and _gm.out_of_range(current, (close_rad, open_rad)):
                return {"ok": False, "action": action,
                        "position_rad": round(current, 4),
                        "position_deg": round(_gm.deg(current), 2),
                        "target_rad": round(target, 4),
                        "target_deg": round(_gm.deg(target), 2),
                        "connected": True,
                        "reason": f"gripper reads {_gm.deg(current):.2f} deg, outside the "
                                  f"limits [{_gm.deg(close_rad):.2f}, {_gm.deg(open_rad):.2f}] "
                                  f"by more than {_gm.deg(_gm.SLOP_RAD):.0f} deg -- stalled, or "
                                  "the position reference has moved. Re-measure before "
                                  "commanding it again."}
            if (action == "close" and current is not None
                    and abs(current - close_rad) <= GRIPPER_ALREADY_CLOSED_RAD):
                return {"ok": True, "action": action, "target_pos_rad": target,
                        "requested_target_pos_rad": requested,
                        "open_limit_deg": round(_gm.deg(open_rad), 2),
                        "close_limit_deg": round(_gm.deg(close_rad), 2),
                        "position_rad": round(current, 4),
                        "position_deg": round(_gm.deg(current), 2),
                        "connected": True,
                        "skipped": f"already at the close limit "
                                   f"({_gm.deg(close_rad):.2f} deg)"}

            # Optional MIT gain override for the duration of this move. The config
            # ships the gripper at kp=50, but the routine that demonstrably moves
            # these jaws uses kp=2 and the known-good RS gripper config uses kp=8 --
            # so being able to try a softer gain without editing yaml is the fastest
            # way to tell a gain problem from a comms one.
            group = getattr(controller, "_gripper_group", None)
            saved: tuple[Any, Any] | None = None
            effective_kp = kp
            effective_kd = kd
            if group is not None and (effective_kp is not None or effective_kd is not None):
                saved = (group._mit_kp.copy(), group._mit_kd.copy())
                if effective_kp is not None:
                    group._mit_kp.fill(float(effective_kp))
                if effective_kd is not None:
                    group._mit_kd.fill(float(effective_kd))
            try:
                # Ramp the target instead of jumping to it, so the jaws ease open and
                # shut rather than snapping.
                start = current if current is not None else target
                travel = abs(target - start)
                n_steps = max(1, int(GRIPPER_MOVE_S / max(1e-3, GRIPPER_STEP_S)))
                for i in range(1, n_steps + 1):
                    command = start + (target - start) * i / n_steps
                    controller.set_gripper_target(command)
                    time.sleep(GRIPPER_STEP_S)
            finally:
                if saved is not None and group is not None:
                    group._mit_kp[:], group._mit_kd[:] = saved
        time.sleep(0.4)
        q, _, _ = arm.get_state()
        position = float(q[6]) if len(q) > 6 else None
        return {"ok": True, "action": action,
                "target_rad": round(target, 4), "target_deg": round(_gm.deg(target), 2),
                "target_pos_rad": target,
                "position_deg": None if position is None else round(_gm.deg(position), 2),
                "open_limit_deg": round(_gm.deg(open_rad), 2),
                "close_limit_deg": round(_gm.deg(close_rad), 2),
                "position_rad": position, "connected": True,
                "mit_kp": None if effective_kp is None else float(effective_kp),
                "mit_kd": None if effective_kd is None else float(effective_kd),
                "contact_aware": False}

    def home(self) -> dict[str, Any]:
        with self._lock:
            arm, controller = self._require()
            q_start, _, _ = arm.get_state()
            q_start = np.asarray(q_start[:6], dtype=np.float64)
            # Recovery remains at the previously tested 2x ceiling even when a
            # normal trajectory opts into the faster 3x profile.
            max_vel = min(_joint_vlim(), 0.60)
            # safe_home uses a minimum-jerk trajectory lasting
            # 2*max_joint_error/max_vel. Its old fixed 15 s timeout was shorter
            # than a valid scan->home path at the deliberate 0.25 rad/s limit.
            trajectory_s = 2.0 * float(np.max(np.abs(q_start))) / max(1e-6, max_vel)
            confirmed = controller.safe_home(
                max_vel=max_vel,
                timeout=max(15.0, trajectory_s + 5.0),
            )
            settle_deadline = time.monotonic() + HOME_SETTLE_TIMEOUT_S
            state = self.state()
            while True:
                joints = [float(v) for v in state.get("joints_rad", [])]
                velocities = [abs(float(v)) for v in state.get("joint_vel_rad_s", [])]
                max_error = max((abs(v) for v in joints), default=float("inf"))
                max_velocity = max(velocities, default=float("inf"))
                if max_error < HOME_POS_TOL_RAD and max_velocity < HOME_VEL_TOL_RAD_S:
                    break
                if time.monotonic() >= settle_deadline:
                    break
                time.sleep(0.05)
                state = self.state()
            # safe_home's internal confirmation uses a tighter instantaneous
            # threshold and can time out a few encoder counts before zero. The
            # state sampled through the settle window is the authoritative,
            # independently measured postcondition for parking.
            measured_home = (
                max_error < HOME_POS_TOL_RAD
                and max_velocity < HOME_VEL_TOL_RAD_S
            )
            ok = bool(measured_home)
            return {
                "ok": ok,
                "reason": (
                    "home_confirmed"
                    if confirmed else
                    "home_confirmed_by_settled_state_after_controller_timeout"
                ) if ok else
                          f"home_not_confirmed max_joint_error={max_error:.4f}rad "
                          f"max_joint_velocity={max_velocity:.4f}rad/s",
                "max_joint_error_rad": max_error,
                "max_joint_velocity_rad_s": max_velocity,
                "state": state,
            }


_local: LocalArm | None = None


def get_local_arm() -> LocalArm:
    global _local
    if _local is None:
        _local = LocalArm()
        atexit.register(_local.disconnect)
    return _local
