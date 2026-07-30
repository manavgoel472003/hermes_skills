"""Single safety gate for every arm motion the visual servo makes.

Nothing in the servo talks to the arm except through here, so every move is:
  1. BOUNDS-checked against a hard workspace AABB (never out of bounds),
  2. IK-VERIFIED via /sim/ik BEFORE moving -- if IK fails, or the IK solution's
     end-effector doesn't actually reach the requested pose, the move is REFUSED
     (no motion), so the arm never lunges at an unreachable/degenerate target,
  3. STEP-CAPPED -- horizontal servo steps are limited so motion is always small
     and gentle, then re-observed,
  4. executed only via /move/pose mode="smooth" -> IK solve + joint-space
     interpolation (the service's non-erratic path), then verified it arrived.
The gripper only ever moves via /gripper, which the service hard-clamps to the
68% open cap; this module never bypasses that.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TELEOP_EXAMPLE_DIR = ROOT / "reBotArm_control_py_fresh" / "example"
if str(TELEOP_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_EXAMPLE_DIR))
import arm_teleop_constraints as TELEOP_LIMITS

# The rebot arm service (main.py / README / service_client all default to :8000).
# Overridable so every agent-side script + the MCP server share one URL.
ARM = os.getenv("REBOT_ARM_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
# ``local`` is the single-process rebot_rs path.  Set ``REBOT_ARM_BACKEND=service``
# only when deliberately using the legacy FastAPI stack.
BACKEND = os.getenv("REBOT_ARM_BACKEND", "local").strip().lower()


def _local_arm():
    if BACKEND != "local":
        return None
    from .local_arm import get_local_arm
    return get_local_arm()

# ---- workspace envelope -----------------------------------------------------------
# Two profiles, because one box cannot serve both users of this gate:
#
#   "teleop"  the exact effective box of depth_hand_follow_demo.py, including its
#             85% Y/Z safety shrink and absolute Z floor -- sized for a human
#             waving at a camera, where over-reach is the hazard.
#   "table"   (default) for deliberate table work: pick & place must actually
#             REACH the table. The teleop floor sits at z=0.183, which measured
#             hardware poses show is ~17 cm ABOVE the surface -- a descent would
#             stop in mid-air and the gripper would close on nothing. Bounds below
#             are IK-verified reachable at the working pitch and contain the poses
#             measured on the real arm (x 0.263-0.401, y -0.211..0.137, z 0.015-0.282).
#
# This AABB is only the COARSE filter. Every move is still IK-verified before it
# executes and every descent is torque-gated, so a generous box does not remove a
# safety check -- out-of-reach corners are simply refused at move time.
# Override per axis with REBOT_WS_{X,Y,Z}_{MIN,MAX}; pick a profile with
# REBOT_WORKSPACE=teleop|table.
_TELEOP_LO, _TELEOP_HI = TELEOP_LIMITS.workspace_bounds()
_PROFILES = {
    "teleop": (tuple(float(v) for v in _TELEOP_LO), tuple(float(v) for v in _TELEOP_HI)),
    "table": ((0.20, -0.24, 0.00), (0.44, 0.24, 0.30)),
}
WORKSPACE_PROFILE = os.getenv("REBOT_WORKSPACE", "table").strip().lower()
if WORKSPACE_PROFILE not in _PROFILES:
    WORKSPACE_PROFILE = "table"
_WORKSPACE_LO, _WORKSPACE_HI = _PROFILES[WORKSPACE_PROFILE]
X_MIN, Y_MIN, Z_MIN = _WORKSPACE_LO
X_MAX, Y_MAX, Z_MAX = _WORKSPACE_HI


def _ws_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


X_MIN = _ws_env("REBOT_WS_X_MIN", X_MIN)
X_MAX = _ws_env("REBOT_WS_X_MAX", X_MAX)
Y_MIN = _ws_env("REBOT_WS_Y_MIN", Y_MIN)
Y_MAX = _ws_env("REBOT_WS_Y_MAX", Y_MAX)
Z_MIN = _ws_env("REBOT_WS_Z_MIN", Z_MIN)
Z_MAX = _ws_env("REBOT_WS_Z_MAX", Z_MAX)
# The hand-follow mapper applies a 0.18 m/s slew rate at video cadence.  A
# 50 ms command window gives a conservative 9 mm maximum step for this discrete
# visual-servo path.
MAX_XY_STEP = TELEOP_LIMITS.MAX_TARGET_SPEED_M_S * 0.05
MAX_Z_STEP = TELEOP_LIMITS.MAX_TARGET_SPEED_M_S * 0.05
IK_POS_TOL = 0.006                  # IK "solved" but must reach within 6 mm
DEF_PITCH = 1.30                    # wrist pitched down
DEF_YAW = 0.0                       # wrist not rotated -> stable, predictable

# ---- motion pace ------------------------------------------------------------------
# Deliberately slow, but never jerky: each move is still a smooth IK + joint-space
# interpolation, we simply give it a longer duration so the same path is traversed
# gently. TOOL_SPEED is the cartesian pace; MIN_MOVE_S keeps very short steps (the
# 9 mm servo steps) from becoming brief twitches.
TOOL_SPEED_M_S = float(os.getenv("REBOT_ARM_SPEED_M_S", "") or 0.015)
MIN_MOVE_S = float(os.getenv("REBOT_ARM_MIN_MOVE_S", "") or 1.0)
TOOL_ANGULAR_SPEED_RAD_S = float(os.getenv("REBOT_ARM_ANGULAR_SPEED_RAD_S", "") or 0.15)
ORIENTATION_TOL_RAD = float(os.getenv("REBOT_ARM_ORIENTATION_TOL_RAD", "") or 0.03)
# Per-joint velocity ceiling handed to the RS position controller (rad/s).
JOINT_VLIM = float(os.getenv("REBOT_JOINT_VLIM", "") or 0.25)


def _move_duration(dx: float, dy: float, dz: float) -> float:
    """Seconds to allocate to a step: distance / speed, with a floor."""
    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
    return max(MIN_MOVE_S, dist / max(1e-6, TOOL_SPEED_M_S))


@dataclass
class MoveResult:
    ok: bool
    reason: str = ""
    reached: tuple[float, float, float] | None = None
    status: str = ""


def in_bounds(x: float, y: float, z: float) -> bool:
    return (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX)


def ee_pos() -> list[float]:
    local = _local_arm()
    if local is not None:
        return list(local.state()["end_effector"]["position_m"])
    s = httpx.get(f"{ARM}/state", timeout=5).json()
    s = s.get("state", s)
    return list(s["end_effector"]["position_m"])


def ee_rpy() -> list[float]:
    """Current tool orientation in the base frame.

    Relative cartesian steps should preserve this by default.  Forcing the old
    fixed down-pitch during a 5 mm jog can make IK choose a very different elbow
    branch, which is precisely the kind of large joint move we must reject.
    """
    local = _local_arm()
    if local is not None:
        return list(local.state()["end_effector"]["rpy_rad"])
    s = httpx.get(f"{ARM}/state", timeout=5).json()
    s = s.get("state", s)
    return list(s["end_effector"]["rpy_rad"])


def joint_torques() -> list[float]:
    """Per-joint torque (Nm) the arm reports -- our force feedback. A rise above
    baseline means contact or strain; used to stop descents before anything is
    over-loaded (the position-gap check can't see torque)."""
    local = _local_arm()
    if local is not None:
        return list(local.state().get("joint_torque", []))
    s = httpx.get(f"{ARM}/state", timeout=5).json()
    s = s.get("state", s)
    return list(s.get("joint_torque", []))


# Max allowed rise of ANY joint's |torque| above its at-rest baseline before we
# treat it as contact/strain and stop. Calibrated from live readings; conservative.
TORQUE_RISE_LIMIT = 0.35


def torque_exceeded(baseline: list[float], limit: float = TORQUE_RISE_LIMIT):
    """Return (exceeded, worst_joint, worst_rise) comparing live torque to baseline."""
    try:
        cur = joint_torques()
    except Exception:
        return False, -1, 0.0
    worst, wj = 0.0, -1
    for i, (c, b) in enumerate(zip(cur, baseline)):
        rise = abs(abs(c) - abs(b))
        if rise > worst:
            worst, wj = rise, i
    return worst > limit, wj, worst


def connected() -> bool:
    local = _local_arm()
    if local is not None:
        return local.connected
    try:
        s = httpx.get(f"{ARM}/state", timeout=5).json()
        return bool(s.get("state", s).get("connected", s.get("connected")))
    except Exception:
        return False


def check_ik(x, y, z, roll=0.0, pitch=DEF_PITCH, yaw=DEF_YAW) -> MoveResult:
    """Dry-run IK: does a clean solution exist that actually REACHES this pose?"""
    if not in_bounds(x, y, z):
        return MoveResult(False, f"out_of_bounds ({x:.3f},{y:.3f},{z:.3f})")
    local = _local_arm()
    if local is not None:
        try:
            sol = local.solve_ik(x, y, z, roll, pitch, yaw)
        except Exception as e:
            return MoveResult(False, f"local_ik_error {e}")
        p = sol["end_effector"]["position_m"]
        err = max(abs(p[0] - x), abs(p[1] - y), abs(p[2] - z))
        if err > IK_POS_TOL:
            return MoveResult(False, f"ik_did_not_reach (off {err*1000:.1f}mm)")
        return MoveResult(True, "ik_ok", (p[0], p[1], p[2]))
    try:
        r = httpx.post(f"{ARM}/sim/ik", json={"x": x, "y": y, "z": z,
                       "roll": roll, "pitch": pitch, "yaw": yaw}, timeout=15)
    except Exception as e:
        return MoveResult(False, f"sim_ik_error {e}")
    if r.status_code != 200:
        return MoveResult(False, f"ik_failed {r.text[:120]}")
    sol = r.json()
    p = sol.get("end_effector", {}).get("position_m")
    if p is None:
        return MoveResult(False, "ik_no_pose")
    err = max(abs(p[0] - x), abs(p[1] - y), abs(p[2] - z))
    if err > IK_POS_TOL:
        return MoveResult(False, f"ik_did_not_reach (off {err*1000:.1f}mm)")
    return MoveResult(True, "ik_ok", (p[0], p[1], p[2]))


def check_ik_path(
    waypoints: list[tuple[float, float, float]],
    roll: float,
    pitch: float,
    yaw: float,
) -> MoveResult:
    """Read-only deterministic continuation IK for a complete waypoint path."""
    if not waypoints:
        return MoveResult(True, "empty_path")
    for index, (x, y, z) in enumerate(waypoints, 1):
        if not in_bounds(x, y, z):
            return MoveResult(
                False,
                f"path waypoint {index}/{len(waypoints)} out_of_bounds "
                f"({x:.3f},{y:.3f},{z:.3f})",
            )
    local = _local_arm()
    if local is not None:
        try:
            local.solve_ik_path([
                (x, y, z, roll, pitch, yaw) for x, y, z in waypoints
            ])
        except Exception as exc:
            return MoveResult(False, f"local_path_ik_error {exc}")
        return MoveResult(
            True,
            f"path_ik_ok ({len(waypoints)} deterministic continuation solutions)",
            waypoints[-1],
        )
    # The service/simulator API has no seeded path endpoint. Retain the existing
    # conservative independent checks there.
    for index, (x, y, z) in enumerate(waypoints, 1):
        result = check_ik(x, y, z, roll, pitch, yaw)
        if not result.ok:
            return MoveResult(
                False,
                f"path waypoint {index}/{len(waypoints)}: {result.reason}",
            )
    return MoveResult(True, f"path_ik_ok ({len(waypoints)} waypoints)",
                      waypoints[-1])


def safe_move(x, y, z, *, roll: float | None = None, pitch: float | None = None,
              yaw: float | None = None,
              duration: float | None = None, dry: bool = False,
              cur: tuple[float, float, float] | None = None) -> MoveResult:
    """Bounds + IK-verify + step-cap, then a smooth (IK+joint-interp) move.

    `cur` (current EE xyz) enables step-cap enforcement; if omitted it's read live.
    Returns MoveResult; on any failure NO motion happens.
    """
    if cur is None:
        try:
            cur = tuple(ee_pos())  # type: ignore[assignment]
        except Exception as e:
            return MoveResult(False, f"state_read_failed {e}")
    # Relative moves retain the live wrist orientation.  An explicit roll/pitch/
    # yaw remains available, but callers must opt into any reorientation.
    try:
        current_rpy = ee_rpy()
    except Exception as e:
        return MoveResult(False, f"orientation_read_failed {e}")
    if roll is None or pitch is None or yaw is None:
        roll = current_rpy[0] if roll is None else roll
        pitch = current_rpy[1] if pitch is None else pitch
        yaw = current_rpy[2] if yaw is None else yaw
    dx, dy, dz = x - cur[0], y - cur[1], z - cur[2]
    eps = 1e-5  # tolerate float round-off on a value clipped exactly to the cap
    if abs(dx) > MAX_XY_STEP + eps or abs(dy) > MAX_XY_STEP + eps or abs(dz) > MAX_Z_STEP + eps:
        return MoveResult(False, f"step_too_large dx={dx*1000:.0f} dy={dy*1000:.0f} dz={dz*1000:.0f}mm")
    ik = check_ik(x, y, z, roll, pitch, yaw)
    if not ik.ok:
        return ik
    if dry:
        return MoveResult(True, "dry_ok", ik.reached)
    angular_change = max(
        abs(float(roll) - float(current_rpy[0])),
        abs(float(pitch) - float(current_rpy[1])),
        abs(float(yaw) - float(current_rpy[2])),
    )
    dur = duration if duration is not None else max(
        _move_duration(dx, dy, dz),
        angular_change / max(1e-6, TOOL_ANGULAR_SPEED_RAD_S),
    )
    local = _local_arm()
    if local is not None:
        try:
            state = local.move_pose(x, y, z, roll, pitch, yaw, dur)
            reached = tuple(state["end_effector"]["position_m"])
            reached_rpy = tuple(state["end_effector"]["rpy_rad"])
        except Exception as e:
            return MoveResult(False, f"move_error {e}")
        off = max(abs(reached[0] - x), abs(reached[1] - y), abs(reached[2] - z))
        angular_off = max(
            abs(reached_rpy[0] - roll),
            abs(reached_rpy[1] - pitch),
            abs(reached_rpy[2] - yaw),
        )
        ok = off < 0.01 and angular_off < ORIENTATION_TOL_RAD
        reason = ("arrived" if ok else
                  f"arrived_off position={off*1000:.1f}mm "
                  f"orientation={angular_off:.4f}rad")
        return MoveResult(ok, reason, reached)
    try:
        r = httpx.post(f"{ARM}/move/pose", json={"x": x, "y": y, "z": z, "roll": roll,
                       "pitch": pitch, "yaw": yaw, "mode": "smooth", "duration": dur},
                       timeout=dur + 40)
        if r.status_code != 200:
            return MoveResult(False, f"move_rejected {r.text[:120]}")
    except Exception as e:
        return MoveResult(False, f"move_error {e}")
    time.sleep(0.15)
    reached = tuple(ee_pos())
    off = max(abs(reached[0] - x), abs(reached[1] - y), abs(reached[2] - z))
    return MoveResult(off < 0.01, "arrived" if off < 0.01 else f"arrived_off {off*1000:.0f}mm", reached)


def safe_step(dx=0.0, dy=0.0, dz=0.0, *, cur=None, **kw) -> MoveResult:
    """Relative version of safe_move (used by the servo)."""
    if cur is None:
        cur = tuple(ee_pos())
    return safe_move(cur[0] + dx, cur[1] + dy, cur[2] + dz, cur=cur, **kw)


def safe_goto(x, y, z, *, cur=None, dry=False, **kw) -> MoveResult:
    """Deliberate move to a pose, broken into per-step-capped sub-moves so even a
    larger approach stays small + smooth + IK-verified at every increment (the arm
    never makes one big lunge). Stops at the first sub-step that's refused."""
    import math
    if cur is None:
        cur = tuple(ee_pos())
    dx, dy, dz = x - cur[0], y - cur[1], z - cur[2]
    n = max(1, int(math.ceil(max(abs(dx) / MAX_XY_STEP, abs(dy) / MAX_XY_STEP,
                                 abs(dz) / MAX_Z_STEP))))
    res = MoveResult(True, "noop", cur)
    for i in range(1, n + 1):
        prev = (cur[0] + dx * (i - 1) / n, cur[1] + dy * (i - 1) / n, cur[2] + dz * (i - 1) / n)
        tgt = (cur[0] + dx * i / n, cur[1] + dy * i / n, cur[2] + dz * i / n)
        res = safe_move(*tgt, cur=prev, dry=dry, **kw)
        if not res.ok:
            return res
    return res


def safe_descend(x, y, z_floor, *, step=0.004, jump_limit=0.25, settle=0.3):
    """Descend the tip straight down toward z_floor, but STOP the instant contact
    or strain shows up. Contact is detected two ways, either of which halts:
      * torque JUMP -- a sudden step-to-step rise in any joint's |torque| (gravity
        load rises gradually as the arm extends, so we watch the jump, not the
        level), or
      * position STALL -- the arm can't get down as commanded (table blocking it).
    Never pushes: on either signal it stops and reports where. Returns MoveResult
    with reached = the last safe pose."""
    cur = tuple(ee_pos())
    prev_t = joint_torques()
    if not prev_t or all(abs(float(v)) < 1e-6 for v in prev_t):
        return MoveResult(
            False,
            "torque_feedback_unavailable: descent refused",
            cur,
            "feedback_unavailable",
        )
    z = cur[2]
    last = cur
    while z - z_floor > 1e-4:
        z_cmd = max(z_floor, z - step)
        r = safe_move(x, y, z_cmd, cur=(x, y, z))
        if not r.ok:
            return MoveResult(
                False,
                f"move_refused: {r.reason}",
                last,
                "move_refused",
            )
        reached = r.reached or (x, y, z_cmd)
        # position stall: commanded down but barely moved
        if (z - reached[2]) < 0.5 * step:
            return MoveResult(
                True,
                f"contact:stall at z={reached[2]:.4f}",
                last,
                "contact_stall",
            )
        cur_t = joint_torques()
        jump = max((abs(abs(c) - abs(p)) for c, p in zip(cur_t, prev_t)), default=0.0)
        if jump > jump_limit:
            return MoveResult(
                True,
                f"contact:torque+{jump:.2f}Nm at z={reached[2]:.4f}",
                last,
                "contact_torque",
            )
        prev_t = cur_t
        last = reached
        z = reached[2]
        time.sleep(settle)
    return MoveResult(True, f"reached floor z={last[2]:.4f}", last, "floor_reached")


def gripper(action: str) -> bool:
    """Open/close via the service, which HARD-CLAMPS open to the 68% cap."""
    assert action in ("open", "close")
    local = _local_arm()
    if local is not None:
        try:
            return bool(local.gripper(action).get("ok"))
        except Exception:
            return False
    try:
        r = httpx.post(f"{ARM}/gripper", json={"action": action}, timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def preflight(*, connect: bool = False) -> MoveResult:
    """Verify the arm is connected and reporting a sane, in-bounds pose before use."""
    local = _local_arm()
    if local is not None and connect and not local.connected:
        try:
            local.connect()
        except Exception as e:
            return MoveResult(False, f"arm_connect_failed {e}")
    # A dry visual check needs only the local Pinocchio model, not an armed CAN
    # bus. Runtime callers pass connect=True and therefore cannot reach this case.
    if local is not None and not local.connected:
        return MoveResult(True, "ik_ready")
    if not connected():
        return MoveResult(False, "arm_not_connected")
    try:
        p = ee_pos()
    except Exception as e:
        return MoveResult(False, f"no_state {e}")
    if not all(map(lambda v: isinstance(v, (int, float)), p)):
        return MoveResult(False, "bad_state")
    return MoveResult(True, "ready", tuple(p))
