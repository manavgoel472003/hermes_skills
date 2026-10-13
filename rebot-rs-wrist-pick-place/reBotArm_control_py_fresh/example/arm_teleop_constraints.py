"""Shared, dependency-free (numpy-only) workspace + movement constraints for
teleoperation of the B601-RS arm.

Both teleop front-ends import these so the arm obeys IDENTICAL limits regardless
of how the target is produced:

  - the gesture demo (bimanual_xyz_tilt_rpy_demo.py, default motion profile)
  - the depth-camera follow demo (depth_hand_follow_demo.py)

The numbers mirror the gesture demo's DEFAULT motion profile. They live here as
plain constants (no arm-stack imports) so this module also loads inside the
Orbbec camera venv, which does not have reBotArm_control_py installed.

If you retune the gesture demo's default profile workspace, update these too
(test_constraints_match() checks they agree when both are importable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Workspace box (metres) — teleop bounds (based on the gesture demo's default
# profile, with X reach and +Z raised a bit for the depth follow mode) ──────────
ARM_X_MIN, ARM_X_MAX = 0.21, 0.43          # a bit more reach
ARM_Y_MIN, ARM_Y_MAX = -0.17, 0.17
ARM_Z_MIN, ARM_Z_MAX = 0.17, 0.34          # higher +Z ceiling
ARM_HARD_Z_MIN = 0.13                      # absolute end-effector Z floor
WORKSPACE_LIMIT_SAFETY_SCALE = 0.85        # shrink Y/Z toward centre
WORKSPACE_LIMIT_SAFETY_SCALE_X = 1.00
ARM_HOME = np.array([0.30, 0.0, 0.22], dtype=np.float64)   # natural RS q=0 pose

# ── Movement limits ───────────────────────────────────────────────────────────
# Max Cartesian speed of the commanded target. Keeps the arm from lurching if the
# hand jumps or depth glitches; matches the gesture demo's per-tick step budget
# (~10 mm at ~20 Hz -> ~0.2 m/s), kept deliberately conservative.
MAX_TARGET_SPEED_M_S = 0.18
# EMA smoothing time constant for the mapped target (larger = smoother/slower).
TARGET_SMOOTH_TAU_S = 0.20

# Gripper travel safety limits. Zero is fully closed; the demo deliberately
# stays at least 5% open and never uses more than 70% of full travel.
GRIPPER_MIN_OPEN_FRACTION = 0.05
GRIPPER_MAX_OPEN_FRACTION = 0.70


def clamp_workspace(pos: np.ndarray) -> np.ndarray:
    """Clamp an XYZ target into the safe workspace box. Identical policy to the
    gesture demo's ``_clamp_workspace``."""
    clamped = np.array(pos, dtype=np.float64, copy=True)
    x_center = 0.5 * (ARM_X_MIN + ARM_X_MAX)
    x_half = 0.5 * (ARM_X_MAX - ARM_X_MIN) * WORKSPACE_LIMIT_SAFETY_SCALE_X
    y_center = 0.5 * (ARM_Y_MIN + ARM_Y_MAX)
    y_half = 0.5 * (ARM_Y_MAX - ARM_Y_MIN) * WORKSPACE_LIMIT_SAFETY_SCALE
    z_center = 0.5 * (ARM_Z_MIN + ARM_Z_MAX)
    z_half = 0.5 * (ARM_Z_MAX - ARM_Z_MIN) * WORKSPACE_LIMIT_SAFETY_SCALE
    clamped[0] = np.clip(clamped[0], x_center - x_half, x_center + x_half)
    clamped[1] = np.clip(clamped[1], y_center - y_half, y_center + y_half)
    safe_z_min = max(ARM_HARD_Z_MIN + 0.005, z_center - z_half)
    clamped[2] = np.clip(clamped[2], safe_z_min, z_center + z_half)
    return clamped


def workspace_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) XYZ corners of the *effective* (safety-scaled) box."""
    lo = clamp_workspace(np.array([-1e3, -1e3, -1e3]))
    hi = clamp_workspace(np.array([1e3, 1e3, 1e3]))
    return lo, hi


@dataclass
class AxisMap:
    """How a hand displacement in CAMERA coordinates maps to ARM coordinates.

    Camera frame (Orbbec, looking at the user): +x right, +y down, +z forward
    (away from camera / deeper). Arm frame: +x reach-out, +y arm-left, +z up.

    Each arm axis picks one camera axis and a sign. Defaults chosen so the arm
    mirrors the operator; every sign is easy to flip after a live look.
    """
    arm_x_from: str = "z"   # reach   <- hand depth
    arm_x_sign: float = -1.0  # hand toward camera (smaller z) -> arm reaches out
    arm_y_from: str = "x"   # left/right <- hand horizontal
    arm_y_sign: float = 1.0   # swapped back: camera +x maps to arm +Y
    arm_z_from: str = "y"   # up/down <- hand vertical
    arm_z_sign: float = -1.0  # hand up (smaller image y) -> arm up
    scale: float = 1.0        # arm metres per camera metre (1.0 = 1:1 mimic)

    def apply(self, d_cam: np.ndarray) -> np.ndarray:
        idx = {"x": 0, "y": 1, "z": 2}
        out = np.array([
            self.arm_x_sign * d_cam[idx[self.arm_x_from]],
            self.arm_y_sign * d_cam[idx[self.arm_y_from]],
            self.arm_z_sign * d_cam[idx[self.arm_z_from]],
        ], dtype=np.float64)
        return out * self.scale


class HandFollowMapper:
    """Turns a live hand position (camera metres) into a smooth, workspace-clamped,
    speed-limited arm XYZ target.

    Relative mapping: when (re)activated, the hand's current position becomes the
    neutral reference and the arm anchor is ARM_HOME. Hand displacement from
    neutral -> arm displacement from HOME (via AxisMap), then clamp + smooth + slew.
    """

    def __init__(self, axis_map: AxisMap | None = None,
                 anchor: np.ndarray | None = None) -> None:
        self.axis_map = axis_map or AxisMap()
        self.anchor = np.array(ARM_HOME if anchor is None else anchor, dtype=np.float64)
        self._neutral_cam: np.ndarray | None = None
        self._target = self.anchor.copy()      # smoothed, constrained target
        self._raw_target = self.anchor.copy()   # pre-smoothing (for display)
        self._clamped_axes = np.zeros(3, dtype=bool)

    def reset(self, anchor: np.ndarray | None = None) -> None:
        """Drop the neutral reference; next update() re-captures it. Optionally
        re-anchor (e.g. to the arm's actual current pose)."""
        if anchor is not None:
            self.anchor = np.array(anchor, dtype=np.float64)
        self._neutral_cam = None
        self._target = self.anchor.copy()
        self._raw_target = self.anchor.copy()

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def raw_target(self) -> np.ndarray:
        return self._raw_target.copy()

    @property
    def clamped_axes(self) -> np.ndarray:
        return self._clamped_axes.copy()

    @property
    def has_neutral(self) -> bool:
        return self._neutral_cam is not None

    def update(self, hand_cam_xyz: np.ndarray | None, dt: float) -> np.ndarray:
        """Advance one frame. hand_cam_xyz is the hand position in camera metres,
        or None if the hand is not tracked (holds the last target)."""
        if hand_cam_xyz is None or not np.all(np.isfinite(hand_cam_xyz)):
            return self._target.copy()

        if self._neutral_cam is None:
            self._neutral_cam = np.array(hand_cam_xyz, dtype=np.float64)

        d_cam = np.array(hand_cam_xyz, dtype=np.float64) - self._neutral_cam
        raw = self.anchor + self.axis_map.apply(d_cam)

        clamped = clamp_workspace(raw)
        self._clamped_axes = ~np.isclose(clamped, raw, atol=1e-6)
        self._raw_target = raw

        # EMA smoothing toward the clamped target.
        gain = 1.0 if dt <= 0 else min(1.0, dt / max(TARGET_SMOOTH_TAU_S, 1e-6))
        desired = self._target + gain * (clamped - self._target)

        # Speed (slew) limit.
        step = desired - self._target
        max_step = MAX_TARGET_SPEED_M_S * max(dt, 1e-3)
        dist = float(np.linalg.norm(step))
        if dist > max_step and dist > 0:
            step *= max_step / dist
        self._target = self._target + step
        return self._target.copy()


# ── Wrist (J4/J5/J6) follow mode ───────────────────────────────────────────────
# URDF joint limits for wrist joints (index 3,4,5): J4,J5 +/-1.57, J6 +/-3.14.
WRIST_JOINT_LO = np.array([-1.57, -1.57, -3.14], dtype=np.float64)
WRIST_JOINT_HI = np.array([1.57, 1.57, 3.14], dtype=np.float64)
WRIST_SAFETY_SCALE = 0.6            # only use 60% of each joint's range
WRIST_SMOOTH_TAU_S = 0.15
WRIST_MAX_SPEED_RAD_S = 1.2         # slew cap on the wrist command


@dataclass
class WristAxisMap:
    """Hand displacement (camera metres) -> wrist joint deltas (radians)."""
    j4_from: str = "y"; j4_sign: float = -1.0   # hand up (smaller cam-y) -> J4 +
    j5_from: str = "x"; j5_sign: float = 1.0    # hand right -> J5 +
    j6_from: str = "z"; j6_sign: float = 1.0    # hand forward -> J6 +
    scale: float = 6.0                          # radians per camera metre

    def apply(self, d_cam: np.ndarray) -> np.ndarray:
        idx = {"x": 0, "y": 1, "z": 2}
        return np.array([
            self.j4_sign * d_cam[idx[self.j4_from]],
            self.j5_sign * d_cam[idx[self.j5_from]],
            self.j6_sign * d_cam[idx[self.j6_from]],
        ], dtype=np.float64) * self.scale


class WristFollowMapper:
    """Turns hand motion into smooth, limited J4/J5/J6 targets. Relative: on
    (re)activation the hand's position becomes neutral and the wrist anchor is the
    arm's current wrist config; hand displacement -> wrist-joint deltas."""

    def __init__(self, axis_map: WristAxisMap | None = None) -> None:
        self.axis_map = axis_map or WristAxisMap()
        self.lo = WRIST_JOINT_LO * WRIST_SAFETY_SCALE
        self.hi = WRIST_JOINT_HI * WRIST_SAFETY_SCALE
        self._neutral_cam: np.ndarray | None = None
        self._anchor = np.zeros(3, dtype=np.float64)
        self._target = np.zeros(3, dtype=np.float64)

    def reset(self, anchor_wrist_q: np.ndarray | None = None) -> None:
        if anchor_wrist_q is not None:
            self._anchor = np.clip(np.asarray(anchor_wrist_q, dtype=np.float64), self.lo, self.hi)
        self._neutral_cam = None
        self._target = self._anchor.copy()

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def has_neutral(self) -> bool:
        return self._neutral_cam is not None

    def update(self, hand_cam_xyz: np.ndarray | None, dt: float) -> np.ndarray:
        if hand_cam_xyz is None or not np.all(np.isfinite(hand_cam_xyz)):
            return self._target.copy()
        if self._neutral_cam is None:
            self._neutral_cam = np.array(hand_cam_xyz, dtype=np.float64)
        d_cam = np.array(hand_cam_xyz, dtype=np.float64) - self._neutral_cam
        raw = self._anchor + self.axis_map.apply(d_cam)
        clamped = np.clip(raw, self.lo, self.hi)
        gain = 1.0 if dt <= 0 else min(1.0, dt / max(WRIST_SMOOTH_TAU_S, 1e-6))
        desired = self._target + gain * (clamped - self._target)
        step = desired - self._target
        max_step = WRIST_MAX_SPEED_RAD_S * max(dt, 1e-3)
        dist = float(np.linalg.norm(step))
        if dist > max_step and dist > 0:
            step *= max_step / dist
        self._target = self._target + step
        return self._target.copy()


def deproject_pixel(u: float, v: float, depth_m: float,
                    fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole back-projection: pixel (u,v) + depth (m) -> camera-frame XYZ (m)."""
    x = (u - cx) / fx * depth_m
    y = (v - cy) / fy * depth_m
    return np.array([x, y, depth_m], dtype=np.float64)
