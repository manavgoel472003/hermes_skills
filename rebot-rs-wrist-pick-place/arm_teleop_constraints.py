"""Minimal shared workspace constraints required by ``arm_agent.arm_safe``.

This is the safety-relevant subset of the RS teleoperation constraints. Keeping
it beside the skill makes dry-run and simulation tests independent of the
operator's separate ``reBotArm_control_py_fresh`` checkout.
"""
from __future__ import annotations

import numpy as np

ARM_X_MIN, ARM_X_MAX = 0.21, 0.43
ARM_Y_MIN, ARM_Y_MAX = -0.17, 0.17
ARM_Z_MIN, ARM_Z_MAX = 0.17, 0.34
ARM_HARD_Z_MIN = 0.13
WORKSPACE_LIMIT_SAFETY_SCALE = 0.85
WORKSPACE_LIMIT_SAFETY_SCALE_X = 1.00
MAX_TARGET_SPEED_M_S = 0.18


def clamp_workspace(pos: np.ndarray) -> np.ndarray:
    """Clamp an XYZ target into the safety-scaled teleoperation workspace."""
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
    """Return the lower and upper XYZ corners of the effective workspace."""
    lo = clamp_workspace(np.array([-1e3, -1e3, -1e3]))
    hi = clamp_workspace(np.array([1e3, 1e3, 1e3]))
    return lo, hi
