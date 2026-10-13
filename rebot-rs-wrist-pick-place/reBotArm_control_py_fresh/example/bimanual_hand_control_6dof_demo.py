#!/usr/bin/env python3
"""Bimanual 6DOF hand control demo for ReBot arm + gripper.

Standalone script (does not depend on other demo files).

Left hand:
  - pinch closed -> close gripper gradually
  - hand open    -> open gripper gradually

Right hand translate mode:
  - open palm + thumb right -> X+
  - open palm + thumb left  -> X-
  - thumb up                -> Z+
  - thumb down              -> Z-
  - closed thumb left       -> Y+
  - closed thumb right      -> Y-

Mode switching:
  - both palms open (held)  -> toggle rotate mode
  - clap or both fists      -> home

Right hand rotate mode (same gestures, mapped to RPY deltas):
  - open palm + thumb R/L   -> roll +/-
  - thumb up/down           -> pitch +/-
  - closed thumb left/right -> yaw +/-
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2
except ImportError:
    print("OpenCV missing. Run: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("mediapipe missing. Run: pip install mediapipe")
    sys.exit(1)

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import joint_to_pose


# B601-RS gripper frame: zero = fully CLOSED, negative = open (full travel -270 deg).
GRIPPER_OPEN_POS = -4.712388980384690    # -270 deg, fully open
GRIPPER_CLOSE_POS = 0.0                  # fully closed (zeroed here)
GRIPPER_VLIM = 3.0
GRIPPER_STEP = 0.05
GRIPPER_MAX_OPEN_FRACTION = 0.7          # HARD CAP: never open past 70% of travel
GRIPPER_MIN_OPEN_FRACTION = 0.05         # HARD FLOOR: never close past 5% open

ARM_TELEOP_VLIM_CAP = 1.2                # rad/s hard cap on joint speed during teleop
DEMO_LOOP_RATE_HZ = 150.0                # control-loop rate; 500 Hz x 7 motors saturates
                                         # the 1 Mbit CAN bus (ENOBUFS + dropped frames)

PINCH_THRESHOLD = 0.28

# Workspace tuned for the B601-RS URDF: its natural (q=0) end-effector pose is
# (0.302, 0.0, 0.218) with a level tool; holding the tool level couples low Z
# with smaller X, so HOME sits at the natural pose.
ARM_X_MIN, ARM_X_MAX = 0.22, 0.40
ARM_Y_MIN, ARM_Y_MAX = -0.17, 0.17
ARM_Z_MIN, ARM_Z_MAX = 0.15, 0.32
ARM_HARD_Z_MIN = 0.12
WORKSPACE_LIMIT_SAFETY_SCALE = 0.85
WORKSPACE_LIMIT_SAFETY_SCALE_X = 1.00
ARM_HOME = np.array([0.30, 0.0, 0.22], dtype=np.float64)

ARM_MOVE_INTERVAL = 0.05
ARM_STEP_MIN = 0.003
ARM_STEP_MAX = 0.012
ARM_MAX_DELTA = ARM_STEP_MAX
ARM_Y_STEP_SCALE = 1.0        # <1.0 makes sideways (Y) motion slower than X/Z
ARM_SMOOTHING_ALPHA = 0.35
ARM_SMOOTHING_ZERO = 0.0015
DEMO_IK_TOLERANCE = 1.5e-2
DEMO_IK_MAX_ITER = 400
GESTURE_CONFIRM_FRAMES = 3
GESTURE_RELEASE_FRAMES = 5
FAILED_COMMAND_COOLDOWN = 0.8
PALM_OPEN_MIN_FINGERS = 3
THUMB_REACH_MIN = 0.45
THUMB_AXIS_MIN = 0.40
POSE_VIS_MIN = 0.45
CLAP_DIST_FRAC = 0.30
CLAP_HOLD_FRAMES = 4
CLAP_COOLDOWN_S = 2.5
FISTS_HOME_HOLD_FRAMES = 4
FISTS_HOME_COOLDOWN_S = 2.5

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
HOLISTIC_MODEL_PATH = MODEL_DIR / "holistic_landmarker.task"
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

POSE_CONNECTIONS = (
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
)

ROT_MODE_HOLD_FRAMES = 4
ROT_MODE_COOLDOWN_S = 1.5
ROT_STEP_MAX = 0.015
ROT_SMOOTHING_ALPHA = 0.22
ROT_SMOOTHING_ZERO = 0.0025
ROLL_LIMIT = 0.35
PITCH_LIMIT = 0.30
YAW_LIMIT = 0.45


@dataclass
class HandObservation:
    side: str
    wrist_px: tuple[int, int]
    pts_norm: np.ndarray
    pts_px: np.ndarray
    palm_size_px: float


def _ensure_holistic_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if HOLISTIC_MODEL_PATH.exists():
        return HOLISTIC_MODEL_PATH
    print(f"Downloading MediaPipe holistic model to {HOLISTIC_MODEL_PATH} ...")
    urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
    return HOLISTIC_MODEL_PATH


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _clamp_workspace(pos: np.ndarray) -> np.ndarray:
    clamped = np.array(pos, dtype=np.float64, copy=True)
    x_center = 0.5 * (ARM_X_MIN + ARM_X_MAX)
    x_half = 0.5 * (ARM_X_MAX - ARM_X_MIN) * WORKSPACE_LIMIT_SAFETY_SCALE_X
    y_center = 0.5 * (ARM_Y_MIN + ARM_Y_MAX)
    y_half = 0.5 * (ARM_Y_MAX - ARM_Y_MIN) * WORKSPACE_LIMIT_SAFETY_SCALE
    z_center = 0.5 * (ARM_Z_MIN + ARM_Z_MAX)
    z_half = 0.5 * (ARM_Z_MAX - ARM_Z_MIN) * WORKSPACE_LIMIT_SAFETY_SCALE
    clamped[0] = np.clip(clamped[0], x_center - x_half, x_center + x_half)
    clamped[1] = np.clip(clamped[1], y_center - y_half, y_center + y_half)
    # Keep a 5 mm command margin so IK tolerance cannot cross the absolute floor.
    safe_z_min = max(ARM_HARD_Z_MIN + 0.005, z_center - z_half)
    clamped[2] = np.clip(clamped[2], safe_z_min, z_center + z_half)
    return clamped


def _sanitize_delta(dx: float, dy: float, dz: float) -> np.ndarray:
    delta = np.array([dx, dy, dz], dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        return np.zeros(3, dtype=np.float64)
    return np.clip(delta, -ARM_MAX_DELTA, ARM_MAX_DELTA)


def _pose_px(landmarks, index: int, shape: tuple[int, int, int]) -> tuple[float, float, float] | None:
    if not landmarks or len(landmarks) <= index:
        return None
    landmark = landmarks[index]
    visibility = float(getattr(landmark, "visibility", 1.0) or 1.0)
    if visibility < POSE_VIS_MIN:
        return None
    height, width = shape[:2]
    return (
        float(landmark.x * width),
        float(landmark.y * height),
        visibility,
    )


def _extract_pose_points(pose_landmarks, shape: tuple[int, int, int]) -> dict[str, tuple[float, float, float] | None]:
    return {
        "left_shoulder": _pose_px(pose_landmarks, POSE_LEFT_SHOULDER, shape),
        "right_shoulder": _pose_px(pose_landmarks, POSE_RIGHT_SHOULDER, shape),
        "left_wrist": _pose_px(pose_landmarks, POSE_LEFT_WRIST, shape),
        "right_wrist": _pose_px(pose_landmarks, POSE_RIGHT_WRIST, shape),
    }


def _build_hand_observations(result, shape: tuple[int, int, int]) -> dict[str, HandObservation]:
    height, width = shape[:2]
    observations: dict[str, HandObservation] = {}

    for side, landmarks in (
        ("left", getattr(result, "left_hand_landmarks", None)),
        ("right", getattr(result, "right_hand_landmarks", None)),
    ):
        if not landmarks:
            continue
        pts_norm = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        pts_px = np.column_stack((pts_norm[:, 0] * width, pts_norm[:, 1] * height))
        wrist_px = (int(pts_px[0, 0]), int(pts_px[0, 1]))
        observations[side] = HandObservation(
            side=side,
            wrist_px=wrist_px,
            pts_norm=pts_norm,
            pts_px=pts_px,
            palm_size_px=_dist(pts_px[0], pts_px[9]) + 1e-6,
        )

    return observations


def _pinch_openness(hand: HandObservation | None) -> float:
    if hand is None:
        return -1.0
    norm_dist = _dist(hand.pts_norm[4, :2], hand.pts_norm[8, :2]) / max(
        _dist(hand.pts_norm[0, :2], hand.pts_norm[9, :2]),
        1e-6,
    )
    return float(norm_dist)


def _finger_extended(hand: HandObservation, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    wrist = hand.pts_norm[0, :2]
    tip = hand.pts_norm[tip_idx, :2]
    pip = hand.pts_norm[pip_idx, :2]
    mcp = hand.pts_norm[mcp_idx, :2]
    return (_dist(tip, wrist) > 1.12 * _dist(pip, wrist)) and (tip[1] < mcp[1] + 0.04)


def _extended_finger_count(hand: HandObservation) -> int:
    return sum(
        (
            _finger_extended(hand, 8, 6, 5),
            _finger_extended(hand, 12, 10, 9),
            _finger_extended(hand, 16, 14, 13),
            _finger_extended(hand, 20, 18, 17),
        )
    )


def _palm_open(hand: HandObservation) -> bool:
    return _extended_finger_count(hand) >= PALM_OPEN_MIN_FINGERS


def _closed_fist(hand: HandObservation | None) -> bool:
    if hand is None:
        return False
    extended_count = _extended_finger_count(hand)
    thumb_tip = hand.pts_norm[4, :2]
    palm_center = np.mean(hand.pts_norm[[0, 5, 9, 13, 17], :2], axis=0)
    palm_scale = max(_dist(hand.pts_norm[0, :2], hand.pts_norm[9, :2]), 1e-6)
    thumb_fold = _dist(thumb_tip, palm_center) / palm_scale
    return extended_count <= 1 and thumb_fold < 0.95


def _right_hand_command(hand: HandObservation | None) -> str | None:
    if hand is None:
        return None

    thumb_tip = hand.pts_norm[4, :2]
    wrist = hand.pts_norm[0, :2]
    palm_scale = max(_dist(hand.pts_norm[0, :2], hand.pts_norm[9, :2]), 1e-6)
    thumb_dx = float((thumb_tip[0] - wrist[0]) / palm_scale)
    thumb_dy = float((thumb_tip[1] - wrist[1]) / palm_scale)
    thumb_reach = float(np.hypot(thumb_dx, thumb_dy))

    if thumb_reach < THUMB_REACH_MIN:
        return None

    if _palm_open(hand):
        if abs(thumb_dx) < THUMB_AXIS_MIN:
            return None
        if thumb_dx < 0:
            return "x_back"
        return "x_forward"

    if abs(thumb_dy) >= abs(thumb_dx):
        if abs(thumb_dy) < THUMB_AXIS_MIN:
            return None
        if thumb_dy < 0:
            return "z_up"
        return "z_down"

    if abs(thumb_dx) < THUMB_AXIS_MIN:
        return None
    if thumb_dx < 0:
        return "y_left"
    return "y_right"


def _command_to_delta(command: str | None) -> tuple[float, float, float]:
    if command == "x_forward":
        return ARM_STEP_MAX, 0.0, 0.0
    if command == "x_back":
        return -ARM_STEP_MAX, 0.0, 0.0
    if command == "y_left":
        return 0.0, ARM_STEP_MAX * ARM_Y_STEP_SCALE, 0.0
    if command == "y_right":
        return 0.0, -ARM_STEP_MAX * ARM_Y_STEP_SCALE, 0.0
    if command == "z_up":
        return 0.0, 0.0, ARM_STEP_MAX
    if command == "z_down":
        return 0.0, 0.0, -ARM_STEP_MAX
    return 0.0, 0.0, 0.0


def _command_label(command: str | None) -> str:
    labels = {
        "x_forward": "X+ forward",
        "x_back": "X- backward",
        "y_left": "Y+ left",
        "y_right": "Y- right",
        "z_up": "Z+ up",
        "z_down": "Z- down",
    }
    return labels.get(command, "")


def _pinch_state(norm_dist: float) -> str:
    if norm_dist < 0:
        return "hold"
    if norm_dist < PINCH_THRESHOLD:
        return "closing"
    return "opening"


class ClapDetector:
    def __init__(self) -> None:
        self._recent: list[bool] = []
        self._last_trigger_t = 0.0

    def update(self, pose_points: dict[str, tuple[float, float, float] | None]) -> bool:
        left_wrist = pose_points.get("left_wrist")
        right_wrist = pose_points.get("right_wrist")
        left_shoulder = pose_points.get("left_shoulder")
        right_shoulder = pose_points.get("right_shoulder")

        close = False
        if all(point is not None for point in (left_wrist, right_wrist, left_shoulder, right_shoulder)):
            wrist_dist = float(np.hypot(left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1]))
            shoulder_dist = float(np.hypot(left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1])) + 1e-6
            close = wrist_dist < CLAP_DIST_FRAC * shoulder_dist

        self._recent.append(close)
        if len(self._recent) > CLAP_HOLD_FRAMES:
            self._recent.pop(0)

        now = time.monotonic()
        if (
            len(self._recent) == CLAP_HOLD_FRAMES
            and all(self._recent)
            and (now - self._last_trigger_t) > CLAP_COOLDOWN_S
        ):
            self._last_trigger_t = now
            self._recent.clear()
            return True
        return False


class DualFistHomeDetector:
    def __init__(self) -> None:
        self._recent: list[bool] = []
        self._last_trigger_t = 0.0
        self._latched = False

    def update(
        self,
        left_hand: HandObservation | None,
        right_hand: HandObservation | None,
    ) -> bool:
        both_closed = _closed_fist(left_hand) and _closed_fist(right_hand)
        if not both_closed:
            self._latched = False
        self._recent.append(both_closed)
        if len(self._recent) > FISTS_HOME_HOLD_FRAMES:
            self._recent.pop(0)

        now = time.monotonic()
        if (
            not self._latched
            and len(self._recent) == FISTS_HOME_HOLD_FRAMES
            and all(self._recent)
            and (now - self._last_trigger_t) > FISTS_HOME_COOLDOWN_S
        ):
            self._last_trigger_t = now
            self._latched = True
            self._recent.clear()
            return True
        return False


SMOOTH_TAU_S = 0.12            # low-pass time constant for streamed joint commands
SMOOTH_SLEW_RAD_S = 1.2        # max joint speed of the streamed command (matches teleop cap)
GRIPPER_SMOOTH_TAU_S = 0.15
GRIPPER_SMOOTH_SLEW_RAD_S = 3.0


class SmoothedEndPose(RebotArmEndPose):
    """RebotArmEndPose whose control loop streams a slew-limited, low-pass
    filtered command toward ``_q_target`` instead of jumping to it.

    Gesture updates step the target ~20x/s; without filtering the motors chase
    each step then stop, producing visible stop-and-go pulsing. The filter turns
    the steps into one continuous velocity-bounded path. POS_VEL mode only.
    """

    def __init__(self, rebotarm: RebotArm, **kwargs) -> None:
        if kwargs.get("arm_control_mode", "posvel") != "posvel":
            raise ValueError("SmoothedEndPose supports arm_control_mode='posvel' only")
        kwargs["arm_control_mode"] = "posvel"
        super().__init__(rebotarm, **kwargs)
        self._q_cmd: np.ndarray | None = None
        self._grip_cmd: float | None = None
        self._suppress_gripper = False   # pause loop's gripper output during recovery
        # Cap the control-loop rate BEFORE start(): the yaml default (500 Hz)
        # saturates the CAN bus with 7 motors (TX overflow, dropped frames).
        rebotarm._rate = min(float(rebotarm._rate), DEMO_LOOP_RATE_HZ)

    def _loop_cb(self, _: RebotArm, dt: float) -> None:
        q_target = np.asarray(self._q_target, dtype=np.float64)
        if self._q_cmd is None or self._q_cmd.shape != q_target.shape:
            self._q_cmd = q_target.copy()
        gain = min(1.0, dt / SMOOTH_TAU_S)
        lim = SMOOTH_SLEW_RAD_S * dt
        self._q_cmd = self._q_cmd + np.clip(gain * (q_target - self._q_cmd), -lim, lim)
        if self._arm_group:
            vlim = (
                self._vlim_override
                if self._vlim_override is not None
                else self._arm_group._pv_vlim
            )
            self._arm_group.send_pos_vel(self._q_cmd, vlim=vlim)
        if self._has_gripper and not self._suppress_gripper:
            g_target = float(self._gripper_target)
            if self._grip_cmd is None:
                self._grip_cmd = g_target
            g_gain = min(1.0, dt / GRIPPER_SMOOTH_TAU_S)
            g_lim = GRIPPER_SMOOTH_SLEW_RAD_S * dt
            self._grip_cmd += float(np.clip(g_gain * (g_target - self._grip_cmd), -g_lim, g_lim))
            self._gripper_group.send_mit(
                np.array([self._grip_cmd]),
                kp=self._gripper_group._mit_kp,
                kd=self._gripper_group._mit_kd,
            )


class GripperController:
    """Drives the gripper through the shared RebotArmEndPose controller.

    HARD SAFETY (independent of motion profile):
      - open fraction can never exceed GRIPPER_MAX_OPEN_FRACTION (50%)
      - open fraction can never drop below GRIPPER_MIN_OPEN_FRACTION (5%),
        so the gripper never fully closes on anything.
    """

    def __init__(self, no_gripper: bool,
                 endpose: "RebotArmEndPose | None" = None,
                 max_open_fraction: float | None = None) -> None:
        self._dry = no_gripper
        self._endpose = endpose
        cap = GRIPPER_MAX_OPEN_FRACTION if max_open_fraction is None else float(max_open_fraction)
        self._open_cap = min(cap, GRIPPER_MAX_OPEN_FRACTION)   # never above 50%
        self._open_floor = GRIPPER_MIN_OPEN_FRACTION           # never below 5%
        self._open_frac = self._open_floor

        if self._dry:
            return
        if self._endpose is None or not self._endpose.rebotarm.has_gripper:
            raise RuntimeError("Gripper requested but no gripper group is configured")

        if not self._move_to_floor_and_verify():
            raise RuntimeError(
                "Gripper failed to reach the required startup position (5% open). "
                "Check the gripper zero: 0 must be fully CLOSED (re-zero with the "
                "gripper closed), and open direction must be negative."
            )

    def _frac_to_pos(self, frac: float) -> float:
        """Map open fraction -> motor position, hard-clamped to [5%, 50%] travel."""
        frac = float(np.clip(frac, self._open_floor, self._open_cap))
        return GRIPPER_CLOSE_POS + frac * (GRIPPER_OPEN_POS - GRIPPER_CLOSE_POS)

    def _gripper_position(self) -> float:
        q, _, _ = self._endpose.rebotarm.get_state()
        return float(q[self._endpose._n])

    def _gripper_status(self) -> int | None:
        mot = self._endpose.rebotarm._motor_map.get("gripper")
        if mot is None:
            return None
        try:
            mot.request_feedback()
            self._endpose.rebotarm._ctrl_map["robstride"].poll_feedback_once()
        except Exception:
            pass
        st = mot.get_state()
        return st.status_code if st is not None else None

    def _cycle_gripper_motor(self) -> None:
        """Disable->settle->enable cycle to clear a latched RobStride protection
        state (status!=0, e.g. after a killed session). The control loop keeps
        streaming to the gripper, so we suppress its gripper output while cycling
        (otherwise the loop's commands immediately re-arm the motor)."""
        mot = self._endpose.rebotarm._motor_map.get("gripper")
        if mot is None:
            return
        try:
            print(f"[gripper] cycling enable to clear status={self._gripper_status()}")
            self._endpose._suppress_gripper = True
            time.sleep(0.1)               # let the loop finish its current send
            mot.clear_error()             # required to clear the latched protection bit
            time.sleep(0.3)
            mot.disable()
            time.sleep(2.0)               # protection state needs a long settle
            mot.enable()
            time.sleep(0.3)
            # Re-seed the smoothed command to the current position so the loop
            # resumes from where the gripper actually is (no jump).
            self._endpose._grip_cmd = self._gripper_position()
        except Exception as exc:
            print(f"[gripper] enable cycle failed: {exc}")
        finally:
            self._endpose._suppress_gripper = False

    def _move_to_floor_and_verify(self, timeout_s: float = 6.0, tolerance_rad: float = 0.15) -> bool:
        """Command the 5%-open floor until feedback confirms it; never start gestures otherwise."""
        if self._dry or self._endpose is None:
            return True
        # Proactively clear a latched protection state before wasting a timeout.
        if self._gripper_status() not in (0, None):
            self._cycle_gripper_motor()
        target = self._frac_to_pos(self._open_floor)
        for attempt in range(2):
            self._endpose.set_gripper_target(target)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                time.sleep(0.05)
                position = self._gripper_position()
                if np.isfinite(position) and abs(position - target) <= tolerance_rad:
                    self._open_frac = self._open_floor
                    print("[gripper] startup position (5% open) confirmed")
                    return True
            if attempt == 0:
                self._cycle_gripper_motor()
        # SAFETY: stop pushing toward an unreachable target (e.g. wrong zero
        # convention would press the gripper against its hard stop).
        held = self._gripper_position()
        if np.isfinite(held):
            self._endpose.set_gripper_target(held)
        return False

    def update(self, pinch_state: str) -> None:
        if pinch_state == "opening":
            self._open_frac = min(self._open_cap, self._open_frac + GRIPPER_STEP)
        elif pinch_state == "closing":
            self._open_frac = max(self._open_floor, self._open_frac - GRIPPER_STEP)
        else:
            return
        if not self._dry and self._endpose is not None:
            self._endpose.set_gripper_target(self._frac_to_pos(self._open_frac))

    @property
    def open_frac(self) -> float:
        return self._open_frac

    def stop(self) -> None:
        # The shared RebotArmEndPose owns the hardware; nothing to release here.
        return


class ArmController:
    def __init__(self, no_arm: bool, hw_yaml: str | None = None,
                 vlim_cap: float | None = None) -> None:
        self._dry = no_arm
        self._arm: RebotArm | None = None
        self._endpos: RebotArmEndPose | None = None
        self._pos = _clamp_workspace(ARM_HOME)
        self._rpy = np.zeros(3, dtype=np.float64)
        self._smoothed_delta = np.zeros(3, dtype=np.float64)
        self._last_move_t = 0.0
        self._homing = False

        if self._dry:
            return

        self._arm = RebotArm(hw_yaml)
        self._endpos = SmoothedEndPose(self._arm, arm_control_mode="posvel")
        self._endpos._ik_solver_params.tolerance = DEMO_IK_TOLERANCE
        self._endpos._ik_solver_params.max_iter = DEMO_IK_MAX_ITER
        self._endpos.start()
        # SAFETY: cap the joint speed limits the control loop sends (yaml
        # defaults are tuned for fast point-to-point moves, not teleop).
        arm_group = self._arm.groups["arm"]
        cap = ARM_TELEOP_VLIM_CAP if vlim_cap is None else min(float(vlim_cap), ARM_TELEOP_VLIM_CAP)
        arm_group._pv_vlim = np.minimum(
            np.asarray(arm_group._pv_vlim, dtype=np.float64), cap)
        self._refresh_pose_from_arm()
        target = _clamp_workspace(ARM_HOME)
        ok = self._endpos.move_to_traj(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
            roll=float(self._rpy[0]),
            pitch=float(self._rpy[1]),
            yaw=float(self._rpy[2]),
            duration=2.0,
        )
        if ok and self._wait_for_home(target):
            self._pos = target
        else:
            self._arm.disconnect()
            self._endpos = None
            self._arm = None
            raise RuntimeError("Arm failed to reach HOME during mandatory startup initialization")
        print("[arm] startup HOME position confirmed")

    def _wait_for_home(
        self,
        target: np.ndarray,
        timeout_s: float = 35.0,
        position_tolerance_m: float = 0.018,
        velocity_tolerance_rad_s: float = 0.08,
    ) -> bool:
        """Require measured HOME pose and low joint velocity before gestures start."""
        if self._dry or self._arm is None or self._endpos is None:
            return True
        deadline = time.monotonic() + timeout_s
        stable_samples = 0
        while time.monotonic() < deadline:
            try:
                q, velocity, _ = self._arm.get_state()
                n = self._endpos._n
                q, velocity = q[:n], velocity[:n]
                measured_pos, _ = joint_to_pose(q)
                position_error = float(np.linalg.norm(np.asarray(measured_pos) - target))
                max_velocity = float(np.max(np.abs(velocity)))
                trajectory_done = not self._endpos._moving
                if (
                    trajectory_done
                    and position_error <= position_tolerance_m
                    and max_velocity <= velocity_tolerance_rad_s
                ):
                    stable_samples += 1
                    if stable_samples >= 3:
                        return True
                else:
                    stable_samples = 0
            except Exception:
                stable_samples = 0
            time.sleep(0.10)
        print("[arm] HOME confirmation timed out")
        return False

    def _refresh_pose_from_arm(self) -> None:
        if self._dry or self._arm is None:
            return
        q_curr, _, _ = self._arm.get_state()
        pos_curr, rpy_curr = joint_to_pose(q_curr[: self._endpos._n])
        self._pos = _clamp_workspace(np.array(pos_curr, dtype=np.float64))
        self._rpy = np.array(rpy_curr, dtype=np.float64)

    @property
    def hardware(self) -> RebotArm | None:
        return self._arm

    @property
    def endpose(self) -> RebotArmEndPose | None:
        return self._endpos

    @property
    def pos(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def homing(self) -> bool:
        return self._homing

    def move_delta(self, dx: float, dy: float, dz: float) -> bool:
        if self._homing:
            self._smoothed_delta[:] = 0.0
            return False
        desired_delta = _sanitize_delta(dx, dy, dz)
        self._smoothed_delta = (
            (1.0 - ARM_SMOOTHING_ALPHA) * self._smoothed_delta
            + ARM_SMOOTHING_ALPHA * desired_delta
        )
        if np.linalg.norm(desired_delta) < 1e-9 and np.linalg.norm(self._smoothed_delta) < ARM_SMOOTHING_ZERO:
            self._smoothed_delta[:] = 0.0
            return False
        now = time.monotonic()
        if now - self._last_move_t < ARM_MOVE_INTERVAL:
            return False

        delta = _sanitize_delta(*self._smoothed_delta)
        if np.allclose(delta, 0.0, atol=1e-9):
            return False

        new_pos = _clamp_workspace(self._pos + delta)
        if np.allclose(new_pos, self._pos, atol=3e-4):
            self._smoothed_delta[:] = 0.0
            return False

        if not self._dry and self._endpos is not None:
            ok = self._endpos.move_to_ik(
                x=float(new_pos[0]),
                y=float(new_pos[1]),
                z=float(new_pos[2]),
                roll=float(self._rpy[0]),
                pitch=float(self._rpy[1]),
                yaw=float(self._rpy[2]),
            )
            if not ok:
                self._smoothed_delta[:] = 0.0
                return False
        self._pos = new_pos
        self._last_move_t = now
        return True

    def go_home(self) -> None:
        if self._homing:
            return
        self._homing = True
        self._smoothed_delta[:] = 0.0
        target = _clamp_workspace(ARM_HOME)
        if not self._dry and self._endpos is not None:
            self._refresh_pose_from_arm()
            current_rpy = self._rpy.copy()
            candidates = (
                current_rpy,
                np.zeros(3, dtype=np.float64),
            )
            ok = False
            used_rpy = None
            for rpy in candidates:
                ok = self._endpos.move_to_traj(
                    x=float(target[0]),
                    y=float(target[1]),
                    z=float(target[2]),
                    roll=float(rpy[0]),
                    pitch=float(rpy[1]),
                    yaw=float(rpy[2]),
                    duration=1.5,
                )
                if not ok:
                    ok = self._endpos.move_to_ik(
                        x=float(target[0]),
                        y=float(target[1]),
                        z=float(target[2]),
                        roll=float(rpy[0]),
                        pitch=float(rpy[1]),
                        yaw=float(rpy[2]),
                    )
                if ok:
                    used_rpy = rpy
                    break
            if ok:
                self._pos = target
                if used_rpy is not None:
                    self._rpy = np.array(used_rpy, dtype=np.float64)
                self._last_move_t = time.monotonic()
                time.sleep(1.0)
            else:
                print("[arm] home move failed; staying at current pose")
                self._refresh_pose_from_arm()
        else:
            self._pos = target
        self._homing = False

    def stop(self) -> None:
        if not self._dry and self._endpos is not None:
            self._endpos.end()


class ActionLogger:
    def __init__(self) -> None:
        self._last_grip = ""
        self._last_axis = "idle"
        self._last_motion = ""

    def log_gripper(self, pinch_state: str, pinch_value: float, open_frac: float) -> None:
        pct = int(open_frac * 100.0)
        state = pinch_state
        if state != self._last_grip:
            print(f"[gripper] {state:7s}  pinch={pinch_value:.2f}  grip={pct}%")
            self._last_grip = state

    def log_arm(self, axis: str | None, dx: float, dy: float, dz: float) -> None:
        axis_state = axis or "idle"
        if axis_state != self._last_axis:
            print(f"[arm] axis={axis_state}")
            self._last_axis = axis_state
        motion = self._motion_label(dx, dy, dz)
        if motion != self._last_motion:
            if motion:
                print(f"[arm] move={motion}  delta=({dx:+.4f}, {dy:+.4f}, {dz:+.4f})")
            self._last_motion = motion

    def log_home(self, source: str) -> None:
        print(f"[arm] {source} -> home")

    @staticmethod
    def _motion_label(dx: float, dy: float, dz: float) -> str:
        if dx > 0 or dx < 0 or dy > 0 or dy < 0 or dz > 0 or dz < 0:
            command = None
            if dx > 0:
                command = "x_forward"
            elif dx < 0:
                command = "x_back"
            elif dy > 0:
                command = "y_left"
            elif dy < 0:
                command = "y_right"
            elif dz > 0:
                command = "z_up"
            elif dz < 0:
                command = "z_down"
            return _command_label(command)
        return ""


def _draw_polyline(frame: np.ndarray, pts_px: np.ndarray, connections, color: tuple[int, int, int]) -> None:
    for start, end in connections:
        pt1 = tuple(pts_px[start].astype(int))
        pt2 = tuple(pts_px[end].astype(int))
        cv2.line(frame, pt1, pt2, color, 2)


def _draw_pose(frame: np.ndarray, pose_landmarks, color: tuple[int, int, int]) -> None:
    if not pose_landmarks:
        return
    height, width = frame.shape[:2]
    pts_px = np.array([[lm.x * width, lm.y * height] for lm in pose_landmarks], dtype=np.float64)
    _draw_polyline(frame, pts_px, POSE_CONNECTIONS, color)
    for index in [11, 12, 13, 14, 15, 16]:
        if index < len(pose_landmarks):
            visibility = float(getattr(pose_landmarks[index], "visibility", 1.0) or 1.0)
            if visibility >= POSE_VIS_MIN:
                cv2.circle(frame, tuple(pts_px[index].astype(int)), 5, color, -1)


def _draw_hand(frame: np.ndarray, hand: HandObservation | None, color: tuple[int, int, int]) -> None:
    if hand is None:
        return
    _draw_polyline(frame, hand.pts_px, HAND_CONNECTIONS, color)
    for point in hand.pts_px:
        cv2.circle(frame, tuple(point.astype(int)), 3, color, -1)


def _draw_hand_markers(frame: np.ndarray, hand: HandObservation | None, pinch_value: float, color: tuple[int, int, int]) -> None:
    if hand is None:
        return
    thumb = tuple(hand.pts_px[4].astype(int))
    index = tuple(hand.pts_px[8].astype(int))
    cv2.line(frame, thumb, index, color, 3)
    cv2.circle(frame, thumb, 8, (255, 220, 0), -1)
    cv2.circle(frame, index, 8, (0, 180, 255), -1)
    cv2.putText(
        frame,
        f"{hand.side} pinch {pinch_value:.2f}",
        (hand.wrist_px[0] - 40, hand.wrist_px[1] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
    )


def _sanitize_rot_delta(dr: float, dp: float, dy: float) -> np.ndarray:
    delta = np.array([dr, dp, dy], dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        return np.zeros(3, dtype=np.float64)
    return np.clip(delta, -ROT_STEP_MAX, ROT_STEP_MAX)


def _rotation_command_label(command: str | None) -> str:
    labels = {
        "roll_pos": "Roll+",
        "roll_neg": "Roll-",
        "pitch_pos": "Pitch+",
        "pitch_neg": "Pitch-",
        "yaw_pos": "Yaw+",
        "yaw_neg": "Yaw-",
    }
    return labels.get(command, "")


def _rotation_delta_for_command(command: str | None) -> tuple[float, float, float]:
    if command == "x_forward":
        return -ROT_STEP_MAX, 0.0, 0.0
    if command == "x_back":
        return ROT_STEP_MAX, 0.0, 0.0
    if command == "z_up":
        return 0.0, ROT_STEP_MAX, 0.0
    if command == "z_down":
        return 0.0, -ROT_STEP_MAX, 0.0
    if command == "y_left":
        return 0.0, 0.0, -ROT_STEP_MAX
    if command == "y_right":
        return 0.0, 0.0, ROT_STEP_MAX
    return 0.0, 0.0, 0.0


class BothPalmsModeDetector:
    def __init__(self) -> None:
        self._recent: list[bool] = []
        self._last_trigger_t = 0.0
        self._latched = False

    def update(
        self,
        left_hand: HandObservation | None,
        right_hand: HandObservation | None,
    ) -> bool:
        both_open = (
            left_hand is not None
            and right_hand is not None
            and _palm_open(left_hand)
            and _palm_open(right_hand)
        )
        if not both_open:
            self._latched = False
        self._recent.append(both_open)
        if len(self._recent) > ROT_MODE_HOLD_FRAMES:
            self._recent.pop(0)

        now = time.monotonic()
        if (
            not self._latched
            and len(self._recent) == ROT_MODE_HOLD_FRAMES
            and all(self._recent)
            and (now - self._last_trigger_t) > ROT_MODE_COOLDOWN_S
        ):
            self._latched = True
            self._last_trigger_t = now
            self._recent.clear()
            return True
        return False


class ConfirmedRightHandGestureController:
    def __init__(self) -> None:
        self.active_command: str | None = None
        self._candidate_command: str | None = None
        self._candidate_frames = 0
        self._neutral_frames = 0
        self._blocked_command: str | None = None
        self._blocked_until = 0.0

    def update(self, hand: HandObservation | None) -> str | None:
        now = time.monotonic()
        candidate = _right_hand_command(hand)
        if self._blocked_command == candidate and now < self._blocked_until:
            candidate = None
        elif now >= self._blocked_until:
            self._blocked_command = None

        if candidate == self._candidate_command:
            self._candidate_frames += 1
        else:
            self._candidate_command = candidate
            self._candidate_frames = 1

            if candidate is None:
                self._neutral_frames += 1
            if self._neutral_frames >= GESTURE_RELEASE_FRAMES:
                self.active_command = None
            return None

        self._neutral_frames = 0
        if candidate != self.active_command and self._candidate_frames >= GESTURE_CONFIRM_FRAMES:
            self.active_command = candidate
        if self.active_command is None or candidate != self.active_command:
            return None
        return self.active_command

    def block_active_command(self) -> None:
        if self.active_command is None:
            return
        self._blocked_command = self.active_command
        self._blocked_until = time.monotonic() + FAILED_COMMAND_COOLDOWN
        self.reset()

    def reset(self) -> None:
        self.active_command = None
        self._candidate_command = None
        self._candidate_frames = 0
        self._neutral_frames = 0


class MotionModeController:
    def __init__(self) -> None:
        self.mode = "translate"

    def toggle_rotation_mode(self) -> None:
        if self.mode == "translate":
            self.mode = "rotate"
        else:
            self.mode = "translate"

    def reset_translation(self) -> None:
        self.mode = "translate"

    def update(self, command: str | None) -> tuple[np.ndarray, np.ndarray]:
        translation = np.zeros(3, dtype=np.float64)
        rotation = np.zeros(3, dtype=np.float64)

        if self.mode == "translate":
            if command is not None:
                translation[:] = _command_to_delta(command)
            return translation, rotation

        if self.mode == "rotate":
            if command is not None:
                rotation[:] = _rotation_delta_for_command(command)

        return translation, rotation

    def active_rotation_label(self) -> str:
        labels = {
            "rotate": "Rotate",
            "translate": "Translate",
        }
        return labels.get(self.mode, self.mode)


class ArmController6DOF(ArmController):
    def __init__(self, no_arm: bool, hw_yaml: str | None = None,
                 vlim_cap: float | None = None) -> None:
        super().__init__(no_arm=no_arm, hw_yaml=hw_yaml, vlim_cap=vlim_cap)
        self._rpy_home = self._rpy.copy()
        self._smoothed_rot_delta = np.zeros(3, dtype=np.float64)

    @property
    def rpy(self) -> np.ndarray:
        return self._rpy.copy()

    def _clamp_rpy(self, rpy: np.ndarray) -> np.ndarray:
        half_span = np.array([ROLL_LIMIT, PITCH_LIMIT, YAW_LIMIT], dtype=np.float64)
        return np.clip(rpy, self._rpy_home - half_span, self._rpy_home + half_span)

    def move_rotation_delta(self, dr: float, dp: float, dy: float) -> bool:
        if self._homing:
            self._smoothed_rot_delta[:] = 0.0
            return False
        desired_delta = _sanitize_rot_delta(dr, dp, dy)
        self._smoothed_rot_delta = (
            (1.0 - ROT_SMOOTHING_ALPHA) * self._smoothed_rot_delta
            + ROT_SMOOTHING_ALPHA * desired_delta
        )
        if np.linalg.norm(desired_delta) < 1e-9 and np.linalg.norm(self._smoothed_rot_delta) < ROT_SMOOTHING_ZERO:
            self._smoothed_rot_delta[:] = 0.0
            return False

        now = time.monotonic()
        if now - self._last_move_t < ARM_MOVE_INTERVAL:
            return False

        delta = _sanitize_rot_delta(*self._smoothed_rot_delta)
        if np.allclose(delta, 0.0, atol=1e-9):
            return False

        new_rpy = self._clamp_rpy(self._rpy + delta)
        if np.allclose(new_rpy, self._rpy, atol=2e-3):
            self._smoothed_rot_delta[:] = 0.0
            return False

        if not self._dry and self._endpos is not None:
            ok = self._endpos.move_to_ik(
                x=float(self._pos[0]),
                y=float(self._pos[1]),
                z=float(self._pos[2]),
                roll=float(new_rpy[0]),
                pitch=float(new_rpy[1]),
                yaw=float(new_rpy[2]),
            )
            if not ok:
                self._smoothed_rot_delta[:] = 0.0
                return False
        self._rpy = new_rpy
        self._last_move_t = now
        return True

    def go_home(self) -> None:
        if self._homing:
            return
        self._homing = True
        self._smoothed_delta[:] = 0.0
        self._smoothed_rot_delta[:] = 0.0
        target = _clamp_workspace(ARM_HOME)
        # Home can fail due to orientation feasibility. Try a small ladder of
        # orientations to ensure we can always return to a safe pose.
        self._refresh_pose_from_arm()
        current_rpy = self._rpy.copy()
        target_rpy = self._rpy_home.copy()
        if not self._dry and self._endpos is not None:
            candidates = (
                target_rpy,
                current_rpy,
                np.zeros(3, dtype=np.float64),
            )
            ok = False
            used_rpy = None
            for rpy in candidates:
                ok = self._endpos.move_to_traj(
                    x=float(target[0]),
                    y=float(target[1]),
                    z=float(target[2]),
                    roll=float(rpy[0]),
                    pitch=float(rpy[1]),
                    yaw=float(rpy[2]),
                    duration=1.5,
                )
                if not ok:
                    ok = self._endpos.move_to_ik(
                        x=float(target[0]),
                        y=float(target[1]),
                        z=float(target[2]),
                        roll=float(rpy[0]),
                        pitch=float(rpy[1]),
                        yaw=float(rpy[2]),
                    )
                if ok:
                    used_rpy = rpy
                    break
            if ok:
                self._pos = target
                if used_rpy is not None:
                    self._rpy = np.array(used_rpy, dtype=np.float64)
                self._last_move_t = time.monotonic()
                time.sleep(1.0)
            else:
                print("[arm] home move failed; staying at current pose")
                self._refresh_pose_from_arm()
        else:
            self._pos = target
            self._rpy = target_rpy
        self._homing = False


class SixDofLogger(ActionLogger):
    def __init__(self) -> None:
        super().__init__()
        self._last_mode = ""
        self._last_rot_motion = ""

    def log_mode(self, mode: str) -> None:
        if mode != self._last_mode:
            print(f"[mode] {mode}")
            self._last_mode = mode

    def log_rotation(self, mode: str, dr: float, dp: float, dy: float) -> None:
        motion = ""
        if mode == "rotate":
            motion = (
                _rotation_command_label("roll_pos" if dr > 0 else "roll_neg" if dr < 0 else None)
                or _rotation_command_label("pitch_pos" if dp > 0 else "pitch_neg" if dp < 0 else None)
                or _rotation_command_label("yaw_pos" if dy > 0 else "yaw_neg" if dy < 0 else None)
            )
        if motion != self._last_rot_motion:
            if motion:
                print(f"[arm] rotate={motion}  delta_rpy=({dr:+.4f}, {dp:+.4f}, {dy:+.4f})")
            self._last_rot_motion = motion


def _draw_overlay(
    frame: np.ndarray,
    pose_landmarks,
    left_hand: HandObservation | None,
    right_hand: HandObservation | None,
    pinch_value: float,
    pinch_state: str,
    grip_frac: float,
    active_command: str | None,
    mode_controller: MotionModeController,
    dx: float,
    dy: float,
    dz: float,
    dr: float,
    dp: float,
    dyaw: float,
    arm_pos: np.ndarray,
    arm_rpy: np.ndarray,
    home_flash: bool,
    is_homing: bool,
) -> None:
    _draw_pose(frame, pose_landmarks, (120, 180, 255))
    _draw_hand(frame, left_hand, (50, 220, 120))
    _draw_hand(frame, right_hand, (255, 180, 80))
    _draw_hand_markers(frame, left_hand, pinch_value, (50, 220, 120))

    if right_hand is not None:
        label = _command_label(active_command) or "idle"
        cv2.putText(
            frame,
            f"right cmd={label}",
            (right_hand.wrist_px[0] - 40, right_hand.wrist_px[1] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 180, 80),
            2,
        )

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 116), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Mode: {mode_controller.active_rotation_label()}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Left hand: pinch<{PINCH_THRESHOLD:.2f} closes, else opens  ({int(grip_frac * 100):d}%)",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (80, 220, 120),
        2,
    )
    cv2.putText(
        frame,
        "Translate: open palm+thumb R/L = X, closed thumb L/R = Y, thumb U/D = Z",
        (12, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 200, 120),
        1,
    )
    cv2.putText(
        frame,
        "Both palms open = toggle rotate mode. Rotate: X->roll, Y->yaw, Z->pitch. Both fists/clap = HOME",
        (12, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 200, 120),
        1,
    )

    status_text = "HOME TRIGGERED" if home_flash or is_homing else f"Gripper: {pinch_state}"
    cv2.putText(
        frame,
        status_text,
        (12, height - 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 180, 180),
        1,
    )

    trans_motion = ActionLogger._motion_label(dx, dy, dz) or "hold"
    rot_motion = (
        _rotation_command_label("roll_pos" if dr > 0 else "roll_neg" if dr < 0 else None)
        or _rotation_command_label("pitch_pos" if dp > 0 else "pitch_neg" if dp < 0 else None)
        or _rotation_command_label("yaw_pos" if dyaw > 0 else "yaw_neg" if dyaw < 0 else None)
        or "hold"
    )
    cv2.putText(
        frame,
        f"Translate: {trans_motion}   Rotate: {rot_motion}",
        (12, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Arm pos: {arm_pos[0]:.3f}  {arm_pos[1]:+.3f}  {arm_pos[2]:.3f}",
        (12, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Arm rpy: {arm_rpy[0]:+.3f}  {arm_rpy[1]:+.3f}  {arm_rpy[2]:+.3f}",
        (360, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 200, 255),
        1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bimanual 6DOF hand control demo")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--hw-yaml", default=None,
                        help="hardware config yaml (default: from config/rebotarm.yaml)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    model_path = _ensure_holistic_model()
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.55,
        min_pose_landmarks_confidence=0.55,
        min_hand_landmarks_confidence=0.55,
        output_segmentation_mask=False,
        output_face_blendshapes=False,
    )
    landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(options)

    arm = ArmController6DOF(no_arm=args.no_arm, hw_yaml=args.hw_yaml)
    gripper = GripperController(
        no_gripper=args.no_gripper,
        endpose=arm.endpose,
    )
    gesture_controller = ConfirmedRightHandGestureController()
    mode_controller = MotionModeController()
    clap = ClapDetector()
    fist_home = DualFistHomeDetector()
    palms_mode = BothPalmsModeDetector()
    logger = SixDofLogger()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Bimanual 6DOF hand control")
    print("  Left hand pinch/unpinch controls the gripper")
    print("  Translation mode uses the original XYZ gestures")
    print("  Both palms open toggles rotate mode")
    print("  Rotate mode: X->roll, Y->yaw, Z->pitch")
    print("  Clap or hold both fists closed = HOME")
    print("  Q or Esc quits")

    pinch_value = -1.0
    window_name = "ReBot - Bimanual 6DOF Hand Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    should_exit = False
    frame_timestamp_ms = 0
    home_flash = 0

    def _handle_signal(_sig, _frame) -> None:
        nonlocal should_exit
        should_exit = True

    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not should_exit:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33

            pose_points = _extract_pose_points(getattr(result, "pose_landmarks", None), frame.shape)
            hand_map = _build_hand_observations(result, frame.shape)
            left_hand = hand_map.get("left")
            right_hand = hand_map.get("right")

            if clap.update(pose_points):
                home_flash = 15
                logger.log_home("clap")
                gesture_controller.reset()
                mode_controller.reset_translation()
                threading.Thread(target=arm.go_home, daemon=True).start()
            elif fist_home.update(left_hand, right_hand):
                home_flash = 15
                logger.log_home("both_fists")
                gesture_controller.reset()
                mode_controller.reset_translation()
                threading.Thread(target=arm.go_home, daemon=True).start()
            elif palms_mode.update(left_hand, right_hand):
                gesture_controller.reset()
                mode_controller.toggle_rotation_mode()
            elif home_flash > 0:
                home_flash -= 1

            pinch_value = _pinch_openness(left_hand)
            pinch_state = _pinch_state(pinch_value)
            gripper.update(pinch_state)
            logger.log_gripper(pinch_state, pinch_value, gripper.open_frac)

            command = gesture_controller.update(right_hand)
            translation_delta, rotation_delta = mode_controller.update(command)
            logger.log_mode(mode_controller.mode)

            moved = False
            if arm.homing:
                translation_delta[:] = 0.0
                rotation_delta[:] = 0.0
            elif np.linalg.norm(rotation_delta) > 0.0:
                moved = arm.move_rotation_delta(*rotation_delta)
                if not moved:
                    gesture_controller.block_active_command()
                    rotation_delta[:] = 0.0
            else:
                moved = arm.move_delta(*translation_delta)
                if np.linalg.norm(translation_delta) > 0.0 and not moved:
                    gesture_controller.block_active_command()
                    translation_delta[:] = 0.0

            logger.log_arm(
                gesture_controller.active_command if mode_controller.mode == "translate" else None,
                float(translation_delta[0]),
                float(translation_delta[1]),
                float(translation_delta[2]),
            )
            logger.log_rotation(
                mode_controller.mode,
                float(rotation_delta[0]),
                float(rotation_delta[1]),
                float(rotation_delta[2]),
            )

            _draw_overlay(
                frame,
                getattr(result, "pose_landmarks", None),
                left_hand,
                right_hand,
                pinch_value,
                pinch_state,
                gripper.open_frac,
                gesture_controller.active_command,
                mode_controller,
                float(translation_delta[0]),
                float(translation_delta[1]),
                float(translation_delta[2]),
                float(rotation_delta[0]),
                float(rotation_delta[1]),
                float(rotation_delta[2]),
                arm.pos,
                arm.rpy,
                bool(home_flash),
                arm.homing,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        arm.stop()
        gripper.stop()


if __name__ == "__main__":
    main()
