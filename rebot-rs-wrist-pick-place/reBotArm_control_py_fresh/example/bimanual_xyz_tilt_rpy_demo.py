#!/usr/bin/env python3
"""Bimanual demo: right hand XYZ gestures + direct wrist-joint claw orientation.

Controls:

Right hand (translate XYZ; restored working gestures):
  - open palm + thumb right -> X+
  - open palm + thumb left  -> X-
  - thumb up                -> Z+
  - thumb down              -> Z-
  - closed thumb left       -> Y+
  - closed thumb right      -> Y-

Left hand:
  - normal arm mode: open palm twist controls joint6 clockwise/counter-clockwise;
    thumb up/down controls joint4; thumb left/right controls joint5 only when
    the palm is not open. Open palm rotation never moves joint4 or joint5.
  - gripper mode: pinch closed/open closes/opens the gripper gradually.

Mode:
  - hands below shoulders -> arm/wrist mode
  - both hands above shoulders -> gripper mode

Home:
  - touch both hands together or clap -> home

Optional arm camera:
  - --arm-camera INDEX shows a separate camera window pointed at the arm (default: 4)
  - --no-arm-camera-display disables that second preview
  - --record-arm-camera records that camera to an MP4 file
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bimanual_hand_control_6dof_demo as core
from reBotArm_control_py.kinematics import (
    compute_fk,
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
)
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik


AIM_ALPHA = 0.16
AIM_HOLD_FRAMES = 3
AIM_DEADZONE_RAD = 0.035
WRIST_TWIST_STEP_MAX = 0.004
AIM_ROLL_GAIN = 0.60
THUMB_ROT_DEADZONE = 0.34
THUMB_ROT_DOMINANCE_MARGIN = 0.08
THUMB_JOINT_STEP_MAX = 0.0035
THUMB_MIN_REACH_RATIO = 0.55
WRIST_JOINT_SAFETY_SCALE = 0.15
WRIST_DIRECT_VLIM = 0.25
WRIST_DIRECT_STEP_MAX = 0.004
WRIST_FK_IK_TOLERANCE = 2.0e-2
WRIST_FK_IK_MAX_ITER = 180
WRIST_IK_JOINT_DRIFT_MAX = 0.35
WRIST_POS_MARGIN = 0.015
ROLL_LIMIT = 0.20
PITCH_LIMIT = 0.18
YAW_LIMIT = 0.25

PINCH_EMA_ALPHA = 0.22
PINCH_CLOSE_THRESH = 0.22
PINCH_OPEN_THRESH = 0.32

GRIPPER_MODE_HOLD_FRAMES = 3
GRIPPER_MODE_WRIST_MARGIN_FRAC = 0.08
HANDS_TOUCH_DIST_FRAC = 0.45
HANDS_TOUCH_HOLD_FRAMES = 4
HANDS_TOUCH_COOLDOWN_S = 2.5

PALM_AIM_MIN_EXTENDED_FINGERS = 3
PALM_AIM_EXTENSION_RATIO = 1.08
WRIST_LOG_EPS = 0.0015
WRIST_JOINT_INDICES = np.array([3, 4, 5], dtype=np.int64)
core.GESTURE_CONFIRM_FRAMES = 3
core.GESTURE_RELEASE_FRAMES = 5
core.THUMB_REACH_MIN = 0.45
core.THUMB_AXIS_MIN = 0.40


@dataclass(frozen=True)
class MotionProfile:
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    z_limits: tuple[float, float]
    min_z: float
    move_interval: float
    arm_step_min: float
    arm_step_max: float
    arm_smoothing_alpha: float
    arm_smoothing_zero: float
    aim_alpha: float
    wrist_twist_step: float
    thumb_joint_step: float
    wrist_safety_scale: float
    wrist_vlim: float
    wrist_direct_step: float
    rpy_limits: tuple[float, float, float]
    arm_vlim_cap: float | None
    gripper_step: float
    gripper_vlim: float
    gripper_open_cap: float
    y_step_scale: float = 1.0   # <1.0 slows sideways (Y) motion vs X/Z


MOTION_PROFILES = {
    "slow": MotionProfile(
        x_limits=(0.26, 0.34),
        y_limits=(-0.06, 0.06),
        z_limits=(0.19, 0.26),
        min_z=0.15,
        move_interval=0.12,
        arm_step_min=0.0005,
        arm_step_max=0.0015,
        arm_smoothing_alpha=0.08,
        arm_smoothing_zero=0.00025,
        aim_alpha=0.08,
        wrist_twist_step=0.0015,
        thumb_joint_step=0.0012,
        wrist_safety_scale=0.08,
        wrist_vlim=0.12,
        wrist_direct_step=0.0015,
        rpy_limits=(0.12, 0.10, 0.15),
        arm_vlim_cap=0.12,
        gripper_step=0.01,
        gripper_vlim=0.35,
        gripper_open_cap=0.25,
    ),
    "default": MotionProfile(
        x_limits=(0.22, 0.40),
        y_limits=(-0.17, 0.17),
        z_limits=(0.17, 0.30),
        min_z=0.13,
        move_interval=0.05,
        arm_step_min=0.003,
        arm_step_max=0.010,
        arm_smoothing_alpha=0.28,
        arm_smoothing_zero=0.0008,
        aim_alpha=0.26,
        wrist_twist_step=0.013,
        thumb_joint_step=0.0095,
        wrist_safety_scale=0.75,
        wrist_vlim=0.85,
        wrist_direct_step=0.013,
        rpy_limits=(0.45, 0.40, 0.55),
        arm_vlim_cap=None,
        gripper_step=0.035,
        gripper_vlim=2.2,
        gripper_open_cap=0.70,
        y_step_scale=0.5,   # wider Y reach, but move through it gently
    ),
}


def _apply_motion_profile(profile: MotionProfile) -> None:
    global AIM_ALPHA, WRIST_TWIST_STEP_MAX, THUMB_JOINT_STEP_MAX
    global WRIST_JOINT_SAFETY_SCALE, WRIST_DIRECT_VLIM, WRIST_DIRECT_STEP_MAX
    global ROLL_LIMIT, PITCH_LIMIT, YAW_LIMIT

    core.ARM_X_MIN, core.ARM_X_MAX = profile.x_limits
    core.ARM_Y_MIN, core.ARM_Y_MAX = profile.y_limits
    core.ARM_Z_MIN, core.ARM_Z_MAX = profile.z_limits
    core.ARM_MOVE_INTERVAL = profile.move_interval
    core.ARM_STEP_MIN = profile.arm_step_min
    core.ARM_STEP_MAX = profile.arm_step_max
    core.ARM_MAX_DELTA = profile.arm_step_max
    core.ARM_Y_STEP_SCALE = profile.y_step_scale
    core.ARM_SMOOTHING_ALPHA = profile.arm_smoothing_alpha
    core.ARM_SMOOTHING_ZERO = profile.arm_smoothing_zero
    core.GRIPPER_STEP = profile.gripper_step
    core.GRIPPER_VLIM = profile.gripper_vlim
    core.GRIPPER_MAX_OPEN_FRACTION = profile.gripper_open_cap
    AIM_ALPHA = profile.aim_alpha
    WRIST_TWIST_STEP_MAX = profile.wrist_twist_step
    THUMB_JOINT_STEP_MAX = profile.thumb_joint_step
    WRIST_JOINT_SAFETY_SCALE = profile.wrist_safety_scale
    WRIST_DIRECT_VLIM = profile.wrist_vlim
    WRIST_DIRECT_STEP_MAX = profile.wrist_direct_step
    ROLL_LIMIT, PITCH_LIMIT, YAW_LIMIT = profile.rpy_limits


_apply_motion_profile(MOTION_PROFILES["slow"])


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _resolve_serial_port(requested: str | None, arm_cfg: Path) -> str:
    """Select a serial device safely; never guess when multiple adapters exist."""
    if requested is not None:
        port = Path(requested)
        if not port.exists():
            raise RuntimeError(f"Requested serial port does not exist: {port}")
        return str(port)

    with arm_cfg.open("r") as stream:
        configured = Path(str(yaml.safe_load(stream).get("channel", "")))
    if configured.exists():
        return str(configured)

    candidates = sorted(Path("/dev").glob("ttyACM*"))
    if len(candidates) == 1:
        selected = str(candidates[0])
        print(f"[serial] Configured {configured} is absent; auto-selected {selected}")
        return selected
    if not candidates:
        raise RuntimeError(
            f"Configured serial port {configured} is absent and no /dev/ttyACM* device was found. "
            "Connect the USB-CAN adapter or pass --serial-port explicitly."
        )
    raise RuntimeError(
        "Configured serial port is absent and multiple ACM devices were found: "
        f"{', '.join(str(path) for path in candidates)}. Pass --serial-port explicitly."
    )


@dataclass
class PalmFrame:
    across: np.ndarray
    forward: np.ndarray


def _hand_frame(hand: core.HandObservation | None) -> PalmFrame | None:
    if hand is None:
        return None
    wrist = hand.pts_norm[0, :3]
    index_mcp = hand.pts_norm[5, :3]
    middle_mcp = hand.pts_norm[9, :3]
    pinky_mcp = hand.pts_norm[17, :3]

    across = pinky_mcp - index_mcp
    forward = middle_mcp - wrist

    across_n = across / (np.linalg.norm(across) + 1e-9)
    forward_n = forward / (np.linalg.norm(forward) + 1e-9)
    return PalmFrame(across=across_n, forward=forward_n)


def _aim_finger_extended(hand: core.HandObservation, tip_idx: int, pip_idx: int) -> bool:
    wrist = hand.pts_norm[0, :2]
    tip = hand.pts_norm[tip_idx, :2]
    pip = hand.pts_norm[pip_idx, :2]
    return core._dist(tip, wrist) > PALM_AIM_EXTENSION_RATIO * core._dist(pip, wrist)


def _aim_palm_open(hand: core.HandObservation | None) -> bool:
    if hand is None:
        return False
    extended_count = sum(
        (
            _aim_finger_extended(hand, 8, 6),
            _aim_finger_extended(hand, 12, 10),
            _aim_finger_extended(hand, 16, 14),
            _aim_finger_extended(hand, 20, 18),
        )
    )
    return extended_count >= PALM_AIM_MIN_EXTENDED_FINGERS


def _roll_feature(frame: PalmFrame) -> float:
    """Clockwise/counter-clockwise palm twist in the image plane."""
    return float(np.arctan2(frame.across[1], frame.across[0] + 1e-9))


def _scale_axis(value: float, deadzone: float) -> float:
    value_abs = abs(value)
    if value_abs <= deadzone:
        return 0.0
    return float(np.sign(value) * (value_abs - deadzone) / max(1.0 - deadzone, 1e-6))


def _thumb_joint4_joint5(hand: core.HandObservation) -> tuple[float, float]:
    """Use the left thumb as a clean joint4/joint5 switch.

    Image y decreases upward, so thumb-up maps to positive joint4. Only the
    dominant thumb axis is accepted to avoid diagonal noise mixing joint4/joint5.
    """
    wrist = hand.pts_norm[0, :2]
    thumb_tip = hand.pts_norm[4, :2]
    middle_mcp = hand.pts_norm[9, :2]
    vec = thumb_tip - wrist
    thumb_reach = core._dist(thumb_tip, wrist) / max(core._dist(middle_mcp, wrist), 1e-6)
    if thumb_reach < THUMB_MIN_REACH_RATIO:
        return 0.0, 0.0

    norm = float(np.linalg.norm(vec)) + 1e-9
    vx = float(vec[0] / norm)
    vy = float(vec[1] / norm)

    pitch_axis = -vy
    yaw_axis = -vx    # thumb left/right -> joint5 (negated so gesture matches motion)
    if abs(pitch_axis) > abs(yaw_axis) + THUMB_ROT_DOMINANCE_MARGIN:
        return _scale_axis(pitch_axis, THUMB_ROT_DEADZONE), 0.0
    if abs(yaw_axis) > abs(pitch_axis) + THUMB_ROT_DOMINANCE_MARGIN:
        return 0.0, _scale_axis(yaw_axis, THUMB_ROT_DEADZONE)
    return 0.0, 0.0


class WristJointAimController:
    """Left palm/thumb controls wrist joints 4, 5, and 6 directly."""

    def __init__(self) -> None:
        self._neutral_features: np.ndarray | None = None
        self._neutral_wrist_q: np.ndarray | None = None
        self._smoothed = np.zeros(3, dtype=np.float64)
        self._open_frames = 0
        self._active = False
        self._state = "paused"

    def reset(self) -> None:
        self._neutral_features = None
        self._neutral_wrist_q = None
        self._smoothed[:] = 0.0
        self._open_frames = 0
        self._active = False
        self._state = "paused"

    def _reset_twist_only(self) -> None:
        self._neutral_features = None
        self._neutral_wrist_q = None
        self._open_frames = 0

    def _smooth_and_return(self, raw_delta: np.ndarray, state: str) -> np.ndarray:
        self._smoothed = (1.0 - AIM_ALPHA) * self._smoothed + AIM_ALPHA * raw_delta
        self._active = bool(np.linalg.norm(raw_delta) >= 1e-9 or np.linalg.norm(self._smoothed) >= WRIST_LOG_EPS)
        self._state = state
        return self._smoothed.copy()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def state(self) -> str:
        return self._state

    def update(self, hand: core.HandObservation | None, current_wrist_q: np.ndarray) -> np.ndarray:
        if hand is None:
            self.reset()
            self._state = "missing_left_hand"
            return np.zeros(3, dtype=np.float64)

        if not _aim_palm_open(hand):
            joint4_cmd, joint5_cmd = _thumb_joint4_joint5(hand)
            joint4_delta = _clamp(joint4_cmd * THUMB_JOINT_STEP_MAX, -THUMB_JOINT_STEP_MAX, THUMB_JOINT_STEP_MAX)
            joint5_delta = _clamp(joint5_cmd * THUMB_JOINT_STEP_MAX, -THUMB_JOINT_STEP_MAX, THUMB_JOINT_STEP_MAX)
            self._reset_twist_only()
            raw_delta = np.array([joint4_delta, joint5_delta, 0.0], dtype=np.float64)
            state = "thumb_only" if np.linalg.norm(raw_delta) >= 1e-9 else "left_thumb_idle"
            return self._smooth_and_return(raw_delta, state)

        frame = _hand_frame(hand)
        if frame is None:
            self._reset_twist_only()
            return self._smooth_and_return(np.zeros(3, dtype=np.float64), "bad_twist_frame")

        self._open_frames += 1
        if self._open_frames < AIM_HOLD_FRAMES:
            return self._smooth_and_return(np.zeros(3, dtype=np.float64), "warming_left_palm")

        twist_now = _roll_feature(frame)
        if self._neutral_features is None:
            self._neutral_features = np.array([twist_now], dtype=np.float64)
            self._neutral_wrist_q = current_wrist_q.copy()
            return self._smooth_and_return(np.zeros(3, dtype=np.float64), "neutral_set")

        joint6_offset = _wrap_angle(twist_now - self._neutral_features[0]) * AIM_ROLL_GAIN
        if abs(joint6_offset) < AIM_DEADZONE_RAD:
            joint6_offset = 0.0
        if not np.isfinite(joint6_offset) or self._neutral_wrist_q is None:
            self.reset()
            self._state = "bad_rotation_offset"
            return np.zeros(3, dtype=np.float64)

        target_joint6 = self._neutral_wrist_q[2] + joint6_offset
        joint6_delta = _clamp(target_joint6 - current_wrist_q[2], -WRIST_TWIST_STEP_MAX, WRIST_TWIST_STEP_MAX)
        raw_delta = np.array([0.0, 0.0, joint6_delta], dtype=np.float64)
        return self._smooth_and_return(raw_delta, "aiming")


class WristLogger:
    def __init__(self) -> None:
        self._last_state = ""
        self._last_motion = ""
        self._last_block = ""

    def log(self, state: str, dq4: float, dq5: float, dq6: float) -> None:
        if state != self._last_state:
            print(f"[wrist] state={state}")
            self._last_state = state

        motion = self._motion_label(dq4, dq5, dq6)
        if motion != self._last_motion:
            if motion:
                print(f"[wrist] move={motion}  delta=({dq4:+.4f}, {dq5:+.4f}, {dq6:+.4f})")
            self._last_motion = motion

    def log_block(self, reason: str) -> None:
        if reason != self._last_block:
            if reason:
                print(f"[wrist] blocked={reason}")
            self._last_block = reason

    @staticmethod
    def _motion_label(dq4: float, dq5: float, dq6: float) -> str:
        parts: list[str] = []
        if abs(dq4) >= WRIST_LOG_EPS:
            parts.append("J4+" if dq4 > 0 else "J4-")
        if abs(dq5) >= WRIST_LOG_EPS:
            parts.append("J5+" if dq5 > 0 else "J5-")
        if abs(dq6) >= WRIST_LOG_EPS:
            parts.append("J6+" if dq6 > 0 else "J6-")
        return " ".join(parts)


class StablePinch:
    """Hysteresis + EMA pinch gate to avoid gripper chatter from noisy landmarks."""

    def __init__(self) -> None:
        self._ema = -1.0
        self._state = "unknown"  # "pinched" | "open" | "unknown"

    def reset(self) -> None:
        self._ema = -1.0
        self._state = "unknown"

    def update(self, pinch_value: float) -> tuple[str, float]:
        if pinch_value < 0:
            self._ema = -1.0
            self._state = "unknown"
            return "hold", -1.0

        if self._ema < 0:
            self._ema = pinch_value
        else:
            self._ema = (1.0 - PINCH_EMA_ALPHA) * self._ema + PINCH_EMA_ALPHA * pinch_value

        if self._state in ("unknown", "open"):
            if self._ema <= PINCH_CLOSE_THRESH:
                self._state = "pinched"
        if self._state in ("unknown", "pinched"):
            if self._ema >= PINCH_OPEN_THRESH:
                self._state = "open"

        if self._state == "pinched":
            return "closing", float(self._ema)
        if self._state == "open":
            return "opening", float(self._ema)
        return "hold", float(self._ema)


class ShoulderHeightModeDetector:
    """True when both wrists are clearly above their shoulders for a few frames."""

    def __init__(self) -> None:
        self._recent: list[bool] = []

    def update(self, pose_points: dict[str, tuple[float, float, float] | None]) -> bool:
        ls = pose_points.get("left_shoulder")
        rs = pose_points.get("right_shoulder")
        lw = pose_points.get("left_wrist")
        rw = pose_points.get("right_wrist")
        above_shoulder = False
        if any(p is None for p in (ls, rs, lw, rw)):
            self._recent.clear()
            return False

        shoulder_span = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])) + 1e-6
        margin = GRIPPER_MODE_WRIST_MARGIN_FRAC * shoulder_span
        left_above = lw[1] < ls[1] - margin
        right_above = rw[1] < rs[1] - margin
        above_shoulder = left_above and right_above

        self._recent.append(bool(above_shoulder))
        if len(self._recent) > GRIPPER_MODE_HOLD_FRAMES:
            self._recent.pop(0)
        return len(self._recent) == GRIPPER_MODE_HOLD_FRAMES and all(self._recent)


class HandsTouchHomeDetector:
    """True when landmarks from both hands physically touch for a few frames."""

    def __init__(self) -> None:
        self._recent: list[bool] = []
        self._last_trigger_t = 0.0
        self._latched = False

    def update(
        self,
        left_hand: core.HandObservation | None,
        right_hand: core.HandObservation | None,
    ) -> bool:
        touching = False
        if left_hand is not None and right_hand is not None:
            avg_palm = 0.5 * (left_hand.palm_size_px + right_hand.palm_size_px) + 1e-6
            left_points = left_hand.pts_px[[0, 4, 8, 12, 16, 20]]
            right_points = right_hand.pts_px[[0, 4, 8, 12, 16, 20]]
            distances = np.linalg.norm(left_points[:, None, :] - right_points[None, :, :], axis=2)
            touching = float(np.min(distances)) < HANDS_TOUCH_DIST_FRAC * avg_palm

        if not touching:
            self._latched = False

        self._recent.append(touching)
        if len(self._recent) > HANDS_TOUCH_HOLD_FRAMES:
            self._recent.pop(0)

        now = time.monotonic()
        if (
            not self._latched
            and len(self._recent) == HANDS_TOUCH_HOLD_FRAMES
            and all(self._recent)
            and (now - self._last_trigger_t) > HANDS_TOUCH_COOLDOWN_S
        ):
            self._last_trigger_t = now
            self._latched = True
            self._recent.clear()
            return True
        return False


class PreviewWindow:
    """Frame preview that falls back to Tk when OpenCV HighGUI is unavailable."""

    _tk_root = None
    _tk_image_cls = None
    _exit_requested = False

    def __init__(self, title: str, width: int, height: int, x: int, y: int) -> None:
        self._title = title
        self._width = width
        self._height = height
        self._backend = "none"
        self._tk_window = None
        self._tk_label = None
        self._tk_photo = None

        try:
            core.cv2.namedWindow(title, core.cv2.WINDOW_NORMAL | core.cv2.WINDOW_KEEPRATIO)
            core.cv2.resizeWindow(title, width, height)
            core.cv2.moveWindow(title, x, y)
            self._backend = "opencv"
            return
        except core.cv2.error as exc:
            print(f"[display] OpenCV window unavailable for '{title}'; using Tk fallback. Reason: {exc}")

        try:
            import tkinter as tk
            from PIL import ImageTk

            if PreviewWindow._tk_root is None:
                PreviewWindow._tk_root = tk.Tk()
                self._tk_window = PreviewWindow._tk_root
            else:
                self._tk_window = tk.Toplevel(PreviewWindow._tk_root)
            PreviewWindow._tk_image_cls = ImageTk.PhotoImage
            self._tk_window.title(title)
            self._tk_window.geometry(f"{width}x{height}+{x}+{y}")
            self._tk_label = tk.Label(self._tk_window)
            self._tk_label.pack(fill=tk.BOTH, expand=True)
            self._tk_window.bind("<Escape>", self._request_exit)
            self._tk_window.bind("q", self._request_exit)
            self._backend = "tk"
        except Exception as exc:
            print(f"[display] Tk fallback unavailable for '{title}'; preview disabled. Reason: {exc}")

    @property
    def active(self) -> bool:
        return self._backend != "none"

    @property
    def uses_opencv(self) -> bool:
        return self._backend == "opencv"

    @classmethod
    def exit_requested(cls) -> bool:
        return cls._exit_requested

    @staticmethod
    def _request_exit(_event=None) -> None:
        PreviewWindow._exit_requested = True

    def show(self, frame: np.ndarray) -> None:
        if self._backend == "opencv":
            core.cv2.imshow(self._title, frame)
            return
        if self._backend != "tk" or self._tk_window is None or self._tk_label is None:
            return

        from PIL import Image

        rgb = core.cv2.cvtColor(frame, core.cv2.COLOR_BGR2RGB)
        width = max(1, self._tk_label.winfo_width() or self._width)
        height = max(1, self._tk_label.winfo_height() or self._height)
        rgb = core.cv2.resize(rgb, (width, height))
        image = Image.fromarray(rgb)
        self._tk_photo = PreviewWindow._tk_image_cls(image)
        self._tk_label.configure(image=self._tk_photo)
        try:
            self._tk_window.update_idletasks()
            PreviewWindow._tk_root.update()
        except Exception:
            PreviewWindow._exit_requested = True

    @classmethod
    def destroy_all(cls) -> None:
        if cls._tk_root is not None:
            try:
                cls._tk_root.destroy()
            except Exception:
                pass
            cls._tk_root = None


class FrameRecorder:
    """Constant-size MP4 recorder for preview frames."""

    def __init__(self, output_path: Path | None, width: int, height: int, fps: float, prefix: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = output_path or (Path("outputs") / f"{prefix}_{stamp}.mp4")
        self._width = width
        self._height = height
        self._writer = None

        if self._width <= 0 or self._height <= 0:
            self._width = 640
            self._height = 640
        if fps <= 1e-6:
            fps = 30.0

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = core.cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = core.cv2.VideoWriter(str(self._output_path), fourcc, fps, (self._width, self._height))
        if not self._writer.isOpened():
            print(f"[record] Cannot open writer {self._output_path}; recording disabled")
            self._writer.release()
            self._writer = None

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    @property
    def output_path(self) -> Path:
        return self._output_path

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = core.cv2.resize(frame, (self._width, self._height))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class ArmCameraRecorder:
    """Optional second camera for viewing and recording the physical arm."""

    def __init__(
        self,
        camera_index: int | None,
        width: int,
        height: int,
        display: bool,
        record: bool,
        output_path: Path | None,
        fps: float,
        window_x: int,
        window_y: int,
    ) -> None:
        self._display = display
        self._record = record
        self._output_path = output_path
        self._fps = fps
        self._record_width = width
        self._record_height = height
        self._cap = None
        self._recorder = None
        self._window_name = "ReBot - Arm Camera"
        self._enabled = camera_index is not None and (self._display or self._record)

        if not self._enabled:
            return

        self._cap = core.cv2.VideoCapture(int(camera_index))
        if not self._cap.isOpened():
            print(f"[arm-camera] Cannot open camera {camera_index}; arm camera disabled")
            self._cap.release()
            self._cap = None
            self._enabled = False
            return

        if width > 0:
            self._cap.set(core.cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(core.cv2.CAP_PROP_FRAME_HEIGHT, height)

        if self._display:
            self._preview = PreviewWindow(self._window_name, width, height, window_x, window_y)
            self._display = self._preview.active
        else:
            self._preview = None

        if self._record:
            self._open_writer()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def _default_output_path(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("outputs") / f"arm_camera_{stamp}.mp4"

    def _open_writer(self) -> None:
        if self._cap is None:
            return

        fps = self._fps
        if fps <= 0:
            fps = float(self._cap.get(core.cv2.CAP_PROP_FPS)) or 30.0

        self._recorder = FrameRecorder(
            self._output_path or self._default_output_path(),
            self._record_width,
            self._record_height,
            fps,
            "arm_camera",
        )
        self._output_path = self._recorder.output_path
        if not self._recorder.enabled:
            self._record = False

    def update(self) -> None:
        if not self._enabled or self._cap is None:
            return

        ok, frame = self._cap.read()
        if not ok:
            return

        display_frame = core.cv2.flip(frame, 1)
        if self._record:
            core.cv2.circle(display_frame, (20, 22), 7, (0, 0, 255), -1)
            core.cv2.putText(
                display_frame,
                "REC",
                (34, 27),
                core.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            if self._recorder is not None:
                self._recorder.write(display_frame)
        if self._display:
            if self._preview is not None:
                self._preview.show(display_frame)

    def release(self) -> None:
        if self._recorder is not None:
            self._recorder.release()
            self._recorder = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ArmControllerTilt(core.ArmController):
    def __init__(self, no_arm: bool, hw_yaml: str | None = None,
                 vlim_cap: float | None = None) -> None:
        super().__init__(
            no_arm=no_arm,
            hw_yaml=hw_yaml,
            vlim_cap=vlim_cap,
        )
        self._rpy_home = self._rpy.copy()
        self._wrist_q = np.zeros(3, dtype=np.float64)
        self._wrist_min = np.array([-0.6, -0.6, -0.8], dtype=np.float64)
        self._wrist_max = np.array([0.6, 0.6, 0.8], dtype=np.float64)
        self._last_translation_failed = False
        self._last_wrist_block_reason = ""
        self._safety_model = self._endpos._model if self._endpos is not None else load_robot_model()
        self._safety_data = self._safety_model.createData()
        self._safety_end_frame_id = (
            self._endpos._end_frame_id if self._endpos is not None else get_end_effector_frame_id(self._safety_model)
        )
        self._wrist_ik_params = IKParams(
            max_iter=WRIST_FK_IK_MAX_ITER,
            tolerance=WRIST_FK_IK_TOLERANCE,
            step_size=0.5,
            damping=1e-6,
        )
        self._init_wrist_guardrails()

    @property
    def rpy(self) -> np.ndarray:
        return self._rpy.copy()

    @property
    def wrist_q(self) -> np.ndarray:
        return self._wrist_q.copy()

    @property
    def last_translation_failed(self) -> bool:
        return self._last_translation_failed

    @property
    def last_wrist_block_reason(self) -> str:
        return self._last_wrist_block_reason

    def _clamp_rpy(self, rpy: np.ndarray) -> np.ndarray:
        half_span = np.array([ROLL_LIMIT, PITCH_LIMIT, YAW_LIMIT], dtype=np.float64)
        return np.clip(rpy, self._rpy_home - half_span, self._rpy_home + half_span)

    def _init_wrist_guardrails(self) -> None:
        n_arm = 6
        q_current = np.zeros(n_arm, dtype=np.float64)
        if not self._dry and self._arm is not None:
            try:
                q_full, _, _ = self._arm.get_state()
                q_current = np.asarray(q_full[:n_arm], dtype=np.float64)
            except Exception:
                q_current = np.zeros(n_arm, dtype=np.float64)

        self._wrist_q = np.asarray(q_current[WRIST_JOINT_INDICES], dtype=np.float64)
        if self._arm is None:
            return

        # Joint limits come from the URDF model (scaled toward the current
        # pose by WRIST_JOINT_SAFETY_SCALE, same policy as before).
        raw_min = np.asarray(self._safety_model.lowerPositionLimit, dtype=np.float64)[:n_arm][WRIST_JOINT_INDICES]
        raw_max = np.asarray(self._safety_model.upperPositionLimit, dtype=np.float64)[:n_arm][WRIST_JOINT_INDICES]
        finite = np.isfinite(raw_min) & np.isfinite(raw_max)

        safe_min = self._wrist_min.copy()
        safe_max = self._wrist_max.copy()
        if np.any(finite):
            center = self._wrist_q.copy()
            safe_min[finite] = center[finite] + (raw_min[finite] - center[finite]) * WRIST_JOINT_SAFETY_SCALE
            safe_max[finite] = center[finite] + (raw_max[finite] - center[finite]) * WRIST_JOINT_SAFETY_SCALE
        self._wrist_min = np.minimum(safe_min, safe_max)
        self._wrist_max = np.maximum(safe_min, safe_max)
        self._wrist_q = np.clip(self._wrist_q, self._wrist_min, self._wrist_max)

        if not self._dry and self._arm is not None:
            arm_group = self._arm.groups["arm"]
            arm_group._pv_vlim[WRIST_JOINT_INDICES] = np.minimum(
                arm_group._pv_vlim[WRIST_JOINT_INDICES],
                WRIST_DIRECT_VLIM,
            )

    def go_home(self) -> None:
        """Use the same joint-space safe-home path used when the script exits."""
        if self._homing:
            return

        self._homing = True
        self._smoothed_delta[:] = 0.0
        self._last_translation_failed = False
        self._last_wrist_block_reason = ""

        try:
            self._stop_active_trajectory()
            if not self._dry and self._endpos is not None:
                self._endpos.safe_home()
                self._refresh_pose_from_arm()
                q_current, _, _ = self._arm.get_state()
                q_current = np.asarray(q_current[:6], dtype=np.float64)
                self._wrist_q = np.asarray(q_current[WRIST_JOINT_INDICES], dtype=np.float64)
                self._wrist_q = np.clip(self._wrist_q, self._wrist_min, self._wrist_max)
                self._endpos._q_target = q_current.copy()
            else:
                q_home = np.zeros(6, dtype=np.float64)
                self._wrist_q = q_home[WRIST_JOINT_INDICES]
                try:
                    pos_home, rpy_home = core.joint_to_pose(q_home)
                    self._pos = core._clamp_workspace(np.asarray(pos_home, dtype=np.float64))
                    self._rpy = np.asarray(rpy_home, dtype=np.float64)
                except Exception:
                    self._pos = core._clamp_workspace(core.ARM_HOME)
                    self._rpy = np.zeros(3, dtype=np.float64)
            self._last_move_t = time.monotonic()
        finally:
            self._homing = False

    def freeze_motion(self) -> None:
        """Immediately hold arm/wrist motion while the gripper is being controlled."""
        self._smoothed_delta[:] = 0.0
        self._last_translation_failed = False
        self._last_wrist_block_reason = ""
        self._stop_active_trajectory()
        if not self._dry and self._arm is not None and self._endpos is not None:
            try:
                q_current, _, _ = self._arm.get_state()
                self._endpos._q_target = np.asarray(q_current[:6], dtype=np.float64).copy()
                self._wrist_q = np.clip(
                    self._endpos._q_target[WRIST_JOINT_INDICES],
                    self._wrist_min,
                    self._wrist_max,
                )
            except Exception:
                pass
        self._last_move_t = time.monotonic()

    def _stop_active_trajectory(self) -> None:
        if self._endpos is None:
            return
        try:
            self._endpos._stop_send.set()
            send_thread = self._endpos._send_thread
            if send_thread is not None and send_thread.is_alive():
                send_thread.join(timeout=0.2)
            self._endpos._moving = False
        except Exception:
            pass

    def _target_q(self, prefer_current: bool = False) -> np.ndarray:
        """Return the 6-dim arm joint target/current vector."""
        if prefer_current and self._arm is not None:
            q_current, _, _ = self._arm.get_state()
            return np.asarray(q_current[:6], dtype=np.float64).copy()
        if self._endpos is not None:
            q_target = np.asarray(self._endpos._q_target, dtype=np.float64).copy()
            if q_target.size >= 6:
                return q_target[:6]
        if self._arm is not None:
            q_current, _, _ = self._arm.get_state()
            return np.asarray(q_current[:6], dtype=np.float64).copy()
        q_dry = np.zeros(6, dtype=np.float64)
        q_dry[WRIST_JOINT_INDICES] = self._wrist_q
        return q_dry

    def _workspace_safe_from_fk(self, pos: np.ndarray) -> bool:
        return (
            core.ARM_X_MIN - WRIST_POS_MARGIN <= pos[0] <= core.ARM_X_MAX + WRIST_POS_MARGIN
            and core.ARM_Y_MIN - WRIST_POS_MARGIN <= pos[1] <= core.ARM_Y_MAX + WRIST_POS_MARGIN
            and core.ARM_HARD_Z_MIN <= pos[2] <= core.ARM_Z_MAX + WRIST_POS_MARGIN
        )

    def _wrist_candidate_safe(self, q_candidate: np.ndarray, q_seed: np.ndarray) -> tuple[bool, str]:
        if q_candidate.shape != (6,) or not np.all(np.isfinite(q_candidate)):
            return False, "bad_candidate_q"

        q_candidate_model = pad_q_for_model(self._safety_model, q_candidate)
        try:
            pos, rot, _ = compute_fk(self._safety_model, q_candidate_model)
        except Exception as exc:
            return False, f"fk_failed:{type(exc).__name__}"

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
            return False, "fk_nonfinite"
        if not self._workspace_safe_from_fk(pos):
            return False, "fk_workspace_limit"

        target = pos_rot_to_se3(pos, rot=rot)
        result = solve_ik(
            self._safety_model,
            self._safety_data,
            self._safety_end_frame_id,
            target,
            pad_q_for_model(self._safety_model, np.asarray(q_seed, dtype=np.float64)[:6]),
            self._wrist_ik_params,
            controlled_joints=6,
        )
        if not result.success:
            return False, f"ik_roundtrip_failed:{result.error:.3f}"

        wrist_drift = float(np.max(np.abs(result.q[WRIST_JOINT_INDICES] - q_candidate[WRIST_JOINT_INDICES])))
        if wrist_drift > WRIST_IK_JOINT_DRIFT_MAX:
            return False, f"ik_wrist_drift:{wrist_drift:.3f}"

        return True, ""

    def _apply_wrist_delta_to_target(self, delta: np.ndarray, base_q: np.ndarray | None = None) -> bool:
        delta = np.asarray(delta, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(delta)):
            return False
        delta = np.clip(delta, -WRIST_DIRECT_STEP_MAX, WRIST_DIRECT_STEP_MAX)
        if np.allclose(delta, 0.0, atol=1e-9):
            return False

        q_target = np.asarray(base_q, dtype=np.float64).copy() if base_q is not None else self._target_q(prefer_current=True)
        old_wrist = q_target[WRIST_JOINT_INDICES].copy()
        new_wrist = np.clip(old_wrist + delta, self._wrist_min, self._wrist_max)
        if np.allclose(new_wrist, old_wrist, atol=2e-4):
            return False

        q_target[WRIST_JOINT_INDICES] = new_wrist
        safe, reason = self._wrist_candidate_safe(q_target, self._target_q(prefer_current=True))
        if not safe:
            self._last_wrist_block_reason = reason
            return False
        self._last_wrist_block_reason = ""

        if self._endpos is not None:
            self._endpos._q_target = q_target.copy()
        self._wrist_q = new_wrist
        try:
            _, rpy_curr = core.joint_to_pose(q_target)
            self._rpy = np.asarray(rpy_curr, dtype=np.float64)
        except Exception:
            pass
        return True

    def apply_wrist_delta(self, dq4: float, dq5: float, dq6: float) -> bool:
        return self.apply_deltas(0.0, 0.0, 0.0, dq4, dq5, dq6)

    def apply_deltas(self, dx: float, dy: float, dz: float, dq4: float, dq5: float, dq6: float) -> bool:
        """Apply XYZ through IK, then wrist joints 4/5/6 directly.

        This avoids using orientation IK for the wrist. Joint commands are
        clamped by this demo before being sent.
        """
        if self._homing:
            return False
        self._last_translation_failed = False
        now = time.monotonic()
        if now - self._last_move_t < core.ARM_MOVE_INTERVAL:
            return False

        # Smooth translation similar to the base controller.
        desired_trans = core._sanitize_delta(dx, dy, dz)
        self._smoothed_delta = (
            (1.0 - core.ARM_SMOOTHING_ALPHA) * self._smoothed_delta
            + core.ARM_SMOOTHING_ALPHA * desired_trans
        )
        if (
            np.linalg.norm(desired_trans) < 1e-9
            and np.linalg.norm(self._smoothed_delta) < core.ARM_SMOOTHING_ZERO
        ):
            self._smoothed_delta[:] = 0.0

        trans = core._sanitize_delta(*self._smoothed_delta)
        wrist_delta = np.array([dq4, dq5, dq6], dtype=np.float64)
        if not np.all(np.isfinite(wrist_delta)):
            wrist_delta = np.zeros(3, dtype=np.float64)

        if np.allclose(trans, 0.0, atol=1e-9) and np.allclose(wrist_delta, 0.0, atol=1e-9):
            return False

        self._stop_active_trajectory()
        new_pos = core._clamp_workspace(self._pos + trans)
        if np.allclose(new_pos, self._pos, atol=3e-4) and np.allclose(wrist_delta, 0.0, atol=1e-9):
            return False

        moved_translation = False
        base_q_after_translation: np.ndarray | None = None
        if not self._dry and self._endpos is not None:
            if not np.allclose(new_pos, self._pos, atol=3e-4):
                ok = self._endpos.move_to_ik(
                    x=float(new_pos[0]),
                    y=float(new_pos[1]),
                    z=float(new_pos[2]),
                    roll=float(self._rpy[0]),
                    pitch=float(self._rpy[1]),
                    yaw=float(self._rpy[2]),
                )
                if ok:
                    moved_translation = True
                    base_q_after_translation = self._endpos._q_target.copy()
                else:
                    self._last_translation_failed = True
                    self._smoothed_delta[:] = 0.0
        elif not np.allclose(new_pos, self._pos, atol=3e-4):
            moved_translation = True

        moved_wrist = self._apply_wrist_delta_to_target(wrist_delta, base_q_after_translation)
        if moved_translation:
            self._pos = new_pos
        if moved_translation or moved_wrist:
            self._last_move_t = now
            return True
        return False


class SimArmControllerTilt(ArmControllerTilt):
    """Gesture-compatible B601 simulator; never constructs motor hardware."""

    def __init__(self) -> None:
        super().__init__(no_arm=True)
        import webbrowser
        from example.sim.visualizer import Visualizer

        self._viz = Visualizer(open_browser=True)
        self.sim_url = self._viz.meshcat.url()
        browser_opened = webbrowser.open(self.sim_url, new=2)
        if browser_opened:
            print(f"[sim] Opened 3D arm window in your browser: {self.sim_url}")
        else:
            print(f"[sim] Could not launch a browser. Open this URL manually: {self.sim_url}")
        self._q_sim = np.zeros(self._safety_model.nq, dtype=np.float64)
        self._viz.update(self._q_sim)
        self.go_home()
        print("[sim] B601-RS HOME position loaded; no motor hardware connected")

    def _render_sim_target(self) -> bool:
        import pinocchio as pin

        target = pos_rot_to_se3(
            self._pos,
            rot=pin.rpy.rpyToMatrix(*[float(v) for v in self._rpy]),
        )
        result = solve_ik(
            self._safety_model,
            self._safety_data,
            self._safety_end_frame_id,
            target,
            self._q_sim,
            self._wrist_ik_params,
            controlled_joints=6,
        )
        if not result.success:
            self._last_translation_failed = True
            return False

        q_candidate = pad_q_for_model(
            self._safety_model, np.asarray(result.q, dtype=np.float64))
        q_candidate[WRIST_JOINT_INDICES] = self._wrist_q
        lower = np.asarray(self._safety_model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self._safety_model.upperPositionLimit, dtype=np.float64)
        q_candidate = np.clip(q_candidate, lower, upper)
        achieved_pos, _, _ = compute_fk(self._safety_model, q_candidate)
        if float(achieved_pos[2]) < core.ARM_HARD_Z_MIN:
            self._last_translation_failed = True
            self._last_wrist_block_reason = "hard_z_floor"
            return False
        self._q_sim = q_candidate
        self._viz.update(self._q_sim)
        return True

    def apply_deltas(self, dx: float, dy: float, dz: float,
                     dq4: float, dq5: float, dq6: float) -> bool:
        moved = super().apply_deltas(dx, dy, dz, dq4, dq5, dq6)
        if moved:
            return self._render_sim_target()
        return False

    def go_home(self) -> None:
        super().go_home()
        if hasattr(self, "_viz"):
            self._q_sim = np.zeros(self._safety_model.nq, dtype=np.float64)
            self._viz.update(self._q_sim)

    def stop(self) -> None:
        # MeshCat owns no robot resource; keep the normal controller cleanup contract.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Bimanual XYZ + direct wrist-joint orientation demo")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument(
        "--sim", action="store_true",
        help="Run live gesture detection against a MeshCat B601-RS simulation; never connect hardware",
    )
    parser.add_argument(
        "--hw-yaml", default=None,
        help="hardware config yaml (default: from config/rebotarm.yaml)",
    )
    parser.add_argument(
        "--motion-profile", choices=tuple(MOTION_PROFILES), default="slow",
        help="Motion/safety tuning: slow is the new gentle default; default restores the old behavior",
    )
    parser.add_argument("--camera", type=int, default=0, help="Gesture/control camera index")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=550)
    parser.add_argument(
        "--min-z", type=float, default=None,
        help="Override the profile's absolute end-effector Z floor in metres",
    )
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV preview window.")
    parser.add_argument("--control-window-x", type=int, default=40, help="Main control window left position")
    parser.add_argument("--control-window-y", type=int, default=40, help="Main control window top position")
    parser.add_argument("--arm-camera", type=int, default=4, help="Second camera index pointed at the arm")
    parser.add_argument("--arm-camera-width", type=int, default=800, help="Arm camera preview/recording width")
    parser.add_argument("--arm-camera-height", type=int, default=550, help="Arm camera preview/recording height")
    parser.add_argument("--arm-window-x", type=int, default=880, help="Arm camera window left position")
    parser.add_argument("--arm-window-y", type=int, default=40, help="Arm camera window top position")
    parser.add_argument(
        "--no-arm-camera-display",
        action="store_true",
        help="Do not show the separate arm camera preview window",
    )
    parser.add_argument(
        "--record-arm-camera",
        action="store_true",
        help="Record the arm camera to MP4; requires --arm-camera",
    )
    parser.add_argument(
        "--record-control-window",
        action="store_true",
        help="Record the processed gesture/control preview window to MP4",
    )
    parser.add_argument(
        "--record-both",
        action="store_true",
        help="Record both the control preview and arm camera preview",
    )
    parser.add_argument(
        "--control-window-output",
        type=Path,
        default=None,
        help="Output MP4 path for --record-control-window; defaults to outputs/control_window_TIMESTAMP.mp4",
    )
    parser.add_argument(
        "--arm-camera-output",
        type=Path,
        default=None,
        help="Output MP4 path for --record-arm-camera; defaults to outputs/arm_camera_TIMESTAMP.mp4",
    )
    parser.add_argument("--arm-camera-fps", type=float, default=30.0, help="Recording FPS for the arm camera")
    args = parser.parse_args()
    motion_profile = MOTION_PROFILES[args.motion_profile]
    _apply_motion_profile(motion_profile)
    min_z = motion_profile.min_z if args.min_z is None else float(args.min_z)
    if not 0.08 <= min_z <= 0.25:
        parser.error("--min-z must be between 0.08 and 0.25 metres")
    core.ARM_HARD_Z_MIN = min_z
    gripper_open_cap = motion_profile.gripper_open_cap
    record_control_window = args.record_control_window or args.record_both
    record_arm_camera = args.record_arm_camera or args.record_both

    model_path = core._ensure_holistic_model()
    base_options = core.mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = core.mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=core.mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.55,
        min_pose_landmarks_confidence=0.55,
        min_hand_landmarks_confidence=0.55,
        output_segmentation_mask=False,
        output_face_blendshapes=False,
    )
    landmarker = core.mp.tasks.vision.HolisticLandmarker.create_from_options(options)

    if args.sim:
        arm = SimArmControllerTilt()
        gripper = core.GripperController(no_gripper=True)
    else:
        arm = ArmControllerTilt(
            no_arm=args.no_arm,
            hw_yaml=args.hw_yaml,
            vlim_cap=motion_profile.arm_vlim_cap,
        )
        try:
            gripper = core.GripperController(
                no_gripper=args.no_gripper,
                endpose=arm.endpose,
                max_open_fraction=gripper_open_cap,
            )
        except Exception:
            # Never leave the arm enabled if the gripper failed to initialize.
            arm.stop()
            raise
    right_hand_gestures = core.ConfirmedRightHandGestureController()
    clap = core.ClapDetector()
    hands_touch_home = HandsTouchHomeDetector()
    logger = core.ActionLogger()
    wrist_logger = WristLogger()
    wrist_aim = WristJointAimController()
    pinch_gate = StablePinch()
    gripper_mode = ShoulderHeightModeDetector()

    cap = core.cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(core.cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(core.cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "ReBot - XYZ + Direct Wrist Joints"
    display_enabled = not args.headless
    main_preview = None
    if display_enabled:
        main_preview = PreviewWindow(
            window_name,
            args.width,
            args.height,
            args.control_window_x,
            args.control_window_y,
        )
        if not main_preview.active:
            display_enabled = False
            print("[display] Use Ctrl+C to quit.")
    else:
        print("[display] Headless mode enabled. Use Ctrl+C to quit.")

    arm_camera = ArmCameraRecorder(
        camera_index=args.arm_camera,
        width=args.arm_camera_width,
        height=args.arm_camera_height,
        display=not args.no_arm_camera_display and display_enabled,
        record=record_arm_camera,
        output_path=args.arm_camera_output,
        fps=args.arm_camera_fps,
        window_x=args.arm_window_x,
        window_y=args.arm_window_y,
    )
    control_recorder = FrameRecorder(
        args.control_window_output,
        args.width,
        args.height,
        args.arm_camera_fps,
        "control_window",
    ) if record_control_window else None
    if control_recorder is not None and control_recorder.enabled:
        print(f"[control-window] Recording to {control_recorder.output_path}")

    if record_arm_camera and args.arm_camera is None:
        print("[arm-camera] recording ignored because --arm-camera was not provided")
    elif arm_camera.enabled:
        if record_arm_camera and arm_camera.output_path is not None:
            print(f"[arm-camera] Recording to {arm_camera.output_path}")
        elif not args.no_arm_camera_display and not args.headless:
            print(f"[arm-camera] Showing camera {args.arm_camera} in a separate window")

    print("Bimanual XYZ + direct wrist-joint claw orientation")
    if args.sim:
        print("  MODE: B601-RS SIMULATION (hardware disabled)")
    print(f"  Gesture camera: {args.camera}")
    low = core._clamp_workspace(np.array([-9.0, -9.0, -9.0], dtype=np.float64))
    high = core._clamp_workspace(np.array([9.0, 9.0, 9.0], dtype=np.float64))
    print(f"  MOTION PROFILE: {args.motion_profile.upper()}")
    print(f"    workspace X={low[0]:.3f}..{high[0]:.3f} m, "
          f"Y={low[1]:+.3f}..{high[1]:+.3f} m, Z={low[2]:.3f}..{high[2]:.3f} m")
    print(f"    XYZ step<={core.ARM_MAX_DELTA * 1000:.1f} mm, "
          f"joint velocity cap={motion_profile.arm_vlim_cap or 'config default'}")
    print(f"    wrist range={WRIST_JOINT_SAFETY_SCALE * 100:.0f}% of limits, "
          f"gripper<={gripper_open_cap * 100:.0f}% open")
    if arm_camera.enabled and not args.no_arm_camera_display and not args.headless:
        print(f"  Arm camera: {args.arm_camera}")
    print("  Right hand: restored XYZ thumb gestures")
    print("  Arm mode: open left palm twist=joint6 only")
    print("            non-open left thumb up/down=joint4; thumb left/right=joint5")
    print("  Gripper mode: move both hands above shoulders, then left pinch/unpinch controls gripper")
    print("  Touch both hands together or clap = HOME")
    print("  Q or Esc quits")

    should_exit = False
    frame_timestamp_ms = 0
    flash = 0

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
            arm_camera.update()

            frame = core.cv2.flip(frame, 1)
            rgb = core.cv2.cvtColor(frame, core.cv2.COLOR_BGR2RGB)
            mp_image = core.mp.Image(image_format=core.mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33

            pose_points = core._extract_pose_points(getattr(result, "pose_landmarks", None), frame.shape)
            hand_map = core._build_hand_observations(result, frame.shape)
            left_hand = hand_map.get("left")
            right_hand = hand_map.get("right")

            if hands_touch_home.update(left_hand, right_hand):
                flash = 15
                logger.log_home("hands_touch")
                right_hand_gestures.reset()
                wrist_aim.reset()
                pinch_gate.reset()
                threading.Thread(target=arm.go_home, daemon=True).start()
            elif clap.update(pose_points):
                flash = 15
                logger.log_home("clap")
                right_hand_gestures.reset()
                wrist_aim.reset()
                pinch_gate.reset()
                threading.Thread(target=arm.go_home, daemon=True).start()
            elif flash > 0:
                flash -= 1

            in_gripper_mode = gripper_mode.update(pose_points)
            pinch_value = core._pinch_openness(left_hand)
            if in_gripper_mode:
                arm.freeze_motion()
                right_hand_gestures.reset()
                wrist_aim.reset()
                wrist_logger.log("gripper_mode", 0.0, 0.0, 0.0)
                pinch_state, pinch_ema = pinch_gate.update(pinch_value)
                gripper.update(pinch_state)
                logger.log_gripper(pinch_state, pinch_ema if pinch_ema >= 0 else pinch_value, gripper.open_frac)
                dx = dy = dz = 0.0
            else:
                pinch_gate.reset()
                # Translation (right hand) through XYZ IK + wrist joints directly.
                cmd = right_hand_gestures.update(right_hand)
                dx, dy, dz = core._command_to_delta(cmd)
                dq4, dq5, dq6 = wrist_aim.update(left_hand, arm.wrist_q)
                wrist_logger.log(wrist_aim.state, float(dq4), float(dq5), float(dq6))

                moved = False if arm.homing else arm.apply_deltas(float(dx), float(dy), float(dz), float(dq4), float(dq5), float(dq6))
                wrist_logger.log_block(arm.last_wrist_block_reason)
                if (dx != 0.0 or dy != 0.0 or dz != 0.0) and (not moved or arm.last_translation_failed):
                    right_hand_gestures.block_active_command()
                    dx, dy, dz = 0.0, 0.0, 0.0
                logger.log_arm(right_hand_gestures.active_command, float(dx), float(dy), float(dz))

            # Minimal overlay (reuse core helpers)
            core._draw_pose(frame, getattr(result, "pose_landmarks", None), (120, 180, 255))
            core._draw_hand(frame, left_hand, (50, 220, 120))
            core._draw_hand(frame, right_hand, (255, 180, 80))
            core._draw_hand_markers(frame, left_hand, pinch_value, (50, 220, 120))

            h, w = frame.shape[:2]
            core.cv2.rectangle(frame, (0, 0), (w, 140), (20, 20, 20), -1)
            if in_gripper_mode:
                mode_text = "GRIPPER MODE: both wrists above shoulders | left pinch controls gripper"
            else:
                aim_state = "aiming" if wrist_aim.active else "paused"
                mode_text = f"ARM MODE: right=XYZ | left twist=J6 thumb=J4/J5 ({aim_state})"
            core.cv2.putText(frame, f"{mode_text} | touch hands=HOME",
                             (12, 24), core.cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
            core.cv2.putText(frame, f"Arm pos: {arm.pos[0]:.3f} {arm.pos[1]:+.3f} {arm.pos[2]:.3f}",
                             (12, 50), core.cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1)
            core.cv2.putText(frame, f"Wrist q4-6: {arm.wrist_q[0]:+.3f} {arm.wrist_q[1]:+.3f} {arm.wrist_q[2]:+.3f}",
                             (12, 74), core.cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 200, 255), 1)
            core.cv2.putText(
                frame,
                f"Gripper: {gripper.open_frac * 100:.0f}% open "
                f"(profile maximum: {gripper_open_cap * 100:.0f}%)",
                (12, 98),
                core.cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (120, 240, 180),
                1,
            )
            core.cv2.putText(
                frame,
                f"Hard Z floor: {core.ARM_HARD_Z_MIN:.3f} m",
                (w - 220, 98),
                core.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (80, 180, 255),
                1,
            )
            right_fingers = core._extended_finger_count(right_hand) if right_hand is not None else 0
            right_palm = right_hand is not None and core._palm_open(right_hand)
            right_raw_command = core._right_hand_command(right_hand)
            core.cv2.putText(
                frame,
                f"Right hand: {'OPEN' if right_palm else 'CLOSED/UNKNOWN'} "
                f"fingers={right_fingers} raw={right_raw_command or 'none'} "
                f"active={right_hand_gestures.active_command or 'none'}",
                (12, 120),
                core.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (80, 230, 255) if right_hand is not None else (100, 100, 255),
                1,
            )

            if control_recorder is not None:
                if control_recorder.enabled:
                    record_frame = frame.copy()
                    core.cv2.circle(record_frame, (20, h - 24), 7, (0, 0, 255), -1)
                    core.cv2.putText(
                        record_frame,
                        "REC",
                        (34, h - 19),
                        core.cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )
                    control_recorder.write(record_frame)

            if display_enabled:
                main_preview.show(frame)
                if PreviewWindow.exit_requested():
                    break
                if main_preview.uses_opencv:
                    key = core.cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
            else:
                time.sleep(0.001)
    finally:
        cap.release()
        arm_camera.release()
        if control_recorder is not None:
            control_recorder.release()
        if display_enabled:
            try:
                core.cv2.destroyAllWindows()
            except core.cv2.error:
                pass
            PreviewWindow.destroy_all()
        landmarker.close()
        arm.stop()
        gripper.stop()


if __name__ == "__main__":
    main()
