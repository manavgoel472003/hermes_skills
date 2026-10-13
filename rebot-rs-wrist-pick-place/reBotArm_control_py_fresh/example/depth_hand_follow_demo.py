#!/usr/bin/env python3
"""Depth-camera hand-follow teleop (Orbbec Gemini 336) — SEPARATE mode.

This is an isolated alternative to the gesture demo: instead of discrete thumb
gestures, it tracks the operator's hand position in 3D with the Orbbec depth
camera and maps that motion 1:1 to the arm end-effector, so the arm mimics how
the hand moves through space. The SAME workspace box and movement limits as the
gesture demo's default profile are applied (see arm_teleop_constraints.py).

Two modes, isolated:
  * camera-only (default, --no-arm): track + visualise the constrained target.
    Run this FIRST to check tracking and tuning before touching the arm.
  * arm-follow (--arm): drive the real arm in-process (needs can0 up). The arm
    follows ONLY while ENGAGED (press 'e'); disengaged it never moves.

Run with the fresh repo venv (Py3.10) — it now has the Orbbec SDK, MediaPipe,
OpenCV AND the arm stack, so camera + arm live in ONE process (no server):
    .venv/bin/python example/depth_hand_follow_demo.py            # camera-only
    .venv/bin/python example/depth_hand_follow_demo.py --arm      # drive the arm
    .venv/bin/python example/depth_hand_follow_demo.py --headless --frames 60 \
        --save-annotated /tmp/depth_track.png                     # no display, save a frame

Controls (live window):
  right hand open  controls end-effector XYZ translation
  right fist (3 consecutive frames) controls wrist joints J4/J5/J6
  left thumb/index pinch controls gripper opening while engaged
  touch both palms together briefly to disengage and send the arm HOME
  e          ENGAGE / disengage arm-follow (deadman clutch). The arm follows the
             hand ONLY while engaged; engaging also captures neutral. Disengaged,
             the client sends nothing and the arm holds — random tracking can
             never move it.
  SPACE / n  (re)set the neutral reference to the hand's current position
  r          reset (drop neutral, disengage, re-anchor to HOME)
  f          toggle true fullscreen
  q / Esc    quit
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FRESH_REPO = Path(__file__).resolve().parents[1]
WALL_TOUCH = Path("/home/manav/rebot_lerobot/wall-touch-demo")
sys.path.insert(0, str(FRESH_REPO))            # repo root (for reBotArm_control_py)
sys.path.insert(0, str(FRESH_REPO / "example"))
sys.path.insert(0, str(WALL_TOUCH))

import cv2
import mediapipe as mp

import arm_teleop_constraints as C
from arm_teleop_constraints import (
    AxisMap, HandFollowMapper, WristAxisMap, WristFollowMapper,
    deproject_pixel, workspace_bounds,
)

HAND_MODEL = FRESH_REPO / "models" / "hand_landmarker.task"

# Palm landmarks (wrist + finger MCPs) — a stable centroid less noisy than one point.
PALM_LANDMARKS = (0, 5, 9, 13, 17)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
DEPTH_WIN = 7          # half-not: sample a (2*DEPTH_WIN+1)^2 window around each palm px
DEPTH_MIN_MM, DEPTH_MAX_MM = 150.0, 1500.0   # plausible hand distance; reject wall/noise
GRIP_PINCH_CLOSED = 0.18  # thumb-index distance / palm size
GRIP_PINCH_OPEN = 0.65
HANDS_TOUCH_SCALE = 1.35  # palm-centre distance / average palm length
HANDS_TOUCH_MAX_3D_M = 0.12
HANDS_TOUCH_HOLD_FRAMES = 8
HANDS_TOUCH_COOLDOWN_S = 3.0


def _make_landmarker():
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)


def _screen_size(fallback: tuple[int, int],
                 window_center: tuple[int, int] | None = None) -> tuple[int, int]:
    """Pixel dimensions of the monitor containing the OpenCV window.

    Tk reports the combined virtual desktop on multi-monitor Linux systems,
    which is not the size OpenCV uses for fullscreen. Prefer xrandr's individual
    monitor rectangles and choose the one containing the window centre.
    """
    try:
        output = subprocess.check_output(
            ["xrandr", "--current"], text=True, stderr=subprocess.DEVNULL,
        )
        monitors = []
        for line in output.splitlines():
            if " connected" not in line:
                continue
            match = re.search(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", line)
            if match:
                width, height, x, y = map(int, match.groups())
                monitors.append((x, y, width, height, " primary " in line))
        if monitors:
            if window_center is not None:
                cx, cy = window_center
                for x, y, width, height, _ in monitors:
                    if x <= cx < x + width and y <= cy < y + height:
                        return width, height
            primary = next((m for m in monitors if m[4]), monitors[0])
            return primary[2], primary[3]
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        return size
    except Exception:
        return fallback


def _is_fist(landmarks) -> bool:
    """Orientation-independent fist test: a finger is 'extended' when its tip is
    farther from the wrist than its PIP joint. Fist = at most one finger extended
    (thumb ignored, it stays out on a fist)."""
    w = np.array([landmarks[0].x, landmarks[0].y])
    def extended(tip, pip):
        t = np.array([landmarks[tip].x, landmarks[tip].y])
        p = np.array([landmarks[pip].x, landmarks[pip].y])
        return np.linalg.norm(t - w) > np.linalg.norm(p - w) * 1.05
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))   # index, middle, ring, pinky
    return sum(extended(t, p) for t, p in fingers) <= 1


def _pinch_open_fraction(landmarks) -> float:
    """Map thumb/index separation to the safe 5%..70% gripper range."""
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
    palm = max(float(np.linalg.norm(pts[0] - pts[9])), 1e-6)
    pinch = float(np.linalg.norm(pts[4] - pts[8])) / palm
    amount = np.clip(
        (pinch - GRIP_PINCH_CLOSED) / (GRIP_PINCH_OPEN - GRIP_PINCH_CLOSED),
        0.0, 1.0,
    )
    return float(
        C.GRIPPER_MIN_OPEN_FRACTION
        + amount * (C.GRIPPER_MAX_OPEN_FRACTION - C.GRIPPER_MIN_OPEN_FRACTION)
    )


def _sample_depth_m(depth_mm: np.ndarray, u: float, v: float) -> float | None:
    """Median of valid depths in a window around (u,v), in metres, or None."""
    h, w = depth_mm.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    x0, x1 = max(0, ui - DEPTH_WIN), min(w, ui + DEPTH_WIN + 1)
    y0, y1 = max(0, vi - DEPTH_WIN), min(h, vi + DEPTH_WIN + 1)
    win = depth_mm[y0:y1, x0:x1]
    valid = win[(win >= DEPTH_MIN_MM) & (win <= DEPTH_MAX_MM)]
    if valid.size < 5:
        return None
    return float(np.median(valid)) / 1000.0


def _hand_cam_xyz(landmarks, depth_mm, intr) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    """Palm-centroid camera-frame XYZ (m) and its pixel, from hand landmarks."""
    h, w = depth_mm.shape
    pts = []
    depths = []
    for i in PALM_LANDMARKS:
        u, v = landmarks[i].x * w, landmarks[i].y * h
        d = _sample_depth_m(depth_mm, u, v)
        if d is not None:
            pts.append((u, v))
            depths.append(d)
    if not depths:
        return None, None
    u = float(np.mean([p[0] for p in pts]))
    v = float(np.mean([p[1] for p in pts]))
    depth_m = float(np.median(depths))
    xyz = deproject_pixel(u, v, depth_m, intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    return xyz, (int(round(u)), int(round(v)))


# ── visualisation ─────────────────────────────────────────────────────────────

def _draw_hand(frame, landmarks, color=(0, 200, 255), label=None) -> None:
    h, w = frame.shape[:2]
    px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], color, 1)
    for p in px:
        cv2.circle(frame, p, 2, color, -1)
    if label:
        cv2.putText(frame, label, (px[0][0] + 6, px[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _hands_by_side(result) -> dict[str, object]:
    """Return MediaPipe landmarks keyed by its Left/Right classification."""
    hands = {}
    for i, landmarks in enumerate(result.hand_landmarks):
        if i >= len(result.handedness) or not result.handedness[i]:
            continue
        side = result.handedness[i][0].category_name.lower()
        hands[side] = landmarks
    return hands


class _HandsTouchHomeDetector:
    """Debounced, release-latched palm-touch gesture."""

    def __init__(self) -> None:
        self._streak = 0
        self._latched = False
        self._last_trigger_t = -HANDS_TOUCH_COOLDOWN_S

    @staticmethod
    def _centre_and_scale(landmarks) -> tuple[np.ndarray, float]:
        pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
        centre = np.mean(pts[list(PALM_LANDMARKS)], axis=0)
        scale = max(float(np.linalg.norm(pts[0] - pts[9])), 1e-6)
        return centre, scale

    def update(self, left, right, left_xyz=None, right_xyz=None,
               now: float | None = None) -> bool:
        touching = False
        if (left is not None and right is not None
                and left_xyz is not None and right_xyz is not None):
            lc, ls = self._centre_and_scale(left)
            rc, rs = self._centre_and_scale(right)
            touching = (
                float(np.linalg.norm(lc - rc))
                <= HANDS_TOUCH_SCALE * 0.5 * (ls + rs)
                and float(np.linalg.norm(
                    np.asarray(left_xyz) - np.asarray(right_xyz)
                )) <= HANDS_TOUCH_MAX_3D_M
            )

        if not touching:
            self._streak = 0
            self._latched = False
            return False
        if self._latched:
            return False

        self._streak += 1
        now = time.monotonic() if now is None else now
        if (self._streak >= HANDS_TOUCH_HOLD_FRAMES
                and now - self._last_trigger_t >= HANDS_TOUCH_COOLDOWN_S):
            self._streak = 0
            self._latched = True
            self._last_trigger_t = now
            return True
        return False


def _draw_box_panel(frame, origin, size, lo, hi, target, clamped, axes, labels) -> None:
    """Draw a 2D projection of the workspace box with the target dot."""
    ox, oy = origin
    W, H = size
    ai, bi = axes
    cv2.rectangle(frame, (ox, oy), (ox + W, oy + H), (70, 70, 70), 1)
    cv2.putText(frame, labels[0], (ox + W - 14, oy + H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, labels[1], (ox - 4, oy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def to_px(val_a, val_b):
        fa = (val_a - lo[ai]) / max(hi[ai] - lo[ai], 1e-6)
        fb = (val_b - lo[bi]) / max(hi[bi] - lo[bi], 1e-6)
        return int(ox + fa * W), int(oy + (1.0 - fb) * H)

    # HOME marker
    hx, hy = to_px(C.ARM_HOME[ai], C.ARM_HOME[bi])
    cv2.drawMarker(frame, (hx, hy), (120, 120, 120), cv2.MARKER_CROSS, 8, 1)
    # target dot — red if either shown axis is clamped
    tx, ty = to_px(target[ai], target[bi])
    hit = clamped[ai] or clamped[bi]
    cv2.circle(frame, (tx, ty), 5, (0, 0, 255) if hit else (0, 220, 0), -1)


def _draw_wrist_panel(frame, origin, size, wrist_target) -> None:
    """Compact J4/J5/J6 gauges used instead of XYZ plots in wrist mode."""
    ox, oy = origin
    width, height = size
    _glass_panel(frame, (ox, oy), (ox + width, oy + height),
                 color=(16, 19, 24), alpha=0.82, border=(64, 70, 78))
    _ui_text(frame, "WRIST JOINT TARGETS", (ox + 14, oy + 27),
             (225, 228, 232), scale=0.52, thickness=1)
    values = np.asarray(wrist_target, dtype=np.float64)
    lo = C.WRIST_JOINT_LO * C.WRIST_SAFETY_SCALE
    hi = C.WRIST_JOINT_HI * C.WRIST_SAFETY_SCALE
    for i, label in enumerate(("J4", "J5", "J6")):
        y = oy + 54 + i * 31
        x1, x2 = ox + 62, ox + width - 58
        cv2.line(frame, (x1, y), (x2, y), (82, 88, 96), 5, cv2.LINE_AA)
        cv2.line(frame, ((x1 + x2) // 2, y - 6), ((x1 + x2) // 2, y + 6),
                 (145, 150, 158), 1, cv2.LINE_AA)
        frac = float(np.clip((values[i] - lo[i]) / max(hi[i] - lo[i], 1e-6), 0, 1))
        px = int(round(x1 + frac * (x2 - x1)))
        cv2.circle(frame, (px, y), 7, (70, 172, 235), -1, cv2.LINE_AA)
        _ui_text(frame, label, (ox + 14, y + 6), (220, 222, 226),
                 scale=0.48, thickness=1)
        _ui_text(frame, f"{np.degrees(values[i]):+.0f} deg",
                 (x2 + 10, y + 6), (70, 172, 235), scale=0.43, thickness=1)


def _glass_panel(frame, p1, p2, color=(18, 22, 28), alpha=0.78,
                 border=(90, 100, 110)) -> None:
    """High-contrast translucent panel with a subtle border."""
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, p1, p2, border, 1, cv2.LINE_AA)


def _ui_text(frame, text, origin, color=(245, 245, 245),
             scale=0.58, thickness=2) -> None:
    """Clean text with a small shadow for bright camera scenes."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x + 1, y + 1), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


def _annotate(frame, tracked, hand_px, mapper, has_hand,
              engaged=False, arm_mode=False, mode="xyz", wrist_target=None,
              grip_fraction=None) -> np.ndarray:
    h, w = frame.shape[:2]
    lo, hi = workspace_bounds()
    target = mapper.target
    clamped = mapper.clamped_axes

    if hand_px is not None:
        cv2.circle(frame, hand_px, 7, (0, 0, 255), 2)

    # Large, persistent status and mode badges.
    status = "ARM ENGAGED" if engaged else "ARM DISENGAGED"
    status_fill = (43, 153, 82) if engaged else (55, 62, 210)
    status_w = min(245, max(190, w // 3))
    cv2.rectangle(frame, (10, 8), (status_w, 46), status_fill, -1, cv2.LINE_AA)
    _ui_text(frame, status, (22, 34), (255, 255, 255), scale=0.64, thickness=1)

    mchip = "WRIST CONTROL  |  J4  J5  J6" if mode == "wrist" else "POSITION CONTROL  |  X  Y  Z"
    mcol = (70, 172, 235) if mode == "wrist" else (94, 201, 132)
    mode_x = status_w + 10
    _glass_panel(frame, (mode_x, 8), (w - 10, 46),
                 color=(18, 21, 26), alpha=0.88, border=(65, 72, 80))
    cv2.rectangle(frame, (mode_x, 43), (w - 10, 46), mcol, -1)
    _ui_text(frame, mchip, (mode_x + 12, 34), (235, 237, 240),
             scale=0.52, thickness=1)

    lines = []
    if not has_hand:
        lines.append(("NO ARM HAND DETECTED", (60, 190, 255)))
    elif tracked is None:
        lines.append(("HAND FOUND - WAITING FOR DEPTH", (60, 190, 255)))
    else:
        lines.append((f"HAND   X {tracked[0]:+.3f}   Y {tracked[1]:+.3f}   Z {tracked[2]:+.3f} m",
                      (255, 255, 255)))
    active_has_neutral = (
        mapper.has_neutral if mode == "xyz"
        else wrist_target is not None
    )
    if not active_has_neutral:
        lines.append(("PRESS SPACE / N TO SET NEUTRAL", (50, 230, 255)))
    if mode == "wrist" and wrist_target is not None:
        wd = np.degrees(wrist_target)
        lines.append((f"TARGET   J4 {wd[0]:+.0f}  J5 {wd[1]:+.0f}  J6 {wd[2]:+.0f} deg",
                      (50, 190, 255)))
    else:
        lines.append((f"TARGET X {target[0]:+.3f}   Y {target[1]:+.3f}   Z {target[2]:+.3f} m",
                      (70, 235, 115)))
        cl = [ax for ax, c in zip("XYZ", clamped) if c]
        if cl:
            lines.append((f"LIMIT REACHED: {','.join(cl)}", (70, 90, 255)))
    if grip_fraction is not None:
        lines.append((f"GRIPPER {grip_fraction * 100:.0f}% OPEN",
                      (255, 190, 80)))

    panel_bottom = 62 + 28 * len(lines)
    _glass_panel(frame, (10, 56), (min(w - 10, 535), panel_bottom))
    for i, (txt, col) in enumerate(lines):
        _ui_text(frame, txt, (22, 82 + 28 * i), col, scale=0.55, thickness=1)

    # The visualization follows the active control mode.
    if mode == "wrist" and wrist_target is not None:
        pw, ph = min(330, w - 20), 148
        _draw_wrist_panel(frame, (w - pw - 10, h - ph - 10),
                          (pw, ph), wrist_target)
    else:
        P, M = 130, 16
        panel_y = max(panel_bottom + 18, h - P - 40)
        _draw_box_panel(frame, (w - 2 * P - M - 8, panel_y), (P, P), lo, hi,
                        target, clamped, axes=(0, 1), labels=("X reach", "Y L/R"))
        _draw_box_panel(frame, (w - P - 8, panel_y), (P, P), lo, hi,
                        target, clamped, axes=(0, 2), labels=("X reach", "Z up"))
    return frame


class _ArmDriver:
    """In-process arm driver: holds the B601-RS via the tested SmoothedEndPose,
    homes + confirms, and drives to XYZ targets with IK. Same constraints as the
    gesture demo. Only touched when ENGAGED (the caller gates it)."""

    def __init__(self, hw_yaml=None):
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.kinematics import joint_to_pose
        import bimanual_hand_control_6dof_demo as core   # SmoothedEndPose + gripper consts
        self._core = core
        self._joint_to_pose = joint_to_pose
        self.arm = RebotArm(hw_yaml)
        self.endpos = core.SmoothedEndPose(self.arm, arm_control_mode="posvel")
        self.endpos.start()
        self._n = self.endpos._n
        # Use the proven gripper startup path from the bimanual demo. It checks
        # feedback, clears a latched motor protection state, re-enables motor 7,
        # and confirms the safe 5%-open starting position before teleop begins.
        try:
            self.gripper = core.GripperController(
                no_gripper=False,
                endpose=self.endpos,
                max_open_fraction=C.GRIPPER_MAX_OPEN_FRACTION,
            )
        except Exception:
            self.endpos.end()
            raise
        self._grip_fraction = C.GRIPPER_MIN_OPEN_FRACTION

    def home_and_confirm(self, timeout_s: float = 25.0) -> bool:
        home = C.ARM_HOME
        print(f"[arm] homing to {np.round(home,3)} ...")
        self.endpos.move_to_traj(x=float(home[0]), y=float(home[1]), z=float(home[2]),
                                 roll=0.0, pitch=0.0, yaw=0.0, duration=2.5)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            q, vel, _ = self.arm.get_state()
            pos, _ = self._joint_to_pose(q[:self._n])
            if (float(np.linalg.norm(np.asarray(pos) - home)) < 0.02
                    and float(np.max(np.abs(vel[:self._n]))) < 0.08
                    and not self.endpos._moving):
                print("[arm] HOME confirmed")
                return True
            time.sleep(0.1)
        print("[arm] WARNING: home not confirmed")
        return False

    def set_target(self, xyz: np.ndarray) -> None:
        t = C.clamp_workspace(xyz)   # safety belt (mapper already clamps)
        self.endpos.move_to_ik(x=float(t[0]), y=float(t[1]), z=float(t[2]),
                               roll=0.0, pitch=0.0, yaw=0.0)

    def get_ee_xyz(self) -> np.ndarray:
        q, _, _ = self.arm.get_state()
        pos, _ = self._joint_to_pose(q[:self._n])
        return np.asarray(pos, dtype=np.float64)

    def get_wrist_q(self) -> np.ndarray:
        """Current commanded wrist joints J4/J5/J6 (indices 3,4,5)."""
        return np.asarray(self.endpos._q_target[3:6], dtype=np.float64).copy()

    def set_wrist_q(self, w3: np.ndarray) -> bool:
        """Set J4/J5/J6 while holding J1-J3.

        Reject a candidate if forward kinematics says the wrist motion would put
        the tool outside the same Cartesian safety box used by XYZ mode.
        """
        lo = C.WRIST_JOINT_LO * C.WRIST_SAFETY_SCALE
        hi = C.WRIST_JOINT_HI * C.WRIST_SAFETY_SCALE
        q = np.asarray(self.endpos._q_target, dtype=np.float64).copy()
        q[3:6] = np.clip(np.asarray(w3, dtype=np.float64), lo, hi)
        pos, _ = self._joint_to_pose(q[:self._n])
        pos = np.asarray(pos, dtype=np.float64)
        box_lo, box_hi = C.workspace_bounds()
        if (not np.all(np.isfinite(pos))
                or np.any(pos < box_lo) or np.any(pos > box_hi)):
            return False
        self.endpos._q_target = q
        return True

    def set_gripper_fraction(self, fraction: float) -> None:
        """Command a safely bounded gripper opening fraction."""
        fraction = float(np.clip(
            fraction,
            C.GRIPPER_MIN_OPEN_FRACTION,
            C.GRIPPER_MAX_OPEN_FRACTION,
        ))
        target = self._core.GRIPPER_CLOSE_POS + fraction * (
            self._core.GRIPPER_OPEN_POS - self._core.GRIPPER_CLOSE_POS
        )
        self.endpos.set_gripper_target(target)
        self._grip_fraction = fraction

    def get_gripper_feedback(self) -> tuple[float, float, int | None]:
        """Return commanded fraction, actual motor radians, and status code."""
        q, _, _ = self.arm.get_state()
        actual = float(q[self._n]) if len(q) > self._n else float("nan")
        motor = self.arm._motor_map.get("gripper")
        state = motor.get_state() if motor is not None else None
        status = state.status_code if state is not None else None
        return self._grip_fraction, actual, status

    def shutdown(self) -> None:
        try:
            self.endpos.end()        # safe-home + disable
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Orbbec depth hand-follow teleop")
    ap.add_argument("--no-arm", action="store_true", default=True,
                    help="camera-only: track + visualise, do not drive the arm (default)")
    ap.add_argument("--arm", dest="no_arm", action="store_false",
                    help="drive the real arm in-process (homes on start; needs can0 up)")
    ap.add_argument("--hw-yaml", default=None, help="arm hardware yaml (default: config/rebotarm.yaml)")
    ap.add_argument("--headless", action="store_true", help="no display window (print only)")
    ap.add_argument("--frames", type=int, default=0, help="process N frames then exit (0=until quit)")
    ap.add_argument("--save-annotated", type=str, default=None,
                    help="write the last annotated frame to this path")
    ap.add_argument("--scale", type=float, default=1.0, help="arm metres per camera metre")
    ap.add_argument("--arm-hand", choices=("left", "right"), default="right",
                    help="hand used for XYZ/wrist; the opposite hand controls gripper "
                         "(default: right)")
    ap.add_argument("--auto-neutral", action="store_true",
                    help="set neutral automatically on the first frame a hand is seen")
    ap.add_argument("--auto-engage", action="store_true",
                    help="camera-only testing aid: auto-engage once a hand is seen. IGNORED "
                         "with --arm — the arm ALWAYS starts with follow off (press 'e').")
    args = ap.parse_args()

    # SAFETY: the arm must never auto-follow at startup. Force follow off for --arm.
    if args.auto_engage and not args.no_arm:
        print("[depth-follow] --auto-engage ignored with --arm; press 'e' to engage.")
        args.auto_engage = False

    if not HAND_MODEL.exists():
        print(f"Missing hand model: {HAND_MODEL}")
        return

    arm_driver = None
    if not args.no_arm:
        print("[depth-follow] --arm: connecting + homing (arm will move to HOME now) ...")
        arm_driver = _ArmDriver(args.hw_yaml)
        if not arm_driver.home_and_confirm():
            print("[depth-follow] home not confirmed — disabling and exiting for safety")
            arm_driver.shutdown()
            return

    from wall_touch_orbbec import OrbbecCamera
    cam = OrbbecCamera()
    p = cam.pipeline.get_camera_param().rgb_intrinsic
    intr = dict(fx=float(p.fx), fy=float(p.fy), cx=float(p.cx), cy=float(p.cy))
    print(f"[depth-follow] {cam.stream_description}")
    print(f"[depth-follow] intrinsics fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")

    landmarker = _make_landmarker()
    mapper = HandFollowMapper(axis_map=AxisMap(scale=args.scale))          # XYZ mode
    wrist_mapper = WristFollowMapper(axis_map=WristAxisMap())              # fist mode
    mode = "xyz"                    # "xyz" (open hand) or "wrist" (closed fist)
    fist_streak = 0                 # debounce: consecutive frames of the opposite state
    touch_home = _HandsTouchHomeDetector()

    win = "Orbbec depth hand-follow" + ("  [ARM]" if arm_driver is not None else "  [camera-only]")
    disp_size = None
    if not args.headless:
        warm = cam.read()
        fh, fw = warm.color_bgr.shape[:2]
        up = 2 if max(fw, fh) <= 720 else 1          # upscale small (USB2) streams
        disp_size = (fw * up, fh * up)
        # Let OpenCV scale each frame to the user-controlled window. FREERATIO
        # avoids letterboxing, while GUI_NORMAL drops the Qt toolbar/status bar.
        cv2.namedWindow(
            win,
            cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_NORMAL,
        )
        # Some Qt/OpenCV builds ignore WINDOW_FREERATIO when it is supplied only
        # as a namedWindow flag. Set the aspect policy explicitly as well so the
        # image stretches to the entire client area at non-4:3 window sizes.
        cv2.setWindowProperty(
            win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO,
        )
        cv2.resizeWindow(win, *disp_size)
    gripper_hand = "left" if args.arm_hand == "right" else "right"
    print(f"[depth-follow] {args.arm_hand}=XYZ/wrist  {gripper_hand}=gripper  "
          "open=XYZ  fist=J4/J5/J6  e=engage/disengage  "
          "SPACE/n=set neutral  r=reset  f=fullscreen  q/Esc=quit")

    prev_t = time.monotonic()
    count = 0
    last_annotated = None
    engaged = False        # SAFETY: arm follows ONLY while engaged (press 'e')
    fullscreen = False
    windowed_size = disp_size

    def _engage(tracked_xyz):
        """Engage arm-follow and (re)capture neutral for the ACTIVE mode at the
        current hand pose, so nothing jumps when following starts (same as 'n')."""
        nonlocal engaged
        if mode == "wrist":
            wrist_mapper.reset(anchor_wrist_q=arm_driver.get_wrist_q() if arm_driver else np.zeros(3))
            if tracked_xyz is not None:
                wrist_mapper.update(tracked_xyz, 0.0)
        else:
            mapper.reset(anchor=arm_driver.get_ee_xyz() if arm_driver else None)
            if tracked_xyz is not None:
                mapper.update(tracked_xyz, 0.0)
        engaged = True
        print(f"[depth-follow] ENGAGED — following in {mode.upper()} mode (neutral set)"
              if tracked_xyz is not None
              else f"[depth-follow] ENGAGED — waiting for a hand ({mode.upper()} mode)")
    try:
        while True:
            f = cam.read()
            color, depth = f.color_bgr, f.depth_mm
            now = time.monotonic()
            dt, prev_t = now - prev_t, now

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, int(now * 1000))

            tracked = None
            hand_px = None
            hands = _hands_by_side(result)
            arm_lms = hands.get(args.arm_hand)
            gripper_lms = hands.get(gripper_hand)
            has_hand = arm_lms is not None
            fist = False
            grip_fraction = None
            gripper_xyz = None
            if has_hand:
                tracked, hand_px = _hand_cam_xyz(arm_lms, depth, intr)
                fist = _is_fist(arm_lms)
                if args.auto_neutral and tracked is not None and not mapper.has_neutral:
                    mapper.reset()
                    mapper.update(tracked, dt)   # captures neutral
                if args.auto_engage and tracked is not None and not engaged:
                    _engage(tracked)
            if gripper_lms is not None:
                grip_fraction = _pinch_open_fraction(gripper_lms)
                gripper_xyz, _ = _hand_cam_xyz(gripper_lms, depth, intr)

            # Both palms held together is an explicit emergency-style return:
            # stop following before initiating the blocking, smooth HOME move.
            xyz_by_side = {
                args.arm_hand: tracked,
                gripper_hand: gripper_xyz,
            }
            if touch_home.update(
                    hands.get("left"), hands.get("right"),
                    xyz_by_side.get("left"), xyz_by_side.get("right"), now):
                engaged = False
                mode = "xyz"
                fist_streak = 0
                print("[depth-follow] BOTH HANDS TOUCHED — disengaged; going HOME")
                if arm_driver is not None:
                    arm_driver.home_and_confirm()
                    mapper.reset(anchor=arm_driver.get_ee_xyz())
                    wrist_mapper.reset(anchor_wrist_q=arm_driver.get_wrist_q())
                else:
                    mapper.reset(anchor=C.ARM_HOME)
                    wrist_mapper.reset()
                if tracked is not None:
                    mapper.update(tracked, 0.0)

            # Mode switch on a DEBOUNCED fist state. Closed fist -> wrist (J4/J5/J6),
            # open hand -> XYZ translate. On switch, re-anchor the target mappers so
            # neither the wrist nor the XYZ jumps.
            desired_mode = "wrist" if fist else "xyz"
            if has_hand and desired_mode != mode:
                fist_streak += 1
                if fist_streak >= 3:
                    mode = desired_mode
                    fist_streak = 0
                    if mode == "wrist":
                        anchor = arm_driver.get_wrist_q() if arm_driver else np.zeros(3)
                        wrist_mapper.reset(anchor_wrist_q=anchor)
                        if tracked is not None:
                            wrist_mapper.update(tracked, 0.0)
                        print("[depth-follow] MODE: WRIST (J4/J5/J6)")
                    else:
                        anchor = arm_driver.get_ee_xyz() if arm_driver else None
                        mapper.reset(anchor=anchor)
                        if tracked is not None:
                            mapper.update(tracked, 0.0)
                        print("[depth-follow] MODE: XYZ (translate)")
            else:
                fist_streak = 0

            # advance whichever mapper is active
            if mode == "xyz":
                target = mapper.update(tracked, dt)
                wrist_target = wrist_mapper.target
            else:
                wrist_target = wrist_mapper.update(tracked, dt)
                target = mapper.target

            # SAFETY: only command the arm while ENGAGED. Disengaged -> never call
            # the arm, so random tracking can never move it.
            if arm_driver is not None and engaged:
                if mode == "xyz":
                    arm_driver.set_target(target)
                else:
                    arm_driver.set_wrist_q(wrist_target)
                if grip_fraction is not None:
                    arm_driver.set_gripper_fraction(grip_fraction)

            frame = color.copy()
            if arm_lms is not None:
                _draw_hand(frame, arm_lms, (0, 200, 255),
                           f"{args.arm_hand}: ARM")
            if gripper_lms is not None:
                _draw_hand(frame, gripper_lms, (255, 180, 80),
                           f"{gripper_hand}: GRIPPER")
            frame = _annotate(frame, tracked, hand_px, mapper, has_hand,
                              engaged=engaged, arm_mode=arm_driver is not None,
                              mode=mode, wrist_target=wrist_target,
                              grip_fraction=grip_fraction)
            last_annotated = frame

            if count % 15 == 0:
                td = "none" if tracked is None else f"{tracked[0]:+.3f},{tracked[1]:+.3f},{tracked[2]:+.3f}"
                eng = "-" if arm_driver is None else ("ENGAGED" if engaged else "disengaged")
                if mode == "xyz":
                    t = mapper.target
                    cmd = f"xyz=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})"
                else:
                    wt = np.degrees(wrist_target)
                    cmd = f"wrist_deg=({wt[0]:+.0f},{wt[1]:+.0f},{wt[2]:+.0f})"
                grip_log = ""
                if arm_driver is not None:
                    gf, gp, gs = arm_driver.get_gripper_feedback()
                    grip_log = f"  grip_cmd={gf:.0%} actual={gp:+.2f}rad status={gs}"
                print(f"[{count:04d}] mode={mode} hand_cam=({td})  {cmd}  "
                      f"arm={eng}{grip_log}")

            if not args.headless:
                # Explicitly resize the pixels to the current client area. Some
                # Qt/OpenCV builds retain white margins even with FREERATIO set.
                # Ignore the bogus tiny client rect Qt reports before realization
                # so it cannot collapse the initial window.
                if not fullscreen:
                    try:
                        _, _, ww, wh = cv2.getWindowImageRect(win)
                        if ww >= 160 and wh >= 120:
                            disp_size = (ww, wh)
                            windowed_size = disp_size
                    except cv2.error:
                        pass
                # Preserve the camera aspect ratio and make the image as large
                # as possible: one dimension always fills the client area. Put
                # it on a client-sized BLACK canvas so Qt cannot expose white
                # background around the image.
                dw, dh = disp_size
                fh, fw = frame.shape[:2]
                scale = min(dw / fw, dh / fh)
                rw = max(1, min(dw, int(round(fw * scale))))
                rh = max(1, min(dh, int(round(fh * scale))))
                fitted = cv2.resize(
                    frame, (rw, rh), interpolation=cv2.INTER_LINEAR,
                )
                shown = np.zeros((dh, dw, 3), dtype=np.uint8)
                x0, y0 = (dw - rw) // 2, (dh - rh) // 2
                shown[y0:y0 + rh, x0:x0 + rw] = fitted
                cv2.imshow(win, shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        # Do not use getWindowImageRect in fullscreen: on some Qt
                        # builds it reports only the old image rectangle, leaving
                        # a white border on every side.
                        try:
                            wx, wy, ww, wh = cv2.getWindowImageRect(win)
                            centre = (wx + ww // 2, wy + wh // 2)
                        except cv2.error:
                            centre = None
                        disp_size = _screen_size(disp_size, centre)
                        cv2.setWindowProperty(
                            win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
                        )
                    else:
                        cv2.setWindowProperty(
                            win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL,
                        )
                        disp_size = windowed_size
                        cv2.resizeWindow(win, *windowed_size)
                if key == ord('e'):        # engage toggle (deadman clutch)
                    if engaged:
                        engaged = False
                        print("[depth-follow] DISENGAGED — arm holding, tracking ignored")
                    else:
                        _engage(tracked)
                if key in (ord('n'), ord(' ')):
                    if mode == "wrist":
                        wrist_mapper.reset(anchor_wrist_q=arm_driver.get_wrist_q() if arm_driver else np.zeros(3))
                        if tracked is not None:
                            wrist_mapper.update(tracked, dt)
                    else:
                        mapper.reset(anchor=arm_driver.get_ee_xyz() if arm_driver else None)
                        if tracked is not None:
                            mapper.update(tracked, dt)
                    print(f"[depth-follow] neutral set ({mode})" if tracked is not None
                          else "[depth-follow] no valid hand depth; neutral not set")
                if key == ord('r'):
                    mapper.reset()
                    wrist_mapper.reset()
                    mode = "xyz"
                    fist_streak = 0
                    engaged = False
                    print("[depth-follow] reset + disengaged")

            count += 1
            if args.frames and count >= args.frames:
                break
    finally:
        if args.save_annotated and last_annotated is not None:
            cv2.imwrite(args.save_annotated, last_annotated)
            print(f"[depth-follow] saved annotated frame -> {args.save_annotated}")
        if arm_driver is not None:
            print("[depth-follow] safe-homing + disabling arm ...")
            arm_driver.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()
        try:
            cam.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
