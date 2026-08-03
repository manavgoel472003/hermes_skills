"""Eye-in-hand pick & place: the wrist RGB-D camera guides the grasp, a taught
drop zone receives the object.

This is the WRIST-camera sibling of ``depth_servo`` (overhead camera). Bolting the
camera to the gripper changes the geometry in one decisive way:

    the TCP is at a FIXED point in the camera frame.

An overhead servo must find the gripper tip in every frame. Here the tip cannot
move in the image, so it is localized once from an object manually held in the
jaws, without commanding the arm or gripper, and cached. Every servo iteration
then only has to find the object.

The flip side: a static object appears to move *opposite* to the arm, so the
learned Jacobian's signs are inverted relative to the overhead case. Rather than
reason about that, ``learn_jacobian`` measures it -- jog each base axis, watch the
object -- so the sign convention is whatever the hardware actually does. The
alignment target then falls out with no camera pose and no hand-eye calibration:

    target_obj_cam = tcp_cam + APPROACH_M * J[:, 2]

i.e. "where the object would appear if the TCP were hovering APPROACH_M above it",
since J[:, 2] is exactly how the object's camera-frame position shifts per metre of
base-frame lift. Servo the object onto that, drop straight down under torque
gating, close, verify. ``arm_agent.wrist_sim`` renders this geometry from first
principles offline, so the whole loop is verifiable without hardware.

Placing is deliberately NOT visual: the drop zone is taught once (park the arm
where you want the release and record the pose), which is the reliable half of a
demo whose interesting half is the grasp.

Skills, each precondition -> arm_safe-gated action -> post-check -> safe reset,
returning ``{ok, reason, steps, verified}`` for a planner to chain on:
  * ``servo_pick(object)``       -> leaves the object held
  * ``place_in_drop_zone()``     -> releases it at the taught pose
  * ``pick_and_place(object)``   -> both, for a one-shot demo

CLI (``conda activate rebot_rs`` first)::

    python -m arm_agent.wrist_servo --sim                       # no hardware at all
    python -m arm_agent.wrist_servo --watch --object "red block" --live   # aim the camera
    python -m arm_agent.wrist_servo --calib-grasp-held "red block"       # TCP, camera-only
    python -m arm_agent.wrist_servo --teach-drop --run          # records the current pose
    python -m arm_agent.wrist_servo --gripper open --run        # jaw read-back
    python -m arm_agent.wrist_servo --pick "red block" --dry --live
    python -m arm_agent.wrist_servo --pick-place "red block" --run --live
    python -m arm_agent.wrist_servo --home --run

Env:
  REBOT_WRIST_CAMERA    realsense (default) | orbbec | auto
  REBOT_WRIST_TCP       "x,y,z" mm, overrides the cached TCP-in-camera calibration
  REBOT_WRIST_DROP      "x,y,z" m, overrides the taught drop zone
  REBOT_WRIST_SCAN      "x,y,z" m scan pose (else derived from arm_safe bounds)
  REBOT_WRIST_APPROACH  hover height above the object before descending (m)
  REBOT_WRIST_DEBUG     where annotated frames are written
  REBOT_WRIST_NEAR_MM / REBOT_WRIST_FAR_MM   detector depth gate (see wrist_detect)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_agent import arm_safe as A  # the shared, IK-verified, arm-agnostic motion gate
from arm_agent import grasp_geometry as GG
from arm_agent import scene_collision as SC
from arm_agent import wrist_detect as WD
from arm_agent.depth_camera import open_depth_camera

# ---- tunables (env-overridable) --------------------------------------------------
CAMERA_KIND = os.getenv("REBOT_WRIST_CAMERA", "realsense")
# Tool pitch the demo works at. NOT arm_safe's DEF_PITCH (1.30 rad / 74 deg): at
# that near-vertical wrist the RS arm can only reach ~23 mm above the table, which
# is not enough to clear, grasp and lift a block. 1.0 rad (57 deg) still aims the
# wrist camera down at the table but buys ~104 mm of working height. Deliberate
# absolute moves command it explicitly; relative servo steps preserve orientation.
WORK_PITCH = float(os.getenv("REBOT_WRIST_PITCH", "1.0"))
# A short vertical move bends the otherwise nearly straight home configuration
# enough for local Cartesian jogs.  Approach mode preserves the live orientation:
# it does not use the pick posture and does not twist the wrist.
APPROACH_READY_LIFT_M = float(os.getenv("REBOT_WRIST_APPROACH_READY_LIFT_M", "0.03"))
APPROACH_WORKSPACE_MARGIN_M = float(
    os.getenv("REBOT_WRIST_APPROACH_WORKSPACE_MARGIN_M", "0.015")
)
JAC_STEP = 0.020         # base jog per axis when learning the Jacobian (m)
APPROACH_M = float(os.getenv("REBOT_WRIST_APPROACH", "0.060"))
# In approach-only mode, "distance" means camera depth beyond the saved visual
# gripper target.  Base-frame Z is not camera depth when the wrist camera is
# tilted/side-looking: on hardware a 60 mm base-Z offset changed depth by only
# ~15 mm and produced an unsafe-looking 82 mm lens-to-object target.
APPROACH_CAMERA_GAP_MM = float(
    os.getenv("REBOT_WRIST_APPROACH_CAMERA_GAP_MM", str(APPROACH_M * 1000.0))
)
# Fixed-trajectory approach stops short of the visually calibrated gripper point.
# This is deliberately non-zero: there is no contact validation in this mode.
APPROACH_NEAR_TOUCH_GAP_MM = float(
    os.getenv(
        "REBOT_WRIST_APPROACH_NEAR_TOUCH_GAP_MM",
        # Backward-compatible and conservative: an old explicit camera-gap
        # command remains far rather than silently becoming a 20 mm near-touch.
        os.getenv("REBOT_WRIST_APPROACH_CAMERA_GAP_MM", "20.0"),
    )
)
APPROACH_TRAJECTORY_CAP_FRACTION = float(
    os.getenv("REBOT_WRIST_APPROACH_TRAJECTORY_CAP_FRACTION", "0.50")
)
APPROACH_SEGMENT_BISECT_DEPTH = int(
    os.getenv("REBOT_WRIST_APPROACH_SEGMENT_BISECT_DEPTH", "4")
)
FIXED_PICK_OPEN_BEFORE_END_M = float(
    os.getenv("REBOT_WRIST_PICK_OPEN_BEFORE_END_M", "0.040")
)
FIXED_PICK_CHECKPOINT_STEP_M = float(
    os.getenv("REBOT_WRIST_PICK_CHECKPOINT_STEP_M", "0.040")
)
FIXED_PICK_MAX_CHECKPOINTS = int(
    os.getenv("REBOT_WRIST_PICK_MAX_CHECKPOINTS", "8")
)
FIXED_PICK_FINAL_GAP_MM = float(
    os.getenv("REBOT_WRIST_PICK_FINAL_GAP_MM", "5.0")
)
FIXED_PICK_CLOSER_MM = float(
    os.getenv("REBOT_WRIST_PICK_CLOSER_MM", "0.0")
)
FIXED_PICK_TARGET_Y_OFFSET_MM = float(
    os.getenv("REBOT_WRIST_PICK_TARGET_Y_OFFSET_MM", "0.0")
)
FIXED_PICK_TARGET_TOL_MM = float(
    os.getenv("REBOT_WRIST_PICK_TARGET_TOL_MM", "15.0")
)
FIXED_PICK_MAX_CORRECTIONS = int(
    os.getenv("REBOT_WRIST_PICK_MAX_CORRECTIONS", "2")
)
FIXED_PICK_MAX_CORRECTION_M = float(
    os.getenv("REBOT_WRIST_PICK_MAX_CORRECTION_M", "0.050")
)
FIXED_PICK_LIFT_M = float(os.getenv("REBOT_WRIST_PICK_LIFT_M", "0.050"))
DYNAMIC_GRASP_ENABLED = os.getenv("REBOT_DYNAMIC_GRASP", "0").strip().lower() \
    in {"1", "true", "yes", "on"}
START_GRIPPER_CLOSED = os.getenv(
    "REBOT_WRIST_CLOSE_GRIPPER_AT_START", "0"
).strip().lower() in {"1", "true", "yes", "on"}
DYNAMIC_REMEASURE_MAX_DELTA_MM = float(
    os.getenv("REBOT_DYNAMIC_REMEASURE_MAX_DELTA_MM", "6.0")
)
DYNAMIC_MICRO_LIFT_M = float(os.getenv("REBOT_DYNAMIC_MICRO_LIFT_M", "0.015"))
DYNAMIC_HOLD_TOL_MM = float(os.getenv("REBOT_DYNAMIC_HOLD_TOL_MM", "6.0"))
DYNAMIC_SLOT_TOL_MM = float(os.getenv("REBOT_DYNAMIC_SLOT_TOL_MM", "6.0"))
DYNAMIC_RETRY_CLOSE_MM = float(os.getenv("REBOT_DYNAMIC_RETRY_CLOSE_MM", "1.0"))
DYNAMIC_RETRY_CLOSE_DEG = float(os.getenv("REBOT_DYNAMIC_RETRY_CLOSE_DEG", "5.0"))
DYNAMIC_MAX_RETRIES = int(os.getenv("REBOT_DYNAMIC_MAX_RETRIES", "3"))
DYNAMIC_CLOSE_STEP_DEG = float(os.getenv("REBOT_DYNAMIC_CLOSE_STEP_DEG", "15.0"))
GEOMETRY_MAX_YAW_DELTA_DEG = float(
    os.getenv("REBOT_WRIST_GRASP_MAX_YAW_DELTA_DEG", "10.0")
)
GEOMETRY_YAW_MEASUREMENT_SLOP_DEG = float(
    os.getenv("REBOT_WRIST_GRASP_YAW_MEASUREMENT_SLOP_DEG", "0.5")
)
GEOMETRY_FINAL_APPROACH_MM = float(
    os.getenv("REBOT_WRIST_GRASP_FINAL_APPROACH_MM", "40.0")
)
GEOMETRY_MIN_FINAL_APPROACH_MM = float(
    os.getenv("REBOT_WRIST_GRASP_MIN_FINAL_APPROACH_MM", "20.0")
)
GEOMETRY_RPY_STEP_DEG = float(
    os.getenv("REBOT_WRIST_GRASP_RPY_STEP_DEG", "2.0")
)
GEOMETRY_MIN_FEATURE_PROMINENCE_MM = float(
    os.getenv("REBOT_WRIST_GRASP_MIN_FEATURE_PROMINENCE_MM", "25.0")
)
FIXED_HELD_RIGID_TOL_MM = float(
    os.getenv("REBOT_WRIST_HELD_RIGID_TOL_MM", "15.0")
)
FIXED_DROP_RELEASE_Z_M = float(
    os.getenv("REBOT_WRIST_DROP_RELEASE_Z_M", "0.200")
)
SCENE_PLACE_OFFSET_M = float(
    os.getenv("REBOT_WRIST_SCENE_PLACE_OFFSET_M", "0.090")
)
SCENE_PLACE_CONSENSUS_FRAMES = int(
    os.getenv("REBOT_WRIST_SCENE_PLACE_CONSENSUS_FRAMES", "3")
)
SCENE_PLACE_MAX_SPREAD_MM = float(
    os.getenv("REBOT_WRIST_SCENE_PLACE_MAX_SPREAD_MM", "15.0")
)
SCENE_COLLISION_ENABLED = os.getenv(
    "REBOT_SCENE_COLLISION", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SCENE_COLLISION_VOXEL_M = float(
    os.getenv("REBOT_SCENE_COLLISION_VOXEL_M", "0.020")
)
SCENE_COLLISION_INFLATION_M = float(
    os.getenv("REBOT_SCENE_COLLISION_INFLATION_M", "0.025")
)
SCENE_COLLISION_STRIDE_PX = int(
    os.getenv("REBOT_SCENE_COLLISION_STRIDE_PX", "4")
)
ALIGN_TOL_MM = 8.0       # camera-frame object<->target error we call "aligned"
STALL_ITERS = 3          # no improvement for this many steps => at the sensor's noise floor
GAIN = 0.6               # servo step = GAIN * Jinv @ error (then step-capped)
MAX_ITERS = 24           # floor; the real budget is derived from the workspace
SETTLE_S = 0.35          # let the arm settle before re-observing
LOST_RETRIES = 3         # re-grabs before declaring the object lost mid-servo
GRASP_GAP_RAD = 0.09     # gripper stopped >this short of full close => holding
GRIPPER_NEAR_RAD = 0.12  # gripper within this of the open target => released
MIN_SIZED_GRASP_MM = float(os.getenv("REBOT_GRIPPER_MIN_OBJECT_MM", "8"))
SIZE_TARGET_TOL_MM = float(os.getenv("REBOT_GRIPPER_SIZE_TARGET_TOL_MM", "1.5"))
SIZE_SAFETY_MARGIN_MM = float(os.getenv("REBOT_GRIPPER_SIZE_MARGIN_MM", "0"))
SIZE_COMPRESSION_MM = float(os.getenv("REBOT_GRIPPER_SIZE_COMPRESSION_MM", "0"))
HELD_TCP_TOL_MM = float(os.getenv("REBOT_WRIST_HELD_TCP_TOL_MM", "20"))
# A wrist camera sits centimetres from the jaws, so a HELD object must be close.
# Anything outside this band means the detector found a different object.
TCP_MIN_MM = float(os.getenv("REBOT_WRIST_TCP_MIN_MM", "20"))
TCP_MAX_MM = float(os.getenv("REBOT_WRIST_TCP_MAX_MM", "250"))
# Use the active wrist detector range for the depth inset as well. This follows
# REBOT_WRIST_NEAR_MM/FAR_MM and keeps close-range calibration views legible.
DEPTH_VIS_MM = (WD.NEAR_MM, WD.FAR_MM)
# Jaw limits THIS SKILL commands, in radians. Defaults to the measured pair; set
# REBOT_WRIST_GRIPPER_OPEN_DEG / _CLOSE_DEG to be more conservative here without
# narrowing the gripper for depth_servo, motion_tip or the MCP open/close tools.
def _wrist_limits() -> tuple[float, float]:
    from arm_agent import gripper_model as gm
    return (gm._limit("REBOT_WRIST_GRIPPER_CLOSE_RAD", "REBOT_WRIST_GRIPPER_CLOSE_DEG",
                      gm.deg(gm.CLOSE_RAD)),
            gm._limit("REBOT_WRIST_GRIPPER_OPEN_RAD", "REBOT_WRIST_GRIPPER_OPEN_DEG",
                      gm.deg(gm.OPEN_RAD)))

CALIB_DIR = Path(os.getenv("REBOT_WRIST_CALIB", str(ROOT / "calib")))
TCP_FILE = CALIB_DIR / "wrist_tcp.json"
GRASP_TARGET_FILE = CALIB_DIR / "wrist_grasp_target.json"
DROP_FILE = CALIB_DIR / "wrist_drop_zone.json"
SAFETY_FILE = CALIB_DIR / "wrist_safety.json"
CAMERA_ROTATION_FILE = CALIB_DIR / "wrist_camera_rotation.json"
DEBUG_DIR = Path(os.getenv(
    "REBOT_WRIST_DEBUG",
    "/tmp/claude-1000/-home-manav-rebot-lerobot/"
    "1b94e05b-7601-46f2-a0fe-97b0ab3194ba/scratchpad/wrist_servo"))

# Swapped out by the offline simulator (see wrist_sim / run_sim).
_CAMERA_FACTORY: Callable[[str], Any] = open_depth_camera


def install_backend(arm: Any = None, camera_factory: Callable[[str], Any] | None = None,
                    settle_s: float | None = None) -> None:
    """Replace the motion gate / camera. Only the simulator and tests use this;
    the hardware path always runs against ``arm_safe`` + a real depth camera."""
    global A, _CAMERA_FACTORY, SETTLE_S
    if arm is not None:
        A = arm
    if camera_factory is not None:
        _CAMERA_FACTORY = camera_factory
    if settle_s is not None:
        SETTLE_S = settle_s


# =====================================================================================
# Visualisation (reuses depth_servo's file + MJPEG plumbing, wrist-specific overlay)
# =====================================================================================
def _depth_inset(depth: np.ndarray, w: int = 200) -> np.ndarray:
    lo, hi = DEPTH_VIS_MM
    norm = np.clip((depth - lo) / max(1.0, hi - lo), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth <= 0] = (0, 0, 0)
    h = int(w * depth.shape[0] / depth.shape[1])
    return cv2.resize(vis, (w, h))


def annotate(color, depth=None, intr=None, *, det=None, tcp_cam=None, target_cam=None,
             err_mm=None, note=None, title=None, candidates=None):
    """Draw what the wrist servo sees: object, the fixed TCP, and the hover target."""
    vis = color.copy()
    if det is not None:
        b = [int(v) for v in det.box]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (80, 220, 80), 2)
        p = (int(det.uv[0]), int(det.uv[1]))
        cv2.circle(vis, p, 6, (80, 220, 80), -1)
        cv2.putText(vis, f"{det.source} {det.z_mm:.0f}mm", (b[0], max(14, b[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
    if candidates:
        for index, candidate in enumerate(candidates, 1):
            try:
                u, v = (int(round(float(value))) for value in candidate["uv"])
                width = float(candidate["width_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            cv2.circle(vis, (u, v), 7, (255, 180, 0), 2)
            cv2.putText(vis, f"{index}:{width:.0f}mm", (u + 8, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1)
    tgt_uv = WD.project(target_cam, intr) if (target_cam is not None and intr is not None) else None
    if tgt_uv is not None:
        t = (int(tgt_uv[0]), int(tgt_uv[1]))
        cv2.drawMarker(vis, t, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 2)
        cv2.putText(vis, "target", (t[0] + 10, t[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if det is not None:
            cv2.arrowedLine(vis, (int(det.uv[0]), int(det.uv[1])), t, (0, 255, 255), 2, tipLength=0.2)
    tcp_uv = WD.project(tcp_cam, intr) if (tcp_cam is not None and intr is not None) else None
    if tcp_uv is not None:
        p = (int(tcp_uv[0]), int(tcp_uv[1]))
        cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis, "TCP", (p[0] + 10, p[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    banner = title or "wrist_servo"
    if err_mm is not None:
        banner += f"   err={err_mm:.0f}mm"
    if note:
        banner += f"   {note}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if depth is not None:
        inset = _depth_inset(depth)
        ih, iw = inset.shape[:2]
        vis[vis.shape[0] - ih:, vis.shape[1] - iw:] = inset
    return vis


_VIZ: Any = None
_OBJECT_HELD: bool | None = False
# This context deliberately lives only in the controller process.  A place call
# after a process restart must not assume that an object is still secured merely
# because a file on disk says that an earlier pick succeeded.
_FIXED_HELD_CONTEXT: dict[str, Any] | None = None
_INTERRUPT_HOME_ENABLED = False


def set_viz(viz: Any) -> None:
    global _VIZ
    _VIZ = viz


def make_viz(outdir: Path | str = DEBUG_DIR, *, mjpeg_port: int | None = None,
             window: bool = False):
    from arm_agent.depth_servo import Viz
    return Viz(outdir=Path(outdir), mjpeg_port=mjpeg_port, window=window,
               annotator=annotate, title="wrist_servo")


def _show(color, depth=None, intr=None, **kw) -> None:
    if _VIZ is not None:
        _VIZ.show(color, depth, intr, **kw)


def _settle(eye: "WristEye | None" = None, seconds: float | None = None, **kw: Any) -> None:
    """Wait for the arm to settle, streaming the wrist view while we wait.

    The servo only grabs a frame per iteration, so a viewer would otherwise sit on
    a frozen image between steps. Pumping frames through the settle window makes
    the live feed move continuously. Frames are pulled on THIS thread -- a second
    thread on the same RealSense pipeline is not safe.
    """
    seconds = SETTLE_S if seconds is None else seconds
    if eye is None or _VIZ is None or seconds <= 0:
        time.sleep(max(0.0, seconds))
        return
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            color, depth, intr = eye.frame()
            _VIZ.show(color, depth, intr, tag="live", save=False, **kw)
        except Exception:  # noqa: BLE001 - a dropped frame must not break motion
            time.sleep(0.05)
        time.sleep(0.02)


# =====================================================================================
# Calibration state: TCP-in-camera (fixed by the mount) and the taught drop zone
# =====================================================================================
def _parse_triplet(raw: str | None) -> tuple[float, float, float] | None:
    if not raw:
        return None
    try:
        x, y, z = (float(v) for v in raw.split(","))
    except ValueError:
        return None
    return (x, y, z)


def load_tcp() -> tuple[float, float, float] | None:
    """TCP position in the camera frame (mm). Fixed by the mount, so cached."""
    env = _parse_triplet(os.getenv("REBOT_WRIST_TCP"))
    if env is not None:
        return env
    try:
        data = json.loads(TCP_FILE.read_text())
        return tuple(float(v) for v in data["tcp_camera_xyz_mm"])  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def save_tcp(p_mm: tuple[float, float, float], detail: dict[str, Any] | None = None) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    TCP_FILE.write_text(json.dumps({
        "tcp_camera_xyz_mm": [float(v) for v in p_mm],
        "camera": CAMERA_KIND, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detail": detail or {},
    }, indent=2))
    return TCP_FILE


def load_grasp_target() -> tuple[float, float, float] | None:
    """Object position in camera mm when centered inside the open jaw slot."""
    env = _parse_triplet(os.getenv("REBOT_WRIST_PICK_TARGET_CAM_MM"))
    if env is not None:
        return env
    try:
        data = json.loads(GRASP_TARGET_FILE.read_text())
        return tuple(float(v) for v in data["object_camera_xyz_mm"])  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def load_grasp_target_detail() -> dict[str, Any]:
    """Non-positional camera/gripper geometry saved with the jaw-slot target."""
    try:
        data = json.loads(GRASP_TARGET_FILE.read_text())
        detail = data.get("detail")
        return detail if isinstance(detail, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _rpy_rotation_matrix(rpy: Any) -> np.ndarray:
    """Base-frame tool rotation using the arm IK/FK roll-pitch-yaw convention."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def load_saved_camera_jacobian(
    current_rpy: tuple[float, float, float],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load the rigid wrist-camera rotation measured on textured targets.

    Camera translation geometry belongs to the bolted camera mount, not to the
    object being picked. A saved transform is accepted only if it is finite,
    orthonormal after the normal rigid-fit validation, and the wrist remains
    near the orientation at which it was calibrated.
    """
    try:
        data = json.loads(CAMERA_ROTATION_FILE.read_text())
        matrix = 1000.0 * np.asarray(
            data["base_to_camera_rotation"], dtype=float
        )
        calibrated_rpy = np.asarray(data["calibrated_tool_rpy_rad"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"saved camera rotation unavailable: {exc}"}
    if matrix.shape != (3, 3) or calibrated_rpy.shape != (3,):
        return None, {"reason": "saved camera rotation has invalid dimensions"}
    calibrated_rigid, detail = _rigid_camera_jacobian(matrix)
    if calibrated_rigid is None:
        return None, {"reason": f"saved camera rotation failed validation: "
                                f"{detail.get('reason')}"}
    calibrated_rotation = _rpy_rotation_matrix(calibrated_rpy)
    current_rotation = _rpy_rotation_matrix(current_rpy)
    relative_rotation = calibrated_rotation.T @ current_rotation
    orientation_delta_deg = math.degrees(math.acos(float(np.clip(
        (np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0
    ))))
    allowed_deg = float(np.clip(float(os.getenv(
        "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG", "2.0"
    )), 0.25, 10.0))
    if orientation_delta_deg > allowed_deg:
        return None, {
            "reason": (
                f"wrist orientation is {orientation_delta_deg:.2f}deg from the saved "
                f"camera calibration (maximum {allowed_deg:.2f}deg)"
            )
        }
    # M_cal maps base translation to apparent object motion in the calibrated
    # camera frame. A camera bolted to the tool at a new orientation has
    # M_cur = M_cal * R_cal * R_cur^T. This analytically rotates the saved rigid
    # mapping; it does not estimate geometry from the current object.
    adjusted = calibrated_rigid @ calibrated_rotation @ current_rotation.T
    rigid, adjusted_detail = _rigid_camera_jacobian(adjusted)
    if rigid is None:
        return None, {"reason": "orientation-adjusted camera rotation failed "
                                f"validation: {adjusted_detail.get('reason')}"}
    return rigid, {
        **adjusted_detail,
        "reason": (
            f"saved rigid camera mount calibration rotated to current wrist; "
            f"orientation delta={orientation_delta_deg:.2f}deg"
        ),
        "file": str(CAMERA_ROTATION_FILE),
        "orientation_delta_deg": round(orientation_delta_deg, 3),
        "orientation_adjusted": orientation_delta_deg > 0.05,
    }


def save_grasp_target(p_mm: tuple[float, float, float],
                      detail: dict[str, Any] | None = None) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    GRASP_TARGET_FILE.write_text(json.dumps({
        "object_camera_xyz_mm": [float(v) for v in p_mm],
        "camera": CAMERA_KIND,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detail": detail or {},
    }, indent=2))
    return GRASP_TARGET_FILE


def load_drop_zone() -> tuple[float, float, float] | None:
    """Taught release pose (base frame, metres)."""
    env = _parse_triplet(os.getenv("REBOT_WRIST_DROP"))
    if env is not None:
        return env
    try:
        data = json.loads(DROP_FILE.read_text())
        return tuple(float(v) for v in data["release_xyz_m"])  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def save_drop_zone(xyz: tuple[float, float, float]) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    DROP_FILE.write_text(json.dumps({
        "release_xyz_m": [float(v) for v in xyz],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    return DROP_FILE


def load_safety_calibration() -> dict[str, Any]:
    """Measured descent prerequisites, independent of the broad workspace box."""
    try:
        data = json.loads(SAFETY_FILE.read_text())
    except Exception:  # noqa: BLE001
        data = {}
    raw_floor = os.getenv("REBOT_TABLE_Z")
    if raw_floor:
        try:
            data["table_z_m"] = float(raw_floor)
            data["table_z_source"] = "env"
        except ValueError:
            pass
    return data


def descent_readiness(*, require_table: bool = True) -> tuple[bool, str, float | None]:
    data = load_safety_calibration()
    floor = data.get("table_z_m")
    try:
        floor = None if floor is None else float(floor)
    except (TypeError, ValueError):
        floor = None
    if not bool(data.get("torque_feedback_verified")):
        return False, "torque feedback has not been hardware-verified", floor
    if require_table and floor is None:
        return False, "table height has not been measured", None
    if floor is not None and not A.Z_MIN <= floor <= A.Z_MAX:
        return False, f"measured table z={floor:.4f}m is outside workspace", floor
    return True, (
        f"torque verified; table z={floor:.4f}m" if floor is not None
        else "torque verified"
    ), floor


def check_tcp(object_name: str, *, verbose: bool = False) -> dict[str, Any]:
    """Compare the stored TCP against a held object, WITHOUT overwriting it.

    Use this to validate a TCP taken from CAD. A held object sits at the TCP, so the
    gap between where the camera sees it and where the stored TCP claims the TCP is
    IS the calibration error -- which is what decides whether a grasp lands. CAD
    gives the mount geometry, but not the camera's optical-centre offset inside its
    housing, nor print and seating tolerance, so this check is worth one minute.
    """
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    tcp = load_tcp()
    steps.append(("stored_tcp", tcp is not None,
                  f"{tuple(round(v, 1) for v in tcp)}mm" if tcp else "none stored"))
    if tcp is None:
        return _result(False, "no TCP stored to check", steps)

    eye = WristEye()
    try:
        steps.append(("camera_only", True,
                      "arm and gripper were not connected, read, enabled, or commanded"))
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            return _result(False, f"wrist camera unavailable: {exc}", steps)
        # TCP calibration is the one context where the object must be beside the
        # camera, not anywhere in the normal table-working range. Restrict the
        # detector up front so a larger same-coloured item on the table cannot win
        # and then merely be rejected by the distance plausibility check.
        det, why = eye.object(
            object_name,
            color,
            depth,
            intr,
            near_mm=max(WD.NEAR_MM, TCP_MIN_MM),
            far_mm=min(WD.FAR_MM, TCP_MAX_MM),
        )
        _show(color, depth, intr, tag="check_tcp", det=det,
              tcp_cam=np.asarray(tcp, float), note="stored TCP vs held object")
        steps.append(("detect_held_object", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
        if det is None:
            return _result(False, f"could not see the held '{object_name}'", steps)

        seen = np.asarray(det.p_cam_mm, float)
        error = seen - np.asarray(tcp, float)
        norm = float(np.linalg.norm(error))
        verdict = ("good" if norm < 5 else "usable" if norm < 12 else "too large")
        steps.append(("compare", norm < 12, f"error {norm:.1f}mm ({verdict})"))
        log(f"  stored TCP {tuple(round(v,1) for v in tcp)}mm")
        log(f"  held object at {tuple(round(float(v),1) for v in seen)}mm")
        log(f"  error {tuple(round(float(v),1) for v in error)}mm  |{norm:.1f}mm| -> {verdict}")
        return _result(norm < 12, f"TCP error {norm:.1f}mm ({verdict})", steps,
                       stored_tcp_mm=[round(v, 2) for v in tcp],
                       observed_mm=[round(float(v), 2) for v in seen],
                       error_mm=[round(float(v), 2) for v in error],
                       error_norm_mm=round(norm, 2))
    finally:
        eye.close()


def calibrate_jaw_scale(object_name: str, *, verbose: bool = False) -> dict[str, Any]:
    """Measure millimetres-per-radian by gripping one object, no ruler needed.

    Radians alone cannot give millimetres -- the motor-to-finger linkage is not in
    the URDF. But the two signals we already have close the gap: the camera measures
    the object's real width, and the motor reports the angle at which the jaws
    stopped on it. One grip fixes the scale, after which the gripper can be opened
    to fit any measured object and impossible picks refused up front.

    Put the object between the jaws, then run this. Moves the gripper only.
    """
    return _result(
        False,
        "automatic close-based jaw calibration is disabled because contact cannot "
        "be validated safely. Measure a known open gap and use --set-jaw-span MM",
        [("safety_policy", False, "no gripper command was sent")],
    )

    from arm_agent import gripper_model as gm
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    pf = A.preflight(connect=True)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        return _result(False, f"arm not ready: {pf.reason}", steps)
    eye = WristEye()
    try:
        # Measure the object while the jaws are open, before they occlude it.
        _gripper("open")
        time.sleep(0.5)
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            steps.append(("camera", False, str(exc)))
            return _result(False, f"wrist camera unavailable ({eye.camera_kind})", steps)
        # Search only the volume physically close enough to contain the jaws.
        # Normal wrist detection includes the table out to 600 mm, where a larger
        # same-coloured object could otherwise mask the held calibration object.
        det, why = eye.object(
            object_name,
            color,
            depth,
            intr,
            near_mm=max(WD.NEAR_MM, TCP_MIN_MM),
            far_mm=min(WD.FAR_MM, TCP_MAX_MM),
        )
        _show(color, depth, intr, tag="calib_jaw", det=det,
              note=f"measuring '{object_name}'")
        steps.append(("measure_object", det is not None and det.max_width_mm is not None,
                      f"{det.min_width_mm:.1f} x {det.max_width_mm:.1f} mm"
                      if det and det.max_width_mm else str(why)))
        if det is None or det.max_width_mm is None:
            return _result(False, f"could not measure '{object_name}' -- "
                                  "check it is in view with --watch", steps)

        resp = _gripper("close")
        time.sleep(0.8)
        if not resp.get("ok"):
            steps.append(("close_on_object", False, str(resp.get("reason"))))
            return _result(False, f"gripper command failed: {resp.get('reason')}", steps)
        contact = resp.get("position_rad")
        held = _verify_grasp(resp)
        steps.append(("close_on_object", held is not False and contact is not None,
                      f"stopped at {contact:.4f} rad" if contact is not None
                      else "no position read-back"))
        if contact is None:
            return _result(False, "the gripper gave no position read-back", steps)
        if held is False:
            return _result(False, "the jaws closed fully -- put the object between "
                                  "them first", steps)

        # The jaws stop on the object, so closed->contact spans its width. Use the
        # wide side: that is what sits between the jaws at an arbitrary orientation.
        fit = gm.fit_from_grasp(float(det.max_width_mm), float(contact))
        steps.append(("fit_scale", bool(fit.get("ok")),
                      f"{fit.get('mm_per_rad')} mm/rad, max aperture "
                      f"{fit.get('max_aperture_mm')}mm" if fit.get("ok")
                      else str(fit.get("reason"))))
        if not fit.get("ok"):
            return _result(False, str(fit.get("reason")), steps)
        log(f"  jaw scale {fit['mm_per_rad']} mm/rad; "
            f"max aperture {fit['max_aperture_mm']} mm")
        return _result(True, "jaw scale calibrated", steps, **fit)
    finally:
        eye.close()


def calibrate_tcp_by_grasp(object_name: str, *, close_first: bool = True,
                           verbose: bool = False) -> dict[str, Any]:
    """TCP-in-camera from a HELD object -- the preferred calibration.

    An object gripped between the jaws *is* at the TCP, so its camera-frame position
    IS the TCP's. Detecting it needs only the detector the servo already uses: no
    jaw has to be visible, no image differencing, nothing to go wrong if the fingers
    sit outside the frame or inside the depth camera's dead zone (the failure mode
    of the ``motion_tip`` jaw-diff method).

    It is also self-consistent: the servo drives an object's camera position to
    ``tcp_cam``, so measuring ``tcp_cam`` as "where a held object appears" cancels
    any systematic bias in the detector instead of adding to it.

    With ``close_first=True``, put the object between the open jaws; this routine
    commands them closed before reading the camera. With ``close_first=False``,
    secure the object in the jaws first: the routine uses the camera only. It does
    not connect to, read, enable, or command the arm or gripper.
    """
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    if close_first:
        return _result(
            False,
            "automatic close-based TCP calibration is disabled. Place the object "
            "in the jaws by hand and use --calib-grasp-held; no command was sent",
            [("safety_policy", False, "automatic gripper close disabled")],
        )
    else:
        steps.append((
            "camera_only",
            True,
            "operator declared object held; arm and gripper were not connected, "
            "read, enabled, or commanded",
        ))
    eye = WristEye()
    try:
        if close_first:
            resp = _gripper("close")
            time.sleep(0.8)
            # The gripper command must actually have SUCCEEDED. A failed call
            # returns {"ok": False, ...} with no position, which looks identical to
            # "no read-back" -- and silently calibrating against jaws that never
            # moved stores a table position as the TCP.
            if not resp.get("ok"):
                steps.append(("close_on_object", False,
                              f"gripper command failed: {resp.get('reason')}"))
                return _result(False, "the gripper did not respond -- check it with "
                                      "'--gripper close --run'", steps)
            held = _verify_grasp(resp)
            steps.append(("close_on_object", held is True,
                          "held" if held else ("nothing between the jaws" if held is False
                                               else "no position read-back from the gripper")))
            if held is not True:
                return _result(False, "could not confirm the object is gripped "
                                      f"({'jaws closed fully' if held is False else 'no read-back'})"
                                      " -- calibrating on an unconfirmed grasp would store "
                                      "whatever the camera happened to see", steps)
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            steps.append(("camera", False, str(exc)))
            return _result(False, f"wrist camera unavailable ({eye.camera_kind})", steps)

        # Search only the volume physically close enough to contain the jaws.
        # Normal wrist detection includes the table out to 600 mm, where a larger
        # same-coloured object could otherwise mask the held calibration object.
        det, why = eye.object(
            object_name,
            color,
            depth,
            intr,
            near_mm=max(WD.NEAR_MM, TCP_MIN_MM),
            far_mm=min(WD.FAR_MM, TCP_MAX_MM),
        )
        _show(color, depth, intr, tag="calib_grasp", det=det,
              note=f"TCP calibration from held '{object_name}'")
        steps.append(("detect_held_object", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
        if det is None:
            return _result(False, f"could not see the held '{object_name}'. Is it in "
                                  "the jaws and in view? Try --watch to check.", steps)

        p = tuple(float(v) for v in det.p_cam_mm)
        # Plausibility gate. The camera is bolted a few centimetres from the jaws,
        # so a held object CANNOT be far away. A large value means the detector
        # locked onto something else -- typically the same-coloured object still
        # lying on the table -- and storing it would send every grasp to that spot.
        distance = float(np.linalg.norm(p))
        if not TCP_MIN_MM <= distance <= TCP_MAX_MM:
            steps.append(("plausibility", False,
                          f"{distance:.0f}mm from the lens, outside "
                          f"{TCP_MIN_MM:.0f}-{TCP_MAX_MM:.0f}mm"))
            return _result(False, f"the detected '{object_name}' is {distance:.0f}mm from "
                                  "the camera -- too far to be held in the jaws. It is "
                                  "probably another one on the table; remove it, or check "
                                  "the view with --watch", steps,
                           observed_mm=[round(v, 1) for v in p])
        steps.append(("plausibility", True, f"{distance:.0f}mm from the lens"))
        path = save_tcp(p, {"method": "held_object", "object": object_name,
                            "gripper_commanded": bool(close_first),
                            "gripper_state_read": bool(close_first),
                            "held_asserted_by_operator": not close_first,
                            "uv": list(det.uv), "detector": det.source})
        log(f"  TCP in camera frame: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
        steps.append(("save", True, str(path)))
        return _result(True, "TCP calibrated from the held object", steps,
                       tcp_camera_xyz_mm=[round(v, 2) for v in p], saved=str(path))
    finally:
        eye.close()


def calibrate_grasp_target(object_name: str, *,
                           verbose: bool = False) -> dict[str, Any]:
    """Save where an object appears when manually centered in the open jaws.

    This is camera-only. It never connects to, reads, enables, or commands the
    arm or gripper. The operator holds the intended grasp feature (for a marker,
    the cap) in the middle of the open finger slot at the depth where it should
    be captured.
    """
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = [(
        "camera_only",
        True,
        "arm and gripper were not connected, read, enabled, or commanded",
    )]
    eye = WristEye()
    try:
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            steps.append(("camera", False, str(exc)))
            return _result(False, f"wrist camera unavailable ({eye.camera_kind})",
                           steps)
        det, why = eye.object(
            object_name, color, depth, intr,
            near_mm=max(WD.NEAR_MM, TCP_MIN_MM),
            far_mm=min(WD.FAR_MM, TCP_MAX_MM),
        )
        _show(color, depth, intr, tag="calib_grasp_target", det=det,
              note=f"manual open-jaw grasp target for '{object_name}'")
        steps.append(("detect_grasp_feature", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
        if det is None:
            return _result(
                False,
                f"could not see the manually positioned '{object_name}'; "
                "keep the intended grasp feature between the open fingers and in view",
                steps,
            )
        p = tuple(float(v) for v in det.p_cam_mm)
        distance = float(np.linalg.norm(p))
        plausible = TCP_MIN_MM <= distance <= TCP_MAX_MM
        steps.append(("plausibility", plausible,
                      f"{distance:.0f}mm from the lens"))
        if not plausible:
            return _result(
                False,
                f"detected feature is {distance:.0f}mm from the lens, outside the "
                f"{TCP_MIN_MM:.0f}-{TCP_MAX_MM:.0f}mm jaw region",
                steps, observed_mm=[round(v, 1) for v in p],
            )
        path = save_grasp_target(
            p,
            {"method": "manual_open_jaw_slot", "object": object_name,
             "arm_commanded": False, "gripper_commanded": False,
             "uv": list(det.uv), "detector": det.source,
             "jaw_long_axis_camera_deg":
                 det.info.get("long_axis_deg")},
        )
        log(f"  grasp target in camera frame: "
            f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
        steps.append(("save", True, str(path)))
        return _result(
            True,
            "open-jaw visual grasp target calibrated without motion",
            steps,
            grasp_target_camera_xyz_mm=[round(v, 2) for v in p],
            saved=str(path),
        )
    finally:
        eye.close()


def calibrate_tcp(camera_kind: str = CAMERA_KIND) -> dict[str, Any]:
    """Localize the TCP in the wrist camera by opening/closing the jaws.

    Moves the gripper only -- no arm joint moves -- and the result is cached
    because a rigidly mounted camera cannot see the TCP move.
    """
    return _result(
        False,
        "jaw-motion TCP calibration is disabled. Use --calib-grasp-held with an "
        "object already placed in the jaws; no gripper command was sent",
        [("safety_policy", False, "automatic gripper motion disabled")],
    )

    from arm_agent import motion_tip
    result = motion_tip.run(camera_kind)
    if not result.get("ok"):
        return {"ok": False, "reason": result.get("reason", "jaw_localization_failed"),
                "detail": result.get("detail"), "images": result.get("images")}
    p = tuple(float(v) for v in result["tip_camera_xyz_mm"])
    path = save_tcp(p, {"pixel": result.get("tip_pixel"), **(result.get("detail") or {})})
    return {"ok": True, "tcp_camera_xyz_mm": list(p), "saved": str(path),
            "images": result.get("images")}


def teach_drop_zone(xyz: tuple[float, float, float] | None = None) -> dict[str, Any]:
    """Record the release pose. With no argument, uses where the arm is NOW --
    park it over the bin (open gripper, jaws at release height) and call this."""
    if xyz is None:
        pf = A.preflight(connect=True)
        if not pf.ok:
            return {"ok": False, "reason": f"arm not ready: {pf.reason}"}
        xyz = tuple(float(v) for v in A.ee_pos())
    if not A.in_bounds(*xyz):
        return {"ok": False, "reason": f"drop zone out of workspace: {xyz}"}
    path = save_drop_zone(xyz)
    return {"ok": True, "release_xyz_m": [round(v, 4) for v in xyz], "saved": str(path)}


# =====================================================================================
# Camera + detection
# =====================================================================================
@dataclass
class WristEye:
    """Owns the wrist RGB-D stream and finds objects in it."""

    camera_kind: str = CAMERA_KIND
    allow_vlm: bool = True
    _camera: Any = field(default=None, repr=False)
    _frame_lock: Any = field(default_factory=threading.Lock, repr=False)
    _live_stop: Any = field(default_factory=threading.Event, repr=False)
    _live_thread: Any = field(default=None, repr=False)

    def frame(self):
        # RealSense wait_for_frames must have exactly one caller at a time. The
        # continuous browser feed and detector share this lock and this stream.
        with self._frame_lock:
            if self._camera is None:
                self._camera = _CAMERA_FACTORY(self.camera_kind)
            return self._camera.frame()

    def start_live_feed(self) -> bool:
        """Continuously pump this camera into MJPEG without steering motion."""
        if _VIZ is None or getattr(_VIZ, "_stream", None) is None:
            return False
        if self._live_thread is not None and self._live_thread.is_alive():
            return True
        self._live_stop.clear()

        def pump() -> None:
            while not self._live_stop.is_set():
                try:
                    color, depth, intr = self.frame()
                    _VIZ.show(
                        color, depth, intr, tag="live", save=False,
                        note="continuous view; motion remains preplanned",
                    )
                except Exception:  # noqa: BLE001 - video must never abort motion
                    self._live_stop.wait(0.05)
                # Yield the frame lock promptly when a safety observation needs it.
                self._live_stop.wait(0.01)

        self._live_thread = threading.Thread(
            target=pump, name="wrist-live-feed", daemon=True
        )
        self._live_thread.start()
        return True

    def stop_live_feed(self) -> None:
        self._live_stop.set()
        thread = self._live_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._live_thread = None

    def close(self) -> None:
        self.stop_live_feed()
        with self._frame_lock:
            if self._camera is not None:
                self._camera.close()
                self._camera = None

    def object(self, description: str, color=None, depth=None, intr=None,
               *, near_mm: float | None = None, far_mm: float | None = None):
        """Locate ``description``; returns (Detection | None, why)."""
        if color is None:
            color, depth, intr = self.frame()
        bounds: dict[str, float] = {}
        if near_mm is not None:
            bounds["near_mm"] = float(near_mm)
        if far_mm is not None:
            bounds["far_mm"] = float(far_mm)
        return WD.detect(
            color,
            depth,
            intr,
            description,
            allow_vlm=self.allow_vlm,
            **bounds,
        )


# =====================================================================================
# Motion / gripper helpers
# =====================================================================================
def _goto(x: float, y: float, z: float, **kw: Any):
    """Deliberate absolute move at the demo's known-reachable working orientation.
    (Relative servo steps deliberately do NOT go through here -- they preserve the
    live wrist orientation, as arm_safe intends.)"""
    return A.safe_goto(x, y, z, roll=0.0, pitch=WORK_PITCH, yaw=0.0, **kw)


def _approach_ready_lift(log: Callable[[str], None]):
    """Lift a little, preserving live orientation, to leave the straight-arm singularity."""
    current = tuple(float(v) for v in A.ee_pos())
    target_z = min(current[2] + APPROACH_READY_LIFT_M, A.Z_MAX - 0.01)
    log(f"  [ready lift] z={current[2]:.4f}->{target_z:.4f}m "
        "orientation=preserved")
    # The local RS path uses the same continuous minimum-jerk executor as the
    # main trajectory. This also waits for measured low velocity at the endpoint,
    # avoiding the 10 mm fast-profile undershoot seen with repeated safe_goto
    # segments. Simulation/service retain their existing conservative move.
    if getattr(A, "BACKEND", "") == "local" and callable(
            getattr(A, "safe_move_path", None)):
        rpy = tuple(float(v) for v in A.ee_rpy())
        waypoints, reason = _plan_fixed_approach_trajectory(
            (current[0], current[1], target_z), fixed_rpy=rpy
        )
        if waypoints is None:
            return A.MoveResult(False, f"ready_lift_plan_failed {reason}")
        ok, execute_reason = _execute_fixed_approach_trajectory(
            waypoints, log, fixed_rpy=rpy
        )
        return A.MoveResult(ok, execute_reason,
                            (current[0], current[1], target_z) if ok else None)
    return A.safe_goto(current[0], current[1], target_z, cur=current)


def _highest_reachable_z(x: float, y: float, ceiling: float, floor: float) -> float | None:
    """Highest z at (x, y) the arm can ACTUALLY reach, by asking IK.

    The workspace AABB is a box, but the reachable set inside it is not: with the
    tool pitched down to aim the wrist camera at the table, the top of the box is
    unreachable on this arm. Deriving heights from a fixed offset therefore hands
    the planner poses the gate will refuse. Probing costs a few IK solves once per
    pick and keeps this correct for any arm, pitch or camera mount.
    """
    for z in np.linspace(ceiling, floor, 12):
        if A.check_ik(x, y, float(z), 0.0, WORK_PITCH, 0.0).ok:
            return float(z)
    return None


def _scan_poses() -> list[tuple[float, float, float]]:
    """Poses to look from, centre first, then the four quadrants.

    An eye-in-hand camera cannot see the whole table at once -- its footprint at
    grasp range is roughly the size of the workspace, and the mount offset pushes
    it off-centre -- so "scan" means *look from several places*, unlike the
    overhead servo's single fixed view. Heights come from live IK probing (see
    _highest_reachable_z); REBOT_WRIST_SCAN pins it to one pose instead.
    """
    raw = _parse_triplet(os.getenv("REBOT_WRIST_SCAN"))
    if raw is not None:
        return [raw]
    cx, cy = (A.X_MIN + A.X_MAX) / 2.0, (A.Y_MIN + A.Y_MAX) / 2.0
    ix, iy = 0.30 * (A.X_MAX - A.X_MIN), 0.30 * (A.Y_MAX - A.Y_MIN)
    # Probe down from the ceiling itself. A "Z_MIN + offset" formula silently
    # assumed the floor sat at table height; with a floor below the table it would
    # scan from far too low and see only a patch of the table.
    # Stay materially inside the workspace ceiling. IK/read-back noise at a pose
    # commanded exactly to Z_MAX produced z=0.300000...; subsequent horizontal
    # Jacobian jogs were then correctly refused as out of bounds.
    ceiling = A.Z_MAX - 0.010
    poses = []
    for x, y in ((cx, cy), (cx + ix, cy + iy), (cx + ix, cy - iy),
                 (cx - ix, cy + iy), (cx - ix, cy - iy)):
        z = _highest_reachable_z(x, y, ceiling, A.Z_MIN + 0.02)
        if z is not None:
            poses.append((x, y, z))
    return poses


def _scan_pose() -> tuple[float, float, float]:
    """The primary (centre) look-from pose."""
    poses = _scan_poses()
    if poses:
        return poses[0]
    return ((A.X_MIN + A.X_MAX) / 2.0, (A.Y_MIN + A.Y_MAX) / 2.0, A.Z_MIN + 0.02)


def _find_object(eye: WristEye, description: str, log: Callable[[str], None]):
    """Visit the scan poses until the object is in view. Returns (det, frame, why)."""
    why: dict[str, Any] = {}
    for i, pose in enumerate(_scan_poses()):
        r = _goto(*pose)
        if not r.ok:
            log(f"  [scan {i}] pose {tuple(round(v, 3) for v in pose)} unreachable: {r.reason}")
            continue
        _settle(eye, note=f"moving to scan pose {i}")
        det, frame, why = _observe(eye, description)
        if frame is not None:
            _show(*frame, tag=f"scan{i}", det=det,
                  note=f"scan {i}/{len(_scan_poses())} for '{description}'")
        log(f"  [scan {i}] {tuple(round(v, 3) for v in pose)} -> "
            + (f"found ({det.source}, {det.z_mm:.0f}mm)" if det else "not in view"))
        if det is not None:
            return det, pose, why
    return None, None, why


def _carry_z(x: float | None = None, y: float | None = None) -> float:
    """Height to carry a grasped object at -- again the highest IK actually allows,
    not a fixed offset that the gate would refuse."""
    if x is None or y is None:
        x, y = (A.X_MIN + A.X_MAX) / 2.0, (A.Y_MIN + A.Y_MAX) / 2.0
    z = _highest_reachable_z(x, y, A.Z_MAX, A.Z_MIN + 0.02)
    # Stay just inside the AABB. Interpolating to the exact floating-point upper
    # face can produce Z_MAX+epsilon and turn a valid post-grasp lift into a
    # refusal after the object is already held.
    return min(z, A.Z_MAX - 1e-4) if z is not None else A.Z_MIN + 0.02


# Optional MIT gain override for jaw moves (REBOT_GRIPPER_KP / _KD, or --gripper-kp).
GRIPPER_KP = os.getenv("REBOT_GRIPPER_KP")
GRIPPER_KD = os.getenv("REBOT_GRIPPER_KD")


def _gripper(action: str, target_rad: float | None = None) -> dict[str, Any]:
    """Open/close through the active backend, returning its position read-back.

    ``target_rad`` opens to a computed aperture instead of the fixed cap; the
    backend still hard-clamps it, so it can never exceed the travel limit.
    """
    assert action in ("open", "close")
    readback = getattr(A, "gripper_readback", None)
    if readback is not None:            # simulator
        return readback(action, target_rad, _wrist_limits())
    if getattr(A, "BACKEND", "") == "local":
        try:
            from arm_agent.local_arm import get_local_arm
            backend = get_local_arm()
            # Connect first: the gripper-only entry points (--gripper, the
            # calibrations) never went through preflight, so every one of them
            # failed with "RS arm is not connected". connect() is idempotent.
            backend.connect()
            return backend.gripper(action, target_rad, _wrist_limits(),
                                   kp=float(GRIPPER_KP) if GRIPPER_KP else None,
                                   kd=float(GRIPPER_KD) if GRIPPER_KD else None,
                                   stop_on_contact=False)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"gripper_error {exc}"}
    try:
        import httpx
        r = httpx.post(f"{A.ARM}/gripper", json={"action": action}, timeout=25)
        if r.status_code == 200:
            return r.json()
        return {"ok": False, "reason": f"http_{r.status_code}", "body": r.text[:120]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"gripper_error {exc}"}


def _startup_close_gripper() -> dict[str, Any]:
    """Ramp empty jaws to the calibrated software close limit."""
    close_rad, _ = _wrist_limits()
    return _gripper("close", close_rad)


def _plan_aperture(det: Any) -> dict[str, Any]:
    """Plan pre-open and final grasp positions from measured depth size only.

    No force, motor-current, or encoder-stall contact inference is used. If the
    metric jaw calibration or a plausible object width is unavailable, refuse.
    """
    from arm_agent import gripper_model as gm

    max_ap = gm.max_aperture_mm(_wrist_limits())
    base = {"ok": False, "preopen_rad": None, "grasp_rad": None,
            "min_width_mm": None, "max_width_mm": None,
            "max_aperture_mm": max_ap}
    if max_ap is None:
        return {**base, "reason": "jaw millimetre calibration is missing; refusing to pick"}
    if det is None or det.min_width_mm is None or det.max_width_mm is None:
        return {**base, "reason": "detector did not produce a reliable metric object size; "
                                  "refusing to pick"}
    narrow, wide = float(det.min_width_mm), float(det.max_width_mm or det.min_width_mm)
    base.update(min_width_mm=narrow, max_width_mm=wide)
    if not (math.isfinite(narrow) and math.isfinite(wide)) or narrow <= 0 or wide < narrow:
        return {**base, "reason": f"invalid detected size {narrow:.1f}x{wide:.1f}mm; "
                                  "refusing to pick"}
    if narrow < MIN_SIZED_GRASP_MM:
        return {**base, "reason": f"object is only {narrow:.1f}mm across; below the "
                                  f"conservative {MIN_SIZED_GRASP_MM:.1f}mm minimum"}
    # Position-only closing has no force/contact inference. Keep the requested
    # undersize correction tightly bounded even when an environment value is
    # mistyped; three millimetres is the current tested maximum.
    compression_mm = float(np.clip(SIZE_COMPRESSION_MM, 0.0, 3.0))
    grasp_request_mm = narrow + SIZE_SAFETY_MARGIN_MM - compression_mm
    if grasp_request_mm <= 0:
        return {**base, "reason": f"computed grasp target "
                                  f"{grasp_request_mm:.1f}mm is not positive; "
                                  "refusing to close"}
    if grasp_request_mm > max_ap:
        return {**base, "reason": f"object is {narrow:.1f}mm across and the conservative "
                                  f"{SIZE_SAFETY_MARGIN_MM:.1f}mm margin requires "
                                  f"{grasp_request_mm:.1f}mm, beyond the "
                                  f"{max_ap:.1f}mm jaw limit"}

    # Open as generously as possible for approach, but close only to the measured
    # narrow width. This deliberately commands zero planned compression.
    preopen_mm = min(max_ap, max(narrow + gm.CLEARANCE_MM, wide + gm.CLEARANCE_MM))
    preopen_rad = gm.rad_for_aperture(preopen_mm, _wrist_limits())
    grasp_rad = gm.rad_for_aperture(grasp_request_mm, _wrist_limits())
    if preopen_rad is None or grasp_rad is None:
        return {**base, "reason": "could not convert the detected size to jaw positions"}
    actual_grasp_mm = gm.aperture_mm(grasp_rad)
    actual_preopen_mm = gm.aperture_mm(preopen_rad)
    if actual_grasp_mm is None or abs(actual_grasp_mm - grasp_request_mm) > SIZE_TARGET_TOL_MM:
        return {**base, "reason": f"safe jaw limits cannot represent the "
                                  f"{grasp_request_mm:.1f}mm "
                                  f"grasp target (nearest is {actual_grasp_mm}); refusing"}
    if compression_mm > 0:
        grasp_basis = (
            f"{compression_mm:.1f}mm smaller than the detected width; "
            f"{compression_mm:.1f}mm planned position compression"
        )
    elif SIZE_SAFETY_MARGIN_MM == 0:
        grasp_basis = "exact detected width; zero planned compression"
    else:
        grasp_basis = (
            f"{SIZE_SAFETY_MARGIN_MM:.1f}mm above the detected width; "
            "zero planned compression"
        )
    note = (
        f"detected {narrow:.1f}x{wide:.1f}mm; "
        f"pre-open {actual_preopen_mm:.1f}mm, then position-close to "
        f"{actual_grasp_mm:.1f}mm ({grasp_basis})"
    )
    if wide + gm.CLEARANCE_MM > max_ap:
        note += "; full opening required, so the narrow axis must lie between the jaws"
    return {**base, "ok": True, "reason": note, "preopen_rad": preopen_rad,
            "grasp_rad": grasp_rad, "preopen_mm": actual_preopen_mm,
            "grasp_mm": actual_grasp_mm,
            "planned_compression_mm": compression_mm}


_LAST_LOCAL_STATE_ERROR: str | None = None


def _local_state() -> dict[str, Any] | None:
    """Current backend state, or None if unavailable (used to measure jaw motion)."""
    global _LAST_LOCAL_STATE_ERROR
    _LAST_LOCAL_STATE_ERROR = None
    try:
        if getattr(A, "BACKEND", "") == "local":
            from arm_agent.local_arm import get_local_arm
            backend = get_local_arm()
            backend.connect()
            return backend.state()
    except Exception as exc:  # noqa: BLE001
        _LAST_LOCAL_STATE_ERROR = str(exc)
        return None
    return None


def _home() -> dict[str, Any]:
    """Park the arm at its home pose through whichever backend is active."""
    home = getattr(A, "home", None)
    if callable(home):                      # simulator
        return home()
    if getattr(A, "BACKEND", "") == "local":
        try:
            from arm_agent.local_arm import get_local_arm
            backend = get_local_arm()
            backend.connect()
            result = backend.home()
            if not isinstance(result, dict):
                return {"ok": False, "reason": "local home returned no verification"}
            return result
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"home_error {exc}"}
    try:
        import httpx
        r = httpx.post(f"{A.ARM}/home", json={}, timeout=60)
        return {"ok": r.status_code == 200}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"home_error {exc}"}


def _freeze_arm() -> dict[str, Any]:
    """Latch the measured pose without disabling torque or starting a new path."""
    if getattr(A, "BACKEND", "") == "local":
        try:
            from arm_agent.local_arm import get_local_arm
            backend = get_local_arm()
            if not backend.connected:
                return {"ok": True, "reason": "arm was not connected"}
            return backend.freeze()
        except BaseException as exc:
            return {"ok": False, "reason": f"freeze_error {exc}"}
    # The simulator/service has no local trajectory thread to cancel.
    return {"ok": True, "reason": "no local trajectory to freeze"}


def _home_after_interrupt(*, open_gripper: bool = True) -> dict[str, Any]:
    """Freeze immediately, then open the gripper and home the arm."""
    try:
        frozen = _freeze_arm()
    except BaseException as exc:
        frozen = {"ok": False, "reason": f"interrupt_freeze_error {exc}"}
    if not frozen.get("ok"):
        return {
            "ok": False, "frozen": False, "gripper_opened": False,
            "homed": False, "freeze": frozen,
            "gripper": {"ok": False, "reason": "skipped because freeze was not verified"},
            "home": {"ok": False, "reason": "skipped because freeze was not verified"},
            "reason": "freeze was not verified; refusing a new trajectory or motor disable",
        }
    if open_gripper:
        try:
            opened = _gripper("open")
        except BaseException as exc:
            opened = {"ok": False, "reason": f"interrupt_gripper_open_error {exc}"}
    else:
        opened = {"ok": True, "skipped": True,
                  "reason": "gripper intentionally untouched"}
    if open_gripper and opened.get("ok") and SCENE_COLLISION_ENABLED:
        # Once the jaws are verified open the payload is no longer attached to
        # the arm. Keep static scene/table checks active for the home path.
        A.update_collision_payload(0.0)
    try:
        homed = _home()
    except BaseException as exc:
        homed = {"ok": False, "reason": f"interrupt_home_error {exc}"}
    # A second Ctrl+C can interrupt the first jaw-open ramp while the arm still
    # needs to finish its controlled recovery.  If home succeeded, retry the
    # requested opening at the safe home pose instead of returning with partly
    # closed jaws.  Each attempt is independently guarded so repeated terminal
    # interrupts cannot skip the remainder of recovery.
    if open_gripper and not opened.get("ok") and isinstance(homed, dict) \
            and homed.get("ok"):
        first_reason = str(opened.get("reason", "initial open failed"))
        for retry_index in range(1, 3):
            try:
                opened = _gripper("open")
            except BaseException as exc:
                opened = {
                    "ok": False,
                    "reason": f"interrupt_gripper_open_retry_{retry_index}_error {exc}",
                }
            if opened.get("ok"):
                opened["recovery_retry"] = retry_index
                opened["initial_failure"] = first_reason
                break
    if not isinstance(opened, dict):
        opened = {"ok": False, "reason": "interrupt gripper open returned no result"}
    if not isinstance(homed, dict):
        homed = {"ok": False, "reason": "interrupt home returned no result"}
    return {
        "ok": bool(homed.get("ok")),
        "frozen": bool(frozen.get("ok")),
        "gripper_opened": bool(open_gripper and opened.get("ok")),
        "homed": bool(homed.get("ok")),
        "freeze": frozen,
        "gripper": opened,
        "home": homed,
        "reason": (
            ("pose frozen, gripper opened, and arm homed"
             if open_gripper else "pose frozen, gripper untouched, and arm homed")
            if frozen.get("ok") and opened.get("ok") and homed.get("ok")
            else f"freeze={frozen.get('ok')} gripper_open={opened.get('ok')} "
                 f"home={homed.get('ok')}"
        ),
    }


def _hold_unparked_controller(reason: str) -> None:
    """Keep the local control loop alive when home cannot be verified.

    Exiting here would run teardown while the arm is elevated. This deliberately
    waits for physical intervention instead. SIGINT retries freeze/open/home; a
    hardware E-stop or supported arm is still required for power/CAN failures.
    """
    if getattr(A, "BACKEND", "") != "local":
        return
    from arm_agent.local_arm import get_local_arm
    backend = get_local_arm()
    if not backend.connected:
        return
    print(
        "[SAFETY HOLD] Home was not verified. Motor control remains active and "
        f"the process will not exit. {reason} Use the physical E-stop/power cutoff "
        "only after supporting the arm.",
        file=sys.stderr,
        flush=True,
    )
    while backend.connected:
        try:
            time.sleep(1.0)
        except KeyboardInterrupt:
            retry = _home_after_interrupt()
            if retry.get("homed"):
                return
            print("[SAFETY HOLD] Recovery retry failed; continuing to hold.",
                  file=sys.stderr, flush=True)


def _verify_grasp(close_resp: dict[str, Any]):
    """True if the jaws could NOT reach full close -> something is held between them."""
    pos, tgt = close_resp.get("position_rad"), close_resp.get("target_pos_rad")
    if pos is None or tgt is None:
        return None
    return abs(float(pos) - float(tgt)) > GRASP_GAP_RAD


def _verify_release(open_resp: dict[str, Any]):
    pos, tgt = open_resp.get("position_rad"), open_resp.get("target_pos_rad")
    if pos is None or tgt is None:
        return None
    return abs(float(pos) - float(tgt)) < GRIPPER_NEAR_RAD


def _verify_held_at_tcp(eye: WristEye, object_name: str, tcp_cam: np.ndarray,
                        log: Callable[[str], None]) -> tuple[bool, str, Any | None]:
    """Confirm after lift that the named object is beside the calibrated TCP.

    This is deliberately camera-only. Jaw tracking error is not treated as
    contact or as proof of a grasp.
    """
    last_why: Any = None
    for attempt in range(1, 4):
        color, depth, intr = eye.frame()
        det, last_why = eye.object(
            object_name, color, depth, intr,
            near_mm=TCP_MIN_MM,
            far_mm=TCP_MAX_MM,
        )
        if det is None:
            log(f"stage=visual_hold_check attempt={attempt}/3 detected=false")
            continue
        error_mm = float(np.linalg.norm(
            np.asarray(det.p_cam_mm, dtype=float) - np.asarray(tcp_cam, dtype=float)
        ))
        _show(color, depth, intr, tag=f"held_check_{attempt}", det=det,
              tcp_cam=tcp_cam, err_mm=error_mm,
              note=f"post-lift held check; TCP error {error_mm:.1f}mm")
        log(f"stage=visual_hold_check attempt={attempt}/3 detected=true "
            f"tcp_error_mm={error_mm:.1f} limit_mm={HELD_TCP_TOL_MM:.1f}")
        if error_mm <= HELD_TCP_TOL_MM:
            return True, f"object seen {error_mm:.1f}mm from the calibrated TCP after lift", det
        last_why = f"detected object was {error_mm:.1f}mm from TCP"
    return False, f"object was not visually confirmed at the TCP after lift ({last_why})", None


def _result(ok: bool, reason: str, steps: list[tuple[str, bool, str]],
            verified: bool | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": bool(ok), "reason": reason, "verified": verified,
        "steps": [{"step": s, "ok": bool(o), "reason": r} for (s, o, r) in steps],
    }
    if _VIZ is not None:
        out["frames"] = str(_VIZ.outdir)
    out.update(extra)
    return out


def _printer(verbose: bool) -> Callable[[str], None]:
    if not verbose:
        return lambda m: None

    def emit(message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[wrist {stamp}] {message}", flush=True)

    return emit


# =====================================================================================
# Online Jacobian + servo core (eye-in-hand: watch the OBJECT, not the tip)
# =====================================================================================
def _observe(eye: WristEye, description: str, retries: int = LOST_RETRIES):
    """Detect the object, tolerating a dropped frame or a momentary depth hole."""
    last_why: dict[str, Any] = {}
    for attempt in range(retries):
        color, depth, intr = eye.frame()
        det, why = eye.object(description, color, depth, intr)
        if det is not None:
            return det, (color, depth, intr), why
        last_why = why
        time.sleep(0.1)
    return None, None, last_why


def learn_jacobian(eye: WristEye, description: str, log: Callable[[str], None] = lambda m: None):
    """Learn the 3x3 J mapping a base-frame EE step (m) to the OBJECT's apparent
    shift in the camera frame (mm): jog each base axis, watch the (static) object,
    return to start.

    Eye-in-hand means the object appears to move opposite the arm; measuring rather
    than assuming that is the whole point. Returns (J, why) or (None, why).
    J's z-column is how the object shifts per metre of base lift.
    """
    base = tuple(A.ee_pos())
    camera_shifts = []
    measured_base_steps = []
    lower = (A.X_MIN, A.Y_MIN, A.Z_MIN)
    upper = (A.X_MAX, A.Y_MAX, A.Z_MAX)
    for axis in range(3):
        det0, frame0, why = _observe(eye, description)
        if det0 is None:
            A.safe_goto(*base)
            return None, f"object not visible before jogging axis {axis} ({why.get('used')})"
        before_pos = np.asarray(A.ee_pos(), dtype=float)
        # Try progressively smaller, in-bounds calibration jogs. A locally awkward
        # IK pose should reduce calibration leverage, not abort the whole approach.
        # Every candidate and every restore still goes through bounds + IK.
        step = 0.0
        r = None
        failures: list[str] = []
        for magnitude in (JAC_STEP, 0.015, 0.010):
            for sign in (1.0, -1.0):
                candidate = sign * magnitude
                target_axis = base[axis] + candidate
                if not lower[axis] + 1e-6 <= target_axis <= upper[axis] - 1e-6:
                    continue
                delta = [0.0, 0.0, 0.0]
                delta[axis] = candidate
                r_try = A.safe_goto(*(base[i] + delta[i] for i in range(3)))
                if r_try.ok:
                    step, r = candidate, r_try
                    break
                failures.append(f"{candidate*1000:+.0f}mm: {r_try.reason}")
                restore = A.safe_goto(*base)
                if not restore.ok:
                    return None, (f"axis {axis} jog refused and base restore failed: "
                                  f"{restore.reason}")
            if r is not None and r.ok:
                break
        if r is None or not r.ok:
            restore = A.safe_goto(*base)
            if not restore.ok:
                return None, f"could not restore base after axis {axis}: {restore.reason}"
            return None, f"could not jog axis {axis}: {'; '.join(failures)}"
        _settle(eye, note=f"jacobian: base axis {axis}")
        jog_pos = np.asarray(A.ee_pos(), dtype=float)
        measured_step = jog_pos - before_pos
        measured_step_norm = float(np.linalg.norm(measured_step))
        det1, frame, _ = _observe(eye, description)
        if frame is not None:
            _show(*frame, tag=f"jac_axis{axis}", det=det1,
                  note=f"jacobian: base axis {axis} {step*1000:+.0f}mm")
        restore = A.safe_goto(*base)
        if not restore.ok:
            return None, f"could not restore base after axis {axis}: {restore.reason}"
        _settle(eye)
        if det1 is None:
            return None, f"object lost after jogging axis {axis}"
        tracked = (
            WD.track_detection_flow(frame0, frame, det0, description)
            if frame0 is not None and frame is not None else None
        )
        tracked_shift = (
            np.asarray(tracked.p_cam_mm) - np.asarray(det0.p_cam_mm)
            if tracked is not None else None
        )
        detector_shift = (
            np.asarray(det1.p_cam_mm) - np.asarray(det0.p_cam_mm)
        )
        # Camera motion must be compared with what the arm actually reached,
        # not just the requested jog. The local controller can legitimately
        # settle a few millimetres inside its endpoint tolerance; dividing by
        # the requested 20 mm made those unequal endpoint errors look like an
        # impossible non-rigid camera.
        if measured_step_norm < 0.005:
            return None, (
                f"axis {axis} calibration jog moved only "
                f"{measured_step_norm * 1000.0:.1f}mm according to arm state"
            )
        expected_mm = measured_step_norm * 1000.0
        tracked_plausible = bool(
            tracked_shift is not None
            and 0.35 * expected_mm <= float(np.linalg.norm(tracked_shift))
            <= 2.5 * expected_mm
        )
        detector_plausible = bool(
            0.35 * expected_mm <= float(np.linalg.norm(detector_shift))
            <= 2.5 * expected_mm
        )
        if tracked_plausible:
            shift = tracked_shift
            association = f"flow({tracked.info.get('flow_points', 0)}pts)"
            disagreement = float(np.linalg.norm(detector_shift - tracked_shift))
            if disagreement > 15.0:
                log(f"    [association] detector jump rejected; detector/flow "
                    f"disagreement={disagreement:.0f}mm")
        elif detector_plausible:
            shift = detector_shift
            association = "detector"
        else:
            return None, (
                f"object association changed during axis {axis}: detector shift "
                f"{float(np.linalg.norm(detector_shift)):.0f}mm"
                + (" and optical flow was unavailable" if tracked_shift is None
                   else f", flow shift {float(np.linalg.norm(tracked_shift)):.0f}mm")
                + f" for a {expected_mm:.0f}mm calibration jog"
            )
        log(f"  [jac] axis {axis} {step*1000:+.0f}mm -> object moved "
            f"({shift[0]:+.0f},{shift[1]:+.0f},{shift[2]:+.0f})mm "
            f"association={association} actual_arm_step="
            f"{measured_step_norm*1000.0:.1f}mm")
        camera_shifts.append(np.asarray(shift, dtype=float))
        measured_base_steps.append(measured_step)
    # S = J D, where each column of D is the measured base-frame displacement
    # and each column of S is the corresponding camera-frame object shift.
    # Solving the full matrices also accounts for small off-axis endpoint error.
    shift_matrix = np.asarray(camera_shifts, dtype=float).T
    delta_matrix = np.asarray(measured_base_steps, dtype=float).T
    if float(np.linalg.cond(delta_matrix)) > 20.0:
        return None, "measured arm calibration steps are not independent"
    try:
        J = shift_matrix @ np.linalg.inv(delta_matrix)
    except np.linalg.LinAlgError:
        return None, "measured arm calibration steps are singular"
    weakest = min(float(np.linalg.norm(c)) for c in camera_shifts)
    if weakest < 3.0:
        return None, "a base jog barely moved the object in view (occluded or mis-detected)"
    cond = float(np.linalg.cond(J))
    if cond > 80.0:
        return None, f"Jacobian ill-conditioned (cond={cond:.0f})"
    return J, f"ok (cond={cond:.1f})"


def _max_iters() -> int:
    """Iteration budget sized to the workspace, not a magic number.

    Steps are capped at ~9 mm, so crossing the envelope takes tens of them --
    descending the height alone needs ~33. A fixed 24 silently turned "the object
    is far from the scan pose" into "did not converge". Read from the live gate so
    it tracks whatever envelope is configured.
    """
    span = max(A.X_MAX - A.X_MIN, A.Y_MAX - A.Y_MIN, A.Z_MAX - A.Z_MIN)
    step = max(1e-6, min(A.MAX_XY_STEP, A.MAX_Z_STEP))
    return int(min(90, max(MAX_ITERS, span / step + 15)))


def _clamped_step(raw: np.ndarray) -> tuple[float, float, float]:
    """Scale the whole step to fit the caps, preserving its DIRECTION (clamping each
    axis independently would steer the arm off the commanded line)."""
    scale = 1.0
    for value, cap in zip(raw, (A.MAX_XY_STEP, A.MAX_XY_STEP, A.MAX_Z_STEP)):
        if abs(value) > 1e-12:
            scale = min(scale, cap / abs(value))
    return tuple(float(v * scale) for v in raw)  # type: ignore[return-value]


def _servo_object_to(eye: WristEye, description: str, Jinv: np.ndarray,
                     target_cam: np.ndarray, tcp_cam, tol_mm: float,
                     log: Callable[[str], None], tag: str = "servo",
                     workspace_margin_m: float = 0.0):
    """Step the arm until the OBJECT reaches ``target_cam`` in the camera frame.
    Every step is arm_safe-gated (bounds + IK-verify + step cap)."""
    err = float("inf")
    best, stalled = float("inf"), 0
    budget = _max_iters()
    for it in range(budget):
        det, frame, why = _observe(eye, description)
        if det is None:
            if frame is not None:
                _show(*frame, tag=f"{tag}{it:02d}", tcp_cam=tcp_cam, target_cam=target_cam,
                      note="object lost")
            return False, f"object lost during servo at step {it} ({why.get('used')})"
        e = np.asarray(target_cam, float) - np.asarray(det.p_cam_mm, float)
        err = float(np.linalg.norm(e))
        _show(*frame, tag=f"{tag}{it:02d}", det=det, tcp_cam=tcp_cam,
              target_cam=target_cam, err_mm=err, title=tag)
        if err < tol_mm:
            return True, f"aligned err={err:.0f}mm in {it} steps"
        # Detection noise puts a floor under the achievable error. Grinding away at
        # it until MAX_ITERS would throw away an alignment that is already good
        # enough to grasp, so accept a stalled-but-close one and say so.
        if err < best - 1.0:
            best, stalled = err, 0
        else:
            stalled += 1
            if stalled >= STALL_ITERS:
                if err < 2 * tol_mm:
                    return True, f"aligned err={err:.0f}mm (at the noise floor, {it} steps)"
                return False, f"stalled at err={err:.0f}mm after {it} steps"
        step = _clamped_step(GAIN * (Jinv @ e))
        if workspace_margin_m > 0.0:
            current = np.asarray(A.ee_pos(), dtype=float)
            candidate = current + np.asarray(step, dtype=float)
            margin = float(workspace_margin_m)
            lower = np.array([A.X_MIN + margin, A.Y_MIN + margin, A.Z_MIN + margin])
            upper = np.array([A.X_MAX - margin, A.Y_MAX - margin, A.Z_MAX - margin])
            if np.any(candidate < lower) or np.any(candidate > upper):
                log(f"  [it {it}] stopping before safe workspace margin; "
                    f"candidate={candidate.round(4).tolist()}")
                return True, (
                    f"closest safe point reached; next step would cross the "
                    f"{margin * 1000:.0f}mm workspace margin"
                )
        r = A.safe_step(dx=step[0], dy=step[1], dz=step[2])
        log(f"  [it {it}] err={err:.0f}mm step=({step[0]*1000:+.1f},"
            f"{step[1]*1000:+.1f},{step[2]*1000:+.1f})mm -> {r.reason}")
        if not r.ok:
            return False, f"step refused at {it}: {r.reason}"
        _settle(eye, tcp_cam=tcp_cam, target_cam=target_cam, title=tag)
    return False, f"did not converge (err={err:.0f}mm after {budget} steps)"


def _reachable_approach_target(det: Any, J: np.ndarray, requested_cam: np.ndarray):
    """Choose the requested hover or the closest safely reachable point.

    The learned Jacobian predicts the final base-frame displacement before any
    approach motion. This prevents a distant object from walking the controller
    all the way to a workspace face before discovering the hover is unreachable.
    """
    Jinv = np.linalg.inv(J)
    observed_cam = np.asarray(det.p_cam_mm, dtype=float)
    error_cam = np.asarray(requested_cam, dtype=float) - observed_cam
    delta_base = Jinv @ error_cam
    current = np.asarray(A.ee_pos(), dtype=float)
    if not np.all(np.isfinite(delta_base)):
        return None, {"reason": "non-finite visual displacement"}

    margin = max(0.0, APPROACH_WORKSPACE_MARGIN_M)
    lower = np.array([A.X_MIN + margin, A.Y_MIN + margin, A.Z_MIN + margin])
    upper = np.array([A.X_MAX - margin, A.Y_MAX - margin, A.Z_MAX - margin])
    max_scale = 1.0
    for value, start, lo, hi in zip(delta_base, current, lower, upper):
        if value > 1e-9:
            max_scale = min(max_scale, float((hi - start) / value))
        elif value < -1e-9:
            max_scale = min(max_scale, float((lo - start) / value))
    max_scale = float(np.clip(max_scale, 0.0, 1.0))
    requested_endpoint = current + delta_base
    if max_scale < 0.05:
        return None, {
            "reason": "no meaningful approach motion fits inside the safe workspace margin",
            "requested_endpoint_m": requested_endpoint.round(4).tolist(),
        }

    # The box is only a coarse envelope. Verify the predicted endpoint with IK and
    # back off along the same approach line if the kinematic boundary is tighter.
    rpy = tuple(float(v) for v in A.ee_rpy())
    chosen = None
    endpoint = None
    last_reason = ""
    for scale in np.linspace(max_scale, max(0.05, max_scale * 0.25), 16):
        candidate = current + float(scale) * delta_base
        ik = A.check_ik(*candidate, *rpy)
        if ik.ok:
            chosen, endpoint = float(scale), candidate
            break
        last_reason = ik.reason
    if chosen is None or endpoint is None:
        return None, {
            "reason": f"no safely reachable endpoint along the approach line ({last_reason})",
            "requested_endpoint_m": requested_endpoint.round(4).tolist(),
        }

    effective_cam = observed_cam + J @ (chosen * delta_base)
    full = chosen >= 0.999
    workspace_limited = max_scale < 0.999
    ik_limited = chosen < max_scale - 1e-4
    residual_mm = float(np.linalg.norm(np.asarray(requested_cam) - effective_cam))
    if full:
        reason = "requested target is predicted reachable"
    elif workspace_limited and ik_limited:
        reason = (
            f"requested target crosses the safe workspace and farther candidates "
            f"also fail IK; using the closest IK-reachable point with a "
            f"{margin * 1000:.0f}mm boundary margin"
        )
    elif workspace_limited:
        reason = (
            f"requested target is outside the safe workspace; using the closest "
            f"point with a {margin * 1000:.0f}mm boundary margin"
        )
    else:
        reason = "requested target fails IK; using the closest IK-reachable point"
    return effective_cam, {
        "full_hover": full,
        "path_fraction": round(chosen, 3),
        "predicted_endpoint_m": endpoint.round(4).tolist(),
        "requested_endpoint_m": requested_endpoint.round(4).tolist(),
        "workspace_margin_mm": round(margin * 1000.0, 1),
        "residual_to_requested_target_mm": round(residual_mm, 1),
        "workspace_limited": workspace_limited,
        "ik_limited": ik_limited,
        "reason": reason,
    }


def _rigid_camera_jacobian(measured: np.ndarray):
    """Project a noisy translation Jacobian onto the rigid-camera model.

    With fixed wrist orientation, translating the camera by one metre must move
    a static object's camera coordinates by exactly one metre, through an
    orthogonal axis mapping. Raw finite-difference columns contain depth noise;
    treating that noise as shear causes large open-loop endpoint errors.
    """
    raw = np.asarray(measured, dtype=float)
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        return None, {"reason": "measured Jacobian is not a finite 3x3 matrix"}
    try:
        u, singular_mm_per_m, vt = np.linalg.svd(raw)
    except np.linalg.LinAlgError as exc:
        return None, {"reason": f"Jacobian SVD failed: {exc}"}
    rigid = 1000.0 * (u @ vt)
    relative_fit_error = float(
        np.linalg.norm(raw - rigid) / max(1e-9, np.linalg.norm(rigid))
    )
    detail = {
        "singular_values_mm_per_m": [
            round(float(v), 1) for v in singular_mm_per_m
        ],
        "rigid_fit_relative_error": round(relative_fit_error, 3),
        "determinant_sign": -1 if np.linalg.det(rigid) < 0 else 1,
        "reason": (
            f"rigid camera mapping fit error={relative_fit_error:.3f}; "
            f"singular values="
            f"{[round(float(v), 1) for v in singular_mm_per_m]}mm/m"
        ),
    }
    max_fit_error = float(np.clip(
        float(os.getenv("REBOT_WRIST_RIGID_FIT_MAX_ERROR", "0.15")),
        0.05, 0.20,
    ))
    detail["max_rigid_fit_relative_error"] = max_fit_error
    if relative_fit_error > max_fit_error:
        detail["reason"] = (
            f"translation measurements do not fit a rigid camera "
            f"(relative error={relative_fit_error:.3f}, maximum "
            f"{max_fit_error:.3f})"
        )
        return None, detail
    return rigid, detail


def _plan_fixed_approach_trajectory(
    endpoint_m: Any,
    *,
    cap_fraction: float | None = None,
    start_m: Any | None = None,
    fixed_rpy: tuple[float, float, float] | None = None,
    validate_ik: bool = True,
):
    """Pre-validate a straight Cartesian waypoint trajectory without moving."""
    start = np.asarray(A.ee_pos() if start_m is None else start_m, dtype=float)
    endpoint = np.asarray(endpoint_m, dtype=float)
    delta = endpoint - start
    fraction = float(np.clip(
        APPROACH_TRAJECTORY_CAP_FRACTION
        if cap_fraction is None else cap_fraction,
        0.20, 0.85,
    ))
    caps = np.array([A.MAX_XY_STEP, A.MAX_XY_STEP, A.MAX_Z_STEP]) * fraction
    count = max(1, int(math.ceil(float(np.max(np.abs(delta) / caps)))))
    rpy = (
        tuple(float(v) for v in A.ee_rpy())
        if fixed_rpy is None else tuple(float(v) for v in fixed_rpy)
    )
    margin = max(0.0, APPROACH_WORKSPACE_MARGIN_M)
    lower = np.array([A.X_MIN + margin, A.Y_MIN + margin, A.Z_MIN + margin])
    upper = np.array([A.X_MAX - margin, A.Y_MAX - margin, A.Z_MAX - margin])
    # Reachability serializes endpoints to 0.1 mm before trajectory planning.
    # Decimal boundary values such as 0.465 can then differ from 0.480-0.015 by
    # a few floating-point ulps. This epsilon is sub-micrometre and changes no
    # physical safety margin; it only makes equality at the boundary stable.
    boundary_epsilon_m = 1e-7
    waypoints: list[tuple[float, float, float]] = []
    for index in range(1, count + 1):
        waypoint = start + delta * (index / count)
        if np.any(waypoint < lower - boundary_epsilon_m) or \
                np.any(waypoint > upper + boundary_epsilon_m):
            return None, (
                f"waypoint {index}/{count} crosses the "
                f"{margin * 1000:.0f}mm workspace margin"
            )
        waypoints.append(tuple(float(v) for v in waypoint))
    if not validate_ik:
        return waypoints, (
            f"{len(waypoints)} bounded waypoints generated; "
            f"endpoint={endpoint.round(4).tolist()}"
        )
    ik_ok, ik_reason = _validate_fixed_waypoints(waypoints, rpy)
    if not ik_ok:
        return None, ik_reason
    return waypoints, (
        f"{len(waypoints)} fixed waypoints pre-validated; "
        f"endpoint={endpoint.round(4).tolist()}"
    )


def _validate_fixed_waypoints(
    waypoints: list[tuple[float, float, float]],
    rpy: tuple[float, float, float],
) -> tuple[bool, str]:
    """Validate one complete waypoint chain from the current live joint seed."""
    path_checker = getattr(A, "check_ik_path", None)
    if callable(path_checker):
        path_ik = path_checker(waypoints, *rpy)
        if not path_ik.ok:
            return False, f"complete path failed continuation IK: {path_ik.reason}"
    else:
        for index, waypoint in enumerate(waypoints, 1):
            ik = A.check_ik(*waypoint, *rpy)
            if not ik.ok:
                return False, (
                    f"waypoint {index}/{len(waypoints)} failed IK: {ik.reason}"
                )
    return True, (
        f"complete {len(waypoints)}-waypoint chain passed deterministic "
        "continuation IK"
    )


def _plan_clearance_axial_trajectory(
    endpoint_m: Any,
    camera_mapping: np.ndarray,
    fixed_rpy: tuple[float, float, float],
) -> tuple[list[tuple[float, float, float]] | None, str, dict[str, Any]]:
    """Find a fully prevalidated clearance path followed by an axial approach.

    A Cartesian straight line can cross a numerical singularity even when both
    endpoints are reachable. Try only a small, deterministic set of alternate
    clearance chains. Every candidate retains the ordinary workspace checks,
    fixed wrist orientation, joint continuity cap, and complete-path IK gate.
    No candidate is executed until the entire chain passes.
    """
    start = np.asarray(A.ee_pos(), dtype=float)
    endpoint = np.asarray(endpoint_m, dtype=float)
    nominal_mm = max(1.0, float(GEOMETRY_FINAL_APPROACH_MM))
    minimum_mm = float(np.clip(
        GEOMETRY_MIN_FINAL_APPROACH_MM, 15.0, nominal_mm
    ))
    backoffs_mm: list[float] = []
    for value in (nominal_mm, max(minimum_mm, nominal_mm * 0.75), minimum_mm):
        if not any(abs(value - existing) < 0.1 for existing in backoffs_mm):
            backoffs_mm.append(value)

    failures: list[str] = []
    for backoff_mm in backoffs_mm:
        try:
            base_backoff = np.linalg.solve(
                np.asarray(camera_mapping, dtype=float),
                np.array([0.0, 0.0, backoff_mm], dtype=float),
            )
        except np.linalg.LinAlgError:
            return None, (
                "rigid camera mapping is singular; cannot construct an axial "
                "final approach"
            ), {}
        pregrasp = endpoint + base_backoff
        clearance_z = max(float(start[2]), float(pregrasp[2]))
        bridge_above_start = np.array(
            [start[0], start[1], clearance_z], dtype=float
        )
        bridge_above_pregrasp = np.array(
            [pregrasp[0], pregrasp[1], clearance_z], dtype=float
        )
        routes: list[tuple[str, list[np.ndarray]]] = [
            ("straight_clearance", [pregrasp]),
            ("level_clearance", [bridge_above_start,
                                  bridge_above_pregrasp, pregrasp]),
        ]
        # If the direct level traverse intersects clutter, try bounded lateral
        # corridors. Every corridor is still solved and collision-checked as one
        # complete chain before its first command is sent.
        for axis, label in ((1, "y"), (0, "x")):
            for sign, side in ((1.0, "positive"), (-1.0, "negative")):
                shifted_start = bridge_above_start.copy()
                shifted_end = bridge_above_pregrasp.copy()
                shifted_start[axis] += sign * 0.060
                shifted_end[axis] += sign * 0.060
                routes.append((
                    f"{label}_{side}_corridor",
                    [bridge_above_start, shifted_start, shifted_end,
                     bridge_above_pregrasp, pregrasp],
                ))
        for route_name, route_points in routes:
            combined: list[tuple[float, float, float]] = []
            leg_start = start.copy()
            route_failed = None
            for leg_endpoint in [*route_points, endpoint]:
                leg_endpoint = np.asarray(leg_endpoint, dtype=float)
                if float(np.linalg.norm(leg_endpoint - leg_start)) < 1e-7:
                    continue
                leg, leg_reason = _plan_fixed_approach_trajectory(
                    leg_endpoint,
                    start_m=leg_start,
                    fixed_rpy=fixed_rpy,
                    validate_ik=False,
                )
                if leg is None:
                    route_failed = leg_reason
                    break
                combined.extend(leg)
                leg_start = leg_endpoint
            if route_failed is not None or not combined:
                failures.append(
                    f"{route_name}@{backoff_mm:.0f}mm: "
                    f"{route_failed or 'empty path'}"
                )
                continue
            chain_ok, chain_reason = _validate_fixed_waypoints(
                combined, fixed_rpy
            )
            if chain_ok:
                detail = {
                    "pregrasp_endpoint_m": pregrasp.round(4).tolist(),
                    "final_approach_camera_mm": [0.0, 0.0, backoff_mm],
                    "clearance_strategy": route_name,
                }
                return combined, (
                    f"{route_name} with {backoff_mm:.0f}mm camera-axial final "
                    f"approach; {len(combined)} waypoints; "
                    f"pregrasp={pregrasp.round(4).tolist()}; {chain_reason}"
                ), detail
            failures.append(
                f"{route_name}@{backoff_mm:.0f}mm: {chain_reason}"
            )
    return None, (
        "all bounded clearance/axial candidates failed: " + "; ".join(failures)
    ), {}


def _plan_rpy_trajectory(
    target_rpy: tuple[float, float, float],
    *,
    xyz: Any | None = None,
) -> tuple[list[tuple[float, float, float]] | None, str]:
    """Prevalidate a stationary, finely stepped wrist reorientation."""
    position = np.asarray(A.ee_pos() if xyz is None else xyz, dtype=float)
    start = np.asarray(A.ee_rpy(), dtype=float)
    target = np.asarray(target_rpy, dtype=float)
    delta = target - start
    max_total = math.radians(max(0.0, GEOMETRY_MAX_YAW_DELTA_DEG))
    if float(np.max(np.abs(delta))) > max_total + 1e-9:
        return None, (
            f"orientation change {math.degrees(float(np.max(np.abs(delta)))):.1f}deg "
            f"exceeds the {GEOMETRY_MAX_YAW_DELTA_DEG:.1f}deg clamp"
        )
    step = math.radians(float(np.clip(GEOMETRY_RPY_STEP_DEG, 0.5, 3.0)))
    count = max(1, int(math.ceil(float(np.max(np.abs(delta))) / step)))
    planned: list[tuple[float, float, float]] = []
    for index in range(1, count + 1):
        rpy = start + delta * (index / count)
        ik = A.check_ik(*position, *rpy)
        if not ik.ok:
            return None, (
                f"orientation waypoint {index}/{count} failed IK: {ik.reason}"
            )
        planned.append(tuple(float(v) for v in rpy))
    return planned, (
        f"{len(planned)} orientation waypoints prevalidated at fixed XYZ; "
        f"target_rpy={[round(float(v), 4) for v in target]}"
    )


def _execute_rpy_trajectory(
    rpy_waypoints: list[tuple[float, float, float]],
    log: Callable[[str], None],
) -> tuple[bool, str]:
    """Execute a prevalidated stationary reorientation with no visual steering."""
    for index, (roll, pitch, yaw) in enumerate(rpy_waypoints, 1):
        xyz = tuple(float(v) for v in A.ee_pos())
        result = A.safe_move(
            *xyz, cur=xyz, roll=roll, pitch=pitch, yaw=yaw
        )
        log(
            f"  [orientation {index}/{len(rpy_waypoints)}] "
            f"rpy_deg=({math.degrees(roll):.1f},"
            f"{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}) "
            f"ok={result.ok} detail={result.reason}"
        )
        if not result.ok:
            return False, result.reason
    return True, "bounded orientation trajectory completed"


def _execute_fixed_approach_trajectory(
    waypoints: list[tuple[float, float, float]],
    log: Callable[[str], None],
    fixed_rpy: tuple[float, float, float] | None = None,
):
    """Execute only the pre-planned waypoints; camera detections cannot steer it."""
    if fixed_rpy is None:
        fixed_rpy = tuple(float(v) for v in A.ee_rpy())
    roll, pitch, yaw = fixed_rpy

    # On RS hardware, submit the already-prevalidated Cartesian chain as one
    # continuous 50 Hz minimum-jerk joint path. This matches safe_home's motion
    # profile and avoids decelerating to zero at every tiny waypoint.
    continuous_mover = getattr(A, "safe_move_path", None)
    if getattr(A, "BACKEND", "") == "local" and callable(continuous_mover):
        result = continuous_mover(
            waypoints, roll=roll, pitch=pitch, yaw=yaw
        )
        log(f"  [trajectory continuous 1/{len(waypoints)} -> "
            f"{len(waypoints)}/{len(waypoints)}] {result.reason}")
        if result.ok:
            return True, (
                f"completed {len(waypoints)} fixed waypoints as one continuous "
                "minimum-jerk trajectory"
            )
        return False, f"continuous fixed trajectory refused: {result.reason}"

    def move_segment(target: tuple[float, float, float], depth: int = 0):
        current = tuple(float(v) for v in A.ee_pos())
        # Each attempted segment is gated immediately before it moves. Numerical
        # IK can converge from one live joint seed and plateau from the next even
        # along a prevalidated path; bisect only that fixed segment rather than
        # loosening IK tolerance or changing the endpoint.
        result = A.safe_move(
            *target, cur=current, roll=roll, pitch=pitch, yaw=yaw
        )
        if result.ok:
            return True, result.reason
        can_bisect = (
            depth < max(0, APPROACH_SEGMENT_BISECT_DEPTH)
            and ("ik" in result.reason.lower() or "step_too_large" in result.reason.lower())
        )
        delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
        if not can_bisect or float(np.max(np.abs(delta))) < 0.001:
            return False, result.reason
        midpoint_arr = np.asarray(current, dtype=float) + 0.5 * delta
        midpoint = tuple(float(v) for v in midpoint_arr)
        log(f"    [subdivide depth={depth + 1}] reason={result.reason}; "
            f"midpoint=({midpoint[0]:.4f},{midpoint[1]:.4f},{midpoint[2]:.4f})")
        ok, why = move_segment(midpoint, depth + 1)
        if not ok:
            return False, why
        return move_segment(target, depth + 1)

    for index, waypoint in enumerate(waypoints, 1):
        ok, why = move_segment(waypoint)
        log(f"  [trajectory {index}/{len(waypoints)}] "
            f"target=({waypoint[0]:.4f},{waypoint[1]:.4f},{waypoint[2]:.4f}) "
            f"-> {why}")
        if not ok:
            return False, f"fixed waypoint {index}/{len(waypoints)} refused: {why}"
    return True, f"completed {len(waypoints)} fixed waypoints"


def _plan_endpoint_alignment_correction(
    observed_cam_mm: Any,
    expected_cam_mm: Any,
    rigid_jacobian: np.ndarray,
    fixed_rpy: tuple[float, float, float],
):
    """Plan one small, fixed, IK-validated correction with the jaws open.

    The learned rigid Jacobian maps base-frame tool motion in metres to apparent
    object motion in camera millimetres.  Corrections are limited to 12 mm per
    attempt regardless of environment configuration and are never executed until
    their entire waypoint chain passes the normal workspace and continuation-IK
    checks.
    """
    observed = np.asarray(observed_cam_mm, dtype=float)
    expected = np.asarray(expected_cam_mm, dtype=float)
    error_cam = expected - observed
    if not np.all(np.isfinite(error_cam)):
        return None, {"reason": "non-finite endpoint camera error"}
    try:
        raw_delta = np.linalg.solve(np.asarray(rigid_jacobian, dtype=float), error_cam)
    except np.linalg.LinAlgError:
        return None, {"reason": "rigid camera mapping is singular"}
    if not np.all(np.isfinite(raw_delta)):
        return None, {"reason": "non-finite endpoint correction"}

    gain = float(np.clip(
        float(os.getenv("REBOT_WRIST_FINAL_ALIGN_GAIN", "0.85")), 0.50, 0.90
    ))
    requested_cap = float(np.clip(
        FIXED_PICK_MAX_CORRECTION_M, 0.003, 0.012
    ))
    delta = gain * raw_delta
    norm = float(np.linalg.norm(delta))
    if norm > requested_cap:
        delta *= requested_cap / norm
        norm = requested_cap
    if norm < 0.0005:
        return None, {"reason": "predicted endpoint correction is below 0.5mm"}

    start = np.asarray(A.ee_pos(), dtype=float)
    endpoint = start + delta
    waypoints, reason = _plan_fixed_approach_trajectory(
        endpoint, start_m=start, fixed_rpy=fixed_rpy
    )
    detail = {
        "reason": reason,
        "camera_error_mm": error_cam.round(2).tolist(),
        "base_correction_mm": (delta * 1000.0).round(2).tolist(),
        "correction_norm_mm": round(norm * 1000.0, 2),
        "endpoint_m": endpoint.round(4).tolist(),
    }
    return waypoints, detail


def _endpoint_retry_envelope_mm(*, object_visible: bool) -> float:
    """Maximum initial error eligible for bounded, verified correction.

    This is not a grasp acceptance tolerance. It only decides whether the open
    jaws may enter the correction loop; closure still requires the much tighter
    dynamic jaw-slot gate after every correction has been re-observed.
    """
    if not DYNAMIC_GRASP_ENABLED or not object_visible:
        return float(FIXED_PICK_TARGET_TOL_MM)
    attempts = int(np.clip(FIXED_PICK_MAX_CORRECTIONS, 0, 3))
    per_attempt_mm = 1000.0 * float(np.clip(
        FIXED_PICK_MAX_CORRECTION_M, 0.003, 0.012
    ))
    return max(
        float(FIXED_PICK_TARGET_TOL_MM),
        attempts * per_attempt_mm,
    )


def _plan_scene_grasp_geometry(
    frame: tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]],
    det: WD.Detection,
    *,
    current_rpy: tuple[float, float, float],
    allow_orientation: bool = True,
    object_name: str | None = None,
) -> GG.GraspGeometry:
    """Turn a target feature plus full RGB-D scene into one safe grasp proposal."""
    from arm_agent import gripper_model as gm

    max_aperture = gm.max_aperture_mm(_wrist_limits())
    if max_aperture is None:
        return GG.GraspGeometry(
            False, "jaw millimetre calibration is missing", None, None, None
        )
    axis_raw = os.getenv("REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG")
    axis: float | None = None
    if allow_orientation:
        if axis_raw:
            try:
                axis = float(axis_raw)
            except ValueError:
                axis = None
        else:
            saved = load_grasp_target_detail().get(
                "jaw_long_axis_camera_deg"
            )
            try:
                axis = None if saved is None else float(saved)
            except (TypeError, ValueError):
                axis = None
    color, depth, intr = frame
    semantic_envelope = None
    if object_name and os.getenv("REBOT_ENABLE_LOCATE_ANYTHING", "0").strip().lower() \
            in {"1", "true", "yes", "on"}:
        semantic_box = WD.detect_vlm(
            color, depth, intr, f"the complete {object_name}",
            near_mm=WD.NEAR_MM, far_mm=WD.FAR_MM,
        )
        if semantic_box is not None:
            semantic_envelope = WD.segment_semantic_envelope(
                color, depth, intr, semantic_box, det,
                near_mm=WD.NEAR_MM, far_mm=WD.FAR_MM,
            )
    return GG.plan_grasp_geometry(
        color, depth, intr, det,
        current_rpy=current_rpy,
        max_aperture_mm=float(max_aperture),
        jaw_long_axis_camera_deg=axis,
        max_yaw_delta_deg=GEOMETRY_MAX_YAW_DELTA_DEG,
        yaw_measurement_slop_deg=GEOMETRY_YAW_MEASUREMENT_SLOP_DEG,
        final_approach_mm=GEOMETRY_FINAL_APPROACH_MM,
        min_feature_prominence_mm=GEOMETRY_MIN_FEATURE_PROMINENCE_MM,
        require_axis_for_elongated=allow_orientation,
        semantic_envelope=semantic_envelope,
    )


def _feature_target_for_grasp_target(
    requested_grasp_cam: Any,
    feature_det: WD.Detection,
    grasp_det: WD.Detection,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected visual-feature location when the chosen grasp region is aligned.

    A color cap may identify an object while the depth-envelope centre is the
    safer place to grasp it. Their camera-frame offset is rigid before contact;
    preserve that offset so the endpoint veto compares like with like.
    """
    feature_to_grasp = (
        np.asarray(feature_det.p_cam_mm, dtype=float)
        - np.asarray(grasp_det.p_cam_mm, dtype=float)
    )
    expected_feature = (
        np.asarray(requested_grasp_cam, dtype=float) + feature_to_grasp
    )
    return expected_feature, feature_to_grasp


def _apply_pick_target_y_offset(target_cam: Any) -> tuple[np.ndarray, float]:
    """Apply a bounded image-vertical trim without changing calibration.

    RealSense camera +Y points down in the image. Requesting the stationary
    object slightly farther down in the image moves the camera/jaws upward
    relative to it. Keep this as a run-time trim so TCP and the camera-only jaw
    slot calibration remain immutable and independently inspectable.
    """
    offset_mm = float(np.clip(FIXED_PICK_TARGET_Y_OFFSET_MM, -15.0, 15.0))
    adjusted = np.asarray(target_cam, dtype=float).copy()
    adjusted[1] += offset_mm
    return adjusted, offset_mm


def _install_scene_collision_map(
    frame: tuple[np.ndarray, np.ndarray, Any],
    geometry: GG.GraspGeometry,
    camera_mapping: np.ndarray,
    *,
    payload_radius_m: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Map visible non-target depth and install it in the central motion gate."""
    if not SCENE_COLLISION_ENABLED:
        return True, {
            "collision_model": False,
            "reason": "disabled; set REBOT_SCENE_COLLISION=1 for guarded runs",
        }
    target = load_grasp_target()
    safety = load_safety_calibration()
    table_z = safety.get("table_z_m")
    if target is None:
        return False, {"reason": "open-jaw camera target is unavailable"}
    if table_z is None:
        return False, {"reason": "calibrated table height is unavailable"}
    exclusions: list[tuple[int, int, int, int]] = []
    # The target is the sole allowed environmental contact. Prefer the complete
    # semantic/depth envelope over the small colored identifier or grasp slice.
    for detection in (geometry.envelope_detection, geometry.grasp_detection):
        if detection is not None:
            exclusions.append(tuple(int(v) for v in detection.box))
            break
    try:
        _, depth, intr = frame
        scene = SC.build_scene_map(
            depth, intr,
            ee_xyz_m=A.ee_pos(),
            camera_mapping_mm_per_m=np.asarray(camera_mapping, dtype=float),
            jaw_target_cam_mm=target,
            table_z_m=float(table_z),
            workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
            workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
            exclude_pixel_boxes=exclusions,
            near_mm=max(100.0, float(WD.NEAR_MM)),
            far_mm=float(WD.FAR_MM),
            stride_px=SCENE_COLLISION_STRIDE_PX,
            voxel_m=SCENE_COLLISION_VOXEL_M,
            inflation_m=SCENE_COLLISION_INFLATION_M,
        )
    except Exception as exc:  # noqa: BLE001
        return False, {
            "reason": f"RGB-D scene mapping failed: {type(exc).__name__}: {exc}"
        }
    if not scene.boxes:
        return False, {
            **scene.as_dict(),
            "reason": "no above-table obstacle geometry was recovered; refusing "
                      "to interpret an empty map as a clear scene",
        }
    configured = A.configure_collision_scene(
        [box.as_dict() for box in scene.boxes],
        table_z_m=float(table_z), payload_radius_m=payload_radius_m,
    )
    detail = {
        **scene.as_dict(),
        "reason": configured.reason,
        "target_exclusion_boxes_px": [list(box) for box in exclusions],
        "payload_radius_mm": round(payload_radius_m * 1000.0, 1),
    }
    return configured.ok, detail


# =====================================================================================
# SAFE APPROACH: current-view detection -> IK alignment -> home (no descent/gripper)
# =====================================================================================
def servo_approach(object_name: str, *, dry: bool = False, verbose: bool = False,
                   eye: WristEye | None = None) -> dict[str, Any]:
    """Hover ``APPROACH_M`` above an object, then return directly home.

    Every commanded pose passes through the shared bounds/IK/step gate. This mode
    never commands the gripper, never descends, and never visits a scan pose.
    """
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    log(f"stage=approach_start object={object_name!r} "
        f"mode={'dry' if dry else 'live'} gripper=untouched descent=disabled")

    tcp = load_tcp()
    steps.append(("tcp_calibration", tcp is not None,
                  f"{tuple(round(v, 1) for v in tcp)}mm" if tcp else "missing"))
    if tcp is None:
        return _result(False, "wrist TCP not calibrated", steps, homed=False)
    tcp_cam = np.asarray(tcp, dtype=float)

    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        return _result(False, "arm not ready", steps, homed=False)

    own_eye = eye is None
    eye = eye or WristEye()
    start_feed = getattr(eye, "start_live_feed", None)
    continuous_feed = bool(not dry and callable(start_feed) and start_feed())
    if continuous_feed:
        steps.append((
            "continuous_live_feed", True,
            "camera frames stream throughout motion; detections cannot alter "
            "the preplanned approach, but may trigger a bounded, separately "
            "IK-validated endpoint retry while the jaws remain open",
        ))
        log("stage=continuous_live_feed started=true "
            "approach_steering=disabled endpoint_retry=enabled")
    try:
        if dry:
            color, depth, intr = eye.frame()
            det, why = eye.object(object_name, color, depth, intr)
            _show(color, depth, intr, tag="approach_dry", det=det,
                  tcp_cam=tcp_cam, note="dry approach: current view only")
            steps.append(("detect_from_current_pose", det is not None,
                          f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
            return _result(
                det is not None,
                "object visible; live approach will use IK-verified moves"
                if det else "object not visible from the current pose",
                steps, homed=False,
                detection={"source": det.source, "uv": det.uv, "z_mm": det.z_mm}
                if det else None,
            )

        current = tuple(float(v) for v in A.ee_pos())
        log(f"stage=approach_ready_lift starting xyz="
            f"({current[0]:.4f},{current[1]:.4f},{current[2]:.4f}) "
            f"lift_mm={APPROACH_READY_LIFT_M * 1000:.0f} orientation=preserved")
        ready = _approach_ready_lift(log)
        steps.append(("ready_lift", ready.ok, ready.reason))
        if not ready.ok:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, f"could not make the ready lift: {ready.reason}",
                           steps, homed=bool(recovery.get("homed")),
                           recovery=recovery, gripper_commanded=False)
        _settle(eye, note="ready lift reached")

        det, frame, why = _observe(eye, object_name)
        if frame is not None:
            _show(*frame, tag="approach_detect", det=det, tcp_cam=tcp_cam,
                  note="approach detection from current pose")
        steps.append(("detect_from_current_pose", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
        if det is None:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, f"'{object_name}' not visible from the current pose", steps,
                           homed=bool(recovery.get("homed")), recovery=recovery,
                           gripper_commanded=False)

        log("stage=approach_jacobian starting bounded_ik_jogs=3")
        J, why_j = learn_jacobian(eye, object_name, log)
        steps.append(("learn_jacobian", J is not None, why_j))
        if J is None:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, f"visual alignment unavailable: {why_j}", steps,
                           homed=bool(recovery.get("homed")), recovery=recovery,
                           gripper_commanded=False)

        # Freeze one visual goal now. X/Y use the accepted gripper target; optical
        # depth stops short of it. Subsequent camera frames are observational only
        # and can never steer this trajectory.
        requested_cam = tcp_cam.copy()
        requested_cam[2] = max(
            float(tcp_cam[2]) + APPROACH_NEAR_TOUCH_GAP_MM,
            float(WD.NEAR_MM) + 15.0,
        )
        log(f"stage=fixed_target_plan tcp_depth_mm={tcp_cam[2]:.1f} "
            f"non_contact_clearance_mm={APPROACH_NEAR_TOUCH_GAP_MM:.1f} "
            f"object_target_depth_mm={requested_cam[2]:.1f}")
        plan_det, _, plan_why = _observe(eye, object_name)
        if plan_det is None:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, f"object lost before reachability planning ({plan_why})",
                           steps, homed=bool(recovery.get("homed")),
                           recovery=recovery, gripper_commanded=False)
        target_cam, reachability = _reachable_approach_target(plan_det, J, requested_cam)
        steps.append(("reachable_target_plan", target_cam is not None,
                      str(reachability.get("reason"))))
        if target_cam is None:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, str(reachability.get("reason")), steps,
                           homed=bool(recovery.get("homed")), recovery=recovery,
                           gripper_commanded=False, reachability=reachability)
        log(f"stage=fixed_endpoint full_target={reachability['full_hover']} "
            f"path_fraction={reachability['path_fraction']:.3f} "
            f"endpoint={reachability['predicted_endpoint_m']} "
            f"detail={reachability['reason']}")

        waypoints, trajectory_reason = _plan_fixed_approach_trajectory(
            reachability["predicted_endpoint_m"]
        )
        steps.append(("prevalidate_fixed_trajectory", waypoints is not None,
                      trajectory_reason))
        if waypoints is None:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, f"fixed trajectory is unsafe: {trajectory_reason}",
                           steps, homed=bool(recovery.get("homed")),
                           recovery=recovery, gripper_commanded=False,
                           reachability=reachability)
        log(f"stage=fixed_trajectory_start {trajectory_reason} "
            "camera_steering=disabled")
        ok, trajectory_result = _execute_fixed_approach_trajectory(waypoints, log)
        steps.append(("execute_fixed_trajectory", ok, trajectory_result))
        if not ok:
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, f"could not complete fixed trajectory: {trajectory_result}",
                           steps,
                           homed=bool(recovery.get("homed")), recovery=recovery,
                           gripper_commanded=False)

        endpoint_det, _, _ = _observe(eye, object_name)
        endpoint_observation = None
        if endpoint_det is not None:
            endpoint_observation = {
                "source": endpoint_det.source,
                "p_cam_mm": [round(float(v), 1) for v in endpoint_det.p_cam_mm],
                "target_cam_mm": [round(float(v), 1) for v in target_cam],
                "error_mm": round(float(np.linalg.norm(
                    np.asarray(endpoint_det.p_cam_mm) - target_cam
                )), 1),
            }
            log(f"stage=endpoint_observation camera_only=true "
                f"error_mm={endpoint_observation['error_mm']:.1f}")
        else:
            log("stage=endpoint_observation camera_only=true detected=false")

        frozen = _freeze_arm()
        steps.append(("freeze_at_endpoint", bool(frozen.get("ok")), str(frozen.get("reason"))))
        if not frozen.get("ok"):
            recovery = _home_after_interrupt(open_gripper=False)
            return _result(False, "could not latch the fixed endpoint", steps,
                           homed=bool(recovery.get("homed")), recovery=recovery,
                           gripper_commanded=False)

        full_target = bool(reachability["full_hover"])
        log(f"stage={'fixed_endpoint_reached' if full_target else 'closest_safe_point_reached'} "
            "gripper=untouched descent=disabled next=home")
        recovery = _home_after_interrupt(open_gripper=False)
        if SCENE_COLLISION_ENABLED and recovery.get("homed"):
            A.clear_collision_scene()
        steps.append(("return_home", bool(recovery.get("homed")),
                      str(recovery.get("reason"))))
        return _result(
            bool(recovery.get("homed")),
            (
                "fixed near-touch trajectory completed and arm returned home"
                if full_target else
                "approached as close as workspace and IK allowed and returned home"
            ) if recovery.get("homed")
            else "fixed endpoint reached but home was not verified",
            steps, homed=bool(recovery.get("homed")),
            gripper_commanded=False, recovery=recovery,
            full_hover=full_target, full_target=full_target,
            reachability=reachability,
            endpoint_observation=endpoint_observation,
        )
    except BaseException as exc:
        if dry:
            return _result(False, f"dry approach exception: {type(exc).__name__}: {exc}",
                           steps, homed=False, gripper_commanded=False)
        recovery = _home_after_interrupt(open_gripper=False)
        return _result(False, f"approach exception: {type(exc).__name__}: {exc}",
                       steps, homed=bool(recovery.get("homed")),
                       gripper_commanded=False, recovery=recovery)
    finally:
        stop_feed = getattr(eye, "stop_live_feed", None)
        if callable(stop_feed):
            stop_feed()
        if own_eye:
            eye.close()


# =====================================================================================
# FIXED PICK + PLACE: proven fixed approach, sized jaws, high release, home
# =====================================================================================
def fixed_pick_and_place(object_name: str, *, dry: bool = False,
                         verbose: bool = False, eye: WristEye | None = None,
                         place_after_pick: bool = True,
                         place_reference: str | None = None,
                         place_relation: str = "drop_zone",
                         place_distance_m: float | None = None,
                         ) -> dict[str, Any]:
    """Fixed-coordinate pick and high release with no contact inference.

    The camera is used to detect/size once, learn the local translation Jacobian,
    and observe whether the object moved rigidly with the wrist after lift. It
    never steers the prevalidated approach, lift, or drop trajectories.
    """
    global _OBJECT_HELD, _FIXED_HELD_CONTEXT
    _OBJECT_HELD = False
    _FIXED_HELD_CONTEXT = None
    if SCENE_COLLISION_ENABLED:
        # A previous interrupted controller process must never leak stale scene
        # geometry into a new observation.
        A.clear_collision_scene()
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    preplanned_scene_endpoint: np.ndarray | None = None
    preplanned_scene_detail: dict[str, Any] | None = None
    collision_detail: dict[str, Any] = {
        "collision_model": False,
        "reason": "scene collision checking disabled",
    }
    task_name = "fixed_pick_place" if place_after_pick else "fixed_pick"
    normalized_place_relation = _normalize_scene_relation(place_relation)
    if place_after_pick and normalized_place_relation is None:
        return _result(
            False, f"unsupported placement relation: {place_relation!r}",
            [("placement_description", False,
              "use left/right/front/behind/next_to/free_space/drop_zone")],
            homed=False, object_held=False,
        )
    scene_place = bool(
        place_after_pick and normalized_place_relation in _SCENE_RELATIONS
    )
    if scene_place and not place_reference:
        return _result(
            False, f"{normalized_place_relation} placement needs a reference object",
            [("placement_description", False, "reference is missing")],
            homed=False, object_held=False,
        )
    log(f"stage={task_name}_start object={object_name!r} "
        f"mode={'dry' if dry else 'live'} contact_inference=false "
        f"place_relation={normalized_place_relation!r} "
        f"place_reference={place_reference!r}")

    if DYNAMIC_GRASP_ENABLED:
        from arm_agent import gripper_model as _dynamic_gm
        gripper_detail = _dynamic_gm.describe()
        measured_points = [
            point for point in gripper_detail.get("aperture_points", [])
            if float(point.get("angle_deg", 0.0)) >= _dynamic_gm.HARD_MIN_DEG
        ]
        locate_enabled = os.getenv(
            "REBOT_ENABLE_LOCATE_ANYTHING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not locate_enabled:
            return _result(
                False,
                "dynamic grasp requires the GPU whole-object detector; "
                "set REBOT_ENABLE_LOCATE_ANYTHING=1",
                [("dynamic_semantic_envelope", False,
                  "GPU semantic detection is not enabled")],
                homed=False, object_held=False,
            )
        has_curve = gripper_detail.get("aperture_model") == "piecewise_linear" \
            and len(measured_points) >= 3
        has_single_span = bool(
            gripper_detail.get("calibrated_mm")
            and float(gripper_detail.get("max_aperture_mm", 0.0)) > 1.0
        )
        if not (has_curve or has_single_span):
            return _result(
                False,
                "dynamic grasp requires either the known 150deg jaw span or a "
                "multi-point jaw curve; no motion was attempted",
                [("dynamic_jaw_curve", False,
                  "no usable jaw-aperture calibration")],
                homed=False, object_held=False,
            )
        steps.append((
            "dynamic_jaw_model", True,
            (f"piecewise curve with {len(measured_points)} measured points"
             if has_curve else
             "single 150deg span accepted as an uncertain initial estimate; "
             "visual micro-lift controls bounded 5deg retries"),
        ))
        dynamic_has_jaw_curve = has_curve
    else:
        dynamic_has_jaw_curve = False

    tcp = load_tcp()
    uses_taught_drop = bool(
        place_after_pick and normalized_place_relation in {"drop_zone", "free_space"}
    )
    drop = load_drop_zone() if uses_taught_drop else None
    steps.append(("tcp_calibration", tcp is not None,
                  f"{tuple(round(v, 1) for v in tcp)}mm" if tcp else "missing"))
    if uses_taught_drop:
        steps.append(("drop_zone", drop is not None,
                      f"{tuple(round(v, 3) for v in drop)}m" if drop else "missing"))
    if tcp is None or (uses_taught_drop and drop is None):
        requirement = ("TCP and taught drop zone are required"
                       if uses_taught_drop else "TCP is required")
        return _result(False, requirement, steps,
                       homed=False, object_held=False)
    tcp_cam = np.asarray(tcp, dtype=float)
    target_override_raw = os.getenv("REBOT_WRIST_PICK_TARGET_CAM_MM")
    if target_override_raw and _parse_triplet(target_override_raw) is None:
        steps.append(("grasp_visual_target", False,
                      "REBOT_WRIST_PICK_TARGET_CAM_MM must be x,y,z in camera mm"))
        return _result(False, "invalid fixed grasp visual target", steps,
                       homed=False, object_held=False)
    # A persisted target belongs to the physical camera/gripper mount. Never
    # leak it into the simulator; simulation uses its own synthetic TCP unless
    # a target is explicitly supplied through the environment.
    target_override = (
        _parse_triplet(target_override_raw)
        if target_override_raw else
        (load_grasp_target() if getattr(A, "BACKEND", "") == "local" else None)
    )
    fixed_visual_target = (
        np.asarray(target_override, dtype=float)
        if target_override is not None else None
    )
    if fixed_visual_target is None and getattr(A, "BACKEND", "") == "local":
        steps.append((
            "grasp_visual_target",
            False,
            "missing camera-only open-jaw calibration; run "
            "--calib-grasp-target OBJECT first",
        ))
        return _result(False, "open-jaw grasp target is not calibrated", steps,
                       homed=False, object_held=False)
    steps.append((
        "grasp_visual_target",
        True,
        (f"calibrated open-jaw camera target "
         f"{fixed_visual_target.round(1).tolist()}mm")
        if fixed_visual_target is not None else
        f"TCP-derived target with {FIXED_PICK_FINAL_GAP_MM:.1f}mm clearance",
    ))

    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        return _result(False, "arm not ready", steps, homed=False,
                       object_held=False)

    own_eye = eye is None
    eye = eye or WristEye()
    start_feed = getattr(eye, "start_live_feed", None)
    continuous_feed = bool(not dry and callable(start_feed) and start_feed())
    if continuous_feed:
        steps.append((
            "continuous_live_feed", True,
            "camera frames stream throughout motion; detections cannot steer "
            "the preplanned trajectory",
        ))
        log("stage=continuous_live_feed started=true camera_steering=disabled")

    def recover(reason: str, **extra: Any) -> dict[str, Any]:
        global _OBJECT_HELD, _FIXED_HELD_CONTEXT
        recovery = _home_after_interrupt(open_gripper=True)
        if SCENE_COLLISION_ENABLED and recovery.get("homed"):
            A.clear_collision_scene()
        _OBJECT_HELD = False if recovery.get("gripper_opened") else None
        if _OBJECT_HELD is False:
            _FIXED_HELD_CONTEXT = None
        return _result(
            False, reason, steps,
            homed=bool(recovery.get("homed")),
            gripper_opened=bool(recovery.get("gripper_opened")),
            object_held=_OBJECT_HELD, recovery=recovery, **extra,
        )

    try:
        if dry:
            color, depth, intr = eye.frame()
            det, why = eye.object(object_name, color, depth, intr)
            geometry = (
                _plan_scene_grasp_geometry(
                    (color, depth, intr), det,
                    # Dry mode must remain camera/model-only. The live workflow
                    # reads measured RPY after connecting; the dry preview uses
                    # the verified home orientation and never touches hardware.
                    current_rpy=(0.0, 0.0, 0.0),
                    object_name=object_name,
                )
                if det is not None else None
            )
            grasp_det = (
                geometry.grasp_detection
                if geometry is not None and geometry.ok else None
            )
            plan = _plan_aperture(grasp_det) if grasp_det is not None else None
            steps.append(("detect_and_size", det is not None,
                          f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
            if geometry is not None:
                steps.append(("geometry_grasp_candidates", geometry.ok,
                              geometry.reason))
            if plan is not None:
                steps.append(("jaw_plan", bool(plan["ok"]), str(plan["reason"])))
            drop_ik = None
            release_z = None
            reference_det = None
            reference_why: Any = None
            if scene_place:
                assert place_reference is not None
                reference_det, reference_why = eye.object(
                    place_reference, color, depth, intr
                )
                steps.append((
                    "placement_reference_preview",
                    reference_det is not None,
                    f"{reference_det.source} at {reference_det.z_mm:.0f}mm"
                    if reference_det is not None else str(reference_why),
                ))
            if uses_taught_drop:
                assert drop is not None
                release_z = min(
                    A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
                    max(FIXED_DROP_RELEASE_Z_M, float(drop[2]) + 0.10),
                )
                drop_ik = A.check_ik(float(drop[0]), float(drop[1]), release_z,
                                     0.0, 0.0, 0.0)
                steps.append(("high_drop_ik", drop_ik.ok, drop_ik.reason))
            if SCENE_COLLISION_ENABLED:
                steps.append((
                    "scene_collision_preview", True,
                    "enabled for live execution; base-frame mapping and complete "
                    "URDF path checks require the measured live arm pose and are "
                    "therefore deferred without moving in dry mode",
                ))
            ok = bool(det is not None and geometry and geometry.ok
                      and plan and plan["ok"]
                      and (not scene_place or reference_det is not None)
                      and (drop_ik is None or drop_ik.ok))
            return _result(
                ok,
                f"dry {task_name.replace('_', ' ')} checks passed" if ok else
                f"dry {task_name.replace('_', ' ')} checks failed",
                steps, homed=False, object_held=False,
                jaw_plan=plan if plan and plan["ok"] else None,
                grasp_geometry=geometry.summary() if geometry else None,
                high_drop_xyz_m=(
                    [float(drop[0]), float(drop[1]), release_z]
                    if uses_taught_drop and drop is not None else None
                ),
                placement_plan_deferred_until_carry=scene_place,
                placement_reference=(
                    {"object": place_reference,
                     "source": reference_det.source,
                     "z_mm": round(float(reference_det.z_mm), 1)}
                    if reference_det is not None else None
                ),
                collision_scene={
                    "collision_model": bool(SCENE_COLLISION_ENABLED),
                    "deferred_until_live_pose": bool(SCENE_COLLISION_ENABLED),
                },
            )

        if START_GRIPPER_CLOSED:
            close_limit_rad, _ = _wrist_limits()
            log("stage=startup_gripper_close "
                f"target_deg={math.degrees(close_limit_rad):.1f}")
            startup_close = _startup_close_gripper()
            startup_close_detail = str(
                startup_close.get("reason")
                or startup_close.get("skipped")
                or f"position {startup_close.get('position_deg')}deg"
            )
            steps.append(("startup_gripper_close",
                          bool(startup_close.get("ok")),
                          startup_close_detail))
            if not startup_close.get("ok"):
                return recover(
                    "could not place the empty gripper at its calibrated "
                    f"close limit: {startup_close_detail}"
                )

        current = tuple(float(v) for v in A.ee_pos())
        log(f"stage=ready_lift starting xyz={tuple(round(v, 4) for v in current)}")
        ready = _approach_ready_lift(log)
        steps.append(("ready_lift", ready.ok, ready.reason))
        if not ready.ok:
            return recover(f"could not make the ready lift: {ready.reason}")
        _settle(eye, note="fixed pick ready lift")

        det, frame, why = _observe(eye, object_name)
        if frame is not None:
            _show(*frame, tag="fixed_pick_detect", det=det, tcp_cam=tcp_cam,
                  note="fixed pick detection and sizing")
        steps.append(("detect_from_ready_pose", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
        if det is None:
            return recover(f"'{object_name}' not visible from the ready pose")
        geometry = _plan_scene_grasp_geometry(
            frame, det,
            current_rpy=tuple(float(v) for v in A.ee_rpy()),
            object_name=object_name,
        )
        steps.append(("geometry_grasp_candidates", geometry.ok,
                      geometry.reason))
        log(f"stage=geometry_grasp ok={geometry.ok} detail={geometry.reason}")
        geometry_attempt_reasons = [geometry.reason]
        # Semantic masks can occasionally collapse to a coloured cap or a short
        # depth fragment for one frame. Retry in place before treating a large
        # apparent yaw as physical truth. No arm or gripper command occurs here.
        if not geometry.ok:
            for retry_index in range(1, 4):
                retry_det, retry_frame, retry_why = _observe(eye, object_name)
                if retry_det is None or retry_frame is None:
                    retry_reason = f"target not detected ({retry_why})"
                    geometry_attempt_reasons.append(retry_reason)
                    steps.append((f"geometry_retry_{retry_index}", False,
                                  retry_reason))
                    log(f"stage=geometry_retry attempt={retry_index}/3 "
                        f"ok=false detail={retry_reason}")
                    continue
                retry_geometry = _plan_scene_grasp_geometry(
                    retry_frame, retry_det,
                    current_rpy=tuple(float(v) for v in A.ee_rpy()),
                    object_name=object_name,
                )
                geometry_attempt_reasons.append(retry_geometry.reason)
                steps.append((f"geometry_retry_{retry_index}",
                              retry_geometry.ok, retry_geometry.reason))
                log(f"stage=geometry_retry attempt={retry_index}/3 "
                    f"ok={retry_geometry.ok} detail={retry_geometry.reason}")
                if retry_geometry.ok:
                    det, frame, geometry = retry_det, retry_frame, retry_geometry
                    break
        if not geometry.ok or geometry.grasp_detection is None \
                or geometry.rpy_rad is None:
            return recover(
                "no safe geometry-derived grasp after bounded ready-pose "
                f"re-detection: {' | '.join(geometry_attempt_reasons)}",
                grasp_geometry=geometry.summary(),
                geometry_attempts=geometry_attempt_reasons,
            )
        grasp_geometry_summary = geometry.summary()
        approach_rpy = tuple(float(v) for v in geometry.rpy_rad)
        live_rpy = np.asarray(A.ee_rpy(), dtype=float)
        rpy_delta = float(np.max(np.abs(
            np.asarray(approach_rpy, dtype=float) - live_rpy
        )))
        if rpy_delta > math.radians(0.1):
            rpy_waypoints, rpy_reason = _plan_rpy_trajectory(approach_rpy)
            steps.append(("prevalidate_bounded_orientation",
                          rpy_waypoints is not None, rpy_reason))
            if rpy_waypoints is None:
                return recover(
                    f"bounded wrist orientation is unsafe: {rpy_reason}",
                    grasp_geometry=grasp_geometry_summary,
                )
            log(f"stage=bounded_orientation_start {rpy_reason}")
            rpy_ok, rpy_exec_reason = _execute_rpy_trajectory(
                rpy_waypoints, log
            )
            steps.append(("execute_bounded_orientation",
                          rpy_ok, rpy_exec_reason))
            if not rpy_ok:
                return recover(
                    f"bounded wrist orientation failed: {rpy_exec_reason}",
                    grasp_geometry=grasp_geometry_summary,
                )
            _settle(eye, note="bounded geometry orientation")
            det_after, frame_after, why_after = _observe(eye, object_name)
            steps.append((
                "redetect_after_orientation",
                det_after is not None,
                f"{det_after.source} at {det_after.z_mm:.0f}mm"
                if det_after else str(why_after),
            ))
            if det_after is None or frame_after is None:
                return recover(
                    "target was lost after the bounded wrist orientation",
                    grasp_geometry=grasp_geometry_summary,
                )
            geometry_after = _plan_scene_grasp_geometry(
                frame_after, det_after, current_rpy=approach_rpy,
                allow_orientation=False,
                object_name=object_name,
            )
            steps.append(("geometry_after_orientation", geometry_after.ok,
                          geometry_after.reason))
            if not geometry_after.ok or \
                    geometry_after.grasp_detection is None:
                return recover(
                    f"grasp geometry became unsafe after orientation: "
                    f"{geometry_after.reason}",
                    grasp_geometry=geometry_after.summary(),
                )
            det = geometry_after.grasp_detection
            frame = frame_after
            geometry = geometry_after
            grasp_geometry_summary = geometry_after.summary()
        else:
            det = geometry.grasp_detection
        jaw_plan = _plan_aperture(det)
        steps.append(("depth_size_and_jaw_limits", bool(jaw_plan["ok"]),
                      str(jaw_plan["reason"])))
        log(f"stage=jaw_plan ok={jaw_plan['ok']} detail={jaw_plan['reason']}")
        if not jaw_plan["ok"]:
            return recover(str(jaw_plan["reason"]))

        use_saved_rotation = os.getenv(
            "REBOT_WRIST_USE_SAVED_CAMERA_ROTATION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if use_saved_rotation:
            J_rigid, rigid_detail = load_saved_camera_jacobian(
                tuple(float(v) for v in A.ee_rpy())
            )
            steps.append(("saved_camera_geometry", J_rigid is not None,
                          str(rigid_detail.get("reason"))))
            if J_rigid is None:
                return recover(
                    "saved rigid camera geometry is unavailable or does not "
                    "match the current wrist orientation; refusing to move",
                    rigid_camera_fit=rigid_detail,
                )
            J = J_rigid
            log(f"stage=saved_camera_geometry detail={rigid_detail['reason']}")
        else:
            log("stage=jacobian starting bounded_ik_jogs=3")
            J, why_j = learn_jacobian(eye, object_name, log)
            steps.append(("learn_jacobian", J is not None, why_j))
            if J is None:
                return recover(f"visual geometry unavailable: {why_j}")
            J_rigid, rigid_detail = _rigid_camera_jacobian(J)
            steps.append(("fit_rigid_camera_geometry", J_rigid is not None,
                          str(rigid_detail.get("reason"))))
            if J_rigid is None:
                return recover(
                    "measured camera geometry is not rigid enough for a completely "
                    "preplanned approach; refusing to move",
                    rigid_camera_fit=rigid_detail,
                )
            J = J_rigid
            log(f"stage=rigid_camera_fit detail={rigid_detail['reason']}")

        if SCENE_COLLISION_ENABLED:
            map_det, map_frame, map_why = _observe(eye, object_name)
            if map_det is None or map_frame is None:
                steps.append(("scene_collision_map", False, str(map_why)))
                return recover(
                    "target could not be isolated in the collision-map frame; "
                    "refusing to treat it as generic free space"
                )
            map_geometry = _plan_scene_grasp_geometry(
                map_frame, map_det,
                current_rpy=tuple(float(v) for v in A.ee_rpy()),
                allow_orientation=False, object_name=object_name,
            )
            if not map_geometry.ok:
                steps.append(("scene_collision_map", False,
                              map_geometry.reason))
                return recover(
                    "the grasp target could not be separated from scene "
                    f"obstacles: {map_geometry.reason}"
                )
            collision_ok, collision_detail = _install_scene_collision_map(
                map_frame, map_geometry, J
            )
            steps.append((
                "scene_collision_map", collision_ok,
                str(collision_detail.get("reason")),
            ))
            log("stage=scene_collision_map "
                f"ok={collision_ok} "
                f"obstacle_columns={collision_detail.get('obstacle_columns')} "
                f"inflation_mm={collision_detail.get('inflation_mm')} "
                f"detail={collision_detail.get('reason')}")
            if not collision_ok:
                return recover(
                    "collision scene could not be validated; no trajectory was "
                    f"executed: {collision_detail.get('reason')}",
                    collision_scene=collision_detail,
                )

        if scene_place:
            assert place_reference is not None
            provisional_context = {
                "grasp_geometry": grasp_geometry_summary,
                "jaw_plan": {
                    "object_width_mm": round(float(jaw_plan["min_width_mm"]), 1),
                    "grasp_mm": round(float(jaw_plan["grasp_mm"]), 1),
                },
            }
            preplanned_scene_endpoint, _, preplanned_scene_detail = \
                _plan_scene_relative_release(
                    place_reference, str(normalized_place_relation),
                    place_distance_m, provisional_context, eye, log,
                    camera_mapping_override=J,
                    camera_mapping_detail=rigid_detail,
                )
            steps.append((
                "preplan_scene_relative_release",
                preplanned_scene_endpoint is not None,
                str((preplanned_scene_detail or {}).get("reason") or
                    (preplanned_scene_detail or {}).get("warning")),
            ))
            if preplanned_scene_endpoint is None:
                return recover(
                    "placement reference could not be frozen from the wide "
                    f"ready-pose view: "
                    f"{(preplanned_scene_detail or {}).get('reason')}",
                    placement=preplanned_scene_detail,
                )
            if preplanned_scene_detail is not None and SCENE_COLLISION_ENABLED:
                preplanned_scene_detail["scene_map"] = collision_detail
            log("stage=scene_place_preplanned_at_ready_pose "
                f"reference={place_reference!r} "
                f"endpoint={preplanned_scene_endpoint.round(4).tolist()} "
                "reference_motion_after_snapshot=not_tracked")

        if fixed_visual_target is not None:
            # This is the object-in-camera observation from a physically
            # accepted near-marker pose. It is safer than extrapolating from an
            # inaccurate jaw/TCP calibration and lets the fixed trajectory
            # reproduce the already-tested relationship.
            requested_cam, target_y_offset_mm = \
                _apply_pick_target_y_offset(fixed_visual_target)
            closer_mm = float(np.clip(FIXED_PICK_CLOSER_MM, 0.0, 10.0))
            requested_cam[2] -= closer_mm
            minimum_target_depth = float(WD.NEAR_MM) + 2.0
            if requested_cam[2] < minimum_target_depth:
                return recover(
                    f"requested {closer_mm:.1f}mm depth advance would put the "
                    f"grasp target at {requested_cam[2]:.1f}mm, inside the "
                    f"{minimum_target_depth:.1f}mm camera safety floor"
                )
            steps.append((
                "grasp_depth_offset",
                True,
                f"{closer_mm:.1f}mm closer; target depth "
                f"{requested_cam[2]:.1f}mm",
            ))
            steps.append((
                "grasp_image_vertical_offset",
                True,
                f"camera Y {target_y_offset_mm:+.1f}mm; target Y "
                f"{requested_cam[1]:.1f}mm (positive raises jaws relative "
                "to the object)",
            ))
            log("stage=fixed_grasp_target source=calibrated_open_jaw_slot "
                f"closer_mm={closer_mm:.1f} "
                f"target_y_offset_mm={target_y_offset_mm:+.1f} "
                f"target_mm={requested_cam.round(1).tolist()}")
        else:
            requested_cam = tcp_cam.copy()
            requested_cam[2] = max(
                float(tcp_cam[2]) + FIXED_PICK_FINAL_GAP_MM,
                float(WD.NEAR_MM) + 10.0,
            )
            log("stage=fixed_grasp_target source=tcp_plus_clearance "
                f"target_mm={requested_cam.round(1).tolist()}")
        approach_rpy = tuple(float(v) for v in A.ee_rpy())
        plan_det, plan_frame, plan_why = _observe(eye, object_name)
        if plan_frame is not None:
            _show(*plan_frame, tag="fixed_pick_preplanned",
                  det=plan_det, tcp_cam=tcp_cam, target_cam=requested_cam,
                  note="single observation used for complete fixed trajectory")
        steps.append(("preplan_object_observation", plan_det is not None,
                      f"{plan_det.source} at {plan_det.z_mm:.0f}mm"
                      if plan_det else str(plan_why)))
        if plan_det is None:
            return recover(
                f"object lost before complete trajectory planning ({plan_why})"
            )
        assert plan_frame is not None
        plan_geometry = _plan_scene_grasp_geometry(
            plan_frame, plan_det, current_rpy=approach_rpy,
            allow_orientation=False,
            object_name=object_name,
        )
        steps.append(("preplan_full_object_geometry", plan_geometry.ok,
                      plan_geometry.reason))
        if not plan_geometry.ok or plan_geometry.grasp_detection is None:
            return recover(
                f"full-object geometry is unsafe before motion: "
                f"{plan_geometry.reason}",
                grasp_geometry=plan_geometry.summary(),
            )
        plan_feature_det = plan_det
        plan_det = plan_geometry.grasp_detection
        expected_feature_cam, feature_to_grasp_cam = \
            _feature_target_for_grasp_target(
                requested_cam, plan_feature_det, plan_det
            )
        steps.append((
            "feature_to_grasp_geometry",
            True,
            f"feature offset from grasp region "
            f"{feature_to_grasp_cam.round(1).tolist()}mm; expected endpoint "
            f"feature={expected_feature_cam.round(1).tolist()}mm",
        ))
        grasp_geometry_summary = plan_geometry.summary()
        planned_cam, reachability = _reachable_approach_target(
            plan_det, J, requested_cam
        )
        # The endpoint-only check above uses the live joints as its seed. Before
        # accepting its partial fallback, try the exact requested endpoint with
        # deterministic continuation IK across the complete small-waypoint path.
        # A continuous path can reach poses that a one-shot solve from home cannot.
        requested_endpoint_raw = reachability.get("requested_endpoint_m")
        exact_waypoints = None
        exact_reason = "requested endpoint unavailable"
        if requested_endpoint_raw is not None:
            exact_waypoints, exact_reason = _plan_fixed_approach_trajectory(
                requested_endpoint_raw, fixed_rpy=approach_rpy
            )
            if exact_waypoints is None:
                fine_waypoints, fine_reason = _plan_fixed_approach_trajectory(
                    requested_endpoint_raw, cap_fraction=0.25,
                    fixed_rpy=approach_rpy,
                )
                if fine_waypoints is not None:
                    exact_waypoints = fine_waypoints
                    exact_reason = (
                        f"{fine_reason}; fine deterministic continuation used "
                        "after the nominal path was refused"
                    )
                else:
                    exact_reason = (
                        f"nominal continuation: {exact_reason}; "
                        f"fine continuation: {fine_reason}"
                    )
        if exact_waypoints is not None:
            endpoint = np.asarray(requested_endpoint_raw, dtype=float)
            waypoints = exact_waypoints
            trajectory_reason = exact_reason
            planned_residual_mm = 0.0
            planned_target_qualified = True
            reachability = {
                **reachability,
                "full_hover": True,
                "path_fraction": 1.0,
                "predicted_endpoint_m": endpoint.round(4).tolist(),
                "residual_to_requested_target_mm": 0.0,
                "endpoint_ik_mode": "deterministic_continuation_path",
                "reason": "exact target reachable through deterministic "
                          "continuation IK",
            }
        else:
            planned_residual_mm = float(
                reachability.get("residual_to_requested_target_mm", math.inf)
            )
            correction_envelope_mm = _endpoint_retry_envelope_mm(
                object_visible=True
            )
            planned_target_qualified = bool(
                planned_cam is not None
                and (
                    reachability.get("full_hover")
                    or planned_residual_mm <= correction_envelope_mm
                )
            )
            endpoint = np.asarray(
                reachability.get("predicted_endpoint_m", [math.nan] * 3),
                dtype=float,
            )
            waypoints, trajectory_reason = (
                _plan_fixed_approach_trajectory(
                    endpoint, fixed_rpy=approach_rpy
                )
                if planned_target_qualified else
                (None, exact_reason)
            )
        if planned_target_qualified and np.all(np.isfinite(endpoint)):
            waypoints, trajectory_reason, clearance_detail = \
                _plan_clearance_axial_trajectory(endpoint, J, approach_rpy)
            reachability.update(clearance_detail)
        plan_detail = (
            f"{reachability.get('reason')}; predicted residual "
            f"{planned_residual_mm:.1f}mm, bounded endpoint-correction "
            f"envelope {_endpoint_retry_envelope_mm(object_visible=True):.1f}mm"
            + ("" if exact_waypoints is not None else
               f"; exact path refusal: {exact_reason}")
        )
        steps.append(("preplan_reachable_grasp",
                      planned_target_qualified, plan_detail))
        if not planned_target_qualified:
            return recover(
                "no IK endpoint reaches the calibrated jaw slot within "
                f"the {_endpoint_retry_envelope_mm(object_visible=True):.1f}mm "
                "bounded correction envelope; refusing to execute",
                reachability=reachability,
                rigid_camera_fit=rigid_detail,
                exact_path_reason=exact_reason,
            )
        steps.append(("prevalidate_complete_approach",
                      waypoints is not None, trajectory_reason))
        if waypoints is None:
            return recover(
                f"complete fixed approach is unsafe: {trajectory_reason}",
                reachability=reachability,
                rigid_camera_fit=rigid_detail,
            )
        open_index = 0
        for index, waypoint in enumerate(waypoints):
            if float(np.linalg.norm(endpoint - np.asarray(waypoint))) <= \
                    FIXED_PICK_OPEN_BEFORE_END_M:
                open_index = index
                break
        else:
            open_index = max(0, len(waypoints) - 1)
        prefix, final_leg = waypoints[:open_index], waypoints[open_index:]
        final_start = (
            np.asarray(prefix[-1], dtype=float)
            if prefix else np.asarray(A.ee_pos(), dtype=float)
        )
        final_distance = float(np.linalg.norm(endpoint - final_start))
        final_residual_mm = planned_residual_mm
        log(f"stage=complete_trajectory_preplanned total_waypoints="
            f"{len(waypoints)} before_open={len(prefix)} "
            f"after_open={len(final_leg)} final_leg_mm="
            f"{final_distance * 1000:.1f} camera_replanning=disabled")

        if prefix:
            log(f"stage=execute_preplanned_before_open waypoints={len(prefix)}")
            prefix_ok, prefix_reason = _execute_fixed_approach_trajectory(
                prefix, log, fixed_rpy=approach_rpy
            )
            steps.append(("execute_preplanned_before_open",
                          prefix_ok, prefix_reason))
            if not prefix_ok:
                return recover(
                    f"preplanned pre-open trajectory failed: {prefix_reason}"
                )

        preopen_rad = float(jaw_plan["preopen_rad"])
        log(f"stage=gripper_preopen target_mm={jaw_plan['preopen_mm']:.1f} "
            f"target_deg={math.degrees(preopen_rad):.1f}")
        opened = _gripper("open", preopen_rad)
        steps.append(("position_open", bool(opened.get("ok")),
                      str(opened.get("reason", "commanded"))))
        if not opened.get("ok"):
            return recover(f"gripper pre-open failed: {opened.get('reason')}")

        log(f"stage=execute_preplanned_after_open "
            f"distance_mm={final_distance * 1000:.1f} "
            f"waypoints={len(final_leg)} camera_steering=disabled")
        ok, reason = _execute_fixed_approach_trajectory(
            final_leg, log, fixed_rpy=approach_rpy
        )
        steps.append(("final_fixed_approach", ok, reason))
        if not ok:
            return recover(f"final fixed approach failed: {reason}")

        grasp_det, grasp_frame, grasp_why = _observe(eye, object_name)
        if grasp_frame is not None:
            _show(*grasp_frame, tag="grasp_endpoint_before_close",
                  det=grasp_det, target_cam=expected_feature_cam,
                  note="endpoint gate before any jaw close")
        steps.append(("object_visible_before_close", grasp_det is not None,
                      f"{grasp_det.source} at {grasp_det.z_mm:.0f}mm"
                      if grasp_det else str(grasp_why)))
        if grasp_det is None:
            # A correctly calibrated grasp feature can be hidden by the open
            # fingers at the endpoint. The complete route and open timing were
            # fixed and IK-validated before motion, so loss of visibility here
            # cannot alter the endpoint.
            steps.append((
                "calibrated_slot_occlusion",
                True,
                f"complete {len(waypoints)}-waypoint approach was prevalidated "
                f"from a visible object observation",
            ))
            log("stage=grasp_target_gate endpoint_occluded=true "
                f"predicted_residual_mm={final_residual_mm:.1f} "
                "basis=preplanned_calibrated_open_jaw_slot")
            grasp_error = final_residual_mm
            before_lift_cam = expected_feature_cam.copy()
        else:
            grasp_error = float(np.linalg.norm(
                np.asarray(grasp_det.p_cam_mm, dtype=float)
                - expected_feature_cam
            ))

        # The long approach remains fixed and completely prevalidated. At its
        # endpoint a visible object may request a separately bounded correction;
        # this avoids throwing away an otherwise successful approach without
        # turning camera detections into unconstrained trajectory steering.
        corrections: list[dict[str, Any]] = []
        retry_envelope_mm = _endpoint_retry_envelope_mm(
            object_visible=grasp_det is not None
        )
        retry_eligible = grasp_error <= retry_envelope_mm
        steps.append(("grasp_target_gate",
                      retry_eligible,
                      f"error {grasp_error:.1f}mm; bounded correction envelope "
                      f"{retry_envelope_mm:.1f}mm; compared the "
                      "identifier feature against its planned offset from the "
                      "grasp region"))
        if not retry_eligible:
            return recover(
                f"endpoint observation is {grasp_error:.1f}mm from the fixed "
                f"grasp target; outside the {retry_envelope_mm:.1f}mm bounded "
                "retry envelope",
                grasp_error_mm=round(grasp_error, 1),
                corrections=corrections,
            )

        # If the object is visible and close to the slot, reuse the already
        # measured rigid camera transform for at most three small corrections.
        # Each attempt is planned from the newly measured pose, capped at 12 mm,
        # and fully workspace/IK checked before the open gripper moves.
        if DYNAMIC_GRASP_ENABLED and grasp_det is not None:
            slot_tol = float(np.clip(DYNAMIC_SLOT_TOL_MM, 3.0, 8.0))
            max_corrections = int(np.clip(FIXED_PICK_MAX_CORRECTIONS, 0, 3))
            previous_error = grasp_error
            for attempt in range(1, max_corrections + 1):
                if grasp_error <= slot_tol:
                    break
                correction_waypoints, correction_detail = \
                    _plan_endpoint_alignment_correction(
                        grasp_det.p_cam_mm, expected_feature_cam, J,
                        approach_rpy,
                    )
                planned = correction_waypoints is not None
                steps.append((
                    f"prevalidate_final_alignment_{attempt}", planned,
                    str(correction_detail.get("reason")),
                ))
                log("stage=final_alignment_plan "
                    f"attempt={attempt}/{max_corrections} ok={planned} "
                    f"error_mm={grasp_error:.1f} "
                    f"camera_error_mm={correction_detail.get('camera_error_mm')} "
                    f"base_correction_mm="
                    f"{correction_detail.get('base_correction_mm')}")
                if correction_waypoints is None:
                    return recover(
                        "final jaw-slot correction could not be safely planned: "
                        f"{correction_detail.get('reason')}",
                        corrections=corrections,
                    )
                moved, move_reason = _execute_fixed_approach_trajectory(
                    correction_waypoints, log, fixed_rpy=approach_rpy
                )
                correction_detail["attempt"] = attempt
                correction_detail["executed"] = moved
                correction_detail["execution_reason"] = move_reason
                corrections.append(correction_detail)
                steps.append((f"execute_final_alignment_{attempt}", moved,
                              move_reason))
                if not moved:
                    return recover(
                        f"final jaw-slot correction failed: {move_reason}",
                        corrections=corrections,
                    )
                _settle(eye, target_cam=expected_feature_cam,
                        note=f"final jaw-slot alignment attempt {attempt}")
                corrected_det, corrected_frame, corrected_why = _observe(
                    eye, object_name
                )
                if corrected_det is None or corrected_frame is None:
                    return recover(
                        "object was lost after the bounded final correction "
                        f"({corrected_why})", corrections=corrections,
                    )
                corrected_error = float(np.linalg.norm(
                    np.asarray(corrected_det.p_cam_mm, dtype=float)
                    - expected_feature_cam
                ))
                correction_detail["observed_error_after_mm"] = round(
                    corrected_error, 2
                )
                _show(*corrected_frame,
                      tag=f"grasp_endpoint_corrected_{attempt}",
                      det=corrected_det, target_cam=expected_feature_cam,
                      note=f"bounded final alignment {attempt}")
                improved = corrected_error < previous_error - 0.5
                verification_ok = improved or corrected_error <= slot_tol
                steps.append((
                    f"verify_final_alignment_{attempt}", verification_ok,
                    f"error {previous_error:.1f}->{corrected_error:.1f}mm; "
                    f"jaw-slot limit {slot_tol:.1f}mm",
                ))
                log("stage=final_alignment_verify "
                    f"attempt={attempt}/{max_corrections} "
                    f"error_before_mm={previous_error:.1f} "
                    f"error_after_mm={corrected_error:.1f} "
                    f"limit_mm={slot_tol:.1f}")
                grasp_det, grasp_frame = corrected_det, corrected_frame
                grasp_error = corrected_error
                if not improved and grasp_error > slot_tol:
                    return recover(
                        "bounded final correction did not improve jaw-slot "
                        f"alignment ({previous_error:.1f}->{grasp_error:.1f}mm)",
                        corrections=corrections,
                    )
                previous_error = grasp_error

        if grasp_det is not None:
            before_lift_cam = np.asarray(grasp_det.p_cam_mm, dtype=float)
            jaw_slot_feature_cam = before_lift_cam.copy()
        else:
            jaw_slot_feature_cam = expected_feature_cam.copy()

        # A wide general endpoint tolerance is useful for reachability, but it
        # is not enough to prove that a narrow object lies between the fingers.
        # In dynamic mode require the observed feature to be much closer to its
        # calibrated open-jaw location before commanding any closure.  If the
        # fingers occlude the feature, the exact prevalidated path remains the
        # only available gate and no camera claim is fabricated.
        if DYNAMIC_GRASP_ENABLED and grasp_det is not None:
            slot_tol = float(np.clip(DYNAMIC_SLOT_TOL_MM, 3.0, 8.0))
            slot_ok = grasp_error <= slot_tol
            steps.append((
                "dynamic_object_inside_jaw_slot", slot_ok,
                f"3D feature error {grasp_error:.1f}mm; limit {slot_tol:.1f}mm",
            ))
            log(f"stage=dynamic_jaw_slot_gate ok={slot_ok} "
                f"error_mm={grasp_error:.1f} limit_mm={slot_tol:.1f}")
            if not slot_ok:
                return recover(
                    f"object is {grasp_error:.1f}mm from the calibrated jaw "
                    f"slot; refusing to close or tighten"
                )

        # The ready-view width only proves that the object can fit.  The final
        # close target must come from a new RGB-D measurement made after the
        # open jaws have reached their validated endpoint.  If the fingers hide
        # the object or the two measurements disagree, do not guess and do not
        # move the gripper.
        if DYNAMIC_GRASP_ENABLED:
            if grasp_det is None or grasp_frame is None:
                return recover(
                    "object is not measurable between the open jaws at the "
                    "endpoint; refusing image-controlled closure"
                )
            endpoint_geometry = _plan_scene_grasp_geometry(
                grasp_frame, grasp_det, current_rpy=approach_rpy,
                allow_orientation=False, object_name=object_name,
            )
            endpoint_plan = (
                _plan_aperture(endpoint_geometry.grasp_detection)
                if endpoint_geometry.ok
                and endpoint_geometry.grasp_detection is not None else None
            )
            initial_width = float(jaw_plan["min_width_mm"])
            endpoint_width = (
                float(endpoint_plan["min_width_mm"])
                if endpoint_plan is not None and endpoint_plan.get("ok") else None
            )
            width_delta = (
                abs(endpoint_width - initial_width)
                if endpoint_width is not None else float("inf")
            )
            endpoint_measure_ok = bool(
                endpoint_plan is not None and endpoint_plan.get("ok")
                and width_delta <= DYNAMIC_REMEASURE_MAX_DELTA_MM
            )
            steps.append((
                "dynamic_endpoint_width", endpoint_measure_ok,
                (f"ready-view {initial_width:.1f}mm, between-jaws "
                 f"{endpoint_width:.1f}mm, delta {width_delta:.1f}mm; "
                 f"close target {endpoint_plan['grasp_mm']:.1f}mm"
                 if endpoint_width is not None and endpoint_plan is not None
                 else endpoint_geometry.reason),
            ))
            log("stage=dynamic_endpoint_width "
                f"ok={endpoint_measure_ok} ready_width_mm={initial_width:.1f} "
                f"endpoint_width_mm={endpoint_width} delta_mm={width_delta:.1f}")
            if not endpoint_measure_ok:
                return recover(
                    "between-jaws width could not be confirmed against the "
                    f"initial feasibility check within "
                    f"{DYNAMIC_REMEASURE_MAX_DELTA_MM:.1f}mm; refusing to close"
                )
            jaw_plan = endpoint_plan
            # Hold verification must follow the physical body section between
            # the jaws, not the colored identifier used to align the approach.
            # A cap can swing several millimetres when a held marker settles
            # without indicating that the grasp was lost.
            before_lift_cam = np.asarray(
                endpoint_geometry.grasp_detection.p_cam_mm, dtype=float
            )

        grasp_rad = float(jaw_plan["grasp_rad"])
        log(f"stage=gripper_position_close target_mm={jaw_plan['grasp_mm']:.1f} "
            f"target_deg={math.degrees(grasp_rad):.1f} contact_inference=false")
        if DYNAMIC_GRASP_ENABLED:
            # Approach the image-derived angle in small monotonic position
            # increments.  Each backend command is itself ramped; splitting the
            # travel prevents a single fast 150deg->target snap.  The sequence
            # can never cross the final image-derived target or the hard bounds.
            start_rad = float(opened.get("position_rad", preopen_rad))
            travel = abs(start_rad - grasp_rad)
            max_step = math.radians(float(np.clip(
                DYNAMIC_CLOSE_STEP_DEG, 5.0, 20.0
            )))
            close_steps = max(1, int(math.ceil(travel / max_step)))
            closed = {"ok": False, "reason": "no close step executed"}
            for close_index in range(1, close_steps + 1):
                intermediate = start_rad + \
                    (grasp_rad - start_rad) * close_index / close_steps
                log(f"stage=image_controlled_close "
                    f"step={close_index}/{close_steps} "
                    f"target_deg={math.degrees(intermediate):.1f}")
                closed = _gripper("close", intermediate)
                if not closed.get("ok"):
                    break
            steps.append((
                "image_controlled_slow_close", bool(closed.get("ok")),
                f"{close_steps} bounded position ramps to endpoint-measured "
                f"{jaw_plan['grasp_mm']:.1f}mm; no contact inference",
            ))
        else:
            closed = _gripper("close", grasp_rad)
            steps.append(("position_close", bool(closed.get("ok")),
                          f"{jaw_plan['grasp_mm']:.1f}mm commanded; "
                          "no contact inference"))
        if not closed.get("ok"):
            return recover(f"gripper position-close failed: {closed.get('reason')}")
        log("stage=gripper_position_close_readback "
            f"target_deg={math.degrees(grasp_rad):.1f} "
            f"actual_deg={closed.get('position_deg', 'unavailable')} "
            f"delta_deg={closed.get('delta_deg', 'unavailable')}")
        _OBJECT_HELD = None
        if SCENE_COLLISION_ENABLED:
            provisional_payload = {
                "grasp_geometry": grasp_geometry_summary,
                "jaw_plan": {
                    "object_width_mm": float(jaw_plan["min_width_mm"]),
                },
            }
            payload_radius = SC.payload_radius_m(provisional_payload)
            payload_result = A.update_collision_payload(payload_radius)
            steps.append(("attach_payload_collision_envelope",
                          payload_result.ok, payload_result.reason))
            log("stage=attach_payload_collision_envelope "
                f"ok={payload_result.ok} radius_mm={payload_radius * 1000:.1f} "
                f"detail={payload_result.reason}")
            if not payload_result.ok:
                return recover(
                    "held-object collision envelope could not be installed; "
                    "refusing to lift"
                )
        time.sleep(0.5)

        grasp_xyz = np.asarray(A.ee_pos(), dtype=float)
        retry_count = 0
        current_grasp_mm = float(jaw_plan["grasp_mm"])
        current_grasp_rad = float(grasp_rad)
        while True:
            lift_distance = (
                float(np.clip(DYNAMIC_MICRO_LIFT_M, 0.010, 0.020))
                if DYNAMIC_GRASP_ENABLED else FIXED_PICK_LIFT_M
            )
            lift_endpoint = np.asarray(A.ee_pos(), dtype=float)
            lift_endpoint[2] = min(
                A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
                lift_endpoint[2] + lift_distance,
            )
            lift_waypoints, lift_plan_reason = _plan_fixed_approach_trajectory(
                lift_endpoint
            )
            steps.append(("prevalidate_micro_lift" if DYNAMIC_GRASP_ENABLED
                          else "prevalidate_lift",
                          lift_waypoints is not None, lift_plan_reason))
            if lift_waypoints is None:
                return recover(f"post-grasp lift is unsafe: {lift_plan_reason}")
            log(f"stage={'dynamic_micro_lift' if DYNAMIC_GRASP_ENABLED else 'post_grasp_lift'} "
                f"attempt={retry_count + 1} distance_mm={lift_distance * 1000:.1f}")
            lift_ok, lift_reason = _execute_fixed_approach_trajectory(
                lift_waypoints, log
            )
            steps.append(("micro_lift" if DYNAMIC_GRASP_ENABLED else "lift",
                          lift_ok, lift_reason))
            if not lift_ok:
                return recover(f"post-grasp lift failed: {lift_reason}")

            lifted_det, lifted_frame, lifted_why = _observe(eye, object_name)
            lifted_hold_det = lifted_det
            lifted_hold_reason = str(lifted_why)
            if DYNAMIC_GRASP_ENABLED and lifted_det is not None \
                    and lifted_frame is not None:
                lifted_geometry = _plan_scene_grasp_geometry(
                    lifted_frame, lifted_det,
                    current_rpy=tuple(float(v) for v in A.ee_rpy()),
                    allow_orientation=False, object_name=object_name,
                )
                if lifted_geometry.ok \
                        and lifted_geometry.grasp_detection is not None:
                    lifted_hold_det = lifted_geometry.grasp_detection
                    lifted_hold_reason = (
                        "semantic body grasp region tracked after lift"
                    )
                else:
                    lifted_hold_det = None
                    lifted_hold_reason = lifted_geometry.reason
            if lifted_frame is not None:
                _show(*lifted_frame,
                      tag=f"dynamic_lift_attempt_{retry_count + 1}",
                      det=lifted_hold_det, target_cam=before_lift_cam,
                      note="visual hold check after micro-lift")
            rigid_error = None
            if lifted_hold_det is not None:
                rigid_error = float(np.linalg.norm(
                    np.asarray(lifted_hold_det.p_cam_mm, dtype=float)
                    - before_lift_cam
                ))
            hold_tol = (
                float(np.clip(DYNAMIC_HOLD_TOL_MM, 3.0, 8.0))
                if DYNAMIC_GRASP_ENABLED else FIXED_HELD_RIGID_TOL_MM
            )
            held = lifted_hold_det is not None and rigid_error is not None and \
                rigid_error <= hold_tol
            steps.append(("dynamic_visual_hold_check" if DYNAMIC_GRASP_ENABLED
                          else "visual_rigid_hold_check", held,
                          f"camera-relative shift {rigid_error:.1f}mm; "
                          f"limit {hold_tol:.1f}mm"
                          if rigid_error is not None else lifted_hold_reason))
            log(f"stage=visual_rigid_hold held={held} attempt={retry_count + 1} "
                f"shift_mm={rigid_error if rigid_error is not None else 'unavailable'} "
                f"limit_mm={hold_tol:.1f}")
            if held:
                break
            # The initial close plus at most three visually-gated retries gives
            # the uncertain single-span jaw model one additional bounded angle
            # step.  Every retry still requires returning to the validated jaw
            # slot and remains subject to the immutable 15deg hard limit.
            max_retries = int(np.clip(DYNAMIC_MAX_RETRIES, 0, 3))
            if not DYNAMIC_GRASP_ENABLED or retry_count >= max_retries:
                _OBJECT_HELD = False
                return recover(
                    "object did not move rigidly with the wrist after the "
                    "bounded visual grasp attempt; grasp not confirmed"
                )

            # Return to the exact validated endpoint before changing only the jaw
            # position. No camera observation is allowed to steer this motion.
            lower_waypoints, lower_reason = _plan_fixed_approach_trajectory(
                grasp_xyz
            )
            if lower_waypoints is None:
                return recover(f"dynamic retry return is unsafe: {lower_reason}")
            lower_ok, lower_reason = _execute_fixed_approach_trajectory(
                lower_waypoints, log
            )
            steps.append(("dynamic_retry_return", lower_ok, lower_reason))
            if not lower_ok:
                return recover(f"dynamic retry could not return: {lower_reason}")
            retry_det, retry_frame, retry_why = _observe(eye, object_name)
            if retry_frame is not None:
                _show(*retry_frame,
                      tag=f"dynamic_retry_endpoint_{retry_count + 1}",
                      det=retry_det, target_cam=before_lift_cam,
                      note="returned endpoint before tighter close")
            if retry_det is None:
                return recover(f"dynamic retry lost the object: {retry_why}")
            retry_shift = float(np.linalg.norm(
                np.asarray(retry_det.p_cam_mm, dtype=float)
                - jaw_slot_feature_cam
            ))
            retry_slot_tol = float(np.clip(DYNAMIC_SLOT_TOL_MM, 3.0, 8.0))
            if retry_shift > retry_slot_tol:
                return recover(
                    f"object shifted {retry_shift:.1f}mm after the failed lift; "
                    f"outside the {retry_slot_tol:.1f}mm jaw-slot limit; "
                    "refusing a blind retry"
                )
            retry_geometry = _plan_scene_grasp_geometry(
                retry_frame, retry_det,
                current_rpy=tuple(float(v) for v in A.ee_rpy()),
                allow_orientation=False, object_name=object_name,
            ) if retry_frame is not None else None
            if retry_geometry is None or not retry_geometry.ok \
                    or retry_geometry.grasp_detection is None:
                return recover(
                    "dynamic retry could not re-identify the body section "
                    "between the jaws; refusing additional closure"
                )
            from arm_agent import gripper_model as _retry_gm
            if dynamic_has_jaw_curve:
                retry_mm = current_grasp_mm - float(np.clip(
                    DYNAMIC_RETRY_CLOSE_MM, 0.0, 1.0
                ))
                retry_rad = _retry_gm.rad_for_aperture(
                    retry_mm, _wrist_limits()
                )
                actual_retry_mm = (
                    _retry_gm.aperture_mm(retry_rad)
                    if retry_rad is not None else None
                )
                if retry_rad is None or actual_retry_mm is None \
                        or abs(actual_retry_mm - retry_mm) > SIZE_TARGET_TOL_MM:
                    return recover(
                        f"calibrated jaw curve cannot represent the bounded "
                        f"{retry_mm:.1f}mm retry aperture"
                    )
                retry_basis = "piecewise jaw curve, 1mm tighter"
            else:
                close_direction = 1.0 if _retry_gm.OPEN_RAD >= \
                    _retry_gm.CLOSE_RAD else -1.0
                angle_step = math.radians(float(np.clip(
                    DYNAMIC_RETRY_CLOSE_DEG, 1.0, 5.0
                )))
                retry_rad = _retry_gm.clamp_rad(
                    current_grasp_rad - close_direction * angle_step,
                    _wrist_limits(),
                )
                if math.isclose(retry_rad, current_grasp_rad, abs_tol=1e-6):
                    return recover(
                        "bounded visual retry reached the hard jaw-angle limit; "
                        "refusing additional closure"
                    )
                actual_retry_mm = _retry_gm.aperture_mm(retry_rad)
                retry_basis = (
                    f"single-span model uncertain; commanded "
                    f"{math.degrees(angle_step):.1f}deg tighter"
                )
            retry_count += 1
            current_grasp_rad = float(retry_rad)
            if actual_retry_mm is not None:
                current_grasp_mm = float(actual_retry_mm)
            before_lift_cam = np.asarray(
                retry_geometry.grasp_detection.p_cam_mm, dtype=float
            )
            log(f"stage=dynamic_retry_close attempt={retry_count + 1} "
                f"target_mm={current_grasp_mm:.1f} "
                f"target_deg={math.degrees(retry_rad):.1f} basis={retry_basis}")
            retry_close = _gripper("close", retry_rad)
            steps.append(("dynamic_retry_position_close",
                          bool(retry_close.get("ok")),
                          f"{math.degrees(retry_rad):.1f}deg commanded; "
                          f"{retry_basis}; no contact inference"))
            if not retry_close.get("ok"):
                return recover(
                    f"bounded retry close failed: {retry_close.get('reason')}"
                )
            log("stage=dynamic_retry_close_readback "
                f"attempt={retry_count + 1} "
                f"target_deg={math.degrees(retry_rad):.1f} "
                f"actual_deg={retry_close.get('position_deg', 'unavailable')} "
                f"delta_deg={retry_close.get('delta_deg', 'unavailable')}")
            time.sleep(0.5)

        if DYNAMIC_GRASP_ENABLED:
            carry_endpoint = grasp_xyz.copy()
            carry_endpoint[2] = min(
                A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
                grasp_xyz[2] + FIXED_PICK_LIFT_M,
            )
            remaining_waypoints, remaining_reason = \
                _plan_fixed_approach_trajectory(carry_endpoint)
            if remaining_waypoints is None:
                return recover(f"remaining carry lift is unsafe: {remaining_reason}")
            remaining_ok, remaining_reason = _execute_fixed_approach_trajectory(
                remaining_waypoints, log
            )
            steps.append(("lift_to_carry_after_visual_hold",
                          remaining_ok, remaining_reason))
            if not remaining_ok:
                return recover(
                    f"remaining carry lift failed: {remaining_reason}"
                )
        _OBJECT_HELD = True

        _FIXED_HELD_CONTEXT = {
            "object_name": object_name,
            "picked_at_monotonic_s": time.monotonic(),
            "jaw_plan": {
                "object_width_mm": round(float(jaw_plan["min_width_mm"]), 1),
                "preopen_mm": round(float(jaw_plan["preopen_mm"]), 1),
                "grasp_mm": round(float(jaw_plan["grasp_mm"]), 1),
                "contact_inference": False,
            },
            "grasp_geometry": grasp_geometry_summary,
            "reachability": reachability,
            "final_alignment_corrections": corrections,
        }
        if not place_after_pick:
            log("stage=fixed_pick_complete object_held=true "
                "next_skill=wrist_place controller=kept_alive")
            return _result(
                True,
                "object picked, visually lift-verified, and held at the carry pose",
                steps, verified=True, homed=False, gripper_opened=False,
                object_held=True, holding=True,
                next_skill="wrist_place",
                requires_same_controller_process=True,
                jaw_plan=_FIXED_HELD_CONTEXT["jaw_plan"],
                grasp_geometry=grasp_geometry_summary,
                reachability=reachability,
                final_alignment_corrections=corrections,
            )

        if scene_place:
            log(f"stage=scene_place_handoff relation={normalized_place_relation} "
                f"reference={place_reference!r} same_camera=true")
            pick_context = dict(_FIXED_HELD_CONTEXT or {})
            placed = fixed_place(
                reference=place_reference,
                relation=str(normalized_place_relation),
                distance_m=place_distance_m,
                dry=False, verbose=verbose, eye=eye,
                planned_release_endpoint=preplanned_scene_endpoint,
                planned_placement=preplanned_scene_detail,
            )
            placed["steps"] = [
                {"step": label, "ok": bool(ok), "reason": reason}
                for label, ok, reason in steps
            ] + list(placed.get("steps", []))
            placed["pick"] = {
                "ok": True,
                "object": object_name,
                "verified": True,
                "jaw_plan": pick_context.get("jaw_plan"),
                "grasp_geometry": grasp_geometry_summary,
                "reachability": reachability,
                "final_alignment_corrections": corrections,
            }
            return placed

        assert drop is not None
        release_z = min(
            A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
            max(FIXED_DROP_RELEASE_Z_M, float(drop[2]) + 0.10),
        )
        release_endpoint = np.array(
            [float(drop[0]), float(drop[1]), release_z], dtype=float
        )
        drop_waypoints, drop_plan_reason = _plan_fixed_approach_trajectory(
            release_endpoint
        )
        steps.append(("prevalidate_drop_trajectory", drop_waypoints is not None,
                      drop_plan_reason))
        if drop_waypoints is None:
            return recover(f"drop trajectory is unsafe: {drop_plan_reason}")
        log(f"stage=drop_traverse endpoint={release_endpoint.round(4).tolist()} "
            f"waypoints={len(drop_waypoints)}")
        drop_ok, drop_reason = _execute_fixed_approach_trajectory(
            drop_waypoints, log
        )
        steps.append(("drop_traverse", drop_ok, drop_reason))
        if not drop_ok:
            return recover(f"drop traverse failed: {drop_reason}")

        log("stage=drop_release opening_gripper=true")
        released_cmd = _gripper("open")
        time.sleep(0.5)
        released = _verify_release(released_cmd)
        steps.append(("release", released is True,
                      "open position reached" if released is True else
                      "open position was not verified"))
        _OBJECT_HELD = False if released is True else None
        if _OBJECT_HELD is False:
            _FIXED_HELD_CONTEXT = None
        if released is not True:
            return recover("drop release was not verified")
        if SCENE_COLLISION_ENABLED:
            A.update_collision_payload(0.0)

        recovery = _home_after_interrupt(open_gripper=False)
        steps.append(("return_home", bool(recovery.get("homed")),
                      str(recovery.get("reason"))))
        return _result(
            bool(recovery.get("homed")),
            "object picked by sized position close, released over the taught "
            "drop zone, and arm homed" if recovery.get("homed")
            else "object released but home was not verified",
            steps, verified=True, homed=bool(recovery.get("homed")),
            gripper_opened=True, object_held=False, recovery=recovery,
            jaw_plan={
                "object_width_mm": round(float(jaw_plan["min_width_mm"]), 1),
                "preopen_mm": round(float(jaw_plan["preopen_mm"]), 1),
                "grasp_mm": round(float(jaw_plan["grasp_mm"]), 1),
                "contact_inference": False,
            },
            grasp_geometry=grasp_geometry_summary,
            reachability=reachability,
            high_drop_xyz_m=release_endpoint.round(4).tolist(),
            final_alignment_corrections=corrections,
            collision_scene=collision_detail,
        )
    except BaseException as exc:
        if dry:
            return _result(False, f"dry fixed pick/place exception: "
                           f"{type(exc).__name__}: {exc}", steps,
                           homed=False, object_held=False)
        return recover(f"fixed pick/place exception: {type(exc).__name__}: {exc}")
    finally:
        stop_feed = getattr(eye, "stop_live_feed", None)
        if callable(stop_feed):
            stop_feed()
        if own_eye:
            eye.close()


def fixed_pick(object_name: str, *, dry: bool = False,
               verbose: bool = False, eye: WristEye | None = None
               ) -> dict[str, Any]:
    """Run the proven fixed pick through its verified carry lift, then hold.

    Live use is intended for the persistent Hermes/MCP process.  The subsequent
    :func:`fixed_place` call must execute in that same process so controller
    ownership and the in-memory held-object certificate are not lost.
    """
    return fixed_pick_and_place(
        object_name, dry=dry, verbose=verbose, eye=eye,
        place_after_pick=False,
    )


_SCENE_RELATIONS = {"left", "right", "front", "behind", "next_to"}


def _normalize_scene_relation(relation: str | None) -> str | None:
    value = (relation or "drop_zone").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "left_of": "left", "right_of": "right",
        "in_front_of": "front", "front_of": "front",
        "behind_of": "behind", "beside": "next_to", "near": "next_to",
        "next": "next_to", "somewhere": "free_space",
        "somewhere_on_table": "free_space", "on_table": "free_space",
        "taught": "drop_zone", "bin": "drop_zone",
    }
    value = aliases.get(value, value)
    return value if value in {*_SCENE_RELATIONS, "free_space", "drop_zone"} else None


def _held_footprint_radius_m(context: dict[str, Any]) -> float:
    """Conservative bounded radius from the pre-pick visual geometry."""
    geometry = context.get("grasp_geometry") or {}
    envelope = geometry.get("envelope_detection") or {}
    grasp = geometry.get("grasp_detection") or {}
    widths = [
        value for value in (
            envelope.get("max_width_mm"), grasp.get("max_width_mm"),
            grasp.get("min_width_mm"),
        ) if isinstance(value, (int, float)) and value > 0
    ]
    diameter_mm = max(widths) if widths else 40.0
    # Semantic envelopes can include background; never let that force a huge,
    # unexplained relative offset. The explicit caller distance remains available.
    return float(np.clip(diameter_mm / 2000.0, 0.015, 0.060))


def _scene_reference_consensus(
    eye: WristEye, reference: str, log: Callable[[str], None]
) -> tuple[Any | None, np.ndarray | None, dict[str, Any]]:
    samples: list[np.ndarray] = []
    latest = None
    latest_frame = None
    reasons: list[Any] = []
    count = int(np.clip(SCENE_PLACE_CONSENSUS_FRAMES, 3, 7))
    for index in range(count):
        det, frame, why = _observe(eye, reference)
        if det is None:
            reasons.append(why)
            continue
        latest, latest_frame = det, frame
        samples.append(np.asarray(det.p_cam_mm, dtype=float))
        log(f"stage=place_reference_sample frame={index + 1}/{count} "
            f"source={det.source} p_cam_mm={samples[-1].round(1).tolist()}")
        time.sleep(0.08)
    if len(samples) < max(2, count - 1):
        return None, None, {
            "reason": f"reference detected in only {len(samples)}/{count} frames",
            "detector_reasons": reasons,
        }
    points = np.vstack(samples)
    median = np.median(points, axis=0)
    spread = float(np.max(np.linalg.norm(points - median, axis=1)))
    if spread > SCENE_PLACE_MAX_SPREAD_MM:
        return None, None, {
            "reason": f"reference depth/position spread {spread:.1f}mm exceeds "
                      f"{SCENE_PLACE_MAX_SPREAD_MM:.1f}mm",
            "samples_camera_mm": points.round(1).tolist(),
        }
    if latest_frame is not None:
        target = load_grasp_target()
        _show(*latest_frame, tag="scene_place_reference", det=latest,
              tcp_cam=np.asarray(target, dtype=float) if target else None,
              note=f"placement reference: {reference}")
    return latest, median, {
        "reason": f"{len(samples)}/{count} stable frames; spread={spread:.1f}mm",
        "reference_camera_mm": median.round(1).tolist(),
        "spread_mm": round(spread, 1),
        "source": latest.source if latest is not None else None,
    }


def _high_release_waypoints(
    endpoint: np.ndarray, fixed_rpy: tuple[float, float, float]
) -> tuple[list[tuple[float, float, float]] | None, str]:
    """Choose a fully IK- and collision-validated carry corridor."""
    start = np.asarray(A.ee_pos(), dtype=float)
    ceiling = A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M
    base_z = min(ceiling, max(float(start[2]), float(endpoint[2])))
    travel_heights = sorted({
        round(base_z, 4),
        round(min(ceiling, base_z + 0.040), 4),
        round(min(ceiling, base_z + 0.080), 4),
    })
    failures: list[str] = []
    valid: list[tuple[float, list[tuple[float, float, float]], str]] = []
    for travel_z in travel_heights:
        start_top = np.array([start[0], start[1], travel_z], dtype=float)
        end_top = np.array([endpoint[0], endpoint[1], travel_z], dtype=float)
        route_candidates = [
            ("direct", [start_top, end_top, endpoint]),
            ("x_then_y", [start_top,
                           np.array([endpoint[0], start[1], travel_z]),
                           end_top, endpoint]),
            ("y_then_x", [start_top,
                           np.array([start[0], endpoint[1], travel_z]),
                           end_top, endpoint]),
        ]
        for route_name, route in route_candidates:
            combined: list[tuple[float, float, float]] = []
            leg_start = start
            failed = None
            for leg_endpoint in route:
                if float(np.linalg.norm(leg_endpoint - leg_start)) < 1e-7:
                    continue
                leg, reason = _plan_fixed_approach_trajectory(
                    leg_endpoint, start_m=leg_start, fixed_rpy=fixed_rpy,
                    validate_ik=False,
                )
                if leg is None:
                    failed = reason
                    break
                combined.extend(leg)
                leg_start = leg_endpoint
            if failed is not None or not combined:
                failures.append(
                    f"{route_name}@z={travel_z:.3f}: {failed or 'empty path'}"
                )
                continue
            ok, reason = _validate_fixed_waypoints(combined, fixed_rpy)
            if not ok:
                failures.append(f"{route_name}@z={travel_z:.3f}: {reason}")
                continue
            length = float(sum(np.linalg.norm(
                np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
            ) for a, b in zip([tuple(start), *combined[:-1]], combined)))
            valid.append((length, combined,
                          f"{route_name} at z={travel_z:.3f}; {reason}"))
    if not valid:
        return None, "all collision-aware carry corridors failed: " + "; ".join(failures)
    _, selected, detail = min(valid, key=lambda item: item[0])
    return selected, f"collision-aware high release route; {detail}"


def _plan_scene_relative_release(
    reference: str,
    relation: str,
    distance_m: float | None,
    context: dict[str, Any],
    eye: WristEye,
    log: Callable[[str], None],
    *,
    camera_mapping_override: np.ndarray | None = None,
    camera_mapping_detail: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, list[tuple[float, float, float]] | None, dict[str, Any]]:
    det, reference_cam, observation = _scene_reference_consensus(
        eye, reference, log
    )
    if det is None or reference_cam is None:
        return None, None, observation
    grasp_target = load_grasp_target()
    if grasp_target is None:
        return None, None, {"reason": "open-jaw visual target is not calibrated"}
    current_rpy = tuple(float(v) for v in A.ee_rpy())
    if camera_mapping_override is not None:
        camera_mapping = np.asarray(camera_mapping_override, dtype=float)
        mapping_detail = dict(camera_mapping_detail or {
            "reason": "caller-supplied rigid camera mapping"
        })
    else:
        camera_mapping, mapping_detail = load_saved_camera_jacobian(current_rpy)
    if camera_mapping is None:
        return None, None, {
            "reason": f"saved rigid camera mapping unavailable: "
                      f"{mapping_detail.get('reason')}"
        }
    try:
        align_delta = np.linalg.solve(
            camera_mapping,
            np.asarray(grasp_target, dtype=float) - reference_cam,
        )
    except np.linalg.LinAlgError:
        return None, None, {"reason": "saved rigid camera mapping is singular"}

    if distance_m is None:
        ref_diameter_mm = max(
            float(det.min_width_mm or 0.0), float(det.max_width_mm or 0.0), 30.0
        )
        reference_radius = float(np.clip(ref_diameter_mm / 2000.0, 0.015, 0.060))
        offset = reference_radius + _held_footprint_radius_m(context) + 0.025
        offset = float(np.clip(max(offset, SCENE_PLACE_OFFSET_M), 0.070, 0.150))
        distance_basis = "object radii + 25mm clearance"
    else:
        offset = float(np.clip(distance_m, 0.060, 0.180))
        distance_basis = "caller distance, clamped to 60-180mm"

    direction = {
        "left": np.array([0.0, 1.0]),
        "right": np.array([0.0, -1.0]),
        "front": np.array([1.0, 0.0]),
        "behind": np.array([-1.0, 0.0]),
    }
    names = [relation] if relation != "next_to" else [
        "left", "right", "front", "behind"
    ]
    current = np.asarray(A.ee_pos(), dtype=float)
    reference_xy = current[:2] + align_delta[:2]
    release_z = min(
        A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M, FIXED_DROP_RELEASE_Z_M
    )
    candidates: list[dict[str, Any]] = []
    for name in names:
        endpoint = np.array([
            *(reference_xy + direction[name] * offset), release_z
        ], dtype=float)
        waypoints, reason = _high_release_waypoints(endpoint, current_rpy)
        edge_clearance = min(
            endpoint[0] - A.X_MIN, A.X_MAX - endpoint[0],
            endpoint[1] - A.Y_MIN, A.Y_MAX - endpoint[1],
        )
        candidates.append({
            "relation": name, "endpoint": endpoint,
            "waypoints": waypoints, "reason": reason,
            "edge_clearance_m": float(edge_clearance),
        })
    valid = [candidate for candidate in candidates
             if candidate["waypoints"] is not None]
    if not valid:
        return None, None, {
            "reason": "no requested relative side has a complete IK path",
            "candidates": [{
                "relation": item["relation"], "reason": item["reason"],
                "endpoint_m": item["endpoint"].round(4).tolist(),
            } for item in candidates],
        }
    selected = max(valid, key=lambda item: item["edge_clearance_m"])
    detail = {
        **observation,
        "requested_relation": relation,
        "selected_relation": selected["relation"],
        "reference": reference,
        "reference_aligned_xy_m": reference_xy.round(4).tolist(),
        "offset_m": round(offset, 4),
        "distance_basis": distance_basis,
        "release_endpoint_m": selected["endpoint"].round(4).tolist(),
        "camera_mapping": mapping_detail,
        "collision_model": bool(SCENE_COLLISION_ENABLED),
        "warning": (
            "static RGB-D scene, URDF self-collision, table, and attached "
            "payload checked; unseen or moving obstacles still trigger the "
            "operator stop path"
            if SCENE_COLLISION_ENABLED else
            "IK/workspace only; scene obstacles are not modelled"
        ),
    }
    return selected["endpoint"], selected["waypoints"], detail


def fixed_place(*, reference: str | None = None,
                relation: str = "drop_zone", distance_m: float | None = None,
                dry: bool = False, verbose: bool = False,
                eye: WristEye | None = None,
                planned_release_endpoint: Any | None = None,
                planned_placement: dict[str, Any] | None = None) -> dict[str, Any]:
    """Place an object held by :func:`fixed_pick`, then return home.

    This deliberately accepts no caller-supplied ``held=True`` override.  Only a
    successful fixed pick in this controller process authorizes the traverse.
    """
    global _OBJECT_HELD, _FIXED_HELD_CONTEXT
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    normalized_relation = _normalize_scene_relation(relation)
    log(f"stage=fixed_place_start mode={'dry' if dry else 'live'} "
        f"relation={relation!r} reference={reference!r}")
    if normalized_relation is None:
        return _result(False, f"unsupported placement relation: {relation!r}",
                       [("placement_description", False,
                         "use left/right/front/behind/next_to/free_space/drop_zone")],
                       object_held=_OBJECT_HELD, homed=False)
    if normalized_relation in _SCENE_RELATIONS and not reference:
        return _result(False, f"{normalized_relation} placement needs a reference object",
                       [("placement_description", False, "reference is missing")],
                       object_held=_OBJECT_HELD, homed=False)

    drop = load_drop_zone()
    uses_taught_drop = normalized_relation in {"drop_zone", "free_space"}
    if uses_taught_drop:
        steps.append(("drop_zone", drop is not None,
                      f"{tuple(round(v, 3) for v in drop)}m" if drop else "missing"))
    if uses_taught_drop and drop is None:
        return _result(False, "drop zone not taught", steps,
                       object_held=_OBJECT_HELD, homed=False)

    release_endpoint = None
    if uses_taught_drop:
        assert drop is not None
        release_z = min(
            A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
            max(FIXED_DROP_RELEASE_Z_M, float(drop[2]) + 0.10),
        )
        release_endpoint = np.array(
            [float(drop[0]), float(drop[1]), release_z], dtype=float
        )

    if dry:
        if release_endpoint is None:
            return _result(
                False,
                "dry scene-relative placement requires a live same-session held "
                "object and current carry-pose camera view",
                [("scene_relative_dry", False,
                  "reference localization is intentionally not guessed")],
                homed=False, object_held=False,
            )
        pf = A.preflight(connect=False)
        steps.append(("preflight", pf.ok, pf.reason))
        if not pf.ok:
            return _result(False, "arm not ready", steps,
                           object_held=False, homed=False)
        drop_ik = A.check_ik(*release_endpoint, 0.0, 0.0, 0.0)
        steps.append(("high_drop_ik", drop_ik.ok, drop_ik.reason))
        return _result(
            drop_ik.ok,
            "dry fixed place check passed" if drop_ik.ok
            else "dry fixed place check failed",
            steps, homed=False, object_held=False,
            high_drop_xyz_m=release_endpoint.round(4).tolist(),
        )

    held_in_this_process = (
        _OBJECT_HELD is True and _FIXED_HELD_CONTEXT is not None
    )
    steps.append((
        "same_process_pick_certificate", held_in_this_process,
        (f"holding {_FIXED_HELD_CONTEXT['object_name']!r}"
         if held_in_this_process else
         "no successful fixed pick is active in this controller process"),
    ))
    if not held_in_this_process:
        return _result(
            False,
            "place refused: no object picked by this live controller session",
            steps, object_held=_OBJECT_HELD, homed=False,
            safe_to_retry=False, needs_operator=True,
        )

    pf = A.preflight(connect=True)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        return _result(False, "arm not ready", steps,
                       object_held=True, homed=False,
                       safe_to_retry=False, needs_operator=True)

    def recover(reason: str) -> dict[str, Any]:
        global _OBJECT_HELD, _FIXED_HELD_CONTEXT
        recovery = _home_after_interrupt(open_gripper=True)
        _OBJECT_HELD = False if recovery.get("gripper_opened") else None
        if _OBJECT_HELD is False:
            _FIXED_HELD_CONTEXT = None
        return _result(
            False, reason, steps,
            homed=bool(recovery.get("homed")),
            gripper_opened=bool(recovery.get("gripper_opened")),
            object_held=_OBJECT_HELD, recovery=recovery,
        )

    context = dict(_FIXED_HELD_CONTEXT)
    placement_detail: dict[str, Any] = {
        "requested_relation": normalized_relation,
        "reference": reference,
    }
    scene_waypoints: list[tuple[float, float, float]] | None = None
    own_eye = False
    try:
        if normalized_relation in _SCENE_RELATIONS:
            if planned_release_endpoint is not None:
                release_endpoint = np.asarray(
                    planned_release_endpoint, dtype=float
                )
                placement_detail = dict(planned_placement or {})
                placement_detail.update({
                    "reference_snapshot_stage": "pre_pick_ready_pose",
                    "reference_redetection_at_carry": False,
                    "release_endpoint_m": release_endpoint.round(4).tolist(),
                })
                scene_waypoints = None
                log("stage=scene_place_use_cached_reference "
                    f"endpoint={release_endpoint.round(4).tolist()} "
                    "carry_redetection=false")
            else:
                own_eye = eye is None
                eye = eye or WristEye()
                assert reference is not None
                release_endpoint, scene_waypoints, placement_detail = \
                    _plan_scene_relative_release(
                        reference, normalized_relation, distance_m,
                        context, eye, log,
                    )
            steps.append(("scene_relative_release_plan",
                          release_endpoint is not None and (
                              scene_waypoints is not None
                              or planned_release_endpoint is not None
                          ),
                          str(placement_detail.get("reason") or
                              placement_detail.get("warning"))))
            if release_endpoint is None or (
                scene_waypoints is None and planned_release_endpoint is None
            ):
                if own_eye:
                    eye.close()
                    own_eye = False
                return _result(
                    False,
                    f"scene-relative placement refused: "
                    f"{placement_detail.get('reason')}",
                    steps, object_held=True, homed=False,
                    safe_to_retry=True, needs_operator=False,
                    placement=placement_detail,
                )
        else:
            # With no collision model, "somewhere on the table" means the explicitly
            # taught safe release location; it never invents an unobserved empty spot.
            if normalized_relation == "free_space":
                placement_detail.update({
                    "selected_relation": "taught_safe_table_point",
                    "collision_model": False,
                    "reason": "free-space request mapped to taught drop zone",
                })
            assert release_endpoint is not None
    except BaseException as exc:
        if own_eye and eye is not None:
            eye.close()
            own_eye = False
        return recover(
            f"scene-relative planning exception: {type(exc).__name__}: {exc}"
        )

    try:
        if scene_waypoints is not None:
            drop_waypoints = scene_waypoints
            drop_plan_reason = "scene-relative high-release path prevalidated"
        else:
            drop_waypoints, drop_plan_reason = _high_release_waypoints(
                release_endpoint, tuple(float(v) for v in A.ee_rpy())
            )
        steps.append(("prevalidate_drop_trajectory", drop_waypoints is not None,
                      drop_plan_reason))
        if drop_waypoints is None:
            return recover(f"drop trajectory is unsafe: {drop_plan_reason}")

        log(f"stage=drop_traverse endpoint={release_endpoint.round(4).tolist()} "
            f"waypoints={len(drop_waypoints)}")
        drop_ok, drop_reason = _execute_fixed_approach_trajectory(
            drop_waypoints, log
        )
        steps.append(("drop_traverse", drop_ok, drop_reason))
        if not drop_ok:
            return recover(f"drop traverse failed: {drop_reason}")

        log("stage=drop_release opening_gripper=true")
        released_cmd = _gripper("open")
        time.sleep(0.5)
        released = _verify_release(released_cmd)
        steps.append(("release", released is True,
                      "open position reached" if released is True else
                      "open position was not verified"))
        _OBJECT_HELD = False if released is True else None
        if _OBJECT_HELD is False:
            _FIXED_HELD_CONTEXT = None
        if released is not True:
            return recover("drop release was not verified")
        if SCENE_COLLISION_ENABLED:
            A.update_collision_payload(0.0)

        recovery = _home_after_interrupt(open_gripper=False)
        if SCENE_COLLISION_ENABLED and recovery.get("homed"):
            A.clear_collision_scene()
        steps.append(("return_home", bool(recovery.get("homed")),
                      str(recovery.get("reason"))))
        return _result(
            bool(recovery.get("homed")),
            ("held object released at the scene-relative table position and arm homed"
             if normalized_relation in _SCENE_RELATIONS else
             "held object released over the taught drop zone and arm homed")
            if recovery.get("homed") else
            "object released but home was not verified",
            steps, verified=True, homed=bool(recovery.get("homed")),
            gripper_opened=True, object_held=False, recovery=recovery,
            picked_object=context.get("object_name"),
            pick_jaw_plan=context.get("jaw_plan"),
            high_drop_xyz_m=release_endpoint.round(4).tolist(),
            placement=placement_detail,
        )
    except BaseException as exc:
        return recover(f"fixed place exception: {type(exc).__name__}: {exc}")
    finally:
        if own_eye and eye is not None:
            eye.close()


def abort_fixed_hold(reason: str = "operator requested stop") -> dict[str, Any]:
    """Freeze, release, and home a fixed pick/hold operation."""
    global _OBJECT_HELD, _FIXED_HELD_CONTEXT
    recovery = _home_after_interrupt(open_gripper=True)
    _OBJECT_HELD = False if recovery.get("gripper_opened") else None
    if _OBJECT_HELD is False:
        _FIXED_HELD_CONTEXT = None
    return _result(
        bool(recovery.get("homed")), reason,
        [
            ("freeze", bool(recovery.get("frozen")),
             str(recovery.get("freeze", {}).get("reason"))),
            ("open_gripper", bool(recovery.get("gripper_opened")),
             str(recovery.get("gripper", {}).get("reason", "requested"))),
            ("return_home", bool(recovery.get("homed")),
             str(recovery.get("home", {}).get("reason"))),
        ],
        status="stopped", homed=bool(recovery.get("homed")),
        gripper_opened=bool(recovery.get("gripper_opened")),
        object_held=_OBJECT_HELD, recovery=recovery,
    )


def hold_fixed_pick_until_quit(*, verbose: bool = False) -> dict[str, Any]:
    """Keep a CLI pick process alive until Q or Ctrl+C requests recovery.

    Terminal input is used only by the standalone CLI. MCP uses stdio for its
    protocol and therefore stops through the separate ``wrist_stop`` tool.
    """
    log = _printer(verbose)
    if _OBJECT_HELD is not True or _FIXED_HELD_CONTEXT is None:
        return _result(False, "nothing is held by this fixed-pick session", [],
                       homed=False, object_held=_OBJECT_HELD)

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    interactive = sys.stdin.isatty()
    previous_tty = termios.tcgetattr(fd) if interactive else None
    print(
        "[fixed-pick] Object is held at the carry pose. Press Q to stop, "
        "open the gripper, and home. Ctrl+C does the same.",
        flush=True,
    )
    try:
        if interactive:
            tty.setcbreak(fd)
        while True:
            if interactive and select.select([sys.stdin], [], [], 0.10)[0]:
                key = sys.stdin.read(1)
                if key.lower() == "q":
                    log("stage=fixed_pick_hold_stop input=q action=freeze_open_home")
                    return abort_fixed_hold("Q pressed; gripper opened and arm homed")
            else:
                time.sleep(0.10)
    except KeyboardInterrupt:
        log("stage=fixed_pick_hold_stop input=Ctrl+C action=freeze_open_home")
        return abort_fixed_hold("Ctrl+C pressed; gripper opened and arm homed")
    finally:
        if interactive and previous_tty is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous_tty)


def run_fixed_pick_cli(object_name: str, *, verbose: bool = False) -> dict[str, Any]:
    """Run live fixed-pick with a Q listener active for the entire operation."""
    import os as _os
    import select
    import termios
    import threading
    import tty

    stop_listener = threading.Event()
    listener_ready = threading.Event()

    def listen_for_q() -> None:
        if not sys.stdin.isatty():
            listener_ready.set()
            return
        fd = sys.stdin.fileno()
        previous_tty = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            listener_ready.set()
            while not stop_listener.is_set():
                if select.select([sys.stdin], [], [], 0.10)[0]:
                    if sys.stdin.read(1).lower() == "q":
                        # Deliver SIGINT to the main thread. The active fixed-pick
                        # exception path performs freeze/open/home before returning.
                        _os.kill(_os.getpid(), signal.SIGINT)
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous_tty)
            listener_ready.set()

    listener = threading.Thread(
        target=listen_for_q, name="wrist-q-stop", daemon=True
    )
    listener.start()
    listener_ready.wait(timeout=1.0)
    print(
        "[fixed-pick] Q or Ctrl+C stops at any stage: freeze, open gripper, home.",
        flush=True,
    )
    try:
        picked = fixed_pick(object_name, dry=False, verbose=verbose)
        if not picked.get("ok"):
            return picked
        print(
            "[fixed-pick] Object is held at the carry pose. Press Q or Ctrl+C "
            "to open the gripper, home, and exit.",
            flush=True,
        )
        try:
            while True:
                time.sleep(0.20)
        except KeyboardInterrupt:
            held = abort_fixed_hold(
                "Q or Ctrl+C pressed; gripper opened and arm homed"
            )
            held["pick"] = picked
            return held
    finally:
        stop_listener.set()
        listener.join(timeout=1.0)


# =====================================================================================
# SKILL 1: pick with the wrist camera
# =====================================================================================
def servo_pick(object_name: str, *, dry: bool = False, verbose: bool = False,
               eye: WristEye | None = None) -> dict[str, Any]:
    """Pick a named object using the wrist camera alone.

    pre:   arm ready, TCP-in-camera calibrated, object visible with valid depth.
    act:   scan pose -> learn Jacobian -> servo the object under the TCP ->
           descend by feel (torque-gated) -> close.
    post:  lift, then visually confirm the named object at the calibrated TCP.
    reset: lift to a carry height. Leaves the object HELD.
    """
    global _OBJECT_HELD
    _OBJECT_HELD = False
    log = _printer(verbose)
    log(f"stage=pick_start object={object_name!r} mode={'dry' if dry else 'live'}")
    steps: list[tuple[str, bool, str]] = []

    tcp = load_tcp()
    steps.append(("tcp_calibration", tcp is not None,
                  f"{tuple(round(v, 1) for v in tcp)}mm" if tcp else
                  "missing -- run: python -m arm_agent.wrist_servo --calib-tcp --run"))
    if tcp is None:
        return _result(False, "wrist TCP not calibrated", steps)
    tcp_cam = np.asarray(tcp, float)

    ready, ready_reason, table_z = descent_readiness(require_table=True)
    steps.append(("descent_readiness", ready, ready_reason))
    log(f"stage=descent_readiness ok={ready} detail={ready_reason}")
    if not ready or table_z is None:
        return _result(False, "descent safety calibration is incomplete", steps,
                       object_held=False, safe_to_retry=False, needs_operator=True)

    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    log(f"stage=preflight ok={pf.ok} detail={pf.reason}")
    if not pf.ok:
        return _result(False, "arm not ready", steps)

    own_eye = eye is None
    eye = eye or WristEye()
    try:
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            steps.append(("camera", False, str(exc)))
            return _result(False, f"wrist camera unavailable ({eye.camera_kind})", steps)

        if dry:
            # No motion allowed, so this can only report the view from where the arm
            # already is -- a --run pick additionally searches the other scan poses.
            det, why = eye.object(object_name, color, depth, intr)
            _show(color, depth, intr, tag="detect_object", det=det, tcp_cam=tcp_cam,
                  note=f"dry detect '{object_name}'")
            steps.append(("detect_from_current_pose", det is not None,
                          f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
            log("stage=dry_detection " + (
                f"ok=true source={det.source} depth_mm={det.z_mm:.0f}"
                if det else f"ok=false detail={why}"
            ))
            plan = _plan_aperture(det) if det is not None else None
            if plan is not None:
                steps.append(("depth_size_and_jaw_limits", bool(plan["ok"]),
                              str(plan["reason"])))
                log(f"stage=dry_jaw_plan ok={plan['ok']} detail={plan['reason']}")
            ready_to_run = det is not None and bool(plan and plan["ok"])
            return _result(ready_to_run,
                           "object localized and depth-sized within safe jaw limits; "
                           "safe to --run" if ready_to_run
                           else f"'{object_name}' not visible from the current pose "
                                "(a --run pick would also search the other scan poses)"
                           if det is None else str(plan["reason"]),
                           steps,
                           detection={
                               "source": det.source, "uv": det.uv, "z_mm": det.z_mm,
                               "min_width_mm": round(float(det.min_width_mm), 1),
                               "max_width_mm": round(float(det.max_width_mm), 1),
                           } if det and det.min_width_mm is not None
                           and det.max_width_mm is not None else None,
                           jaw_plan={
                               "max_aperture_mm": round(float(plan["max_aperture_mm"]), 1),
                               "preopen_mm": round(float(plan["preopen_mm"]), 1),
                               "preopen_deg": round(math.degrees(plan["preopen_rad"]), 1),
                               "grasp_mm": round(float(plan["grasp_mm"]), 1),
                               "grasp_deg": round(math.degrees(plan["grasp_rad"]), 1),
                               "contact_inference": False,
                           } if plan and plan["ok"] else None)

        log("stage=gripper_open starting")
        opened = _gripper("open")
        log(f"stage=gripper_open ok={opened.get('ok')} "
            f"position_deg={opened.get('position_deg')}")
        if not opened.get("ok"):
            steps.append(("open_gripper", False, str(opened.get("reason"))))
            return _result(False, f"gripper open failed: {opened.get('reason')}", steps,
                           object_held=False, safe_to_retry=False, needs_operator=True)
        log(f"stage=scan starting poses={len(_scan_poses())}")
        det, pose, why = _find_object(eye, object_name, log)
        steps.append(("scan_for_object", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm from "
                      f"{tuple(round(v, 3) for v in pose)}" if det else str(why)))
        if det is None:
            return _result(False, f"'{object_name}' not found from any scan pose", steps)
        log(f"stage=scan complete source={det.source} depth_mm={det.z_mm:.0f} "
            f"pose={tuple(round(v, 4) for v in pose)}")

        # Size the jaws to the measured object BEFORE approaching: an object that
        # cannot fit is refused here, having cost nothing but a look.
        jaw_plan = _plan_aperture(det)
        steps.append(("depth_size_and_jaw_limits", bool(jaw_plan["ok"]),
                      str(jaw_plan["reason"])))
        log(f"stage=jaw_sizing ok={jaw_plan['ok']} detail={jaw_plan['reason']}")
        if not jaw_plan["ok"]:
            _goto(*_scan_pose())
            return _result(False, str(jaw_plan["reason"]), steps,
                           object_size_mm=[round(det.min_width_mm or 0, 1),
                                           round(det.max_width_mm or 0, 1)])
        sized = _gripper("open", float(jaw_plan["preopen_rad"]))
        log(f"stage=jaw_presize ok={sized.get('ok')} "
            f"target_deg={sized.get('target_deg')} position_deg={sized.get('position_deg')}")
        if not sized.get("ok"):
            steps.append(("presize_gripper", False, str(sized.get("reason"))))
            return _result(False, f"gripper presize failed: {sized.get('reason')}",
                           steps, object_held=False, safe_to_retry=False,
                           needs_operator=True)

        log("stage=jacobian_learning starting axes=3")
        J, why_j = learn_jacobian(eye, object_name, log)
        steps.append(("learn_jacobian", J is not None, why_j))
        log(f"stage=jacobian_learning ok={J is not None} detail={why_j}")
        if J is None:
            _goto(*_scan_pose())
            return _result(False, f"visual servo unavailable: {why_j}", steps)
        Jinv = np.linalg.inv(J)

        # Where the object would appear if the TCP hovered APPROACH_M above it.
        target_cam = tcp_cam + APPROACH_M * J[:, 2]
        log(f"stage=visual_alignment starting tolerance_mm={ALIGN_TOL_MM:.0f}")
        ok, why_s = _servo_object_to(eye, object_name, Jinv, target_cam, tcp_cam,
                                     ALIGN_TOL_MM, log, tag="pick_align")
        steps.append(("align_over_object", ok, why_s))
        log(f"stage=visual_alignment ok={ok} detail={why_s}")
        if not ok:
            _goto(*_scan_pose())
            return _result(False, f"could not align over the object: {why_s}", steps)

        cur = tuple(A.ee_pos())
        log(f"stage=descent starting from_z={cur[2]:.4f} floor_z={table_z:.4f}")
        res = A.safe_descend(cur[0], cur[1], table_z)
        steps.append(("descend", res.ok, res.reason))
        log(f"stage=descent ok={res.ok} status={res.status} detail={res.reason}")
        if not res.ok:
            return _result(False, f"descent failed: {res.reason}", steps,
                           object_held=False, safe_to_retry=False, needs_operator=True,
                           descent_status=res.status)
        cur = res.reached or cur

        _OBJECT_HELD = None
        grasp_target_rad = float(jaw_plan["grasp_rad"])
        log(f"stage=gripper_position_close starting target_mm={jaw_plan['grasp_mm']:.1f} "
            f"target_deg={math.degrees(grasp_target_rad):.1f} contact_inference=false")
        close = _gripper("close", grasp_target_rad)
        time.sleep(0.8)
        log(f"stage=gripper_position_close_result ok={close.get('ok')} "
            f"target_deg={math.degrees(grasp_target_rad):.1f} "
            f"position_deg={close.get('position_deg')}")
        if not close.get("ok"):
            steps.append(("close_gripper", False, str(close.get("reason"))))
            return _result(False, f"gripper command failed: {close.get('reason')}", steps)
        steps.append(("position_close", True,
                      f"commanded measured width {jaw_plan['grasp_mm']:.1f}mm; "
                      "no contact inference"))

        carry_z = _carry_z(cur[0], cur[1])
        log(f"stage=lift starting target_z={carry_z:.4f}")
        lift = _goto(cur[0], cur[1], carry_z, cur=cur)
        steps.append(("lift_to_carry", lift.ok, lift.reason))
        log(f"stage=lift ok={lift.ok} detail={lift.reason}")
        if not lift.ok:
            return _result(False, f"object grasped but lift failed: {lift.reason}",
                           steps, verified=None, object_held=None,
                           safe_to_retry=False, needs_operator=True)
        held, held_reason, _ = _verify_held_at_tcp(eye, object_name, tcp_cam, log)
        _OBJECT_HELD = held
        steps.append(("visual_verify_after_lift", held, held_reason))
        log(f"stage=grasp_verification held={held} detail={held_reason}")
        return _result(bool(held), "picked" if held else "grasp not confirmed",
                       steps, verified=held, object_held=held,
                       safe_to_retry=held is False, needs_operator=held is None)
    except KeyboardInterrupt:
        held = _OBJECT_HELD
        home = _home_after_interrupt()
        homed = bool(home.get("ok"))
        steps.append(("interrupted", False,
                      f"active operation stopped; gripper open "
                      f"{'completed' if home.get('gripper_opened') else 'failed'}; "
                      f"home {'completed' if homed else 'failed'}"))
        return _result(
            False,
            "interrupted; gripper opened and arm homed"
            if homed and home.get("gripper_opened") else
            f"interrupted; home failed: {home.get('reason')}",
            steps,
            object_held=held,
            safe_to_retry=False,
            needs_operator=not homed or held is not False,
            status="interrupted",
            homed=homed,
            home=home,
            gripper_opened=bool(home.get("gripper_opened")),
            recommended_action="inspect object possession after homing",
        )
    finally:
        if own_eye:
            eye.close()


# =====================================================================================
# SKILL 2: place into the taught drop zone
# =====================================================================================
def place_in_drop_zone(*, dry: bool = False, verbose: bool = False,
                       held_confirmed: bool | None = None) -> dict[str, Any]:
    """Release the CURRENTLY held object at the taught drop-zone pose.

    pre:   arm ready, drop zone taught (``--teach-drop``), something held.
    act:   lift to carry height -> traverse above the drop zone -> lower onto it
           under torque gating -> open.
    post:  verify the jaws reached their open target (nothing jammed).
    reset: retreat to carry height.
    """
    global _OBJECT_HELD
    log = _printer(verbose)
    log(f"stage=place_start mode={'dry' if dry else 'live'}")
    steps: list[tuple[str, bool, str]] = []

    drop = load_drop_zone()
    steps.append(("drop_zone", drop is not None,
                  f"{tuple(round(v, 3) for v in drop)}m" if drop else
                  "missing -- run: python -m arm_agent.wrist_servo --teach-drop --run"))
    if drop is None:
        return _result(False, "drop zone not taught", steps)

    ready, ready_reason, _ = descent_readiness(require_table=False)
    steps.append(("descent_readiness", ready, ready_reason))
    log(f"stage=descent_readiness ok={ready} detail={ready_reason}")
    if not ready:
        return _result(False, "descent safety calibration is incomplete", steps,
                       object_held=_OBJECT_HELD, safe_to_retry=False, needs_operator=True)

    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    log(f"stage=preflight ok={pf.ok} detail={pf.reason}")
    if not pf.ok:
        return _result(False, "arm not ready", steps)

    dx, dy, dz = drop
    # Highest height IK actually allows ABOVE THE DROP ZONE -- taking a max() with
    # a fixed offset here would hand safe_goto a pose the gate refuses.
    carry = _carry_z(dx, dy)
    if dry:
        # Match the exact orientation _goto() will use live. The shared safety
        # gate's default pitch is intentionally different and can falsely reject
        # a wrist pose that is reachable at WORK_PITCH.
        above = A.check_ik(dx, dy, carry, 0.0, WORK_PITCH, 0.0)
        at = A.check_ik(dx, dy, dz, 0.0, WORK_PITCH, 0.0)
        steps.append(("ik_above_drop", above.ok, above.reason))
        steps.append(("ik_at_drop", at.ok, at.reason))
        log(f"stage=drop_ik above_ok={above.ok} at_ok={at.ok} "
            f"above={above.reason} at={at.reason}")
        return _result(above.ok and at.ok, "dry IK check of the drop zone", steps)

    has_object = _OBJECT_HELD if held_confirmed is None else bool(held_confirmed)
    steps.append(("object_held", has_object is True,
                  "confirmed" if has_object is True else
                  ("not held" if has_object is False else "unknown")))
    if has_object is not True:
        return _result(False, "place refused because a held object is not confirmed",
                       steps, object_held=has_object, safe_to_retry=False,
                       needs_operator=True)

    cur = tuple(A.ee_pos())
    log(f"stage=place_lift starting from={tuple(round(v, 4) for v in cur)}")
    lift = _goto(cur[0], cur[1], _carry_z(cur[0], cur[1]), cur=cur)
    steps.append(("lift_to_carry", lift.ok, lift.reason))
    if not lift.ok:
        return _result(False, f"could not lift before traversing: {lift.reason}", steps)

    r = _goto(dx, dy, carry)
    steps.append(("traverse_above_drop", r.ok, r.reason))
    if not r.ok:
        return _result(False, f"could not reach above the drop zone: {r.reason}", steps)
    log(f"stage=drop_traverse ok=true target=({dx:.3f},{dy:.3f},{carry:.3f})")

    # Torque-gated even though the height is taught: a bin rim or a stacked object
    # must stop the descent, not be pushed through.
    res = A.safe_descend(dx, dy, dz)
    steps.append(("lower_to_drop", res.ok, res.reason))
    log(f"stage=drop_descent ok={res.ok} status={res.status} detail={res.reason}")
    if not res.ok:
        return _result(False, f"drop descent failed: {res.reason}", steps,
                       object_held=True, safe_to_retry=False, needs_operator=True,
                       descent_status=res.status)
    landed = res.reached or (dx, dy, dz)

    _OBJECT_HELD = None
    log("stage=release starting")
    opened = _gripper("open")
    time.sleep(0.5)
    released = _verify_release(opened)
    _OBJECT_HELD = False if released is True else None
    log(f"stage=release_verification released={released} "
        f"position_deg={opened.get('position_deg')}")
    steps.append(("verify_release", bool(released),
                  "released" if released else ("jaws blocked" if released is False else "unknown")))

    retreat = _goto(landed[0], landed[1], carry, cur=landed)
    steps.append(("retreat", retreat.ok, retreat.reason))
    log(f"stage=retreat ok={retreat.ok} detail={retreat.reason}")
    ok = released is True and retreat.ok
    reason = ("placed" if ok else
              "release not confirmed" if released is not True else
              f"released but retreat failed: {retreat.reason}")
    return _result(ok, reason, steps, verified=released,
                   object_held=_OBJECT_HELD, safe_to_retry=False,
                   needs_operator=not ok)


def pick_and_place(object_name: str, *, dry: bool = False, verbose: bool = False,
                   auto_home: bool = True) -> dict[str, Any]:
    """The demo end to end: wrist-servo pick, release in the taught drop zone, home.

    Normal completion homes only once the object is released. Ctrl+C is the explicit
    operator override: it stops the active operation, opens the gripper, and homes.
    """
    global _OBJECT_HELD
    log = _printer(verbose)
    log(f"stage=pipeline_start task=pick_and_place object={object_name!r} "
        f"mode={'dry' if dry else 'live'}")
    try:
        picked = servo_pick(object_name, dry=dry, verbose=verbose)
    except KeyboardInterrupt:
        held = _OBJECT_HELD
        home = _home_after_interrupt()
        homed = bool(home.get("ok"))
        return {
            "ok": False,
            "status": "interrupted",
            "reason": "interrupted; gripper opened and arm homed"
                      if homed and home.get("gripper_opened") else
                      f"interrupted; home failed: {home.get('reason')}",
            "object_held": held,
            "safe_to_retry": False,
            "needs_operator": not homed or held is not False,
            "recommended_action": "inspect object possession after homing",
            "pick": None,
            "place": None,
            "homed": homed,
            "home": home,
            "gripper_opened": bool(home.get("gripper_opened")),
        }
    if not picked.get("ok"):
        interrupted = picked.get("status") == "interrupted"
        recovery = None
        if not dry and not interrupted:
            log("stage=pick_failure_recovery opening gripper, then homing")
            recovery = _home_after_interrupt()
            picked["failure_recovery"] = recovery
            picked["homed"] = bool(recovery.get("homed"))
            picked["gripper_opened"] = bool(recovery.get("gripper_opened"))
        return {"ok": False, "reason": f"pick failed: {picked.get('reason')}",
                "status": picked.get("status") if interrupted else None,
                "pick": picked, "place": None,
                "homed": bool(picked.get("homed")),
                "object_held": picked.get("object_held"),
                "failure_recovery": recovery}
    if dry:
        return {"ok": True, "reason": "dry run: pick validated", "pick": picked,
                "place": place_in_drop_zone(dry=True, verbose=verbose), "homed": False}
    try:
        placed = place_in_drop_zone(verbose=verbose, held_confirmed=True)
    except KeyboardInterrupt:
        held = _OBJECT_HELD
        home = _home_after_interrupt()
        homed = bool(home.get("ok"))
        return {
            "ok": False,
            "status": "interrupted",
            "reason": "interrupted during place; gripper opened and arm homed"
                      if homed and home.get("gripper_opened") else
                      f"interrupted during place; recovery incomplete: {home.get('reason')}",
            "object_held": held,
            "safe_to_retry": False,
            "needs_operator": True,
            "recommended_action": "inspect object possession after homing",
            "pick": picked,
            "place": None,
            "homed": homed,
            "home": home,
            "gripper_opened": bool(home.get("gripper_opened")),
        }

    homed: dict[str, Any] | None = None
    if auto_home and placed.get("ok"):
        log("  homing...")
        homed = _home()
        log(f"  home -> {'ok' if homed.get('ok') else homed.get('reason')}")
    return {"ok": bool(placed.get("ok")),
            "reason": placed.get("reason") if placed.get("ok") else
            f"place failed: {placed.get('reason')}",
            "pick": picked, "place": placed,
            "homed": bool(homed and homed.get("ok")),
            "object_held": placed.get("object_held"),
            "safe_to_retry": False,
            "needs_operator": not bool(placed.get("ok"))}


# =====================================================================================
# Look (no motion)
# =====================================================================================
def watch(object_name: str | None = None, *, seconds: float | None = None,
          verbose: bool = True) -> dict[str, Any]:
    """Stream the annotated wrist view until Ctrl+C. No motion, ever.

    This is the mode to leave running while aiming the camera or checking that an
    object is detected. ``--snapshot`` cannot do it: it grabs one frame and exits,
    and the MJPEG server dies with the process before a browser can connect.
    """
    if _VIZ is None:
        return {"ok": False, "reason": "run with --live (or --mjpeg PORT) to stream"}
    eye = WristEye()
    tcp = load_tcp()
    tcp_cam = np.asarray(tcp, float) if tcp else None
    log = _printer(verbose)
    frames, found = 0, 0
    deadline = None if seconds is None else time.time() + seconds
    last_report = 0.0
    latest_detection = "not requested" if not object_name else "not detected"
    interrupted = False
    log(f"stage=watch_start object={object_name!r} "
        "mode=viewer_only next_stage=none exit=Ctrl+C")
    try:
        while deadline is None or time.time() < deadline:
            try:
                color, depth, intr = eye.frame()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "reason": f"wrist camera unavailable: {exc}"}
            det = None
            if object_name:
                det, _ = eye.object(object_name, color, depth, intr)
                found += det is not None
                latest_detection = (
                    f"{det.source} depth_mm={det.z_mm:.0f} "
                    f"uv=({det.uv[0]:.0f},{det.uv[1]:.0f})"
                    if det else "not detected"
                )
            frames += 1
            _VIZ.show(color, depth, intr, tag="watch", save=False, det=det,
                      tcp_cam=tcp_cam, title="watch",
                      note=object_name or "aim the camera")
            now = time.monotonic()
            if now - last_report >= 2.0:
                rate = (100.0 * found / frames) if object_name and frames else 0.0
                log(f"stage=watch_progress frames={frames} detections={found} "
                    f"detection_rate={rate:.1f}% latest={latest_detection}")
                last_report = now
            time.sleep(0.03)
    except KeyboardInterrupt:
        interrupted = True
        log("stage=watch_stop reason=Ctrl+C")
    finally:
        eye.close()
    return {"ok": True, "frames": frames, "object": object_name,
            "detected_in_frames": found if object_name else None,
            "interrupted": interrupted,
            "tcp_calibrated": tcp is not None,
            "drop_zone_taught": load_drop_zone() is not None}


def look(object_name: str | None = None, *, pose_index: int = 0,
         verbose: bool = False, auto_home: bool = True) -> dict[str, Any]:
    """Move to a scan pose, then stream the wrist view until Ctrl+C.

    The "can it see the table from where it looks?" check. Uses exactly the pose
    servo_pick would scan from, so what you see here is what the pick will see.

    On exit -- including Ctrl+C -- the arm is parked at home rather than left
    hanging over the table. Pass ``auto_home=False`` to leave it where it is.
    """
    log = _printer(verbose)
    pf = A.preflight(connect=True)
    log(f"stage=look_preflight ok={pf.ok} detail={pf.reason}")
    if not pf.ok:
        return {"ok": False, "reason": f"arm not ready: {pf.reason}"}

    poses = _scan_poses()
    if not poses:
        return {"ok": False, "reason": "no reachable scan pose (check the workspace "
                                       "profile and REBOT_WRIST_PITCH)"}
    index = max(0, min(pose_index, len(poses) - 1))
    pose = poses[index]
    log(f"stage=look_move scan_poses={len(poses)} index={index} target="
        f"{tuple(round(v, 3) for v in pose)}")
    homed = False
    gripper_opened = False
    out: dict[str, Any] = {}
    try:
        r = _goto(*pose)
        log(f"stage=look_move_complete ok={r.ok} detail={r.reason}")
        if not r.ok:
            return {"ok": False, "reason": f"could not reach scan pose {pose}: {r.reason}",
                    "scan_poses": [[round(v, 4) for v in p] for p in poses]}
        time.sleep(0.5)
        log("stage=look_hold mode=viewer_only detail="
            "scan pose reached; detecting and streaming until Ctrl+C; "
            "pick pipeline will not run in --look mode")
        out = watch(object_name, verbose=verbose)
    except KeyboardInterrupt:
        out = {"ok": True, "reason": "interrupted", "interrupted": True}
    finally:
        # Ctrl+C is the operator override: open first, then home. Normal exit only
        # homes, so merely using a timed look does not alter the gripper.
        if auto_home:
            if out.get("interrupted"):
                log("  interrupted: opening gripper, then homing...")
                result = _home_after_interrupt()
                homed = bool(result.get("homed"))
                gripper_opened = bool(result.get("gripper_opened"))
            else:
                log("  homing...")
                result = _home()
                homed = bool(result.get("ok"))
            log(f"  home -> {'ok' if homed else result.get('reason')}")
    out["scan_pose"] = [round(v, 4) for v in pose]
    out["scan_pose_index"] = index
    out["scan_poses"] = [[round(v, 4) for v in p] for p in poses]
    out["homed"] = homed
    out["gripper_opened"] = gripper_opened
    return out


def snapshot(object_name: str | None = None) -> dict[str, Any]:
    """One annotated wrist frame -- for aiming the camera and checking detection."""
    eye = WristEye()
    try:
        color, depth, intr = eye.frame()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"wrist camera unavailable: {exc}"}
    try:
        tcp = load_tcp()
        det, why = (None, {})
        if object_name:
            det, why = eye.object(object_name, color, depth, intr)
        _show(color, depth, intr, tag="snapshot", det=det,
              tcp_cam=np.asarray(tcp, float) if tcp else None,
              note=object_name or "wrist view")
        out: dict[str, Any] = {
            "ok": True,
            "tcp_calibrated": tcp is not None,
            "drop_zone_taught": load_drop_zone() is not None,
            "valid_depth_px": int((depth > 0).sum()),
        }
        if object_name:
            out["found"] = det is not None
            out["detector"] = why.get("used")
            if det is not None:
                out.update({"uv": [round(v, 1) for v in det.uv], "z_mm": round(det.z_mm, 1),
                            "box": list(det.box)})
            else:
                out["why"] = why
        if _VIZ is not None:
            out["frames"] = str(_VIZ.outdir)
        return out
    finally:
        eye.close()


def grasp_preview(object_name: str) -> dict[str, Any]:
    """Camera-only ranked body cross-sections; never touches arm or gripper."""
    eye = WristEye()
    try:
        try:
            color, depth, intr = eye.frame()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"wrist camera unavailable: {exc}"}
        det, why = eye.object(object_name, color, depth, intr)
        if det is None:
            _show(color, depth, intr, tag="grasp_preview",
                  note=f"not found: {object_name}")
            return {"ok": False, "reason": "object not detected", "why": why,
                    "camera_only": True, "arm_commanded": False,
                    "gripper_commanded": False}
        geometry = _plan_scene_grasp_geometry(
            (color, depth, intr), det, current_rpy=(0.0, 0.0, 0.0),
            object_name=object_name,
        )
        selected = geometry.grasp_detection
        jaw_plan = _plan_aperture(selected) if geometry.ok and selected else None
        ranked = sorted(
            [dict(item) for item in geometry.candidates],
            key=lambda item: float(item.get("score", -1e9)), reverse=True,
        )
        _show(color, depth, intr, tag="grasp_preview", det=selected or det,
              candidates=ranked, title="camera-only dynamic grasp preview",
              note=geometry.reason)
        return {
            "ok": bool(geometry.ok and jaw_plan and jaw_plan.get("ok")),
            "reason": geometry.reason,
            "camera_only": True,
            "arm_commanded": False,
            "gripper_commanded": False,
            "identifier_detection": {
                "source": det.source, "uv": [round(v, 1) for v in det.uv],
                "z_mm": round(float(det.z_mm), 1),
            },
            "selected_grasp": geometry.summary().get("grasp_detection"),
            "ranked_cross_sections": ranked,
            "jaw_plan": jaw_plan,
            "frames": str(_VIZ.outdir) if _VIZ is not None else None,
        }
    finally:
        eye.close()


# =====================================================================================
# Offline simulation (no hardware)
# =====================================================================================
def run_sim(object_name: str = "red block", *, object_xy: tuple[float, float] | None = None,
            tcp_error_mm: float = 0.0, verbose: bool = True,
            viz: Any = None, trace: bool = False) -> dict[str, Any]:
    """Run the real pick+place loop against the synthetic world in ``wrist_sim``.

    Verifies the geometry end to end -- Jacobian signs, alignment target, step caps,
    grasp verification -- with no camera, arm or CAN present.
    """
    from arm_agent import wrist_sim

    world = wrist_sim.SimWorld()
    if object_xy is not None:
        world.object_xy = object_xy
    arm = wrist_sim.SimArm(world, trace=trace)
    original_a, original_factory, original_settle = A, _CAMERA_FACTORY, SETTLE_S
    if viz is not None:
        set_viz(viz)

    # The TCP the servo will use: ground truth from the mount, optionally perturbed
    # to check the loop tolerates an imperfect calibration.
    true_tcp = world.to_camera(world.ee)
    tcp = np.asarray(true_tcp, float) + np.array([tcp_error_mm, tcp_error_mm, 0.0])
    drop = (float(np.clip(0.24, arm.X_MIN, arm.X_MAX)),
            float(np.clip(-0.11, arm.Y_MIN, arm.Y_MAX)),
            float(world.table_z_m + 0.03))

    previous_env = {k: os.environ.get(k) for k in ("REBOT_WRIST_TCP", "REBOT_WRIST_DROP")}
    os.environ["REBOT_WRIST_TCP"] = ",".join(f"{v:.4f}" for v in tcp)
    os.environ["REBOT_WRIST_DROP"] = ",".join(f"{v:.4f}" for v in drop)
    install_backend(arm=arm, camera_factory=lambda kind: wrist_sim.SimCamera(world),
                    settle_s=0.0)
    start_xy = tuple(world.object_xy)
    try:
        out = pick_and_place(object_name, verbose=verbose)
        obj_final = world.object_center()
        out["sim"] = {
            "object_start_xy": [round(v, 4) for v in start_xy],
            "object_end_xy": [round(float(obj_final[0]), 4), round(float(obj_final[1]), 4)],
            "drop_zone_xy": [round(drop[0], 4), round(drop[1], 4)],
            "held": world.held,
            "arm_moves": arm.moves,
            "refused_moves": len(arm.refused),
            "tcp_error_mm": tcp_error_mm,
        }
        if arm.trace is not None:
            out["sim"]["trace"] = arm.trace
            out["sim"]["world"] = {
                "table_z_m": world.table_z_m,
                "object_radius_m": world.object_radius_m,
                "object_height_m": world.object_height_m,
                "drop_xyz_m": list(drop),
                "bounds": [arm.X_MIN, arm.X_MAX, arm.Y_MIN, arm.Y_MAX, arm.Z_MIN, arm.Z_MAX],
            }
        return out
    finally:
        install_backend(arm=original_a, camera_factory=original_factory, settle_s=original_settle)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# =====================================================================================
# CLI
# =====================================================================================
def main() -> None:
    global CAMERA_KIND, _INTERRUPT_HOME_ENABLED
    # Convert terminal/service shutdown signals into the same controlled recovery
    # path as Ctrl+C. SIGKILL and power loss are not catchable in software.
    def request_controlled_stop(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, request_controlled_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_controlled_stop)

    p = argparse.ArgumentParser(description="Eye-in-hand wrist-camera pick & place")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sim", action="store_true", help="full loop against the offline simulator")
    mode.add_argument("--snapshot", action="store_true", help="one annotated wrist frame (no motion)")
    mode.add_argument("--grasp-preview", metavar="OBJECT",
                      help="camera-only: rank graspable body cross-sections; "
                           "never connects to the arm or gripper")
    mode.add_argument("--calib-tcp", action="store_true", help="localize the TCP (jaws move only)")
    mode.add_argument("--teach-drop", action="store_true", help="record the release pose (no motion)")
    mode.add_argument("--show-calib", action="store_true", help="print the stored calibration")
    mode.add_argument("--state", action="store_true",
                      help="print joints, torques, tool pose and jaw angle (no motion)")
    mode.add_argument("--approach", metavar="OBJECT",
                      help="detect once, execute a prevalidated fixed near-touch trajectory, "
                           "then home; no scan/contact/gripper")
    mode.add_argument("--fixed-pick-place", metavar="OBJECT",
                      help="geometry-planned grasp with bounded wrist yaw, full-object "
                           "depth envelope, axial final approach, high release, then home")
    mode.add_argument("--fixed-pick", metavar="OBJECT",
                      help="run the proven pick and lift, then hold until Q or Ctrl+C "
                           "opens the gripper and homes")
    mode.add_argument("--pick", metavar="OBJECT", help="wrist-servo pick")
    mode.add_argument("--place", action="store_true", help="release into the taught drop zone")
    mode.add_argument("--pick-place", metavar="OBJECT", help="pick, then place in the drop zone")
    mode.add_argument("--home", action="store_true", help="park the arm at its home pose")
    mode.add_argument("--watch", action="store_true",
                      help="stream the live wrist view until Ctrl+C (no motion)")
    mode.add_argument("--look", action="store_true",
                      help="move to a scan pose and stream the view (does the arm see the table?)")
    mode.add_argument("--gripper", choices=("open", "close", "read"),
                      help="open/close the jaws, or 'read' to report position "
                           "WITHOUT moving or re-zeroing anything")
    mode.add_argument("--calib-grasp", metavar="OBJECT",
                      help="calibrate TCP and automatically close on the object "
                           "(use only with a sturdy object)")
    mode.add_argument("--calib-grasp-held", metavar="OBJECT",
                      help="preferred: calibrate TCP from an object already secured "
                           "in the jaws; camera-only, with no arm/gripper connection")
    mode.add_argument("--calib-grasp-target", metavar="OBJECT",
                      help="camera-only: save where the grasp feature appears when "
                           "manually centered inside the open jaw slot")
    mode.add_argument("--calib-jaw", metavar="OBJECT",
                      help="measure mm-per-radian by gripping an object (enables sized opening)")
    mode.add_argument("--set-tcp", metavar="X,Y,Z",
                      help="store a TCP-in-camera from CAD/measurement, in mm (no camera needed)")
    mode.add_argument("--check-tcp", metavar="OBJECT",
                      help="compare the stored TCP against a held object; does not overwrite")
    mode.add_argument("--set-jaw-span", type=float, metavar="MM",
                      help="store the jaw scale from a measured gap (no camera needed)")
    mode.add_argument("--set-jaw-point", metavar="DEG,MM",
                      help="add one ruler-measured empty-jaw calibration point; "
                           "saves only and never commands the gripper")

    p.add_argument("--object", help="object name for --snapshot")
    p.add_argument("--place-reference",
                   help="--fixed-pick-place: visible landmark for relative placing")
    p.add_argument("--place-relation", default="drop_zone",
                   choices=("drop_zone", "free_space", "left", "right",
                            "front", "behind", "next_to"),
                   help="--fixed-pick-place destination relation in robot/table frame")
    p.add_argument("--place-distance-m", type=float, default=None,
                   help="optional landmark separation, clamped to 0.06-0.18m")
    p.add_argument("--drop", help="explicit 'x,y,z' metres for --teach-drop")
    p.add_argument("--to-deg", type=float, default=None, metavar="DEG",
                   help="--gripper: command an explicit jaw angle in degrees "
                        "instead of the open/close limits")
    p.add_argument("--gripper-kp", type=float, default=None,
                   help="MIT kp for this jaw move (config uses 50; try 8 or 2)")
    p.add_argument("--gripper-kd", type=float, default=None,
                   help="MIT kd for this jaw move (RobStride encodes 0-5)")
    p.add_argument("--empty-jaws", action="store_true",
                   help="assert no object is between the jaws; required for an "
                        "explicit live gripper-only positioning command")
    p.add_argument("--scan-index", type=int, default=0,
                   help="--look: which scan pose (0 = centre, 1-4 = quadrants)")
    p.add_argument("--at-fraction", type=float, default=None,
                   help="--set-jaw-span: travel fraction the gap was measured at "
                        "(default: the open cap, i.e. after --gripper open)")
    p.add_argument("--camera", default=CAMERA_KIND, choices=("realsense", "orbbec", "auto"))
    p.add_argument("--run", action="store_true", help="authorize real motion")
    p.add_argument("--dry", action="store_true", help="plan + validate only, never move")
    p.add_argument("--view", action="store_true", help="save annotated frames")
    p.add_argument("--mjpeg", type=int, metavar="PORT", help="also stream frames to a browser")
    p.add_argument("--live", action="store_true",
                   help="show the live wrist feed in a browser (implies --view; port 8731)")
    p.add_argument("--no-home", action="store_true",
                   help="do not auto-home after a completed pick & place")
    p.add_argument("--tcp-error-mm", type=float, default=0.0, help="--sim: perturb the TCP")
    p.add_argument("--object-xy", help="--sim: object position 'x,y' in metres")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if args.run and args.no_home:
        p.error("--no-home is disabled for live RS motion")
    if args.run and (args.pick or args.place or args.pick_place):
        p.error("live grasp/place is temporarily safety-locked; use --approach OBJECT "
                "to validate slow IK-only motion without descent or gripper commands")
    if args.run and args.gripper in {"open", "close"} and not args.empty_jaws:
        p.error("live gripper commands are temporarily safety-locked while the "
                "IK-only approach path is being validated; for ruler calibration "
                "remove every object and pass --empty-jaws")
    if args.run and args.look:
        p.error("scan-pose motion is temporarily disabled; use --watch for a "
                "camera-only view or --approach OBJECT for current-view IK motion")
    _INTERRUPT_HOME_ENABLED = bool(args.run and (
        args.approach or args.fixed_pick or args.fixed_pick_place
        or args.pick or args.place
        or args.pick_place or args.look or args.home
        or args.gripper in {"open", "close"}
        or args.calib_grasp or args.calib_jaw or args.calib_tcp or args.check_tcp
    ))
    CAMERA_KIND = args.camera
    global GRIPPER_KP, GRIPPER_KD
    if args.gripper_kp is not None:
        GRIPPER_KP = str(args.gripper_kp)
    if args.gripper_kd is not None:
        GRIPPER_KD = str(args.gripper_kd)
    port = args.mjpeg or (8731 if args.live else None)
    if args.view or port:
        set_viz(make_viz(mjpeg_port=port))
        print(f"[viz] frames -> {DEBUG_DIR}")
        if port:
            print(f"[viz] LIVE wrist feed -> http://127.0.0.1:{port}/   (open in a browser)")

    verbose = not args.quiet

    if args.sim:
        xy = _parse_triplet((args.object_xy + ",0") if args.object_xy else None)
        out = run_sim(args.object or "red block",
                      object_xy=(xy[0], xy[1]) if xy else None,
                      tcp_error_mm=args.tcp_error_mm, verbose=verbose)
    elif args.show_calib:
        from arm_agent import gripper_model as _gm
        out = {"tcp_camera_xyz_mm": load_tcp(), "tcp_file": str(TCP_FILE),
               "grasp_target_camera_xyz_mm": load_grasp_target(),
               "grasp_target_detail": load_grasp_target_detail(),
               "grasp_target_file": str(GRASP_TARGET_FILE),
               "release_xyz_m": load_drop_zone(), "drop_file": str(DROP_FILE),
               "safety": load_safety_calibration(), "safety_file": str(SAFETY_FILE),
               "camera": CAMERA_KIND, "gripper": _gm.describe()}
    elif args.snapshot:
        out = snapshot(args.object)
    elif args.grasp_preview:
        out = grasp_preview(args.grasp_preview)
    elif args.watch:
        if not (args.live or args.mjpeg):
            p.error("--watch streams video; add --live (or --mjpeg PORT)")
        print("[watch] streaming... press Ctrl+C to stop")
        out = watch(args.object, verbose=verbose)
    elif args.state:
        from arm_agent import gripper_model as _gmp
        st = _local_state()
        if not st:
            out = {
                "ok": False,
                "reason": _LAST_LOCAL_STATE_ERROR
                          or "no state (is the arm connected? is can0 up?)",
            }
        else:
            joints = st.get("joints_rad") or []
            torque = st.get("joint_torque") or []
            ee = st.get("end_effector", {})
            grip = st.get("gripper_position_rad")
            out = {
                "ok": True,
                "joints_deg": [round(math.degrees(v), 2) for v in joints],
                "joint_torque_nm": [round(float(v), 3) for v in torque],
                "torque_all_zero": all(abs(float(v)) < 1e-6 for v in torque) if torque else None,
                "tool_position_m": [round(float(v), 4) for v in ee.get("position_m", [])],
                "tool_rpy_rad": [round(float(v), 4) for v in ee.get("rpy_rad", [])],
                "gripper_deg": None if grip is None else round(math.degrees(grip), 2),
                "note": "read-only: nothing moved. If torque_all_zero stays true while "
                        "you press on the arm, safe_descend has no contact signal.",
            }
    elif args.gripper == "read":
        from arm_agent import gripper_model as _gmp
        state = _local_state() or {}
        pos = state.get("gripper_position_rad")
        out = {"ok": pos is not None, "position_rad": pos}
        if pos is not None:
            out["position_deg"] = round(math.degrees(pos), 2)
        else:
            out["reason"] = "no gripper position read-back"
        out["note"] = ("read-only: nothing moved and no zero was changed. "
                       "Move the jaws BY HAND and read again to see which way "
                       "the value goes.")
    elif args.gripper and args.dry:
        from arm_agent import gripper_model as _gmp
        tgt = (math.radians(args.to_deg) if args.to_deg is not None
               else _gmp.target_for(args.gripper, _wrist_limits()))
        clamped = _gmp.clamp_rad(tgt, _wrist_limits())
        out = {"ok": True, "action": args.gripper, "dry_run": True,
               "requested_deg": round(math.degrees(tgt), 2),
               "would_command_deg": round(math.degrees(clamped), 2),
               "limits_deg": [round(math.degrees(v), 2) for v in _wrist_limits()],
               "note": "nothing was commanded"}
    elif args.gripper:
        if not args.run:
            p.error("--gripper moves the jaws; pass --run to authorize it "
                    "(or --dry to see the target without moving)")
        from arm_agent import gripper_model as _gmp
        explicit = (math.radians(args.to_deg) if args.to_deg is not None else None)
        before = None
        state = _local_state()
        if state:
            before = state.get("gripper_position_rad")
        resp = _gripper(args.gripper, explicit)
        pos, tgt = resp.get("position_rad"), resp.get("target_pos_rad")
        from arm_agent import gripper_model as _gm
        out = dict(resp)
        if pos is not None:
            out["position_deg"] = round(math.degrees(float(pos)), 2)
            if tgt is not None:
                out["target_deg"] = round(math.degrees(float(tgt)), 2)
                out["short_of_target_deg"] = round(math.degrees(float(pos) - float(tgt)), 2)
        if before is not None and pos is not None:
            delta = float(pos) - float(before)
            out["before_rad"] = round(float(before), 4)
            out["delta_rad"] = round(delta, 4)
            out["delta_deg"] = round(math.degrees(delta), 2)
            if abs(delta) > 0.01:
                out["direction"] = ("OPENED" if delta > 0 else "CLOSED") + \
                    " (opening is the +rad direction on this gripper)"
        if pos is not None and tgt is not None:
            out["reached_target"] = abs(float(pos) - float(tgt)) < 0.05
            out["note"] = (
                "jaw position read-back only; tracking error is not interpreted "
                "as object contact"
            )
    elif args.fixed_pick:
        if not (args.run or args.dry):
            p.error("pass --run for fixed pick/hold, or --dry for camera/jaw checks")
        out = (
            run_fixed_pick_cli(args.fixed_pick, verbose=verbose)
            if args.run else
            fixed_pick(args.fixed_pick, dry=True, verbose=verbose)
        )
    elif args.fixed_pick_place:
        if not (args.run or args.dry):
            p.error("pass --run for fixed pick/place, or --dry for camera/jaw checks")
        out = fixed_pick_and_place(
            args.fixed_pick_place, dry=args.dry, verbose=verbose,
            place_reference=args.place_reference,
            place_relation=args.place_relation,
            place_distance_m=args.place_distance_m,
        )
    elif args.approach:
        if not (args.run or args.dry):
            p.error("pass --run for the fixed safe trajectory, or --dry for camera-only validation")
        out = servo_approach(args.approach, dry=args.dry, verbose=verbose)
    elif args.look:
        if not args.run:
            p.error("--look moves the arm to a scan pose; pass --run to authorize it")
        if not (args.live or args.mjpeg):
            p.error("--look streams video; add --live (or --mjpeg PORT)")
        print("[look] viewer-only mode: moving to a scan pose, detecting continuously, "
              "and waiting there until Ctrl+C. It will not pick.", flush=True)
        out = look(args.object, pose_index=args.scan_index, verbose=verbose,
                   auto_home=not args.no_home)
    elif args.calib_grasp:
        out = calibrate_tcp_by_grasp(args.calib_grasp, verbose=verbose)
    elif args.calib_grasp_held:
        out = calibrate_tcp_by_grasp(args.calib_grasp_held, close_first=False,
                                     verbose=verbose)
    elif args.calib_grasp_target:
        out = calibrate_grasp_target(args.calib_grasp_target, verbose=verbose)
    elif args.calib_jaw:
        out = calibrate_jaw_scale(args.calib_jaw, verbose=verbose)
    elif args.set_tcp:
        xyz = _parse_triplet(args.set_tcp)
        if xyz is None:
            p.error("--set-tcp needs 'x,y,z' in millimetres")
        out = {"ok": True, "tcp_camera_xyz_mm": list(xyz),
               "saved": str(save_tcp(xyz, {"method": "declared"}))}
    elif args.check_tcp:
        out = check_tcp(args.check_tcp, verbose=verbose)
    elif args.set_jaw_span is not None:
        from arm_agent import gripper_model as _gm
        out = _gm.set_span(args.set_jaw_span, args.at_fraction)
    elif args.set_jaw_point:
        from arm_agent import gripper_model as _gm
        try:
            angle_deg, aperture_mm = (
                float(value) for value in args.set_jaw_point.split(",")
            )
        except (TypeError, ValueError):
            p.error("--set-jaw-point needs 'DEG,MM', for example '60,18.5'")
        out = _gm.add_calibration_point(angle_deg, aperture_mm)
    elif args.calib_tcp:
        out = calibrate_tcp(args.camera)
    elif args.teach_drop:
        xyz = _parse_triplet(args.drop)
        if xyz is None and not args.run:
            p.error("--teach-drop reads the live arm pose; pass --run (or give --drop x,y,z)")
        out = teach_drop_zone(xyz)
    elif args.pick:
        if not (args.run or args.dry):
            p.error("pass --run to move the arm, or --dry to validate only")
        out = servo_pick(args.pick, dry=args.dry, verbose=verbose)
    elif args.place:
        if not (args.run or args.dry):
            p.error("pass --run to move the arm, or --dry to validate only")
        out = place_in_drop_zone(dry=args.dry, verbose=verbose)
    elif args.home:
        if not args.run:
            p.error("--home moves the arm; pass --run to authorize it")
        out = _home()
    else:
        if not (args.run or args.dry):
            p.error("pass --run to move the arm, or --dry to validate only")
        out = pick_and_place(args.pick_place, dry=args.dry, verbose=verbose,
                             auto_home=not args.no_home)

    print(json.dumps(out, indent=2, default=str))
    if args.run and (args.approach or args.fixed_pick
                     or args.fixed_pick_place) and not out.get("homed"):
        _hold_unparked_controller(str(out.get("reason", "fixed operation did not park")))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if not _INTERRUPT_HOME_ENABLED:
            print(json.dumps({
                "ok": False, "status": "interrupted",
                "reason": "camera/read-only operation stopped; no arm was commanded",
                "gripper_opened": False, "homed": False,
            }, indent=2))
            raise SystemExit(130)
        approach_mode = "--approach" in sys.argv
        recovery = _home_after_interrupt(open_gripper=not approach_mode)
        out = {
            "ok": False, "status": "interrupted", "reason": recovery.get("reason"),
            "gripper_opened": recovery.get("gripper_opened"),
            "homed": recovery.get("homed"), "recovery": recovery,
        }
        print(json.dumps(out, indent=2, default=str))
        if not out.get("homed"):
            _hold_unparked_controller("KeyboardInterrupt recovery failed")
    except Exception as exc:
        if not _INTERRUPT_HOME_ENABLED:
            raise
        approach_mode = "--approach" in sys.argv
        recovery = _home_after_interrupt(open_gripper=not approach_mode)
        out = {
            "ok": False, "status": "unhandled_exception",
            "reason": f"{type(exc).__name__}: {exc}",
            "gripper_opened": recovery.get("gripper_opened"),
            "homed": recovery.get("homed"), "recovery": recovery,
        }
        print(json.dumps(out, indent=2, default=str))
        if not out.get("homed"):
            _hold_unparked_controller(f"{type(exc).__name__}: {exc}")
        raise
