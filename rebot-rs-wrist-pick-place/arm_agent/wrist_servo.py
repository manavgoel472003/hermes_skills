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
import itertools
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
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
from arm_agent import object_geometry_3d as OG3
from arm_agent import wrist_detect as WD
from arm_agent import voxel_navigation as VN
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
HYBRID_CHECKPOINT_M = float(os.getenv(
    "REBOT_WRIST_VISIBLE_CHECKPOINT_M", "0.090"
))
HYBRID_MAX_REPLANS = int(os.getenv(
    "REBOT_WRIST_VISIBLE_CHECKPOINT_REPLANS", "1"
))
FIXED_PICK_LIFT_M = float(os.getenv("REBOT_WRIST_PICK_LIFT_M", "0.050"))
DYNAMIC_GRASP_ENABLED = os.getenv("REBOT_DYNAMIC_GRASP", "0").strip().lower() \
    in {"1", "true", "yes", "on"}
FIXED_SCENE_TRAJECTORY = os.getenv(
    "REBOT_FIXED_SCENE_TRAJECTORY", "0"
).strip().lower() in {"1", "true", "yes", "on"}
START_GRIPPER_CLOSED = os.getenv(
    "REBOT_WRIST_CLOSE_GRIPPER_AT_START", "0"
).strip().lower() in {"1", "true", "yes", "on"}
DYNAMIC_REMEASURE_MAX_DELTA_MM = float(
    os.getenv("REBOT_DYNAMIC_REMEASURE_MAX_DELTA_MM", "6.0")
)
DYNAMIC_WIDTH_CONSENSUS_FRAMES = int(
    os.getenv("REBOT_DYNAMIC_WIDTH_CONSENSUS_FRAMES", "5")
)
DYNAMIC_WIDTH_CONSENSUS_MIN_FRAMES = int(
    os.getenv("REBOT_DYNAMIC_WIDTH_CONSENSUS_MIN_FRAMES", "3")
)
DYNAMIC_WIDTH_CONSENSUS_MAX_SPREAD_MM = float(
    os.getenv("REBOT_DYNAMIC_WIDTH_CONSENSUS_MAX_SPREAD_MM", "4.0")
)
DYNAMIC_MICRO_LIFT_M = float(os.getenv("REBOT_DYNAMIC_MICRO_LIFT_M", "0.015"))
DYNAMIC_HOLD_TOL_MM = float(os.getenv("REBOT_DYNAMIC_HOLD_TOL_MM", "6.0"))
DYNAMIC_SLOT_TOL_MM = float(os.getenv("REBOT_DYNAMIC_SLOT_TOL_MM", "6.0"))
DYNAMIC_SLOT_BOUNDARY_SLOP_MM = float(os.getenv(
    "REBOT_DYNAMIC_SLOT_BOUNDARY_SLOP_MM", "0.5"
))
DYNAMIC_SLOT_DEPTH_SLACK_MM = float(os.getenv(
    "REBOT_DYNAMIC_SLOT_DEPTH_SLACK_MM", "3.0"
))
DYNAMIC_SLOT_DEPTH_BOUNDARY_SLOP_MM = float(os.getenv(
    "REBOT_DYNAMIC_SLOT_DEPTH_BOUNDARY_SLOP_MM", "0.5"
))
DYNAMIC_BODY_MAX_CORRECTIONS = int(os.getenv(
    "REBOT_DYNAMIC_BODY_MAX_CORRECTIONS", "2"
))
DYNAMIC_RETRY_CLOSE_MM = float(os.getenv("REBOT_DYNAMIC_RETRY_CLOSE_MM", "1.0"))
DYNAMIC_RETRY_CLOSE_DEG = float(os.getenv("REBOT_DYNAMIC_RETRY_CLOSE_DEG", "5.0"))
DYNAMIC_MAX_RETRIES = int(os.getenv("REBOT_DYNAMIC_MAX_RETRIES", "3"))
DYNAMIC_CLOSE_STEP_DEG = float(os.getenv(
    "REBOT_DYNAMIC_GRIPPER_CLOSE_STEP_DEG",
    # Compatibility with the older internal name. Launch profiles have always
    # used the explicit gripper-prefixed form.
    os.getenv("REBOT_DYNAMIC_CLOSE_STEP_DEG", "15.0"),
))


def _object_geometry_mode() -> str | None:
    """Select the object-box pipeline without changing legacy defaults."""
    mode = os.getenv("REBOT_OBJECT_GEOMETRY_MODE", "legacy").strip().lower()
    return mode if mode in {"legacy", "hybrid"} else None


def _hybrid_box_selection_policy(
    extent_consistent: bool, *, place_after_pick: bool,
    scene_collision_enabled: bool,
) -> str:
    """Choose persistent versus fresh geometry without weakening placement."""
    if extent_consistent:
        return "registered_multiview"
    if place_after_pick or scene_collision_enabled:
        return "reject"
    return "fresh_stationary_pick_only"


def _stationary_box_is_long_axis_truncation(
    hybrid_box: dict[str, Any], stationary_box: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Recognize partial long-axis visibility without hiding a wrong object.

    Either the registered sweep or the stationary close view can lose an end
    of a long object.  The earlier test only handled the stationary view being
    shorter, which incorrectly rejected the inverse (common when a marker is
    clipped during the small discovery sweep).  Cross-section, base-frame XY
    centre, and independent multi-frame evidence must still agree.
    """
    try:
        hybrid_extent = np.sort(np.asarray(
            hybrid_box["observed_size_m"], dtype=float
        ))
        stationary_extent = np.sort(np.asarray(
            stationary_box["observed_size_m"], dtype=float
        ))
        hybrid_center = np.asarray(hybrid_box["center_m"], dtype=float)
        stationary_center = np.asarray(stationary_box["center_m"], dtype=float)
        hybrid_views = int(hybrid_box.get("views", 0))
        stationary_views = int(stationary_box.get("views", 0))
        if hybrid_extent.shape != (3,) or stationary_extent.shape != (3,) \
                or hybrid_center.shape != (3,) \
                or stationary_center.shape != (3,) \
                or not np.all(np.isfinite(hybrid_extent)) \
                or not np.all(np.isfinite(stationary_extent)):
            raise ValueError("invalid box arrays")
    except (KeyError, TypeError, ValueError) as exc:
        return False, {"reason": f"invalid box for truncation test: {exc}"}

    cross_limit = np.maximum(
        0.006, 0.30 * np.maximum(hybrid_extent[:2], stationary_extent[:2])
    )
    cross_section_consistent = bool(np.all(
        np.abs(hybrid_extent[:2] - stationary_extent[:2]) <= cross_limit
    ))
    longer_source = (
        "registered_multiview"
        if hybrid_extent[2] >= stationary_extent[2]
        else "fresh_stationary"
    )
    longer_axis = float(max(hybrid_extent[2], stationary_extent[2]))
    shorter_axis = float(min(hybrid_extent[2], stationary_extent[2]))
    long_axis_loss = longer_axis - shorter_axis
    long_axis_truncated = bool(
        longer_axis >= 0.050
        and long_axis_loss >= max(0.020, 0.20 * longer_axis)
    )
    center_xy_error = float(np.linalg.norm(
        hybrid_center[:2] - stationary_center[:2]
    ))
    center_consistent = center_xy_error <= 0.025
    multiframe_evidence = hybrid_views >= 2 and stationary_views >= 2
    accepted = bool(
        cross_section_consistent and long_axis_truncated
        and center_consistent and multiframe_evidence
    )
    return accepted, {
        "reason": (
            "the two multi-frame boxes preserve the measured cross-section "
            f"but one clips the long axis; retained {longer_source}"
            if accepted else
            "box mismatch is not isolated to partial long-axis visibility"
        ),
        "hybrid_extent_mm": (hybrid_extent * 1000.0).round(1).tolist(),
        "stationary_extent_mm": \
            (stationary_extent * 1000.0).round(1).tolist(),
        "cross_section_consistent": cross_section_consistent,
        "long_axis_loss_mm": round(long_axis_loss * 1000.0, 1),
        "longer_source": longer_source,
        "center_xy_error_mm": round(center_xy_error * 1000.0, 1),
        "multiframe_evidence": multiframe_evidence,
    }


GEOMETRY_MAX_YAW_DELTA_DEG = float(
    os.getenv("REBOT_WRIST_GRASP_MAX_YAW_DELTA_DEG", "16.0")
)
GEOMETRY_YAW_MEASUREMENT_SLOP_DEG = float(
    os.getenv(
        "REBOT_WRIST_GRASP_YAW_MEASUREMENT_SLOP_DEG",
        # Backward compatibility for older launch scripts.  The longer name
        # makes clear that this is tolerance for measured object-axis noise,
        # not permission to exceed the physical wrist-yaw command bound.
        os.getenv("REBOT_WRIST_GRASP_YAW_SLOP_DEG", "0.5"),
    )
)
GEOMETRY_MAX_RESIDUAL_YAW_DEG = float(
    os.getenv("REBOT_WRIST_GRASP_MAX_RESIDUAL_YAW_DEG", "0.0")
)
GEOMETRY_APERTURE_DERIVED_RESIDUAL = os.getenv(
    "REBOT_WRIST_GRASP_APERTURE_DERIVED_RESIDUAL", "0"
).strip().lower() in {"1", "true", "yes", "on"}
GEOMETRY_FINAL_APPROACH_MM = float(
    os.getenv("REBOT_WRIST_GRASP_FINAL_APPROACH_MM", "40.0")
)
GEOMETRY_MIN_FINAL_APPROACH_MM = float(
    os.getenv("REBOT_WRIST_GRASP_MIN_FINAL_APPROACH_MM", "20.0")
)
GEOMETRY_RPY_STEP_DEG = float(
    os.getenv("REBOT_WRIST_GRASP_RPY_STEP_DEG", "2.0")
)
REACH_PITCH_BIAS_DEG = float(
    os.getenv("REBOT_WRIST_REACH_PITCH_BIAS_DEG", "10.0")
)
GEOMETRY_MIN_FEATURE_PROMINENCE_MM = float(
    os.getenv("REBOT_WRIST_GRASP_MIN_FEATURE_PROMINENCE_MM", "25.0")
)
GRASP_MIN_OPEN_CLEARANCE_MM = float(
    os.getenv("REBOT_GRASP_MIN_OPEN_CLEARANCE_MM", "8.0")
)
GRASP_MIN_END_CLEARANCE_MM = float(
    os.getenv("REBOT_GRASP_MIN_END_CLEARANCE_MM", "12.0")
)
GRASP_MAX_MAJOR_MM = float(
    os.getenv("REBOT_GRASP_MAX_MAJOR_MM", "150.0")
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
# Payload geometry is substantially less certain than the robot URDF and the
# measured scene.  Keep it independently switchable so scene-relative runs can
# still protect every arm/gripper link while payload-map fusion is being
# validated.  The generic default remains conservative; the scene wrapper
# explicitly opts into arm/end-effector-only collision checking.
PAYLOAD_COLLISION_ENABLED = os.getenv(
    "REBOT_PAYLOAD_COLLISION", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SCENE_COLLISION_VOXEL_M = float(
    os.getenv("REBOT_SCENE_COLLISION_VOXEL_M", "0.020")
)
SCENE_COLLISION_INFLATION_M = float(
    os.getenv("REBOT_SCENE_COLLISION_INFLATION_M", "0.003")
)
# Payload uncertainty uses the same bounded clearance requested for the scene.
# The previous hard-coded 15mm per side was applied on top of already-inflated
# scene boxes and more than doubled the effective clearance.
PAYLOAD_COLLISION_PADDING_M = float(os.getenv(
    "REBOT_PAYLOAD_COLLISION_PADDING_M",
    str(SCENE_COLLISION_INFLATION_M),
))
SCENE_COLLISION_STRIDE_PX = int(
    os.getenv("REBOT_SCENE_COLLISION_STRIDE_PX", "4")
)
SCENE_COLLISION_TARGET_PAD_PX = int(
    os.getenv("REBOT_SCENE_COLLISION_TARGET_PAD_PX", "20")
)
SCENE_COLLISION_EXPORT_VIEW = os.getenv(
    "REBOT_SCENE_COLLISION_EXPORT_VIEW", "0"
).strip().lower() in {"1", "true", "yes", "on"}
# Individual registered camera views are useful for registration debugging, but
# rendering all of them beside the fused cloud makes one physical surface look
# like several obstacles.  Keep the collision viewer focused on the data the
# planner actually uses; the per-view overlays remain explicitly opt-in.
SCENE_COLLISION_EXPORT_PER_VIEW = os.getenv(
    "REBOT_SCENE_EXPORT_PER_VIEW_CLOUDS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SCENE_COLLISION_EXPORT_BOX_LABELS = os.getenv(
    "REBOT_SCENE_EXPORT_BOX_LABELS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SCENE_DENSE_REGISTRATION_MIN_OVERLAP = float(np.clip(float(os.getenv(
    "REBOT_SCENE_DENSE_REGISTRATION_MIN_OVERLAP", "0.15"
)), 0.10, 0.60))
SCENE_DENSE_REGISTRATION_MAX_TRANSLATION_M = float(np.clip(float(os.getenv(
    "REBOT_SCENE_DENSE_REGISTRATION_MAX_TRANSLATION_M", "0.012"
)), 0.0, 0.015))
ALIGN_TOL_MM = 8.0       # camera-frame object<->target error we call "aligned"
STALL_ITERS = 3          # no improvement for this many steps => at the sensor's noise floor
GAIN = 0.6               # servo step = GAIN * Jinv @ error (then step-capped)
MAX_ITERS = 24           # floor; the real budget is derived from the workspace
SETTLE_S = 0.35          # let the arm settle before re-observing
LOST_RETRIES = 3         # re-grabs before declaring the object lost mid-servo
GRASP_GAP_RAD = 0.09     # gripper stopped >this short of full close => holding
GRIPPER_NEAR_RAD = 0.12  # gripper within this of the open target => released
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
             err_mm=None, note=None, title=None, candidates=None,
             selected_uv=None, object_box_edges_uv=None, decision_lines=None):
    """Draw what the wrist servo sees: object, the fixed TCP, and the hover target."""
    vis = color.copy()
    if det is not None:
        mask = getattr(det, "info", {}).get("mask_u8")
        if mask is not None:
            mask_bool = np.asarray(mask) > 0
            if mask_bool.shape == vis.shape[:2]:
                tint = np.zeros_like(vis)
                tint[:] = (40, 120, 40)
                vis[mask_bool] = cv2.addWeighted(
                    vis[mask_bool], 0.60, tint[mask_bool], 0.40, 0.0
                )
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
    if selected_uv is not None:
        selected = tuple(int(round(float(value))) for value in selected_uv)
        cv2.drawMarker(
            vis, selected, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2
        )
        cv2.putText(vis, "selected grasp", (selected[0] + 10, selected[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2)
    if object_box_edges_uv:
        for first, second in object_box_edges_uv:
            a = tuple(int(round(float(value))) for value in first)
            b = tuple(int(round(float(value))) for value in second)
            cv2.line(vis, a, b, (255, 0, 255), 2)
        cv2.putText(vis, "metric 3D OBB", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 2)
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
    if decision_lines:
        for index, line in enumerate(decision_lines):
            y = 72 + index * 20
            cv2.rectangle(vis, (4, y - 15),
                          (min(vis.shape[1] - 4, 620), y + 4), (0, 0, 0), -1)
            cv2.putText(vis, str(line), (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (255, 255, 255), 1)
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
_ACTIVE_SCENE_MAP: SC.SceneMap | None = None
_VOXEL_ROUTE_VIZ_SEQUENCE = 0
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


def _project_object_box_edges(
    box: dict[str, Any], *, ee_xyz_m: Any, camera_mapping: Any,
    tool_camera_mm: Any, intr: Any,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Project a base-frame metric OBB into its capture camera image."""
    center = np.asarray(box["center_m"], dtype=float)
    size = np.asarray(box["observed_size_m"], dtype=float)
    axes = np.asarray(box["axes_base"], dtype=float).T
    ee = np.asarray(ee_xyz_m, dtype=float)
    mapping = np.asarray(camera_mapping, dtype=float)
    tool = np.asarray(tool_camera_mm, dtype=float)
    if center.shape != (3,) or size.shape != (3,) or axes.shape != (3, 3):
        raise ValueError("invalid object box projection geometry")
    signs = list(itertools.product((-1.0, 1.0), repeat=3))
    pixels = []
    for sign in signs:
        corner_base = center + axes @ (0.5 * size * np.asarray(sign))
        corner_camera = tool - mapping @ (corner_base - ee)
        uv = WD.project(corner_camera, intr)
        if uv is None:
            raise ValueError("3D box corner projects behind the camera")
        pixels.append(tuple(float(value) for value in uv))
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            differing_axes = sum(
                signs[first][axis] != signs[second][axis] for axis in range(3)
            )
            if differing_axes == 1:
                edges.append((pixels[first], pixels[second]))
    return edges


def _save_grasp_fit_diagnostic(
    frame: tuple[np.ndarray, np.ndarray, Any], geometry: GG.GraspGeometry,
    jaw_plan: dict[str, Any], reason: str, *, name: str,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Save the exact image-space and metric geometry behind a fit refusal."""
    if _VIZ is None:
        return {"ok": False, "reason": "visualizer unavailable"}
    color, depth, intr = frame
    grasp = geometry.grasp_detection
    envelope = geometry.envelope_detection or grasp
    if grasp is None or envelope is None:
        return {"ok": False, "reason": "grasp geometry unavailable"}
    box_detail = None
    box_edges = None
    box_reason = "metric box unavailable"
    try:
        target = load_grasp_target()
        table_z = load_safety_calibration().get("table_z_m")
        current_rpy = tuple(float(value) for value in A.ee_rpy())
        mapping, mapping_detail = load_saved_camera_jacobian(current_rpy)
        ee_xyz = np.asarray(A.ee_pos(), dtype=float)
        if target is None or table_z is None or mapping is None:
            raise ValueError(
                str(mapping_detail.get("reason") if mapping is None else
                    "visual target or table calibration unavailable")
            )
        points = OG3.masked_points_base(
            frame, envelope, ee_xyz_m=ee_xyz,
            camera_mapping_mm_per_m=mapping,
            tool_camera_mm=target,
            near_mm=float(WD.NEAR_MM), far_mm=float(WD.FAR_MM),
        )
        box_detail = OG3.fit_tabletop_box(
            [points], table_z_m=float(table_z)
        ).as_dict()
        box_edges = _project_object_box_edges(
            box_detail, ee_xyz_m=ee_xyz, camera_mapping=mapping,
            tool_camera_mm=target, intr=intr,
        )
        box_reason = (
            "measured-only RGB-D OBB projected in magenta; "
            f"{box_detail.get('orientation_mode')}; no table completion"
        )
    except Exception as exc:  # noqa: BLE001
        box_reason = f"3D OBB unavailable: {type(exc).__name__}: {exc}"
    selected = getattr(grasp, "info", {}).get("selected_candidate", {})
    selected_uv = selected.get("uv") if isinstance(selected, dict) else None
    width = float(grasp.min_width_mm or 0.0)
    aperture = float(jaw_plan.get("max_aperture_mm", 0.0))
    required = width + max(0.0, GRASP_MIN_OPEN_CLEARANCE_MM)
    decision_lines = [
        f"selected local width={width:.1f}mm + clearance="
        f"{GRASP_MIN_OPEN_CLEARANCE_MM:.1f}mm => required={required:.1f}mm",
        f"measured gripper maximum={aperture:.1f}mm; decision=REFUSE",
        "blue circles=other cross-sections; red X=selected section",
        box_reason,
    ]
    rejected_pixels = getattr(envelope, "info", {}).get(
        "depth_rejected_pixels"
    )
    retained_pixels = getattr(envelope, "info", {}).get(
        "depth_component_pixels"
    )
    if rejected_pixels is not None and retained_pixels is not None:
        decision_lines.append(
            f"RGB-D cleanup retained {int(retained_pixels)} target pixels; "
            f"rejected {int(rejected_pixels)} leaked pixels"
        )
    if box_detail is not None:
        observed_mm = np.asarray(box_detail["observed_size_m"]) * 1000.0
        decision_lines.append(
            f"3D OBB observed dimensions="
            f"{[round(float(value), 1) for value in observed_mm]}mm"
        )
    safe_name = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in name
    )
    index = int(getattr(_VIZ, "_n", 0))
    tag = f"grasp_fit_{safe_name}"
    rendered = _VIZ.show(
        color, depth, intr, tag=tag, save=True,
        det=envelope, candidates=geometry.candidates,
        selected_uv=selected_uv, object_box_edges_uv=box_edges,
        decision_lines=decision_lines,
        note=f"FIT REFUSED: {reason}",
    )
    path = str(Path(_VIZ.outdir) / f"{index:03d}_{tag}.jpg")
    latest_path = Path(_VIZ.outdir) / "latest_grasp_fit_decision.jpg"
    cv2.imwrite(str(latest_path), rendered)
    log(
        "stage=grasp_fit_diagnostic saved=" + path + " "
        f"selected_width_mm={width:.1f} required_mm={required:.1f} "
        f"max_aperture_mm={aperture:.1f} box3d={box_detail is not None}"
    )
    return {
        "ok": True, "image": path, "latest_image": str(latest_path),
        "view": name,
        "selected_width_mm": round(width, 2),
        "required_open_mm": round(required, 2),
        "max_aperture_mm": round(aperture, 2),
        "object_box_3d": box_detail,
        "box_reason": box_reason,
    }


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


def _payload_collision_box_plan(
    detection: Any,
    camera_mapping: np.ndarray,
    tcp_camera_mm: Any,
    ee_rpy: Any,
    *,
    padding_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return payload size, long axis and measured centre in the EE frame."""
    minor_mm = float(detection.min_width_mm)
    major_mm = float(detection.max_width_mm or minor_mm)
    if not (math.isfinite(minor_mm) and math.isfinite(major_mm)) \
            or minor_mm <= 0.0 or major_mm <= 0.0:
        raise ValueError("payload dimensions are missing or non-positive")
    mapping = np.asarray(camera_mapping, dtype=float)
    if mapping.shape != (3, 3) or not np.all(np.isfinite(mapping)) \
            or float(np.linalg.cond(mapping)) > 80.0:
        raise ValueError("camera mapping is invalid or ill-conditioned")
    angle = math.radians(float(
        detection.info.get("long_axis_deg") or 0.0
    ))
    camera_axis = np.array([math.cos(angle), math.sin(angle), 0.0])
    base_axis = np.linalg.solve(mapping, -1000.0 * camera_axis)
    base_axis /= max(1e-9, float(np.linalg.norm(base_axis)))
    ee_rotation = _rpy_rotation_matrix(ee_rpy)
    long_axis_ee = ee_rotation.T @ base_axis

    object_center_cam = np.asarray(detection.p_cam_mm, dtype=float)
    tcp_cam = np.asarray(tcp_camera_mm, dtype=float)
    if object_center_cam.shape != (3,) or tcp_cam.shape != (3,):
        raise ValueError("payload centre and TCP must be 3D camera points")
    center_offset_cam_mm = object_center_cam - tcp_cam
    center_offset_base_m = np.linalg.solve(mapping, -center_offset_cam_mm)
    center_offset_ee_m = ee_rotation.T @ center_offset_base_m
    if not np.all(np.isfinite(center_offset_ee_m)) or \
            float(np.linalg.norm(center_offset_ee_m)) > 0.20:
        raise ValueError(
            "measured payload centre is inconsistent with camera/TCP calibration"
        )

    bounded_padding = float(np.clip(padding_m, 0.0, 0.015))
    payload_size = np.array([
        minor_mm / 1000.0 + 2.0 * bounded_padding,
        minor_mm / 1000.0 + 2.0 * bounded_padding,
        major_mm / 1000.0 + 2.0 * bounded_padding,
    ])
    return payload_size, long_axis_ee, center_offset_ee_m


def load_saved_camera_jacobian(
    current_rpy: tuple[float, float, float],
    *, maximum_orientation_delta_deg: float | None = None,
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
    # Do not let this post-orientation gate contradict the bounded grasp
    # planner. The rigid camera mount is rotated analytically below, so the
    # configured grasp envelope is also the default camera-geometry envelope.
    configured_allowed = (
        float(os.getenv(
            "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG",
            str(GEOMETRY_MAX_YAW_DELTA_DEG),
        ))
        if maximum_orientation_delta_deg is None
        else float(maximum_orientation_delta_deg)
    )
    # Ordinary visual servoing retains its existing 25deg ceiling. A caller
    # may explicitly request a larger bounded envelope for a fully preplanned
    # collision-escape grasp; the rigid mount transform is rotated
    # analytically, and that path still needs complete IK/collision validation.
    hard_ceiling_deg = 25.0 if maximum_orientation_delta_deg is None else 45.0
    allowed_deg = float(np.clip(
        configured_allowed, 0.25, hard_ceiling_deg
    ))
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


def _camera_bounded_pitch_bias(
    base_rpy: tuple[float, float, float], requested_bias_deg: float,
) -> tuple[tuple[float, float, float], float, dict[str, Any]]:
    """Spend only the pitch still available inside the camera-fit envelope.

    Discovery pitch and geometry-derived yaw can both already consume part of
    the saved camera calibration's orientation budget. Search downward from
    the requested reach bias and retain a small margin for hardware tracking
    error, rather than rotating first and failing the camera gate afterward.
    """
    requested = float(np.clip(requested_bias_deg, 0.0, 15.0))
    allowed_deg = float(np.clip(float(os.getenv(
        "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG",
        str(GEOMETRY_MAX_YAW_DELTA_DEG),
    )), 0.25, 25.0))
    margin_deg = min(0.75, max(0.1, allowed_deg * 0.1))
    maximum_delta = allowed_deg - margin_deg
    base = np.asarray(base_rpy, dtype=float)
    # Quarter-degree resolution is much smaller than the 2deg executed RPY
    # steps while avoiding an unnecessarily conservative whole-degree loss.
    trial_biases = np.arange(requested, -0.001, -0.25)
    last_detail: dict[str, Any] = {}
    for bias_deg in trial_biases:
        candidate = base.copy()
        candidate[1] += math.radians(float(bias_deg))
        mapping, detail = load_saved_camera_jacobian(
            tuple(float(v) for v in candidate)
        )
        last_detail = detail
        delta_deg = float(detail.get("orientation_delta_deg", math.inf))
        if mapping is not None and delta_deg <= maximum_delta:
            return (
                tuple(float(v) for v in candidate),
                float(bias_deg),
                {
                    **detail,
                    "requested_bias_deg": requested,
                    "applied_bias_deg": float(bias_deg),
                    "reserved_margin_deg": margin_deg,
                },
            )
    return tuple(float(v) for v in base), 0.0, {
        **last_detail,
        "requested_bias_deg": requested,
        "applied_bias_deg": 0.0,
        "reserved_margin_deg": margin_deg,
        "reason": (
            "no adaptive pitch fits inside the saved camera-orientation "
            f"envelope: {last_detail.get('reason', 'calibration unavailable')}"
        ),
    }


def _camera_safe_yaw_bounds(
    current_rpy: tuple[float, float, float], hard_limit_deg: float,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    """Yaw correction remaining after a look-around pose used RPY budget."""
    try:
        data = json.loads(CAMERA_ROTATION_FILE.read_text())
        calibrated_rpy = np.asarray(
            data["calibrated_tool_rpy_rad"], dtype=float
        )
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"saved camera orientation unavailable: {exc}"}
    if calibrated_rpy.shape != (3,):
        return None, {"reason": "saved camera orientation has invalid dimensions"}
    allowed_deg = float(np.clip(float(os.getenv(
        "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG",
        str(GEOMETRY_MAX_YAW_DELTA_DEG),
    )), 0.25, 25.0))
    # Retain a small deterministic margin for measured pose tracking. The
    # existing vision slop may absorb this amount; larger disagreement refuses.
    margin_deg = min(0.25, allowed_deg * 0.05)
    usable_deg = allowed_deg - margin_deg
    hard = max(0.0, float(hard_limit_deg))
    candidates = np.linspace(-hard, hard, max(2, int(40 * hard) + 1))
    calibrated_rotation = _rpy_rotation_matrix(calibrated_rpy)
    base = np.asarray(current_rpy, dtype=float)
    valid: list[float] = []
    for correction_deg in candidates:
        candidate = base.copy()
        candidate[2] += math.radians(float(correction_deg))
        relative = calibrated_rotation.T @ _rpy_rotation_matrix(candidate)
        delta_deg = math.degrees(math.acos(float(np.clip(
            (np.trace(relative) - 1.0) / 2.0, -1.0, 1.0
        ))))
        if delta_deg <= usable_deg + 1e-9:
            valid.append(float(correction_deg))
    if not valid:
        return None, {
            "reason": "current look-around orientation exhausted the saved "
                      "camera calibration budget",
            "allowed_orientation_delta_deg": allowed_deg,
            "reserved_margin_deg": margin_deg,
        }
    bounds = (min(valid), max(valid))
    return bounds, {
        "reason": "remaining yaw derived from the shared camera RPY envelope",
        "yaw_bounds_deg": [round(value, 3) for value in bounds],
        "allowed_orientation_delta_deg": allowed_deg,
        "reserved_margin_deg": margin_deg,
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
    _identity_filters: dict[str, Any] = field(default_factory=dict, repr=False)

    def frame(self):
        # RealSense wait_for_frames must have exactly one caller at a time. The
        # continuous browser feed and detector share this lock and this stream.
        with self._frame_lock:
            if self._camera is None:
                self._camera = _CAMERA_FACTORY(self.camera_kind)
            return self._camera.frame()

    def preflight(self) -> tuple[np.ndarray, np.ndarray, Any]:
        """Require one complete RGB-D frame before any hardware may move.

        Starting the MJPEG pump only proves that a Python thread was created;
        it does not prove that the physical camera exists.  This synchronous
        read keeps a missing/unresponsive wrist camera on the fail-closed side
        of the arm-motion boundary.
        """
        color, depth, intr = self.frame()
        if not isinstance(color, np.ndarray) or color.ndim != 3 \
                or color.shape[2] != 3 or color.size == 0:
            raise RuntimeError("wrist camera returned an invalid colour frame")
        if not isinstance(depth, np.ndarray) or depth.ndim != 2 \
                or depth.size == 0 or depth.shape != color.shape[:2]:
            raise RuntimeError("wrist camera returned an invalid depth frame")
        if intr is None or len(intr) != 4:
            raise RuntimeError("wrist camera returned invalid intrinsics")
        return color, depth, intr

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
        found, why = WD.detect(
            color,
            depth,
            intr,
            description,
            allow_vlm=self.allow_vlm,
            **bounds,
        )
        identity_filter = self._identity_filters.get(
            description.strip().lower()
        )
        if found is not None and callable(identity_filter):
            accepted, identity_detail = identity_filter(
                found, (color, depth, intr)
            )
            why = {**why, "identity_gate": identity_detail}
            if not accepted:
                why["used"] = None
                return None, why
        return found, why

    def set_identity_filter(
        self, description: str, identity_filter: Any | None
    ) -> None:
        """Install a task-local named-object identity gate."""
        key = description.strip().lower()
        if identity_filter is None:
            self._identity_filters.pop(key, None)
        else:
            self._identity_filters[key] = identity_filter


class CameraFrameUnavailable(RuntimeError):
    """A bounded synchronous RGB-D retry window was exhausted."""


_SEMANTIC_IDENTITY_SOURCES = {
    "sam3", "grounding_dino_sam2", "locate_anything"
}


@dataclass
class _HybridTargetIdentity:
    """Keep one named target attached to a semantic 3D anchor.

    Semantic vision is the only authority allowed to create the anchor. Cheap
    colour/depth fallbacks may subsequently support tracking, but only when
    their deprojected point maps back to the same bounded base-frame cluster.
    They can neither create nor relocate target identity.
    """

    description: str
    tool_camera_mm: np.ndarray
    log: Callable[[str], None]
    anchor_base_m: np.ndarray | None = None
    semantic_points_base_m: list[np.ndarray] = field(default_factory=list)
    rgbd_candidate_points_base_m: list[np.ndarray] = field(default_factory=list)
    anchor_kind: str | None = None
    accepted: int = 0
    rejected: int = 0

    @property
    def radius_m(self) -> float:
        return float(np.clip(float(os.getenv(
            "REBOT_HYBRID_TARGET_CLUSTER_RADIUS_M", "0.075"
        )), 0.025, 0.120))

    @property
    def anchor_attempts(self) -> int:
        return int(np.clip(int(os.getenv(
            "REBOT_HYBRID_SEMANTIC_ANCHOR_ATTEMPTS", "8"
        )), 3, 15))

    @property
    def rgbd_consensus_frames(self) -> int:
        return int(np.clip(int(os.getenv(
            "REBOT_HYBRID_RGBD_ANCHOR_FRAMES", "3"
        )), 3, 5))

    @property
    def rgbd_consensus_spread_m(self) -> float:
        return float(np.clip(float(os.getenv(
            "REBOT_HYBRID_RGBD_ANCHOR_SPREAD_M", "0.015"
        )), 0.005, 0.025))

    def _point_base(self, detection: Any) -> tuple[np.ndarray | None, str]:
        rpy = tuple(float(value) for value in A.ee_rpy())
        mapping, detail = load_saved_camera_jacobian(rpy)
        if mapping is None:
            return None, str(detail.get("reason"))
        mapping = np.asarray(mapping, dtype=float)
        ee_xyz = np.asarray(A.ee_pos(), dtype=float)
        p_cam = np.asarray(detection.p_cam_mm, dtype=float)
        if mapping.shape != (3, 3) or ee_xyz.shape != (3,) \
                or p_cam.shape != (3,) \
                or not np.all(np.isfinite(mapping)) \
                or not np.all(np.isfinite(ee_xyz)) \
                or not np.all(np.isfinite(p_cam)):
            return None, "camera/base target geometry is invalid"
        try:
            point_base = ee_xyz + np.linalg.solve(
                mapping, self.tool_camera_mm - p_cam
            )
        except np.linalg.LinAlgError:
            return None, "camera/base target mapping is singular"
        if not np.all(np.isfinite(point_base)):
            return None, "mapped target point is non-finite"
        return point_base, "mapped"

    def reanchor_from_metric_box(
        self, object_box_3d: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Replace a partial-mask centroid with a verified metric-box centre.

        Long objects can be clipped in the initial wrist frame, so the first
        semantic mask centroid is not a stable physical point.  Once repeated
        native-mask depth views establish a metric OBB, its centre is the
        orientation-independent identity anchor.  The new centre must still
        lie inside the original semantic cluster; this cannot relocate target
        identity to a different object.
        """
        try:
            center = np.asarray(object_box_3d["center_m"], dtype=float)
            size = np.asarray(
                object_box_3d["observed_size_m"], dtype=float
            )
            views = int(object_box_3d.get("views", 0))
        except (KeyError, TypeError, ValueError):
            return False, {"reason": "metric object box is malformed"}
        if center.shape != (3,) or size.shape != (3,) \
                or not np.all(np.isfinite(center)) \
                or not np.all(np.isfinite(size)) \
                or np.any(size <= 0.0):
            return False, {"reason": "metric object box is non-finite"}
        if views < 2:
            return False, {
                "reason": (
                    f"metric object box has only {views} target view; "
                    "two are required to update identity"
                )
            }
        if self.anchor_base_m is None:
            return False, {"reason": "semantic target anchor is unavailable"}
        previous = np.asarray(self.anchor_base_m, dtype=float)
        shift_m = float(np.linalg.norm(center - previous))
        if shift_m > self.radius_m:
            return False, {
                "reason": (
                    f"metric box centre is {shift_m * 1000.0:.1f}mm from "
                    f"the semantic anchor, beyond "
                    f"{self.radius_m * 1000.0:.1f}mm"
                ),
                "previous_anchor_base_m": previous.round(4).tolist(),
                "candidate_center_base_m": center.round(4).tolist(),
            }
        self.anchor_base_m = center.copy()
        # Do not let historical partial-mask centroids pull the robust centre
        # back toward the clipped end on the next median update.
        self.semantic_points_base_m = [center.copy()]
        self.anchor_kind = "verified_metric_obb_center"
        return True, {
            "reason": (
                "semantic identity reanchored to the verified multi-view "
                "metric object-box centre"
            ),
            "shift_mm": round(shift_m * 1000.0, 1),
            "previous_anchor_base_m": previous.round(4).tolist(),
            "anchor_base_m": center.round(4).tolist(),
            "box_observed_mm": (size * 1000.0).round(1).tolist(),
            "box_views": views,
            "cluster_radius_mm": round(self.radius_m * 1000.0, 1),
        }

    def __call__(
        self, detection: Any, _frame: tuple[np.ndarray, np.ndarray, Any]
    ) -> tuple[bool, dict[str, Any]]:
        source = str(getattr(detection, "source", ""))
        semantic = source in _SEMANTIC_IDENTITY_SOURCES
        point_base, mapping_reason = self._point_base(detection)
        if point_base is None:
            self.rejected += 1
            detail = {
                "accepted": False, "source": source,
                "reason": mapping_reason,
            }
            self.log(
                "stage=hybrid_target_identity accepted=false "
                f"source={source} detail={mapping_reason}"
            )
            return False, detail
        if self.anchor_base_m is None:
            rgbd_object = source == "rgbd:colored_object"
            if rgbd_object:
                self.rgbd_candidate_points_base_m.append(point_base)
                recent = self.rgbd_candidate_points_base_m[
                    -self.rgbd_consensus_frames:
                ]
                if len(recent) >= self.rgbd_consensus_frames:
                    points = np.vstack(recent)
                    median = np.median(points, axis=0)
                    spread_m = float(np.max(np.linalg.norm(
                        points - median, axis=1
                    )))
                    if spread_m <= self.rgbd_consensus_spread_m:
                        self.anchor_base_m = median
                        self.anchor_kind = "stable_rgbd_colored_object"
                        self.accepted += 1
                        detail = {
                            "accepted": True, "source": source,
                            "reason": (
                                "stable table-plane RGB-D colored-object "
                                "anchor established"
                            ),
                            "stable_frames": len(recent),
                            "spread_mm": round(spread_m * 1000.0, 1),
                            "anchor_base_m": median.round(4).tolist(),
                            "cluster_radius_mm": round(
                                self.radius_m * 1000.0, 1
                            ),
                        }
                        self.log(
                            "stage=hybrid_target_identity accepted=true "
                            f"source={source} action=rgbd_consensus_anchor "
                            f"stable_frames={len(recent)} "
                            f"spread_mm={spread_m * 1000.0:.1f}"
                        )
                        return True, detail
                    # Do not let old samples from a different physical cluster
                    # accumulate until they accidentally satisfy consensus.
                    self.rgbd_candidate_points_base_m = [point_base]
                self.rejected += 1
                reason = (
                    "waiting for stable table-plane RGB-D colored-object "
                    f"consensus {len(self.rgbd_candidate_points_base_m)}/"
                    f"{self.rgbd_consensus_frames}"
                )
                detail = {
                    "accepted": False, "source": source, "reason": reason,
                    "candidate_base_m": point_base.round(4).tolist(),
                }
                self.log(
                    "stage=hybrid_target_identity accepted=false "
                    f"source={source} detail={reason}"
                )
                return False, detail
            if not semantic:
                self.rejected += 1
                reason = "semantic target anchor has not been established"
                detail = {
                    "accepted": False, "source": source,
                    "reason": reason,
                    "candidate_base_m": point_base.round(4).tolist(),
                }
                self.log(
                    "stage=hybrid_target_identity accepted=false "
                    f"source={source} detail={reason}"
                )
                return False, detail
            self.semantic_points_base_m.append(point_base)
            self.anchor_base_m = point_base.copy()
            self.anchor_kind = "semantic"
            self.accepted += 1
            depth_basis = str(
                (getattr(detection, "info", {}) or {}).get(
                    "depth_basis", "legacy_detection_depth"
                )
            )
            detection_depth_mm = float(getattr(
                detection, "z_mm",
                np.asarray(detection.p_cam_mm, dtype=float)[2],
            ))
            detail = {
                "accepted": True, "source": source,
                "reason": "semantic target anchor established",
                "anchor_base_m": self.anchor_base_m.round(4).tolist(),
                "cluster_radius_mm": round(self.radius_m * 1000.0, 1),
                "depth_basis": depth_basis,
            }
            self.log(
                "stage=hybrid_target_identity accepted=true "
                f"source={source} action=anchor base_m="
                f"{self.anchor_base_m.round(4).tolist()} "
                f"depth_mm={detection_depth_mm:.1f} "
                f"depth_basis={depth_basis}"
            )
            return True, detail
        distance_m = float(np.linalg.norm(point_base - self.anchor_base_m))
        if distance_m > self.radius_m:
            self.rejected += 1
            reason = (
                f"candidate is {distance_m * 1000.0:.1f}mm from semantic "
                f"anchor, beyond {self.radius_m * 1000.0:.1f}mm"
            )
            detail = {
                "accepted": False, "source": source, "reason": reason,
                "distance_from_anchor_mm": round(distance_m * 1000.0, 1),
            }
            self.log(
                "stage=hybrid_target_identity accepted=false "
                f"source={source} detail={reason}"
            )
            return False, detail
        if semantic:
            self.semantic_points_base_m.append(point_base)
            recent = self.semantic_points_base_m[-5:]
            self.anchor_base_m = np.median(np.vstack(recent), axis=0)
        self.accepted += 1
        detail = {
            "accepted": True, "source": source,
            "reason": "candidate belongs to semantic target cluster",
            "distance_from_anchor_mm": round(distance_m * 1000.0, 1),
            "anchor_base_m": self.anchor_base_m.round(4).tolist(),
            "cluster_radius_mm": round(self.radius_m * 1000.0, 1),
        }
        self.log(
            "stage=hybrid_target_identity accepted=true "
            f"source={source} distance_mm={distance_m * 1000.0:.1f}"
        )
        return True, detail


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


def _minimum_commandable_aperture_mm() -> float | None:
    """Return the calibrated jaw gap at the immutable safe close command.

    Object eligibility must follow the gripper's measured capability.  The old
    fixed 8 mm object-size floor rejected narrow objects even though the hard
    15 degree command limit still leaves only about a 5 mm gap.  This helper
    keeps the decision tied to both the current calibration and the same hard
    angle clamp used by every real gripper command.
    """
    from arm_agent import gripper_model as gm

    close_rad, open_rad = _wrist_limits()
    safe_close_rad = gm.target_for("close", (close_rad, open_rad))
    return gm.aperture_mm(safe_close_rad)


def _plan_aperture(det: Any) -> dict[str, Any]:
    """Plan pre-open and final grasp positions from measured depth size only.

    No force, motor-current, or encoder-stall contact inference is used. If the
    metric jaw calibration or a plausible object width is unavailable, refuse.
    """
    from arm_agent import gripper_model as gm

    max_ap = gm.max_aperture_mm(_wrist_limits())
    min_ap = _minimum_commandable_aperture_mm()
    base = {"ok": False, "preopen_rad": None, "grasp_rad": None,
            "min_width_mm": None, "max_width_mm": None,
            "max_aperture_mm": max_ap,
            "min_commandable_aperture_mm": min_ap}
    if max_ap is None or min_ap is None:
        return {**base, "reason": "jaw millimetre calibration is missing; refusing to pick"}
    if det is None or det.min_width_mm is None or det.max_width_mm is None:
        return {**base, "reason": "detector did not produce a reliable metric object size; "
                                  "refusing to pick"}
    narrow, wide = float(det.min_width_mm), float(det.max_width_mm or det.min_width_mm)
    base.update(min_width_mm=narrow, max_width_mm=wide)
    if not (math.isfinite(narrow) and math.isfinite(wide)) or narrow <= 0 or wide < narrow:
        return {**base, "reason": f"invalid detected size {narrow:.1f}x{wide:.1f}mm; "
                                  "refusing to pick"}
    if narrow <= min_ap:
        return {**base, "reason": f"object is only {narrow:.1f}mm across; the "
                                  f"smallest safe commanded jaw opening is "
                                  f"{min_ap:.1f}mm"}
    # Position-only closing has no force/contact inference. Keep the requested
    # undersize correction tightly bounded even when an environment value is
    # mistyped. Four millimetres is the tested maximum: one millimetre tighter
    # than the previous profile, while all calibrated angle/hard-limit gates
    # below remain unchanged.
    compression_mm = float(np.clip(SIZE_COMPRESSION_MM, 0.0, 4.0))
    raw_grasp_request_mm = (
        narrow + SIZE_SAFETY_MARGIN_MM - compression_mm
    )
    # Never request an aperture below the immutable angle-bounded close
    # position. For narrow but still graspable objects, reduce the requested
    # compression automatically instead of rejecting them or commanding the
    # motor against its stop.
    grasp_request_mm = max(min_ap, raw_grasp_request_mm)
    if grasp_request_mm <= 0:
        return {**base, "reason": f"computed grasp target "
                                  f"{grasp_request_mm:.1f}mm is not positive; "
                                  "refusing to close"}
    if grasp_request_mm > max_ap:
        return {**base, "reason": f"object is {narrow:.1f}mm across and the conservative "
                                  f"{SIZE_SAFETY_MARGIN_MM:.1f}mm margin requires "
                                  f"{grasp_request_mm:.1f}mm, beyond the "
                                  f"{max_ap:.1f}mm jaw limit"}

    # Open generously for approach. The final position is the measured width
    # minus only the bounded, angle-representable compression computed above.
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
    effective_compression_mm = max(
        0.0, narrow + SIZE_SAFETY_MARGIN_MM - float(actual_grasp_mm)
    )
    if effective_compression_mm > 1e-6:
        grasp_basis = (
            f"{effective_compression_mm:.1f}mm smaller than the detected "
            f"width; {effective_compression_mm:.1f}mm planned position "
            "compression"
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
            "planned_compression_mm": effective_compression_mm}


def _validate_object_gripper_fit(
    geometry: GG.GraspGeometry, jaw_plan: dict[str, Any],
    object_box_3d: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate a complete object independently of its semantic class.

    The long dimension may protrude along the fingers; only the selected local
    cross-section belongs between the jaws.  Still require room to enter around
    that section, adequate material on both sides of the grasp point, and a
    bounded payload length before committing to an approach.
    """
    grasp = getattr(geometry, "grasp_detection", None)
    envelope = getattr(geometry, "envelope_detection", None) or grasp
    if grasp is None or envelope is None:
        return False, {"reason": "complete grasp envelope is unavailable"}
    try:
        width_mm = float(grasp.min_width_mm)
        major_mm = float(envelope.max_width_mm or grasp.max_width_mm or width_mm)
        major_basis = "current_depth_envelope"
        box_dimensions_mm = None
        if object_box_3d is not None:
            box_dimensions_mm = np.asarray(
                object_box_3d["conservative_size_m"], dtype=float
            ) * 1000.0
            if box_dimensions_mm.shape != (3,) or not np.all(
                np.isfinite(box_dimensions_mm) & (box_dimensions_mm > 0)
            ):
                raise ValueError("invalid conservative 3D box")
            box_major_mm = float(np.max(box_dimensions_mm))
            # When close-view segmentation spills into the table, hybrid mode
            # deliberately retains the registered multi-view semantic shape
            # and updates only its position from repeated semantic centroids.
            # In that provenance state, letting the rejected close-view depth
            # envelope enlarge the object defeats the fallback (for example a
            # verified 109 mm marker became a fictitious 266 mm payload).
            # Keep the conservative max() behavior for every other path.
            registered_shape_retained = bool(
                (getattr(grasp, "info", {}) or {}).get("position_basis")
                == "stable_semantic_identifier_registered_box"
            )
            if registered_shape_retained:
                major_mm = box_major_mm
                major_basis = "registered_multiview_metric_box"
            else:
                major_mm = max(major_mm, box_major_mm)
                major_basis = "max_current_envelope_and_metric_box"
        aperture_mm = float(jaw_plan["max_aperture_mm"])
        preopen_mm = float(jaw_plan["preopen_mm"])
    except (KeyError, TypeError, ValueError):
        return False, {"reason": "object or jaw dimensions are incomplete"}
    if not all(math.isfinite(value) and value > 0 for value in (
        width_mm, major_mm, aperture_mm, preopen_mm
    )):
        return False, {"reason": "object or jaw dimensions are invalid"}
    required_open_mm = width_mm + max(0.0, GRASP_MIN_OPEN_CLEARANCE_MM)
    if required_open_mm > aperture_mm + 1e-6:
        return False, {
            "reason": (
                f"local grasp width {width_mm:.1f}mm needs at least "
                f"{required_open_mm:.1f}mm opening including clearance, beyond "
                f"the {aperture_mm:.1f}mm jaw limit"
            ),
            "width_mm": width_mm,
            "major_mm": major_mm,
            "box_dimensions_mm": None if box_dimensions_mm is None else
                box_dimensions_mm.round(1).tolist(),
            "major_basis": major_basis,
        }
    if major_mm > max(1.0, GRASP_MAX_MAJOR_MM):
        return False, {
            "reason": (
                f"complete object is {major_mm:.1f}mm long, beyond the "
                f"{GRASP_MAX_MAJOR_MM:.1f}mm bounded carry-envelope limit"
            ),
            "width_mm": width_mm,
            "major_mm": major_mm,
            "box_dimensions_mm": None if box_dimensions_mm is None else
                box_dimensions_mm.round(1).tolist(),
            "major_basis": major_basis,
        }
    selected = grasp.info.get("selected_candidate", {})
    try:
        end_fraction = float(selected.get("end_clearance_fraction", 1.0))
    except (AttributeError, TypeError, ValueError):
        end_fraction = 1.0
    end_clearance_mm = 0.5 * major_mm * float(np.clip(end_fraction, 0.0, 1.0))
    elongated = major_mm / max(width_mm, 1e-6) >= 1.6
    if elongated and end_clearance_mm < max(0.0, GRASP_MIN_END_CLEARANCE_MM):
        return False, {
            "reason": (
                f"selected {width_mm:.1f}mm cross-section is only "
                f"{end_clearance_mm:.1f}mm from an object end; at least "
                f"{GRASP_MIN_END_CLEARANCE_MM:.1f}mm is required"
            ),
            "width_mm": width_mm,
            "major_mm": major_mm,
            "end_clearance_mm": end_clearance_mm,
            "box_dimensions_mm": None if box_dimensions_mm is None else
                box_dimensions_mm.round(1).tolist(),
            "major_basis": major_basis,
        }
    return True, {
        "reason": (
            f"object {width_mm:.1f}x{major_mm:.1f}mm: local cross-section fits "
            f"the {preopen_mm:.1f}mm opening with "
            f"{preopen_mm - width_mm:.1f}mm total clearance; estimated nearest "
            f"end clearance {end_clearance_mm:.1f}mm"
        ),
        "width_mm": width_mm,
        "major_mm": major_mm,
        "box_dimensions_mm": None if box_dimensions_mm is None else
            box_dimensions_mm.round(1).tolist(),
        "major_basis": major_basis,
        "end_clearance_mm": end_clearance_mm,
        "preopen_mm": preopen_mm,
    }


def _aligned_3d_jaw_width(
    object_box_3d: dict[str, Any], camera_mapping: np.ndarray,
) -> dict[str, Any]:
    """Select a measured OBB axis only when it is physically between the jaws.

    The image silhouette can show a broad face while the parallel jaws close
    across its much smaller thickness.  That thickness is usable only if its
    projected direction agrees with the calibrated jaw-closing direction and
    it is not mostly optical depth.  This gate is geometry-only and contains no
    object-name or material exceptions.
    """
    axis_raw = os.getenv("REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG")
    if axis_raw is None:
        axis_raw = load_grasp_target_detail().get("jaw_long_axis_camera_deg")
    try:
        jaw_long_deg = float(axis_raw)
        axes_base = np.asarray(object_box_3d["axes_base"], dtype=float)
        observed_mm = np.asarray(
            object_box_3d["observed_size_m"], dtype=float
        ) * 1000.0
        conservative_mm = np.asarray(
            object_box_3d["conservative_size_m"], dtype=float
        ) * 1000.0
        mapping = np.asarray(camera_mapping, dtype=float)
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": "3D jaw-axis geometry is incomplete"}
    if axes_base.shape != (3, 3) or observed_mm.shape != (3,) \
            or conservative_mm.shape != (3,) or mapping.shape != (3, 3):
        return {"ok": False, "reason": "3D jaw-axis geometry has invalid dimensions"}
    close_angle = math.radians(jaw_long_deg + 90.0)
    closing_xy = np.array([math.cos(close_angle), math.sin(close_angle)])
    max_image_angle_deg = float(np.clip(float(os.getenv(
        "REBOT_3D_JAW_AXIS_MAX_ERROR_DEG", "15"
    )), 5.0, 25.0))
    # Image-plane agreement answers whether the selected object dimension is
    # physically between the jaws.  Optical tilt answers a different question:
    # whether enough of that metric axis is visible in RGB-D to trust it.  The
    # old code incorrectly applied the 15deg jaw-alignment tolerance to both,
    # rejecting a well aligned axis merely because it was 16deg out of the
    # image plane.  At the default 25deg optical limit more than 90% of the
    # measured axis remains projected into the image; near-optical/hidden
    # dimensions are still rejected by both this angle and the resolved-length
    # check below.
    max_optical_angle_deg = float(np.clip(float(os.getenv(
        "REBOT_3D_JAW_OPTICAL_MAX_ERROR_DEG", "25"
    )), 10.0, 45.0))
    minimum_resolved_mm = float(np.clip(float(os.getenv(
        "REBOT_3D_JAW_MIN_RESOLVED_MM", "8"
    )), 5.0, 15.0))
    candidates = []
    for index, axis_base in enumerate(axes_base):
        camera_axis = mapping @ axis_base
        norm = float(np.linalg.norm(camera_axis))
        if norm <= 1e-9:
            continue
        camera_axis /= norm
        image_norm = float(np.linalg.norm(camera_axis[:2]))
        if image_norm <= 1e-6:
            continue
        image_axis = camera_axis[:2] / image_norm
        image_error_deg = math.degrees(math.acos(float(np.clip(
            abs(np.dot(image_axis, closing_xy)), 0.0, 1.0
        ))))
        optical_error_deg = math.degrees(math.asin(float(np.clip(
            abs(camera_axis[2]), 0.0, 1.0
        ))))
        projected_resolved_mm = float(observed_mm[index]) * image_norm
        candidates.append({
            "axis_index": index,
            "observed_width_mm": float(observed_mm[index]),
            "conservative_width_mm": float(conservative_mm[index]),
            "image_alignment_error_deg": image_error_deg,
            "optical_tilt_deg": optical_error_deg,
            "projected_resolved_mm": projected_resolved_mm,
            "camera_axis": camera_axis.round(6).tolist(),
        })
    aligned = [candidate for candidate in candidates if
               candidate["image_alignment_error_deg"] <= max_image_angle_deg and
               candidate["optical_tilt_deg"] <= max_optical_angle_deg and
               candidate["projected_resolved_mm"] >= minimum_resolved_mm]
    if not aligned:
        return {
            "ok": False,
            "reason": (
                "no measured 3D object axis is aligned with the calibrated "
                f"jaw-closing direction within {max_image_angle_deg:.1f}deg, "
                f"visible within {max_optical_angle_deg:.1f}deg of the image "
                f"plane, and projected above {minimum_resolved_mm:.1f}mm"
            ),
            "candidates": candidates,
        }
    selected = min(aligned, key=lambda candidate:
                   candidate["conservative_width_mm"])
    return {
        "ok": True,
        **selected,
        "jaw_long_axis_camera_deg": jaw_long_deg,
        "maximum_axis_error_deg": max_image_angle_deg,
        "maximum_optical_tilt_deg": max_optical_angle_deg,
        "minimum_resolved_width_mm": minimum_resolved_mm,
        "reason": (
            f"measured 3D jaw axis {selected['observed_width_mm']:.1f}mm "
            f"({selected['conservative_width_mm']:.1f}mm including RGB-D "
            "uncertainty) is aligned with jaw closure; image error "
            f"{selected['image_alignment_error_deg']:.1f}deg, optical tilt "
            f"{selected['optical_tilt_deg']:.1f}deg"
        ),
    }


def _metric_horizontal_box_dimensions(
    object_box_3d: dict[str, Any] | None,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Return the narrow/long measured tabletop axes from a metric OBB."""
    if not isinstance(object_box_3d, dict):
        return None, None, {"reason": "multiview metric object box is missing"}
    try:
        axes = np.asarray(object_box_3d["axes_base"], dtype=float)
        sizes = np.asarray(object_box_3d["observed_size_m"], dtype=float) * 1000.0
    except (KeyError, TypeError, ValueError):
        return None, None, {"reason": "multiview metric object box is incomplete"}
    if axes.shape != (3, 3) or sizes.shape != (3,) \
            or not np.all(np.isfinite(axes)) \
            or not np.all(np.isfinite(sizes)):
        return None, None, {"reason": "multiview metric object box is non-finite"}
    horizontal = [
        index for index, axis in enumerate(axes)
        if abs(float(axis[2])) <= 0.5 and float(sizes[index]) > 0.0
    ]
    if len(horizontal) < 2:
        # A genuinely tilted object—or a one-sided RGB-D view of a soft/rounded
        # object—need not expose two PCA axes already parallel to the table.
        # Rejecting the entire measured box here made an additional valid view
        # turn a usable right-side sponge into a hard failure. Preserve the
        # measured 3-D geometry: use its narrowest principal extent as the
        # candidate jaw thickness and its longest extent as the clearance
        # length. The later camera/jaw projection gate remains authoritative
        # and will reject this proposal unless that measured narrow axis is
        # actually aligned with the calibrated jaw-closing direction.
        positive = [
            (index, float(size)) for index, size in enumerate(sizes)
            if math.isfinite(float(size)) and float(size) > 0.0
        ]
        if len(positive) != 3:
            return None, None, {
                "reason": "metric object box has invalid principal extents"
            }
        ordered = sorted(positive, key=lambda item: item[1])
        width_index, width_mm = ordered[0]
        major_index, major_mm = ordered[-1]
        return width_mm, major_mm, {
            "reason": (
                "tilted metric box uses measured 3D principal extents: "
                f"jaw candidate {width_mm:.1f}mm, clearance length "
                f"{major_mm:.1f}mm; final jaw-axis projection still required"
            ),
            "orientation_mode": object_box_3d.get("orientation_mode"),
            "principal_sizes_mm": sizes.round(2).tolist(),
            "width_axis_index": width_index,
            "major_axis_index": major_index,
            "requires_jaw_axis_projection": True,
        }
    horizontal_sizes = sorted(float(sizes[index]) for index in horizontal)
    return horizontal_sizes[0], horizontal_sizes[-1], {
        "reason": (
            f"multiview metric horizontal axes are "
            f"{horizontal_sizes[0]:.1f}x{horizontal_sizes[-1]:.1f}mm"
        ),
        "horizontal_axis_indices": horizontal,
        "horizontal_sizes_mm": horizontal_sizes,
    }


def _geometry_with_3d_jaw_width(
    geometry: GG.GraspGeometry, width_detail: dict[str, Any],
) -> GG.GraspGeometry:
    grasp = geometry.grasp_detection
    if grasp is None or not width_detail.get("ok"):
        return geometry
    measured_width = float(width_detail["observed_width_mm"])
    updated = replace(
        grasp, min_width_mm=measured_width,
        info={
            **getattr(grasp, "info", {}),
            "jaw_sizing_basis": "aligned measured 3D OBB axis",
            "aligned_3d_jaw_width": dict(width_detail),
        },
    )
    return replace(
        geometry, grasp_detection=updated,
        reason=(f"{geometry.reason}; jaw width replaced by aligned measured "
                f"3D thickness {measured_width:.1f}mm"),
    )


def _jaw_slot_fit(
    observed_cam_mm: Any, expected_cam_mm: Any,
    jaw_plan: dict[str, Any], geometry: GG.GraspGeometry,
    object_box_3d: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check jaw capture by physical axes instead of a spherical error norm."""
    error = np.asarray(observed_cam_mm, dtype=float) - np.asarray(
        expected_cam_mm, dtype=float
    )
    base_tol = float(np.clip(DYNAMIC_SLOT_TOL_MM, 3.0, 8.0))
    saved_axis = load_grasp_target_detail().get("jaw_long_axis_camera_deg")
    try:
        axis_deg = float(os.getenv(
            "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG", saved_axis
        ))
        theta = math.radians(axis_deg)
    except (TypeError, ValueError):
        norm_mm = float(np.linalg.norm(error))
        return norm_mm <= base_tol, {
            "reason": "jaw-axis calibration unavailable; used spherical gate",
            "error_mm": norm_mm,
            "limit_mm": base_tol,
        }
    long_axis = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    closing_axis = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    along_mm = abs(float(error[:2] @ long_axis))
    across_mm = abs(float(error[:2] @ closing_axis))
    depth_mm = abs(float(error[2]))
    width_mm = float(jaw_plan["min_width_mm"])
    preopen_mm = float(jaw_plan["preopen_mm"])
    physical_half_clearance = max(0.0, 0.5 * (preopen_mm - width_mm))
    # Keep a real capture margin inside each open finger, but do not command a
    # multi-millimetre arm recentering move for sub-millimetre RGB-D boundary
    # noise.  The allowance is capped below the 1 mm physical reserve, so a
    # passing observation is still inside the measured open-jaw envelope.
    boundary_slop = float(np.clip(
        DYNAMIC_SLOT_BOUNDARY_SLOP_MM, 0.0, 0.75
    ))
    capture_reserve = 1.0 - boundary_slop
    across_limit = min(
        12.0, max(0.0, physical_half_clearance - capture_reserve)
    )
    # Use the same registered object box that authorized the approach. A close
    # view is often cropped by the fingers; re-running the complete-object fit
    # on that partial view can contradict the already validated 3-D geometry
    # and turn a 0.2 mm slot residual into an infinite violation.
    fit_ok, fit_detail = _validate_object_gripper_fit(
        geometry, jaw_plan, object_box_3d
    )
    end_clearance = float(fit_detail.get("end_clearance_mm", base_tol))
    along_limit = min(20.0, max(base_tol, end_clearance - 5.0))
    # Optical depth carries systematic hand-eye/TCP error that is not visible
    # in the sub-millimetre multi-frame spread.  The fingers already have
    # capture length along this axis, so retain the strict across/along gates
    # and add only a small, globally bounded depth deadband.  This prevents the
    # arm from translating 8--16 mm to chase a final sub-millimetre depth
    # residual while the object is visibly between the jaws.
    depth_slack = float(np.clip(DYNAMIC_SLOT_DEPTH_SLACK_MM, 0.0, 3.0))
    # Treat only a sub-millimetre reading at the depth boundary as sensor
    # quantisation. This is deliberately separate from the 3 mm systematic
    # depth allowance and capped at 0.5 mm: the logged 9.2 mm observation can
    # close without an unnecessary recentering move, while a genuine 11.1 mm
    # miss remains outside the unchanged capture region.
    depth_boundary_slop = float(np.clip(
        DYNAMIC_SLOT_DEPTH_BOUNDARY_SLOP_MM, 0.0, 0.5
    ))
    depth_limit = base_tol + depth_slack + depth_boundary_slop
    ok = bool(
        fit_ok
        and across_limit > 0.0
        and across_mm <= across_limit
        and along_mm <= along_limit
        and depth_mm <= depth_limit
    )
    return ok, {
        "reason": (
            f"axis-aware slot check: across {across_mm:.1f}/{across_limit:.1f}mm, "
            f"along {along_mm:.1f}/{along_limit:.1f}mm, depth "
            f"{depth_mm:.1f}/{depth_limit:.1f}mm"
        ),
        "across_error_mm": across_mm,
        "across_limit_mm": across_limit,
        "physical_across_limit_mm": physical_half_clearance,
        "boundary_slop_mm": boundary_slop,
        "capture_reserve_mm": capture_reserve,
        "along_error_mm": along_mm,
        "along_limit_mm": along_limit,
        "depth_error_mm": depth_mm,
        "depth_limit_mm": depth_limit,
        "depth_base_limit_mm": base_tol,
        "depth_slack_mm": depth_slack,
        "depth_boundary_slop_mm": depth_boundary_slop,
        "fit_ok": fit_ok,
        "fit_reason": fit_detail.get("reason"),
    }


def _jaw_slot_violation_mm(detail: dict[str, Any]) -> float:
    """Distance outside the axis-aware capture box; zero means inside."""
    violations = []
    for name in ("across", "along", "depth"):
        error = float(detail.get(f"{name}_error_mm", math.inf))
        limit = float(detail.get(f"{name}_limit_mm", 0.0))
        violations.append(max(0.0, error - limit))
    if not bool(detail.get("fit_ok", False)):
        return math.inf
    return float(np.linalg.norm(violations))


def _propagate_locked_body_after_alignment(
    body_cam_mm: Any,
    rigid_jacobian: Any,
    tool_start_m: Any,
    tool_end_m: Any,
    planned_endpoint_m: Any,
    requested_cam_mm: Any,
    jaw_plan: dict[str, Any],
    geometry: GG.GraspGeometry,
    previous_violation_mm: float,
    *, identity_frames: int,
    object_box_3d: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transport a locked grasp point through one measured tool correction.

    The target is stationary while the open jaws make a small, prevalidated
    Cartesian correction. If finger occlusion makes the exact visual
    subsection disappear afterwards, the rigid camera Jacobian can propagate
    its last stable camera point. This is accepted only with continued target
    identity, accurate execution of the planned endpoint, bounded motion, and
    a strict improvement under the unchanged physical jaw-slot gate.
    """
    failure = lambda reason: {"ok": False, "reason": reason}
    try:
        body = np.asarray(body_cam_mm, dtype=float)
        mapping = np.asarray(rigid_jacobian, dtype=float)
        start = np.asarray(tool_start_m, dtype=float)
        end = np.asarray(tool_end_m, dtype=float)
        planned = np.asarray(planned_endpoint_m, dtype=float)
        previous = float(previous_violation_mm)
    except (TypeError, ValueError):
        return failure("alignment propagation inputs are invalid")
    if body.shape != (3,) or mapping.shape != (3, 3) \
            or start.shape != (3,) or end.shape != (3,) \
            or planned.shape != (3,) or not all(np.all(np.isfinite(value))
                                                for value in (
                                                    body, mapping, start,
                                                    end, planned,
                                                )) \
            or not math.isfinite(previous):
        return failure("alignment propagation inputs are incomplete or non-finite")
    if int(identity_frames) < 3:
        return failure("fewer than 3 post-motion target-identity frames")
    motion_m = end - start
    motion_norm_m = float(np.linalg.norm(motion_m))
    if motion_norm_m < 0.0005 or motion_norm_m > 0.015:
        return failure(
            f"measured alignment motion {motion_norm_m * 1000.0:.1f}mm is "
            "outside the 0.5-15.0mm propagation bound"
        )
    endpoint_error_m = float(np.linalg.norm(end - planned))
    if endpoint_error_m > 0.003:
        return failure(
            f"tool stopped {endpoint_error_m * 1000.0:.1f}mm from the "
            "validated alignment endpoint"
        )
    predicted = body + mapping @ motion_m
    slot_ok, slot_detail = _jaw_slot_fit(
        predicted, requested_cam_mm, jaw_plan, geometry, object_box_3d
    )
    violation = _jaw_slot_violation_mm(slot_detail)
    improved = bool(slot_ok or violation < previous - 0.5)
    if not improved:
        return failure(
            f"motion-model jaw-slot violation did not improve: "
            f"{previous:.1f}->{violation:.1f}mm; "
            f"{slot_detail.get('reason')}"
        )
    return {
        "ok": True,
        "reason": (
            "locked grasp body propagated through measured open-jaw motion; "
            f"slot violation {previous:.1f}->{violation:.1f}mm; "
            f"{slot_detail.get('reason')}"
        ),
        "body_cam_mm": predicted,
        "slot_ok": bool(slot_ok),
        "slot_detail": slot_detail,
        "violation_mm": violation,
        "motion_mm": (motion_m * 1000.0).round(2).tolist(),
        "endpoint_error_mm": round(endpoint_error_m * 1000.0, 2),
    }


def _jaw_slot_depth_occupancy_detection(
    frame: tuple[np.ndarray, np.ndarray, Any],
    requested_body_cam_mm: Any,
    jaw_plan: dict[str, Any],
    geometry: GG.GraspGeometry,
    object_box_3d: dict[str, Any] | None = None,
) -> tuple[WD.Detection | None, str]:
    """Detect anonymous foreground only inside the calibrated open-jaw slot.

    This is an endpoint-only fallback for semantic occlusion by the fingers. It
    cannot discover or steer toward an object: the complete approach must have
    already been planned from a named, registered target. A stable foreground
    must occupy the small calibrated slot volume and stand proud of its local
    background. Width remains the registered/depth-validated grasp width.
    """
    color, depth, intr = frame
    target = np.asarray(requested_body_cam_mm, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return None, "calibrated jaw-slot target is invalid"
    if depth.ndim != 2 or depth.shape != color.shape[:2]:
        return None, "RGB-D frame dimensions are invalid"
    fx, fy, cx, cy = (float(value) for value in intr)
    if fx <= 0.0 or fy <= 0.0:
        return None, "camera intrinsics are invalid"
    saved_axis = load_grasp_target_detail().get("jaw_long_axis_camera_deg")
    try:
        jaw_long_deg = float(os.getenv(
            "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG", saved_axis
        ))
        width_mm = float(jaw_plan["min_width_mm"])
        preopen_mm = float(jaw_plan["preopen_mm"])
    except (KeyError, TypeError, ValueError):
        return None, "jaw-slot geometry is incomplete"
    theta = math.radians(jaw_long_deg)
    long_axis = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    closing_axis = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    across_half_mm = min(12.0, max(3.0, 0.5 * preopen_mm - 2.0))
    along_half_mm = 20.0
    depth_half_mm = float(np.clip(float(os.getenv(
        "REBOT_DYNAMIC_SLOT_OCCUPANCY_DEPTH_MM", "15"
    )), 8.0, 20.0))
    prominence_mm = float(np.clip(float(os.getenv(
        "REBOT_DYNAMIC_SLOT_OCCUPANCY_PROMINENCE_MM", "8"
    )), 5.0, 20.0))

    z = np.asarray(depth, dtype=float)
    rows, cols = np.indices(z.shape)
    valid = np.isfinite(z) & (z > 0.0)
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    delta_x = x - target[0]
    delta_y = y - target[1]
    along = delta_x * long_axis[0] + delta_y * long_axis[1]
    across = delta_x * closing_axis[0] + delta_y * closing_axis[1]
    central = (
        valid
        & (np.abs(z - target[2]) <= depth_half_mm)
        & (np.abs(along) <= along_half_mm)
        & (np.abs(across) <= across_half_mm)
    )
    mask = central.astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    expected_uv = WD.project(target, intr)
    if expected_uv is None:
        return None, "calibrated jaw-slot target cannot be projected"
    best_label = None
    best_distance = float("inf")
    minimum_pixels = 20
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_pixels:
            continue
        component_rows, component_cols = np.nonzero(labels == label)
        distance = float(np.hypot(
            np.median(component_cols) - expected_uv[0],
            np.median(component_rows) - expected_uv[1],
        ))
        if distance < best_distance:
            best_label, best_distance = label, distance
    if best_label is None:
        return None, "no stable-size depth foreground occupies the jaw slot"
    component = labels == best_label
    component_z = z[component]
    centre_z = float(np.median(component_z))
    ring = (
        valid
        & (np.abs(along) <= 1.8 * along_half_mm)
        & (np.abs(across) <= 1.8 * across_half_mm)
        & ~central
    )
    ring_depth = z[ring]
    if ring_depth.size < minimum_pixels:
        return None, "jaw-slot background support is insufficient"
    background_z = float(np.median(ring_depth))
    prominence = background_z - centre_z
    if not math.isfinite(prominence) or prominence < prominence_mm:
        return None, (
            f"jaw-slot depth is not foreground: prominence "
            f"{prominence:.1f}/{prominence_mm:.1f}mm"
        )
    component_rows, component_cols = np.nonzero(component)
    u = float(np.median(component_cols))
    v = float(np.median(component_rows))
    p_cam = WD.deproject(u, v, centre_z, intr)
    slot_ok, slot_detail = _jaw_slot_fit(
        p_cam, target, jaw_plan, geometry, object_box_3d
    )
    if not slot_ok:
        return None, str(slot_detail.get("reason"))
    x1 = int(stats[best_label, cv2.CC_STAT_LEFT])
    y1 = int(stats[best_label, cv2.CC_STAT_TOP])
    width_px = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height_px = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    detection = WD.Detection(
        box=(x1, y1, x1 + width_px, y1 + height_px),
        uv=(u, v), z_mm=centre_z, p_cam_mm=p_cam,
        source="depth:object", area_px=float(component.sum()),
        min_width_mm=width_mm,
        max_width_mm=float(
            geometry.grasp_detection.max_width_mm
            if geometry.grasp_detection is not None
            and geometry.grasp_detection.max_width_mm is not None
            else width_mm
        ),
        info={
            "mask_u8": component.astype(np.uint8) * 255,
            "long_axis_deg": jaw_long_deg,
            "local_grasp_width_mm": width_mm,
            "endpoint_width_basis": "registered_shape_plus_slot_occupancy",
            "slot_occupancy_prominence_mm": prominence,
            "cross_section_candidates": [{
                "uv": [u, v], "width_mm": width_mm,
                "axis_fraction": 0.0, "end_clearance_fraction": 1.0,
                "width_variation_mm": 0.0, "score": 2.0,
            }],
        },
    )
    return detection, (
        f"calibrated jaw-slot foreground at {centre_z:.1f}mm; "
        f"prominence {prominence:.1f}mm; {slot_detail.get('reason')}"
    )


def _geometry_with_profile_cross_section(
    geometry: GG.GraspGeometry,
    frame: tuple[np.ndarray, np.ndarray, Any],
    candidate: dict[str, Any],
    *,
    rank: int | None = None,
) -> GG.GraspGeometry | None:
    """Move a grasp proposal to one measured long-axis cross-section.

    Cross-sections are produced by the current RGB-D object mask.  This helper
    deliberately carries no object label, preferred image location, or base
    coordinate: a later collision check decides whether the measured section is
    usable.
    """
    base_body = geometry.grasp_detection
    envelope = geometry.envelope_detection or base_body
    if base_body is None or envelope is None:
        return None
    try:
        uv = tuple(float(value) for value in candidate["uv"])
        width_mm = float(candidate["width_mm"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(uv) != 2 or not all(math.isfinite(value) for value in uv) \
            or not math.isfinite(width_mm) or width_mm <= 0.0:
        return None
    depth_image = np.asarray(frame[1])
    intr = frame[2]
    z_mm = float(WD.median_depth(depth_image, uv[0], uv[1], r=4))
    if not math.isfinite(z_mm) or z_mm < WD.NEAR_MM or z_mm > WD.FAR_MM:
        # A different long-axis section can sit at a different depth. Falling
        # back to the envelope centre would fabricate its 3D location and could
        # invalidate the very collision check this candidate is meant to pass.
        return None
    # A multi-view 3D thickness aligned with the calibrated jaw axis is a more
    # reliable width than any single image slice. Preserve it while moving the
    # grasp point along the measured object body.
    aligned_width = base_body.info.get("aligned_3d_jaw_width") is not None
    info = {
        **base_body.info,
        "selected_candidate": dict(candidate),
    }
    if rank is not None:
        info["collision_aware_candidate_rank"] = int(rank)
    body = replace(
        base_body,
        uv=uv,
        z_mm=z_mm,
        p_cam_mm=WD.deproject(uv[0], uv[1], z_mm, intr),
        min_width_mm=(
            float(base_body.min_width_mm) if aligned_width else width_mm
        ),
        source="depth:collision_ranked_cross_section",
        info=info,
    )
    return replace(geometry, grasp_detection=body)


def _selected_axis_fraction(geometry: GG.GraspGeometry) -> float | None:
    """Return the normalized measured-body location of a grasp proposal."""
    body = geometry.grasp_detection
    if body is None:
        return None
    selected = body.info.get("selected_candidate", {})
    try:
        value = float(selected["axis_fraction"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _projected_3d_jaw_span(
    object_box_3d: dict[str, Any] | None,
    camera_mapping: np.ndarray,
) -> dict[str, Any]:
    """Project a measured OBB onto the calibrated jaw-closing direction.

    This is used when collision avoidance tilts the wrist without changing its
    grasp yaw.  Unlike selecting only the narrowest OBB axis, the full support
    function accounts for every measured axis that contributes to the span
    between the parallel jaws.
    """
    if not isinstance(object_box_3d, dict):
        return {"ok": False, "reason": "measured 3D object box is unavailable"}
    saved_axis = load_grasp_target_detail().get("jaw_long_axis_camera_deg")
    try:
        axes_base = np.asarray(object_box_3d["axes_base"], dtype=float)
        observed_mm = 1000.0 * np.asarray(
            object_box_3d["observed_size_m"], dtype=float
        )
        conservative_mm = 1000.0 * np.asarray(
            object_box_3d["conservative_size_m"], dtype=float
        )
        mapping = np.asarray(camera_mapping, dtype=float)
        jaw_long_deg = float(os.getenv(
            "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG", saved_axis
        ))
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": "3D jaw projection inputs are incomplete"}
    if axes_base.shape != (3, 3) or mapping.shape != (3, 3) \
            or observed_mm.shape != (3,) or conservative_mm.shape != (3,) \
            or not np.all(np.isfinite(axes_base)) \
            or not np.all(np.isfinite(mapping)) \
            or not np.all(np.isfinite(observed_mm)) \
            or not np.all(np.isfinite(conservative_mm)):
        return {"ok": False, "reason": "3D jaw projection inputs are invalid"}
    close_angle = math.radians(jaw_long_deg + 90.0)
    closing_camera = np.array([
        math.cos(close_angle), math.sin(close_angle), 0.0
    ])
    contributions = []
    for axis_base in axes_base:
        camera_axis = mapping @ axis_base
        norm = float(np.linalg.norm(camera_axis))
        if norm <= 1e-9:
            return {"ok": False, "reason": "3D object axis projection collapsed"}
        camera_axis /= norm
        contributions.append(abs(float(camera_axis @ closing_camera)))
    weights = np.asarray(contributions, dtype=float)
    observed_span = float(observed_mm @ weights)
    conservative_span = float(conservative_mm @ weights)
    if not math.isfinite(observed_span) or not math.isfinite(conservative_span) \
            or observed_span <= 0.0 or conservative_span <= 0.0:
        return {"ok": False, "reason": "projected jaw span is invalid"}
    return {
        "ok": True,
        "observed_width_mm": observed_span,
        "conservative_width_mm": conservative_span,
        "axis_contributions": weights.round(5).tolist(),
        "reason": (
            f"measured OBB projects to {observed_span:.1f}mm between the jaws "
            f"({conservative_span:.1f}mm including RGB-D uncertainty)"
        ),
    }


def _geometry_with_projected_3d_width(
    geometry: GG.GraspGeometry, width_detail: dict[str, Any],
) -> GG.GraspGeometry:
    grasp = geometry.grasp_detection
    if grasp is None or not width_detail.get("ok"):
        return geometry
    measured_width = float(width_detail["observed_width_mm"])
    updated = replace(
        grasp, min_width_mm=measured_width,
        info={
            **getattr(grasp, "info", {}),
            "jaw_sizing_basis": "full measured 3D OBB projection",
            "projected_3d_jaw_width": dict(width_detail),
        },
    )
    return replace(
        geometry, grasp_detection=updated,
        reason=(f"{geometry.reason}; collision-escape jaw span "
                f"{measured_width:.1f}mm from the measured 3D OBB"),
    )


def _revalidate_recovered_metric_grasp(
    geometry: GG.GraspGeometry, object_box_3d: dict[str, Any] | None,
    camera_mapping: np.ndarray, max_aperture_mm: float,
) -> tuple[GG.GraspGeometry | None, dict[str, Any]]:
    """Keep an accepted metric recovery pose across later box enrichment.

    Hybrid fusion runs once during look-around recovery and again when the main
    workflow installs its authoritative object box.  The latter must not let a
    noisy single-frame yaw overwrite the former.  Re-project the refreshed box
    into the *current* calibrated jaw direction instead; preservation is
    allowed only when the recovery marker and both aperture gates still pass.
    """
    grasp = geometry.grasp_detection
    if grasp is None or not bool(grasp.info.get(
        "metric_recovery_pose_validated", False
    )):
        return None, {"ok": False, "reason": "no validated metric recovery pose"}
    projected = _projected_3d_jaw_span(object_box_3d, camera_mapping)
    if not projected.get("ok"):
        return None, dict(projected)
    observed = float(projected["observed_width_mm"])
    conservative = float(projected["conservative_width_mm"])
    aperture = float(max_aperture_mm)
    safe = bool(
        conservative <= aperture + 1e-6
        and observed + max(0.0, GRASP_MIN_OPEN_CLEARANCE_MM)
            <= aperture + 1e-6
    )
    if not safe:
        return None, {
            **projected,
            "ok": False,
            "reason": (
                f"refreshed 3D box projects to {observed:.1f}mm "
                f"({conservative:.1f}mm conservative), outside the "
                f"{aperture:.1f}mm calibrated aperture/entry-clearance gate"
            ),
        }
    refreshed = _geometry_with_projected_3d_width(geometry, projected)
    refreshed = replace(
        refreshed, object_box_3d=object_box_3d,
        reason=(f"{refreshed.reason}; recovered fused-metric pose preserved "
                "after refreshed full-OBB jaw projection"),
    )
    return refreshed, {**projected, "ok": True}


def _geometry_near_feature_offset(
    geometry: GG.GraspGeometry,
    frame: tuple[np.ndarray, np.ndarray, Any],
    feature_detection: WD.Detection,
    expected_feature_to_body_cam_mm: Any,
    *,
    axis_fraction: float | None = None,
) -> GG.GraspGeometry | None:
    """Reacquire one physical body section without trusting axis sign.

    The image long-axis convention is symmetric modulo 180 degrees, so its
    sign may flip between masks.  The camera-frame vector from the semantic
    feature to the chosen body section is rigid while the object is stationary
    and therefore provides a safer dynamic identity.  ``axis_fraction`` is only
    a magnitude-based tie-breaker.
    """
    expected = np.asarray(expected_feature_to_body_cam_mm, dtype=float)
    if expected.shape != (3,) or not np.all(np.isfinite(expected)):
        return None
    feature_cam = np.asarray(feature_detection.p_cam_mm, dtype=float)
    if feature_cam.shape != (3,) or not np.all(np.isfinite(feature_cam)):
        return None
    candidates = getattr(geometry, "candidates", [])
    options: list[tuple[float, float, GG.GraspGeometry]] = []
    base_body = geometry.grasp_detection
    if base_body is not None:
        base_cam = np.asarray(base_body.p_cam_mm, dtype=float)
        if base_cam.shape == (3,) and np.all(np.isfinite(base_cam)):
            base_offset = feature_cam - base_cam
            base_error = float(np.linalg.norm(base_offset - expected))
            base_fraction = _selected_axis_fraction(geometry)
            fraction_error = (
                abs(abs(base_fraction) - abs(axis_fraction))
                if base_fraction is not None and axis_fraction is not None
                else 0.0
            )
            if math.isfinite(base_error):
                options.append((base_error, float(fraction_error), geometry))
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            candidate_geometry = _geometry_with_profile_cross_section(
                geometry, frame, item
            )
            if candidate_geometry is None \
                    or candidate_geometry.grasp_detection is None:
                continue
            body_cam = np.asarray(
                candidate_geometry.grasp_detection.p_cam_mm, dtype=float
            )
            if body_cam.shape != (3,) or not np.all(np.isfinite(body_cam)):
                continue
            offset_error = float(np.linalg.norm(
                (feature_cam - body_cam) - expected
            ))
            if not math.isfinite(offset_error):
                continue
            try:
                fraction = float(item["axis_fraction"])
            except (KeyError, TypeError, ValueError):
                fraction = math.nan
            fraction_error = (
                abs(abs(fraction) - abs(axis_fraction))
                if math.isfinite(fraction) and axis_fraction is not None else 0.0
            )
            options.append((offset_error, float(fraction_error),
                            candidate_geometry))
    if not options:
        return None
    offset_error, _fraction_error, selected = min(
        options, key=lambda item: (item[0], item[1])
    )
    default_tolerance_mm = max(
        float(DYNAMIC_WIDTH_CONSENSUS_MAX_SPREAD_MM),
        float(DYNAMIC_SLOT_TOL_MM),
    )
    try:
        configured_tolerance = float(os.getenv(
            "REBOT_GRASP_SECTION_MATCH_TOL_MM", str(default_tolerance_mm)
        ))
    except (TypeError, ValueError):
        configured_tolerance = default_tolerance_mm
    if not math.isfinite(configured_tolerance):
        configured_tolerance = default_tolerance_mm
    tolerance_mm = float(np.clip(configured_tolerance, 2.0, 12.0))
    if not math.isfinite(offset_error) or offset_error > tolerance_mm:
        return None
    return selected


def _fresh_geometry_for_selected_section(
    geometry: GG.GraspGeometry,
    frame: tuple[np.ndarray, np.ndarray, Any],
    feature_detection: WD.Detection,
    expected_feature_to_body_cam_mm: Any,
    *,
    section_locked: bool,
    axis_fraction: float | None = None,
) -> GG.GraspGeometry | None:
    """Preserve only an explicitly selected alternate object section.

    Normal grasps should use the best fresh cross-section recovered at the
    closer checkpoint; perspective and segmentation legitimately improve there.
    Once that close-range section is accepted, or when collision planning chose
    an alternate section, ``section_locked`` preserves its physical identity
    through fine corrections and lift verification.
    """
    if not section_locked:
        return geometry
    return _geometry_near_feature_offset(
        geometry, frame, feature_detection,
        expected_feature_to_body_cam_mm,
        axis_fraction=axis_fraction,
    )


def _endpoint_section_identity(
    sample: dict[str, Any] | None,
) -> tuple[np.ndarray | None, float | None, dict[str, Any]]:
    """Freeze one freshly measured grasp section relative to its identifier.

    A fresh close-range profile is allowed to replace the distant grasp
    proposal once.  After that choice, alignment corrections must compare the
    same physical section before and after arm motion.  Re-running the generic
    "best cross-section" selector after every correction can jump to another
    part of a long or tilted object and make a good camera Jacobian appear to
    have the wrong sign.
    """
    if not isinstance(sample, dict):
        return None, None, {"reason": "endpoint sample is unavailable"}
    try:
        feature = np.asarray(sample["detection"].p_cam_mm, dtype=float)
        geometry = sample["geometry"]
        body = np.asarray(geometry.grasp_detection.p_cam_mm, dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, None, {
            "reason": "endpoint feature/body geometry is incomplete"
        }
    if feature.shape != (3,) or body.shape != (3,) \
            or not np.all(np.isfinite(feature)) \
            or not np.all(np.isfinite(body)):
        return None, None, {
            "reason": "endpoint feature/body geometry is non-finite"
        }
    offset = feature - body
    axis_fraction = _selected_axis_fraction(geometry)
    return offset, axis_fraction, {
        "reason": (
            "stable close-range grasp section locked relative to the semantic "
            f"identifier; feature-to-body={offset.round(1).tolist()}mm"
        ),
        "feature_to_body_cam_mm": offset.tolist(),
        "axis_fraction": axis_fraction,
    }


def _plan_collision_aware_grasp_pose(
    geometry: GG.GraspGeometry,
    frame: tuple[np.ndarray, np.ndarray, Any],
    camera_mapping: np.ndarray,
    centered_target_cam_mm: Any,
    approach_rpy: tuple[float, float, float],
    jaw_plan: dict[str, Any],
    object_box_3d: dict[str, Any] | None,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Search measured grasp sections and jaw-slot slack after a blocked pose.

    Nothing here depends on an object name or a fixed world direction. Candidate
    body points come from the semantic/depth cross-section profile; lateral slot
    offsets come from the measured object width and calibrated open aperture.
    Every resulting endpoint and complete route retains the normal IK/collision
    gates before any movement.
    """
    base_body = geometry.grasp_detection
    envelope = geometry.envelope_detection or base_body
    if base_body is None or envelope is None:
        return {"ok": False, "reason": "measured grasp body is unavailable"}
    centered_target = np.asarray(centered_target_cam_mm, dtype=float)
    if centered_target.shape != (3,) or not np.all(np.isfinite(centered_target)):
        return {"ok": False, "reason": "calibrated jaw target is invalid"}

    _inside, slot_detail = _jaw_slot_fit(
        centered_target, centered_target, jaw_plan, geometry
    )
    across_limit_mm = float(slot_detail.get("across_limit_mm", 0.0))
    if not math.isfinite(across_limit_mm) or across_limit_mm <= 0.0:
        return {
            "ok": False,
            "reason": "measured object leaves no safe lateral jaw-slot slack",
            "slot": slot_detail,
        }
    saved_axis = load_grasp_target_detail().get("jaw_long_axis_camera_deg")
    try:
        jaw_axis_deg = float(os.getenv(
            "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG", saved_axis
        ))
    except (TypeError, ValueError):
        return {
            "ok": False,
            "reason": "jaw-axis calibration is unavailable for pose sampling",
        }
    theta = math.radians(jaw_axis_deg)
    closing_axis_cam = np.array(
        [-math.sin(theta), math.cos(theta)], dtype=float
    )

    # Sample the whole dynamically measured safe slot. Resolution follows the
    # configured collision inflation (and is independently tunable); no object
    # class or fixed world displacement is encoded here.
    default_step_mm = max(
        1.0, 0.5 * float(SCENE_COLLISION_INFLATION_M) * 1000.0
    )
    try:
        configured_step_mm = float(os.getenv(
            "REBOT_GRASP_POSE_SLOT_STEP_MM", str(default_step_mm)
        ))
    except (TypeError, ValueError):
        configured_step_mm = default_step_mm
    if not math.isfinite(configured_step_mm):
        configured_step_mm = default_step_mm
    sample_step_mm = float(np.clip(
        configured_step_mm, 0.5, across_limit_mm
    ))
    magnitudes = np.arange(
        sample_step_mm, across_limit_mm, sample_step_mm, dtype=float
    ).tolist()
    magnitudes.append(across_limit_mm)
    slot_offsets = [0.0]
    for magnitude in magnitudes:
        slot_offsets.extend((-float(magnitude), float(magnitude)))
    slot_offsets = sorted(
        set(round(value, 6) for value in slot_offsets),
        key=lambda value: (abs(float(value)), float(value)),
    )
    body_options: list[tuple[str, WD.Detection, float]] = [
        ("selected", base_body, float("inf"))
    ]
    profile = getattr(geometry, "candidates", [])
    if isinstance(profile, list):
        ranked = sorted(
            (item for item in profile if isinstance(item, dict)),
            key=lambda item: float(item.get("score", -math.inf)),
            reverse=True,
        )
        for index, item in enumerate(ranked):
            try:
                uv = tuple(float(value) for value in item["uv"])
                score = float(item.get("score", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if len(uv) != 2 or not all(math.isfinite(value) for value in uv):
                continue
            if float(np.linalg.norm(
                np.asarray(uv) - np.asarray(base_body.uv)
            )) < 1.0:
                continue
            candidate_geometry = _geometry_with_profile_cross_section(
                geometry, frame, item, rank=index + 1,
            )
            if candidate_geometry is None \
                    or candidate_geometry.grasp_detection is None:
                continue
            body_options.append((
                f"section_{index + 1}",
                candidate_geometry.grasp_detection,
                score,
            ))

    try:
        configured_max_candidates = int(os.getenv(
            "REBOT_GRASP_POSE_MAX_CANDIDATES", "20"
        ))
    except (TypeError, ValueError):
        configured_max_candidates = 20
    max_candidates = int(np.clip(configured_max_candidates, 4, 40))
    attempts = 0
    failures: list[str] = []
    log(
        "stage=grasp_pose_search status=starting "
        f"measured_sections={len(body_options)} "
        f"jaw_slot_half_slack_mm={across_limit_mm:.1f} "
        f"slot_sample_step_mm={sample_step_mm:.1f} "
        f"candidate_cap={max_candidates}"
    )
    valid_bodies = []
    for body_name, body, score in body_options:
        candidate_geometry = replace(geometry, grasp_detection=body)
        candidate_jaw = _plan_aperture(body)
        fit_ok, fit_detail = (
            _validate_object_gripper_fit(
                candidate_geometry, candidate_jaw, object_box_3d
            ) if candidate_jaw.get("ok") else
            (False, {"reason": candidate_jaw.get("reason")})
        )
        if not fit_ok:
            failures.append(f"{body_name}: {fit_detail.get('reason')}")
            continue
        valid_bodies.append((
            body_name, body, score, candidate_geometry, candidate_jaw
        ))

    # Position and jaw-slot alternatives cannot resolve a collision caused by
    # an upstream wrist/link at the shared terminal pose. Before giving up,
    # tilt the grasp about pitch while retaining the vision-selected yaw. This
    # changes the arm configuration without changing which measured object
    # cross-section is held. Every candidate uses the rigidly rotated camera
    # direction, the full 3D OBB projection between the jaws, and the complete
    # path collision gate.
    if object_box_3d is not None and valid_bodies:
        try:
            pitch_step_deg = float(os.getenv(
                "REBOT_GRASP_COLLISION_PITCH_STEP_DEG", "5"
            ))
            pitch_limit_deg = float(os.getenv(
                "REBOT_GRASP_COLLISION_MAX_PITCH_DEG",
                str(GEOMETRY_MAX_YAW_DELTA_DEG),
            ))
            camera_limit_deg = float(os.getenv(
                "REBOT_GRASP_COLLISION_CAMERA_DELTA_DEG", "40"
            ))
        except (TypeError, ValueError):
            pitch_step_deg, pitch_limit_deg, camera_limit_deg = 5.0, 20.0, 40.0
        pitch_step_deg = float(np.clip(pitch_step_deg, 2.0, 10.0))
        pitch_limit_deg = float(np.clip(
            pitch_limit_deg, pitch_step_deg,
            max(pitch_step_deg, GEOMETRY_MAX_YAW_DELTA_DEG),
        ))
        camera_limit_deg = float(np.clip(camera_limit_deg, 25.0, 45.0))
        pitch_magnitudes = np.arange(
            pitch_step_deg, pitch_limit_deg + 0.5 * pitch_step_deg,
            pitch_step_deg,
        )
        pitch_offsets_deg = [
            signed
            for magnitude in pitch_magnitudes
            for signed in (float(magnitude), -float(magnitude))
        ]
        _effective, nominal_reachability = _reachable_approach_target(
            base_body, camera_mapping, centered_target
        )
        nominal_endpoint_raw = nominal_reachability.get(
            "requested_endpoint_m"
        )
        if nominal_endpoint_raw is not None \
                and nominal_reachability.get("full_hover"):
            nominal_endpoint = np.asarray(nominal_endpoint_raw, dtype=float)
            log(
                "stage=grasp_orientation_search status=starting "
                f"pitch_step_deg={pitch_step_deg:.1f} "
                f"pitch_limit_deg={pitch_limit_deg:.1f} "
                f"camera_delta_limit_deg={camera_limit_deg:.1f}"
            )
            for orientation_attempt, pitch_offset_deg in enumerate(
                pitch_offsets_deg, start=1
            ):
                candidate_rpy_array = np.asarray(approach_rpy, dtype=float)
                candidate_rpy_array[1] += math.radians(pitch_offset_deg)
                candidate_rpy = tuple(
                    float(value) for value in candidate_rpy_array
                )
                candidate_mapping, mapping_detail = \
                    load_saved_camera_jacobian(
                        candidate_rpy,
                        maximum_orientation_delta_deg=camera_limit_deg,
                    )
                if candidate_mapping is None:
                    log(
                        "stage=grasp_orientation_candidate status=rejected "
                        f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                        f"pitch_offset_deg={pitch_offset_deg:+.1f} "
                        f"detail={mapping_detail.get('reason')}"
                    )
                    continue
                width_detail = _projected_3d_jaw_span(
                    object_box_3d, candidate_mapping
                )
                if not width_detail.get("ok"):
                    log(
                        "stage=grasp_orientation_candidate status=rejected "
                        f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                        f"pitch_offset_deg={pitch_offset_deg:+.1f} "
                        f"detail={width_detail.get('reason')}"
                    )
                    continue
                if float(width_detail["conservative_width_mm"]) \
                        > float(jaw_plan["max_aperture_mm"]) + 1e-6:
                    log(
                        "stage=grasp_orientation_candidate status=rejected "
                        f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                        f"pitch_offset_deg={pitch_offset_deg:+.1f} "
                        f"detail={width_detail.get('reason')}; exceeds "
                        f"{float(jaw_plan['max_aperture_mm']):.1f}mm aperture"
                    )
                    continue
                candidate_geometry = _geometry_with_projected_3d_width(
                    geometry, width_detail
                )
                candidate_jaw = _plan_aperture(
                    candidate_geometry.grasp_detection
                )
                fit_ok, fit_detail = (
                    _validate_object_gripper_fit(
                        candidate_geometry, candidate_jaw, object_box_3d
                    ) if candidate_jaw.get("ok") else
                    (False, {"reason": candidate_jaw.get("reason")})
                )
                if not fit_ok:
                    log(
                        "stage=grasp_orientation_candidate status=rejected "
                        f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                        f"pitch_offset_deg={pitch_offset_deg:+.1f} "
                        f"detail={fit_detail.get('reason')}"
                    )
                    continue
                endpoint_gate = A.check_pose_clearance(
                    *nominal_endpoint.tolist(), *candidate_rpy
                )
                if not endpoint_gate.ok:
                    log(
                        "stage=grasp_orientation_candidate status=endpoint_blocked "
                        f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                        f"pitch_offset_deg={pitch_offset_deg:+.1f} "
                        f"detail={endpoint_gate.reason}"
                    )
                    continue
                log(
                    "stage=grasp_orientation_candidate status=route_checking "
                    f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                    f"pitch_offset_deg={pitch_offset_deg:+.1f} "
                    f"jaw_span_mm={float(width_detail['observed_width_mm']):.1f}"
                )
                route, route_reason, clearance = \
                    _plan_clearance_axial_trajectory(
                        nominal_endpoint, candidate_mapping, candidate_rpy,
                        log=log, phase=f"orientation_{orientation_attempt}",
                    )
                if route is None:
                    failures.append(
                        f"pitch {pitch_offset_deg:+.1f}deg: {route_reason}"
                    )
                    continue
                log(
                    "stage=grasp_orientation_search status=accepted "
                    f"attempt={orientation_attempt}/{len(pitch_offsets_deg)} "
                    f"pitch_offset_deg={pitch_offset_deg:+.1f}"
                )
                return {
                    "ok": True,
                    "reason": (
                        "collision-free terminal pitch selected from measured "
                        f"3D object geometry; pitch offset "
                        f"{pitch_offset_deg:+.1f}deg"
                    ),
                    "geometry": candidate_geometry,
                    "body": candidate_geometry.grasp_detection,
                    "jaw_plan": candidate_jaw,
                    "requested_cam_mm": centered_target.copy(),
                    "endpoint_m": nominal_endpoint,
                    "waypoints": route,
                    "trajectory_reason": route_reason,
                    "clearance": clearance,
                    "reachability": nominal_reachability,
                    "slot_offset_mm": 0.0,
                    "body_name": "selected",
                    "axis_fraction": _selected_axis_fraction(
                        candidate_geometry
                    ),
                    "attempts": orientation_attempt,
                    "approach_rpy": candidate_rpy,
                    "camera_mapping": candidate_mapping,
                    "pitch_offset_deg": pitch_offset_deg,
                    "projected_jaw_span": width_detail,
                    "camera_mapping_detail": mapping_detail,
                }

    # Offset-major ordering first tests every independently measured body
    # section at the calibrated jaw centre, then expands laterally by increasing
    # measured slot displacement. A dense slot grid therefore cannot consume
    # the whole compute budget before alternate cross-sections are considered.
    for offset_mm in slot_offsets:
        for body_name, body, score, candidate_geometry, candidate_jaw \
                in valid_bodies:
            # The already-tested nominal pose is the reason this search exists.
            if body_name == "selected" and abs(float(offset_mm)) < 1e-6:
                continue
            if attempts >= max_candidates:
                break
            attempts += 1
            target_cam = centered_target.copy()
            target_cam[:2] += closing_axis_cam * float(offset_mm)
            effective_cam, reachability = _reachable_approach_target(
                body, camera_mapping, target_cam
            )
            endpoint_raw = reachability.get("requested_endpoint_m")
            if effective_cam is None or endpoint_raw is None \
                    or not reachability.get("full_hover"):
                reason = str(reachability.get("reason"))
                failures.append(
                    f"{body_name}@{offset_mm:+.1f}mm: {reason}"
                )
                log(
                    "stage=grasp_pose_candidate status=rejected "
                    f"attempt={attempts}/{max_candidates} body={body_name} "
                    f"slot_offset_mm={offset_mm:+.1f} detail={reason}"
                )
                continue
            endpoint = np.asarray(endpoint_raw, dtype=float)
            endpoint_gate = A.check_pose_clearance(
                *endpoint.tolist(), *approach_rpy
            )
            if not endpoint_gate.ok:
                failures.append(
                    f"{body_name}@{offset_mm:+.1f}mm: {endpoint_gate.reason}"
                )
                log(
                    "stage=grasp_pose_candidate status=endpoint_blocked "
                    f"attempt={attempts}/{max_candidates} body={body_name} "
                    f"slot_offset_mm={offset_mm:+.1f} "
                    f"detail={endpoint_gate.reason}"
                )
                continue
            log(
                "stage=grasp_pose_candidate status=route_checking "
                f"attempt={attempts}/{max_candidates} body={body_name} "
                f"slot_offset_mm={offset_mm:+.1f} score={score:.3f}"
            )
            route, route_reason, clearance = \
                _plan_clearance_axial_trajectory(
                    endpoint, camera_mapping, approach_rpy, log=log,
                    phase=f"pose_{attempts}",
                )
            if route is None:
                failures.append(
                    f"{body_name}@{offset_mm:+.1f}mm: {route_reason}"
                )
                continue
            log(
                "stage=grasp_pose_search status=accepted "
                f"attempt={attempts}/{max_candidates} body={body_name} "
                f"slot_offset_mm={offset_mm:+.1f}"
            )
            return {
                "ok": True,
                "reason": (
                    "collision-free grasp selected from measured object "
                    f"geometry; body={body_name}, jaw-slot offset "
                    f"{offset_mm:+.1f}mm"
                ),
                "geometry": candidate_geometry,
                "body": body,
                "jaw_plan": candidate_jaw,
                "requested_cam_mm": target_cam,
                "endpoint_m": endpoint,
                "waypoints": route,
                "trajectory_reason": route_reason,
                "clearance": clearance,
                "reachability": reachability,
                "slot_offset_mm": float(offset_mm),
                "body_name": body_name,
                "axis_fraction": _selected_axis_fraction(candidate_geometry),
                "attempts": attempts,
            }
        if attempts >= max_candidates:
            break
    return {
        "ok": False,
        "reason": (
            f"no collision-free grasp pose among {attempts} candidates derived "
            "from measured cross-sections and jaw slack"
        ),
        "attempts": attempts,
        "failures": failures[-8:],
        "jaw_slot_half_slack_mm": across_limit_mm,
    }


def _ready_cross_section_range_mm(
    geometry: GG.GraspGeometry, initial_width_mm: float,
) -> tuple[float, float, list[float]]:
    """Return the range of feasible local widths observed before motion."""
    widths = [float(initial_width_mm)]
    candidates = getattr(geometry, "candidates", [])
    if isinstance(candidates, list):
        for item in candidates:
            try:
                width = float(item["width_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(width) and width > 0:
                widths.append(width)
    widths = sorted(widths)
    return widths[0], widths[-1], widths


def _endpoint_width_consensus(
    samples: list[dict[str, Any]], ready_min_mm: float, ready_max_mm: float,
    *, min_frames: int | None = None, max_spread_mm: float | None = None,
    ready_tolerance_mm: float | None = None,
    allow_stable_narrow_override: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a stable endpoint-width cluster compatible with ready geometry."""
    required = int(np.clip(
        DYNAMIC_WIDTH_CONSENSUS_MIN_FRAMES if min_frames is None else min_frames,
        2, 7,
    ))
    spread_limit = float(np.clip(
        DYNAMIC_WIDTH_CONSENSUS_MAX_SPREAD_MM
        if max_spread_mm is None else max_spread_mm,
        0.5, 10.0,
    ))
    tolerance = float(np.clip(
        DYNAMIC_REMEASURE_MAX_DELTA_MM
        if ready_tolerance_mm is None else ready_tolerance_mm,
        0.5, 12.0,
    ))
    valid = []
    for sample in samples:
        try:
            width = float(sample["width_mm"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(width) and width > 0:
            valid.append((width, sample))
    valid.sort(key=lambda item: item[0])
    if len(valid) < required:
        return None, {
            "reason": f"only {len(valid)}/{required} valid endpoint widths",
            "widths_mm": [round(item[0], 2) for item in valid],
        }
    clusters: list[list[tuple[float, dict[str, Any]]]] = []
    for left in range(len(valid)):
        cluster = []
        for right in range(left, len(valid)):
            if valid[right][0] - valid[left][0] > spread_limit:
                break
            cluster.append(valid[right])
        if len(cluster) >= required:
            clusters.append(cluster)
    if not clusters:
        return None, {
            "reason": (
                f"no {required}-frame width cluster is within the "
                f"{spread_limit:.1f}mm spread limit"
            ),
            "widths_mm": [round(item[0], 2) for item in valid],
        }
    cluster = max(
        clusters,
        key=lambda group: (len(group), -(group[-1][0] - group[0][0])),
    )
    median_mm = float(np.median([item[0] for item in cluster]))
    lower = float(ready_min_mm) - tolerance
    upper = float(ready_max_mm) + tolerance
    ordinarily_compatible = lower <= median_mm <= upper
    minimum_graspable_mm = _minimum_commandable_aperture_mm()
    if minimum_graspable_mm is None:
        minimum_graspable_mm = math.inf
    # A close-range measurement is made at the exact stationary jaw pose and
    # can legitimately see a narrower local section than the distant ready
    # view. Let it override only downward, only when all five independent
    # endpoint frames form a very tight cluster, and never when it is less than
    # 60% of the smallest ready-view cross-section. This reuses observations
    # already required by the dynamic close and adds no inference or latency.
    all_frame_override = bool(
        not ordinarily_compatible
        and median_mm < lower
        and len(cluster) >= 5
        and len(cluster) == len(valid)
        and cluster[-1][0] - cluster[0][0] <= min(1.0, spread_limit)
        and median_mm > max(
            float(minimum_graspable_mm), 0.60 * float(ready_min_mm)
        )
        and median_mm <= float(ready_max_mm)
    )
    # When the caller has independently established a stable 3-D body pose at
    # the exact open-jaw endpoint, that close-range local section is stronger
    # evidence than a coarse far-view OBB thickness. Permit one outlier among
    # five frames, but retain a tight width cluster, the physical minimum-size
    # gate, and all later complete-object/jaw-slot checks. The default remains
    # false so width alone can never select an unrelated narrow feature.
    stable_body_override = bool(
        allow_stable_narrow_override
        and not ordinarily_compatible
        and median_mm < lower
        and len(cluster) >= max(required, int(math.ceil(0.8 * len(valid))))
        and cluster[-1][0] - cluster[0][0] <= min(1.5, spread_limit)
        and median_mm > float(minimum_graspable_mm)
        and median_mm <= float(ready_max_mm)
    )
    authoritative_override = all_frame_override or stable_body_override
    compatible = ordinarily_compatible or authoritative_override
    representative = min(cluster, key=lambda item: abs(item[0] - median_mm))[1]
    acceptance = (
        "authoritative close-range override"
        if authoritative_override else
        ("ready-compatible consensus" if ordinarily_compatible else "rejected")
    )
    detail = {
        "reason": (
            f"{len(cluster)}/{len(valid)} stable endpoint frames; median "
            f"{median_mm:.1f}mm, spread {cluster[-1][0] - cluster[0][0]:.1f}mm; "
            f"ready cross-section range {ready_min_mm:.1f}-{ready_max_mm:.1f}mm "
            f"with +/-{tolerance:.1f}mm allowance; {acceptance}"
        ),
        "ok": compatible,
        "acceptance": acceptance,
        "authoritative_endpoint_override": authoritative_override,
        "stable_body_endpoint_override": stable_body_override,
        "median_mm": median_mm,
        "spread_mm": float(cluster[-1][0] - cluster[0][0]),
        "widths_mm": [round(item[0], 2) for item in valid],
        "ready_range_mm": [float(ready_min_mm), float(ready_max_mm)],
        "compatible_range_mm": [lower, upper],
    }
    if not compatible:
        return None, detail
    return representative, detail


def _endpoint_body_pose_consensus(
    samples: list[dict[str, Any]], *, min_frames: int = 3,
    max_spread_mm: float = 3.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fuse the actual grasp-body section seen between the open fingers."""
    points = []
    for sample in samples:
        try:
            point = np.asarray(sample["body_cam_mm"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if point.shape == (3,) and np.all(np.isfinite(point)):
            points.append(point)
    required = int(np.clip(min_frames, 2, 7))
    if len(points) < required:
        return None, {
            "reason": f"only {len(points)}/{required} valid grasp-body poses"
        }
    limit = float(np.clip(max_spread_mm, 0.5, 8.0))
    # Select the largest independently stable subset. The previous full-set
    # median made ``min_frames=2`` ineffective: one bad depth frame could reject
    # two agreeing observations. Keep the physical tolerance unchanged and
    # discard only samples that cannot belong to any largest in-limit cluster.
    selected_indices: tuple[int, ...] | None = None
    median: np.ndarray | None = None
    spread = float("inf")
    for cluster_size in range(len(points), required - 1, -1):
        candidates: list[tuple[float, tuple[int, ...], np.ndarray]] = []
        for indices in itertools.combinations(range(len(points)), cluster_size):
            cluster = np.stack([points[index] for index in indices])
            candidate_median = np.median(cluster, axis=0)
            candidate_spread = max(
                float(np.linalg.norm(point - candidate_median))
                for point in cluster
            )
            if candidate_spread <= limit:
                candidates.append(
                    (candidate_spread, indices, candidate_median)
                )
        if candidates:
            spread, selected_indices, median = min(
                candidates, key=lambda candidate: candidate[0]
            )
            break
    if selected_indices is None or median is None:
        all_median = np.median(np.stack(points), axis=0)
        all_spread = max(
            float(np.linalg.norm(point - all_median)) for point in points
        )
        return None, {
            "reason": (
                f"no {required}-frame grasp-body cluster is within the "
                f"{limit:.1f}mm stationary limit; full-set spread "
                f"{all_spread:.1f}mm"
            ),
            "spread_mm": all_spread,
            "valid_frames": len(points),
        }
    stable_count = len(selected_indices)
    return median, {
        "reason": (
            f"{stable_count}/{len(points)} stable grasp-body frames; median "
            f"camera XYZ {np.round(median, 1).tolist()}mm; spread "
            f"{spread:.1f}mm within the unchanged {limit:.1f}mm limit"
        ),
        "spread_mm": spread,
        "median_camera_mm": median.tolist(),
        "stable_frames": stable_count,
        "valid_frames": len(points),
        "rejected_frame_indices": [
            index for index in range(len(points))
            if index not in selected_indices
        ],
    }


def _checkpoint_section_consensus(
    locked_samples: list[dict[str, Any]],
    fresh_samples: list[dict[str, Any]],
    *, section_locked: bool, min_frames: int = 2,
    max_spread_mm: float = 4.0,
) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    """Prefer a locked body section, or safely reselect stable fresh geometry.

    Reselection does not authorize motion.  Its result is subsequently used to
    rebuild the endpoint and validate the complete IK/collision path.  This
    helper only prevents a perspective-dependent profile sample from becoming
    a permanent object identity.
    """
    locked_body, locked_detail = _endpoint_body_pose_consensus(
        locked_samples, min_frames=min_frames, max_spread_mm=max_spread_mm
    )
    if locked_body is not None:
        return locked_body, locked_detail, locked_samples
    if section_locked:
        fresh_body, fresh_detail = _endpoint_body_pose_consensus(
            fresh_samples, min_frames=min_frames,
            max_spread_mm=max_spread_mm,
        )
        if fresh_body is not None:
            detail = {
                **fresh_detail,
                "section_reselected": True,
                "locked_section_reason": locked_detail.get("reason"),
                "reason": (
                    "previous collision-selected subsection was not "
                    "identifiable from the closer view; stable fresh body "
                    "geometry accepted for a new endpoint and full collision "
                    f"replan; {fresh_detail.get('reason')}"
                ),
            }
            return fresh_body, detail, fresh_samples
    return None, locked_detail, locked_samples


def _checkpoint_target_position_consensus(
    samples: list[dict[str, Any]], anchor_base_m: Any,
    *, min_frames: int = 2,
) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    """Confirm a known target without re-estimating its close-range shape.

    The registered hybrid model already supplies object dimensions and a fully
    checked final trajectory.  Near the jaws, semantic/depth masks can spill
    onto the table even while their target centroids remain useful.  This gate
    accepts only a repeatable base-frame position near the original semantic
    anchor; it never turns those close masks into a new width or payload box.
    """
    try:
        anchor = np.asarray(anchor_base_m, dtype=float)
    except (TypeError, ValueError):
        anchor = np.array([], dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        return None, {"reason": "hybrid semantic target anchor is unavailable"}, []
    anchor_limit_m = float(np.clip(float(os.getenv(
        "REBOT_HYBRID_CHECKPOINT_ANCHOR_TOL_M", "0.025"
    )), 0.010, 0.040))
    spread_limit_m = float(np.clip(float(os.getenv(
        "REBOT_HYBRID_CHECKPOINT_POSITION_SPREAD_M", "0.015"
    )), 0.005, 0.020))
    candidates: list[dict[str, Any]] = []
    for sample in samples:
        try:
            point = np.asarray(sample["point_base_m"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        distance = float(np.linalg.norm(point - anchor))
        if distance <= anchor_limit_m:
            candidates.append({**sample, "anchor_distance_m": distance})
    required = int(np.clip(min_frames, 2, 3))
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    for size in range(len(candidates), required - 1, -1):
        for indices in itertools.combinations(range(len(candidates)), size):
            points = np.stack([
                np.asarray(candidates[index]["point_base_m"], dtype=float)
                for index in indices
            ])
            median = np.median(points, axis=0)
            spread = max(float(np.linalg.norm(point - median))
                         for point in points)
            if spread <= spread_limit_m and (
                best is None or len(indices) > len(best[1])
                or (len(indices) == len(best[1]) and spread < best[0])
            ):
                best = (spread, indices, median)
        if best is not None:
            break
    if best is None:
        return None, {
            "reason": (
                f"no {required}-frame target-position cluster stayed within "
                f"{anchor_limit_m * 1000.0:.1f}mm of the original semantic "
                f"anchor and {spread_limit_m * 1000.0:.1f}mm stationary spread"
            ),
            "valid_near_anchor_frames": len(candidates),
        }, []
    spread, indices, median = best
    selected = [candidates[index] for index in indices]
    return median, {
        "reason": (
            f"{len(selected)}/{len(samples)} target-position frames confirm "
            "the original hybrid target; shape/width retained from registered "
            "multi-view geometry"
        ),
        "median_base_m": median.round(5).tolist(),
        "spread_mm": round(spread * 1000.0, 1),
        "max_anchor_distance_mm": round(max(
            sample["anchor_distance_m"] for sample in selected
        ) * 1000.0, 1),
    }, selected


def _checkpoint_metric_width_is_authoritative(
    width_detail: dict[str, Any] | None,
    object_box_3d: dict[str, Any] | None,
    max_aperture_mm: float,
    position_consensus: np.ndarray | None,
) -> tuple[bool, str]:
    """Gate reuse of a registered 3-D jaw width at a close checkpoint.

    A close semantic mask can include the table or fingers and make an object
    appear wider than it was in the registered multi-view model.  Reuse the
    metric jaw axis only when it was physically aligned with the jaws, its
    conservative width fits the calibrated aperture, the source box still
    exists, and repeated close observations confirm the same target position.
    ``override_applied`` is deliberately irrelevant: an already-correct
    initial hybrid silhouette needs no override, but the same measured width
    must remain authoritative later when perspective becomes less reliable.
    """
    if not isinstance(object_box_3d, dict):
        return False, "registered metric object box is unavailable"
    if not isinstance(width_detail, dict) or not width_detail.get("ok"):
        return False, "jaw-aligned metric width was not verified"
    try:
        observed_mm = float(width_detail["observed_width_mm"])
        conservative_mm = float(width_detail["conservative_width_mm"])
        aperture_mm = float(max_aperture_mm)
    except (KeyError, TypeError, ValueError):
        return False, "jaw-aligned metric width is incomplete"
    if not math.isfinite(observed_mm) or not math.isfinite(conservative_mm) \
            or not math.isfinite(aperture_mm):
        return False, "jaw-aligned metric width is non-finite"
    if conservative_mm > aperture_mm:
        return False, (
            f"metric width {conservative_mm:.1f}mm exceeds the "
            f"{aperture_mm:.1f}mm jaw aperture"
        )
    if position_consensus is None:
        return False, "close observations did not confirm the registered target"
    return True, (
        f"verified jaw-aligned 3D width {observed_mm:.1f}mm "
        f"({conservative_mm:.1f}mm conservative) retained for the same target"
    )


def _endpoint_measurement_consensus(
    locked_samples: list[dict[str, Any]],
    fresh_samples: list[dict[str, Any]],
    ready_min_mm: float, ready_max_mm: float,
    *, section_locked: bool,
) -> dict[str, Any]:
    """Fuse close-range width/body data, with safe section reselection.

    A profile subsection selected from a distant mask is not a durable visual
    identity. If it disappears, a fresh close-range section is accepted only
    when both its width and its 3D body position independently form the normal
    multi-frame consensus. The caller still applies the physical jaw-slot gate
    and collision-checked alignment before any close command.
    """
    def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
        body_cam, body_detail = _endpoint_body_pose_consensus(
            samples, min_frames=DYNAMIC_WIDTH_CONSENSUS_MIN_FRAMES
        )
        selected, width_detail = _endpoint_width_consensus(
            samples, ready_min_mm, ready_max_mm,
            allow_stable_narrow_override=body_cam is not None,
        )
        return {
            "samples": samples,
            "selected": selected,
            "width_detail": width_detail,
            "body_cam": body_cam,
            "body_detail": body_detail,
            "section_reselected": False,
        }

    locked = summarize(locked_samples)
    if locked["selected"] is not None and locked["body_cam"] is not None:
        return locked
    if section_locked:
        fresh = summarize(fresh_samples)
        if fresh["selected"] is not None and fresh["body_cam"] is not None:
            fresh["section_reselected"] = True
            fresh["locked_width_reason"] = locked["width_detail"].get(
                "reason"
            )
            fresh["locked_body_reason"] = locked["body_detail"].get(
                "reason"
            )
            fresh["width_detail"] = {
                **fresh["width_detail"],
                "reason": (
                    "previous collision-selected subsection was not "
                    "measurable at the open jaws; stable fresh close-range "
                    "width/body geometry reselected; "
                    f"{fresh['width_detail'].get('reason')}"
                ),
            }
            return fresh
    return locked


def _ready_geometry_endpoint_fallback(
    measurement: dict[str, Any], jaw_plan: dict[str, Any],
    ready_geometry: GG.GraspGeometry, ready_max_mm: float,
    object_box_3d: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recover from close-range jaw occlusion without inventing a new width.

    Open fingers can enter the RGB-D component and make every endpoint width
    larger even though the physical body position remains stable.  In that
    narrowly identifiable case, retain the stationary preplanned width and use
    the fresh body consensus only for alignment.  Mixed/narrow measurements,
    an unstable body, a failed full-object fit, or too few observations remain
    fail-closed.
    """
    failure = lambda reason: {"ok": False, "reason": reason}
    body = measurement.get("body_cam")
    try:
        body_cam = np.asarray(body, dtype=float)
    except (TypeError, ValueError):
        return failure("close-range grasp body is unavailable")
    if body_cam.shape != (3,) or not np.all(np.isfinite(body_cam)):
        return failure("close-range grasp body is not a stable finite XYZ")
    samples = measurement.get("samples")
    if not isinstance(samples, list):
        return failure("close-range endpoint samples are unavailable")
    valid: list[tuple[float, dict[str, Any]]] = []
    for sample in samples:
        try:
            width = float(sample["width_mm"])
            sample_body = np.asarray(sample["body_cam_mm"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(width) and width > 0.0 \
                and sample_body.shape == (3,) \
                and np.all(np.isfinite(sample_body)):
            valid.append((width, sample))
    required = int(np.clip(DYNAMIC_WIDTH_CONSENSUS_MIN_FRAMES, 2, 7))
    if len(valid) < required:
        return failure(
            f"only {len(valid)}/{required} valid close-range widths"
        )
    inflation_floor = float(ready_max_mm) + float(np.clip(
        DYNAMIC_REMEASURE_MAX_DELTA_MM, 0.5, 12.0
    ))
    widths = [item[0] for item in valid]
    if any(width <= inflation_floor for width in widths):
        return failure(
            "close-range widths are not consistently inflated beyond the "
            "ready-view uncertainty bound"
        )
    fit_ok, fit_detail = _validate_object_gripper_fit(
        ready_geometry, jaw_plan, object_box_3d
    )
    if not fit_ok:
        return failure(
            "stable ready-view geometry no longer passes the complete-object "
            f"fit gate: {fit_detail.get('reason')}"
        )
    representative = min(valid, key=lambda item: float(np.linalg.norm(
        np.asarray(item[1]["body_cam_mm"], dtype=float) - body_cam
    )))[1]
    ready_body = ready_geometry.grasp_detection
    if ready_body is None:
        return failure("stable ready-view grasp cross-section is unavailable")
    updated_body = replace(
        ready_body,
        p_cam_mm=tuple(float(value) for value in body_cam),
        info={
            **getattr(ready_body, "info", {}),
            "endpoint_width_basis": "stable_ready_view_jaw_occlusion_fallback",
            "rejected_close_widths_mm": [round(value, 2) for value in widths],
        },
    )
    updated_geometry = replace(
        ready_geometry, grasp_detection=updated_body,
        reason=(
            f"{ready_geometry.reason}; retained stable ready-view width after "
            "repeated close-range jaw occlusion"
        ),
    )
    selected = {
        **representative,
        "width_mm": float(jaw_plan["min_width_mm"]),
        "body_cam_mm": body_cam.tolist(),
        "plan": dict(jaw_plan),
        "geometry": updated_geometry,
    }
    return {
        "ok": True,
        "selected": selected,
        "body_cam": body_cam,
        "body_detail": measurement.get("body_detail", {}),
        "reason": (
            "stable close-range body retained with the preplanned multi-frame "
            f"width {float(jaw_plan['min_width_mm']):.1f}mm; rejected "
            f"consistently inflated endpoint widths "
            f"{[round(value, 2) for value in widths]} as jaw occlusion"
        ),
    }


def _locked_identifier_body_slot_fallback(
    identifier_cam: Any, identifier_detail: dict[str, Any],
    feature_to_body_cam: Any, requested_cam: Any,
    jaw_plan: dict[str, Any], geometry: GG.GraspGeometry,
    object_box_3d: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover body XYZ when open fingers destabilize only the body mask.

    This does not use the identifier itself as the grasp point.  It transports
    the already measured and locked identifier-to-body offset, then applies
    the normal calibrated jaw-slot gate to the inferred physical body section.
    Width authorization remains the caller's independent multi-frame depth
    consensus.
    """
    failure = lambda reason: {"ok": False, "reason": reason}
    try:
        identifier = np.asarray(identifier_cam, dtype=float)
        offset = np.asarray(feature_to_body_cam, dtype=float)
    except (TypeError, ValueError):
        return failure("stable identifier or locked feature/body offset unavailable")
    if identifier.shape != (3,) or offset.shape != (3,) \
            or not np.all(np.isfinite(identifier)) \
            or not np.all(np.isfinite(offset)):
        return failure("stable identifier or locked feature/body offset is invalid")
    if int(identifier_detail.get("stable_frames", 0)) < 2:
        return failure(
            "semantic identifier lacks a stable independent multi-frame pose"
        )
    body_cam = identifier - offset
    slot_ok, slot_detail = _jaw_slot_fit(
        body_cam, requested_cam, jaw_plan, geometry, object_box_3d,
    )
    return {
        "ok": bool(slot_ok),
        "reason": (
            "open-finger body-mask jitter bypassed using the locked semantic "
            "feature/body transform; " + str(slot_detail.get("reason"))
        ),
        "body_cam": body_cam,
        "slot_ok": bool(slot_ok),
        "slot_detail": slot_detail,
        "identifier_detail": identifier_detail,
    }


def _observe_full_geometry_consensus(
    eye: "WristEye", description: str,
    current_rpy: tuple[float, float, float],
    log: Callable[[str], None],
    *, min_frames: int = 2, max_frames: int = 4,
    camera_mapping: np.ndarray | None = None,
    tool_camera_mm: Any | None = None,
    table_z_m: float | None = None,
    stationary_ee_xyz_m: Any | None = None,
    metric_jaw_width_mm: float | None = None,
    metric_major_mm: float | None = None,
    prefer_exact_target_mask: bool = False,
    allow_registered_box_fallback: bool = False,
) -> tuple[Any | None, Any | None, GG.GraspGeometry | None, dict[str, Any]]:
    """Require repeated full-object geometry at one stationary arm pose.

    A single depth frame can merge a tabletop object into the table. We retry
    that sensor failure. Hybrid mode may retain a shape already established by
    registered native semantic masks when repeated named-object centroids
    confirm its new position; legacy mode never carries shape across motion.
    """
    samples: list[dict[str, Any]] = []
    semantic_positions: list[dict[str, Any]] = []
    failures: list[str] = []
    required = int(np.clip(min_frames, 2, 3))
    total = int(np.clip(max_frames, required, 6))

    def registered_position_fallback() -> tuple[
        Any | None, Any | None, GG.GraspGeometry | None, dict[str, Any]
    ] | None:
        if not allow_registered_box_fallback or not samples:
            return None
        position, position_detail = _endpoint_body_pose_consensus(
            semantic_positions, min_frames=required, max_spread_mm=5.0
        )
        if position is None:
            return None
        representative = min(
            samples,
            key=lambda sample: float(np.linalg.norm(
                np.asarray(sample["feature"].p_cam_mm, dtype=float)
                - position
            )),
        )
        geometry_out = representative["geometry"]
        body = geometry_out.grasp_detection
        if body is None:
            return None
        feature_to_body = (
            np.asarray(representative["feature"].p_cam_mm, dtype=float)
            - np.asarray(body.p_cam_mm, dtype=float)
        )
        confirmed_body = np.asarray(position, dtype=float) - feature_to_body
        updated_body = replace(
            body, p_cam_mm=tuple(float(value) for value in confirmed_body),
            info={
                **(getattr(body, "info", {}) or {}),
                "position_basis": "stable_semantic_identifier_registered_box",
            },
        )
        geometry_out = replace(
            geometry_out, grasp_detection=updated_body,
            object_box_3d=None,
            reason=(
                f"{geometry_out.reason}; registered multi-view shape retained "
                "after stable semantic position confirmation"
            ),
        )
        confirmed_feature = replace(
            representative["feature"],
            p_cam_mm=tuple(float(value) for value in position),
            z_mm=float(position[2]),
            info={
                **(getattr(representative["feature"], "info", {}) or {}),
                "position_basis": "stable_semantic_identifier_consensus",
            },
        )
        return (
            confirmed_feature, representative["frame"], geometry_out,
            {
                **position_detail,
                "reason": (
                    f"{position_detail.get('reason')}; one complete semantic "
                    "geometry frame plus the registered multi-view object box "
                    "retained shape while repeated semantic centroids confirmed "
                    "position"
                ),
                "registered_box_fallback": True,
                "valid_full_geometry_frames": len(samples),
                "semantic_position_frames": len(semantic_positions),
                "failures": failures,
            },
        )

    for index in range(1, total + 1):
        feature, frame, why = _observe(eye, description)
        if feature is None or frame is None:
            failures.append(str(why))
            log(
                "stage=preplan_geometry_sample "
                f"frame={index}/{total} ok=False detail={why}"
            )
            continue
        feature_p_cam = getattr(feature, "p_cam_mm", None)
        if str(getattr(feature, "source", "")) in _SEMANTIC_IDENTITY_SOURCES \
                and feature_p_cam is not None:
            semantic_position = np.asarray(feature_p_cam, dtype=float)
            if semantic_position.shape == (3,) \
                    and np.all(np.isfinite(semantic_position)):
                semantic_positions.append({
                    "body_cam_mm": semantic_position.tolist(),
                })
        geometry = _plan_scene_grasp_geometry(
            frame, feature, current_rpy=current_rpy,
            allow_orientation=False, object_name=description,
            metric_jaw_width_mm=metric_jaw_width_mm,
            metric_major_mm=metric_major_mm,
        )
        body = geometry.grasp_detection
        if not geometry.ok or body is None:
            failures.append(geometry.reason)
            log(
                "stage=preplan_geometry_sample "
                f"frame={index}/{total} ok=False detail={geometry.reason}"
            )
            fallback = registered_position_fallback()
            if fallback is not None:
                log(
                    "stage=preplan_geometry_registered_fallback ok=True "
                    f"detail={fallback[3].get('reason')}"
                )
                return fallback
            continue
        samples.append({
            "body_cam_mm": np.asarray(body.p_cam_mm, dtype=float).tolist(),
            "width_mm": float(body.min_width_mm or 0.0),
            "feature": feature, "frame": frame, "geometry": geometry,
        })
        log(
            "stage=preplan_geometry_sample "
            f"frame={index}/{total} ok=True body_xyz_mm="
            f"{np.round(body.p_cam_mm, 1).tolist()} "
            f"width_mm={float(body.min_width_mm or 0.0):.1f}"
        )
        median, consensus = _endpoint_body_pose_consensus(
            samples, min_frames=required, max_spread_mm=5.0
        )
        if median is None:
            fallback = registered_position_fallback()
            if fallback is not None:
                log(
                    "stage=preplan_geometry_registered_fallback ok=True "
                    f"detail={fallback[3].get('reason')}"
                )
                return fallback
            continue
        rejected = set(consensus.get("rejected_frame_indices", []))
        stable = [sample for sample_index, sample in enumerate(samples)
                  if sample_index not in rejected]
        widths = np.asarray([sample["width_mm"] for sample in stable], dtype=float)
        median_width = float(np.median(widths))
        width_limit = max(3.0, 0.20 * max(1.0, median_width))
        width_spread = float(np.max(np.abs(widths - median_width)))
        if width_spread > width_limit:
            failures.append(
                f"stable positions disagreed on width by {width_spread:.1f}mm"
            )
            continue
        representative = min(stable, key=lambda sample: float(np.linalg.norm(
            np.asarray(sample["body_cam_mm"], dtype=float) - median
        )))
        geometry_out = representative["geometry"]
        # Position uses the frame nearest the stable 3D median, but jaw entry
        # must use the widest credible observation. Choosing all fields from
        # the positional representative let a narrow-noise frame understate a
        # 43.6--48.5 mm sponge as only 43.6 mm.
        conservative_width = float(np.max(widths))
        if geometry_out.grasp_detection is not None:
            geometry_out.grasp_detection.min_width_mm = conservative_width
            geometry_out.grasp_detection.info = {
                **getattr(geometry_out.grasp_detection, "info", {}),
                "stable_width_max_mm": conservative_width,
                "stable_width_median_mm": median_width,
            }
        object_box_detail = None
        if camera_mapping is not None and tool_camera_mm is not None \
                and table_z_m is not None:
            box_started = time.monotonic()
            log(
                "stage=object_box_3d status=building "
                f"stable_frames={len(stable)} arm=stationary"
            )
            point_sets = []
            box_failures = []
            # Every accepted frame above was captured without arm motion. Read
            # the arm pose once, rather than reacquiring the CAN/controller lock
            # once per frame during a purely geometric calculation.
            log("stage=object_box_3d status=reading_stationary_pose")
            stationary_ee_xyz = (
                np.asarray(stationary_ee_xyz_m, dtype=float)
                if stationary_ee_xyz_m is not None else
                np.asarray(A.ee_pos(), dtype=float)
            )
            log(
                "stage=object_box_3d status=stationary_pose_ready "
                f"ee_xyz={[round(float(v), 4) for v in stationary_ee_xyz]}"
            )
            for sample in stable:
                envelope = sample["geometry"].envelope_detection \
                    or sample["geometry"].grasp_detection
                box_detection = envelope
                mask_basis = "grasp_depth_envelope"
                if prefer_exact_target_mask:
                    feature = sample["feature"]
                    feature_source = str(getattr(feature, "source", ""))
                    # SAM3's native mask is consumed while constructing the
                    # cleaned RGB-D semantic envelope; the lightweight feature
                    # Detection intentionally carries only its box/centroid.
                    # Therefore inspect the accepted geometry envelope rather
                    # than expecting the original feature to retain mask_u8.
                    envelope_info = getattr(envelope, "info", {}) or {}
                    envelope_mask = envelope_info.get("mask_u8")
                    segmentation_source = str(
                        envelope_info.get("segmentation_source", "")
                    )
                    exact_semantic_envelope = (
                        str(getattr(envelope, "source", ""))
                        == "rgbd:semantic_envelope"
                        and segmentation_source.endswith(":native_mask")
                        and envelope_mask is not None
                    )
                    if feature_source not in _SEMANTIC_IDENTITY_SOURCES \
                            or not exact_semantic_envelope:
                        box_failures.append(
                            "stationary hybrid confirmation requires an exact "
                            "semantic target mask"
                        )
                        log(
                            "stage=object_box_3d status=frame_skipped "
                            f"source={feature_source or 'unknown'} "
                            "detail=exact_semantic_target_mask_unavailable"
                        )
                        continue
                    # Hybrid discovery has already established target identity
                    # and metric scale from registered views.  Confirm that
                    # model with the detector's exact object mask.  The grasp
                    # envelope intentionally includes nearby depth support and
                    # is useful for choosing a cross-section, but lifting it as
                    # payload geometry can absorb much of the table or scene.
                    box_detection = envelope
                    mask_basis = "exact_semantic_native_mask"
                try:
                    points_base = OG3.masked_points_base(
                        sample["frame"], box_detection,
                        ee_xyz_m=stationary_ee_xyz,
                        camera_mapping_mm_per_m=camera_mapping,
                        tool_camera_mm=tool_camera_mm,
                        near_mm=float(WD.NEAR_MM), far_mm=float(WD.FAR_MM),
                    )
                    point_sets.append(points_base)
                    log(
                        "stage=object_box_3d status=frame_lifted "
                        f"view={len(point_sets)}/{len(stable)} "
                        f"points={len(points_base)} basis={mask_basis}"
                    )
                except Exception as exc:  # noqa: BLE001
                    box_failures.append(f"{type(exc).__name__}: {exc}")
            minimum_box_views = 2 if prefer_exact_target_mask else 1
            if len(point_sets) < minimum_box_views:
                failures.extend(box_failures)
                failures.append(
                    f"only {len(point_sets)}/{minimum_box_views} exact target "
                    "masks were available for stationary 3D confirmation"
                )
                fallback = registered_position_fallback()
                if fallback is not None:
                    log(
                        "stage=object_box_3d status=registered_fallback "
                        f"detail={fallback[3].get('reason')}"
                    )
                    return fallback
                continue
            consistent_indices, extent_consensus = \
                OG3.consistent_view_extent_indices(
                    point_sets, table_z_m=float(table_z_m)
                )
            log(
                "stage=object_box_3d_extent_consensus "
                f"ok={extent_consensus.get('ok')} "
                f"accepted={len(consistent_indices)}/{len(point_sets)} "
                f"extent_mm={extent_consensus.get('extent_mm')} "
                f"detail={extent_consensus.get('reason')}"
            )
            if not extent_consensus.get("ok"):
                failures.append(str(extent_consensus.get("reason")))
                # The arm is stationary and the outer observation loop can
                # capture another frame. Never fuse inconsistent silhouettes
                # into one oversized collision/payload box.
                continue
            point_sets = [point_sets[index] for index in consistent_indices]
            prior_vertices = None
            prior_reason = "disabled (REBOT_SAM3D_ENDPOINT is unset)"
            envelope = geometry_out.envelope_detection \
                or geometry_out.grasp_detection
            try:
                log(
                    "stage=object_box_3d status=shape_prior "
                    f"enabled={bool(os.getenv('REBOT_SAM3D_ENDPOINT', '').strip())}"
                )
                prior_mask = getattr(envelope, "info", {}).get("mask_u8")
                if prior_mask is None:
                    prior_reason = "unavailable: exact object mask is missing"
                else:
                    prior_vertices = OG3.request_sam3d_prior(
                        representative["frame"][0], prior_mask
                    )
                    if prior_vertices is not None:
                        prior_reason = f"received {len(prior_vertices)} vertices"
            except Exception as exc:  # noqa: BLE001
                # A learned completion service is advisory. Metric RGB-D still
                # produces a valid conservative box when that service is absent.
                prior_reason = f"rejected: {type(exc).__name__}: {exc}"
            try:
                log("stage=object_box_3d status=fitting_metric_box")
                object_box = OG3.fit_tabletop_box(
                    point_sets, table_z_m=float(table_z_m),
                    prior_vertices=prior_vertices,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"3D box fit failed: {type(exc).__name__}: {exc}")
                continue
            object_box_detail = object_box.as_dict()
            object_box_detail["shape_prior_status"] = prior_reason
            object_box_detail["view_extent_consensus"] = extent_consensus
            geometry_out = replace(
                geometry_out, object_box_3d=object_box_detail
            )
            log(
                "stage=object_box_3d ok=True source=metric_rgbd "
                f"views={object_box.views} points={object_box.point_count} "
                f"observed_mm="
                f"{(np.asarray(object_box.observed_size_m) * 1000).round(1).tolist()} "
                f"conservative_mm="
                f"{(np.asarray(object_box.conservative_size_m) * 1000).round(1).tolist()} "
                f"prior={prior_reason} "
                f"support_completed={object_box.support_completed} "
                f"elapsed_s={time.monotonic() - box_started:.2f}"
            )
        return (
            representative["feature"], representative["frame"],
            geometry_out, {
                **consensus,
                "reason": f"{consensus['reason']}; width median "
                          f"{median_width:.1f}mm, spread {width_spread:.1f}mm",
                "attempted_frames": index,
                "valid_full_geometry_frames": len(samples),
                "width_median_mm": median_width,
                "width_max_mm": conservative_width,
                "width_spread_mm": width_spread,
                "object_box_3d": object_box_detail,
                "failures": failures,
            },
        )
    return None, None, None, {
        "reason": f"no {required}-frame full-object geometry consensus in "
                  f"{total} stationary frames",
        "valid_full_geometry_frames": len(samples),
        "failures": failures,
    }


def _lift_hold_hypothesis(
    before_cam_mm: Any,
    after_cam_mm: Any,
    arm_delta_m: Any,
    rigid_jacobian: np.ndarray,
    *, tolerance_mm: float,
) -> dict[str, Any]:
    """Classify observed object motion as wrist-held, table-static or unclear."""
    observed_shift = (
        np.asarray(after_cam_mm, dtype=float)
        - np.asarray(before_cam_mm, dtype=float)
    )
    predicted_static_shift = (
        np.asarray(rigid_jacobian, dtype=float)
        @ np.asarray(arm_delta_m, dtype=float)
    )
    held_residual = float(np.linalg.norm(observed_shift))
    static_residual = float(np.linalg.norm(
        observed_shift - predicted_static_shift
    ))
    tolerance = float(np.clip(tolerance_mm, 3.0, 8.0))
    if held_residual <= tolerance:
        classification = "held"
    elif static_residual <= tolerance:
        classification = "static"
    else:
        classification = "ambiguous"
    return {
        "classification": classification,
        "held_residual_mm": held_residual,
        "static_residual_mm": static_residual,
        "observed_shift_mm": observed_shift.tolist(),
        "predicted_static_shift_mm": predicted_static_shift.tolist(),
        "tolerance_mm": tolerance,
    }


def _hold_observation_pair(
    before_body_cam_mm: Any | None,
    after_body_cam_mm: Any | None,
    before_feature_cam_mm: Any | None,
    after_feature_cam_mm: Any | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Choose the strongest visual evidence available for lift verification.

    The selected physical cross-section remains authoritative whenever visible.
    After lifting, the fingers can hide that exact slice while the semantic
    identifier or full-object mask remains stable. In that case the identifier
    is a valid rigid-motion witness for *hold verification only*; it never
    controls alignment or jaw closure.
    """
    for before_raw, after_raw, basis in (
        (before_body_cam_mm, after_body_cam_mm, "locked_grasp_section"),
        (before_feature_cam_mm, after_feature_cam_mm,
         "semantic_identifier_fallback"),
    ):
        if before_raw is None or after_raw is None:
            continue
        before = np.asarray(before_raw, dtype=float)
        after = np.asarray(after_raw, dtype=float)
        if before.shape == (3,) and after.shape == (3,) \
                and np.all(np.isfinite(before)) \
                and np.all(np.isfinite(after)):
            return before, after, basis
    return None, None, "unobserved"


def _post_close_hold_baseline(
    post_close_body_cam_mm: Any | None,
    post_close_detail: dict[str, Any],
) -> dict[str, Any]:
    """Select evidence for a bounded lift without requiring visible fingers.

    The physical grasp section has already passed multi-frame width, position,
    full-object fit, and jaw-slot gates immediately before closure. Closing the
    jaws often hides that exact section. A stable, identity-gated semantic
    feature observed *after* closure is therefore sufficient as the before-lift
    witness, but never as proof that the object is held. The following visual
    micro-lift must still classify that witness as rigid with the wrist rather
    than static on the table.
    """
    def finite_point(raw: Any | None) -> np.ndarray | None:
        if raw is None:
            return None
        try:
            point = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return None
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        return point

    body = finite_point(post_close_body_cam_mm)
    feature = finite_point(post_close_detail.get("feature_cam_mm"))
    if body is not None:
        return {
            "ok": True,
            "body_cam_mm": body,
            "feature_cam_mm": feature,
            "basis": "post_close_physical_body",
            "reason": (
                "stable post-close physical grasp section retained as the "
                "micro-lift baseline"
            ),
        }
    if feature is not None:
        return {
            "ok": True,
            "body_cam_mm": None,
            "feature_cam_mm": feature,
            "basis": "post_close_semantic_identifier",
            "reason": (
                "closed fingers occluded the pre-validated grasp section; "
                "stable identity-gated post-close semantic feature retained "
                "only as the micro-lift motion witness"
            ),
        }
    return {
        "ok": False,
        "body_cam_mm": None,
        "feature_cam_mm": None,
        "basis": "unobserved",
        "reason": (
            "neither the physical grasp section nor a stable identity-gated "
            "semantic feature was measurable after closure"
        ),
    }


def _locked_body_from_feature(
    feature_cam_mm: Any,
    feature_to_body_cam_mm: Any,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Recover an occluded locked grasp section from its semantic feature.

    Close-range alignment measures ``feature - body`` while both are visible.
    If the jaws later hide that exact body slice, a stable feature can recover
    its position without selecting another cross-section or estimating a new
    width. This bookkeeping is used only after the initial close has already
    been authorized from unobstructed multi-frame body geometry. It may provide
    the baseline for a bounded visual hold probe, but it can never authorize
    alignment, jaw closure, or claim that the object is held.
    """
    try:
        feature = np.asarray(feature_cam_mm, dtype=float)
        offset = np.asarray(feature_to_body_cam_mm, dtype=float)
    except (TypeError, ValueError):
        return None, {"reason": "locked feature/body geometry is unavailable"}
    if feature.shape != (3,) or offset.shape != (3,) \
            or not np.all(np.isfinite(feature)) \
            or not np.all(np.isfinite(offset)):
        return None, {"reason": "locked feature/body geometry is non-finite"}
    body = feature - offset
    return body, {
        "reason": (
            "recovered the occluded locked grasp section from the stable "
            "semantic feature; no new width was inferred"
        ),
        "body_cam_mm": body.tolist(),
    }


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
    pending_frame_errors: list[str] = []
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            color, depth, intr = eye.frame()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - bounded camera retry below
            detail = f"{type(exc).__name__}: {exc}"
            pending_frame_errors.append(detail)
            print(
                f"[wrist {time.strftime('%H:%M:%S')}] "
                f"stage=camera_frame_retry attempt={attempt + 1}/{attempts} "
                f"target={description!r} detail={detail}",
                flush=True,
            )
            if attempt + 1 < attempts:
                time.sleep(0.15)
            continue
        if pending_frame_errors:
            print(
                f"[wrist {time.strftime('%H:%M:%S')}] "
                f"stage=camera_frame_retry status=recovered "
                f"after_failures={len(pending_frame_errors)} "
                f"target={description!r}",
                flush=True,
            )
            pending_frame_errors.clear()
        det, why = eye.object(description, color, depth, intr)
        if det is not None:
            return det, (color, depth, intr), why
        last_why = why
        time.sleep(0.1)
    if pending_frame_errors:
        raise CameraFrameUnavailable(
            f"wrist RGB-D frame unavailable after {attempts} attempts; "
            f"consecutive failures={len(pending_frame_errors)}; "
            f"last error: {pending_frame_errors[-1]}"
        )
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
    *,
    log: Callable[[str], None] | None = None,
    phase: str = "approach",
) -> tuple[Any | None, str, dict[str, Any]]:
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
    emit = log if log is not None else (lambda _message: None)
    routes_per_backoff = 6
    total_attempts = len(backoffs_mm) * routes_per_backoff
    attempt = 0
    terminal_blocked_backoffs: list[float] = []
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
            attempt += 1
            combined: list[tuple[float, float, float]] = []
            leg_start = start.copy()
            route_failed = None
            terminal_first_segment = 1
            all_leg_endpoints = [*route_points, endpoint]
            for leg_index, leg_endpoint in enumerate(all_leg_endpoints):
                leg_endpoint = np.asarray(leg_endpoint, dtype=float)
                if float(np.linalg.norm(leg_endpoint - leg_start)) < 1e-7:
                    continue
                if leg_index == len(all_leg_endpoints) - 1:
                    # Collision-path segment numbers are one-based relative to
                    # the first waypoint after the current live configuration.
                    terminal_first_segment = len(combined) + 1
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
                emit(
                    f"stage=grasp_route_plan phase={phase} status=rejected "
                    f"attempt={attempt}/{total_attempts} route={route_name} "
                    f"backoff_mm={backoff_mm:.0f} detail="
                    f"{route_failed or 'empty path'}"
                )
                failures.append(
                    f"{route_name}@{backoff_mm:.0f}mm: "
                    f"{route_failed or 'empty path'}"
                )
                continue
            started = time.monotonic()
            emit(
                f"stage=grasp_route_plan phase={phase} status=validating "
                f"attempt={attempt}/{total_attempts} route={route_name} "
                f"backoff_mm={backoff_mm:.0f} waypoints={len(combined)} "
                "detail=deterministic IK and collision checks; arm stationary"
            )
            chain_ok, chain_reason = _validate_fixed_waypoints(
                combined, fixed_rpy
            )
            elapsed_s = time.monotonic() - started
            if chain_ok:
                emit(
                    f"stage=grasp_route_plan phase={phase} status=accepted "
                    f"attempt={attempt}/{total_attempts} route={route_name} "
                    f"elapsed_s={elapsed_s:.1f}"
                )
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
            emit(
                f"stage=grasp_route_plan phase={phase} status=rejected "
                f"attempt={attempt}/{total_attempts} route={route_name} "
                f"elapsed_s={elapsed_s:.1f} detail={chain_reason}"
            )
            failures.append(
                f"{route_name}@{backoff_mm:.0f}mm: {chain_reason}"
            )
            # Every deterministic candidate for this backoff has the same
            # camera-axial terminal leg. If the straight baseline reports that
            # an actual gripper link hits the scene on that suffix, lateral or
            # overhead prefixes cannot make the physical jaw geometry clear.
            # Try the next (shorter) backoff once, instead of spending minutes
            # validating five routes that necessarily end in the same contact.
            segment = None
            marker = "Cartesian path collision at segment "
            if marker in chain_reason:
                suffix = chain_reason.split(marker, 1)[1]
                try:
                    segment = int(suffix.split(",", 1)[0].strip())
                except ValueError:
                    segment = None
            terminal_gripper_collision = (
                route_name == "straight_clearance"
                and segment is not None
                and segment >= terminal_first_segment
                and "'robot_link': 'gripper_" in chain_reason
            )
            if terminal_gripper_collision:
                terminal_blocked_backoffs.append(backoff_mm)
                emit(
                    f"stage=grasp_route_plan phase={phase} "
                    "status=terminal_blocked "
                    f"backoff_mm={backoff_mm:.0f} segment={segment} "
                    "detail=shared final jaw-entry collides; skipping "
                    "alternate prefixes for this backoff"
                )
                break
    if len(terminal_blocked_backoffs) == len(backoffs_mm):
        return None, (
            "shared final jaw-entry is blocked at every bounded backoff; "
            "alternate corridors and global routing cannot change the required "
            "terminal gripper geometry: " + "; ".join(failures)
        ), {
            "terminal_approach_blocked": True,
            "tested_backoffs_mm": [float(value)
                                    for value in terminal_blocked_backoffs],
        }
    # Preserve a visible checkpoint during the first leg. After the target has
    # been freshly re-observed, the global fallback may route to the exact grasp
    # endpoint. It remains bounded, upright, collision checked, and stale-scene
    # certified by the local backend.
    global_endpoint = endpoint.copy()
    global_checkpoint_only = phase == "initial"
    if global_checkpoint_only:
        checkpoint_backoff_mm = float(np.clip(
            HYBRID_CHECKPOINT_M * 1000.0, 70.0, 120.0
        ))
        try:
            global_endpoint = endpoint + np.linalg.solve(
                np.asarray(camera_mapping, dtype=float),
                np.array([0.0, 0.0, checkpoint_backoff_mm], dtype=float),
            )
        except np.linalg.LinAlgError:
            return None, (
                "all bounded routes failed and camera mapping is singular"
            ), {}
    emit(
        f"stage=grasp_route_plan phase={phase} status=global_search "
        "detail=deterministic corridors exhausted; arm stationary"
    )
    global_plan = A.plan_joint_motion(
        [(*[float(value) for value in global_endpoint], *fixed_rpy)],
        orientation_ref_rpy=fixed_rpy,
        max_tilt_rad=math.radians(float(np.clip(
            float(os.getenv("REBOT_GLOBAL_PLANNER_MAX_TILT_DEG", "12")),
            2.0, 20.0,
        ))),
        timeout_s=float(np.clip(
            float(os.getenv("REBOT_GLOBAL_PLANNER_TIMEOUT_S", "4")),
            0.5, 12.0,
        )),
        max_nodes=int(os.getenv("REBOT_GLOBAL_PLANNER_MAX_NODES", "4000")),
        seed=int(os.getenv("REBOT_GLOBAL_PLANNER_SEED", "0")),
    )
    if global_plan.get("ok"):
        planner = global_plan.get("planner", {})
        emit(
            f"stage=grasp_route_plan phase={phase} status=global_accepted "
            f"nodes={planner.get('nodes')} elapsed_s={planner.get('elapsed_s')} "
            f"joint_waypoints={len(global_plan.get('joint_path', []))}"
        )
        return global_plan, (
            "certified global joint route after deterministic grasp corridors "
            "failed"
        ), {
            "clearance_strategy": "global_joint_rrt_connect",
            "global_checkpoint_only": global_checkpoint_only,
            "global_endpoint_m": global_endpoint.round(4).tolist(),
            "checkpoint_remaining_m": float(np.linalg.norm(
                endpoint - global_endpoint
            )),
        }
    emit(
        f"stage=grasp_route_plan phase={phase} status=global_rejected "
        f"detail={global_plan.get('reason')}"
    )
    return None, (
        "all bounded clearance/axial candidates and global joint search failed: "
        + "; ".join(failures) + f"; global: {global_plan.get('reason')}"
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
    requested_total = float(np.max(np.abs(delta)))
    # Geometry clamps its requested yaw against the capture pose, while this
    # final validator starts from fresh motor feedback. A few hundredths of a
    # degree of ordinary tracking/quantization error can therefore turn an
    # exactly-at-limit request into 20.02deg and falsely reject it. Permit only
    # a small configurable boundary snap, then clip the *commanded* delta back
    # to the unchanged physical limit. Larger requests still fail closed.
    try:
        snap_deg = float(os.getenv(
            "REBOT_WRIST_RPY_BOUNDARY_SNAP_DEG", "0.10"
        ))
    except (TypeError, ValueError):
        snap_deg = 0.10
    snap = math.radians(float(np.clip(snap_deg, 0.0, 0.25)))
    boundary_snapped = bool(
        requested_total > max_total
        and requested_total <= max_total + snap + 1e-12
    )
    if requested_total > max_total + snap + 1e-12:
        return None, (
            f"orientation change {math.degrees(requested_total):.2f}deg "
            f"exceeds the {GEOMETRY_MAX_YAW_DELTA_DEG:.1f}deg clamp"
        )
    if boundary_snapped:
        delta = np.clip(delta, -max_total, max_total)
        target = start + delta
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
    reason = (
        f"{len(planned)} orientation waypoints prevalidated at fixed XYZ; "
        f"target_rpy={[round(float(v), 4) for v in target]}"
    )
    if boundary_snapped:
        reason += (
            f"; {math.degrees(requested_total):.2f}deg feedback-relative "
            f"request snapped to unchanged {GEOMETRY_MAX_YAW_DELTA_DEG:.1f}deg "
            "physical clamp"
        )
    return planned, reason


def _execute_rpy_trajectory(
    rpy_waypoints: list[tuple[float, float, float]],
    log: Callable[[str], None],
    *, fixed_xyz: Any | None = None,
) -> tuple[bool, str]:
    """Execute one watchdog-bounded continuous stationary reorientation."""
    stationary_xyz = None if fixed_xyz is None else tuple(
        float(v) for v in np.asarray(fixed_xyz, dtype=float)
    )
    xyz = stationary_xyz or tuple(float(v) for v in A.ee_pos())
    poses = [
        (*xyz, float(roll), float(pitch), float(yaw))
        for roll, pitch, yaw in rpy_waypoints
    ]
    timeout_s = float(np.clip(12.0 + 2.0 * len(poses), 15.0, 30.0))
    final = rpy_waypoints[-1]
    log(
        "stage=orientation_continuous status=executing "
        f"waypoints={len(poses)} watchdog_s={timeout_s:.0f} "
        f"target_rpy_deg={[round(math.degrees(v), 1) for v in final]}"
    )

    previous_handler = None
    watchdog_installed = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    )

    def orientation_timeout(_signum, _frame):
        raise TimeoutError(
            f"orientation controller exceeded {timeout_s:.0f}s watchdog"
        )

    try:
        if watchdog_installed:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, orientation_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        result = A.safe_pose_path(poses)
    finally:
        if watchdog_installed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
    log(
        f"  [orientation continuous 1/{len(poses)} -> {len(poses)}/{len(poses)}] "
        f"ok={result.ok} detail={result.reason}"
    )
    if not result.ok:
        return False, result.reason
    return True, "bounded continuous orientation trajectory completed"


def _detection_partial(det: WD.Detection, frame: Any) -> str | None:
    """Explain why one detection is unsuitable as a complete scene target."""
    color = frame[0]
    height, width = color.shape[:2]
    x1, y1, x2, y2 = det.box
    soft_border = 12
    hard_border = 1

    # A target resting on the table will commonly finish near the bottom of
    # the wrist image even when its entire usable body is visible.  Treating
    # the old 12 px bottom guard as proof of truncation caused a good target to
    # trigger every look-around pose and eventually abort.  Left/right/top
    # proximity still means the camera may be missing useful object geometry.
    # At the bottom, reject only a box that is actually clamped to the image;
    # grasp geometry, local jaw width, payload size, and reachability are still
    # independently checked before any approach is executed.
    clipped_edges = []
    if x1 <= hard_border:
        clipped_edges.append("left")
    if y1 <= hard_border:
        clipped_edges.append("top")
    if x2 >= width - hard_border:
        clipped_edges.append("right")
    if y2 >= height - hard_border:
        clipped_edges.append("bottom")
    if clipped_edges:
        return (
            "semantic/depth envelope is clipped at the image "
            + "/".join(clipped_edges)
            + " edge"
        )

    crowded_edges = []
    if x1 <= soft_border:
        crowded_edges.append("left")
    if y1 <= soft_border:
        crowded_edges.append("top")
    if x2 >= width - soft_border:
        crowded_edges.append("right")
    if crowded_edges:
        return (
            "semantic/depth envelope is too near the image "
            + "/".join(crowded_edges)
            + " edge for complete geometry"
        )
    if float(det.z_mm) <= float(WD.NEAR_MM) + 45.0:
        return f"target depth {det.z_mm:.0f}mm is too close for a complete view"
    if float(det.z_mm) >= float(WD.FAR_MM) - 30.0:
        return f"target depth {det.z_mm:.0f}mm is at the far depth boundary"
    return None


def _append_pose_aware_discovery_view(
    captured_views: list[dict[str, Any]], *, name: str,
    frame: tuple[np.ndarray, np.ndarray, Any], detection: Any,
    geometry: GG.GraspGeometry | None, log: Callable[[str], None],
    ee_xyz_m: Any | None = None, ee_rpy: Any | None = None,
) -> bool:
    """Retain an RGB-D view with the transform valid at capture time."""
    view_rpy = tuple(float(v) for v in (
        A.ee_rpy() if ee_rpy is None else ee_rpy
    ))
    view_mapping, view_mapping_detail = load_saved_camera_jacobian(view_rpy)
    if view_mapping is None:
        log(
            f"stage=look_around_capture name={name} status=skipped "
            f"detail={view_mapping_detail.get('reason')}"
        )
        return False
    captured_views.append({
        "name": name,
        "frame": frame,
        "detection": detection,
        "geometry": geometry,
        "ee_xyz_m": np.asarray(
            A.ee_pos() if ee_xyz_m is None else ee_xyz_m, dtype=float
        ),
        "ee_rpy_rad": np.asarray(view_rpy, dtype=float),
        "camera_mapping": np.asarray(view_mapping, dtype=float),
    })
    log(
        f"stage=look_around_capture name={name} status=saved "
        f"views={len(captured_views)}"
    )
    return True


def _recenter_clipped_placement_reference(
    eye: WristEye, description: str, detection: Any,
    frame: tuple[np.ndarray, np.ndarray, Any], log: Callable[[str], None],
    *, observation_consensus_frames: int,
    observation_consensus_max_spread_mm: float | None,
    _attempt_index: int = 1,
    _remaining_corrections: int | None = None,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    """Make one bounded camera-calibrated move to reframe a clipped reference.

    This is deliberately placement-reference-only.  A partially visible object
    is not trustworthy enough to define a scene-relative release point, while
    yaw/pitch alone cannot bring an object at the side of the workspace into the
    image.  Use the rigid wrist-camera mapping to predict a retreat/sideways/up
    translation, prevalidate the whole motion, and then demand the same stable,
    complete multi-frame observation used by the ordinary discovery sweep.

    The correction can never extend the arm or descend it.  It therefore cannot
    turn an edge-framing problem into an unplanned approach toward the table or
    object.  Failure simply leaves the normal fail-closed behavior in place.
    """
    if _remaining_corrections is None:
        _remaining_corrections = int(np.clip(int(os.getenv(
            "REBOT_LOOK_AROUND_RECENTER_STEPS", "3"
        )), 1, 3))
    color, _depth, intr = frame
    height, width = color.shape[:2]
    observed_cam = np.asarray(detection.p_cam_mm, dtype=float)
    if observed_cam.shape != (3,) or not np.all(np.isfinite(observed_cam)) \
            or float(observed_cam[2]) <= 1.0:
        return None, None, {"ok": False, "reason": "invalid clipped reference depth"}

    current_rpy = tuple(float(v) for v in A.ee_rpy())
    mapping, mapping_detail = load_saved_camera_jacobian(current_rpy)
    if mapping is None:
        return None, None, {
            "ok": False,
            "reason": "camera mapping unavailable for edge recenter: "
                      f"{mapping_detail.get('reason')}",
        }

    # Aim at the image centre rather than barely crossing the border.  The
    # visible bounding box is truncated and consequently underestimates the
    # missing half of the object; centring leaves room for that unseen portion.
    desired_uv = (0.5 * float(width - 1), 0.5 * float(height - 1))
    desired_cam = np.asarray(WD.deproject(
        desired_uv[0], desired_uv[1], float(observed_cam[2]), intr
    ), dtype=float)
    camera_delta_mm = desired_cam - observed_cam
    try:
        raw_delta_base_m = np.linalg.solve(
            np.asarray(mapping, dtype=float), camera_delta_mm
        )
    except np.linalg.LinAlgError as exc:
        return None, None, {
            "ok": False, "reason": f"camera recenter solve failed: {exc}"
        }
    if not np.all(np.isfinite(raw_delta_base_m)):
        return None, None, {"ok": False, "reason": "non-finite camera recenter move"}

    # Discovery is allowed to retreat, translate laterally, or lift.  Never
    # increase X (reach farther toward the scene) and never reduce Z.
    bounded_delta = np.asarray(raw_delta_base_m, dtype=float)
    bounded_delta[0] = min(0.0, float(bounded_delta[0]))
    bounded_delta[2] = max(0.0, float(bounded_delta[2]))
    maximum_m = float(np.clip(float(os.getenv(
        "REBOT_LOOK_AROUND_RECENTER_MAX_M", "0.045"
    )), 0.015, 0.060))
    norm_m = float(np.linalg.norm(bounded_delta))
    if norm_m > maximum_m:
        bounded_delta *= maximum_m / norm_m
    if float(np.linalg.norm(bounded_delta)) < 0.003:
        return None, None, {
            "ok": False,
            "reason": "edge recenter would require forward motion or descent",
        }

    predicted_cam = observed_cam + np.asarray(mapping) @ bounded_delta
    before_px = WD.project(observed_cam, intr)
    after_px = WD.project(predicted_cam, intr)
    if before_px is None or after_px is None:
        return None, None, {"ok": False, "reason": "edge recenter projection failed"}
    before_error_px = float(np.linalg.norm(
        np.asarray(before_px) - np.asarray(desired_uv)
    ))
    after_error_px = float(np.linalg.norm(
        np.asarray(after_px) - np.asarray(desired_uv)
    ))
    if after_error_px >= before_error_px - 5.0:
        return None, None, {
            "ok": False,
            "reason": (
                "bounded retreat/sideways/up correction does not improve "
                f"framing ({before_error_px:.0f}->{after_error_px:.0f}px)"
            ),
        }

    start_xyz = np.asarray(A.ee_pos(), dtype=float)
    endpoint = start_xyz + bounded_delta
    path, plan_reason = _plan_fixed_approach_trajectory(
        endpoint, fixed_rpy=current_rpy
    )
    if path is None:
        return None, None, {
            "ok": False,
            "reason": f"edge recenter path refused: {plan_reason}",
        }
    log(
        "stage=placement_reference_recenter status=planned "
        f"attempt={_attempt_index} "
        f"camera_error_px={before_error_px:.0f}->{after_error_px:.0f} "
        f"base_delta_mm={(bounded_delta * 1000.0).round(1).tolist()} "
        "motion=retreat_sideways_or_up_only"
    )
    moved, move_reason = _execute_fixed_approach_trajectory(
        path, log, fixed_rpy=current_rpy
    )
    if not moved:
        return None, None, {
            "ok": False, "reason": f"edge recenter execution failed: {move_reason}"
        }
    _settle(eye, note="placement reference edge recenter")

    count = int(np.clip(observation_consensus_frames, 3, 7))
    required = max(2, count - 1)
    detections: list[Any] = []
    frames: list[Any] = []
    points: list[np.ndarray] = []
    rejected: list[str] = []
    latest_clipped: tuple[Any, Any] | None = None
    stable_median: np.ndarray | None = None
    stable_detail: dict[str, Any] = {}
    # SAM can occasionally switch from the named reference to another object
    # of the same colour in a single stationary frame.  Do not stop merely
    # because two complete boxes were returned; stop only after two metric
    # camera points form the existing 15 mm inlier cluster.  A second set of
    # ``count`` attempts is bounded resampling, not a relaxed acceptance gate.
    attempt_limit = 2 * count
    for attempt in range(1, attempt_limit + 1):
        candidate, candidate_frame, why = _observe(eye, description)
        if candidate is None or candidate_frame is None:
            reason = f"not detected: {why}"
            rejected.append(reason)
        else:
            reason = _detection_partial(candidate, candidate_frame)
            if reason is None:
                detections.append(candidate)
                frames.append(candidate_frame)
                points.append(np.asarray(candidate.p_cam_mm, dtype=float))
                log(
                    "stage=placement_reference_recenter_sample "
                    f"attempt={attempt}/{attempt_limit} accepted=true "
                    f"source={candidate.source} "
                    f"p_cam_mm={points[-1].round(1).tolist()}"
                )
                if len(points) >= required:
                    stable_median, stable_detail = \
                        _scene_reference_point_consensus(
                            points, required=required,
                            max_spread_mm=(
                                float(observation_consensus_max_spread_mm)
                                if observation_consensus_max_spread_mm is not None
                                else float("inf")
                            ),
                        )
                    if stable_median is not None:
                        break
                time.sleep(0.08)
                continue
            rejected.append(reason)
            if "image" in reason and "edge" in reason:
                latest_clipped = (candidate, candidate_frame)
        log(
            "stage=placement_reference_recenter_sample "
            f"attempt={attempt}/{attempt_limit} accepted=false detail={reason}"
        )

    if len(points) < required or stable_median is None:
        # A single 45 mm move cannot centre an object that began hundreds of
        # pixels outside the useful view.  Re-observe after each certified leg
        # and permit a small bounded number of additional legs instead of
        # pretending the first prediction should have completed the job.  Each
        # leg independently retains the no-forward/no-descent and IK/collision
        # gates above.
        if latest_clipped is not None and _remaining_corrections > 1:
            log(
                "stage=placement_reference_recenter status=continuing "
                f"attempt={_attempt_index + 1} reason=reference_still_clipped"
            )
            return _recenter_clipped_placement_reference(
                eye, description, latest_clipped[0], latest_clipped[1], log,
                observation_consensus_frames=observation_consensus_frames,
                observation_consensus_max_spread_mm=
                    observation_consensus_max_spread_mm,
                _attempt_index=_attempt_index + 1,
                _remaining_corrections=_remaining_corrections - 1,
            )
        return None, None, {
            "ok": False,
            "reason": (
                f"recenter produced only {len(points)} complete frames"
                if len(points) < required else
                str(stable_detail.get(
                    "reason", "recenter detections did not form a stable cluster"
                ))
            ),
            "rejected_frames": rejected,
        }
    point_array = np.vstack(points)
    median_point = np.asarray(stable_median, dtype=float)
    rejected_indices = set(stable_detail.get("rejected_frame_indices", []))
    stable_indices = [
        index for index in range(len(points)) if index not in rejected_indices
    ]
    representative = min(
        stable_indices,
        key=lambda index: float(np.linalg.norm(points[index] - median_point)),
    )
    spread_mm = float(stable_detail.get("spread_mm", 0.0))
    log(
        "stage=placement_reference_recenter status=accepted "
        f"stable_frames={len(stable_indices)}/{len(points)} "
        f"spread_mm={spread_mm:.1f} "
        f"identity_outliers={len(rejected_indices)}"
    )
    return detections[representative], frames[representative], {
        "ok": True,
        "pose": "edge_recenter",
        "base_delta_mm": (bounded_delta * 1000.0).round(2).tolist(),
        "observation_consensus": {
            "stable_frames": len(stable_indices),
            "configured_frames": count,
            "required_frames": required,
            "samples_camera_mm": point_array.round(2).tolist(),
            "reference_camera_mm": median_point.round(2).tolist(),
            "spread_mm": round(spread_mm, 2),
            "rejected_frames": rejected,
            "rejected_frame_indices": sorted(rejected_indices),
        },
    }


def _restore_camera_mapping_for_recenter(
    start_xyz: Any, start_rpy: Any, log: Callable[[str], None],
) -> tuple[bool, str]:
    """Return a discovery sweep to its calibrated camera orientation.

    Edge recentering uses the saved rigid camera Jacobian and therefore cannot
    consume a detection captured outside that calibration envelope. Restore
    orientation at the current safe XYZ first. If that is kinematically
    blocked, retrace to the sweep's known-safe starting XYZ while preserving
    the live orientation, then rotate there. Every leg uses the ordinary
    complete IK/collision validators.
    """
    current_rpy = tuple(float(value) for value in A.ee_rpy())
    mapping, mapping_detail = load_saved_camera_jacobian(current_rpy)
    if mapping is not None:
        return True, "current viewpoint already has a valid camera mapping"

    target_xyz = np.asarray(start_xyz, dtype=float)
    target_rpy = tuple(float(value) for value in np.asarray(start_rpy, dtype=float))
    orientation_path, orientation_reason = _plan_rpy_trajectory(
        target_rpy, xyz=A.ee_pos()
    )
    if orientation_path is None:
        log(
            "stage=placement_reference_recenter_mapping_restore "
            "phase=orientation_first status=blocked "
            f"detail={orientation_reason}; fallback=translate_first"
        )
        live_xyz = np.asarray(A.ee_pos(), dtype=float)
        position_path, position_reason = _plan_fixed_approach_trajectory(
            target_xyz, start_m=live_xyz, fixed_rpy=current_rpy
        )
        if position_path is None:
            return False, (
                "camera mapping restore failed by both orders: orientation "
                f"first ({orientation_reason}); translation first "
                f"({position_reason})"
            )
        moved, move_reason = _execute_fixed_approach_trajectory(
            position_path, log, fixed_rpy=current_rpy
        )
        if not moved:
            return False, (
                "camera mapping translate-first restore failed: "
                f"{move_reason}"
            )
        orientation_path, orientation_reason = _plan_rpy_trajectory(
            target_rpy, xyz=target_xyz
        )
    if orientation_path is None:
        return False, f"camera mapping orientation restore refused: {orientation_reason}"
    oriented, oriented_reason = _execute_rpy_trajectory(
        orientation_path, log,
        fixed_xyz=(target_xyz if np.linalg.norm(
            np.asarray(A.ee_pos(), dtype=float) - target_xyz
        ) <= 0.003 else A.ee_pos()),
    )
    if not oriented:
        return False, f"camera mapping orientation restore failed: {oriented_reason}"
    restored_mapping, restored_detail = load_saved_camera_jacobian(
        tuple(float(value) for value in A.ee_rpy())
    )
    if restored_mapping is None:
        return False, (
            "restored wrist pose still lacks a valid camera mapping: "
            f"{restored_detail.get('reason')}"
        )
    reason = (
        "returned to the sweep's calibrated camera orientation after "
        f"out-of-envelope view ({mapping_detail.get('reason')})"
    )
    log(
        "stage=placement_reference_recenter_mapping_restore "
        f"status=ready detail={reason}"
    )
    return True, reason


def _look_around_object(
    eye: WristEye, description: str, log: Callable[[str], None],
    *, require_grasp: bool,
    geometry_accept: Callable[[GG.GraspGeometry], tuple[bool, str]] | None = None,
    geometry_rejected: Callable[
        [tuple[np.ndarray, np.ndarray, Any], GG.GraspGeometry, str, str], None
    ] | None = None,
    captured_views: list[dict[str, Any]] | None = None,
    require_camera_mapping: bool = False,
    observation_consensus_frames: int = 1,
    observation_consensus_max_spread_mm: float | None = None,
    allow_edge_recenter: bool = False,
) -> tuple[Any | None, Any | None, GG.GraspGeometry | None, dict[str, Any]]:
    """Bounded elevated discovery sweep; never moves forward or descends.

    ``observation_consensus_frames`` is normally one because grasp discovery
    performs its own geometry consensus later.  Placement-reference discovery
    raises it so a single lucky segmentation at an image edge cannot stop the
    sweep.  In that mode each candidate pose must produce multiple complete,
    position-consistent observations before it is accepted.
    """
    start_xyz = np.asarray(A.ee_pos(), dtype=float)
    start_rpy = np.asarray(A.ee_rpy(), dtype=float)
    retreat = start_xyz.copy()
    retreat[0] = max(A.X_MIN + APPROACH_WORKSPACE_MARGIN_M,
                     float(start_xyz[0]) - 0.040)
    # Inspect from the already elevated ready pose before translating.  A
    # level-wrist retreat from home/ready lies close to a kinematic
    # singularity on the RS arm, so planning that same translation first made
    # every discovery candidate fail before its useful yaw/pitch was reached.
    # If an orientation-only view is insufficient, pitch first and retreat
    # while preserving that more reachable posture.
    pitched_rpy = start_rpy + np.array([0.0, math.radians(8.0), 0.0])
    candidates = [
        ("yaw_left", start_xyz,
         False,
         start_rpy + np.array([0.0, 0.0, math.radians(12.0)])),
        ("yaw_right", start_xyz,
         False,
         start_rpy + np.array([0.0, 0.0, math.radians(-12.0)])),
        ("pitch_wide", start_xyz, False, pitched_rpy),
        ("pitched_retreat", retreat, True, pitched_rpy),
        ("pitched_retreat_yaw_left", retreat,
         True,
         pitched_rpy + np.array([0.0, 0.0, math.radians(12.0)])),
        ("pitched_retreat_yaw_right", retreat,
         True,
         pitched_rpy + np.array([0.0, 0.0, math.radians(-12.0)])),
    ]
    attempts = []
    last_edge_observation: tuple[Any, Any] | None = None
    for index, (name, xyz, translate, rpy) in enumerate(candidates, 1):
        log(f"stage=look_around pose={index}/{len(candidates)} name={name} "
            f"status=planning target={description!r}")
        # Reorient at the current, known-safe XYZ first.  Large transitions
        # (for example +12deg yaw to -12deg yaw) pass through the original
        # ready orientation so no single sweep exceeds the bounded limit.
        live_orientation = np.asarray(A.ee_rpy(), dtype=float)
        max_orientation_step = math.radians(GEOMETRY_MAX_YAW_DELTA_DEG)
        if float(np.max(np.abs(np.asarray(rpy) - live_orientation))) > \
                max_orientation_step:
            neutral_path, neutral_reason = _plan_rpy_trajectory(
                tuple(float(v) for v in start_rpy)
            )
            if neutral_path is None:
                attempts.append({"pose": name, "reason": neutral_reason})
                log(f"stage=look_around name={name} status=rejected "
                    f"phase=neutral_orientation detail={neutral_reason}")
                continue
            neutral_ok, neutral_exec = _execute_rpy_trajectory(neutral_path, log)
            if not neutral_ok:
                attempts.append({"pose": name, "reason": neutral_exec})
                log(f"stage=look_around name={name} status=stopped "
                    f"phase=neutral_orientation detail={neutral_exec}")
                continue
        rpy_path, rpy_reason = _plan_rpy_trajectory(tuple(float(v) for v in rpy))
        if rpy_path is None:
            attempts.append({"pose": name, "reason": rpy_reason})
            log(f"stage=look_around name={name} status=rejected "
                f"phase=orientation detail={rpy_reason}")
            continue
        oriented, orientation_reason = _execute_rpy_trajectory(rpy_path, log)
        if not oriented:
            attempts.append({"pose": name, "reason": orientation_reason})
            log(f"stage=look_around name={name} status=stopped "
                f"phase=orientation detail={orientation_reason}")
            continue

        # Translation is optional for the first three views.  For the later
        # candidates it is planned only after pitch has moved the wrist away
        # from the level-arm singular posture.
        if translate:
            fixed_rpy = tuple(float(v) for v in rpy)
            position_path, position_reason = _plan_fixed_approach_trajectory(
                xyz, fixed_rpy=fixed_rpy
            )
            if position_path is None:
                attempts.append({"pose": name, "reason": position_reason})
                log(f"stage=look_around name={name} status=rejected "
                    f"detail={position_reason}")
                continue
            moved, move_reason = _execute_fixed_approach_trajectory(
                position_path, log,
                fixed_rpy=fixed_rpy,
            )
            if not moved:
                attempts.append({"pose": name, "reason": move_reason})
                log(f"stage=look_around name={name} status=stopped "
                    f"phase=translation detail={move_reason}")
                continue
        log(f"stage=look_around name={name} status=observing "
            f"target={description!r}")
        _settle(eye, note=f"look around {name}")
        # Snapshot the arm transform immediately before the RGB-D acquisition.
        # Semantic inference can take seconds; reading FK only after inference
        # previously associated an old camera frame with a later sagged pose.
        capture_xyz = np.asarray(A.ee_pos(), dtype=float)
        capture_rpy = np.asarray(A.ee_rpy(), dtype=float)
        det, frame, why = _observe(eye, description)
        after_xyz = np.asarray(A.ee_pos(), dtype=float)
        after_rpy = np.asarray(A.ee_rpy(), dtype=float)
        position_drift_mm = float(np.linalg.norm(after_xyz - capture_xyz) * 1000.0)
        before_rotation = _rpy_rotation_matrix(capture_rpy)
        after_rotation = _rpy_rotation_matrix(after_rpy)
        relative_rotation = before_rotation.T @ after_rotation
        orientation_drift_deg = math.degrees(math.acos(float(np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0
        ))))
        capture_stable = bool(
            position_drift_mm <= float(np.clip(float(os.getenv(
                "REBOT_SCENE_CAPTURE_MAX_DRIFT_MM", "2.0"
            )), 0.5, 5.0))
            and orientation_drift_deg <= float(np.clip(float(os.getenv(
                "REBOT_SCENE_CAPTURE_MAX_DRIFT_DEG", "0.5"
            )), 0.1, 2.0))
        )
        if det is None or frame is None:
            reason = f"not detected: {why}"
            attempts.append({"pose": name, "reason": reason})
            log(f"stage=look_around name={name} status=not_found detail={reason}")
            continue
        partial = _detection_partial(det, frame)
        geometry = None
        if partial is None and require_grasp:
            geometry = _plan_scene_grasp_geometry(
                frame, det, current_rpy=tuple(float(v) for v in A.ee_rpy()),
                object_name=description,
            )
            if not geometry.ok:
                partial = geometry.reason
        # Preserve every useful, stationary discovery view with the transform
        # that was valid when it was captured.  These private frame records are
        # consumed later by scene fusion; they are deliberately not included in
        # the JSON result returned to the operator.
        if captured_views is not None and capture_stable:
            _append_pose_aware_discovery_view(
                captured_views, name=name, frame=frame, detection=det,
                geometry=geometry, log=log,
                ee_xyz_m=capture_xyz, ee_rpy=capture_rpy,
            )
        elif captured_views is not None:
            log(
                f"stage=look_around_capture name={name} status=skipped "
                f"detail=arm moved during RGB-D/inference window "
                f"position_drift_mm={position_drift_mm:.2f} "
                f"orientation_drift_deg={orientation_drift_deg:.2f}"
            )
        if partial is None and geometry is not None and geometry_accept is not None:
            accepted, accept_reason = geometry_accept(geometry)
            if not accepted:
                if geometry_rejected is not None:
                    geometry_rejected(frame, geometry, accept_reason, name)
                partial = f"grasp entry rejected: {accept_reason}"
        observation_consensus = None
        if partial is None and observation_consensus_frames > 1:
            consensus_count = int(np.clip(
                observation_consensus_frames, 3, 7
            ))
            consensus_required = max(2, consensus_count - 1)
            consensus_detections = [det]
            consensus_frames = [frame]
            consensus_points = [np.asarray(det.p_cam_mm, dtype=float)]
            consensus_rejections: list[str] = []
            # The already accepted frame is retained as sample one.  Permit the
            # same number of stationary retries as the original placement
            # consensus so this strengthens view selection without reducing
            # its chance of finding a valid pose.
            for sample_attempt in range(1, consensus_count + 1):
                if len(consensus_points) >= consensus_required:
                    break
                sample_det, sample_frame, sample_why = _observe(eye, description)
                if sample_det is None or sample_frame is None:
                    sample_reason = f"not detected: {sample_why}"
                    consensus_rejections.append(sample_reason)
                    log(
                        f"stage=look_around_consensus name={name} "
                        f"sample={len(consensus_points)}/{consensus_count} "
                        f"attempt={sample_attempt}/{consensus_count} "
                        f"accepted=false detail={sample_reason}"
                    )
                    continue
                sample_partial = _detection_partial(sample_det, sample_frame)
                if sample_partial is not None:
                    consensus_rejections.append(sample_partial)
                    log(
                        f"stage=look_around_consensus name={name} "
                        f"sample={len(consensus_points)}/{consensus_count} "
                        f"attempt={sample_attempt}/{consensus_count} "
                        f"accepted=false detail={sample_partial}"
                    )
                    continue
                consensus_detections.append(sample_det)
                consensus_frames.append(sample_frame)
                consensus_points.append(
                    np.asarray(sample_det.p_cam_mm, dtype=float)
                )
                log(
                    f"stage=look_around_consensus name={name} "
                    f"sample={len(consensus_points)}/{consensus_count} "
                    f"attempt={sample_attempt}/{consensus_count} "
                    f"accepted=true source={sample_det.source} "
                    f"p_cam_mm={consensus_points[-1].round(1).tolist()}"
                )
                time.sleep(0.08)

            if len(consensus_points) < consensus_required:
                partial = (
                    "candidate pose produced only "
                    f"{len(consensus_points)}/{consensus_count} complete "
                    "reference frames"
                )
            else:
                point_array = np.vstack(consensus_points)
                median_point = np.median(point_array, axis=0)
                spread_mm = float(np.max(np.linalg.norm(
                    point_array - median_point, axis=1
                )))
                spread_limit = (
                    float(observation_consensus_max_spread_mm)
                    if observation_consensus_max_spread_mm is not None
                    else float("inf")
                )
                if spread_mm > spread_limit:
                    partial = (
                        f"candidate reference spread {spread_mm:.1f}mm "
                        f"exceeds {spread_limit:.1f}mm"
                    )
                else:
                    representative_index = int(np.argmin(np.linalg.norm(
                        point_array - median_point, axis=1
                    )))
                    det = consensus_detections[representative_index]
                    frame = consensus_frames[representative_index]
                    observation_consensus = {
                        "stable_frames": len(consensus_points),
                        "configured_frames": consensus_count,
                        "required_frames": consensus_required,
                        "samples_camera_mm": point_array.round(2).tolist(),
                        "reference_camera_mm": median_point.round(2).tolist(),
                        "spread_mm": round(spread_mm, 2),
                        "rejected_frames": consensus_rejections,
                    }
                    log(
                        f"stage=look_around_consensus name={name} "
                        f"status=accepted stable_frames="
                        f"{len(consensus_points)}/{consensus_count} "
                        f"spread_mm={spread_mm:.1f}"
                    )

        # Scene-relative placement needs a metric camera-to-base mapping for
        # the exact pose that produced its reference detection.  A marker can
        # be clearly visible after yawing farther away from the saved camera
        # calibration, but accepting that image only to fail immediately in
        # placement planning wastes the whole pre-pick setup.  Treat it as an
        # unusable viewpoint and continue the bounded sweep; the opposite yaw
        # commonly moves back inside the calibrated envelope.
        if partial is None and require_camera_mapping:
            _mapping, mapping_detail = load_saved_camera_jacobian(
                tuple(float(v) for v in A.ee_rpy())
            )
            if _mapping is None:
                partial = (
                    "metric camera mapping unavailable at this viewpoint: "
                    f"{mapping_detail.get('reason')}"
                )
                log(
                    f"stage=look_around_mapping_gate name={name} "
                    f"ok=false detail={partial}"
                )

        if partial is None:
            log(f"stage=look_around name={name} status=accepted "
                f"source={det.source} depth_mm={det.z_mm:.0f}")
            detail = {
                "ok": True, "pose": name, "attempts": attempts,
                "moved": True,
            }
            if observation_consensus is not None:
                detail["observation_consensus"] = observation_consensus
            return det, frame, geometry, detail
        attempts.append({"pose": name, "reason": partial})
        log(f"stage=look_around name={name} status=partial detail={partial}")
        if "image" in partial and "edge" in partial:
            last_edge_observation = (det, frame)
    if allow_edge_recenter and last_edge_observation is not None:
        log(
            "stage=placement_reference_recenter status=triggered "
            "reason=orientation sweep left the reference clipped"
        )
        recenter_observation = last_edge_observation
        _mapping, _mapping_detail = load_saved_camera_jacobian(
            tuple(float(value) for value in A.ee_rpy())
        )
        mapping_restore_detail = None
        if _mapping is None:
            mapping_restored, mapping_restore_detail = \
                _restore_camera_mapping_for_recenter(
                    start_xyz, start_rpy, log
                )
            if mapping_restored:
                _settle(eye, note="placement reference mapping restore")
                restored_det, restored_frame, restored_why = _observe(
                    eye, description
                )
                if restored_det is not None and restored_frame is not None:
                    recenter_observation = (restored_det, restored_frame)
                else:
                    mapping_restored = False
                    mapping_restore_detail = (
                        "reference was not detected after camera mapping "
                        f"restore: {restored_why}"
                    )
            if not mapping_restored:
                recentered_det, recentered_frame, recentered = None, None, {
                    "ok": False,
                    "reason": str(mapping_restore_detail),
                }
            else:
                recentered_det, recentered_frame, recentered = \
                    _recenter_clipped_placement_reference(
                        eye, description, recenter_observation[0],
                        recenter_observation[1], log,
                        observation_consensus_frames=
                            observation_consensus_frames,
                        observation_consensus_max_spread_mm=
                            observation_consensus_max_spread_mm,
                    )
        else:
            recentered_det, recentered_frame, recentered = \
                _recenter_clipped_placement_reference(
                    eye, description, recenter_observation[0],
                    recenter_observation[1], log,
                    observation_consensus_frames=observation_consensus_frames,
                    observation_consensus_max_spread_mm=
                        observation_consensus_max_spread_mm,
                )
        if recentered_det is not None and recentered_frame is not None:
            return recentered_det, recentered_frame, None, {
                "ok": True,
                "pose": "edge_recenter",
                "attempts": attempts,
                "moved": True,
                **recentered,
            }
        attempts.append({
            "pose": "edge_recenter",
            "reason": recentered.get("reason"),
        })
        log(
            "stage=placement_reference_recenter status=rejected "
            f"detail={recentered.get('reason')}"
        )
    recovery_cloud = None
    if require_grasp and captured_views:
        log(
            "stage=multiview_metric_grasp_trigger "
            "reason=no 2D grasp-valid view; fusing complete swept masks before "
            "rejecting the target"
        )
        recovered_det, recovered_frame, recovered_geometry, recovered = \
            _recover_multiview_metric_grasp(
                eye, description, captured_views, log
            )
        recovery_cloud = recovered.get("discovery_cloud")
        if recovered_det is not None and recovered_frame is not None \
                and recovered_geometry is not None:
            return recovered_det, recovered_frame, recovered_geometry, {
                "ok": True,
                "pose": recovered.get("pose"),
                "reason": recovered.get("reason"),
                "attempts": attempts,
                "moved": True,
                "multiview_metric_recovery": True,
                "discovery_cloud": recovered.get("discovery_cloud"),
            }
        attempts.append({
            "pose": "multiview_metric_recovery",
            "reason": recovered.get("reason"),
        })
    return None, None, None, {
        "ok": False, "reason": "no complete view found in bounded sweep",
        "attempts": attempts, "moved": True,
        "discovery_cloud": recovery_cloud,
    }


def _execute_fixed_approach_trajectory(
    waypoints: Any,
    log: Callable[[str], None],
    fixed_rpy: tuple[float, float, float] | None = None,
):
    """Execute only the pre-planned waypoints; camera detections cannot steer it."""
    if isinstance(waypoints, dict) and waypoints.get("kind") == "certified_joint_plan":
        count = len(waypoints.get("joint_path", []))
        log(
            "stage=global_joint_route_execute status=revalidating "
            f"scene_version={waypoints.get('scene_version')} waypoints={count}"
        )
        result = A.execute_joint_plan(waypoints)
        log(
            "stage=global_joint_route_execute "
            f"status={'arrived' if result.ok else 'refused'} detail={result.reason}"
        )
        return result.ok, result.reason
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
    *, allow_global: bool = True,
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
    route_kind = "straight_cartesian"
    # A nearby obstacle can block the straight 5--12 mm tool correction even
    # when the corrected terminal pose itself is clear. Use the same certified
    # full-joint-space planner as macro grasp/carry motion before concluding
    # that no alternative route exists. The endpoint orientation remains the
    # calibrated camera orientation; bounded intermediate tilt only changes how
    # the arm gets there.
    if waypoints is None and allow_global and SCENE_COLLISION_ENABLED:
        endpoint_gate = A.check_pose_clearance(
            *endpoint.tolist(), *fixed_rpy
        )
        if endpoint_gate.ok:
            global_plan = A.plan_joint_motion(
                [(*[float(value) for value in endpoint], *fixed_rpy)],
                orientation_ref_rpy=fixed_rpy,
                max_tilt_rad=math.radians(float(np.clip(
                    float(os.getenv(
                        "REBOT_FINE_ALIGN_GLOBAL_MAX_TILT_DEG", "5"
                    )), 2.0, 10.0,
                ))),
                timeout_s=float(np.clip(float(os.getenv(
                    "REBOT_FINE_ALIGN_GLOBAL_TIMEOUT_S", "3"
                )), 0.5, 6.0)),
                max_nodes=int(np.clip(int(os.getenv(
                    "REBOT_FINE_ALIGN_GLOBAL_MAX_NODES", "2500"
                )), 500, 5000)),
                seed=int(os.getenv("REBOT_GLOBAL_PLANNER_SEED", "0")),
            )
            if global_plan.get("ok"):
                waypoints = global_plan
                route_kind = "certified_global_joint"
                reason = (
                    "straight fine alignment was blocked; certified global "
                    "joint route reaches the same camera-aligned endpoint"
                )
            else:
                reason = (
                    f"{reason}; exact endpoint is clear but global joint "
                    f"alternative failed: {global_plan.get('reason')}"
                )
        else:
            reason = (
                f"{reason}; corrected endpoint is collision-occupied: "
                f"{endpoint_gate.reason}"
            )
    detail = {
        "reason": reason,
        "camera_error_mm": error_cam.round(2).tolist(),
        "base_correction_mm": (delta * 1000.0).round(2).tolist(),
        "correction_norm_mm": round(norm * 1000.0, 2),
        "endpoint_m": endpoint.round(4).tolist(),
        "route_kind": route_kind,
    }
    return waypoints, detail


def _update_alignment_jacobian_from_feedback(
    rigid_jacobian: Any,
    tool_delta_m: Any,
    body_before_cam_mm: Any,
    body_after_cam_mm: Any,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Adapt the *local* fine-alignment map from one measured response.

    The saved rigid camera map remains the geometric prior and is never
    overwritten.  Near the open fingers, segmentation and the selected local
    body section can respond differently from that prior.  Once the same
    section has been locked and remeasured after a successful bounded move, a
    rank-one secant update makes the next (and only the next) correction agree
    with the response that was actually observed.

    Contradictory, tiny, excessive, or ill-conditioned responses are rejected.
    This helper does not plan motion, relax the jaw slot, or increase any
    displacement limit.
    """
    failure = lambda reason, **extra: {
        "ok": False, "reason": reason, **extra,
    }
    try:
        mapping = np.asarray(rigid_jacobian, dtype=float)
        delta = np.asarray(tool_delta_m, dtype=float)
        before = np.asarray(body_before_cam_mm, dtype=float)
        after = np.asarray(body_after_cam_mm, dtype=float)
    except (TypeError, ValueError):
        return None, failure("visual-feedback alignment inputs are invalid")
    if mapping.shape != (3, 3) or delta.shape != (3,) \
            or before.shape != (3,) or after.shape != (3,) \
            or not all(np.all(np.isfinite(value)) for value in (
                mapping, delta, before, after,
            )):
        return None, failure(
            "visual-feedback alignment inputs are incomplete or non-finite"
        )

    motion_mm = float(np.linalg.norm(delta) * 1000.0)
    if motion_mm < 0.5 or motion_mm > 15.0:
        return None, failure(
            f"measured tool motion {motion_mm:.1f}mm is outside the "
            "0.5-15.0mm feedback bound",
            tool_motion_mm=motion_mm,
        )
    predicted_shift = mapping @ delta
    observed_shift = after - before
    predicted_norm = float(np.linalg.norm(predicted_shift))
    observed_norm = float(np.linalg.norm(observed_shift))
    if predicted_norm < 0.5 or observed_norm < 0.25:
        return None, failure(
            "camera response was too small to update the alignment map",
            predicted_shift_mm=predicted_shift.round(2).tolist(),
            observed_shift_mm=observed_shift.round(2).tolist(),
        )
    if observed_norm > 25.0:
        return None, failure(
            f"observed camera response {observed_norm:.1f}mm exceeds the "
            "25.0mm feedback bound",
            predicted_shift_mm=predicted_shift.round(2).tolist(),
            observed_shift_mm=observed_shift.round(2).tolist(),
        )
    cosine = float(np.dot(predicted_shift, observed_shift) /
                   (predicted_norm * observed_norm))
    if cosine < 0.20:
        return None, failure(
            f"observed camera response contradicts the geometric prior "
            f"(direction cosine {cosine:.2f})",
            predicted_shift_mm=predicted_shift.round(2).tolist(),
            observed_shift_mm=observed_shift.round(2).tolist(),
            direction_cosine=cosine,
        )

    denominator = float(np.dot(delta, delta))
    candidate = mapping + np.outer(
        observed_shift - predicted_shift, delta
    ) / denominator
    try:
        condition = float(np.linalg.cond(candidate))
        singular_values = np.linalg.svd(candidate, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, failure("feedback-updated camera mapping is singular")
    if not np.all(np.isfinite(candidate)) or not math.isfinite(condition) \
            or condition > 20.0 \
            or float(np.min(singular_values)) < 75.0 \
            or float(np.max(singular_values)) > 1800.0:
        return None, failure(
            "feedback-updated camera mapping failed conditioning bounds",
            condition_number=condition,
            singular_values=[float(value) for value in singular_values],
            predicted_shift_mm=predicted_shift.round(2).tolist(),
            observed_shift_mm=observed_shift.round(2).tolist(),
        )

    fitted_shift = candidate @ delta
    return candidate, {
        "ok": True,
        "reason": (
            "accepted a local measured-response update for the next bounded "
            "open-jaw correction; saved camera calibration is unchanged"
        ),
        "tool_motion_mm": round(motion_mm, 2),
        "predicted_shift_mm": predicted_shift.round(2).tolist(),
        "observed_shift_mm": observed_shift.round(2).tolist(),
        "fitted_shift_mm": fitted_shift.round(2).tolist(),
        "direction_cosine": round(cosine, 3),
        "condition_number": round(condition, 3),
    }


def _plan_partial_slot_alignment_correction(
    observed_cam_mm: Any,
    expected_cam_mm: Any,
    rigid_jacobian: np.ndarray,
    fixed_rpy: tuple[float, float, float],
    jaw_plan: dict[str, Any],
    geometry: GG.GraspGeometry,
    object_box_3d: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Find a smaller clear correction that still satisfies the jaw slot.

    This is not a tolerance relaxation. Each candidate predicts the resulting
    camera-frame body pose from the calibrated rigid Jacobian and must pass the
    original axis-aware slot limits before its complete IK/collision path is
    accepted.
    """
    observed = np.asarray(observed_cam_mm, dtype=float)
    expected = np.asarray(expected_cam_mm, dtype=float)
    mapping = np.asarray(rigid_jacobian, dtype=float)
    failures: list[str] = []
    # Try the smallest useful displacement first.  A correction only needs to
    # put the body inside the capture box; centring it wastes clearance and can
    # move a correctly positioned object out of the jaws.
    for fraction in (0.50, 0.55, 0.60, 0.75, 0.90):
        intermediate_target = observed + fraction * (expected - observed)
        route, detail = _plan_endpoint_alignment_correction(
            observed, intermediate_target, mapping, fixed_rpy,
            allow_global=False,
        )
        if route is None:
            failures.append(
                f"{fraction:.2f}: {detail.get('reason')}"
            )
            continue
        delta_base = np.asarray(
            detail.get("base_correction_mm"), dtype=float
        ) / 1000.0
        predicted = observed + mapping @ delta_base
        slot_ok, slot_detail = _jaw_slot_fit(
            predicted, expected, jaw_plan, geometry, object_box_3d
        )
        if not slot_ok:
            failures.append(
                f"{fraction:.2f}: predicted {slot_detail.get('reason')}"
            )
            continue
        return route, {
            **detail,
            "reason": (
                f"accepted {fraction:.2f} scaled fine correction; complete "
                "path is collision-free and predicted body remains inside "
                f"the unchanged jaw slot; {slot_detail.get('reason')}"
            ),
            "partial_fraction": fraction,
            "predicted_body_cam_mm": predicted.round(2).tolist(),
            "predicted_slot": slot_detail,
            "route_kind": "partial_straight_cartesian",
        }
    return None, {
        "reason": (
            "no bounded partial correction was both collision-free and inside "
            "the unchanged jaw slot; " + "; ".join(failures)
        ),
        "attempts": failures,
    }


def _visible_checkpoint_prefix(
    waypoints: list[tuple[float, float, float]],
    endpoint_m: Any,
    current_m: Any,
    *,
    checkpoint_m: float | None = None,
) -> tuple[list[tuple[float, float, float]], np.ndarray, float]:
    """Stop a prevalidated approach while the target remains clearly visible."""
    endpoint = np.asarray(endpoint_m, dtype=float)
    current = np.asarray(current_m, dtype=float)
    distance = float(np.clip(
        HYBRID_CHECKPOINT_M if checkpoint_m is None else checkpoint_m,
        0.070, 0.120,
    ))
    if float(np.linalg.norm(endpoint - current)) <= distance:
        return [], current, float(np.linalg.norm(endpoint - current))
    for index, waypoint in enumerate(waypoints):
        point = np.asarray(waypoint, dtype=float)
        remaining = float(np.linalg.norm(endpoint - point))
        if remaining <= distance:
            return waypoints[:index + 1], point, remaining
    # Every valid full trajectory ends at zero remaining distance, but retain a
    # deterministic fallback for synthetic tests and alternate planners.
    point = np.asarray(waypoints[-1], dtype=float) if waypoints else current
    return list(waypoints), point, float(np.linalg.norm(endpoint - point))


def _dynamic_visible_checkpoint_m(
    geometry: Any,
    frame: tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]],
    requested_cam_mm: Any,
) -> tuple[float, dict[str, Any]]:
    """Choose a pre-grasp observation distance from measured image geometry.

    The close endpoint is deliberately a poor place to ask a semantic model for
    a complete-object mask: the open fingers occlude it and a large object can
    extend beyond the image.  Keep the full-scene trajectory only up to a pose
    where the measured major axis should occupy at most 65% of the short image
    dimension, then take a new stationary RGB-D consensus.  The 70--120 mm
    bounds are the existing safe visible-checkpoint envelope; object size and
    camera intrinsics select the value inside that envelope.
    """
    color, _depth, intr = frame
    height, width = color.shape[:2]
    fx, fy, _cx, _cy = (float(value) for value in intr)
    requested = np.asarray(requested_cam_mm, dtype=float)
    final_depth_mm = float(requested[2]) if requested.shape == (3,) else 65.0
    final_depth_mm = max(1.0, final_depth_mm)

    dimensions_mm: list[float] = []
    for detection in (
        getattr(geometry, "envelope_detection", None),
        getattr(geometry, "grasp_detection", None),
    ):
        if detection is None:
            continue
        for value in (
            getattr(detection, "min_width_mm", None),
            getattr(detection, "max_width_mm", None),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0.0:
                dimensions_mm.append(number)
    box = getattr(geometry, "object_box_3d", None)
    if isinstance(box, dict):
        for value in box.get("size_mm", []):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0.0:
                dimensions_mm.append(number)

    major_mm = max(dimensions_mm) if dimensions_mm else 0.0
    usable_span_px = max(1.0, 0.65 * float(min(height, width)))
    f_max = max(fx, fy, 1.0)
    fit_depth_mm = (
        f_max * major_mm / usable_span_px
        if major_mm > 0.0 else final_depth_mm + HYBRID_CHECKPOINT_M * 1000.0
    )
    # Preserve at least the old 70 mm checkpoint clearance even for compact
    # objects. Larger measured objects move the observation pose farther away
    # only as much as the calibrated 120 mm checkpoint envelope permits.
    observation_depth_mm = max(final_depth_mm + 70.0, fit_depth_mm)
    remaining_m = float(np.clip(
        (observation_depth_mm - final_depth_mm) / 1000.0,
        0.070, 0.120,
    ))
    return remaining_m, {
        "major_mm": round(major_mm, 1),
        "final_depth_mm": round(final_depth_mm, 1),
        "fit_depth_mm": round(fit_depth_mm, 1),
        "observation_depth_mm": round(final_depth_mm + remaining_m * 1000.0, 1),
        "remaining_mm": round(remaining_m * 1000.0, 1),
        "usable_image_fraction": 0.65,
    }


def _plan_checkpoint_reacquire_retreat(
    executed_prefix: Any,
    checkpoint_start_m: Any,
    camera_mapping: np.ndarray,
    fixed_rpy: tuple[float, float, float],
) -> tuple[list[tuple[float, float, float]] | None, str]:
    """Plan a small retreat to recover a lost close-range observation.

    Cartesian approaches reverse the exact path that was just executed.  A
    certified global joint prefix has no reusable certificate from its new end
    state, so that uncommon case uses a newly IK/collision-validated camera-
    axial backoff.  Neither branch authorizes a stale forward continuation.
    """
    try:
        requested_m = float(os.getenv(
            "REBOT_VISIBLE_CHECKPOINT_REACQUIRE_BACKOFF_M", "0.035"
        ))
    except (TypeError, ValueError):
        requested_m = 0.035
    requested_m = float(np.clip(requested_m, 0.020, 0.060))
    current = np.asarray(A.ee_pos(), dtype=float)
    start = np.asarray(checkpoint_start_m, dtype=float)
    if current.shape != (3,) or start.shape != (3,) \
            or not np.all(np.isfinite(current)) \
            or not np.all(np.isfinite(start)):
        return None, "checkpoint retreat endpoints are invalid"

    if isinstance(executed_prefix, list) and executed_prefix:
        chain = [start]
        for point in executed_prefix:
            candidate = np.asarray(point, dtype=float)
            if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                return None, "executed checkpoint path contains invalid points"
            chain.append(candidate)
        nearest_index = min(
            range(len(chain)),
            key=lambda index: float(np.linalg.norm(current - chain[index])),
        )
        nearest_error_m = float(np.linalg.norm(
            current - chain[nearest_index]
        ))
        if nearest_error_m > 0.015:
            return None, (
                "live arm pose is no longer on the executed checkpoint path "
                f"(nearest error {nearest_error_m * 1000.0:.1f}mm)"
            )
        retreat: list[tuple[float, float, float]] = []
        previous = current
        distance_m = 0.0
        # Begin immediately before the live point on the executed chain. This
        # remains monotonic if a configured second retry starts part-way back.
        for point in reversed(chain[:nearest_index]):
            distance_m += float(np.linalg.norm(previous - point))
            retreat.append(tuple(float(value) for value in point))
            previous = point
            if distance_m >= requested_m:
                break
        if not retreat:
            return None, "executed checkpoint path has no reversible segment"
        valid, reason = _validate_fixed_waypoints(retreat, fixed_rpy)
        if not valid:
            return None, f"reverse checkpoint path is no longer safe: {reason}"
        return retreat, (
            f"{len(retreat)} previously executed waypoints reversed for "
            f"{distance_m * 1000.0:.1f}mm of reacquisition clearance; {reason}"
        )

    try:
        delta_base = np.linalg.solve(
            np.asarray(camera_mapping, dtype=float),
            np.array([0.0, 0.0, requested_m * 1000.0], dtype=float),
        )
    except (ValueError, np.linalg.LinAlgError):
        return None, "camera mapping is singular for checkpoint backoff"
    endpoint = current + delta_base
    retreat, reason = _plan_fixed_approach_trajectory(
        endpoint, start_m=current, fixed_rpy=fixed_rpy
    )
    if retreat is None:
        return None, f"camera-axial checkpoint backoff is unsafe: {reason}"
    return retreat, (
        f"new {requested_m * 1000.0:.1f}mm camera-axial backoff; {reason}"
    )


def _endpoint_from_camera_body(
    current_m: Any,
    observed_body_cam_mm: Any,
    requested_body_cam_mm: Any,
    rigid_jacobian: np.ndarray,
) -> np.ndarray | None:
    """Exact base endpoint implied by a fresh physical-body observation."""
    try:
        delta = np.linalg.solve(
            np.asarray(rigid_jacobian, dtype=float),
            np.asarray(requested_body_cam_mm, dtype=float)
            - np.asarray(observed_body_cam_mm, dtype=float),
        )
    except np.linalg.LinAlgError:
        return None
    endpoint = np.asarray(current_m, dtype=float) + delta
    return endpoint if np.all(np.isfinite(endpoint)) else None


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


def _endpoint_retry_eligibility(
    error_mm: float, *, object_visible: bool,
) -> dict[str, Any]:
    """Decide whether bounded corrections can reach the close-measure gate.

    ``_endpoint_retry_envelope_mm`` is motion capacity, not the maximum raw
    image error.  The arm need only remove error outside the tight endpoint
    measurement tolerance. Comparing raw error directly to motion capacity
    rejected a 37.9 mm observation even though three 12 mm corrections could
    bring it inside the unchanged 8 mm gate.
    """
    capacity_mm = _endpoint_retry_envelope_mm(
        object_visible=object_visible
    )
    if DYNAMIC_GRASP_ENABLED and object_visible:
        residual_tolerance_mm = float(np.clip(
            DYNAMIC_SLOT_TOL_MM, 3.0, 8.0
        ))
        required_correction_mm = max(
            0.0, float(error_mm) - residual_tolerance_mm
        )
        eligible = bool(
            np.isfinite(required_correction_mm)
            and required_correction_mm <= capacity_mm
        )
    else:
        residual_tolerance_mm = float(FIXED_PICK_TARGET_TOL_MM)
        required_correction_mm = max(
            0.0, float(error_mm) - residual_tolerance_mm
        )
        eligible = bool(
            np.isfinite(error_mm)
            and float(error_mm) <= capacity_mm
        )
    return {
        "eligible": eligible,
        "error_mm": float(error_mm),
        "correction_capacity_mm": capacity_mm,
        "residual_tolerance_mm": residual_tolerance_mm,
        "required_correction_mm": required_correction_mm,
    }


def _plan_scene_grasp_geometry(
    frame: tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]],
    det: WD.Detection,
    *,
    current_rpy: tuple[float, float, float],
    allow_orientation: bool = True,
    object_name: str | None = None,
    metric_jaw_width_mm: float | None = None,
    metric_major_mm: float | None = None,
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
    semantic_attempts: list[dict[str, Any]] = []
    vision_enabled = os.getenv(
        "REBOT_ENABLE_LOCATE_ANYTHING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    semantic_sources = {"grounding_dino_sam2", "locate_anything", "sam3"}
    if (getattr(det, "info", {}) or {}).get(
        "endpoint_width_basis"
    ) == "registered_shape_plus_slot_occupancy":
        # This mask was measured only inside the calibrated open-jaw volume
        # after a fully prevalidated named-target approach. Keep it local;
        # generic table-plane segmentation would expand it back into the scene.
        semantic_envelope = det
    # Reuse the native mask from the detection that selected this exact frame.
    # SAM3 keeps that mask alongside its box; issuing a second text query first
    # overwrote it and made whole-object geometry fluctuate frame to frame.
    if vision_enabled and det.source in semantic_sources:
        diagnostics: dict[str, Any] = {"attempt": "primary_native_mask"}
        semantic_envelope = WD.segment_semantic_envelope(
            color, depth, intr, det, det,
            near_mm=WD.NEAR_MM, far_mm=WD.FAR_MM,
            diagnostics=diagnostics,
        )
        diagnostics["attempt"] = "primary_native_mask"
        semantic_attempts.append(diagnostics)
    # Compatibility fallback for a colour identifier or a backend without a
    # native mask. This is intentionally second, not the normal SAM3 path.
    if semantic_envelope is None and object_name and vision_enabled:
        semantic_box = WD.detect_vlm(
            color, depth, intr, f"the complete {object_name}",
            near_mm=WD.NEAR_MM, far_mm=WD.FAR_MM,
        )
        if semantic_box is None:
            semantic_attempts.append({
                "attempt": "complete_object_query",
                "stage": "semantic_detection",
                "reason": "complete-object semantic query returned no box",
            })
        else:
            diagnostics = {"attempt": "complete_object_query"}
            semantic_envelope = WD.segment_semantic_envelope(
                color, depth, intr, semantic_box, det,
                near_mm=WD.NEAR_MM, far_mm=WD.FAR_MM,
                diagnostics=diagnostics,
            )
            diagnostics["attempt"] = "complete_object_query"
            semantic_attempts.append(diagnostics)
    yaw_bounds = None
    if allow_orientation and os.getenv(
        "REBOT_WRIST_USE_SAVED_CAMERA_ROTATION", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        yaw_bounds, yaw_budget_detail = _camera_safe_yaw_bounds(
            current_rpy, GEOMETRY_MAX_YAW_DELTA_DEG
        )
        if yaw_bounds is None:
            return GG.GraspGeometry(
                False, str(yaw_budget_detail.get("reason")),
                det, semantic_envelope, None,
            )
    planned = GG.plan_grasp_geometry(
        color, depth, intr, det,
        current_rpy=current_rpy,
        max_aperture_mm=float(max_aperture),
        jaw_long_axis_camera_deg=axis,
        max_yaw_delta_deg=GEOMETRY_MAX_YAW_DELTA_DEG,
        yaw_delta_bounds_deg=yaw_bounds,
        yaw_measurement_slop_deg=GEOMETRY_YAW_MEASUREMENT_SLOP_DEG,
        max_residual_yaw_deg=GEOMETRY_MAX_RESIDUAL_YAW_DEG,
        allow_aperture_derived_residual=GEOMETRY_APERTURE_DERIVED_RESIDUAL,
        final_approach_mm=GEOMETRY_FINAL_APPROACH_MM,
        min_feature_prominence_mm=GEOMETRY_MIN_FEATURE_PROMINENCE_MM,
        require_axis_for_elongated=allow_orientation,
        semantic_envelope=semantic_envelope,
        metric_jaw_width_mm=metric_jaw_width_mm,
        metric_major_mm=metric_major_mm,
    )
    if not planned.ok and semantic_attempts:
        compact = [
            {
                key: attempt.get(key) for key in (
                    "attempt", "semantic_source", "stage", "reason",
                    "native_mask_available", "mask_pixels",
                    "valid_depth_pixels", "valid_depth_fraction",
                ) if attempt.get(key) is not None
            }
            for attempt in semantic_attempts
        ]
        planned = replace(
            planned,
            reason=f"{planned.reason}; semantic-mask diagnostics={compact}",
        )
    return planned


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


def _exact_semantic_geometry_mask(
    geometry: Any, detection: Any | None = None,
) -> tuple[bool, str]:
    """Return whether an observation carries a native mask for this target.

    Mask authority is deliberately independent of preliminary 2D grasp
    acceptance.  A perspective-oblique object can have an excellent native
    semantic mask while its silhouette produces an impossible jaw width.  The
    metric recovery path needs that exact mask to measure the object in 3D and
    correct the bad 2D estimate; it still applies the ordinary aperture, IK,
    and collision gates before motion.
    """
    # The primary SAM3 observation stores its model mask directly on the
    # detection. A later semantic-envelope cleanup normally mirrors that
    # authority, but a frame-edge crop can make the derived envelope fail even
    # though the model mask and its ranged pixels remain valid. This direct
    # branch is used by bottom-only metric recovery after another view has
    # already established the complete 3D box.
    detection_info = getattr(detection, "info", {}) or {}
    detection_source = str(getattr(detection, "source", ""))
    if detection_source in _SEMANTIC_IDENTITY_SOURCES \
            and detection_info.get("mask_u8") is not None \
            and detection_info.get("depth_basis") in {
                "native_semantic_mask", "semantic_seeded_rgbd_object"
            }:
        if detection_info.get("depth_basis") == "semantic_seeded_rgbd_object" \
                and not detection_info.get("object_completion_validated"):
            return False, "RGB-D object completion was not identity-validated"
        return True, (
            "semantic-seeded complete RGB-D object mask"
            if detection_info.get("object_completion_validated") else
            "exact native semantic target detection mask"
        )
    if geometry is None:
        return False, "grasp observation is unavailable"
    envelope = getattr(geometry, "envelope_detection", None)
    info = getattr(envelope, "info", {}) or {}
    identity_source = (
        str(getattr(detection, "source", "")) if detection is not None else
        str(info.get("semantic_identity_source")
            or info.get("identifier_source", ""))
    )
    if identity_source not in _SEMANTIC_IDENTITY_SOURCES:
        return False, "target position came from non-semantic RGB-D tracking"
    if str(getattr(envelope, "source", "")) != "rgbd:semantic_envelope":
        return False, "accepted geometry does not use a semantic envelope"
    completion_validated = bool(info.get("object_completion_validated"))
    if not completion_validated and not str(
        info.get("segmentation_source", "")
    ).endswith(":native_mask"):
        return False, "semantic envelope does not carry a native model mask"
    if info.get("mask_u8") is None:
        return False, "native semantic object mask is unavailable"
    return True, (
        "semantic-seeded complete RGB-D object mask"
        if completion_validated else "exact native semantic target mask"
    )


def _filter_target_points_to_identity_cluster(
    points_base_m: Any,
    anchor_base_m: Any | None,
    radius_m: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove semantic-mask spill outside the established target identity.

    The same metric radius used to reject a wrong semantic detection is also a
    natural bound for its 3-D mask points. This keeps nearby scene points from
    turning a valid small object into a huge fused OBB without introducing any
    object-name or dimension-specific threshold.
    """
    points = np.asarray(points_base_m, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        return np.empty((0, 3), dtype=float), {
            "ok": False, "reason": "target points have invalid dimensions"
        }
    if anchor_base_m is None or radius_m is None:
        return points, {
            "ok": True, "reason": "identity-cluster filtering not requested",
            "kept": int(len(points)), "removed": 0,
        }
    try:
        anchor = np.asarray(anchor_base_m, dtype=float)
        radius = float(radius_m)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=float), {
            "ok": False, "reason": "target identity cluster is invalid"
        }
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)) \
            or not math.isfinite(radius) or radius <= 0.0:
        return np.empty((0, 3), dtype=float), {
            "ok": False, "reason": "target identity cluster is invalid"
        }
    finite = np.all(np.isfinite(points), axis=1)
    distances = np.full(len(points), math.inf, dtype=float)
    distances[finite] = np.linalg.norm(points[finite] - anchor, axis=1)
    kept = points[distances <= radius]
    return kept, {
        "ok": len(kept) >= 20,
        "reason": (
            f"kept {len(kept)}/{len(points)} semantic target points within "
            f"the established {radius * 1000.0:.1f}mm identity cluster"
        ),
        "kept": int(len(kept)),
        "removed": int(len(points) - len(kept)),
        "radius_mm": radius * 1000.0,
    }


def _export_discovery_cloud(
    captured_views: list[dict[str, Any]], log: Callable[[str], None],
    *, force_geometry: bool = False,
    require_semantic_target_masks: bool = False,
    target_anchor_base_m: Any | None = None,
    target_anchor_radius_m: float | None = None,
) -> dict[str, Any]:
    """Fuse valid swept views even when 2D grasp planning stops early.

    Normally this is diagnostic-only. ``force_geometry`` permits the measured
    target OBB to rescue a perspective-inflated 2D grasp, but still never
    installs a collision scene. Each cloud is lifted with its capture-time arm
    pose and rotated camera mapping, then fused in the base frame.
    """
    if not SCENE_COLLISION_EXPORT_VIEW and not force_geometry:
        return {"ok": False, "reason": "diagnostic export disabled"}
    if not captured_views:
        return {"ok": False, "reason": "no pose-aware discovery views"}
    # Base-frame point lifting must be anchored at the URDF tool/TCP point,
    # whose base coordinate is ``ee_xyz_m``.  The open-jaw grasp target is a
    # different physical point inside the fingers and is used only for visual
    # alignment. Mixing the two offsets the entire cloud relative to both the
    # robot collision model and metric target OBB.
    tool_camera = load_tcp()
    table_z = load_safety_calibration().get("table_z_m")
    if tool_camera is None or table_z is None:
        return {"ok": False, "reason": "TCP or table calibration missing"}
    try:
        clouds = []
        target_point_sets: list[tuple[int, np.ndarray]] = []
        names = []
        named_clouds = {}
        for view in captured_views:
            frame = view.get("frame")
            mapping = np.asarray(view.get("camera_mapping"), dtype=float)
            ee_xyz = np.asarray(view.get("ee_xyz_m"), dtype=float)
            if frame is None or mapping.shape != (3, 3) or ee_xyz.shape != (3,):
                continue
            color, depth, intr = frame
            cloud = SC.build_dense_view_cloud(
                color, depth, intr,
                ee_xyz_m=ee_xyz,
                camera_mapping_mm_per_m=mapping,
                jaw_target_cam_mm=tool_camera,
                workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
                workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
                # Preserve raw FK/hand-eye geometry. Per-view table snapping
                # can translate a complete view to an object top and was the
                # source of doubled scene layers.
                table_z_m=None,
                near_mm=float(WD.NEAR_MM), far_mm=float(WD.FAR_MM),
                stride_px=1, voxel_m=0.0015, bounds_margin_m=0.08,
                minimum_neighbor_support=2,
                crop_to_workspace=True, crop_below_table=False,
            )
            clouds.append(cloud)
            view_name = str(view.get("name") or f"view_{len(names) + 1}")
            names.append(view_name)
            named_clouds[view_name] = cloud
            geometry = view.get("geometry")
            detection = view.get("detection")
            envelope = None if geometry is None else (
                geometry.envelope_detection or geometry.grasp_detection
            )
            # Dense scene clouds may retain partial/rejected views for obstacle
            # coverage. Normally only accepted grasp geometry contributes to
            # the target OBB. Metric recovery additionally admits an exact
            # native semantic mask whose *2D grasp* failed: the measured OBB is
            # precisely what can disambiguate a perspective-inflated width.
            exact_mask_ok, exact_mask_reason = \
                _exact_semantic_geometry_mask(geometry, detection)
            preliminary_grasp_ok = bool(getattr(geometry, "ok", False))
            target_mask_eligible = bool(
                envelope is not None
                and (
                    preliminary_grasp_ok
                    or (force_geometry and exact_mask_ok)
                )
                and (not require_semantic_target_masks or exact_mask_ok)
            )
            if target_mask_eligible:
                try:
                    target_points = OG3.masked_points_base(
                        frame, envelope, ee_xyz_m=ee_xyz,
                        camera_mapping_mm_per_m=mapping,
                        tool_camera_mm=tool_camera,
                        near_mm=float(WD.NEAR_MM), far_mm=float(WD.FAR_MM),
                    )
                    target_points, cluster_detail = \
                        _filter_target_points_to_identity_cluster(
                            target_points, target_anchor_base_m,
                            target_anchor_radius_m,
                        )
                    if cluster_detail.get("removed", 0):
                        log(
                            "stage=discovery_target_identity_filter "
                            f"view={view_name!r} "
                            f"ok={cluster_detail.get('ok')} "
                            f"detail={cluster_detail.get('reason')}"
                        )
                    if len(target_points) >= 20:
                        target_point_sets.append((
                            len(clouds) - 1, target_points
                        ))
                except (TypeError, ValueError):
                    pass
            elif require_semantic_target_masks and envelope is not None:
                log(
                    "stage=discovery_target_mask status=skipped "
                    f"view={view_name!r} source="
                    f"{getattr(detection, 'source', None)} "
                    f"detail={exact_mask_reason}"
                )
        clouds = SC.crop_dense_view_clouds(
            clouds,
            table_z_m=float(table_z),
            workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
            workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
            bounds_margin_m=0.08,
            crop_below_table=True,
        )
        log(
            "stage=discovery_cloud_geometry "
            "table_snap=disabled coordinates=raw_fk_hand_eye"
        )
        registered_clouds, registration = SC.register_dense_view_clouds(
            clouds, table_z_m=float(table_z),
            max_translation_m=SCENE_DENSE_REGISTRATION_MAX_TRANSLATION_M,
            minimum_overlap=SCENE_DENSE_REGISTRATION_MIN_OVERLAP,
        )
        named_clouds = {
            name: cloud for name, cloud in zip(names, registered_clouds)
        }
        for report in registration[1:]:
            log(
                "stage=discovery_cloud_registration "
                f"view={report.get('view')} accepted={report.get('accepted')} "
                f"fused={report.get('valid_for_fusion')} "
                f"translation_mm={report.get('translation_mm')} "
                f"overlap={report.get('zero_overlap')}->"
                f"{report.get('best_overlap')} detail={report.get('reason')}"
            )
        fusable_clouds = [
            cloud for cloud, report in zip(registered_clouds, registration)
            if bool(report.get("valid_for_fusion", False))
        ]
        points, colors = SC.fuse_dense_view_clouds(
            fusable_clouds, voxel_m=0.0015
        )
        if not len(points):
            return {"ok": False, "reason": "discovery views produced no points"}
        empty_scene = SC.SceneMap(
            boxes=(), table_z_m=float(table_z),
            voxel_m=SCENE_COLLISION_VOXEL_M,
            inflation_m=SCENE_COLLISION_INFLATION_M,
            valid_depth_points=0, mapped_obstacle_points=0,
            excluded_target_points=0, closest_obstacle_m=None,
        )
        object_boxes_3d = None
        fused_target_box = None
        if target_point_sets:
            registered_targets = []
            for view_index, target_points in target_point_sets:
                if not bool(registration[view_index].get(
                    "valid_for_fusion", False
                )):
                    continue
                translation = np.asarray(
                    registration[view_index].get("translation_mm", [0.0] * 3),
                    dtype=float,
                ) / 1000.0
                registered_targets.append(
                    np.asarray(target_points, dtype=float) + translation
                )
            if registered_targets:
                fused_target_box = OG3.fit_tabletop_box(
                    registered_targets, table_z_m=float(table_z)
                ).as_dict()
                object_boxes_3d = {"discovery_target": fused_target_box}
                log(
                    "stage=discovery_target_box ok=True "
                    f"views={fused_target_box['views']} "
                    f"observed_mm="
                    f"{(np.asarray(fused_target_box['observed_size_m']) * 1000.0).round(1).tolist()} "
                    f"orientation={fused_target_box['orientation_mode']}"
                )
        artifacts = {}
        if SCENE_COLLISION_EXPORT_VIEW:
            artifacts = SC.save_scene_artifacts(
                empty_scene, DEBUG_DIR / "scene_collision",
                stem="latest_discovery",
                view_points_m=points, view_colors_rgb=colors,
                named_view_clouds=(
                    named_clouds if SCENE_COLLISION_EXPORT_PER_VIEW else {}
                ),
                object_boxes_3d=object_boxes_3d,
                show_box_labels=SCENE_COLLISION_EXPORT_BOX_LABELS,
            )
        result = {
            "ok": True,
            **artifacts,
            "views": len(clouds),
            "fused_views": len(fusable_clouds),
            "rejected_views": [
                names[index] for index, report in enumerate(registration)
                if not bool(report.get("valid_for_fusion", False))
            ],
            "view_names": names,
            "per_view_clouds_exported": SCENE_COLLISION_EXPORT_PER_VIEW,
            "dense_rgb_points": int(len(points)),
            "registration": registration,
            "table_alignment": {
                "correction_mm": 0.0,
                "reason": "per-view table snapping disabled",
            },
            "object_box_3d": fused_target_box,
            "collision_model_installed": False,
        }
        log(
            "stage=discovery_cloud_export ok=True "
            f"views={len(clouds)} points={len(points)} "
            f"rrd={artifacts.get('rerun')}"
        )
        return result
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        log(f"stage=discovery_cloud_export ok=False detail={reason}")
        return {"ok": False, "reason": reason}


def _recover_multiview_metric_grasp(
    eye: WristEye,
    description: str,
    captured_views: list[dict[str, Any]],
    log: Callable[[str], None],
) -> tuple[Any | None, Any | None, GG.GraspGeometry | None, dict[str, Any]]:
    """Use the swept metric OBB when a tilted 2D silhouette blocks grasping.

    The sweep may contain complete semantic masks whose preliminary 2D grasp
    width is perspective-inflated. Fuse those views first, then revisit the
    closest capture pose and replan using the OBB's two measured tabletop axes.
    Every return motion and the resulting grasp retain ordinary IK/collision
    gates; the metric box only replaces the unreliable 2D sizing input.
    """
    cloud = _export_discovery_cloud(
        captured_views, log, force_geometry=True
    )
    width_mm, major_mm, dimension_detail = \
        _metric_horizontal_box_dimensions(cloud.get("object_box_3d"))
    if width_mm is None or major_mm is None:
        return None, None, None, {
            "ok": False,
            "reason": dimension_detail.get("reason"),
            "discovery_cloud": cloud,
        }
    current_xyz = np.asarray(A.ee_pos(), dtype=float)
    candidates = []
    for view in captured_views:
        frame = view.get("frame")
        det = view.get("detection")
        xyz = np.asarray(view.get("ee_xyz_m"), dtype=float)
        rpy = np.asarray(view.get("ee_rpy_rad"), dtype=float)
        if frame is None or det is None or xyz.shape != (3,) or rpy.shape != (3,):
            continue
        partial = _detection_partial(det, frame)
        geometry = _plan_scene_grasp_geometry(
            frame, det,
            current_rpy=tuple(float(value) for value in rpy),
            object_name=description,
            metric_jaw_width_mm=width_mm,
            metric_major_mm=major_mm,
        )
        # Once a complete native-mask view has supplied a metric 3D box, a
        # second view clipped *only at the table-side bottom edge* can safely
        # contribute pose/alignment. Its missing long-axis extent comes from
        # the fused OBB, not from the cropped silhouette. Left/right/top clips
        # remain rejected because they can hide the local jaw cross-section.
        if partial is not None:
            native_mask_ok, _native_reason = \
                _exact_semantic_geometry_mask(geometry, det)
            bottom_only = partial == (
                "semantic/depth envelope is clipped at the image bottom edge"
            )
            if not (bottom_only and native_mask_ok):
                continue
        if not geometry.ok:
            log(
                "stage=multiview_metric_grasp_candidate "
                f"view={view.get('name')} ok=False detail={geometry.reason}"
            )
            continue
        candidates.append((
            float(np.linalg.norm(xyz - current_xyz)), view, geometry
        ))
        log(
            "stage=multiview_metric_grasp_candidate "
            f"view={view.get('name')} ok=True "
            f"distance_mm={1000.0 * candidates[-1][0]:.1f} "
            f"detail={geometry.reason}"
        )
    fused_box = cloud.get("object_box_3d")
    for _distance, view, validated_geometry in sorted(
        candidates, key=lambda item: item[0]
    ):
        target_xyz = np.asarray(view["ee_xyz_m"], dtype=float)
        target_rpy = tuple(float(value) for value in view["ee_rpy_rad"])
        live_xyz = np.asarray(A.ee_pos(), dtype=float)
        live_rpy = tuple(float(value) for value in A.ee_rpy())
        orientation_path, orientation_reason = _plan_rpy_trajectory(
            target_rpy, xyz=live_xyz
        )
        if orientation_path is None:
            # At the retreated RS pose, returning from +yaw to the neutral
            # camera orientation can cross a local singularity. The same
            # target orientation is often reachable after retracing the
            # already-observed translation first. Both legs remain completely
            # prevalidated; this changes restoration order, not its endpoint.
            log(
                "stage=multiview_metric_grasp_restore "
                f"view={view.get('name')} phase=orientation_first "
                f"status=blocked detail={orientation_reason}; "
                "fallback=translate_first"
            )
            position_path, position_reason = _plan_fixed_approach_trajectory(
                target_xyz, fixed_rpy=live_rpy
            )
            if position_path is None:
                continue
            positioned, position_reason = _execute_fixed_approach_trajectory(
                position_path, log, fixed_rpy=live_rpy
            )
            if not positioned:
                continue
            orientation_path, orientation_reason = _plan_rpy_trajectory(
                target_rpy, xyz=target_xyz
            )
            if orientation_path is None:
                continue
            oriented, orientation_reason = _execute_rpy_trajectory(
                orientation_path, log, fixed_xyz=target_xyz
            )
            if not oriented:
                continue
        else:
            oriented, orientation_reason = _execute_rpy_trajectory(
                orientation_path, log
            )
            if not oriented:
                continue
        position_path, position_reason = _plan_fixed_approach_trajectory(
            target_xyz, fixed_rpy=target_rpy
        )
        if position_path is None:
            continue
        positioned, position_reason = _execute_fixed_approach_trajectory(
            position_path, log, fixed_rpy=target_rpy
        )
        if not positioned:
            continue
        _settle(eye, note="multiview metric grasp restore")
        fresh_det, fresh_frame, fresh_why = _observe(eye, description)
        if fresh_det is None or fresh_frame is None:
            continue
        partial = _detection_partial(fresh_det, fresh_frame)
        fresh_geometry = _plan_scene_grasp_geometry(
            fresh_frame, fresh_det,
            current_rpy=tuple(float(value) for value in A.ee_rpy()),
            object_name=description,
            metric_jaw_width_mm=width_mm,
            metric_major_mm=major_mm,
        )
        fresh_native_mask, fresh_native_reason = \
            _exact_semantic_geometry_mask(fresh_geometry, fresh_det)
        if partial is not None:
            if not fresh_native_mask:
                log(
                    "stage=multiview_metric_grasp_restore "
                    f"view={view.get('name')} ok=False phase=fresh_identity "
                    f"detail=clipped fresh view lacks an exact semantic mask: "
                    f"{fresh_native_reason}"
                )
                continue
            # The capture pose was already accepted using the fused metric OBB
            # and the fresh exact semantic mask re-establishes target identity.
            # Do not require the fresh silhouette to become complete as well:
            # edge clipping is precisely why the bounded sweep invoked metric
            # recovery.  Entry clearance is still decided below by projecting
            # every measured OBB axis into the calibrated jaw-closing direction.
            log(
                "stage=multiview_metric_grasp_restore "
                f"view={view.get('name')} phase=fresh_identity "
                "status=accepted_clipped_native_mask "
                f"detail={partial}; safety_gate=full_obb_jaw_projection"
            )
        if not fresh_geometry.ok:
            if not fresh_native_mask:
                log(
                    "stage=multiview_metric_grasp_restore "
                    f"view={view.get('name')} ok=False "
                    "phase=fresh_single_frame_geometry "
                    f"detail={fresh_geometry.reason}; semantic_gate="
                    f"{fresh_native_reason}"
                )
                continue
            # This exact pose already produced ``validated_geometry`` from the
            # fused metric dimensions. A fresh single-frame silhouette may
            # fluctuate back to the perspective-inflated yaw that triggered
            # multiview recovery in the first place. Fresh SAM identity proves
            # the stationary target is still present; retain the captured
            # metric grasp rather than letting that noisy 2D yaw overwrite it.
            log(
                "stage=multiview_metric_grasp_restore "
                f"view={view.get('name')} "
                "phase=fresh_single_frame_geometry "
                "status=retained_fused_metric_candidate "
                f"rejected_fresh_detail={fresh_geometry.reason}"
            )
            fresh_det = view["detection"]
            fresh_frame = view["frame"]
            fresh_geometry = validated_geometry
        # The captured candidate was accepted using the fused metric OBB, but
        # a second single-frame long-axis estimate used to overwrite its
        # validated wrist yaw here. On broad/tilted faces that estimate can
        # jump by ~20deg and turn a 22mm object thickness into a false 45mm
        # silhouette. Keep the restored metric orientation only after
        # projecting the *full* OBB into the jaws at that pose and proving both
        # uncertainty and entry clearance fit the calibrated aperture.
        view_mapping = np.asarray(view.get("camera_mapping"), dtype=float)
        projected_width = _projected_3d_jaw_span(
            fused_box, view_mapping
        ) if view_mapping.shape == (3, 3) else {
            "ok": False, "reason": "restored view camera mapping is invalid"
        }
        from arm_agent import gripper_model as gm
        calibrated_aperture = gm.max_aperture_mm(_wrist_limits())
        if calibrated_aperture is None:
            log(
                "stage=multiview_metric_grasp_restore "
                f"view={view.get('name')} ok=False phase=metric_projection "
                "detail=jaw millimetre calibration is unavailable"
            )
            continue
        max_aperture = float(calibrated_aperture)
        projection_safe = bool(
            projected_width.get("ok")
            and float(projected_width["conservative_width_mm"])
                <= max_aperture + 1e-6
            and float(projected_width["observed_width_mm"])
                + max(0.0, GRASP_MIN_OPEN_CLEARANCE_MM)
                <= max_aperture + 1e-6
        )
        if not projection_safe and projected_width.get("ok"):
            # The fused OBB can prove that the object has a narrow grasp axis,
            # while a captured recovery view leaves that axis a few degrees
            # oblique to the jaws.  Refusing a 42.8 mm observed span because
            # its uncertainty envelope is 50.1 mm throws away an otherwise
            # valid 28 mm cross-section. Search only wrist yaw at the already
            # safe stationary XYZ; never enlarge the aperture or reduce the
            # configured entry-clearance requirement.
            base_rpy = np.asarray(A.ee_rpy(), dtype=float)
            yaw_step_deg = float(np.clip(float(os.getenv(
                "REBOT_METRIC_RECOVERY_YAW_STEP_DEG", "2.0"
            )), 1.0, 4.0))
            yaw_limit_deg = float(np.clip(float(os.getenv(
                "REBOT_METRIC_RECOVERY_MAX_YAW_DEG",
                str(GEOMETRY_MAX_YAW_DELTA_DEG),
            )), yaw_step_deg, GEOMETRY_MAX_YAW_DELTA_DEG))
            yaw_magnitudes = np.arange(
                yaw_step_deg, yaw_limit_deg + 0.5 * yaw_step_deg,
                yaw_step_deg,
            )
            yaw_offsets = [
                signed
                for magnitude in yaw_magnitudes
                for signed in (float(magnitude), -float(magnitude))
            ]
            metric_orientation = None
            metric_orientation_path = None
            metric_mapping = None
            metric_projection = None
            for yaw_offset_deg in yaw_offsets:
                candidate_rpy_array = base_rpy.copy()
                candidate_rpy_array[2] += math.radians(yaw_offset_deg)
                candidate_rpy = tuple(
                    float(value) for value in candidate_rpy_array
                )
                candidate_mapping, _mapping_detail = \
                    load_saved_camera_jacobian(candidate_rpy)
                if candidate_mapping is None:
                    continue
                candidate_projection = _projected_3d_jaw_span(
                    fused_box, candidate_mapping
                )
                if not candidate_projection.get("ok"):
                    continue
                candidate_safe = bool(
                    float(candidate_projection["conservative_width_mm"])
                        <= max_aperture + 1e-6
                    and float(candidate_projection["observed_width_mm"])
                        + max(0.0, GRASP_MIN_OPEN_CLEARANCE_MM)
                        <= max_aperture + 1e-6
                )
                if not candidate_safe:
                    continue
                candidate_path, _candidate_reason = _plan_rpy_trajectory(
                    candidate_rpy, xyz=A.ee_pos()
                )
                if candidate_path is None:
                    continue
                metric_orientation = candidate_rpy
                metric_orientation_path = candidate_path
                metric_mapping = candidate_mapping
                metric_projection = candidate_projection
                log(
                    "stage=multiview_metric_grasp_orientation "
                    f"status=planned yaw_offset_deg={yaw_offset_deg:+.1f} "
                    f"observed_width_mm="
                    f"{float(candidate_projection['observed_width_mm']):.1f} "
                    f"conservative_width_mm="
                    f"{float(candidate_projection['conservative_width_mm']):.1f}"
                )
                break
            if metric_orientation_path is not None \
                    and metric_orientation is not None \
                    and metric_mapping is not None \
                    and metric_projection is not None:
                oriented, orientation_reason = _execute_rpy_trajectory(
                    metric_orientation_path, log, fixed_xyz=A.ee_pos()
                )
                if oriented:
                    target_rpy = metric_orientation
                    view_mapping = np.asarray(metric_mapping, dtype=float)
                    projected_width = metric_projection
                    projection_safe = True
                    _settle(eye, note="metric grasp recovery yaw alignment")
                    aligned_det, aligned_frame, _aligned_why = _observe(
                        eye, description
                    )
                    if aligned_det is None or aligned_frame is None:
                        projection_safe = False
                    else:
                        aligned_partial = _detection_partial(
                            aligned_det, aligned_frame
                        )
                        aligned_geometry = _plan_scene_grasp_geometry(
                            aligned_frame, aligned_det,
                            current_rpy=target_rpy,
                            object_name=description,
                            metric_jaw_width_mm=width_mm,
                            metric_major_mm=major_mm,
                        )
                        if aligned_partial is not None:
                            aligned_native, aligned_native_reason = \
                                _exact_semantic_geometry_mask(
                                    aligned_geometry, aligned_det
                                )
                            if not aligned_native:
                                projection_safe = False
                                log(
                                    "stage=multiview_metric_grasp_orientation "
                                    "status=rejected phase=fresh_identity "
                                    f"detail={aligned_native_reason}"
                                )
                        if not aligned_geometry.ok:
                            projection_safe = False
                        if projection_safe:
                            fresh_det = aligned_det
                            fresh_frame = aligned_frame
                            fresh_geometry = aligned_geometry
                            log(
                                "stage=multiview_metric_grasp_orientation "
                                "status=accepted detail=full OBB projection "
                                "and fresh target observation passed"
                            )
                else:
                    log(
                        "stage=multiview_metric_grasp_orientation "
                        f"status=execution_failed detail={orientation_reason}"
                    )
        if not projection_safe:
            log(
                "stage=multiview_metric_grasp_restore "
                f"view={view.get('name')} ok=False phase=metric_projection "
                f"detail={projected_width.get('reason')}; aperture="
                f"{max_aperture:.1f}mm"
            )
            continue
        fresh_geometry = _geometry_with_projected_3d_width(
            fresh_geometry, projected_width
        )
        metric_body = fresh_geometry.grasp_detection
        if metric_body is not None:
            fresh_geometry = replace(
                fresh_geometry,
                grasp_detection=replace(
                    metric_body,
                    info={
                        **metric_body.info,
                        "metric_recovery_pose_validated": True,
                    },
                ),
            )
        fresh_geometry = replace(
            fresh_geometry,
            rpy_rad=tuple(float(value) for value in target_rpy),
            yaw_correction_deg=math.degrees(float(
                target_rpy[2] - np.asarray(view["ee_rpy_rad"], dtype=float)[2]
            )),
            reason=(
                f"{fresh_geometry.reason}; restored fused-metric grasp "
                "orientation retained after full OBB jaw projection"
            ),
        )
        log(
            "stage=multiview_metric_grasp_restore "
            f"view={view.get('name')} ok=True width_mm={width_mm:.1f} "
            f"major_mm={major_mm:.1f} projected_width_mm="
            f"{float(projected_width['observed_width_mm']):.1f} "
            f"detail={fresh_geometry.reason}"
        )
        return fresh_det, fresh_frame, fresh_geometry, {
            "ok": True,
            "pose": view.get("name"),
            "reason": dimension_detail.get("reason"),
            "discovery_cloud": cloud,
        }
    return None, None, None, {
        "ok": False,
        "reason": (
            "multiview metric box was valid, but no captured complete view "
            "could be safely restored with an aperture-valid grasp"
        ),
        "discovery_cloud": cloud,
    }


def _install_scene_collision_map(
    frame: tuple[np.ndarray, np.ndarray, Any],
    geometry: GG.GraspGeometry,
    camera_mapping: np.ndarray,
    *,
    payload_radius_m: float = 0.0,
    extra_frames: list[tuple[np.ndarray, np.ndarray, Any]] | None = None,
    extra_pose_views: list[dict[str, Any]] | None = None,
    object_boxes_3d: dict[str, dict[str, Any]] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Map visible non-target depth and install it in the central motion gate."""
    global _ACTIVE_SCENE_MAP
    _ACTIVE_SCENE_MAP = None
    if not SCENE_COLLISION_ENABLED:
        return True, {
            "collision_model": False,
            "reason": "disabled; set REBOT_SCENE_COLLISION=1 for guarded runs",
        }
    # ``scene_ee_xyz`` is the URDF tool/TCP position, so every camera point
    # must be lifted with the TCP-in-camera anchor. The calibrated open-jaw
    # slot intentionally remains separate and is used by grasp servoing only.
    tool_camera = load_tcp()
    safety = load_safety_calibration()
    table_z = safety.get("table_z_m")
    if tool_camera is None:
        return False, {"reason": "TCP camera anchor is unavailable"}
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
        color, depth, intr = frame
        scene_ee_xyz = np.asarray(A.ee_pos(), dtype=float)
        target_capsules = []
        target_obbs = []
        target_pixel_masks = []
        target_mask_detail = None
        target_obb_detail = None
        pick_target_box = (object_boxes_3d or {}).get("pick_target")
        if pick_target_box is not None:
            try:
                obb_center = np.asarray(
                    pick_target_box["center_m"], dtype=float
                )
                obb_axes = np.asarray(
                    pick_target_box["axes_base"], dtype=float
                )
                observed_size = np.asarray(
                    pick_target_box["observed_size_m"], dtype=float
                )
                if obb_center.shape != (3,) or obb_axes.shape != (3, 3) \
                        or observed_size.shape != (3,) \
                        or np.any(observed_size <= 0.0) \
                        or float(np.max(observed_size)) > 0.25:
                    raise ValueError("measured target OBB is invalid or implausible")
                # Cover RealSense boundary noise, but do not use the inflated
                # carried-payload volume to erase neighboring obstacles.
                # A voxel touching the measured target is not separable scene
                # evidence: its centre may sit just outside the OBB while its
                # volume still contains target depth. Derive the ownership halo
                # from map resolution rather than from an object-specific size.
                default_obb_pad_m = (
                    SCENE_COLLISION_VOXEL_M / 2.0
                    + SCENE_COLLISION_INFLATION_M
                )
                obb_pad_m = float(np.clip(float(os.getenv(
                    "REBOT_SCENE_TARGET_OBB_PAD_M",
                    str(default_obb_pad_m),
                )), 0.003, 0.020))
                obb_half_size = observed_size / 2.0 + obb_pad_m
                target_obbs.append((obb_center, obb_axes, obb_half_size))
                target_obb_detail = {
                    "center_m": obb_center.round(4).tolist(),
                    "observed_size_mm": (observed_size * 1000.0).round(1).tolist(),
                    "pad_mm": round(obb_pad_m * 1000.0, 1),
                    "source": "stationary_multiframe_metric_pick_target",
                }
            except (KeyError, TypeError, ValueError) as exc:
                return False, {
                    "reason": f"measured target OBB exclusion is invalid: {exc}"
                }
        target_detection = geometry.envelope_detection or geometry.grasp_detection
        if target_detection is not None:
            native_mask = getattr(target_detection, "info", {}).get("mask_u8")
            # A verified stationary metric OBB is the authoritative target
            # ownership test. Do not also apply a stochastic semantic mask:
            # close masks can spill into a neighbouring object and erase it
            # from the collision map. The OBB test happens after deprojection,
            # so it remains tied to measured 3D target volume in every frame.
            if native_mask is not None and not target_obbs:
                native_mask = np.asarray(native_mask, dtype=np.uint8)
                if native_mask.shape != np.asarray(depth).shape:
                    return False, {
                        "reason": "semantic target mask does not match the "
                                  "collision depth frame"
                    }
                dilation_px = int(np.clip(int(os.getenv(
                    "REBOT_SCENE_TARGET_MASK_DILATE_PX", "3"
                )), 0, 6))
                if dilation_px:
                    kernel_size = 2 * dilation_px + 1
                    native_mask = cv2.dilate(
                        native_mask,
                        np.ones((kernel_size, kernel_size), dtype=np.uint8),
                        iterations=1,
                    )
                target_pixel_masks.append(native_mask)
                target_mask_detail = {
                    "pixels": int(np.count_nonzero(native_mask)),
                    "dilation_px": dilation_px,
                    "source": target_detection.info.get(
                        "segmentation_source", target_detection.source
                    ),
                }
            target_camera = np.asarray(target_detection.p_cam_mm, dtype=float)
            target_center_base = scene_ee_xyz + np.linalg.solve(
                np.asarray(camera_mapping, dtype=float),
                np.asarray(tool_camera, dtype=float) - target_camera,
            )
            long_angle = math.radians(float(
                target_detection.info.get("long_axis_deg", 0.0) or 0.0
            ))
            camera_axis = np.array([math.cos(long_angle), math.sin(long_angle), 0.0])
            target_axis_base = np.linalg.solve(
                np.asarray(camera_mapping, dtype=float), -1000.0 * camera_axis
            )
            target_axis_base /= max(1e-9, float(np.linalg.norm(target_axis_base)))
            target_capsules.append((
                target_center_base,
                target_axis_base,
                float(target_detection.max_width_mm or
                      target_detection.min_width_mm or 20.0) / 2000.0
                + SCENE_COLLISION_VOXEL_M,
                float(target_detection.min_width_mm or 20.0) / 2000.0
                + SCENE_COLLISION_INFLATION_M + SCENE_COLLISION_VOXEL_M,
            ))
        # Once a stationary multi-frame metric OBB exists, use it as the sole
        # 3D ownership test. The older padded 2D rectangle/capsule can erase a
        # real neighboring object merely because it overlaps in the image.
        map_pixel_exclusions = [] if target_obbs else exclusions
        map_capsules = [] if target_obbs else target_capsules
        scene = SC.build_scene_map(
            depth, intr,
            ee_xyz_m=scene_ee_xyz,
            camera_mapping_mm_per_m=np.asarray(camera_mapping, dtype=float),
            jaw_target_cam_mm=tool_camera,
            table_z_m=float(table_z),
            workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
            workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
            exclude_pixel_boxes=map_pixel_exclusions,
            exclude_pixel_masks=target_pixel_masks,
            exclude_base_capsules=map_capsules,
            exclude_base_obbs=target_obbs,
            near_mm=max(100.0, float(WD.NEAR_MM)),
            far_mm=float(WD.FAR_MM),
            stride_px=SCENE_COLLISION_STRIDE_PX,
            voxel_m=SCENE_COLLISION_VOXEL_M,
            inflation_m=SCENE_COLLISION_INFLATION_M,
            target_pad_px=SCENE_COLLISION_TARGET_PAD_PX,
        )
        maps = [scene]
        for _extra_color, extra_depth, extra_intr in (extra_frames or []):
            maps.append(SC.build_scene_map(
                extra_depth, extra_intr,
                ee_xyz_m=scene_ee_xyz,
                camera_mapping_mm_per_m=np.asarray(camera_mapping, dtype=float),
                jaw_target_cam_mm=tool_camera,
                table_z_m=float(table_z),
                workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
                workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
                exclude_pixel_boxes=map_pixel_exclusions,
                exclude_pixel_masks=target_pixel_masks,
                exclude_base_capsules=map_capsules,
                exclude_base_obbs=target_obbs,
                near_mm=max(100.0, float(WD.NEAR_MM)),
                far_mm=float(WD.FAR_MM),
                stride_px=SCENE_COLLISION_STRIDE_PX,
                voxel_m=SCENE_COLLISION_VOXEL_M,
                inflation_m=SCENE_COLLISION_INFLATION_M,
                target_pad_px=SCENE_COLLISION_TARGET_PAD_PX,
            ))
        # Discovery frames captured after bounded wrist rotations/translations
        # are retained for the diagnostic cloud below with their own pose and
        # mapping. Do not union them into the safety occupancy map yet: those
        # frames can contain the arm at a historical configuration, and the
        # current-pose self filter must not turn that old arm image into a
        # persistent "ghost obstacle". The collision map therefore continues
        # to use its stationary burst only.
        accepted_pose_views = []
        for pose_view in (extra_pose_views or []):
            pose_frame = pose_view.get("frame")
            pose_mapping = np.asarray(
                pose_view.get("camera_mapping"), dtype=float
            )
            pose_ee = np.asarray(pose_view.get("ee_xyz_m"), dtype=float)
            if pose_frame is None or pose_mapping.shape != (3, 3) \
                    or pose_ee.shape != (3,):
                continue
            accepted_pose_views.append(pose_view)
        temporal_required_frames = 1
        if len(maps) > 1:
            # Stationary RGB-D fusion is an occupancy consensus, not a union.
            # A union turns one-frame flying pixels into permanent full-height
            # obstacles. Requiring a strict majority preserves real objects
            # while rejecting transient RealSense edges and holes.
            temporal_required_frames = max(2, len(maps) // 2 + 1)
            scene = SC.fuse_scene_maps(
                maps, ee_xyz_m=scene_ee_xyz,
                minimum_views_per_column=temporal_required_frames,
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
    raw_boxes = [box.as_dict() for box in scene.boxes]
    self_filter = A.filter_scene_boxes_at_current_pose(raw_boxes)
    if not self_filter.get("ok"):
        return False, {**scene.as_dict(), "reason": self_filter.get("reason")}
    filtered_boxes = list(self_filter.get("kept") or [])
    removed_indices = {
        int(item["index"]) for item in self_filter.get("removed", [])
    }
    filtered_scene = replace(
        scene,
        boxes=tuple(box for index, box in enumerate(scene.boxes)
                    if index not in removed_indices),
    )
    # Consolidate only after current-pose self filtering. If columns were
    # merged first, one small robot overlap could cause the self filter to drop
    # a larger box containing otherwise valid environmental occupancy.
    filtered_scene = SC.consolidate_scene_map(filtered_scene)
    configured_boxes = [box.as_dict() for box in filtered_scene.boxes]
    configured = A.configure_collision_scene(
        configured_boxes,
        table_z_m=float(table_z), payload_radius_m=payload_radius_m,
        voxel_m=float(filtered_scene.voxel_m),
        inflation_m=float(filtered_scene.inflation_m),
    )
    current_scene_gate = {
        "ok": False,
        "reason": "collision scene configuration failed",
    }
    if configured.ok:
        current_scene_gate = A.validate_current_collision_scene()
        if not current_scene_gate.get("ok"):
            A.clear_collision_scene()
            return False, {
                **filtered_scene.as_dict(),
                "reason": current_scene_gate.get("reason"),
                "current_pose_gate": current_scene_gate,
                "temporal_required_frames": temporal_required_frames,
            }
        # Retain the exact target-excluded, self-filtered map for post-grasp
        # top-down navigation. This is process-local and is never accepted as a
        # motion certificate; every proposed route still passes the central
        # full-arm IK/collision validator immediately before execution.
        _ACTIVE_SCENE_MAP = filtered_scene
    artifacts: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled; coarse collision map remains active",
    }
    if SCENE_COLLISION_EXPORT_VIEW:
        try:
            view_clouds = []
            stationary_frames = [frame, *(extra_frames or [])]
            stationary_depth = SC.median_depth_image(
                [stationary_frame[1] for stationary_frame in stationary_frames]
            )
            stationary_view = (
                frame[0], stationary_depth, frame[2]
            )
            diagnostic_views = [
                ("map_stationary_median", stationary_view, scene_ee_xyz,
                 np.asarray(camera_mapping, dtype=float)),
                *[
                    (f"discovery_{view.get('name', 'view')}", view["frame"],
                     np.asarray(view["ee_xyz_m"], dtype=float),
                     np.asarray(view["camera_mapping"], dtype=float))
                    for view in accepted_pose_views
                ],
            ]
            named_clouds = {}
            diagnostic_view_names = []
            for view_name, (view_color, view_depth, view_intr), view_ee, view_mapping \
                    in diagnostic_views:
                cloud = SC.build_dense_view_cloud(
                    view_color, view_depth, view_intr,
                    ee_xyz_m=view_ee,
                    camera_mapping_mm_per_m=view_mapping,
                    jaw_target_cam_mm=tool_camera,
                    workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
                    workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
                    # Keep raw base coordinates. Independent per-view table
                    # snapping manufactured the doubled ~53 mm scene layer.
                    table_z_m=None,
                    near_mm=float(WD.NEAR_MM), far_mm=float(WD.FAR_MM),
                    stride_px=1, voxel_m=0.0015, bounds_margin_m=0.08,
                    minimum_neighbor_support=2,
                    crop_to_workspace=True, crop_below_table=False,
                )
                view_clouds.append(cloud)
                named_clouds[view_name] = cloud
                diagnostic_view_names.append(view_name)
            view_clouds = SC.crop_dense_view_clouds(
                view_clouds,
                table_z_m=float(table_z),
                workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
                workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
                bounds_margin_m=0.08,
                crop_below_table=True,
            )
            registered_clouds, registration = SC.register_dense_view_clouds(
                view_clouds, table_z_m=float(table_z),
                max_translation_m=SCENE_DENSE_REGISTRATION_MAX_TRANSLATION_M,
                minimum_overlap=SCENE_DENSE_REGISTRATION_MIN_OVERLAP,
            )
            named_clouds = {
                name: cloud
                for name, cloud in zip(diagnostic_view_names, registered_clouds)
            }
            fusable_clouds = [
                cloud for cloud, report in zip(registered_clouds, registration)
                if bool(report.get("valid_for_fusion", False))
            ]
            rejected_view_names = [
                name for name, report in zip(
                    diagnostic_view_names, registration
                ) if not bool(report.get("valid_for_fusion", False))
            ]
            if log is not None:
                for name, report in zip(diagnostic_view_names, registration):
                    log(
                        "stage=scene_cloud_registration "
                        f"view={name!r} "
                        f"fused={bool(report.get('valid_for_fusion', False))} "
                        f"corrected={bool(report.get('accepted', False))} "
                        f"translation_mm={report.get('translation_mm')} "
                        f"overlap={report.get('zero_overlap')}->"
                        f"{report.get('best_overlap')} "
                        f"detail={report.get('reason')}"
                    )
            view_points, view_colors = SC.fuse_dense_view_clouds(
                fusable_clouds, voxel_m=0.0015
            )
            artifacts = SC.save_scene_artifacts(
                filtered_scene, DEBUG_DIR / "scene_collision", stem="latest_scene",
                view_points_m=view_points, view_colors_rgb=view_colors,
                named_view_clouds=(
                    named_clouds if SCENE_COLLISION_EXPORT_PER_VIEW else {}
                ),
                object_boxes_3d=object_boxes_3d,
                show_box_labels=SCENE_COLLISION_EXPORT_BOX_LABELS,
            )
            artifacts["enabled"] = True
            artifacts["dense_rgb_points"] = int(len(view_points))
            artifacts["dense_rgb_frames"] = int(len(fusable_clouds))
            artifacts["dense_rgb_frames_seen"] = int(len(diagnostic_views))
            artifacts["dense_rgb_frames_rejected"] = rejected_view_names
            artifacts["dense_stationary_frames_medianed"] = int(
                len(stationary_frames)
            )
            artifacts["dense_table_snap_enabled"] = False
            artifacts["moving_discovery_views"] = int(len(accepted_pose_views))
            artifacts["diagnostic_view_names"] = diagnostic_view_names
            artifacts["per_view_clouds_exported"] = \
                SCENE_COLLISION_EXPORT_PER_VIEW
            artifacts["registration"] = registration
        except Exception as exc:  # noqa: BLE001
            # Visualization is diagnostic-only. It must never weaken or abort
            # an otherwise valid fail-closed collision map.
            artifacts = {
                "enabled": False,
                "reason": f"optional export failed: {type(exc).__name__}: {exc}",
            }
    detail = {
        **filtered_scene.as_dict(),
        "fused_frames": len(maps),
        "temporal_required_frames": temporal_required_frames,
        "temporal_fusion": (
            "strict_majority_stationary_occupancy"
            if len(maps) > 1 else "single_stationary_frame"
        ),
        "current_pose_gate": current_scene_gate,
        "moving_discovery_views": len(accepted_pose_views),
        "moving_views_used_for_collision": False,
        "reason": configured.reason,
        "target_exclusion_boxes_px": [list(box) for box in map_pixel_exclusions],
        "target_exclusion_pad_px": int(SCENE_COLLISION_TARGET_PAD_PX),
        "target_exclusion_3d": bool(map_capsules or target_obbs),
        "target_exclusion_semantic_mask": target_mask_detail,
        "target_exclusion_obb": target_obb_detail,
        "target_exclusion_method": (
            "measured_3d_obb_and_semantic_mask"
            if target_obbs and target_pixel_masks else
            "measured_3d_obb" if target_obbs else
            "semantic_mask_and_legacy_capsule" if target_pixel_masks else
            "legacy_2d_box_and_capsule"
        ),
        "payload_radius_mm": round(payload_radius_m * 1000.0, 1),
        "raw_obstacle_columns": len(raw_boxes),
        "self_filtered_obstacle_columns": len(filtered_boxes),
        "obstacle_columns": len(configured_boxes),
        "robot_self_filtered_columns": self_filter.get("removed", []),
        "visualization": artifacts,
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

    The camera detects/sizes from the ready pose, then refreshes physical-body
    geometry at a visible pre-grasp checkpoint. Every trajectory between visual
    observations remains fixed and completely IK/collision-prevalidated; camera
    observations never steer a segment while it is executing.
    """
    global _OBJECT_HELD, _FIXED_HELD_CONTEXT, _ACTIVE_SCENE_MAP
    _OBJECT_HELD = False
    _FIXED_HELD_CONTEXT = None
    if SCENE_COLLISION_ENABLED:
        # A previous interrupted controller process must never leak stale scene
        # geometry into a new observation.
        A.clear_collision_scene()
        _ACTIVE_SCENE_MAP = None
    log = _printer(verbose)
    steps: list[tuple[str, bool, str]] = []
    preplanned_scene_endpoint: np.ndarray | None = None
    preplanned_scene_detail: dict[str, Any] | None = None
    collision_detail: dict[str, Any] = {
        "collision_model": False,
        "reason": "scene collision checking disabled",
    }
    aligned_3d_width_detail: dict[str, Any] | None = None
    hybrid_identity_gate: _HybridTargetIdentity | None = None
    object_geometry_mode = _object_geometry_mode()
    if object_geometry_mode is None:
        return _result(
            False,
            "REBOT_OBJECT_GEOMETRY_MODE must be 'legacy' or 'hybrid'",
            [("object_geometry_mode", False, "unsupported mode")],
            homed=False, object_held=False,
        )
    hybrid_object_box_3d: dict[str, Any] | None = None
    hybrid_metric_width_mm: float | None = None
    hybrid_metric_major_mm: float | None = None
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
        f"object_geometry_mode={object_geometry_mode!r} "
        f"place_relation={normalized_place_relation!r} "
        f"place_reference={place_reference!r}")
    steps.append((
        "object_geometry_mode", True,
        ("legacy stationary RGB-D geometry"
         if object_geometry_mode == "legacy" else
         "hybrid pose-aware multi-view object-mask fusion"),
    ))

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

    own_eye = eye is None
    eye = eye or WristEye()
    hybrid_identity_gate: _HybridTargetIdentity | None = None
    camera_kind = str(getattr(eye, "camera_kind", CAMERA_KIND))
    log(f"stage=camera_preflight status=checking camera={camera_kind!r}")
    try:
        preflight_camera = getattr(eye, "preflight", None)
        color0, depth0, _intr0 = (
            preflight_camera() if callable(preflight_camera) else eye.frame()
        )
        camera_detail = (
            f"{camera_kind} RGB-D ready "
            f"({color0.shape[1]}x{color0.shape[0]}, "
            f"valid_depth={int(np.count_nonzero(depth0 > 0))})"
        )
        steps.append(("camera_preflight", True, camera_detail))
        log(f"stage=camera_preflight status=ready detail={camera_detail}")
    except Exception as exc:  # noqa: BLE001 - hardware absence is a normal refusal
        camera_detail = f"{type(exc).__name__}: {exc}"
        steps.append(("camera_preflight", False, camera_detail))
        log(f"stage=camera_preflight status=failed detail={camera_detail}")
        if own_eye:
            eye.close()
        return _result(
            False,
            "wrist camera preflight failed; no arm or gripper motion was started: "
            f"{camera_detail}",
            steps, homed=False, object_held=False,
        )

    pf = A.preflight(connect=not dry)
    steps.append(("preflight", pf.ok, pf.reason))
    if not pf.ok:
        if own_eye:
            eye.close()
        return _result(False, "arm not ready", steps, homed=False,
                       object_held=False)

    if object_geometry_mode == "hybrid" and not dry:
        set_identity_filter = getattr(eye, "set_identity_filter", None)
        if not callable(set_identity_filter):
            if own_eye:
                eye.close()
            return _result(
                False,
                "hybrid target identity requires a filter-capable wrist camera",
                steps + [("hybrid_target_identity", False,
                          "camera object does not support identity filtering")],
                homed=False, object_held=False,
            )
        hybrid_identity_gate = _HybridTargetIdentity(
            description=object_name,
            tool_camera_mm=np.asarray(tcp_cam, dtype=float),
            log=log,
        )
        set_identity_filter(object_name, hybrid_identity_gate)
        steps.append((
            "hybrid_target_identity", True,
            "SAM3 establishes the preferred target anchor; after a semantic "
            "miss, three stable table-plane RGB-D object observations containing "
            "the requested colour may establish a fallback anchor; raw colour "
            "blobs cannot anchor; all later observations remain cluster-gated",
        ))
        log(
            "stage=hybrid_target_identity status=armed "
            f"semantic_anchor_attempts={hybrid_identity_gate.anchor_attempts} "
            f"cluster_radius_mm={hybrid_identity_gate.radius_m * 1000.0:.1f}"
        )

    start_feed = getattr(eye, "start_live_feed", None)
    continuous_feed = bool(not dry and callable(start_feed) and start_feed())
    if continuous_feed:
        feed_policy = (
            "camera frames stream throughout motion; the macro arm trajectory "
            "is frozen from the initial scene up to a visible checkpoint; "
            "stationary RGB-D may replace only the short final leg after full "
            "IK/collision validation, then controls jaw-slot alignment and closure"
            if FIXED_SCENE_TRAJECTORY else
            "camera frames stream throughout motion; detections can only replan "
            "at stationary visible checkpoints, never steer an active segment"
        )
        steps.append((
            "continuous_live_feed", True, feed_policy,
        ))
        log("stage=continuous_live_feed started=true "
            f"camera_steering="
            f"{'frozen_macro_then_stationary_checkpoint' if FIXED_SCENE_TRAJECTORY else 'stationary_checkpoints_only'}")

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
            # Semantic detectors occasionally miss one otherwise valid frame.
            # Dry mode cannot move to discover another viewpoint, but it can
            # safely retry fresh frames from the same stationary camera. Stop
            # at the first frame whose detection *and* grasp geometry are both
            # usable instead of letting one stochastic miss reject the scene.
            det = None
            frame = None
            geometry = None
            why: Any = {}
            dry_attempt_reasons: list[str] = []
            dry_rgbd_anchor_points: list[np.ndarray] = []
            dry_attempt_limit = (
                int(np.clip(int(os.getenv(
                    "REBOT_HYBRID_SEMANTIC_ANCHOR_ATTEMPTS", "8"
                )), 3, 15))
                if object_geometry_mode == "hybrid" else 3
            )
            for attempt in range(1, dry_attempt_limit + 1):
                candidate_det, candidate_frame, candidate_why = _observe(
                    eye, object_name, retries=1
                )
                if candidate_det is None or candidate_frame is None:
                    why = candidate_why
                    detail = str(candidate_why)
                    dry_attempt_reasons.append(detail)
                    log(
                        "stage=dry_target_sample "
                        f"frame={attempt}/{dry_attempt_limit} "
                        f"ok=False detail={detail}"
                    )
                    continue
                candidate_partial = _detection_partial(
                    candidate_det, candidate_frame
                )
                if candidate_partial is not None:
                    why = {
                        **candidate_why,
                        "complete_view_gate": candidate_partial,
                    }
                    dry_attempt_reasons.append(candidate_partial)
                    log(
                        "stage=dry_target_sample "
                        f"frame={attempt}/{dry_attempt_limit} "
                        f"ok=False detail={candidate_partial}"
                    )
                    continue
                candidate_source = str(candidate_det.source)
                if object_geometry_mode == "hybrid" \
                        and candidate_source == "rgbd:colored_object":
                    dry_rgbd_anchor_points.append(np.asarray(
                        candidate_det.p_cam_mm, dtype=float
                    ))
                    recent = dry_rgbd_anchor_points[-3:]
                    stable_rgbd_anchor = False
                    spread_mm = float("inf")
                    if len(recent) >= 3:
                        points = np.vstack(recent)
                        median = np.median(points, axis=0)
                        spread_mm = float(np.max(np.linalg.norm(
                            points - median, axis=1
                        )))
                        stable_rgbd_anchor = spread_mm <= 15.0
                        if not stable_rgbd_anchor:
                            dry_rgbd_anchor_points = [recent[-1]]
                    if not stable_rgbd_anchor:
                        detail = (
                            "waiting for stable table-plane RGB-D colored-object "
                            f"anchor {len(dry_rgbd_anchor_points)}/3"
                        )
                        dry_attempt_reasons.append(detail)
                        log(
                            "stage=dry_target_sample "
                            f"frame={attempt}/{dry_attempt_limit} "
                            f"ok=False detail={detail}"
                        )
                        continue
                    log(
                        "stage=dry_hybrid_identity_anchor "
                        "source=rgbd:colored_object stable_frames=3 "
                        f"spread_mm={spread_mm:.1f}"
                    )
                elif object_geometry_mode == "hybrid" and candidate_source \
                        not in _SEMANTIC_IDENTITY_SOURCES:
                    detail = (
                        f"source={candidate_det.source} cannot establish the "
                        "hybrid semantic or stable RGB-D object anchor"
                    )
                    why = {**candidate_why, "identity_gate": detail}
                    dry_attempt_reasons.append(detail)
                    log(
                        "stage=dry_target_sample "
                        f"frame={attempt}/{dry_attempt_limit} "
                        f"ok=False detail={detail}"
                    )
                    continue
                candidate_geometry = _plan_scene_grasp_geometry(
                    candidate_frame, candidate_det,
                    # Dry mode must remain camera/model-only. The live workflow
                    # reads measured RPY after connecting; the dry preview uses
                    # the verified home orientation and never touches hardware.
                    current_rpy=(0.0, 0.0, 0.0),
                    object_name=object_name,
                )
                det, frame, geometry = (
                    candidate_det, candidate_frame, candidate_geometry
                )
                log(
                    "stage=dry_target_sample "
                    f"frame={attempt}/{dry_attempt_limit} "
                    f"ok={candidate_geometry.ok} "
                    f"source={candidate_det.source} "
                    f"depth_mm={candidate_det.z_mm:.0f} "
                    f"detail={candidate_geometry.reason}"
                )
                if candidate_geometry.ok:
                    break
                dry_attempt_reasons.append(candidate_geometry.reason)
            if frame is not None:
                color, depth, intr = frame
                _show(color, depth, intr, tag="fixed_pick_dry_detect",
                      det=det, tcp_cam=tcp_cam,
                      note="stationary dry detection and sizing")
            grasp_det = (
                geometry.grasp_detection
                if geometry is not None and geometry.ok else None
            )
            plan = _plan_aperture(grasp_det) if grasp_det is not None else None
            steps.append(("detect_and_size", det is not None,
                          f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
            steps.append((
                "dry_stationary_target_retry",
                geometry is not None and geometry.ok,
                ("a stationary frame produced usable target geometry"
                 if geometry is not None and geometry.ok else
                 f"{dry_attempt_limit} stationary attempts failed: "
                 f"{dry_attempt_reasons}"),
            ))
            if geometry is not None:
                steps.append(("geometry_grasp_candidates", geometry.ok,
                              geometry.reason))
            if object_geometry_mode == "hybrid":
                steps.append((
                    "hybrid_multiview_object_geometry_preview", True,
                    "deferred to live mode: registered multi-view fusion needs "
                    "measured arm poses and bounded camera motion; dry mode did "
                    "not move or connect to the arm",
                ))
            if plan is not None:
                steps.append(("jaw_plan", bool(plan["ok"]), str(plan["reason"])))
            dry_fit_ok = False
            dry_fit_detail: dict[str, Any] | None = None
            if geometry is not None and geometry.ok and plan is not None \
                    and plan["ok"]:
                dry_fit_ok, dry_fit_detail = _validate_object_gripper_fit(
                    geometry, plan
                )
                steps.append((
                    "object_gripper_fit", dry_fit_ok,
                    str(dry_fit_detail.get("reason")),
                ))
            drop_ik = None
            release_z = None
            reference_det = None
            reference_why: Any = None
            if scene_place:
                assert place_reference is not None
                reference_det, reference_frame, reference_why = _observe(
                    eye, place_reference, retries=3
                )
                log(
                    "stage=dry_reference_sample "
                    f"ok={reference_det is not None} "
                    f"source={getattr(reference_det, 'source', None)} "
                    f"detail={reference_why}"
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
                      and dry_fit_ok
                      and (not scene_place or reference_det is not None)
                      and (drop_ik is None or drop_ik.ok))
            return _result(
                ok,
                f"dry {task_name.replace('_', ' ')} checks passed" if ok else
                f"dry {task_name.replace('_', ' ')} checks failed",
                steps, homed=False, object_held=False,
                jaw_plan=plan if plan and plan["ok"] else None,
                object_gripper_fit=dry_fit_detail,
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

        # Keep metric discovery views for later scene fusion, and require a
        # candidate view to provide an actually enterable local cross-section.
        # Long objects may continue through the fingers; this gate concerns
        # only the material that must pass between the open jaws.
        scene_discovery_views: list[dict[str, Any]] = []
        grasp_fit_diagnostics: list[dict[str, Any]] = []

        def grasp_view_entry_ok(
            candidate: GG.GraspGeometry,
        ) -> tuple[bool, str]:
            candidate_det = candidate.grasp_detection
            if candidate_det is None:
                return False, "local grasp cross-section is unavailable"
            candidate_jaw = _plan_aperture(candidate_det)
            if not candidate_jaw.get("ok"):
                return False, str(candidate_jaw.get("reason"))
            return True, (
                f"visible width {candidate_jaw['min_width_mm']:.1f}mm is within "
                "the hard jaw aperture; entry clearance is deferred until a "
                "measured 3D axis is proven aligned with jaw closure"
            )

        def record_grasp_fit_rejection(
            rejected_frame: tuple[np.ndarray, np.ndarray, Any],
            rejected_geometry: GG.GraspGeometry,
            rejected_reason: str,
            view_name: str,
        ) -> None:
            rejected_det = rejected_geometry.grasp_detection
            if rejected_det is None:
                return
            rejected_jaw = _plan_aperture(rejected_det)
            diagnostic = _save_grasp_fit_diagnostic(
                rejected_frame, rejected_geometry, rejected_jaw,
                rejected_reason, name=view_name, log=log,
            )
            if diagnostic.get("ok"):
                grasp_fit_diagnostics.append(diagnostic)

        initial_detection_retries = (
            hybrid_identity_gate.anchor_attempts
            if hybrid_identity_gate is not None \
                and hybrid_identity_gate.anchor_base_m is None
            else LOST_RETRIES
        )
        det, frame, why = _observe(
            eye, object_name, retries=initial_detection_retries
        )
        if frame is not None:
            _show(*frame, tag="fixed_pick_detect", det=det, tcp_cam=tcp_cam,
                  note="fixed pick detection and sizing")
        steps.append(("detect_from_ready_pose", det is not None,
                      f"{det.source} at {det.z_mm:.0f}mm" if det else str(why)))
        if object_geometry_mode == "hybrid" and det is not None \
                and frame is not None:
            initial_view_geometry = _plan_scene_grasp_geometry(
                frame, det,
                current_rpy=tuple(float(v) for v in A.ee_rpy()),
                object_name=object_name,
            )
            _append_pose_aware_discovery_view(
                scene_discovery_views, name="hybrid_ready_pose",
                frame=frame, detection=det,
                geometry=initial_view_geometry, log=log,
            )
        if det is None:
            log(f"stage=look_around_trigger reason=target_not_visible")
            det, frame, geometry, discovery = _look_around_object(
                eye, object_name, log, require_grasp=True,
                geometry_accept=grasp_view_entry_ok,
                geometry_rejected=record_grasp_fit_rejection,
                captured_views=scene_discovery_views,
            )
            steps.append(("look_around_pick_target", bool(discovery.get("ok")),
                          str(discovery.get("pose") or discovery.get("reason"))))
            if det is None or frame is None or geometry is None:
                discovery_cloud = discovery.get("discovery_cloud") or \
                    _export_discovery_cloud(scene_discovery_views, log)
                return recover(
                    f"'{object_name}' was not found completely by the bounded "
                    "look-around sweep", discovery=discovery,
                    discovery_cloud=discovery_cloud,
                    grasp_fit_diagnostics=grasp_fit_diagnostics,
                )
        else:
            partial_reason = _detection_partial(det, frame)
            geometry = None
            if partial_reason is not None:
                log(f"stage=look_around_trigger reason={partial_reason}")
                det, frame, geometry, discovery = _look_around_object(
                    eye, object_name, log, require_grasp=True,
                    geometry_accept=grasp_view_entry_ok,
                    geometry_rejected=record_grasp_fit_rejection,
                    captured_views=scene_discovery_views,
                )
                steps.append(("look_around_pick_target",
                              bool(discovery.get("ok")),
                              str(discovery.get("pose") or discovery.get("reason"))))
                if det is None or frame is None or geometry is None:
                    discovery_cloud = discovery.get("discovery_cloud") or \
                        _export_discovery_cloud(scene_discovery_views, log)
                    return recover(
                        "initial target view was partial and no complete bounded "
                        "look-around view was found", discovery=discovery,
                        discovery_cloud=discovery_cloud,
                        grasp_fit_diagnostics=grasp_fit_diagnostics,
                    )
            if geometry is None:
                geometry = _plan_scene_grasp_geometry(
                    frame, det,
                    current_rpy=tuple(float(v) for v in A.ee_rpy()),
                    object_name=object_name,
                )
        steps.append(("geometry_grasp_candidates", geometry.ok,
                      geometry.reason))
        log(f"stage=geometry_grasp ok={geometry.ok} detail={geometry.reason}")
        geometry_attempt_reasons = [geometry.reason]
        if not scene_discovery_views and frame is not None:
            _append_pose_aware_discovery_view(
                scene_discovery_views, name="ready_pose", frame=frame,
                detection=det, geometry=geometry, log=log,
            )
        # Semantic masks can occasionally collapse to a coloured cap or a short
        # depth fragment for one frame. Retry in place before treating a large
        # apparent yaw as physical truth. A stable but out-of-budget yaw is not
        # transient, so repeated identical frames are useless; sweep viewpoints
        # instead and retain them for the diagnostic map.
        yaw_budget_failure = (
            not geometry.ok and "wrist yaw" in geometry.reason
            and "budget" in geometry.reason
        )
        if not geometry.ok and not yaw_budget_failure:
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
            log(
                "stage=geometry_view_sweep_trigger "
                f"reason={geometry.reason} action=discover_and_export"
            )
            sweep_det, sweep_frame, sweep_geometry, sweep = \
                _look_around_object(
                    eye, object_name, log, require_grasp=True,
                    geometry_accept=grasp_view_entry_ok,
                    geometry_rejected=record_grasp_fit_rejection,
                    captured_views=scene_discovery_views,
                )
            steps.append((
                "geometry_view_sweep", bool(sweep.get("ok")),
                str(sweep.get("pose") or sweep.get("reason")),
            ))
            if sweep_det is None or sweep_frame is None \
                    or sweep_geometry is None:
                discovery_cloud = sweep.get("discovery_cloud") or \
                    _export_discovery_cloud(scene_discovery_views, log)
                return recover(
                    "no bounded viewpoint produced a safe grasp; the scene "
                    "sweep was exported without weakening orientation limits: "
                    f"{' | '.join(geometry_attempt_reasons)}",
                    discovery=sweep,
                    discovery_cloud=discovery_cloud,
                    grasp_fit_diagnostics=grasp_fit_diagnostics,
                    grasp_geometry=geometry.summary(),
                    geometry_attempts=geometry_attempt_reasons,
                )
            det, frame, geometry = sweep_det, sweep_frame, sweep_geometry
            log(
                "stage=geometry_view_sweep status=accepted "
                f"pose={sweep.get('pose')} detail={geometry.reason}"
            )
        initial_entry_ok, initial_entry_reason = grasp_view_entry_ok(geometry)
        if not initial_entry_ok:
            record_grasp_fit_rejection(
                frame, geometry, initial_entry_reason, "selected_ready_view"
            )
            log(
                "stage=alternate_grasp_view_trigger "
                f"reason={initial_entry_reason}"
            )
            alternate_det, alternate_frame, alternate_geometry, alternate = \
                _look_around_object(
                    eye, object_name, log, require_grasp=True,
                    geometry_accept=grasp_view_entry_ok,
                    geometry_rejected=record_grasp_fit_rejection,
                    captured_views=scene_discovery_views,
                )
            steps.append((
                "alternate_grasp_view",
                bool(alternate.get("ok")),
                str(alternate.get("pose") or alternate.get("reason")),
            ))
            if alternate_det is None or alternate_frame is None \
                    or alternate_geometry is None:
                discovery_cloud = alternate.get("discovery_cloud") or \
                    _export_discovery_cloud(scene_discovery_views, log)
                return recover(
                    "no bounded view exposed a safely enterable local "
                    f"cross-section: {initial_entry_reason}",
                    discovery=alternate,
                    discovery_cloud=discovery_cloud,
                    grasp_fit_diagnostics=grasp_fit_diagnostics,
                    grasp_geometry=geometry.summary(),
                )
            det, frame, geometry = (
                alternate_det, alternate_frame, alternate_geometry
            )
            log(
                "stage=alternate_grasp_view status=accepted "
                f"pose={alternate.get('pose')} detail={geometry.reason}"
            )

        if object_geometry_mode == "hybrid":
            def hybrid_semantic_view_entry_ok(
                candidate_geometry: GG.GraspGeometry,
            ) -> tuple[bool, str]:
                semantic_ok, semantic_reason = \
                    _exact_semantic_geometry_mask(candidate_geometry)
                if not semantic_ok:
                    return False, (
                        "hybrid object shape requires a native semantic target "
                        f"mask; {semantic_reason}"
                    )
                return grasp_view_entry_ok(candidate_geometry)

            # Preserve the selected view, then measure how many *accepted target
            # masks* survived registration. Camera-view count is insufficient:
            # a partial SAM mask remains useful for the scene cloud but must not
            # satisfy the two-view object-geometry requirement.
            if not any(view.get("frame") is frame
                       for view in scene_discovery_views):
                _append_pose_aware_discovery_view(
                    scene_discovery_views, name="hybrid_selected_view",
                    frame=frame, detection=det, geometry=geometry, log=log,
                )
            maximum_extra_sweeps = int(np.clip(int(os.getenv(
                "REBOT_HYBRID_EXTRA_VIEW_SWEEPS", "2"
            )), 1, 3))
            hybrid_cloud: dict[str, Any] = {}
            hybrid_dimension_detail: dict[str, Any] = {
                "reason": "multiview metric object box is missing"
            }
            hybrid_geometry_ok = False
            last_fusion_discovery: dict[str, Any] | None = None
            fused_view_count = 0
            box_view_count = 0
            for collection_attempt in range(maximum_extra_sweeps + 1):
                hybrid_cloud = _export_discovery_cloud(
                    scene_discovery_views, log, force_geometry=True,
                    require_semantic_target_masks=True,
                    target_anchor_base_m=hybrid_identity_gate.anchor_base_m,
                    target_anchor_radius_m=hybrid_identity_gate.radius_m,
                )
                hybrid_object_box_3d = hybrid_cloud.get("object_box_3d")
                fused_view_count = int(hybrid_cloud.get("fused_views", 0))
                box_view_count = int(
                    (hybrid_object_box_3d or {}).get("views", 0)
                )
                hybrid_metric_width_mm, hybrid_metric_major_mm, \
                    hybrid_dimension_detail = _metric_horizontal_box_dimensions(
                        hybrid_object_box_3d
                    )
                hybrid_geometry_ok = bool(
                    hybrid_cloud.get("ok")
                    and fused_view_count >= 2
                    and box_view_count >= 2
                    and hybrid_metric_width_mm is not None
                    and hybrid_metric_major_mm is not None
                )
                log(
                    "stage=hybrid_object_fusion status=checking "
                    f"attempt={collection_attempt + 1}/"
                    f"{maximum_extra_sweeps + 1} "
                    f"camera_views={len(scene_discovery_views)} "
                    f"registered_views={fused_view_count} "
                    f"valid_target_masks={box_view_count}/2"
                )
                if hybrid_geometry_ok:
                    break
                if collection_attempt >= maximum_extra_sweeps:
                    break
                log(
                    "stage=hybrid_object_fusion status=collecting "
                    f"reason=valid_target_masks_{box_view_count}_of_2 "
                    f"extra_sweep={collection_attempt + 1}/"
                    f"{maximum_extra_sweeps}"
                )
                fusion_det, fusion_frame, fusion_geometry, fusion_discovery = \
                    _look_around_object(
                        eye, object_name, log, require_grasp=True,
                        geometry_accept=hybrid_semantic_view_entry_ok,
                        geometry_rejected=record_grasp_fit_rejection,
                        captured_views=scene_discovery_views,
                    )
                last_fusion_discovery = fusion_discovery
                steps.append((
                    f"hybrid_additional_object_view_{collection_attempt + 1}",
                    fusion_det is not None and fusion_frame is not None
                    and fusion_geometry is not None,
                    str(fusion_discovery.get("pose")
                        or fusion_discovery.get("reason")),
                ))
                if fusion_det is None or fusion_frame is None \
                        or fusion_geometry is None:
                    break
                det, frame, geometry = (
                    fusion_det, fusion_frame, fusion_geometry
                )
            steps.append((
                "hybrid_multiview_object_geometry", hybrid_geometry_ok,
                (f"registered_views={fused_view_count}; "
                 f"object_views={box_view_count}; "
                 f"{hybrid_dimension_detail.get('reason')}"),
            ))
            log(
                "stage=hybrid_object_fusion "
                f"ok={hybrid_geometry_ok} registered_views="
                f"{fused_view_count} object_views={box_view_count} "
                f"detail={hybrid_dimension_detail.get('reason')}"
            )
            if not hybrid_geometry_ok:
                return recover(
                    "hybrid object-mask fusion did not produce a consistent "
                    "two-target-mask metric object box after bounded extra "
                    "view collection; legacy mode remains available with "
                    "REBOT_OBJECT_GEOMETRY_MODE=legacy",
                    discovery=last_fusion_discovery,
                    discovery_cloud=hybrid_cloud,
                )
            recovered_metric_pose = bool(
                geometry.grasp_detection is not None
                and geometry.grasp_detection.info.get(
                    "metric_recovery_pose_validated", False
                )
            )
            if recovered_metric_pose:
                current_mapping, mapping_detail = load_saved_camera_jacobian(
                    tuple(float(v) for v in A.ee_rpy())
                )
                from arm_agent import gripper_model as gm
                calibrated_aperture = gm.max_aperture_mm(_wrist_limits())
                if current_mapping is None or calibrated_aperture is None:
                    retained_geometry, retained_detail = None, {
                        "ok": False,
                        "reason": (
                            str(mapping_detail.get("reason"))
                            if current_mapping is None else
                            "jaw millimetre calibration is unavailable"
                        ),
                    }
                else:
                    retained_geometry, retained_detail = \
                        _revalidate_recovered_metric_grasp(
                            geometry, hybrid_object_box_3d,
                            np.asarray(current_mapping, dtype=float),
                            float(calibrated_aperture),
                        )
                if retained_geometry is None:
                    return recover(
                        "refreshed hybrid object box invalidated the recovered "
                        f"metric grasp: {retained_detail.get('reason')}",
                        discovery_cloud=hybrid_cloud,
                    )
                geometry = retained_geometry
                log(
                    "stage=hybrid_object_geometry_applied "
                    "mode=preserve_validated_metric_recovery "
                    f"projected_width_mm="
                    f"{float(retained_detail['observed_width_mm']):.1f}"
                )
            else:
                # A wrist RGB-D cloud observes only the camera-facing object
                # surface. Its thinnest horizontal OBB extent can therefore be
                # merely visible surface relief, not the object's complete jaw
                # thickness. Keep the registered box for identity, length and
                # collision geometry, but size the initial jaws from the
                # depth-backed image silhouette. The later camera-axis 3-D
                # projection may still replace that width after it proves the
                # selected OBB axis is both visible and aligned with closure.
                geometry = _plan_scene_grasp_geometry(
                    frame, det,
                    current_rpy=tuple(float(v) for v in A.ee_rpy()),
                    object_name=object_name,
                )
                if not geometry.ok or geometry.grasp_detection is None \
                        or geometry.rpy_rad is None:
                    return recover(
                        "hybrid fused object box could not produce a safe grasp "
                        f"proposal: {geometry.reason}",
                        discovery_cloud=hybrid_cloud,
                    )
            if not recovered_metric_pose:
                log(
                    "stage=hybrid_object_geometry_applied "
                    "mode=registered_shape_with_silhouette_jaw_width "
                    f"surface_lower_bound_mm={hybrid_metric_width_mm:.1f} "
                    f"major_mm={hybrid_metric_major_mm:.1f} "
                    f"silhouette_width_mm="
                    f"{float(geometry.grasp_detection.min_width_mm):.1f}"
                )
        grasp_geometry_summary = geometry.summary()
        approach_rpy = tuple(float(v) for v in geometry.rpy_rad)
        live_rpy = np.asarray(A.ee_rpy(), dtype=float)
        use_saved_rotation = os.getenv(
            "REBOT_WRIST_USE_SAVED_CAMERA_ROTATION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # A nearly level wrist loses substantial far-and-low reach on the RS
        # arm. Estimate the endpoint and add pitch only in that workspace zone.
        # Motion occurs at the elevated ready pose and is followed by redetection.
        try:
            preview_mapping, _ = load_saved_camera_jacobian(
                tuple(float(v) for v in live_rpy)
            )
            preview_target = np.asarray(
                fixed_visual_target if fixed_visual_target is not None else tcp_cam,
                dtype=float,
            ).copy()
            if fixed_visual_target is not None:
                preview_target, _ = _apply_pick_target_y_offset(preview_target)
                preview_target[2] -= float(np.clip(
                    FIXED_PICK_CLOSER_MM, 0.0, 10.0
                ))
            preview_endpoint = np.asarray(A.ee_pos(), dtype=float) + np.linalg.solve(
                preview_mapping,
                preview_target - np.asarray(
                    geometry.grasp_detection.p_cam_mm, dtype=float
                ),
            )
            if preview_endpoint[0] > 0.410 and preview_endpoint[2] < 0.170:
                requested_pitch_bias = float(np.clip(
                    REACH_PITCH_BIAS_DEG, 0.0, 15.0
                ))
                if use_saved_rotation:
                    approach_rpy, pitch_bias, pitch_budget = \
                        _camera_bounded_pitch_bias(
                            approach_rpy, requested_pitch_bias
                        )
                else:
                    biased = np.asarray(approach_rpy, dtype=float)
                    biased[1] += math.radians(requested_pitch_bias)
                    approach_rpy = tuple(float(v) for v in biased)
                    pitch_bias = requested_pitch_bias
                    pitch_budget = {}
                reach_pitch_detail = (
                    f"far-low endpoint preview={preview_endpoint.round(4).tolist()}; "
                    f"requested +{requested_pitch_bias:.1f}deg, applied "
                    f"+{pitch_bias:.2f}deg pitch"
                )
                if pitch_budget.get("orientation_delta_deg") is not None:
                    reach_pitch_detail += (
                        f"; planned camera-orientation delta "
                        f"{float(pitch_budget['orientation_delta_deg']):.2f}deg"
                    )
                log(f"stage=adaptive_reach_pitch {reach_pitch_detail}")
                steps.append(("adaptive_reach_pitch", True, reach_pitch_detail))
        except (ValueError, np.linalg.LinAlgError, AttributeError, TypeError):
            # Ordinary reachability and fail-closed IK remain authoritative.
            pass
        # Validate the composed discovery + grasp + adaptive orientation before
        # sending any RPY command. Keep the later measured-pose check as a
        # second gate for real hardware tracking error.
        if use_saved_rotation:
            planned_camera_mapping, planned_camera_detail = \
                load_saved_camera_jacobian(approach_rpy)
            planned_camera_ok = planned_camera_mapping is not None
            steps.append((
                "prevalidate_planned_camera_geometry", planned_camera_ok,
                str(planned_camera_detail.get("reason")),
            ))
            if not planned_camera_ok:
                return recover(
                    "planned wrist orientation is outside the saved camera "
                    "calibration envelope; refusing to rotate",
                    rigid_camera_fit=planned_camera_detail,
                    grasp_geometry=grasp_geometry_summary,
                )
        rpy_delta = float(np.max(np.abs(
            np.asarray(approach_rpy, dtype=float) - live_rpy
        )))
        # This pose remains fixed through orientation, redetection, scene-map
        # capture, and metric-box reconstruction. Reuse it instead of taking a
        # late CAN state lock during the CPU-only 3D stage.
        geometry_stationary_xyz = np.asarray(A.ee_pos(), dtype=float)
        if rpy_delta > math.radians(0.1):
            rpy_waypoints, rpy_reason = _plan_rpy_trajectory(
                approach_rpy, xyz=geometry_stationary_xyz
            )
            steps.append(("prevalidate_bounded_orientation",
                          rpy_waypoints is not None, rpy_reason))
            if rpy_waypoints is None:
                return recover(
                    f"bounded wrist orientation is unsafe: {rpy_reason}",
                    grasp_geometry=grasp_geometry_summary,
                )
            log(f"stage=bounded_orientation_start {rpy_reason}")
            rpy_ok, rpy_exec_reason = _execute_rpy_trajectory(
                rpy_waypoints, log, fixed_xyz=geometry_stationary_xyz
            )
            steps.append(("execute_bounded_orientation",
                          rpy_ok, rpy_exec_reason))
            if not rpy_ok:
                return recover(
                    f"bounded wrist orientation failed: {rpy_exec_reason}",
                    grasp_geometry=grasp_geometry_summary,
                )
            _settle(eye, note="bounded geometry orientation")
            det_after, frame_after, geometry_after, orientation_consensus = \
                _observe_full_geometry_consensus(
                    eye, object_name, approach_rpy, log,
                    max_frames=6,
                    allow_registered_box_fallback=(
                        object_geometry_mode == "hybrid"
                    ),
                )
            steps.append((
                "redetect_after_orientation",
                det_after is not None,
                f"{det_after.source} at {det_after.z_mm:.0f}mm"
                if det_after else str(orientation_consensus.get("reason")),
            ))
            if det_after is None or frame_after is None or geometry_after is None:
                return recover(
                    "target lacked stable full-object geometry after the "
                    f"bounded wrist orientation: "
                    f"{orientation_consensus.get('reason')}",
                    grasp_geometry=grasp_geometry_summary,
                )
            steps.append(("geometry_after_orientation", geometry_after.ok,
                          str(orientation_consensus.get("reason"))))
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
        provisional_fit = {
            "reason": (
                "2D silhouette fits the hard aperture; clearance decision "
                "deferred until the measured 3D axis is checked against the "
                "calibrated jaw-closing direction"
            )
        }
        steps.append(("object_gripper_fit", True,
                      str(provisional_fit["reason"])))
        log("stage=object_gripper_fit ok=True status=deferred_to_3d "
            f"detail={provisional_fit['reason']}")

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

        # Recover the persistent metric object box before planning placement.
        # A blocked destination must not prevent the operator from seeing the
        # target OBB that explains grasp and attached-payload dimensions.
        approach_rpy = tuple(float(v) for v in A.ee_rpy())
        plan_det, plan_frame, plan_geometry, geometry_consensus = \
            _observe_full_geometry_consensus(
                eye, object_name, approach_rpy, log,
                max_frames=6,
                camera_mapping=J, tool_camera_mm=tcp_cam,
                table_z_m=load_safety_calibration().get("table_z_m"),
                stationary_ee_xyz_m=geometry_stationary_xyz,
                prefer_exact_target_mask=(object_geometry_mode == "hybrid"),
                allow_registered_box_fallback=(
                    object_geometry_mode == "hybrid"
                ),
            )
        steps.append(("preplan_object_observation", plan_det is not None,
                      f"{plan_det.source} at {plan_det.z_mm:.0f}mm"
                      if plan_det else str(geometry_consensus.get("reason"))))
        if plan_det is None or plan_frame is None or plan_geometry is None:
            return recover(
                "object lost before metric 3D reconstruction "
                f"({geometry_consensus.get('reason')})"
            )
        steps.append(("preplan_full_object_geometry", plan_geometry.ok,
                      str(geometry_consensus.get("reason"))))
        if not plan_geometry.ok or plan_geometry.grasp_detection is None:
            return recover(
                f"full-object geometry is unsafe before motion: "
                f"{plan_geometry.reason}",
                grasp_geometry=plan_geometry.summary(),
            )
        stationary_object_box_3d = plan_geometry.object_box_3d
        full_object_box_3d = stationary_object_box_3d
        if stationary_object_box_3d is None:
            registered_fallback = bool(
                object_geometry_mode == "hybrid"
                and geometry_consensus.get("registered_box_fallback")
                and hybrid_object_box_3d is not None
            )
            if not registered_fallback:
                return recover(
                    "metric 3D object box was not recovered from the "
                    "unobstructed RGB-D views; refusing to reduce the object "
                    "to 2D geometry"
                )
        if object_geometry_mode == "hybrid":
            if hybrid_object_box_3d is None:
                return recover(
                    "hybrid mode lost its registered multi-view object box"
                )
            if stationary_object_box_3d is None:
                steps.append((
                    "hybrid_stationary_box_crosscheck", True,
                    "registered two-view native-semantic object box retained; "
                    "repeated stationary semantic centroids confirmed target "
                    "position while the complete close-view mask was stochastic",
                ))
                log(
                    "stage=hybrid_object_box_crosscheck ok=True "
                    "basis=registered_native_semantic_box_plus_stationary_"
                    "semantic_position"
                )
                full_object_box_3d = hybrid_object_box_3d
                plan_geometry = replace(
                    plan_geometry, object_box_3d=full_object_box_3d,
                    reason=(
                        f"{plan_geometry.reason}; registered multi-view object "
                        "box selected after stationary semantic-position "
                        "confirmation"
                    ),
                )
            else:
                try:
                    hybrid_extent = np.sort(np.asarray(
                        hybrid_object_box_3d["observed_size_m"], dtype=float
                    ))
                    stationary_extent = np.sort(np.asarray(
                        stationary_object_box_3d["observed_size_m"], dtype=float
                    ))
                    extent_limit = np.maximum(
                        0.012,
                        0.35 * np.maximum(hybrid_extent, stationary_extent),
                    )
                    extent_consistent = bool(
                        hybrid_extent.shape == (3,)
                        and stationary_extent.shape == (3,)
                        and np.all(np.isfinite(hybrid_extent))
                        and np.all(np.isfinite(stationary_extent))
                        and np.all(np.abs(
                            hybrid_extent - stationary_extent
                        ) <= extent_limit)
                    )
                except (KeyError, TypeError, ValueError):
                    extent_consistent = False
                    hybrid_extent = np.array([], dtype=float)
                    stationary_extent = np.array([], dtype=float)
                box_policy = _hybrid_box_selection_policy(
                    extent_consistent,
                    place_after_pick=place_after_pick,
                    scene_collision_enabled=SCENE_COLLISION_ENABLED,
                )
                truncation_ok, truncation_detail = \
                    _stationary_box_is_long_axis_truncation(
                        hybrid_object_box_3d, stationary_object_box_3d
                    )
                if box_policy == "reject" and truncation_ok:
                    box_policy = "long_axis_partial_observation"
                if box_policy == "reject":
                    # Placement/carry planning needs a persistent volume and
                    # therefore keeps the fail-closed multi-view cross-check.
                    # Pick-only needs only the currently visible local jaw
                    # section.  Its two-frame stationary native-mask depth box
                    # is fresher and avoids rejecting a valid 17 mm marker
                    # because swept-view registration inflated the orthogonal
                    # (non-closing) thickness axis.
                    steps.append((
                        "hybrid_stationary_box_crosscheck", False,
                        f"hybrid_mm="
                        f"{(hybrid_extent * 1000.0).round(1).tolist()}; "
                        f"stationary_mm="
                        f"{(stationary_extent * 1000.0).round(1).tolist()}",
                    ))
                    log(
                        "stage=hybrid_object_box_crosscheck ok=False "
                        "policy=persistent_scene_geometry_required "
                        f"hybrid_mm="
                        f"{(hybrid_extent * 1000.0).round(1).tolist()} "
                        f"stationary_mm="
                        f"{(stationary_extent * 1000.0).round(1).tolist()}"
                    )
                    return recover(
                        "registered multi-view and stationary object boxes "
                        "disagree; refusing to choose one for collision or "
                        "grasp geometry"
                    )
                if box_policy == "fresh_stationary_pick_only":
                    full_object_box_3d = stationary_object_box_3d
                    plan_geometry = replace(
                        plan_geometry, object_box_3d=full_object_box_3d,
                        reason=(
                            f"{plan_geometry.reason}; fresh stationary "
                            "native-mask depth box selected for pick-only "
                            "local grasp geometry"
                        ),
                    )
                    steps.append((
                        "hybrid_stationary_box_crosscheck", True,
                        "pick-only selected the fresh stationary native-mask "
                        "depth box; swept-view disagreement was confined to "
                        f"nonpersistent geometry; hybrid_mm="
                        f"{(hybrid_extent * 1000.0).round(1).tolist()}; "
                        f"stationary_mm="
                        f"{(stationary_extent * 1000.0).round(1).tolist()}",
                    ))
                    log(
                        "stage=hybrid_object_box_crosscheck ok=True "
                        "policy=pick_only_fresh_stationary_box "
                        f"hybrid_mm="
                        f"{(hybrid_extent * 1000.0).round(1).tolist()} "
                        f"stationary_mm="
                        f"{(stationary_extent * 1000.0).round(1).tolist()}"
                    )
                else:
                    steps.append((
                        "hybrid_stationary_box_crosscheck", True,
                        (
                            f"{truncation_detail.get('reason')}; "
                            if box_policy == "long_axis_partial_observation"
                            else ""
                        )
                        + f"hybrid_mm="
                          f"{(hybrid_extent * 1000.0).round(1).tolist()}; "
                          f"stationary_mm="
                          f"{(stationary_extent * 1000.0).round(1).tolist()}",
                    ))
                    log(
                        "stage=hybrid_object_box_crosscheck ok=True "
                        f"policy={box_policy} "
                        f"hybrid_mm="
                        f"{(hybrid_extent * 1000.0).round(1).tolist()} "
                        f"stationary_mm="
                        f"{(stationary_extent * 1000.0).round(1).tolist()} "
                        f"detail={truncation_detail.get('reason') if truncation_ok else 'independent extents consistent'}"
                    )
                    # When only one observation clipped the long axis, retain
                    # the larger measured object volume.  This is conservative
                    # for collision/payload planning and does not alter the
                    # independently measured local jaw cross-section.
                    use_stationary_long_box = bool(
                        box_policy == "long_axis_partial_observation"
                        and truncation_detail.get("longer_source") ==
                        "fresh_stationary"
                    )
                    full_object_box_3d = (
                        stationary_object_box_3d
                        if use_stationary_long_box
                        else hybrid_object_box_3d
                    )
                    plan_geometry = replace(
                        plan_geometry, object_box_3d=full_object_box_3d,
                        reason=(
                            f"{plan_geometry.reason}; "
                            + ("larger multi-frame object box selected after "
                               "partial long-axis visibility check"
                               if truncation_ok else "extent cross-check")
                        ),
                    )
        steps.append((
            "metric_object_box_3d", True,
            f"observed_mm="
            f"{(np.asarray(full_object_box_3d['observed_size_m']) * 1000).round(1).tolist()}; "
            f"conservative_mm="
            f"{(np.asarray(full_object_box_3d['conservative_size_m']) * 1000).round(1).tolist()}; "
            f"views={full_object_box_3d.get('views')}",
        ))
        if hybrid_identity_gate is not None:
            reanchored, reanchor_detail = \
                hybrid_identity_gate.reanchor_from_metric_box(
                    full_object_box_3d
                )
            steps.append((
                "hybrid_identity_metric_reanchor", reanchored,
                str(reanchor_detail.get("reason")),
            ))
            log(
                "stage=hybrid_identity_metric_reanchor "
                f"ok={reanchored} shift_mm="
                f"{reanchor_detail.get('shift_mm')} "
                f"anchor={reanchor_detail.get('anchor_base_m')} "
                f"detail={reanchor_detail.get('reason')}"
            )
            if not reanchored:
                return recover(
                    "verified metric object geometry could not safely update "
                    "the partial-view identity anchor",
                    identity_reanchor=reanchor_detail,
                )
        aligned_3d_width_detail = _aligned_3d_jaw_width(
            full_object_box_3d, J
        )
        max_aperture_mm = float(jaw_plan["max_aperture_mm"])
        aligned_width_safe = bool(
            aligned_3d_width_detail.get("ok") and
            float(aligned_3d_width_detail["conservative_width_mm"])
            <= max_aperture_mm
        )
        silhouette_width_mm = float(
            plan_geometry.grasp_detection.min_width_mm
        )
        width_reduction_mm = (
            silhouette_width_mm -
            float(aligned_3d_width_detail.get(
                "observed_width_mm", silhouette_width_mm
            ))
        )
        override_useful = bool(aligned_width_safe and width_reduction_mm >= 3.0)
        aligned_3d_width_detail["silhouette_width_mm"] = silhouette_width_mm
        aligned_3d_width_detail["width_reduction_mm"] = width_reduction_mm
        aligned_3d_width_detail["override_applied"] = override_useful
        steps.append((
            "align_3d_thickness_to_jaws", aligned_width_safe,
            str(aligned_3d_width_detail.get("reason")),
        ))
        log(
            f"stage=align_3d_thickness_to_jaws ok={aligned_width_safe} "
            f"detail={aligned_3d_width_detail.get('reason')}"
        )
        if override_useful:
            plan_geometry = _geometry_with_3d_jaw_width(
                plan_geometry, aligned_3d_width_detail
            )
            geometry = plan_geometry
            det = plan_geometry.grasp_detection
            assert det is not None
            jaw_plan = _plan_aperture(det)
            log(
                "stage=jaw_plan_3d_override "
                f"ok={jaw_plan.get('ok')} detail={jaw_plan.get('reason')}"
            )
            if not jaw_plan.get("ok"):
                return recover(
                    f"aligned 3D jaw width cannot be commanded: "
                    f"{jaw_plan.get('reason')}"
                )
        fit_ok, object_fit = _validate_object_gripper_fit(
            plan_geometry, jaw_plan, full_object_box_3d
        )
        steps.append((
            "metric_3d_object_gripper_fit", fit_ok,
            str(object_fit.get("reason")),
        ))
        log(
            f"stage=metric_3d_object_gripper_fit ok={fit_ok} "
            f"box_mm={object_fit.get('box_dimensions_mm')} "
            f"detail={object_fit.get('reason')}"
        )
        if not fit_ok:
            box_mm = (
                np.asarray(full_object_box_3d["conservative_size_m"]) * 1000.0
            ).round(1).tolist()
            return recover(
                "complete object is not safely graspable in the current jaw "
                f"plane: {object_fit.get('reason')}; conservative 3D box "
                f"dimensions are {box_mm}mm",
                object_gripper_fit={**object_fit,
                                    "object_box_3d": full_object_box_3d},
                grasp_geometry=plan_geometry.summary(),
            )

        if SCENE_COLLISION_ENABLED:
            map_det, map_frame, map_why = _observe(eye, object_name)
            if map_det is None or map_frame is None:
                steps.append(("scene_collision_map", False, str(map_why)))
                return recover(
                    "target could not be isolated in the collision-map frame; "
                    "refusing to treat it as generic free space"
                )
            if isinstance(full_object_box_3d, dict):
                # The fresh observation above is still required so a missing
                # or identity-mismatched target fails closed. Its stochastic
                # close-view mask must not reopen a grasp decision that was
                # already established by stable geometry, aligned metric jaw
                # projection, and the complete-object fit gate.
                map_geometry = plan_geometry
                log(
                    "stage=scene_collision_target status=verified_metric_reuse "
                    "detail=fresh target identity present; stationary measured "
                    "3D OBB is the sole target exclusion and grasp geometry "
                    "was not reinterpreted"
                )
            else:
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
            fusion_count = int(np.clip(
                int(os.getenv("REBOT_SCENE_FUSION_FRAMES", "3")), 1, 5
            ))
            extra_map_frames = []
            for frame_index in range(2, fusion_count + 1):
                try:
                    extra_map_frames.append(eye.frame())
                    log(
                        "stage=scene_collision_capture "
                        f"frame={frame_index}/{fusion_count} ok=True arm=stationary"
                    )
                except Exception as exc:  # noqa: BLE001
                    log(
                        "stage=scene_collision_capture "
                        f"frame={frame_index}/{fusion_count} ok=False detail={exc}"
                    )
            collision_ok, collision_detail = _install_scene_collision_map(
                map_frame, map_geometry, J, extra_frames=extra_map_frames,
                extra_pose_views=scene_discovery_views,
                object_boxes_3d={"pick_target": full_object_box_3d},
                log=log,
            )
            steps.append((
                "scene_collision_map", collision_ok,
                str(collision_detail.get("reason")),
            ))
            log("stage=scene_collision_map "
                f"ok={collision_ok} "
                f"columns_raw={collision_detail.get('raw_obstacle_columns')} "
                f"columns_after_self_filter="
                f"{collision_detail.get('self_filtered_obstacle_columns')} "
                f"columns_consolidated="
                f"{collision_detail.get('obstacle_columns')} "
                f"columns_removed_as_overlap="
                f"{collision_detail.get('consolidated_obstacle_reduction', 0)} "
                f"target_columns_removed="
                f"{collision_detail.get('excluded_target_columns', 0)} "
                f"temporal_fusion="
                f"{collision_detail.get('temporal_fusion')} "
                f"required_frames="
                f"{collision_detail.get('temporal_required_frames')} "
                f"capture_pose_clear="
                f"{collision_detail.get('current_pose_gate', {}).get('ok')} "
                f"moving_views={collision_detail.get('moving_discovery_views', 0)} "
                f"target_exclusion="
                f"{collision_detail.get('target_exclusion_method')} "
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

            # Placement-reference discovery may yaw, pitch, or retreat the
            # wrist. The pick route must start from the same ready pose used by
            # its frozen scene geometry, otherwise continuation IK is seeded
            # from a different configuration and the supposedly preplanned
            # trajectory is not actually anchored to the snapshot.
            current_rpy = np.asarray(A.ee_rpy(), dtype=float)
            orientation_error = float(np.max(np.abs(
                current_rpy - np.asarray(approach_rpy, dtype=float)
            )))
            position_restored_before_orientation = False
            if orientation_error > math.radians(0.25):
                restore_rpy, restore_rpy_reason = _plan_rpy_trajectory(
                    approach_rpy, xyz=A.ee_pos()
                )
                if restore_rpy is None:
                    # A placement-reference recenter can finish at a high,
                    # laterally displaced pose where changing wrist
                    # orientation in place crosses an RS singularity.  This is
                    # the same restoration-order issue handled by metric grasp
                    # discovery above: first retrace position while preserving
                    # the currently reachable orientation, then rotate at the
                    # known-reachable frozen pick XYZ.  Both complete legs are
                    # prevalidated; no tolerance or workspace limit is relaxed.
                    log(
                        "stage=return_from_place_reference_view "
                        "phase=orientation_first status=blocked "
                        f"detail={restore_rpy_reason}; "
                        "fallback=translate_first"
                    )
                    live_xyz = np.asarray(A.ee_pos(), dtype=float)
                    live_rpy = tuple(float(value) for value in current_rpy)
                    restore_position_first, position_first_reason = \
                        _plan_fixed_approach_trajectory(
                            geometry_stationary_xyz,
                            start_m=live_xyz,
                            fixed_rpy=live_rpy,
                        )
                    steps.append((
                        "prevalidate_return_from_place_view_position_first",
                        restore_position_first is not None,
                        position_first_reason,
                    ))
                    if restore_position_first is None:
                        return recover(
                            "could not return from the placement-reference "
                            "view by either restoration order: orientation "
                            f"first failed ({restore_rpy_reason}); translation "
                            f"first failed ({position_first_reason})"
                        )
                    restored_position, restored_position_reason = \
                        _execute_fixed_approach_trajectory(
                            restore_position_first, log, fixed_rpy=live_rpy
                        )
                    steps.append((
                        "execute_return_from_place_view_position_first",
                        restored_position, restored_position_reason,
                    ))
                    if not restored_position:
                        return recover(
                            "translation-first return from the placement-"
                            f"reference view failed: {restored_position_reason}"
                        )
                    position_restored_before_orientation = True
                    restore_rpy, restore_rpy_reason = _plan_rpy_trajectory(
                        approach_rpy, xyz=geometry_stationary_xyz
                    )
                steps.append((
                    "prevalidate_return_from_place_view_orientation",
                    restore_rpy is not None, restore_rpy_reason,
                ))
                if restore_rpy is None:
                    return recover(
                        "could not restore the frozen pick orientation after "
                        f"returning to its XYZ: {restore_rpy_reason}"
                    )
                restored, restored_reason = _execute_rpy_trajectory(
                    restore_rpy, log,
                    fixed_xyz=(
                        geometry_stationary_xyz
                        if position_restored_before_orientation else A.ee_pos()
                    ),
                )
                steps.append((
                    "execute_return_from_place_view_orientation",
                    restored, restored_reason,
                ))
                if not restored:
                    return recover(
                        "return from the placement-reference orientation "
                        f"failed: {restored_reason}"
                    )
            current_xyz = np.asarray(A.ee_pos(), dtype=float)
            return_distance = float(np.linalg.norm(
                geometry_stationary_xyz - current_xyz
            ))
            if return_distance > 0.001 and \
                    not position_restored_before_orientation:
                restore_xyz, restore_xyz_reason = \
                    _plan_fixed_approach_trajectory(
                        geometry_stationary_xyz, start_m=current_xyz,
                        fixed_rpy=approach_rpy,
                    )
                steps.append((
                    "prevalidate_return_from_place_view_position",
                    restore_xyz is not None, restore_xyz_reason,
                ))
                if restore_xyz is None:
                    return recover(
                        "could not return from the placement-reference view to "
                        f"the frozen pick position: {restore_xyz_reason}"
                    )
                restored, restored_reason = \
                    _execute_fixed_approach_trajectory(
                        restore_xyz, log, fixed_rpy=approach_rpy
                    )
                steps.append((
                    "execute_return_from_place_view_position",
                    restored, restored_reason,
                ))
                if not restored:
                    return recover(
                        "return from the placement-reference position failed: "
                        f"{restored_reason}"
                    )
            log(
                "stage=return_from_place_reference_view status=ready "
                f"xyz={np.round(A.ee_pos(), 4).tolist()} "
                f"rpy_deg={np.round(np.degrees(A.ee_rpy()), 1).tolist()}"
            )

        voxel_place_requested = bool(
            scene_place and SCENE_COLLISION_ENABLED
            and _ACTIVE_SCENE_MAP is not None
            and os.getenv("REBOT_VOXEL_PLACE_NAV", "1") != "0"
        )
        if voxel_place_requested:
            # The occupancy map is for navigation after pickup. Applying its
            # coarse columns to the terminal grasp can reject the intentional
            # gripper/object intersection. Preserve the proven pick behavior;
            # workspace, IK, self-collision and operator stop gates remain on.
            A.clear_collision_scene()
            steps.append((
                "suspend_scene_for_proven_pick", True,
                "environment columns cached for post-lift voxel navigation",
            ))
            log(
                "stage=scene_collision_lifecycle phase=pick suspended=true "
                "reason=working_pick_unchanged cache=retained "
                "ik_self_collision_workspace=active"
            )

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
        _show(*plan_frame, tag="fixed_pick_preplanned",
              det=plan_det, tcp_cam=tcp_cam, target_cam=requested_cam,
              note="metric object observation used for the fixed trajectory")
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
        selected_grasp_axis_fraction = _selected_axis_fraction(plan_geometry)
        selected_grasp_section_locked = False
        log("stage=grasp_endpoint_plan status=checking "
            "detail=endpoint IK search; arm stationary")
        endpoint_started = time.monotonic()
        planned_cam, reachability = _reachable_approach_target(
            plan_det, J, requested_cam
        )
        log("stage=grasp_endpoint_plan status=complete "
            f"elapsed_s={time.monotonic() - endpoint_started:.1f} "
            f"full_target={bool(reachability.get('full_hover'))} "
            f"detail={reachability.get('reason')}")
        # The endpoint-only check above uses the live joints as its seed. Before
        # accepting its partial fallback, try the exact requested endpoint with
        # deterministic continuation IK across the complete small-waypoint path.
        # A continuous path can reach poses that a one-shot solve from home cannot.
        requested_endpoint_raw = reachability.get("requested_endpoint_m")
        exact_waypoints = None
        exact_reason = "requested endpoint unavailable"
        exact_path_attempted = False
        # When endpoint IK already accepted the full requested target, the
        # clearance planner below validates a complete path to that same pose.
        # Re-validating a straight path here first was duplicate work and could
        # look like a hang in collision-rich scenes.
        if requested_endpoint_raw is not None and not reachability.get("full_hover"):
            exact_path_attempted = True
            log("stage=grasp_exact_path status=validating route=nominal "
                "detail=deterministic IK and collision checks; arm stationary")
            exact_started = time.monotonic()
            exact_waypoints, exact_reason = _plan_fixed_approach_trajectory(
                requested_endpoint_raw, fixed_rpy=approach_rpy
            )
            log("stage=grasp_exact_path status="
                f"{'accepted' if exact_waypoints is not None else 'rejected'} "
                f"route=nominal elapsed_s={time.monotonic() - exact_started:.1f} "
                f"detail={exact_reason}")
            if exact_waypoints is None:
                log("stage=grasp_exact_path status=validating route=fine "
                    "detail=deterministic IK and collision checks; arm stationary")
                fine_started = time.monotonic()
                fine_waypoints, fine_reason = _plan_fixed_approach_trajectory(
                    requested_endpoint_raw, cap_fraction=0.25,
                    fixed_rpy=approach_rpy,
                )
                log("stage=grasp_exact_path status="
                    f"{'accepted' if fine_waypoints is not None else 'rejected'} "
                    f"route=fine elapsed_s={time.monotonic() - fine_started:.1f} "
                    f"detail={fine_reason}")
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
        elif requested_endpoint_raw is not None:
            exact_reason = (
                "endpoint IK accepted the exact target; duplicate straight-path "
                "check skipped because the clearance chain is validated next"
            )
            log(f"stage=grasp_exact_path status=skipped detail={exact_reason}")
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
            # Do not validate another straight path here: every qualified
            # endpoint is immediately handed to the stricter clearance/axial
            # planner below, which validates the complete route as one chain.
            waypoints, trajectory_reason = None, exact_reason
        if planned_target_qualified and np.all(np.isfinite(endpoint)):
            endpoint_gate = (
                A.check_pose_clearance(*endpoint.tolist(), *approach_rpy)
                if SCENE_COLLISION_ENABLED else None
            )
            if endpoint_gate is not None and not endpoint_gate.ok:
                waypoints = None
                trajectory_reason = (
                    "nominal terminal grasp pose is collision-blocked: "
                    f"{endpoint_gate.reason}"
                )
                reachability["endpoint_clearance"] = {
                    "ok": False, "reason": endpoint_gate.reason,
                }
                log("stage=grasp_clearance_plan status=terminal_blocked "
                    "detail=endpoint collision check failed; searching "
                    f"measured alternate grasp poses; reason={endpoint_gate.reason}")
            else:
                log("stage=grasp_clearance_plan status=building "
                    "detail=bounded clearance routes; arm stationary")
                waypoints, trajectory_reason, clearance_detail = \
                    _plan_clearance_axial_trajectory(
                        endpoint, J, approach_rpy, log=log, phase="initial"
                    )
                reachability.update(clearance_detail)
        # A named object's centre is not the only valid grasp.  If the nominal
        # jaw-centred endpoint is kinematically unreachable (as can happen on
        # one lateral side of the RS workspace), search the other *measured*
        # cross-sections and the calibrated free space inside the open jaws.
        # This search used to run only when the optional scene-collision model
        # was enabled, leaving pick-only mode unable to use the same safe
        # alternatives.  Every proposal still passes endpoint clearance and a
        # complete deterministic IK/self-collision path before any movement.
        if not planned_target_qualified or waypoints is None:
            alternate = _plan_collision_aware_grasp_pose(
                plan_geometry, plan_frame, J, requested_cam, approach_rpy,
                jaw_plan, full_object_box_3d, log,
            )
            steps.append((
                "measured_alternate_grasp_pose",
                bool(alternate.get("ok")),
                str(alternate.get("reason")),
            ))
            if alternate.get("ok"):
                plan_geometry = alternate["geometry"]
                plan_det = alternate["body"]
                jaw_plan = alternate["jaw_plan"]
                approach_rpy = tuple(
                    float(value) for value in alternate.get(
                        "approach_rpy", approach_rpy
                    )
                )
                J = np.asarray(
                    alternate.get("camera_mapping", J), dtype=float
                )
                requested_cam = np.asarray(
                    alternate["requested_cam_mm"], dtype=float
                )
                endpoint = np.asarray(alternate["endpoint_m"], dtype=float)
                waypoints = alternate["waypoints"]
                trajectory_reason = str(alternate["trajectory_reason"])
                planned_cam = requested_cam.copy()
                planned_residual_mm = 0.0
                planned_target_qualified = True
                selected_grasp_axis_fraction = alternate.get("axis_fraction")
                selected_grasp_section_locked = bool(
                    alternate["body_name"] != "selected"
                )
                expected_feature_cam, feature_to_grasp_cam = \
                    _feature_target_for_grasp_target(
                        requested_cam, plan_feature_det, plan_det
                    )
                grasp_geometry_summary = plan_geometry.summary()
                reachability = {
                    **alternate["reachability"],
                    **alternate["clearance"],
                    "measured_alternate_grasp_pose": {
                        "body": alternate["body_name"],
                        "axis_fraction": selected_grasp_axis_fraction,
                        "jaw_slot_offset_mm": alternate["slot_offset_mm"],
                        "candidates_tested": alternate["attempts"],
                        "pitch_offset_deg": alternate.get(
                            "pitch_offset_deg", 0.0
                        ),
                    },
                    "full_hover": True,
                    "predicted_endpoint_m": endpoint.round(4).tolist(),
                    "residual_to_requested_target_mm": 0.0,
                }
                steps.append((
                    "alternate_feature_to_grasp_geometry",
                    True,
                    f"selected measured body section; feature offset "
                    f"{feature_to_grasp_cam.round(1).tolist()}mm; expected "
                    f"endpoint feature={expected_feature_cam.round(1).tolist()}mm",
                ))
                _show(
                    *plan_frame, tag="fixed_pick_alternate_grasp",
                    det=plan_det, tcp_cam=tcp_cam,
                    target_cam=requested_cam,
                    note="collision-free grasp selected from measured geometry",
                )
                log(
                    "stage=measured_alternate_grasp_pose status=accepted "
                    f"body={alternate['body_name']} "
                    f"axis_fraction={selected_grasp_axis_fraction} "
                    f"jaw_slot_offset_mm={alternate['slot_offset_mm']:+.1f} "
                    f"endpoint={endpoint.round(4).tolist()}"
                )
            else:
                trajectory_reason = (
                    f"{trajectory_reason}; measured alternate terminal search: "
                    f"{alternate.get('reason')}"
                )
        plan_detail = (
            f"{reachability.get('reason')}; predicted residual "
            f"{planned_residual_mm:.1f}mm, bounded endpoint-correction "
            f"envelope {_endpoint_retry_envelope_mm(object_visible=True):.1f}mm"
            + ("" if exact_waypoints is not None or not exact_path_attempted else
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
        log(
            "stage=grasp_section_tracking mode="
            f"{'fixed_initial_scene' if FIXED_SCENE_TRAJECTORY else ('collision_selected_section_locked' if selected_grasp_section_locked else 'fresh_best_section_at_checkpoint')} "
            f"axis_fraction={selected_grasp_axis_fraction}"
        )
        checkpoint_policy: dict[str, Any] | None = None
        if FIXED_SCENE_TRAJECTORY and not isinstance(waypoints, dict):
            checkpoint_distance_m, checkpoint_policy = \
                _dynamic_visible_checkpoint_m(
                    plan_geometry, plan_frame, requested_cam
                )
            checkpoint_prefix, _, checkpoint_remaining = \
                _visible_checkpoint_prefix(
                    waypoints, endpoint, A.ee_pos(),
                    checkpoint_m=checkpoint_distance_m,
                )
            checkpoint_count = len(checkpoint_prefix)
        elif isinstance(waypoints, dict):
            checkpoint_prefix = waypoints
            checkpoint_remaining = float(
                reachability.get("checkpoint_remaining_m", HYBRID_CHECKPOINT_M)
            )
            checkpoint_count = len(waypoints.get("joint_path", []))
        else:
            checkpoint_prefix, _, checkpoint_remaining = \
                _visible_checkpoint_prefix(
                    waypoints, endpoint, A.ee_pos()
                )
            checkpoint_count = len(checkpoint_prefix)
        # Keep the untouched, already validated terminal suffix. Hybrid mode
        # may reuse it only after a fresh position-only target consensus. This
        # avoids replacing good registered dimensions with a close-range mask
        # contaminated by the table or gripper.
        frozen_checkpoint_suffix = (
            None if isinstance(waypoints, dict) else
            list(waypoints[len(checkpoint_prefix):])
        )
        frozen_checkpoint_endpoint = np.asarray(endpoint, dtype=float).copy()
        frozen_checkpoint_geometry = plan_geometry
        frozen_checkpoint_geometry_summary = grasp_geometry_summary
        frozen_checkpoint_jaw_plan = dict(jaw_plan)
        frozen_checkpoint_expected_feature = np.asarray(
            expected_feature_cam, dtype=float
        ).copy()
        frozen_checkpoint_feature_offset = np.asarray(
            feature_to_grasp_cam, dtype=float
        ).copy()
        frozen_checkpoint_requested_cam = np.asarray(
            requested_cam, dtype=float
        ).copy()
        frozen_checkpoint_anchor = (
            None if hybrid_identity_gate is None
            or hybrid_identity_gate.anchor_base_m is None else
            np.asarray(hybrid_identity_gate.anchor_base_m, dtype=float).copy()
        )
        if FIXED_SCENE_TRAJECTORY:
            complete_count = (
                len(waypoints.get("joint_path", []))
                if isinstance(waypoints, dict) else len(waypoints)
            )
            log(
                "stage=fixed_scene_pick_trajectory status=macro_frozen "
                f"waypoints={complete_count} checkpoint_waypoints="
                f"{checkpoint_count} checkpoint_remaining_mm="
                f"{checkpoint_remaining * 1000.0:.1f} "
                "stationary_final_leg_replanning=enabled "
                f"checkpoint_policy={checkpoint_policy}"
            )
            steps.append((
                "freeze_complete_pick_trajectory", True,
                "complete initial-scene trajectory passed IK/collision checks; "
                "its macro prefix is frozen to a dynamically selected visible "
                "checkpoint, where fresh stationary geometry may validate a "
                "replacement final leg; if close masks spill, repeated target "
                "position must confirm reuse of the original validated suffix",
            ))
        else:
            log("stage=visible_checkpoint_plan "
                f"distance_mm={checkpoint_remaining * 1000:.1f} "
                f"waypoints={checkpoint_count} gripper=closed")
            steps.append((
                "prevalidate_visible_checkpoint", True,
                f"prefix of the already validated complete path; "
                f"{checkpoint_count} waypoints; predicted remaining "
                f"{checkpoint_remaining * 1000:.1f}mm",
            ))
        checkpoint_start_xyz = np.asarray(A.ee_pos(), dtype=float)
        if checkpoint_prefix:
            checkpoint_ok, checkpoint_reason = \
                _execute_fixed_approach_trajectory(
                    checkpoint_prefix, log, fixed_rpy=approach_rpy
                )
            steps.append(("execute_visible_checkpoint",
                          checkpoint_ok, checkpoint_reason))
            if not checkpoint_ok:
                return recover(
                    f"visible checkpoint motion failed: {checkpoint_reason}"
                )
        checkpoint_xyz = np.asarray(A.ee_pos(), dtype=float)
        _settle(eye, target_cam=requested_cam,
                note="visible pre-grasp checkpoint")

        def replan_from_visible_checkpoint(phase: str) -> dict[str, Any]:
            samples: list[dict[str, Any]] = []
            fresh_samples: list[dict[str, Any]] = []
            target_position_samples: list[dict[str, Any]] = []
            failures: list[str] = []

            def collect_sample(sample_index: int, total: int) -> None:
                sample_feature, sample_frame, sample_why = _observe(
                    eye, object_name
                )
                if sample_feature is None or sample_frame is None:
                    failures.append(str(sample_why))
                    log("stage=visible_checkpoint_sample "
                        f"phase={phase} frame={sample_index}/{total} ok=False "
                        f"detail={sample_why}")
                    return
                if hybrid_identity_gate is not None:
                    point_base, point_reason = \
                        hybrid_identity_gate._point_base(sample_feature)
                    if point_base is not None:
                        target_position_samples.append({
                            "point_base_m": np.asarray(
                                point_base, dtype=float
                            ).tolist(),
                            "feature": sample_feature,
                            "frame": sample_frame,
                        })
                    else:
                        failures.append(
                            f"target position mapping failed: {point_reason}"
                        )
                sample_geometry = _plan_scene_grasp_geometry(
                    sample_frame, sample_feature,
                    current_rpy=tuple(float(v) for v in A.ee_rpy()),
                    allow_orientation=False, object_name=object_name,
                )
                if not sample_geometry.ok \
                        or sample_geometry.grasp_detection is None:
                    failures.append(sample_geometry.reason)
                    log("stage=visible_checkpoint_sample "
                        f"phase={phase} frame={sample_index}/{total} ok=False "
                        f"detail={sample_geometry.reason}")
                    return
                # Preserve the best complete geometry from every fresh frame
                # before applying the identity lock for a previously selected
                # alternate section.  Perspective can change the discrete
                # cross-section profile enough that an old section is no
                # longer represented by the same sample.  A stable fresh body
                # may still be used, but only after the endpoint and complete
                # IK/collision route are planned again below.
                fresh_body = sample_geometry.grasp_detection
                fresh_samples.append({
                    "body_cam_mm": np.asarray(
                        fresh_body.p_cam_mm, dtype=float
                    ).tolist(),
                    "feature": sample_feature,
                    "frame": sample_frame,
                    "geometry": sample_geometry,
                    "body": fresh_body,
                })
                # If collision planning selected a different measured section
                # of an elongated object, reacquire that same normalized body
                # location in each fresh frame. This preserves the dynamic
                # choice instead of silently snapping back to the centre.
                selected_sample_geometry = _fresh_geometry_for_selected_section(
                    sample_geometry, sample_frame, sample_feature,
                    feature_to_grasp_cam,
                    section_locked=selected_grasp_section_locked,
                    axis_fraction=selected_grasp_axis_fraction,
                )
                if selected_sample_geometry is None:
                    failures.append(
                        "selected measured body section was not reacquired"
                    )
                    log("stage=visible_checkpoint_sample "
                        f"phase={phase} frame={sample_index}/{total} ok=False "
                        "detail=selected measured body section was not "
                        "reacquired; stable fresh geometry retained for "
                        "collision-checked reselection")
                    return
                sample_geometry = selected_sample_geometry
                body_det = sample_geometry.grasp_detection
                if body_det is None:
                    failures.append("selected measured body section vanished")
                    return
                samples.append({
                    "body_cam_mm": np.asarray(
                        body_det.p_cam_mm, dtype=float
                    ).tolist(),
                    "feature": sample_feature,
                    "frame": sample_frame,
                    "geometry": sample_geometry,
                    "body": body_det,
                })
                log("stage=visible_checkpoint_sample "
                    f"phase={phase} frame={sample_index}/{total} ok=True "
                    f"body_xyz_mm="
                    f"{np.round(body_det.p_cam_mm, 1).tolist()}")

            def frozen_position_fallback(
                trigger: str,
            ) -> dict[str, Any] | None:
                """Reuse the registered grasp only after identity/path gates.

                Close-range masks can be stable yet geometrically wrong when
                the gripper or table spills into the segmentation.  The
                registered hybrid geometry and jaw plan may be retained, but
                only for the same position-confirmed target and only if the
                untouched terminal suffix still passes the complete waypoint
                validator from the arm's current measured state.
                """
                position_median, position_detail, position_cluster = \
                    _checkpoint_target_position_consensus(
                        target_position_samples, frozen_checkpoint_anchor,
                        min_frames=2,
                    )
                frozen_geometry_verified = bool(
                    hybrid_identity_gate is not None
                    and isinstance(full_object_box_3d, dict)
                    and fit_ok
                )
                frozen_suffix_ok = bool(frozen_checkpoint_suffix)
                frozen_suffix_reason = "validated terminal suffix unavailable"
                if frozen_suffix_ok:
                    frozen_suffix_ok, frozen_suffix_reason = \
                        _validate_fixed_waypoints(
                            frozen_checkpoint_suffix, approach_rpy
                        )
                if position_median is None or not frozen_geometry_verified \
                        or not frozen_suffix_ok:
                    log(
                        "stage=visible_checkpoint_position_only_fallback "
                        f"phase={phase} ok=False trigger={trigger!r} detail="
                        f"{position_detail.get('reason')}; "
                        f"registered_geometry={frozen_geometry_verified}; "
                        f"path={frozen_suffix_reason}"
                    )
                    return None
                representative_position = min(
                    position_cluster,
                    key=lambda sample: float(np.linalg.norm(
                        np.asarray(sample["point_base_m"], dtype=float)
                        - position_median
                    )),
                )
                log(
                    "stage=visible_checkpoint_position_only_fallback "
                    f"phase={phase} ok=True trigger={trigger!r} detail="
                    f"{position_detail.get('reason')}; "
                    f"path={frozen_suffix_reason}"
                )
                return {
                    "ok": True,
                    "reason": (
                        f"{position_detail.get('reason')}; close-range shape "
                        f"rejected ({trigger}); retained registered multi-view "
                        "geometry and reused the originally validated terminal "
                        f"path ({frozen_suffix_reason})"
                    ),
                    "body_consensus": position_detail,
                    "endpoint_m": frozen_checkpoint_endpoint.copy(),
                    "waypoints": list(frozen_checkpoint_suffix),
                    "trajectory_reason": frozen_suffix_reason,
                    "clearance": {
                        "checkpoint_position_only_fallback": True,
                        "reason": frozen_suffix_reason,
                    },
                    "geometry": frozen_checkpoint_geometry,
                    "geometry_summary": frozen_checkpoint_geometry_summary,
                    "jaw_plan": dict(frozen_checkpoint_jaw_plan),
                    "feature": representative_position["feature"],
                    "expected_feature_cam":
                        frozen_checkpoint_expected_feature.copy(),
                    "feature_offset_cam":
                        frozen_checkpoint_feature_offset.copy(),
                    "requested_cam_mm":
                        frozen_checkpoint_requested_cam.copy(),
                    "section_locked": selected_grasp_section_locked,
                    "axis_fraction": selected_grasp_axis_fraction,
                    "checkpoint_collision_aware_grasp": None,
                    "frame": representative_position["frame"],
                    "position_only_fallback": True,
                }

            initial_samples = 3
            maximum_samples = 6
            for sample_index in range(1, initial_samples + 1):
                collect_sample(sample_index, initial_samples)
            body_cam, body_detail, planning_samples = \
                _checkpoint_section_consensus(
                    samples, fresh_samples,
                    section_locked=selected_grasp_section_locked,
                    min_frames=2, max_spread_mm=4.0,
                )
            for sample_index in range(initial_samples + 1, maximum_samples + 1):
                if body_cam is not None:
                    break
                log("stage=visible_checkpoint_consensus status=resampling "
                    f"phase={phase} next_frame={sample_index}/{maximum_samples} "
                    f"detail={body_detail.get('reason')}; arm stationary")
                collect_sample(sample_index, maximum_samples)
                body_cam, body_detail, planning_samples = \
                    _checkpoint_section_consensus(
                        samples, fresh_samples,
                        section_locked=selected_grasp_section_locked,
                        min_frames=2, max_spread_mm=4.0,
                    )
            if body_cam is None:
                fallback = frozen_position_fallback(
                    str(body_detail.get("reason"))
                )
                if fallback is not None:
                    return fallback
                return {
                    "ok": False,
                    "reason": body_detail.get("reason"),
                    "failures": failures,
                }
            section_reselected = bool(body_detail.get("section_reselected"))
            if section_reselected:
                log(
                    "stage=visible_checkpoint_section_reselection "
                    f"phase={phase} status=accepted detail="
                    f"{body_detail.get('reason')}; arm stationary"
                )
            representative = min(
                planning_samples,
                key=lambda sample: float(np.linalg.norm(
                    np.asarray(sample["body_cam_mm"], dtype=float) - body_cam
                )),
            )
            new_geometry = representative["geometry"]
            new_body = representative["body"]
            new_feature = representative["feature"]
            checkpoint_position, checkpoint_position_detail, _ = \
                _checkpoint_target_position_consensus(
                    target_position_samples, frozen_checkpoint_anchor,
                    min_frames=2,
                )
            metric_width_authoritative, metric_width_reason = \
                _checkpoint_metric_width_is_authoritative(
                    aligned_3d_width_detail, full_object_box_3d,
                    float(frozen_checkpoint_jaw_plan.get(
                        "max_aperture_mm", 0.0
                    )),
                    checkpoint_position,
                )
            if metric_width_authoritative:
                new_geometry = _geometry_with_3d_jaw_width(
                    new_geometry, aligned_3d_width_detail
                )
                new_body = new_geometry.grasp_detection
                if new_body is None:
                    return {"ok": False,
                            "reason": "aligned 3D grasp body disappeared"}
                log(
                    "stage=visible_checkpoint_metric_width "
                    f"phase={phase} authoritative=true detail="
                    f"{metric_width_reason}; target="
                    f"{checkpoint_position_detail.get('reason')}"
                )
            elif aligned_3d_width_detail is not None:
                log(
                    "stage=visible_checkpoint_metric_width "
                    f"phase={phase} authoritative=false detail="
                    f"{metric_width_reason}"
                )
            new_plan = _plan_aperture(new_body)
            new_fit_ok, new_fit = (
                _validate_object_gripper_fit(
                    new_geometry, new_plan, full_object_box_3d
                )
                if new_plan.get("ok") else
                (False, {"reason": new_plan.get("reason")})
            )
            if not new_plan.get("ok") or not new_fit_ok:
                fit_reason = f"fresh body no longer fits: " \
                    f"{new_fit.get('reason')}"
                fallback = frozen_position_fallback(fit_reason)
                if fallback is not None:
                    return fallback
                return {
                    "ok": False,
                    "reason": fit_reason,
                }
            current_xyz = np.asarray(A.ee_pos(), dtype=float)
            new_endpoint = _endpoint_from_camera_body(
                current_xyz, body_cam, requested_cam, J
            )
            if new_endpoint is None:
                return {"ok": False,
                        "reason": "camera/body endpoint mapping is singular"}
            new_waypoints, new_reason, clearance = \
                _plan_clearance_axial_trajectory(
                    new_endpoint, J, approach_rpy,
                    log=log, phase=f"checkpoint_{phase}",
                )
            checkpoint_alternate = None
            if new_waypoints is None:
                # A global route to this visible checkpoint can be valid even
                # when the freshly measured terminal jaw entry is blocked.
                # Reuse the generic measured-section/jaw-slot search here,
                # where vision is most accurate.  This preserves every
                # endpoint, IK, continuity and collision gate; it does not
                # delete the blocking scene box or weaken inflation.
                log(
                    "stage=checkpoint_grasp_pose_search status=starting "
                    f"phase={phase} trigger=terminal_path_blocked "
                    f"detail={new_reason}"
                )
                alternate = _plan_collision_aware_grasp_pose(
                    new_geometry, representative["frame"], J,
                    requested_cam, approach_rpy, new_plan,
                    full_object_box_3d, log,
                )
                if not alternate.get("ok"):
                    log(
                        "stage=checkpoint_grasp_pose_search status=rejected "
                        f"phase={phase} detail={alternate.get('reason')}"
                    )
                    return {
                        "ok": False,
                        "reason": (
                            f"fresh final path is unsafe: {new_reason}; "
                            "checkpoint collision-aware grasp search: "
                            f"{alternate.get('reason')}"
                        ),
                        "body_consensus": body_detail,
                        "endpoint_m": new_endpoint.round(4).tolist(),
                        "checkpoint_grasp_search": alternate,
                    }
                new_geometry = alternate["geometry"]
                new_body = alternate["body"]
                new_plan = alternate["jaw_plan"]
                new_endpoint = np.asarray(alternate["endpoint_m"], dtype=float)
                new_waypoints = alternate["waypoints"]
                new_reason = str(alternate["trajectory_reason"])
                clearance = alternate["clearance"]
                selected_target_cam = np.asarray(
                    alternate["requested_cam_mm"], dtype=float
                )
                section_locked = bool(
                    (selected_grasp_section_locked and not section_reselected)
                    or alternate["body_name"] != "selected"
                )
                selected_axis_fraction = alternate.get("axis_fraction")
                if selected_axis_fraction is None:
                    selected_axis_fraction = selected_grasp_axis_fraction
                checkpoint_alternate = {
                    "reason": alternate["reason"],
                    "body": alternate["body_name"],
                    "axis_fraction": selected_axis_fraction,
                    "jaw_slot_offset_mm": alternate["slot_offset_mm"],
                    "candidates_tested": alternate["attempts"],
                }
                log(
                    "stage=checkpoint_grasp_pose_search status=accepted "
                    f"phase={phase} body={alternate['body_name']} "
                    f"axis_fraction={selected_axis_fraction} "
                    f"jaw_slot_offset_mm={alternate['slot_offset_mm']:+.1f}"
                )
            else:
                selected_target_cam = np.asarray(requested_cam, dtype=float)
                section_locked = bool(
                    selected_grasp_section_locked and not section_reselected
                )
                selected_axis_fraction = _selected_axis_fraction(new_geometry)
            new_expected_feature, new_feature_offset = \
                _feature_target_for_grasp_target(
                    selected_target_cam, new_feature, new_body
                )
            return {
                "ok": True,
                "reason": (
                    f"{body_detail.get('reason')}; fresh "
                    f"{(len(new_waypoints.get('joint_path', [])) if isinstance(new_waypoints, dict) else len(new_waypoints))}-"
                    "waypoint final path passed IK/collision checks"
                    + ("; checkpoint selected a collision-free measured "
                       "grasp pose" if checkpoint_alternate else "")
                ),
                "body_consensus": body_detail,
                "endpoint_m": new_endpoint,
                "waypoints": new_waypoints,
                "trajectory_reason": new_reason,
                "clearance": clearance,
                "geometry": new_geometry,
                "geometry_summary": new_geometry.summary(),
                "jaw_plan": new_plan,
                "feature": new_feature,
                "expected_feature_cam": new_expected_feature,
                "feature_offset_cam": new_feature_offset,
                "requested_cam_mm": selected_target_cam,
                "section_locked": section_locked,
                "axis_fraction": selected_axis_fraction,
                "checkpoint_collision_aware_grasp": checkpoint_alternate,
                "frame": representative["frame"],
            }

        checkpoint_plan = replan_from_visible_checkpoint("initial")
        steps.append((
            (
                "fixed_scene_macro_then_visible_checkpoint_replan"
                if FIXED_SCENE_TRAJECTORY else
                "visible_checkpoint_body_consensus_and_replan"
            ),
            bool(checkpoint_plan.get("ok")),
            str(checkpoint_plan.get("reason")),
        ))
        log(
            f"stage={'fixed_scene_checkpoint_replan' if FIXED_SCENE_TRAJECTORY else 'visible_checkpoint_replan'} "
            f"phase=initial ok={checkpoint_plan.get('ok')} "
            f"detail={checkpoint_plan.get('reason')}"
        )
        try:
            reacquire_retries = int(os.getenv(
                "REBOT_VISIBLE_CHECKPOINT_REACQUIRE_RETRIES", "1"
            ))
        except (TypeError, ValueError):
            reacquire_retries = 1
        reacquire_retries = int(np.clip(reacquire_retries, 0, 2))
        for retry_index in range(1, reacquire_retries + 1):
            if checkpoint_plan.get("ok"):
                break
            log(
                "stage=visible_checkpoint_reacquire "
                f"attempt={retry_index}/{reacquire_retries} status=planning "
                f"trigger={checkpoint_plan.get('reason')}; arm stationary"
            )
            retreat, retreat_reason = _plan_checkpoint_reacquire_retreat(
                checkpoint_prefix, checkpoint_start_xyz, J, approach_rpy
            )
            steps.append((
                f"prevalidate_visible_checkpoint_reacquire_{retry_index}",
                retreat is not None, retreat_reason,
            ))
            if retreat is None:
                log(
                    "stage=visible_checkpoint_reacquire "
                    f"attempt={retry_index}/{reacquire_retries} "
                    f"status=refused detail={retreat_reason}"
                )
                break
            log(
                "stage=visible_checkpoint_reacquire "
                f"attempt={retry_index}/{reacquire_retries} "
                f"status=retreating detail={retreat_reason}"
            )
            retreat_ok, retreat_execution = \
                _execute_fixed_approach_trajectory(
                    retreat, log, fixed_rpy=approach_rpy
                )
            steps.append((
                f"execute_visible_checkpoint_reacquire_{retry_index}",
                retreat_ok, retreat_execution,
            ))
            if not retreat_ok:
                log(
                    "stage=visible_checkpoint_reacquire "
                    f"attempt={retry_index}/{reacquire_retries} "
                    f"status=failed detail={retreat_execution}"
                )
                break
            _settle(
                eye, target_cam=requested_cam,
                note=f"checkpoint reacquisition retreat {retry_index}",
            )
            checkpoint_plan = replan_from_visible_checkpoint(
                f"reacquire_{retry_index}"
            )
            steps.append((
                f"visible_checkpoint_reacquire_and_replan_{retry_index}",
                bool(checkpoint_plan.get("ok")),
                str(checkpoint_plan.get("reason")),
            ))
            log(
                "stage=visible_checkpoint_reacquire "
                f"attempt={retry_index}/{reacquire_retries} "
                f"status={'accepted' if checkpoint_plan.get('ok') else 'not_found'} "
                f"detail={checkpoint_plan.get('reason')}"
            )
        if not checkpoint_plan.get("ok"):
            return recover(
                "visible checkpoint could not produce a safe fresh final "
                f"trajectory: {checkpoint_plan.get('reason')}"
            )

        # Reacquisition may have backed away from the first checkpoint. Keep
        # the actual viewpoint that produced the accepted final plan for any
        # later bounded endpoint retry and for diagnostics.
        checkpoint_xyz = np.asarray(A.ee_pos(), dtype=float)
        endpoint = np.asarray(checkpoint_plan["endpoint_m"], dtype=float)
        waypoints = checkpoint_plan["waypoints"]
        trajectory_reason = str(checkpoint_plan["trajectory_reason"])
        plan_geometry = checkpoint_plan["geometry"]
        requested_cam = np.asarray(
            checkpoint_plan.get("requested_cam_mm", requested_cam), dtype=float
        )
        selected_grasp_section_locked = bool(
            checkpoint_plan.get(
                "section_locked", selected_grasp_section_locked
            )
        )
        checkpoint_axis_fraction = checkpoint_plan.get(
            "axis_fraction", _selected_axis_fraction(plan_geometry)
        )
        if checkpoint_axis_fraction is not None:
            selected_grasp_axis_fraction = checkpoint_axis_fraction
        grasp_geometry_summary = checkpoint_plan["geometry_summary"]
        payload_geometry = replace(
            checkpoint_plan["geometry"], object_box_3d=full_object_box_3d
        )
        grasp_geometry_summary = payload_geometry.summary()
        jaw_plan = checkpoint_plan["jaw_plan"]
        expected_feature_cam = np.asarray(
            checkpoint_plan["expected_feature_cam"], dtype=float
        )
        feature_to_grasp_cam = np.asarray(
            checkpoint_plan["feature_offset_cam"], dtype=float
        )
        final_residual_mm = 0.0
        reachability = {
            **reachability,
            "visible_checkpoint_replanned": True,
            "visible_checkpoint_m": checkpoint_xyz.round(4).tolist(),
            "predicted_endpoint_m": endpoint.round(4).tolist(),
            "residual_to_requested_target_mm": 0.0,
        }
        checkpoint_collision_grasp = checkpoint_plan.get(
            "checkpoint_collision_aware_grasp"
        )
        if checkpoint_collision_grasp:
            reachability["checkpoint_collision_aware_grasp"] = \
                checkpoint_collision_grasp
            steps.append((
                "checkpoint_collision_aware_grasp_pose", True,
                str(checkpoint_collision_grasp.get("reason")),
            ))

        if isinstance(waypoints, dict):
            # The target was freshly observed at the visible checkpoint. Open
            # before the certified arbitrary route; no camera observation is
            # allowed to mutate that route once execution begins.
            closed_leg, final_leg = [], waypoints
        else:
            open_index = 0
            for index, waypoint in enumerate(waypoints):
                if float(np.linalg.norm(endpoint - np.asarray(waypoint))) <= \
                        FIXED_PICK_OPEN_BEFORE_END_M:
                    open_index = index
                    break
            else:
                open_index = max(0, len(waypoints) - 1)
            closed_leg, final_leg = waypoints[:open_index], waypoints[open_index:]
        if closed_leg:
            closed_ok, closed_reason = _execute_fixed_approach_trajectory(
                closed_leg, log, fixed_rpy=approach_rpy
            )
            steps.append(("execute_replanned_before_open",
                          closed_ok, closed_reason))
            if not closed_ok:
                return recover(
                    f"replanned pre-open trajectory failed: {closed_reason}"
                )

        preopen_rad = float(jaw_plan["preopen_rad"])
        log(f"stage=gripper_preopen target_mm={jaw_plan['preopen_mm']:.1f} "
            f"target_deg={math.degrees(preopen_rad):.1f}")
        opened = _gripper("open", preopen_rad)
        steps.append(("position_open", bool(opened.get("ok")),
                      str(opened.get("reason", "commanded"))))
        if not opened.get("ok"):
            return recover(f"gripper pre-open failed: {opened.get('reason')}")

        final_start = np.asarray(A.ee_pos(), dtype=float)
        final_distance = float(np.linalg.norm(endpoint - final_start))
        log("stage=execute_replanned_final_approach "
            f"distance_mm={final_distance * 1000:.1f} "
            f"waypoints="
            f"{(len(final_leg.get('joint_path', [])) if isinstance(final_leg, dict) else len(final_leg))} "
            "camera_steering=disabled")
        ok, reason = _execute_fixed_approach_trajectory(
            final_leg, log, fixed_rpy=approach_rpy
        )
        steps.append(("final_replanned_approach", ok, reason))
        if not ok:
            return recover(f"final replanned approach failed: {reason}")

        grasp_det, grasp_frame, grasp_why = _observe(eye, object_name)
        if grasp_det is None and DYNAMIC_GRASP_ENABLED:
            try:
                slot_frame = eye.frame()
                slot_det, slot_why = _jaw_slot_depth_occupancy_detection(
                    slot_frame, requested_cam, jaw_plan, plan_geometry,
                    full_object_box_3d,
                )
            except Exception as exc:  # noqa: BLE001
                slot_det, slot_frame = None, None
                slot_why = f"jaw-slot RGB-D read failed: {exc}"
            if slot_det is not None and slot_frame is not None:
                grasp_det, grasp_frame = slot_det, slot_frame
                grasp_why = slot_why
                log(
                    "stage=endpoint_jaw_slot_depth_fallback ok=True "
                    f"detail={slot_why}"
                )
            else:
                log(
                    "stage=endpoint_jaw_slot_depth_fallback ok=False "
                    f"detail={slot_why}"
                )
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
                f"complete {(len(waypoints.get('joint_path', [])) if isinstance(waypoints, dict) else len(waypoints))}-waypoint approach was prevalidated "
                f"from a visible object observation",
            ))
            log("stage=grasp_target_gate endpoint_occluded=true "
                f"predicted_residual_mm={final_residual_mm:.1f} "
                "basis=preplanned_calibrated_open_jaw_slot")
            # A fixed open-loop pick may use the calibrated slot after a fully
            # prevalidated visible approach. Dynamic image-controlled closure
            # cannot: it needs fresh between-jaw geometry. Force the existing
            # bounded checkpoint retreat/reacquisition path instead of waiting
            # until the width gate to abort in place.
            grasp_error = (
                math.inf if DYNAMIC_GRASP_ENABLED else final_residual_mm
            )
            before_lift_cam = expected_feature_cam.copy()
        else:
            endpoint_expected = (
                requested_cam
                if (getattr(grasp_det, "info", {}) or {}).get(
                    "endpoint_width_basis"
                ) == "registered_shape_plus_slot_occupancy"
                else expected_feature_cam
            )
            grasp_error = float(np.linalg.norm(
                np.asarray(grasp_det.p_cam_mm, dtype=float)
                - endpoint_expected
            ))

        # The long approach remains fixed and completely prevalidated. At its
        # endpoint a visible object may request a separately bounded correction;
        # this avoids throwing away an otherwise successful approach without
        # turning camera detections into unconstrained trajectory steering.
        corrections: list[dict[str, Any]] = []
        retry_envelope_mm = _endpoint_retry_envelope_mm(
            object_visible=grasp_det is not None
        )
        retry_gate = _endpoint_retry_eligibility(
            grasp_error, object_visible=grasp_det is not None
        )
        retry_eligible = bool(retry_gate["eligible"])
        steps.append(("grasp_target_gate",
                      retry_eligible,
                      f"error {grasp_error:.1f}mm; correction required "
                      f"{retry_gate['required_correction_mm']:.1f}mm after "
                      f"the {retry_gate['residual_tolerance_mm']:.1f}mm "
                      "endpoint measurement tolerance; bounded correction "
                      f"capacity {retry_envelope_mm:.1f}mm; compared the "
                      "identifier feature against its planned offset from the "
                      "grasp region"))
        if not retry_eligible:
            # Fixed-scene mode freezes the macro trajectory, not recovery from
            # expected open-finger occlusion at its endpoint. Retreat only to
            # the already visited visible checkpoint, obtain a fresh stationary
            # target observation, and prevalidate a replacement short leg. This
            # is bounded replanning inside the existing grasp workflow—not a
            # new scene-level route—and it never permits blind jaw closure.
            max_visible_replans = int(np.clip(HYBRID_MAX_REPLANS, 0, 2))
            for replan_attempt in range(1, max_visible_replans + 1):
                log("stage=visible_checkpoint_retry "
                    f"attempt={replan_attempt}/{max_visible_replans} "
                    f"endpoint_error_mm={grasp_error:.1f} status=retreating")
                retreat_waypoints, retreat_reason = \
                    _plan_fixed_approach_trajectory(
                        checkpoint_xyz, fixed_rpy=approach_rpy
                    )
                steps.append((
                    f"prevalidate_visible_checkpoint_retreat_{replan_attempt}",
                    retreat_waypoints is not None, retreat_reason,
                ))
                if retreat_waypoints is None:
                    break
                retreated, retreat_reason = \
                    _execute_fixed_approach_trajectory(
                        retreat_waypoints, log, fixed_rpy=approach_rpy
                    )
                steps.append((
                    f"execute_visible_checkpoint_retreat_{replan_attempt}",
                    retreated, retreat_reason,
                ))
                if not retreated:
                    break
                _settle(
                    eye, target_cam=requested_cam,
                    note=f"visible checkpoint retry {replan_attempt}",
                )
                retry_plan = replan_from_visible_checkpoint(
                    f"retry_{replan_attempt}"
                )
                steps.append((
                    f"visible_checkpoint_replan_{replan_attempt}",
                    bool(retry_plan.get("ok")),
                    str(retry_plan.get("reason")),
                ))
                log("stage=visible_checkpoint_retry "
                    f"attempt={replan_attempt}/{max_visible_replans} "
                    f"status=planned ok={retry_plan.get('ok')} "
                    f"detail={retry_plan.get('reason')}")
                if not retry_plan.get("ok"):
                    break

                endpoint = np.asarray(retry_plan["endpoint_m"], dtype=float)
                waypoints = retry_plan["waypoints"]
                plan_geometry = retry_plan["geometry"]
                retry_axis_fraction = _selected_axis_fraction(plan_geometry)
                if retry_axis_fraction is not None:
                    selected_grasp_axis_fraction = retry_axis_fraction
                grasp_geometry_summary = retry_plan["geometry_summary"]
                jaw_plan = retry_plan["jaw_plan"]
                expected_feature_cam = np.asarray(
                    retry_plan["expected_feature_cam"], dtype=float
                )
                feature_to_grasp_cam = np.asarray(
                    retry_plan["feature_offset_cam"], dtype=float
                )
                retry_moved, retry_move_reason = \
                    _execute_fixed_approach_trajectory(
                        waypoints, log, fixed_rpy=approach_rpy
                    )
                steps.append((
                    f"execute_visible_checkpoint_replan_{replan_attempt}",
                    retry_moved, retry_move_reason,
                ))
                if not retry_moved:
                    break
                _settle(
                    eye, target_cam=expected_feature_cam,
                    note=f"replanned endpoint {replan_attempt}",
                )
                grasp_det, grasp_frame, grasp_why = _observe(
                    eye, object_name
                )
                if grasp_det is None and DYNAMIC_GRASP_ENABLED:
                    try:
                        slot_frame = eye.frame()
                        slot_det, slot_why = \
                            _jaw_slot_depth_occupancy_detection(
                                slot_frame, requested_cam, jaw_plan,
                                plan_geometry, full_object_box_3d,
                            )
                    except Exception as exc:  # noqa: BLE001
                        slot_det, slot_frame = None, None
                        slot_why = (
                            "retried jaw-slot RGB-D read failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    if slot_det is not None and slot_frame is not None:
                        grasp_det, grasp_frame = slot_det, slot_frame
                        grasp_why = slot_why
                        log(
                            "stage=endpoint_jaw_slot_depth_fallback "
                            f"phase=retry_{replan_attempt} ok=True "
                            f"detail={slot_why}"
                        )
                    else:
                        log(
                            "stage=endpoint_jaw_slot_depth_fallback "
                            f"phase=retry_{replan_attempt} ok=False "
                            f"detail={slot_why}"
                        )
                if grasp_frame is not None:
                    _show(
                        *grasp_frame,
                        tag=f"grasp_endpoint_replanned_{replan_attempt}",
                        det=grasp_det, target_cam=expected_feature_cam,
                        note="endpoint after visible-checkpoint replan",
                    )
                if grasp_det is None:
                    # The complete short path was freshly observed and
                    # prevalidated at the checkpoint. Endpoint occlusion by the
                    # already-open fingers is acceptable under the existing
                    # calibrated-slot policy.
                    if DYNAMIC_GRASP_ENABLED:
                        grasp_error = math.inf
                        retry_eligible = False
                        steps.append((
                            f"verify_visible_checkpoint_replan_"
                            f"{replan_attempt}",
                            False,
                            "dynamic closure still has no measurable object "
                            "between the open jaws after bounded retreat and "
                            "fresh replan",
                        ))
                        continue
                    grasp_error = 0.0
                    final_residual_mm = 0.0
                    steps.append((
                        f"verify_visible_checkpoint_replan_{replan_attempt}",
                        True,
                        "endpoint feature occluded after a fresh visible "
                        "checkpoint and completely prevalidated short path",
                    ))
                    retry_eligible = True
                    break
                retry_expected = (
                    requested_cam
                    if (getattr(grasp_det, "info", {}) or {}).get(
                        "endpoint_width_basis"
                    ) == "registered_shape_plus_slot_occupancy"
                    else expected_feature_cam
                )
                grasp_error = float(np.linalg.norm(
                    np.asarray(grasp_det.p_cam_mm, dtype=float)
                    - retry_expected
                ))
                retry_gate = _endpoint_retry_eligibility(
                    grasp_error, object_visible=True
                )
                retry_eligible = bool(retry_gate["eligible"])
                steps.append((
                    f"verify_visible_checkpoint_replan_{replan_attempt}",
                    retry_eligible,
                    f"endpoint error {grasp_error:.1f}mm; correction required "
                    f"{retry_gate['required_correction_mm']:.1f}mm; bounded "
                    f"correction capacity {retry_envelope_mm:.1f}mm",
                ))
                log("stage=visible_checkpoint_retry "
                    f"attempt={replan_attempt}/{max_visible_replans} "
                    f"status=verified endpoint_error_mm={grasp_error:.1f} "
                    f"ok={retry_eligible}")
                if retry_eligible:
                    break
            if not retry_eligible:
                return recover(
                    f"endpoint observation remains {grasp_error:.1f}mm from "
                    "the fixed grasp target after bounded visible-checkpoint "
                    f"replanning; outside the {retry_envelope_mm:.1f}mm "
                    "correction envelope",
                    grasp_error_mm=round(grasp_error, 1),
                    corrections=corrections,
                )

        if grasp_det is not None:
            before_lift_cam = np.asarray(grasp_det.p_cam_mm, dtype=float)

        # The semantic identifier (for example a marker cap) is useful only for
        # the broad visibility/envelope gate above. It is not necessarily the
        # physical section entering the fingers, so never steer fine alignment
        # from it. Record it as an advisory; the closure gate and every bounded
        # fine correction below use fresh multi-frame grasp-body geometry.
        if DYNAMIC_GRASP_ENABLED and grasp_det is not None:
            occupancy_only = (
                (getattr(grasp_det, "info", {}) or {}).get(
                    "endpoint_width_basis"
                ) == "registered_shape_plus_slot_occupancy"
            )
            advisory_target = (
                requested_cam if occupancy_only else expected_feature_cam
            )
            identifier_slot_ok, identifier_slot_detail = _jaw_slot_fit(
                grasp_det.p_cam_mm, advisory_target, jaw_plan,
                plan_geometry, full_object_box_3d,
            )
            steps.append((
                "dynamic_identifier_alignment_advisory", True,
                f"identifier_fit={identifier_slot_ok}; "
                f"{identifier_slot_detail.get('reason')}; final closure uses "
                "multi-frame grasp-body geometry",
            ))
            log(f"stage=dynamic_identifier_alignment_advisory "
                f"identifier_fit={identifier_slot_ok} "
                f"error_mm={grasp_error:.1f} "
                f"detail={identifier_slot_detail.get('reason')}")

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
            initial_width = float(jaw_plan["min_width_mm"])
            ready_min, ready_max, ready_widths = \
                _ready_cross_section_range_mm(plan_geometry, initial_width)
            sample_count = int(np.clip(
                DYNAMIC_WIDTH_CONSENSUS_FRAMES, 3, 7
            ))
            def measure_grasp_body(
                phase: str,
                seed_det: Any | None = None,
                seed_frame: Any | None = None,
            ) -> dict[str, Any]:
                """Fresh width and body-pose consensus with the jaws open."""
                endpoint_samples: list[dict[str, Any]] = []
                fresh_endpoint_samples: list[dict[str, Any]] = []
                identifier_samples: list[dict[str, Any]] = []
                current_det, current_frame = seed_det, seed_frame

                def usable_sample(
                    candidate_geometry: GG.GraspGeometry,
                    detection: Any, frame: Any,
                ) -> tuple[dict[str, Any] | None, str]:
                    # Measure the actual local section now visible between the
                    # open jaws. The far-view aligned OBB width already proved
                    # approach clearance, but must not overwrite this closer
                    # measurement; doing so also made min_width exceed the
                    # close-range silhouette's max_width and invalidated every
                    # otherwise good endpoint frame. Stable XYZ/width consensus,
                    # complete-object fit, and the jaw-slot gate remain below.
                    candidate_plan = (
                        _plan_aperture(candidate_geometry.grasp_detection)
                        if candidate_geometry.ok
                        and candidate_geometry.grasp_detection is not None
                        else None
                    )
                    candidate_fit_ok, candidate_fit = (
                        _validate_object_gripper_fit(
                            candidate_geometry, candidate_plan,
                            full_object_box_3d,
                        )
                        if candidate_plan is not None
                        and candidate_plan.get("ok") else
                        (False, {"reason": candidate_geometry.reason})
                    )
                    if not candidate_fit_ok or candidate_plan is None:
                        return None, str(candidate_fit.get("reason"))
                    return {
                        "width_mm": float(candidate_plan["min_width_mm"]),
                        "body_cam_mm": np.asarray(
                            candidate_geometry.grasp_detection.p_cam_mm,
                            dtype=float,
                        ).tolist(),
                        "plan": candidate_plan,
                        "geometry": candidate_geometry,
                        "detection": detection,
                        "frame": frame,
                    }, "ready"

                for sample_index in range(1, sample_count + 1):
                    if current_det is None or current_frame is None:
                        current_det, current_frame, sample_why = _observe(
                            eye, object_name
                        )
                        if current_det is None or current_frame is None:
                            try:
                                slot_frame = eye.frame()
                                slot_det, slot_why = \
                                    _jaw_slot_depth_occupancy_detection(
                                        slot_frame, requested_cam, jaw_plan,
                                        plan_geometry, full_object_box_3d,
                                    )
                            except Exception as exc:  # noqa: BLE001
                                slot_det, slot_frame = None, None
                                slot_why = f"jaw-slot RGB-D read failed: {exc}"
                            if slot_det is None or slot_frame is None:
                                log("stage=dynamic_endpoint_width_sample "
                                    f"phase={phase} frame={sample_index}/"
                                    f"{sample_count} ok=False detail="
                                    f"{sample_why}; slot_fallback={slot_why}")
                                continue
                            current_det, current_frame = slot_det, slot_frame
                            log(
                                "stage=dynamic_endpoint_slot_occupancy "
                                f"phase={phase} frame={sample_index}/"
                                f"{sample_count} ok=True detail={slot_why}"
                            )
                    occupancy_only = (
                        (getattr(current_det, "info", {}) or {}).get(
                            "endpoint_width_basis"
                        ) == "registered_shape_plus_slot_occupancy"
                    )
                    # Anonymous slot occupancy is a grasp-body observation,
                    # not the named identifier feature. Never feed it into the
                    # feature-to-body registration fallback.
                    if not occupancy_only:
                        identifier_samples.append({
                            "body_cam_mm": np.asarray(
                                current_det.p_cam_mm, dtype=float
                            ).tolist(),
                            "detection": current_det,
                            "frame": current_frame,
                        })
                    sample_geometry = _plan_scene_grasp_geometry(
                        current_frame, current_det, current_rpy=approach_rpy,
                        allow_orientation=False, object_name=object_name,
                    )
                    fresh_geometry = sample_geometry
                    fresh_sample, fresh_reason = usable_sample(
                        fresh_geometry, current_det, current_frame
                    )
                    if fresh_sample is not None:
                        fresh_endpoint_samples.append(fresh_sample)
                    selected_geometry = _fresh_geometry_for_selected_section(
                        fresh_geometry, current_frame, current_det,
                        feature_to_grasp_cam,
                        section_locked=selected_grasp_section_locked,
                        axis_fraction=selected_grasp_axis_fraction,
                    )
                    if selected_geometry is None:
                        log("stage=dynamic_endpoint_width_sample "
                            f"phase={phase} frame={sample_index}/{sample_count} "
                            "ok=False detail=selected measured body section "
                            "was not reacquired; fresh_geometry_retained="
                            f"{fresh_sample is not None}")
                        current_det, current_frame = None, None
                        continue
                    selected_sample, selected_reason = usable_sample(
                        selected_geometry, current_det, current_frame
                    )
                    if selected_sample is None:
                        log("stage=dynamic_endpoint_width_sample "
                            f"phase={phase} frame={sample_index}/{sample_count} "
                            f"ok=False detail={selected_reason}; "
                            f"fresh_detail={fresh_reason}")
                    else:
                        endpoint_samples.append(selected_sample)
                        log("stage=dynamic_endpoint_width_sample "
                            f"phase={phase} frame={sample_index}/{sample_count} "
                            f"ok=True width_mm="
                            f"{selected_sample['width_mm']:.2f}")
                    # Every consensus member must come from a distinct frame.
                    current_det, current_frame = None, None

                consensus = _endpoint_measurement_consensus(
                    endpoint_samples, fresh_endpoint_samples,
                    ready_min, ready_max,
                    section_locked=selected_grasp_section_locked,
                )
                selected = consensus["selected"]
                width_detail = consensus["width_detail"]
                body_cam_local = consensus["body_cam"]
                body_detail = consensus["body_detail"]
                identifier_cam, identifier_detail = \
                    _endpoint_body_pose_consensus(
                        identifier_samples, min_frames=2,
                        max_spread_mm=4.0,
                    )
                slot_ok = False
                slot_detail: dict[str, Any] = {
                    "reason": body_detail.get("reason")
                }
                if selected is not None and body_cam_local is not None:
                    slot_ok, slot_detail = _jaw_slot_fit(
                        body_cam_local, requested_cam, selected["plan"],
                        selected["geometry"], full_object_box_3d,
                    )
                return {
                    **consensus,
                    "selected": selected,
                    "width_detail": width_detail,
                    "body_cam": body_cam_local,
                    "body_detail": body_detail,
                    "identifier_cam": identifier_cam,
                    "identifier_detail": identifier_detail,
                    "identifier_samples": identifier_samples,
                    "slot_ok": slot_ok,
                    "slot_detail": slot_detail,
                }

            measurement = measure_grasp_body(
                "initial", grasp_det, grasp_frame
            )
            selected_endpoint = measurement["selected"]
            consensus_detail = measurement["width_detail"]
            endpoint_measure_ok = selected_endpoint is not None
            if measurement.get("section_reselected"):
                selected_grasp_section_locked = False
                selected_grasp_axis_fraction = _selected_axis_fraction(
                    selected_endpoint["geometry"]
                )
                log(
                    "stage=dynamic_endpoint_section_reselection "
                    "status=accepted basis=stable_fresh_close_range_geometry "
                    f"detail={measurement['width_detail'].get('reason')}"
                )
            steps.append((
                "dynamic_endpoint_width_consensus", endpoint_measure_ok,
                str(consensus_detail.get("reason")),
            ))
            log("stage=dynamic_endpoint_width_consensus phase=initial "
                f"ok={endpoint_measure_ok} detail={consensus_detail.get('reason')} "
                f"ready_widths_mm={[round(v, 2) for v in ready_widths]} "
                f"endpoint_widths_mm={consensus_detail.get('widths_mm')}")
            if not endpoint_measure_ok:
                fallback = _ready_geometry_endpoint_fallback(
                    measurement, jaw_plan, plan_geometry, ready_max,
                    full_object_box_3d,
                )
                fallback_ok = bool(fallback.get("ok"))
                steps.append((
                    "ready_geometry_jaw_occlusion_fallback", fallback_ok,
                    str(fallback.get("reason")),
                ))
                log("stage=ready_geometry_jaw_occlusion_fallback "
                    f"ok={fallback_ok} detail={fallback.get('reason')}")
                if not fallback_ok:
                    # The registered multi-view model is already the metric
                    # width authority. If close masks spill, infer only the
                    # known grasp-section position from a stable identifier
                    # and its previously measured rigid feature/body offset.
                    identifier_cam = measurement.get("identifier_cam")
                    identifier_detail = measurement.get(
                        "identifier_detail", {}
                    )
                    identifier_body_cam = None
                    if identifier_cam is not None:
                        identifier_body_cam = np.asarray(
                            identifier_cam, dtype=float
                        ) - np.asarray(feature_to_grasp_cam, dtype=float)
                    identifier_slot_ok = False
                    identifier_slot_detail: dict[str, Any] = {
                        "reason": identifier_detail.get("reason")
                    }
                    if identifier_body_cam is not None:
                        identifier_slot_ok, identifier_slot_detail = \
                            _jaw_slot_fit(
                                identifier_body_cam, requested_cam,
                                jaw_plan, plan_geometry, full_object_box_3d,
                            )
                    identifier_fallback_ok = bool(
                        identifier_cam is not None and identifier_slot_ok
                    )
                    steps.append((
                        "registered_geometry_identifier_alignment_fallback",
                        identifier_fallback_ok,
                        f"{identifier_detail.get('reason')}; "
                        f"{identifier_slot_detail.get('reason')}",
                    ))
                    log(
                        "stage=registered_geometry_identifier_alignment_fallback "
                        f"ok={identifier_fallback_ok} "
                        f"identity={identifier_detail.get('reason')} "
                        f"slot={identifier_slot_detail.get('reason')}"
                    )
                    if not identifier_fallback_ok:
                        return recover(
                            "between-jaws width did not form a stable, "
                            "ready-compatible multi-frame consensus: "
                            f"{consensus_detail.get('reason')}; width fallback "
                            f"refused: {fallback.get('reason')}; registered-model "
                            "identifier alignment fallback refused: "
                            f"{identifier_slot_detail.get('reason')}; refusing "
                            "to close"
                        )
                    identifier_sample = min(
                        measurement["identifier_samples"],
                        key=lambda sample: float(np.linalg.norm(
                            np.asarray(
                                sample["body_cam_mm"], dtype=float
                            ) - np.asarray(identifier_cam, dtype=float)
                        )),
                    )
                    selected_endpoint = {
                        "width_mm": float(jaw_plan["min_width_mm"]),
                        "body_cam_mm": np.asarray(
                            identifier_body_cam, dtype=float
                        ).tolist(),
                        "plan": dict(jaw_plan),
                        "geometry": plan_geometry,
                        "detection": identifier_sample["detection"],
                        "frame": identifier_sample["frame"],
                    }
                    measurement = {
                        **measurement,
                        "selected": selected_endpoint,
                        "body_cam": np.asarray(
                            identifier_body_cam, dtype=float
                        ),
                        "body_detail": identifier_detail,
                        "slot_ok": identifier_slot_ok,
                        "slot_detail": identifier_slot_detail,
                        "used_registered_geometry_identifier_fallback": True,
                    }
                    endpoint_measure_ok = True
                else:
                    selected_endpoint = fallback["selected"]
                    body_cam_fallback = fallback["body_cam"]
                    fallback_slot_ok, fallback_slot_detail = _jaw_slot_fit(
                        body_cam_fallback, requested_cam,
                        selected_endpoint["plan"],
                        selected_endpoint["geometry"], full_object_box_3d,
                    )
                    measurement = {
                        **measurement,
                        "selected": selected_endpoint,
                        "body_cam": body_cam_fallback,
                        "body_detail": fallback["body_detail"],
                        "slot_ok": fallback_slot_ok,
                        "slot_detail": fallback_slot_detail,
                        "used_ready_geometry_fallback": True,
                    }
                    endpoint_measure_ok = True

            # Width can form a valid cluster while a single stochastic mask
            # makes the corresponding body XYZ cluster unstable. Before using
            # a semantic-offset fallback or moving the arm, take one entirely
            # fresh stationary batch. This reuses the existing strict width,
            # identity, full-object, and 3 mm body-spread gates; it only gives
            # transient segmentation jitter another observation window.
            if measurement.get("selected") is not None \
                    and measurement.get("body_cam") is None:
                retry_measurement = measure_grasp_body(
                    "stationary_consensus_retry"
                )
                retry_ok = bool(
                    retry_measurement.get("selected") is not None
                    and retry_measurement.get("body_cam") is not None
                )
                retry_reason = (
                    f"width={retry_measurement.get('width_detail', {}).get('reason')}; "
                    f"body={retry_measurement.get('body_detail', {}).get('reason')}"
                )
                steps.append((
                    "dynamic_endpoint_stationary_consensus_retry",
                    retry_ok, retry_reason,
                ))
                log(
                    "stage=dynamic_endpoint_stationary_consensus_retry "
                    f"ok={retry_ok} detail={retry_reason}"
                )
                if retry_ok:
                    measurement = retry_measurement
                    selected_endpoint = measurement["selected"]
                    endpoint_measure_ok = True

            body_cam = measurement["body_cam"]
            body_consensus = measurement["body_detail"]
            body_slot_ok = bool(measurement["slot_ok"])
            body_slot_detail = measurement["slot_detail"]
            # The normal (non-collision-selected) grasp deliberately accepts a
            # better close-range cross-section than the distant scene view.
            # Freeze that newly accepted section now.  Otherwise every bounded
            # correction can select a different point along a long/tilted body,
            # invalidating the before/after motion comparison and the later
            # lift-hold baseline even when all individual frames are stable.
            if measurement.get(
                "used_registered_geometry_identifier_fallback"
            ):
                endpoint_offset = np.asarray(
                    feature_to_grasp_cam, dtype=float
                ).copy()
                endpoint_axis_fraction = selected_grasp_axis_fraction
                endpoint_lock_detail = {
                    "reason": (
                        "retained the registered multi-view grasp section and "
                        "its rigid feature/body offset after close-mask spill"
                    )
                }
                endpoint_section_locked = True
            else:
                endpoint_offset, endpoint_axis_fraction, endpoint_lock_detail = \
                    _endpoint_section_identity(selected_endpoint)
                endpoint_section_locked = endpoint_offset is not None
            steps.append((
                "lock_close_range_grasp_section", endpoint_section_locked,
                str(endpoint_lock_detail.get("reason")),
            ))
            log("stage=close_range_grasp_section_lock "
                f"ok={endpoint_section_locked} "
                f"detail={endpoint_lock_detail.get('reason')}")
            if not endpoint_section_locked:
                return recover(
                    "could not establish a stable close-range grasp-section "
                    "identity; refusing alignment or closure"
                )
            feature_to_grasp_cam = endpoint_offset
            selected_grasp_axis_fraction = endpoint_axis_fraction
            selected_grasp_section_locked = True

            # A narrow object can be correctly inside the open jaws while the
            # fingers split/occlude its close-range depth component.  In that
            # case width consensus can remain stable although reconstructed
            # body centres jump between frames.  Reuse only the locked rigid
            # semantic-feature/body transform and still require the unchanged
            # calibrated jaw-slot gate; never loosen the body spread limit.
            if body_cam is None and selected_endpoint is not None:
                locked_body_fallback = \
                    _locked_identifier_body_slot_fallback(
                        measurement.get("identifier_cam"),
                        measurement.get("identifier_detail", {}),
                        feature_to_grasp_cam, requested_cam,
                        selected_endpoint["plan"],
                        selected_endpoint["geometry"],
                        full_object_box_3d,
                    )
                fallback_ok = bool(locked_body_fallback.get("ok"))
                steps.append((
                    "locked_identifier_body_slot_fallback", fallback_ok,
                    str(locked_body_fallback.get("reason")),
                ))
                log("stage=locked_identifier_body_slot_fallback "
                    f"ok={fallback_ok} "
                    f"detail={locked_body_fallback.get('reason')}")
                if fallback_ok:
                    body_cam = np.asarray(
                        locked_body_fallback["body_cam"], dtype=float
                    )
                    body_consensus = {
                        **measurement.get("identifier_detail", {}),
                        "reason": locked_body_fallback.get("reason"),
                        "basis": "locked_identifier_to_body_transform",
                    }
                    body_slot_ok = True
                    body_slot_detail = locked_body_fallback["slot_detail"]
            log("stage=dynamic_grasp_body_slot_gate phase=initial "
                f"ok={body_slot_ok} consensus={body_consensus.get('reason')} "
                f"detail={body_slot_detail.get('reason')}")

            # Identifier boxes can describe a cap or label rather than the
            # section entering the fingers. If the stable physical body misses
            # the slot, use that body consensus for at most two independently
            # IK/collision-validated open-jaw corrections. Never loosen the
            # capture gate, and remeasure all frames after every move.
            body_correction_limit = int(np.clip(
                DYNAMIC_BODY_MAX_CORRECTIONS, 0, 2
            ))
            # Keep adaptation local to this endpoint. The persistent rigid
            # camera calibration is the prior for every new run and is never
            # modified by close-range segmentation feedback.
            body_alignment_J = np.asarray(J, dtype=float).copy()
            previous_violation = _jaw_slot_violation_mm(body_slot_detail)
            for body_attempt in range(1, body_correction_limit + 1):
                if body_slot_ok:
                    break
                if body_cam is None:
                    break
                # First seek the minimum motion that crosses the unchanged
                # jaw-slot boundary. Fall back to full centring only when no
                # smaller, fully validated correction exists.
                body_waypoints, body_correction = \
                    _plan_partial_slot_alignment_correction(
                        body_cam, requested_cam, body_alignment_J, approach_rpy,
                        selected_endpoint["plan"],
                        selected_endpoint["geometry"],
                        full_object_box_3d,
                    )
                log("stage=grasp_body_alignment_minimum "
                    f"attempt={body_attempt}/{body_correction_limit} "
                    f"ok={body_waypoints is not None} "
                    f"detail={body_correction.get('reason')}")
                if body_waypoints is None:
                    partial_correction = body_correction
                    body_waypoints, body_correction = \
                        _plan_endpoint_alignment_correction(
                            body_cam, requested_cam, body_alignment_J,
                            approach_rpy
                        )
                    body_correction = {
                        **body_correction,
                        "minimum_alternative": partial_correction,
                        "reason": (
                            f"{body_correction.get('reason')}; minimum-motion "
                            f"alternative unavailable: "
                            f"{partial_correction.get('reason')}"
                        ),
                    }
                planned = body_waypoints is not None
                steps.append((
                    f"prevalidate_grasp_body_alignment_{body_attempt}",
                    planned, str(body_correction.get("reason")),
                ))
                log("stage=grasp_body_alignment_plan "
                    f"attempt={body_attempt}/{body_correction_limit} "
                    f"ok={planned} violation_mm={previous_violation:.1f} "
                    f"camera_error_mm={body_correction.get('camera_error_mm')} "
                    f"base_correction_mm="
                    f"{body_correction.get('base_correction_mm')}")
                if body_waypoints is None:
                    return recover(
                        "stable grasp body is outside the jaw slot and its "
                        "bounded correction could not be safely planned: "
                        f"{body_correction.get('reason')}",
                        corrections=corrections,
                    )
                correction_start_xyz = np.asarray(A.ee_pos(), dtype=float)
                prior_selected_endpoint = selected_endpoint
                prior_body_cam = np.asarray(body_cam, dtype=float).copy()
                moved, move_reason = _execute_fixed_approach_trajectory(
                    body_waypoints, log, fixed_rpy=approach_rpy
                )
                body_correction.update({
                    "kind": "grasp_body",
                    "attempt": body_attempt,
                    "executed": moved,
                    "execution_reason": move_reason,
                })
                corrections.append(body_correction)
                steps.append((
                    f"execute_grasp_body_alignment_{body_attempt}",
                    moved, move_reason,
                ))
                if not moved:
                    return recover(
                        f"bounded grasp-body correction failed: {move_reason}",
                        corrections=corrections,
                    )
                _settle(
                    eye, target_cam=requested_cam,
                    note=f"grasp-body alignment attempt {body_attempt}",
                )
                correction_end_xyz = np.asarray(A.ee_pos(), dtype=float)
                measurement = measure_grasp_body(
                    f"body_correction_{body_attempt}"
                )
                selected_endpoint = measurement["selected"]
                if selected_endpoint is None \
                        and prior_selected_endpoint is not None \
                        and hybrid_identity_gate is not None \
                        and isinstance(full_object_box_3d, dict):
                    propagation = _propagate_locked_body_after_alignment(
                        prior_body_cam, body_alignment_J,
                        correction_start_xyz, correction_end_xyz,
                        body_correction.get("endpoint_m"), requested_cam,
                        prior_selected_endpoint["plan"],
                        prior_selected_endpoint["geometry"],
                        previous_violation,
                        identity_frames=len(
                            measurement.get("identifier_samples", [])
                        ),
                        object_box_3d=full_object_box_3d,
                    )
                    propagation_ok = bool(propagation.get("ok"))
                    steps.append((
                        f"grasp_body_motion_model_fallback_{body_attempt}",
                        propagation_ok, str(propagation.get("reason")),
                    ))
                    log(
                        "stage=grasp_body_motion_model_fallback "
                        f"attempt={body_attempt}/{body_correction_limit} "
                        f"ok={propagation_ok} "
                        f"detail={propagation.get('reason')}"
                    )
                    if propagation_ok:
                        predicted_body = np.asarray(
                            propagation["body_cam_mm"], dtype=float
                        )
                        propagated_detection = replace(
                            prior_selected_endpoint[
                                "geometry"
                            ].grasp_detection,
                            p_cam_mm=tuple(float(value)
                                           for value in predicted_body),
                        )
                        propagated_geometry = replace(
                            prior_selected_endpoint["geometry"],
                            grasp_detection=propagated_detection,
                            reason=(
                                prior_selected_endpoint[
                                    "geometry"
                                ].reason
                                + "; locked section propagated through "
                                  "measured open-jaw alignment motion"
                            ),
                        )
                        selected_endpoint = {
                            **prior_selected_endpoint,
                            "body_cam_mm": predicted_body.tolist(),
                            "geometry": propagated_geometry,
                        }
                        measurement = {
                            **measurement,
                            "selected": selected_endpoint,
                            "body_cam": predicted_body,
                            "body_detail": {
                                "reason": propagation.get("reason"),
                                "basis": "measured_tool_motion_propagation",
                                "stable_frames": len(measurement.get(
                                    "identifier_samples", []
                                )),
                            },
                            "slot_ok": propagation["slot_ok"],
                            "slot_detail": propagation["slot_detail"],
                            "width_detail": {
                                "ok": True,
                                "reason": (
                                    "retained the last stable endpoint width "
                                    "while its locked body point was propagated "
                                    "through measured tool motion"
                                ),
                                "widths_mm": [],
                            },
                            "used_motion_model_fallback": True,
                        }
                if measurement.get("section_reselected") \
                        and selected_endpoint is not None:
                    endpoint_offset, endpoint_axis_fraction, \
                        endpoint_lock_detail = _endpoint_section_identity(
                            selected_endpoint
                        )
                    if endpoint_offset is None:
                        return recover(
                            "fresh close-range grasp section was stable but "
                            "could not be assigned a physical identity after "
                            "alignment; refusing to close",
                            corrections=corrections,
                        )
                    feature_to_grasp_cam = endpoint_offset
                    selected_grasp_axis_fraction = endpoint_axis_fraction
                    selected_grasp_section_locked = True
                    log(
                        "stage=dynamic_endpoint_section_reselection "
                        f"phase=body_correction_{body_attempt} "
                        "status=accepted_and_relocked basis="
                        "stable_fresh_close_range_geometry detail="
                        f"{endpoint_lock_detail.get('reason')}"
                    )
                fresh_width_detail = measurement["width_detail"]
                steps.append((
                    f"dynamic_endpoint_width_consensus_after_body_alignment_"
                    f"{body_attempt}",
                    selected_endpoint is not None,
                    str(fresh_width_detail.get("reason")),
                ))
                log("stage=dynamic_endpoint_width_consensus "
                    f"phase=body_correction_{body_attempt} "
                    f"ok={selected_endpoint is not None} "
                    f"detail={fresh_width_detail.get('reason')} "
                    f"endpoint_widths_mm="
                    f"{fresh_width_detail.get('widths_mm')}")
                if selected_endpoint is None:
                    return recover(
                        "fresh between-jaws width consensus failed after the "
                        "bounded grasp-body correction: "
                        f"{fresh_width_detail.get('reason')}",
                        corrections=corrections,
                    )
                body_cam = measurement["body_cam"]
                body_consensus = measurement["body_detail"]
                body_slot_ok = bool(measurement["slot_ok"])
                body_slot_detail = measurement["slot_detail"]
                current_violation = _jaw_slot_violation_mm(body_slot_detail)
                improved = bool(
                    body_slot_ok
                    or current_violation < previous_violation - 0.5
                )
                steps.append((
                    f"verify_grasp_body_alignment_{body_attempt}", improved,
                    f"slot violation {previous_violation:.1f}->"
                    f"{current_violation:.1f}mm; "
                    f"{body_slot_detail.get('reason')}",
                ))
                log("stage=grasp_body_alignment_verify "
                    f"attempt={body_attempt}/{body_correction_limit} "
                    f"ok={improved} slot_fit={body_slot_ok} "
                    f"violation_before_mm={previous_violation:.1f} "
                    f"violation_after_mm={current_violation:.1f} "
                    f"detail={body_slot_detail.get('reason')}")
                if not improved:
                    return recover(
                        "bounded grasp-body correction did not improve the "
                        "physical jaw-slot fit; refusing to close",
                        corrections=corrections,
                    )
                # Only learn a response when another correction is required.
                # A successful final alignment needs no model update, while a
                # contradictory response must stop instead of repeating the
                # same misleading geometric prediction.
                if not body_slot_ok and body_attempt < body_correction_limit:
                    feedback_mapping = None
                    feedback_detail = {
                        "ok": False,
                        "reason": (
                            "fresh stable grasp-body position unavailable"
                        ),
                    }
                    if body_cam is not None:
                        feedback_mapping, feedback_detail = \
                            _update_alignment_jacobian_from_feedback(
                                body_alignment_J,
                                correction_end_xyz - correction_start_xyz,
                                prior_body_cam, body_cam,
                            )
                    feedback_ok = feedback_mapping is not None
                    body_correction[
                        "visual_feedback_update"
                    ] = feedback_detail
                    steps.append((
                        f"grasp_body_alignment_feedback_{body_attempt}",
                        feedback_ok, str(feedback_detail.get("reason")),
                    ))
                    log("stage=grasp_body_alignment_feedback "
                        f"attempt={body_attempt}/{body_correction_limit} "
                        f"ok={feedback_ok} "
                        f"predicted_shift_mm="
                        f"{feedback_detail.get('predicted_shift_mm')} "
                        f"observed_shift_mm="
                        f"{feedback_detail.get('observed_shift_mm')} "
                        f"detail={feedback_detail.get('reason')}")
                    if feedback_mapping is None:
                        return recover(
                            "the first bounded grasp-body correction improved "
                            "the slot fit, but its measured camera response "
                            "could not safely calibrate the remaining fine "
                            f"motion: {feedback_detail.get('reason')}",
                            corrections=corrections,
                        )
                    body_alignment_J = feedback_mapping
                previous_violation = current_violation

            steps.append((
                "dynamic_grasp_body_inside_jaw_slot", body_slot_ok,
                f"{body_consensus.get('reason')}; "
                f"{body_slot_detail.get('reason')}",
            ))
            log("stage=dynamic_grasp_body_slot_gate phase=final "
                f"ok={body_slot_ok} consensus={body_consensus.get('reason')} "
                f"detail={body_slot_detail.get('reason')}")
            if not body_slot_ok:
                return recover(
                    "multi-frame grasp-body geometry remains outside the "
                    "calibrated jaw capture region after bounded corrections: "
                    f"{body_slot_detail.get('reason')}",
                    corrections=corrections,
                )
            assert selected_endpoint is not None
            endpoint_plan = selected_endpoint["plan"]
            endpoint_geometry = selected_endpoint["geometry"]
            payload_geometry = replace(
                endpoint_geometry, object_box_3d=full_object_box_3d
            )
            grasp_geometry_summary = payload_geometry.summary()
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
        payload_box_context = None
        if SCENE_COLLISION_ENABLED:
            box_3d = payload_geometry.object_box_3d
            if box_3d is None:
                return recover(
                    "held-object collision envelope has no persistent 3D box; "
                    "refusing to lift"
                )
            try:
                box_sizes = np.asarray(
                    box_3d["conservative_size_m"], dtype=float
                )
                box_axes = np.asarray(box_3d["axes_base"], dtype=float).T
                box_center = np.asarray(box_3d["center_m"], dtype=float)
                major_index = int(np.argmax(box_sizes))
                payload_size = box_sizes.copy()
                ee_rotation = _rpy_rotation_matrix(A.ee_rpy())
                long_axis_ee = ee_rotation.T @ box_axes[:, major_index]
                payload_axes_ee = ee_rotation.T @ box_axes
                center_offset_ee_m = ee_rotation.T @ (
                    box_center - np.asarray(A.ee_pos(), dtype=float)
                )
                if float(np.linalg.norm(center_offset_ee_m)) > 0.20:
                    raise ValueError("3D box centre is over 200mm from the tool")
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                return recover(
                    "held-object 3D collision box is inconsistent: "
                    f"{exc}; refusing to lift"
                )
            payload_box_context = {
                "box_3d": box_3d,
                "collision_size_m": payload_size.tolist(),
                "center_offset_ee_m": center_offset_ee_m.tolist(),
                "long_axis_ee": long_axis_ee.tolist(),
                "full_axes_ee": payload_axes_ee.T.tolist(),
                "full_size_m": box_sizes.tolist(),
            }
            if PAYLOAD_COLLISION_ENABLED:
                payload_result = A.update_collision_payload_box(
                    payload_size, long_axis_ee, center_offset_ee_m,
                    axes_ee=payload_axes_ee,
                )
                steps.append(("attach_payload_collision_envelope",
                              payload_result.ok, payload_result.reason))
                log("stage=attach_payload_collision_envelope "
                    f"ok={payload_result.ok} shape=oriented_box "
                    f"size_mm={(payload_size * 1000.0).round(1).tolist()} "
                    f"source=unobstructed_multiframe_metric_rgbd "
                    f"center_offset_ee_mm="
                    f"{(center_offset_ee_m * 1000.0).round(1).tolist()} "
                    f"detail={payload_result.reason}")
                if not payload_result.ok:
                    return recover(
                        "held-object collision envelope could not be installed; "
                        "refusing to lift"
                    )
            else:
                cleared = A.update_collision_payload(0.0)
                steps.append((
                    "attach_payload_collision_envelope", cleared.ok,
                    "disabled by REBOT_PAYLOAD_COLLISION=0; collision checks "
                    "cover URDF arm/gripper links and the measured scene only",
                ))
                log("stage=collision_scope arm_and_gripper_only=true "
                    "payload_collision=false alternative_routes=true "
                    f"payload_reset_ok={cleared.ok}")
                if not cleared.ok:
                    return recover(
                        "could not clear the previous payload collision model; "
                        "refusing to plan with stale geometry"
                    )
        time.sleep(0.5)

        grasp_xyz = np.asarray(A.ee_pos(), dtype=float)
        retry_count = 0
        current_grasp_mm = float(jaw_plan["grasp_mm"])
        current_grasp_rad = float(grasp_rad)

        def observe_closed_body(
            phase: str,
        ) -> tuple[np.ndarray | None, Any | None, Any | None, dict[str, Any]]:
            """Track the physical grasp section, not its semantic identifier."""
            samples: list[dict[str, Any]] = []
            latest_det = None
            latest_frame = None
            reasons: list[str] = []
            feature_samples: list[dict[str, Any]] = []
            for sample_index in range(1, 4):
                sample_det, sample_frame, sample_why = _observe(
                    eye, object_name
                )
                if sample_det is None or sample_frame is None:
                    reasons.append(str(sample_why))
                    log("stage=dynamic_hold_body_sample "
                        f"phase={phase} frame={sample_index}/3 ok=False "
                        f"detail={sample_why}")
                    continue
                feature_samples.append({
                    "body_cam_mm": np.asarray(
                        sample_det.p_cam_mm, dtype=float
                    ).tolist()
                })
                sample_geometry = _plan_scene_grasp_geometry(
                    sample_frame, sample_det,
                    current_rpy=tuple(float(v) for v in A.ee_rpy()),
                    allow_orientation=False, object_name=object_name,
                )
                sample_geometry = _fresh_geometry_for_selected_section(
                    sample_geometry, sample_frame, sample_det,
                    feature_to_grasp_cam,
                    section_locked=selected_grasp_section_locked,
                    axis_fraction=selected_grasp_axis_fraction,
                )
                if sample_geometry is None:
                    reasons.append(
                        "selected measured body section was not reacquired"
                    )
                    log("stage=dynamic_hold_body_sample "
                        f"phase={phase} frame={sample_index}/3 ok=False "
                        "detail=selected measured body section was not "
                        "reacquired")
                    continue
                if not sample_geometry.ok \
                        or sample_geometry.grasp_detection is None:
                    reasons.append(sample_geometry.reason)
                    log("stage=dynamic_hold_body_sample "
                        f"phase={phase} frame={sample_index}/3 ok=False "
                        f"detail={sample_geometry.reason}")
                    continue
                latest_det = sample_geometry.grasp_detection
                latest_frame = sample_frame
                samples.append({
                    "body_cam_mm": np.asarray(
                        latest_det.p_cam_mm, dtype=float
                    ).tolist()
                })
                log("stage=dynamic_hold_body_sample "
                    f"phase={phase} frame={sample_index}/3 ok=True "
                    f"xyz_mm={np.round(latest_det.p_cam_mm, 1).tolist()}")
            body, detail = _endpoint_body_pose_consensus(
                samples, min_frames=2, max_spread_mm=4.0
            )
            feature_cam, feature_detail = _endpoint_body_pose_consensus(
                feature_samples, min_frames=2, max_spread_mm=4.0
            )
            detail["feature_cam_mm"] = (
                None if feature_cam is None else feature_cam.tolist()
            )
            detail["feature_reason"] = feature_detail.get("reason")
            if body is None and reasons:
                detail["observations"] = reasons
            return body, latest_det, latest_frame, detail

        while True:
            # Closing can center or settle an object inside the jaws. Establish
            # the hold baseline only after that seating motion; otherwise a
            # successful grasp can look like lift slippage.
            post_close_body, _, post_close_frame, post_close_detail = \
                observe_closed_body(f"post_close_{retry_count + 1}")
            post_close_fallback = None
            # The newly closed fingers commonly hide the exact cross-section
            # that authorized closure. Do not abort before the visual
            # micro-lift merely because that expected occlusion leaves only
            # one physical-section sample. The semantic feature/body offset
            # was locked while both were visible immediately before closing;
            # recover that *same* section and require it to remain inside the
            # calibrated jaw slot. The subsequent micro-lift still has to
            # prove rigid motion with the wrist before the grasp is accepted.
            if post_close_body is None:
                post_close_fallback, fallback_detail = \
                    _locked_body_from_feature(
                        post_close_detail.get("feature_cam_mm"),
                        feature_to_grasp_cam,
                    )
                fallback_slot_ok = False
                fallback_slot_detail = {"reason": fallback_detail["reason"]}
                if post_close_fallback is not None:
                    fallback_slot_ok, fallback_slot_detail = _jaw_slot_fit(
                        post_close_fallback, requested_cam, jaw_plan,
                        endpoint_geometry, full_object_box_3d,
                    )
                steps.append((
                    "dynamic_post_close_semantic_body_fallback",
                    fallback_slot_ok,
                    f"{fallback_detail.get('reason')}; "
                    f"{fallback_slot_detail.get('reason')}",
                ))
                log(
                    "stage=dynamic_post_close_semantic_body_fallback "
                    f"attempt={retry_count + 1} "
                    f"ok={fallback_slot_ok} "
                    f"detail={fallback_detail.get('reason')}; "
                    f"slot={fallback_slot_detail.get('reason')}"
                )
                if fallback_slot_ok:
                    post_close_body = post_close_fallback
                    post_close_detail["reason"] = (
                        f"{fallback_detail.get('reason')}; stable semantic "
                        "fallback remains inside the calibrated jaw slot"
                    )
            hold_baseline = _post_close_hold_baseline(
                post_close_body, post_close_detail
            )
            baseline_ok = bool(hold_baseline["ok"])
            steps.append((
                "dynamic_post_close_body_baseline", baseline_ok,
                str(hold_baseline["reason"]),
            ))
            log("stage=dynamic_post_close_body_baseline "
                f"attempt={retry_count + 1} ok={baseline_ok} "
                f"basis={hold_baseline['basis']} "
                f"detail={hold_baseline['reason']}")
            if not baseline_ok:
                return recover(
                    "could not establish a stable identity-gated visual "
                    "baseline after closing; refusing an unverifiable lift"
                )
            before_lift_cam = hold_baseline["body_cam_mm"]
            before_lift_feature_cam = hold_baseline["feature_cam_mm"]
            pre_lift_xyz = np.asarray(A.ee_pos(), dtype=float)
            lift_distance = (
                float(np.clip(DYNAMIC_MICRO_LIFT_M, 0.010, 0.020))
                if DYNAMIC_GRASP_ENABLED else FIXED_PICK_LIFT_M
            )
            escape_detail: dict[str, Any] = {
                "ok": True, "active": False,
                "reason": (
                    "payload collision disabled; URDF arm/gripper scene "
                    "collision remains active"
                    if SCENE_COLLISION_ENABLED else
                    "scene collision model disabled"
                ),
            }
            if SCENE_COLLISION_ENABLED and PAYLOAD_COLLISION_ENABLED:
                escape_detail = A.begin_payload_vertical_escape()
                steps.append((
                    "payload_vertical_escape_gate",
                    bool(escape_detail.get("ok")),
                    str(escape_detail.get("reason")),
                ))
                log(
                    "stage=payload_vertical_escape_gate "
                    f"ok={escape_detail.get('ok')} "
                    f"active={escape_detail.get('active')} "
                    f"initial_obstacles="
                    f"{escape_detail.get('allowed_initial_obstacles', [])} "
                    f"exact_initial="
                    f"{escape_detail.get('exact_initial_obstacles', [])} "
                    f"reserved_near="
                    f"{escape_detail.get('reserved_near_obstacles', [])} "
                    f"overlap_boxes="
                    f"{len(escape_detail.get('allowed_initial_obstacles', []))} "
                    f"map_components="
                    f"{(escape_detail.get('cluster') or {}).get('map_component_count', 0)} "
                    f"minimum_vertical_clearance_mm="
                    f"{float(escape_detail.get('minimum_vertical_clearance_m', 0.0)) * 1000.0:.1f} "
                    f"reservation_xy_mm="
                    f"{float(escape_detail.get('reservation_xy_shell_m', 0.0)) * 1000.0:.1f} "
                    f"detail={escape_detail.get('reason')}"
                )
                if not escape_detail.get("ok"):
                    return recover(
                        "post-grasp support escape is unsafe: "
                        f"{escape_detail.get('reason')}"
                    )

            # Ordinarily retain the small visual micro-lift. If the payload is
            # born inside an inflated support column, derive the required +Z
            # clearance from the measured payload lower face and the tops of the
            # exact overlapping boxes. This replaces the old arbitrary 40 mm
            # ceiling. The candidate is still workspace/IK/path checked and its
            # endpoint must be strictly clear with the exception disabled.
            candidate_distances = [lift_distance]
            candidate_failures = []
            if escape_detail.get("active"):
                required_escape = max(
                    lift_distance,
                    float(escape_detail.get(
                        "minimum_vertical_clearance_m", 0.0
                    )),
                )
                available_escape = max(
                    0.0,
                    A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M
                    - float(np.asarray(A.ee_pos(), dtype=float)[2]),
                )
                if required_escape > available_escape + 1e-6:
                    candidate_distances = []
                    candidate_failures.append(
                        f"required vertical clearance "
                        f"{required_escape * 1000.0:.1f}mm exceeds the "
                        f"{available_escape * 1000.0:.1f}mm bounded workspace"
                    )
                else:
                    # Do not waste IK/collision planning time on endpoints that
                    # are mathematically below the highest initial overlap.
                    # Check the exact clearance and one bounded 5 mm depth-noise
                    # allowance; both remain inside the Cartesian workspace.
                    candidate_distances = sorted(set(
                        round(float(candidate), 6)
                        for candidate in (
                            required_escape,
                            min(required_escape + 0.005, available_escape),
                        )
                    ))
                log(
                    "stage=payload_vertical_escape_candidates "
                    f"required_mm={required_escape * 1000.0:.1f} "
                    f"available_mm={available_escape * 1000.0:.1f} "
                    f"candidates_mm="
                    f"{[round(value * 1000.0, 1) for value in candidate_distances]}"
                )
            lift_endpoint = None
            lift_waypoints = None
            lift_plan_reason = "no vertical extraction candidate was checked"
            for candidate_distance in candidate_distances:
                candidate_endpoint = np.asarray(A.ee_pos(), dtype=float)
                candidate_endpoint[2] = min(
                    A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
                    candidate_endpoint[2] + candidate_distance,
                )
                candidate_waypoints, candidate_reason = \
                    _plan_fixed_approach_trajectory(candidate_endpoint)
                if candidate_waypoints is None:
                    candidate_failures.append(
                        f"{candidate_distance * 1000:.0f}mm: {candidate_reason}"
                    )
                    continue
                if escape_detail.get("active"):
                    strict = A.validate_payload_escape_endpoint(
                        *[float(value) for value in candidate_endpoint],
                        *[float(value) for value in A.ee_rpy()],
                    )
                    if not strict.ok:
                        candidate_failures.append(
                            f"{candidate_distance * 1000:.0f}mm: {strict.reason}"
                        )
                        continue
                lift_distance = float(
                    candidate_endpoint[2] - np.asarray(A.ee_pos(), dtype=float)[2]
                )
                lift_endpoint = candidate_endpoint
                lift_waypoints = candidate_waypoints
                lift_plan_reason = candidate_reason
                break
            steps.append(("prevalidate_micro_lift" if DYNAMIC_GRASP_ENABLED
                          else "prevalidate_lift",
                          lift_waypoints is not None, lift_plan_reason))
            if lift_waypoints is None:
                if escape_detail.get("active"):
                    A.end_payload_vertical_escape()
                return recover(
                    "post-grasp lift is unsafe: "
                    + "; ".join(candidate_failures or [lift_plan_reason])
                )
            log(f"stage={'dynamic_micro_lift' if DYNAMIC_GRASP_ENABLED else 'post_grasp_lift'} "
                f"attempt={retry_count + 1} distance_mm={lift_distance * 1000:.1f}")
            escape_end = None
            try:
                lift_ok, lift_reason = _execute_fixed_approach_trajectory(
                    lift_waypoints, log
                )
            finally:
                if escape_detail.get("active"):
                    escape_end = A.end_payload_vertical_escape()
            if escape_end is not None:
                steps.append((
                    "restore_full_payload_collision_check",
                    bool(escape_end.get("ok")),
                    str(escape_end.get("reason")),
                ))
                log(
                    "stage=payload_vertical_escape_complete "
                    f"ok={escape_end.get('ok')} "
                    f"detail={escape_end.get('reason')}"
                )
            steps.append(("micro_lift" if DYNAMIC_GRASP_ENABLED else "lift",
                          lift_ok, lift_reason))
            if not lift_ok:
                return recover(f"post-grasp lift failed: {lift_reason}")
            if escape_end is not None and not escape_end.get("ok"):
                return recover(
                    "post-grasp lift did not clear the initial support overlap: "
                    f"{escape_end}"
                )

            lifted_body, lifted_hold_det, lifted_frame, lifted_hold_detail = \
                observe_closed_body(f"micro_lift_{retry_count + 1}")
            if lifted_frame is not None:
                _show(*lifted_frame,
                      tag=f"dynamic_lift_attempt_{retry_count + 1}",
                      det=lifted_hold_det,
                      target_cam=(before_lift_cam if before_lift_cam is not None
                                  else requested_cam),
                      note="visual hold check after micro-lift")
            hold_tol = (
                float(np.clip(DYNAMIC_HOLD_TOL_MM, 3.0, 8.0))
                if DYNAMIC_GRASP_ENABLED else FIXED_HELD_RIGID_TOL_MM
            )
            hold_hypothesis = None
            hold_before, hold_after, hold_basis = _hold_observation_pair(
                before_lift_cam, lifted_body,
                before_lift_feature_cam,
                lifted_hold_detail.get("feature_cam_mm"),
            )
            if hold_before is not None and hold_after is not None:
                hold_hypothesis = _lift_hold_hypothesis(
                    hold_before, hold_after,
                    np.asarray(A.ee_pos(), dtype=float) - pre_lift_xyz,
                    J, tolerance_mm=hold_tol,
                )
                hold_hypothesis["basis"] = hold_basis
            classification = (
                hold_hypothesis.get("classification")
                if hold_hypothesis is not None else "unobserved"
            )
            steps.append((
                "dynamic_visual_hold_probe",
                hold_hypothesis is not None,
                (f"classification={classification}; held residual "
                 f"{hold_hypothesis['held_residual_mm']:.1f}mm; static "
                 f"residual {hold_hypothesis['static_residual_mm']:.1f}mm; "
                 f"limit {hold_tol:.1f}mm")
                if hold_hypothesis is not None else
                str(lifted_hold_detail.get("reason")),
            ))
            log("stage=visual_hold_hypothesis "
                f"attempt={retry_count + 1} classification={classification} "
                f"basis={hold_basis} "
                f"held_residual_mm="
                f"{hold_hypothesis.get('held_residual_mm') if hold_hypothesis else 'unavailable'} "
                f"static_residual_mm="
                f"{hold_hypothesis.get('static_residual_mm') if hold_hypothesis else 'unavailable'} "
                f"limit_mm={hold_tol:.1f}")
            held = classification == "held"

            # If neither rigid-with-wrist nor static-on-table explains the first
            # lift, do one extra 10 mm validated lift. Object seating happens on
            # close/first lift; incremental motion after that cleanly separates
            # the two hypotheses without tightening the jaws.
            if classification == "ambiguous" and hold_after is not None:
                confirm_start = np.asarray(A.ee_pos(), dtype=float)
                confirm_endpoint = confirm_start.copy()
                confirm_endpoint[2] = min(
                    A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
                    confirm_endpoint[2] + 0.010,
                )
                confirm_waypoints, confirm_reason = \
                    _plan_fixed_approach_trajectory(confirm_endpoint)
                steps.append((
                    "prevalidate_hold_confirmation_lift",
                    confirm_waypoints is not None, confirm_reason,
                ))
                if confirm_waypoints is None:
                    return recover(
                        f"ambiguous grasp could not be safely confirmed: "
                        f"{confirm_reason}"
                    )
                log("stage=dynamic_hold_confirmation_lift distance_mm=10.0")
                confirm_ok, confirm_reason = \
                    _execute_fixed_approach_trajectory(confirm_waypoints, log)
                steps.append(("hold_confirmation_lift",
                              confirm_ok, confirm_reason))
                if not confirm_ok:
                    return recover(
                        f"hold confirmation lift failed: {confirm_reason}"
                    )
                confirm_body, confirm_det, confirm_frame, confirm_detail = \
                    observe_closed_body(f"confirm_lift_{retry_count + 1}")
                if confirm_frame is not None:
                    _show(
                        *confirm_frame,
                        tag=f"dynamic_hold_confirmation_{retry_count + 1}",
                        det=confirm_det, target_cam=lifted_body,
                        note="incremental visual hold confirmation",
                    )
                # The exact selected cross-section is commonly hidden after a
                # second lift even though the semantic identifier remains
                # visible. Reuse the same rigid-motion witness allowed for the
                # first lift: physical body when available, otherwise the
                # stable semantic feature. This is verification-only and can
                # never authorize alignment or a close command.
                confirm_before, confirm_after, confirm_basis = \
                    _hold_observation_pair(
                        lifted_body, confirm_body,
                        lifted_hold_detail.get("feature_cam_mm"),
                        confirm_detail.get("feature_cam_mm"),
                    )
                confirm_hypothesis = None
                if confirm_before is not None and confirm_after is not None:
                    confirm_hypothesis = _lift_hold_hypothesis(
                        confirm_before, confirm_after,
                        np.asarray(A.ee_pos(), dtype=float) - confirm_start,
                        J, tolerance_mm=hold_tol,
                    )
                    confirm_hypothesis["basis"] = confirm_basis
                classification = (
                    confirm_hypothesis.get("classification")
                    if confirm_hypothesis is not None else "unobserved"
                )
                held = classification == "held"
                steps.append((
                    "dynamic_visual_hold_check", held,
                    (f"confirmation classification={classification}; held "
                     f"residual {confirm_hypothesis['held_residual_mm']:.1f}mm; "
                     f"static residual "
                     f"{confirm_hypothesis['static_residual_mm']:.1f}mm")
                    if confirm_hypothesis is not None else
                    str(confirm_detail.get("reason")),
                ))
                log("stage=visual_hold_confirmation "
                    f"held={held} classification={classification} "
                    f"basis={confirm_basis} "
                    f"held_residual_mm="
                    f"{confirm_hypothesis.get('held_residual_mm') if confirm_hypothesis else 'unavailable'} "
                    f"static_residual_mm="
                    f"{confirm_hypothesis.get('static_residual_mm') if confirm_hypothesis else 'unavailable'}")
            else:
                steps.append((
                    "dynamic_visual_hold_check", held,
                    (f"classification={classification}; held residual "
                     f"{hold_hypothesis['held_residual_mm']:.1f}mm; static "
                     f"residual {hold_hypothesis['static_residual_mm']:.1f}mm")
                    if hold_hypothesis is not None else
                    str(lifted_hold_detail.get("reason")),
                ))
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
            retry_body, retry_det, retry_frame, retry_detail = \
                observe_closed_body(f"retry_return_{retry_count + 1}")
            if retry_frame is not None:
                _show(*retry_frame,
                      tag=f"dynamic_retry_endpoint_{retry_count + 1}",
                      det=retry_det,
                      target_cam=(before_lift_cam if before_lift_cam is not None
                                  else requested_cam),
                      note="returned endpoint before tighter close")
            retry_basis = "exact locked grasp section"
            if retry_body is None:
                retry_body, retry_fallback_detail = \
                    _locked_body_from_feature(
                        retry_detail.get("feature_cam_mm"),
                        feature_to_grasp_cam,
                    )
                retry_basis = str(retry_fallback_detail.get("reason"))
            if retry_body is None:
                return recover(
                    "dynamic retry could not re-identify the body section or "
                    "recover it from a stable semantic feature; refusing "
                    "additional closure"
                )
            # Closing/first lift can seat a deformable or long object a few
            # millimetres inside the jaws. The post-close body baseline was
            # already proven inside the calibrated slot. On return, compare to
            # that actual captured position—not the ideal empty-jaw target—or
            # harmless seating is counted twice and a safe retry is refused.
            retry_slot_reference = (
                before_lift_cam if before_lift_cam is not None
                else requested_cam
            )
            retry_body_slot_ok, retry_body_slot = _jaw_slot_fit(
                retry_body, retry_slot_reference, jaw_plan, endpoint_geometry,
                full_object_box_3d,
            )
            steps.append((
                "dynamic_retry_grasp_body_inside_jaw_slot",
                retry_body_slot_ok,
                f"basis={retry_basis}; reference="
                f"{'post_close_body_baseline' if before_lift_cam is not None else 'calibrated_jaw_slot'}; "
                f"{retry_body_slot.get('reason')}",
            ))
            log("stage=dynamic_retry_grasp_body_slot_gate "
                f"ok={retry_body_slot_ok} "
                f"basis={retry_basis} "
                f"reference={'post_close_body_baseline' if before_lift_cam is not None else 'calibrated_jaw_slot'} "
                f"detail={retry_body_slot.get('reason')}")
            if not retry_body_slot_ok:
                return recover(
                    "physical grasp body moved outside the calibrated jaw slot "
                    "after the failed lift; refusing a blind retry: "
                    f"{retry_body_slot.get('reason')}"
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
            before_lift_cam = np.asarray(retry_body, dtype=float)
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

        if voxel_place_requested:
            scene_restored, scene_restore_reason = _activate_cached_scene_map()
            steps.append((
                "restore_scene_after_pick_lift",
                scene_restored, scene_restore_reason,
            ))
            log(
                "stage=scene_collision_lifecycle phase=carry "
                f"restored={scene_restored} detail={scene_restore_reason}"
            )
            if not scene_restored:
                # Do not attempt a blind carry. Clear the rejected map so the
                # existing release/home recovery is not blocked by bad boxes.
                A.clear_collision_scene()
                return recover(
                    "object was picked, but the cached scene could not be "
                    f"safely restored for voxel carry: {scene_restore_reason}"
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
            "object_box_3d": full_object_box_3d,
            "payload_box": payload_box_context,
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
    except CameraFrameUnavailable as exc:
        log(f"stage=camera_unavailable status=abort detail={exc}")
        if dry:
            return _result(
                False, f"wrist camera unavailable: {exc}", steps,
                homed=False, object_held=False,
            )
        return recover(
            f"wrist camera unavailable: {exc}; stopped before any further "
            "camera-dependent trajectory"
        )
    except KeyboardInterrupt:
        log(
            "stage=operator_stop input=Q_or_Ctrl+C "
            "action=freeze_open_home"
        )
        if dry:
            return _result(
                False, "dry fixed pick/place interrupted", steps,
                homed=False, object_held=False,
            )
        return recover("operator requested stop during fixed pick/place")
    except BaseException as exc:
        if dry:
            return _result(False, f"dry fixed pick/place exception: "
                           f"{type(exc).__name__}: {exc}", steps,
                           homed=False, object_held=False)
        return recover(f"fixed pick/place exception: {type(exc).__name__}: {exc}")
    finally:
        set_identity_filter = getattr(eye, "set_identity_filter", None)
        if callable(set_identity_filter) and hybrid_identity_gate is not None:
            set_identity_filter(object_name, None)
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


def _voxel_payload_envelope(
    context: dict[str, Any], current_rpy: Any,
) -> dict[str, Any]:
    """Project the measured held-object OBB into a 2.5D carry envelope.

    This does not enable the backend's experimental attached-payload collision
    model. It gives the top-down navigator the object's measured XY reach and
    its lowest point relative to the end effector, which are precisely the two
    quantities needed to keep a carried object out of occupied columns.
    """
    payload = context.get("payload_box")
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "reason": "measured held-object payload box is unavailable",
        }
    try:
        size = np.asarray(payload["full_size_m"], dtype=float)
        axes_ee = np.asarray(payload["full_axes_ee"], dtype=float).T
        center_ee = np.asarray(payload["center_offset_ee_m"], dtype=float)
        if size.shape != (3,) or axes_ee.shape != (3, 3) \
                or center_ee.shape != (3,) or np.any(size <= 0.0) \
                or not np.all(np.isfinite(size)) \
                or not np.all(np.isfinite(axes_ee)) \
                or not np.all(np.isfinite(center_ee)):
            raise ValueError("payload OBB arrays are invalid")
        rotation = _rpy_rotation_matrix(current_rpy)
        axes_base = rotation @ axes_ee
        center_base = rotation @ center_ee
        half_extents_base = np.sum(
            np.abs(axes_base) * size[None, :] / 2.0, axis=1
        )
        # A circular XY envelope is orientation-independent during a fixed-RPY
        # carry and safely contains every projected OBB corner.
        radial_extent = float(np.linalg.norm(
            np.abs(center_base[:2]) + half_extents_base[:2]
        ))
        downward_extent = float(max(
            0.0, half_extents_base[2] - center_base[2]
        ))
        upward_extent = float(max(
            0.0, half_extents_base[2] + center_base[2]
        ))
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "reason": f"invalid measured payload OBB: {exc}"}
    return {
        "ok": True,
        "reason": "measured 3D OBB projected into fixed-RPY voxel envelope",
        "radial_extent_m": radial_extent,
        "downward_extent_m": downward_extent,
        "upward_extent_m": upward_extent,
        "center_offset_base_m": center_base.round(5).tolist(),
        "half_extents_base_m": half_extents_base.round(5).tolist(),
        "size_m": size.round(5).tolist(),
    }


def _scene_reference_point_consensus(
    points_camera_mm: list[np.ndarray], *, required: int,
    max_spread_mm: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Select a stable placement-reference cluster without averaging outliers."""
    samples = [
        {"body_cam_mm": np.asarray(point, dtype=float).tolist()}
        for point in points_camera_mm
    ]
    return _endpoint_body_pose_consensus(
        samples, min_frames=required, max_spread_mm=max_spread_mm
    )


def _scene_reference_consensus(
    eye: WristEye, reference: str, log: Callable[[str], None], *,
    force_look: bool = False,
) -> tuple[Any | None, np.ndarray | None, dict[str, Any]]:
    samples: list[np.ndarray] = []
    sample_detections: list[Any] = []
    sample_frames: list[Any] = []
    latest = None
    latest_frame = None
    reasons: list[Any] = []
    count = int(np.clip(SCENE_PLACE_CONSENSUS_FRAMES, 3, 7))
    required = max(2, count - 1)
    for index in range(count) if not force_look else ():
        det, frame, why = _observe(eye, reference)
        if det is None:
            reasons.append(why)
            continue
        partial = _detection_partial(det, frame)
        if partial is not None:
            reasons.append(partial)
            continue
        latest, latest_frame = det, frame
        samples.append(np.asarray(det.p_cam_mm, dtype=float))
        sample_detections.append(det)
        sample_frames.append(frame)
        log(f"stage=place_reference_sample frame={index + 1}/{count} "
            f"source={det.source} p_cam_mm={samples[-1].round(1).tolist()}")
        time.sleep(0.08)
    median, stable_detail = _scene_reference_point_consensus(
        samples, required=required,
        max_spread_mm=SCENE_PLACE_MAX_SPREAD_MM,
    )
    # A stochastic semantic depth outlier must not force an expensive arm
    # sweep when the reference remains visible. Capture a bounded number of
    # additional stationary frames and look for an unchanged-size inlier
    # cluster; the 15 mm spatial gate itself is never relaxed.
    if median is None and len(samples) >= required:
        for attempt in range(1, count + 1):
            log(
                "stage=place_reference_consensus status=resampling "
                f"attempt={attempt}/{count} detail="
                f"{stable_detail.get('reason')}; arm stationary"
            )
            det, frame, why = _observe(eye, reference)
            if det is None or frame is None:
                reasons.append(why)
                continue
            partial = _detection_partial(det, frame)
            if partial is not None:
                reasons.append(partial)
                continue
            latest, latest_frame = det, frame
            samples.append(np.asarray(det.p_cam_mm, dtype=float))
            sample_detections.append(det)
            sample_frames.append(frame)
            log(
                "stage=place_reference_sample_resample "
                f"frame={len(samples)} source={det.source} "
                f"p_cam_mm={samples[-1].round(1).tolist()}"
            )
            median, stable_detail = _scene_reference_point_consensus(
                samples, required=required,
                max_spread_mm=SCENE_PLACE_MAX_SPREAD_MM,
            )
            if median is not None:
                break
            time.sleep(0.08)
    look_around_detail = None
    if len(samples) < required or median is None:
        log("stage=look_around_trigger role=placement_reference "
            f"reason="
            f"{'mapped_reference_outside_reachable_workspace' if force_look else ('complete_frames_' + str(len(samples)) + '_of_' + str(count) if len(samples) < required else stable_detail.get('reason'))}")
        found, found_frame, _geometry, look_around_detail = _look_around_object(
            eye, reference, log, require_grasp=False,
            require_camera_mapping=True,
            observation_consensus_frames=count,
            observation_consensus_max_spread_mm=SCENE_PLACE_MAX_SPREAD_MM,
            allow_edge_recenter=True,
        )
        if found is not None and found_frame is not None:
            latest, latest_frame = found, found_frame
            samples = []
            sample_detections = []
            sample_frames = []
            reasons = []
            look_consensus = (
                look_around_detail.get("observation_consensus")
                if isinstance(look_around_detail, dict) else None
            )
            if look_consensus:
                samples = [
                    np.asarray(point, dtype=float)
                    for point in look_consensus.get("samples_camera_mm", [])
                ]
                sample_detections = [found] * len(samples)
                sample_frames = [found_frame] * len(samples)
                log(
                    "stage=place_reference_consensus_after_look "
                    f"pose={look_around_detail.get('pose')} "
                    f"stable_frames={len(samples)}/{count} "
                    f"spread_mm={float(look_consensus.get('spread_mm', 0.0)):.1f}"
                )
            # The discovery routine accepted this exact frame only after its
            # full-object/edge checks passed. Retain it as the first consensus
            # member instead of throwing it away and asking a stochastic
            # detector to rediscover the same view from scratch.
            if not look_consensus:
                found_partial = _detection_partial(found, found_frame)
                if found_partial is None:
                    samples.append(np.asarray(found.p_cam_mm, dtype=float))
                    sample_detections.append(found)
                    sample_frames.append(found_frame)
                    log("stage=place_reference_sample_after_look "
                        f"sample={len(samples)}/{count} attempt=accepted_pose "
                        f"source={found.source} "
                        f"p_cam_mm={samples[-1].round(1).tolist()}")
                else:
                    reasons.append(found_partial)

            # Resample only at this stationary pose and stop as soon as the
            # existing env-driven consensus requirement is met. One discovery
            # frame alone can never authorize placement.
            attempt_limit = count
            for attempt in range(1, attempt_limit + 1) if not look_consensus else ():
                if len(samples) >= required:
                    break
                det, frame, why = _observe(eye, reference)
                if det is None:
                    reasons.append(why)
                    log("stage=place_reference_sample_after_look "
                        f"sample={len(samples)}/{count} "
                        f"attempt={attempt}/{attempt_limit} accepted=false "
                        f"detail={why}")
                    continue
                partial = _detection_partial(det, frame)
                if partial is not None:
                    reasons.append(partial)
                    log("stage=place_reference_sample_after_look "
                        f"sample={len(samples)}/{count} "
                        f"attempt={attempt}/{attempt_limit} accepted=false "
                        f"detail={partial}")
                    continue
                latest, latest_frame = det, frame
                samples.append(np.asarray(det.p_cam_mm, dtype=float))
                sample_detections.append(det)
                sample_frames.append(frame)
                log("stage=place_reference_sample_after_look "
                    f"sample={len(samples)}/{count} "
                    f"attempt={attempt}/{attempt_limit} accepted=true "
                    f"source={det.source} p_cam_mm={samples[-1].round(1).tolist()}")
                time.sleep(0.08)
    if len(samples) < required:
        return None, None, {
            "reason": f"reference detected completely in only {len(samples)}/{count} frames",
            "detector_reasons": reasons,
            "look_around": look_around_detail,
        }
    median, stable_detail = _scene_reference_point_consensus(
        samples, required=required,
        max_spread_mm=SCENE_PLACE_MAX_SPREAD_MM,
    )
    points = np.vstack(samples)
    if median is None:
        return None, None, {
            "reason": str(stable_detail.get("reason")),
            "samples_camera_mm": points.round(1).tolist(),
        }
    rejected = set(stable_detail.get("rejected_frame_indices", []))
    stable_indices = [
        index for index in range(len(samples)) if index not in rejected
    ]
    spread = float(stable_detail.get("spread_mm", 0.0))
    if stable_indices and len(sample_detections) == len(samples):
        representative_index = min(
            stable_indices,
            key=lambda index: float(np.linalg.norm(samples[index] - median)),
        )
        latest = sample_detections[representative_index]
        latest_frame = sample_frames[representative_index]
    if latest_frame is not None:
        target = load_grasp_target()
        _show(*latest_frame, tag="scene_place_reference", det=latest,
              tcp_cam=np.asarray(target, dtype=float) if target else None,
              note=f"placement reference: {reference}")
    return latest, median, {
        "reason": f"{len(stable_indices)}/{count} stable frames; "
                  f"spread={spread:.1f}mm",
        "reference_camera_mm": median.round(1).tolist(),
        "spread_mm": round(spread, 1),
        "rejected_frame_indices": sorted(rejected),
        "source": latest.source if latest is not None else None,
        "look_around": look_around_detail,
        "look_around_moved": bool(look_around_detail and
                                  look_around_detail.get("moved")),
    }


def _high_release_waypoints(
    endpoint: np.ndarray, fixed_rpy: tuple[float, float, float],
    log: Callable[[str], None] | None = None,
) -> tuple[Any | None, str]:
    """Choose a fully IK- and collision-validated carry corridor."""
    emit = log or (lambda _message: None)
    start = np.asarray(A.ee_pos(), dtype=float)
    # The production backend exposes the payload-aware endpoint gate.  Keep
    # injected/simulation backends compatible by falling back to their normal
    # read-only IK check; those backends do not model an attached collision
    # payload in the first place.
    endpoint_check = getattr(A, "check_pose_clearance", None)
    if endpoint_check is None:
        endpoint_check = A.check_ik
    endpoint_gate = endpoint_check(
        *[float(value) for value in endpoint], *fixed_rpy
    )
    if not endpoint_gate.ok:
        emit(
            "stage=carry_route_plan status=endpoint_rejected "
            f"endpoint={np.asarray(endpoint).round(4).tolist()} "
            f"detail={endpoint_gate.reason}"
        )
        return None, f"release endpoint is occupied or unreachable: {endpoint_gate.reason}"
    ceiling = A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M
    base_z = min(ceiling, max(float(start[2]), float(endpoint[2])))
    travel_heights = sorted({
        round(base_z, 4),
        round(min(ceiling, base_z + 0.040), 4),
        round(min(ceiling, base_z + 0.080), 4),
    })
    failures: list[str] = []
    global_attempted = False
    global_failure = "not attempted"

    def try_global_route() -> tuple[Any | None, str]:
        """Use all arm joints/XYZ/RPY after the direct carry route fails."""
        nonlocal global_attempted, global_failure
        global_attempted = True
        emit(
            "stage=carry_route_plan status=global_search "
            "detail=direct corridor failed; searching full joint space with "
            "bounded payload tilt; arm stationary"
        )
        max_tilt_deg = float(np.clip(
            float(os.getenv("REBOT_GLOBAL_PLANNER_MAX_TILT_DEG", "12")),
            2.0, 20.0,
        ))
        timeout_s = float(np.clip(
            float(os.getenv("REBOT_GLOBAL_PLANNER_TIMEOUT_S", "4")),
            0.5, 12.0,
        ))
        candidate = A.plan_joint_motion(
            [(*[float(value) for value in endpoint], *fixed_rpy)],
            orientation_ref_rpy=fixed_rpy,
            max_tilt_rad=math.radians(max_tilt_deg),
            timeout_s=timeout_s,
            max_nodes=int(os.getenv("REBOT_GLOBAL_PLANNER_MAX_NODES", "4000")),
            seed=int(os.getenv("REBOT_GLOBAL_PLANNER_SEED", "0")),
        )
        if candidate.get("ok"):
            planner = candidate.get("planner", {})
            emit(
                "stage=carry_route_plan status=global_accepted "
                f"nodes={planner.get('nodes')} "
                f"elapsed_s={planner.get('elapsed_s')} "
                f"joint_waypoints={len(candidate.get('joint_path', []))} "
                f"max_tilt_deg={max_tilt_deg:.1f} "
                f"scene_version={candidate.get('scene_version')}"
            )
            return candidate, (
                "collision-aware global joint route using full joint space "
                "with free yaw and "
                f"bounded {max_tilt_deg:.1f}deg roll/pitch"
            )
        global_failure = str(candidate.get("reason"))
        emit(
            "stage=carry_route_plan status=global_rejected "
            f"detail={global_failure}; continuing deterministic 3D corridors"
        )
        return None, global_failure

    attempt = 0
    total_attempts = len(travel_heights) * 3
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
            attempt += 1
            emit(
                "stage=carry_route_plan status=building "
                f"attempt={attempt}/{total_attempts} route={route_name} "
                f"travel_z_m={travel_z:.3f}"
            )
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
                emit(
                    "stage=carry_route_plan status=rejected "
                    f"attempt={attempt}/{total_attempts} route={route_name} "
                    f"travel_z_m={travel_z:.3f} detail={failed or 'empty path'}"
                )
                failures.append(
                    f"{route_name}@z={travel_z:.3f}: {failed or 'empty path'}"
                )
                if attempt == 1 and not global_attempted:
                    global_route, global_reason = try_global_route()
                    if global_route is not None:
                        return global_route, global_reason
                continue
            emit(
                "stage=carry_route_plan status=validating "
                f"attempt={attempt}/{total_attempts} route={route_name} "
                f"travel_z_m={travel_z:.3f} waypoints={len(combined)} "
                "detail=running deterministic IK and collision checks; arm stationary"
            )
            ok, reason = _validate_fixed_waypoints(combined, fixed_rpy)
            if not ok:
                emit(
                    "stage=carry_route_plan status=rejected "
                    f"attempt={attempt}/{total_attempts} route={route_name} "
                    f"travel_z_m={travel_z:.3f} detail={reason}"
                )
                failures.append(f"{route_name}@z={travel_z:.3f}: {reason}")
                if attempt == 1 and not global_attempted:
                    global_route, global_reason = try_global_route()
                    if global_route is not None:
                        return global_route, global_reason
                continue
            # Candidates are already ordered from shortest/lowest-complexity to
            # more circuitous/higher-clearance routes. Every accepted candidate
            # has passed the complete collision gate, so do not spend tens of
            # seconds proving that additional unused routes are also valid.
            emit(
                "stage=carry_route_plan status=accepted "
                f"attempt={attempt}/{total_attempts} route={route_name} "
                f"travel_z_m={travel_z:.3f} waypoints={len(combined)}"
            )
            return combined, (
                "collision-aware high release route; "
                f"{route_name} at z={travel_z:.3f}; {reason}"
            )
    # Normally the global search was already attempted immediately after the
    # direct corridor failed. Keep this final guard for injected/test backends or
    # a future candidate ordering that bypasses the direct attempt.
    if not global_attempted:
        global_route, global_reason = try_global_route()
        if global_route is not None:
            return global_route, global_reason
    return None, (
        "all collision-aware carry corridors and global joint search failed: "
        + "; ".join(failures) + f"; global: {global_failure}"
    )


def _log_voxel_route_view(
    proposed: VN.VoxelRoute, dense_waypoints: Any, *,
    accepted: bool, footprint_attempt: int, route_radius_m: float,
    reason: str,
) -> str | None:
    """Append a candidate route to the optional scene Rerun recording."""
    if not SCENE_COLLISION_EXPORT_VIEW:
        return None
    try:
        import rerun as rr

        global _VOXEL_ROUTE_VIZ_SEQUENCE
        _VOXEL_ROUTE_VIZ_SEQUENCE += 1
        sequence = _VOXEL_ROUTE_VIZ_SEQUENCE
        status = "accepted" if accepted else "rejected"
        entity = (
            f"planning/voxel_routes/{sequence:03d}_"
            f"footprint_{footprint_attempt}_{status}"
        )
        dense = np.asarray(dense_waypoints, dtype=float).reshape(-1, 3)
        if not len(dense):
            dense = np.asarray(proposed.anchors_m, dtype=float).reshape(-1, 3)
        anchors = np.asarray(proposed.anchors_m, dtype=float).reshape(-1, 3)
        color = [40, 230, 80, 255] if accepted else [255, 55, 55, 230]
        rr.log(
            f"{entity}/validated_cartesian_path",
            rr.LineStrips3D(
                [dense], colors=[color], radii=[0.003],
                labels=[
                    f"{status}: footprint {route_radius_m * 1000.0:.0f}mm"
                ], show_labels=True,
            ),
            static=True,
        )
        rr.log(
            f"{entity}/turn_and_staging_anchors",
            rr.Points3D(
                anchors, colors=[color], radii=0.007,
                labels=[f"anchor {index}" for index in range(len(anchors))],
                show_labels=True,
            ),
            static=True,
        )
        # Keep the reason in the JSON/log output as the authoritative text;
        # this entity name is returned so users can correlate it in Rerun.
        return entity
    except Exception:  # noqa: BLE001 - visualization must never affect safety
        return None


def _voxel_release_waypoints(
    endpoint: np.ndarray, fixed_rpy: tuple[float, float, float],
    log: Callable[[str], None], context: dict[str, Any] | None = None,
) -> tuple[Any | None, np.ndarray, dict[str, Any], str]:
    """Propose a top-down free-cell carry route, then certify the full arm."""
    if _ACTIVE_SCENE_MAP is None:
        detail = {"reason": "no current target-excluded scene map is active"}
        return None, np.asarray(endpoint, dtype=float), detail, detail["reason"]
    resolution = float(os.getenv(
        "REBOT_VOXEL_NAV_RESOLUTION_M",
        str(max(0.010, min(0.025, _ACTIVE_SCENE_MAP.voxel_m))),
    ))
    base_ee_radius = float(np.clip(float(os.getenv(
        "REBOT_VOXEL_NAV_EE_RADIUS_M", "0.025"
    )), 0.015, 0.050))
    base_vertical_clearance = float(np.clip(float(os.getenv(
        "REBOT_VOXEL_NAV_VERTICAL_CLEARANCE_M", "0.020"
    )), 0.008, 0.060))
    base_drop_clearance = float(np.clip(float(os.getenv(
        "REBOT_VOXEL_DROP_CLEARANCE_M", "0.030"
    )), 0.015, 0.080))
    payload_envelope = _voxel_payload_envelope(
        context or {}, fixed_rpy
    )
    payload_margin = max(
        float(_ACTIVE_SCENE_MAP.inflation_m),
        float(np.clip(float(os.getenv(
            "REBOT_VOXEL_PAYLOAD_MARGIN_M", "0.005"
        )), 0.003, 0.015)),
    )
    if payload_envelope.get("ok"):
        ee_radius = max(
            base_ee_radius,
            float(payload_envelope["radial_extent_m"]) + payload_margin,
        )
        payload_down = float(payload_envelope["downward_extent_m"])
        vertical_clearance = max(
            base_vertical_clearance, payload_down + payload_margin
        )
        # Releasing this high puts the measured lowest payload point a small,
        # positive distance above the destination column. It is a drop policy,
        # not a support-surface classification.
        drop_gap = float(np.clip(float(os.getenv(
            "REBOT_VOXEL_PAYLOAD_DROP_GAP_M", "0.008"
        )), 0.005, 0.020))
        drop_clearance = max(
            base_drop_clearance, payload_down + drop_gap
        )
    else:
        # Scene-relative runs built by the current pick always have this box.
        # Refuse to silently regress to end-effector-only navigation.
        detail = {
            "reason": "payload-aware voxel navigation unavailable: "
                      f"{payload_envelope.get('reason')}",
            "payload_envelope": payload_envelope,
        }
        return None, np.asarray(endpoint, dtype=float), detail, detail["reason"]
    footprint_steps = int(np.clip(int(os.getenv(
        "REBOT_VOXEL_ARM_FOOTPRINT_RETRIES", "3"
    )), 0, 4))
    footprint_attempts: list[dict[str, Any]] = []
    last_detail: dict[str, Any] = {}
    last_reason = "voxel route was not attempted"
    last_endpoint = np.asarray(endpoint, dtype=float)
    for footprint_attempt in range(footprint_steps + 1):
        route_radius = ee_radius + footprint_attempt * resolution
        proposed, proposal_detail = VN.plan_topdown_route(
            _ACTIVE_SCENE_MAP,
            np.asarray(A.ee_pos(), dtype=float),
            np.asarray(endpoint, dtype=float)[:2],
            workspace_lo_m=(A.X_MIN, A.Y_MIN, A.Z_MIN),
            workspace_hi_m=(A.X_MAX, A.Y_MAX, A.Z_MAX),
            workspace_margin_m=APPROACH_WORKSPACE_MARGIN_M,
            ee_radius_m=route_radius,
            destination_radius_m=ee_radius,
            vertical_clearance_m=vertical_clearance,
            drop_clearance_m=drop_clearance,
            resolution_m=resolution,
        )
        if proposed is None:
            last_detail = dict(proposal_detail)
            last_reason = str(proposal_detail.get("reason"))
            footprint_attempts.append({
                "attempt": footprint_attempt + 1,
                "radius_m": round(route_radius, 4),
                "ok": False, "reason": last_reason,
            })
            log("stage=voxel_carry_plan status=rejected "
                f"footprint_attempt={footprint_attempt + 1}/"
                f"{footprint_steps + 1} radius_m={route_radius:.3f} "
                f"detail={last_reason} "
                f"layers={proposal_detail.get('attempted_layers')}")
            continue
        detail = dict(proposed.detail)
        detail["payload_envelope"] = payload_envelope
        detail["payload_margin_m"] = round(payload_margin, 4)
        resolved_endpoint = np.asarray(
            proposed.release_endpoint_m, dtype=float
        )
        last_endpoint = resolved_endpoint
        log(
            "stage=voxel_carry_plan status=proposed "
            f"footprint_attempt={footprint_attempt + 1}/"
            f"{footprint_steps + 1} "
            f"grid={detail['grid_cells']} occupied={detail['occupied_cells']} "
            f"resolution_m={detail['resolution_m']:.3f} "
            f"strategy={detail['travel_strategy']} "
            f"start_z_m={detail['starting_carry_z_m']:.3f} "
            f"preferred_clear_z_m="
            f"{detail['preferred_payload_clear_z_m']:.3f} "
            f"rose_before_xy={str(detail['rose_before_xy']).lower()} "
            f"travel_z_m={detail['travel_z_m']:.3f} "
            f"destination_top_m={detail['destination_column_top_m']:.3f} "
            f"release_z_m={detail['release_z_m']:.3f} "
            f"route_radius_m={route_radius:.3f} "
            f"payload_down_m={float(payload_envelope['downward_extent_m']):.3f} "
            f"cells={detail['path_cells']} turns={detail['path_turns']}"
        )
        combined: list[tuple[float, float, float]] = []
        leg_start = np.asarray(A.ee_pos(), dtype=float)
        densify_reason = "ok"
        leg = None
        for anchor_raw in proposed.anchors_m:
            anchor = np.asarray(anchor_raw, dtype=float)
            if float(np.linalg.norm(anchor - leg_start)) < 1e-7:
                continue
            leg, densify_reason = _plan_fixed_approach_trajectory(
                anchor, start_m=leg_start, fixed_rpy=fixed_rpy,
                validate_ik=False,
            )
            if leg is None:
                break
            combined.extend(leg)
            leg_start = anchor
        if not combined or leg is None:
            last_reason = densify_reason if leg is None else "empty route"
            detail["validation_reason"] = last_reason
            viewer_entity = _log_voxel_route_view(
                proposed, combined, accepted=False,
                footprint_attempt=footprint_attempt + 1,
                route_radius_m=route_radius, reason=last_reason,
            )
            attempt_detail = {
                "attempt": footprint_attempt + 1,
                "radius_m": round(route_radius, 4),
                "ok": False, "reason": last_reason,
            }
            if viewer_entity:
                attempt_detail["viewer_entity"] = viewer_entity
                detail["route_viewer_entity"] = viewer_entity
            footprint_attempts.append(attempt_detail)
            last_detail = detail
            log("stage=voxel_carry_plan status=rejected phase=densify "
                f"footprint_attempt={footprint_attempt + 1}/"
                f"{footprint_steps + 1} detail={last_reason}")
            continue
        log("stage=voxel_carry_plan status=validating "
            f"footprint_attempt={footprint_attempt + 1}/"
            f"{footprint_steps + 1} anchors={len(proposed.anchors_m)} "
            f"waypoints={len(combined)} "
            "detail=full-arm deterministic IK and collision gate")
        ok, reason = _validate_fixed_waypoints(combined, fixed_rpy)
        detail["validation_reason"] = reason
        detail["validated_waypoints"] = len(combined)
        viewer_entity = _log_voxel_route_view(
            proposed, combined, accepted=bool(ok),
            footprint_attempt=footprint_attempt + 1,
            route_radius_m=route_radius, reason=reason,
        )
        attempt_detail = {
            "attempt": footprint_attempt + 1,
            "radius_m": round(route_radius, 4),
            "ok": bool(ok), "reason": reason,
        }
        if viewer_entity:
            attempt_detail["viewer_entity"] = viewer_entity
            detail["route_viewer_entity"] = viewer_entity
        footprint_attempts.append(attempt_detail)
        detail["arm_footprint_attempts"] = list(footprint_attempts)
        last_detail = detail
        last_reason = reason
        if not ok:
            log("stage=voxel_carry_plan status=rejected phase=full_arm "
                f"footprint_attempt={footprint_attempt + 1}/"
                f"{footprint_steps + 1} detail={reason} "
                f"viewer_entity={viewer_entity or 'disabled'}")
            continue
        log("stage=voxel_carry_plan status=accepted "
            f"footprint_attempt={footprint_attempt + 1}/"
            f"{footprint_steps + 1} route_radius_m={route_radius:.3f} "
            f"waypoints={len(combined)} detail={reason} "
            f"viewer_entity={viewer_entity or 'disabled'}")
        return combined, resolved_endpoint, detail, (
            "top-down voxel A* route passed complete arm IK/collision "
            "validation"
        )
    last_detail["payload_envelope"] = payload_envelope
    last_detail["payload_margin_m"] = round(payload_margin, 4)
    last_detail["arm_footprint_attempts"] = footprint_attempts
    return None, last_endpoint, last_detail, last_reason


def _activate_cached_scene_map() -> tuple[bool, str]:
    """Restore the captured environment after the proven pick has lifted."""
    if _ACTIVE_SCENE_MAP is None:
        return False, "no cached target-excluded scene map is available"
    configured = A.configure_collision_scene(
        [box.as_dict() for box in _ACTIVE_SCENE_MAP.boxes],
        table_z_m=float(_ACTIVE_SCENE_MAP.table_z_m),
        payload_radius_m=0.0,
        voxel_m=float(_ACTIVE_SCENE_MAP.voxel_m),
        inflation_m=float(_ACTIVE_SCENE_MAP.inflation_m),
    )
    if not configured.ok:
        A.clear_collision_scene()
        return False, configured.reason
    current_gate = A.validate_current_collision_scene()
    if not current_gate.get("ok"):
        A.clear_collision_scene()
        return False, str(current_gate.get("reason"))
    return True, configured.reason


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
    def map_reference(
        reference_camera_mm: np.ndarray, reference_observation: dict[str, Any],
    ) -> tuple[np.ndarray | None, tuple[float, float, float], dict[str, Any]]:
        mapped_rpy = tuple(float(v) for v in A.ee_rpy())
        if camera_mapping_override is not None and not reference_observation.get(
            "look_around_moved", False
        ):
            mapped_camera = np.asarray(camera_mapping_override, dtype=float)
            mapped_detail = dict(camera_mapping_detail or {
                "reason": "caller-supplied rigid camera mapping"
            })
        else:
            mapped_camera, mapped_detail = load_saved_camera_jacobian(mapped_rpy)
        if mapped_camera is None:
            return None, mapped_rpy, {
                "reason": f"saved rigid camera mapping unavailable: "
                          f"{mapped_detail.get('reason')}"
            }
        try:
            mapped_delta = np.linalg.solve(
                mapped_camera,
                np.asarray(grasp_target, dtype=float) - reference_camera_mm,
            )
        except np.linalg.LinAlgError:
            return None, mapped_rpy, {
                "reason": "saved rigid camera mapping is singular"
            }
        mapped_xy = np.asarray(A.ee_pos(), dtype=float)[:2] + mapped_delta[:2]
        return mapped_xy, mapped_rpy, mapped_detail

    reference_xy, current_rpy, mapping_detail = map_reference(
        reference_cam, observation
    )
    if reference_xy is None:
        return None, None, mapping_detail

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
    release_z = min(
        A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M, FIXED_DROP_RELEASE_Z_M
    )
    # A requested relation is semantic (stay on this side), while the supplied
    # distance is the preferred minimum clearance. A coarse inflated obstacle
    # column can legitimately make that exact distance unavailable. Try only
    # progressively farther placements on the same requested side; never cross
    # to another side and never exceed the existing 180 mm placement bound.
    clearance_offsets = [offset]
    for extra in (0.020, 0.040, 0.060, 0.080):
        candidate_offset = float(min(0.180, offset + extra))
        if not any(abs(candidate_offset - value) < 1e-6
                   for value in clearance_offsets):
            clearance_offsets.append(candidate_offset)

    guarded_lo = np.array([
        A.X_MIN + APPROACH_WORKSPACE_MARGIN_M,
        A.Y_MIN + APPROACH_WORKSPACE_MARGIN_M,
    ])
    guarded_hi = np.array([
        A.X_MAX - APPROACH_WORKSPACE_MARGIN_M,
        A.Y_MAX - APPROACH_WORKSPACE_MARGIN_M,
    ])

    def has_reachable_semantic_endpoint(reference_point: np.ndarray) -> bool:
        return any(
            bool(np.all(candidate >= guarded_lo) and np.all(candidate <= guarded_hi))
            for candidate_offset in clearance_offsets
            for candidate_name in names
            for candidate in [
                reference_point + direction[candidate_name] * candidate_offset
            ]
        )

    if not has_reachable_semantic_endpoint(reference_xy):
        log(
            "stage=place_reference_workspace_gate ok=False "
            f"mapped_xy={reference_xy.round(4).tolist()} "
            "action=safe_look_around_retry_before_pick"
        )
        retry_det, retry_cam, retry_observation = _scene_reference_consensus(
            eye, reference, log, force_look=True
        )
        if retry_det is None or retry_cam is None:
            return None, None, {
                **retry_observation,
                "reason": "placement reference mapped outside the reachable "
                          "workspace and safe look-around did not recover it",
                "initial_reference_xy_m": reference_xy.round(4).tolist(),
            }
        retry_xy, current_rpy, mapping_detail = map_reference(
            retry_cam, retry_observation
        )
        if retry_xy is None:
            return None, None, mapping_detail
        det, reference_cam, observation = \
            retry_det, retry_cam, retry_observation
        reference_xy = retry_xy
        if not has_reachable_semantic_endpoint(reference_xy):
            return None, None, {
                **observation,
                "reason": "placement reference remains outside the reachable "
                          "workspace after safe look-around",
                "reference_xy_m": reference_xy.round(4).tolist(),
                "guarded_workspace_xy_m": [
                    guarded_lo.round(4).tolist(), guarded_hi.round(4).tolist()
                ],
            }
        log(
            "stage=place_reference_workspace_gate ok=True "
            f"mapped_xy={reference_xy.round(4).tolist()} "
            "source=safe_look_around_retry"
        )

    candidates: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    # Try the preferred distance first. Only expand when every requested side
    # fails at that distance, and stop as soon as one complete route passes.
    for clearance_offset in clearance_offsets:
        distance_candidates: list[dict[str, Any]] = []
        for name in names:
            log("stage=place_route_plan "
                f"relation={name} offset_m={clearance_offset:.3f} status=checking")
            endpoint = np.array([
                *(reference_xy + direction[name] * clearance_offset), release_z
            ], dtype=float)
            endpoint_in_bounds = bool(
                np.all(endpoint[:2] >= guarded_lo)
                and np.all(endpoint[:2] <= guarded_hi)
            )
            if not endpoint_in_bounds:
                waypoints = None
                reason = "semantic destination is outside guarded XY workspace"
            elif SCENE_COLLISION_ENABLED and _ACTIVE_SCENE_MAP is not None \
                    and os.getenv("REBOT_VOXEL_PLACE_NAV", "1") != "0":
                # At the ready pose we only freeze semantic destination XY.
                # Carry starts after the grasp, so its voxel route and drop Z
                # are deliberately planned from the measured carry pose later.
                waypoints = []
                reason = "voxel carry planning deferred until after pickup"
            else:
                waypoints, reason = _high_release_waypoints(
                    endpoint, current_rpy, log=log
                )
            edge_clearance = min(
                endpoint[0] - A.X_MIN, A.X_MAX - endpoint[0],
                endpoint[1] - A.Y_MIN, A.Y_MAX - endpoint[1],
            )
            item = {
                "relation": name, "endpoint": endpoint,
                "waypoints": waypoints, "reason": reason,
                "edge_clearance_m": float(edge_clearance),
                "offset_m": clearance_offset,
            }
            candidates.append(item)
            distance_candidates.append(item)
            log("stage=place_route_plan "
                f"relation={name} offset_m={clearance_offset:.3f} "
                f"status={'accepted' if waypoints is not None else 'rejected'} "
                f"detail={reason}")
        valid = [candidate for candidate in distance_candidates
                 if candidate["waypoints"] is not None]
        if valid:
            break
    if not valid:
        return None, None, {
            "reason": "no requested relative side has a complete IK path",
            "candidates": [{
                "relation": item["relation"], "reason": item["reason"],
                "offset_m": round(float(item["offset_m"]), 4),
                "endpoint_m": item["endpoint"].round(4).tolist(),
            } for item in candidates],
        }
    selected = max(valid, key=lambda item: float(item["edge_clearance_m"]))
    detail = {
        **observation,
        "requested_relation": relation,
        "selected_relation": selected["relation"],
        "reference": reference,
        "reference_aligned_xy_m": reference_xy.round(4).tolist(),
        "requested_offset_m": round(offset, 4),
        "offset_m": round(float(selected["offset_m"]), 4),
        "clearance_expanded": bool(float(selected["offset_m"]) > offset + 1e-6),
        "distance_basis": distance_basis,
        "release_endpoint_m": selected["endpoint"].round(4).tolist(),
        "camera_mapping": mapping_detail,
        "collision_model": bool(SCENE_COLLISION_ENABLED),
        "payload_collision_model": bool(
            SCENE_COLLISION_ENABLED and PAYLOAD_COLLISION_ENABLED
        ),
        "warning": (
            (
                "static RGB-D scene, URDF self-collision, table, and attached "
                "payload checked; unseen or moving obstacles still trigger "
                "the operator stop path"
                if PAYLOAD_COLLISION_ENABLED else
                "static RGB-D scene, table, and URDF arm/gripper links are "
                "checked; carried-object collisions are intentionally not "
                "modelled"
            )
            if SCENE_COLLISION_ENABLED else
            "IK/workspace only; scene obstacles are not modelled"
        ),
    }
    return selected["endpoint"], selected["waypoints"], detail


def _support_aware_release_pose(
    desired_object_xy_m: Any,
    context: dict[str, Any],
    current_rpy: Any,
    *,
    support_z_m: float,
    release_gap_m: float = 0.010,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Convert an object-centre placement into the required EE release pose."""
    payload = context.get("payload_box")
    if not isinstance(payload, dict):
        return None, {"reason": "held-object payload geometry is unavailable"}
    try:
        center_offset_ee = np.asarray(
            payload["center_offset_ee_m"], dtype=float
        )
        axes_ee = np.asarray(payload["full_axes_ee"], dtype=float).T
        size = np.asarray(payload["full_size_m"], dtype=float)
        rotation = _rpy_rotation_matrix(current_rpy)
        axes_base = rotation @ axes_ee
        center_offset_base = rotation @ center_offset_ee
        half_vertical_extent = float(np.sum(
            np.abs(axes_base[2, :]) * size / 2.0
        ))
        desired_center = np.array([
            *np.asarray(desired_object_xy_m, dtype=float),
            float(support_z_m) + half_vertical_extent
            + float(np.clip(release_gap_m, 0.005, 0.020)),
        ])
        endpoint = desired_center - center_offset_base
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"reason": f"invalid payload placement geometry: {exc}"}
    return endpoint, {
        "reason": "support-plane and full 3D payload geometry applied",
        "support_z_m": round(float(support_z_m), 5),
        "release_gap_m": round(float(np.clip(release_gap_m, 0.005, 0.020)), 5),
        "object_half_vertical_extent_m": round(half_vertical_extent, 5),
        "desired_object_center_m": desired_center.round(5).tolist(),
        "payload_center_offset_base_m": center_offset_base.round(5).tolist(),
        "release_ee_m": endpoint.round(5).tolist(),
    }


def _environment_route_collision_signature(reason: Any) -> str | None:
    """Identify repeated full-arm collisions that share a route prefix."""
    text = str(reason)
    if "'kind': 'environment'" not in text:
        return None

    def quoted(field: str) -> str | None:
        marker = f"'{field}': '"
        if marker not in text:
            return None
        return text.split(marker, 1)[1].split("'", 1)[0]

    link = quoted("robot_link")
    obstacle = quoted("obstacle")
    if not link or not obstacle:
        return None
    # Segment is diagnostic only. Different waypoint densities can number the
    # same physical departure collision differently, so link+obstacle is the
    # stable signature used to stop futile destination-only retries.
    return f"{link}->{obstacle}"


def _same_side_place_candidates(
    reference_xy_m: Any,
    relation: str,
    starting_offset_m: float,
    context: dict[str, Any],
    placement_detail: dict[str, Any],
    current_rpy: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate bounded 2D alternatives without changing semantic sides.

    Expanding only along the relation ray is insufficient in clutter: every
    farther point can remain behind the same tall reference column or run into
    a second object.  Search radial clearance first, then slide tangentially
    along the requested half-plane.  Resolution comes from the collision map
    and measured payload footprint rather than an object-name strategy.
    """
    directions = {
        "left": np.array([0.0, 1.0]),
        "right": np.array([0.0, -1.0]),
        "front": np.array([1.0, 0.0]),
        "behind": np.array([-1.0, 0.0]),
    }
    if relation not in directions:
        return [], {"reason": f"unsupported fixed placement side {relation!r}"}
    reference_xy = np.asarray(reference_xy_m, dtype=float)
    if reference_xy.shape != (2,) or not np.all(np.isfinite(reference_xy)):
        return [], {"reason": "placement reference XY is invalid"}

    scene = placement_detail.get("scene_map") or {}
    try:
        voxel_m = float(scene.get("voxel_mm", 1000.0 * SCENE_COLLISION_VOXEL_M)) \
            / 1000.0
        inflation_m = float(scene.get(
            "inflation_mm", 1000.0 * SCENE_COLLISION_INFLATION_M
        )) / 1000.0
    except (TypeError, ValueError):
        voxel_m = float(SCENE_COLLISION_VOXEL_M)
        inflation_m = float(SCENE_COLLISION_INFLATION_M)

    payload_half_xy = np.zeros(2, dtype=float)
    payload = context.get("payload_box")
    if PAYLOAD_COLLISION_ENABLED and isinstance(payload, dict):
        try:
            size = np.asarray(payload["full_size_m"], dtype=float)
            axes_ee = np.asarray(payload["full_axes_ee"], dtype=float).T
            axes_base = _rpy_rotation_matrix(current_rpy) @ axes_ee
            if size.shape == (3,) and axes_base.shape == (3, 3):
                payload_half_xy = np.sum(
                    np.abs(axes_base[:2, :]) * size[None, :] / 2.0,
                    axis=1,
                )
        except (KeyError, TypeError, ValueError):
            payload_half_xy = np.zeros(2, dtype=float)

    radial = directions[relation]
    tangent = np.array([-radial[1], radial[0]], dtype=float)
    tangent_payload_half = float(np.dot(np.abs(tangent), payload_half_xy))
    derived_radial_step = max(voxel_m, 2.0 * inflation_m, 0.015)
    derived_tangent_step = max(
        voxel_m + 2.0 * inflation_m,
        0.75 * tangent_payload_half,
        0.025,
    )
    radial_step = float(np.clip(float(os.getenv(
        "REBOT_PLACE_SEARCH_RADIAL_STEP_M", str(derived_radial_step)
    )), 0.015, 0.050))
    tangent_step = float(np.clip(float(os.getenv(
        "REBOT_PLACE_SEARCH_TANGENT_STEP_M", str(derived_tangent_step)
    )), 0.020, 0.070))
    max_radius = float(np.clip(float(os.getenv(
        "REBOT_PLACE_SEARCH_MAX_RADIUS_M", "0.280"
    )), 0.120, 0.350))
    derived_tangent_levels = max(
        2, int(math.ceil(max_radius / max(tangent_step, 1e-6)))
    )
    tangent_levels = int(np.clip(int(os.getenv(
        "REBOT_PLACE_SEARCH_TANGENT_LEVELS", str(derived_tangent_levels)
    )), 1, 12))
    max_candidates = int(np.clip(int(os.getenv(
        "REBOT_PLACE_SEARCH_MAX_CANDIDATES", "64"
    )), 4, 128))
    starting_offset = float(np.clip(starting_offset_m, 0.060, max_radius))

    radial_offsets = [starting_offset]
    next_offset = starting_offset + radial_step
    while next_offset <= max_radius + 1e-9:
        radial_offsets.append(float(min(max_radius, next_offset)))
        next_offset += radial_step
    if radial_offsets[-1] < max_radius - 1e-6:
        radial_offsets.append(max_radius)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add_candidate(radial_offset: float, tangential_offset: float) -> None:
        # Preserve the existing maximum distance from the reference even when
        # the candidate slides sideways within the requested half-plane.
        radius = math.hypot(radial_offset, tangential_offset)
        if radius > max_radius + 1e-9:
            return
        center = reference_xy + radial * radial_offset \
            + tangent * tangential_offset
        key = tuple(int(round(value * 10000.0)) for value in center)
        if key in seen:
            return
        seen.add(key)
        edge_clearance = min(
            center[0] - A.X_MIN, A.X_MAX - center[0],
            center[1] - A.Y_MIN, A.Y_MAX - center[1],
        )
        if edge_clearance < 0.0:
            return
        candidates.append({
            "object_center_xy_m": center,
            "radial_offset_m": float(radial_offset),
            "tangential_offset_m": float(tangential_offset),
            "reference_distance_m": float(radius),
            "edge_clearance_m": float(edge_clearance),
            "requested_displacement_m": float(math.hypot(
                radial_offset - starting_offset, tangential_offset
            )),
        })

    # Build the complete bounded same-side lattice first, then rank it by
    # displacement from the requested semantic destination.  The previous
    # nested-loop order spent its small budget on one strip of the map and
    # could miss a closer free cell in the perpendicular direction.
    for radial_offset in radial_offsets:
        add_candidate(radial_offset, 0.0)
        for tangent_level in range(1, tangent_levels + 1):
            magnitude = tangent_step * tangent_level
            add_candidate(radial_offset, magnitude)
            add_candidate(radial_offset, -magnitude)
    candidates.sort(key=lambda item: (
        float(item["requested_displacement_m"]),
        float(item["reference_distance_m"]),
        -float(item["edge_clearance_m"]),
    ))
    candidates = candidates[:max_candidates]

    return candidates, {
        "reason": "same-side radial and tangential placement lattice",
        "radial_step_m": round(radial_step, 4),
        "tangent_step_m": round(tangent_step, 4),
        "max_reference_radius_m": round(max_radius, 4),
        "tangent_levels": tangent_levels,
        "max_candidates": max_candidates,
        "payload_half_xy_m": payload_half_xy.round(4).tolist(),
        "scene_voxel_m": round(voxel_m, 4),
        "scene_inflation_m": round(inflation_m, 4),
        "candidate_count": len(candidates),
    }


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

    voxel_navigation = bool(
        normalized_relation in _SCENE_RELATIONS
        and SCENE_COLLISION_ENABLED
        and _ACTIVE_SCENE_MAP is not None
        and os.getenv("REBOT_VOXEL_PLACE_NAV", "1") != "0"
    )
    if normalized_relation in _SCENE_RELATIONS and voxel_navigation:
        # The destination XY still comes from scene semantics. Its Z and the
        # entire carry path are derived from measured occupancy after pickup.
        # No support-surface classifier is claimed: the jaws simply release at
        # a bounded clearance over the destination column (or table column).
        assert release_endpoint is not None
        placement_detail["placement_mode"] = "voxel_topdown_drop"
        placement_detail["support_surface_validation"] = False
        placement_detail["support_surface_policy"] = (
            "release above highest destination voxel; table Z when empty"
        )
        scene_waypoints = None
        steps.append((
            "voxel_navigation_mode", True,
            "post-pick XY free-cell routing; no support-surface validation",
        ))
        log(
            "stage=voxel_navigation_mode enabled=true pick_pipeline=unchanged "
            "support_validation=disabled release=drop_over_destination_column"
        )
    elif normalized_relation in _SCENE_RELATIONS:
        scene_map_detail = placement_detail.get("scene_map") or {}
        object_box_detail = context.get("object_box_3d") or {}
        support_z = scene_map_detail.get(
            "table_z_m", object_box_detail.get("support_z_m")
        )
        # Contexts produced by the current pick path always contain the metric
        # object box.  Keep accepting old persisted/manual held contexts which
        # predate that schema, but mark the less-safe legacy Z behavior loudly.
        if support_z is None and not object_box_detail:
            placement_detail["support_plane"] = {
                "used": False,
                "reason": "legacy held context has no metric object box",
            }
            steps.append((
                "support_plane_release_pose", True,
                "legacy held context: retained prevalidated release Z",
            ))
            log(
                "stage=support_plane_release_pose skipped=true "
                "reason=legacy_held_context_without_metric_box"
            )
        elif support_z is None:
            return recover(
                "scene-relative placement has no measured support plane Z"
            )
        else:
            assert release_endpoint is not None
            support_endpoint, support_detail = _support_aware_release_pose(
                release_endpoint[:2], context, A.ee_rpy(),
                support_z_m=float(support_z),
                release_gap_m=float(os.getenv(
                    "REBOT_PLACE_RELEASE_GAP_M", "0.010"
                )),
            )
            steps.append((
                "support_plane_release_pose", support_endpoint is not None,
                str(support_detail.get("reason")),
            ))
            if support_endpoint is None:
                return recover(
                    f"support-aware placement failed: {support_detail.get('reason')}"
                )
            release_endpoint = support_endpoint
            placement_detail["support_plane"] = support_detail
            placement_detail["release_endpoint_m"] = \
                release_endpoint.round(4).tolist()
            # The ready-pose route ended at a provisional EE point. Its semantic XY
            # remains useful, but payload centre/height are known only after pickup.
            scene_waypoints = None
            log(
                "stage=support_plane_release_pose ok=True "
                f"support_z_m={float(support_z):.4f} "
                f"object_center={support_detail['desired_object_center_m']} "
                f"ee_endpoint={release_endpoint.round(4).tolist()}"
            )

    drop_fixed_rpy = tuple(float(v) for v in A.ee_rpy())
    try:
        voxel_detail: dict[str, Any] | None = None
        if voxel_navigation:
            drop_waypoints, release_endpoint, voxel_detail, drop_plan_reason = \
                _voxel_release_waypoints(
                    release_endpoint, drop_fixed_rpy, log, context
                )
            placement_detail["voxel_navigation"] = voxel_detail
            placement_detail["release_endpoint_m"] = \
                release_endpoint.round(4).tolist()
        elif scene_waypoints is not None:
            drop_waypoints = scene_waypoints
            drop_plan_reason = "scene-relative high-release path prevalidated"
            if isinstance(drop_waypoints, dict) and (
                drop_waypoints.get("kind") == "certified_joint_plan"
                and drop_waypoints.get("scene_version") != A.collision_scene_version()
            ):
                log(
                    "stage=drop_route_refresh reason=payload_changed_scene_version "
                    f"planned={drop_waypoints.get('scene_version')} "
                    f"current={A.collision_scene_version()}"
                )
                drop_waypoints, drop_plan_reason = _high_release_waypoints(
                    release_endpoint, drop_fixed_rpy, log=log
                )
        else:
            drop_waypoints, drop_plan_reason = _high_release_waypoints(
                release_endpoint, drop_fixed_rpy, log=log
            )
        # The wide-view placement is intentionally frozen before pickup.  If
        # its exact endpoint or route is blocked at carry time, retain the
        # requested semantic side and search spatial alternatives.  With
        # payload collision disabled these attempts check the URDF arm and
        # gripper only; no object-specific route is assumed.
        payload_replans: list[dict[str, Any]] = []
        if drop_waypoints is None \
                and planned_release_endpoint is not None \
                and normalized_relation in _SCENE_RELATIONS:
            selected_relation = str(
                placement_detail.get("selected_relation")
                or normalized_relation
            )
            reference_xy_raw = placement_detail.get("reference_aligned_xy_m")
            if selected_relation in {
                "left", "right", "front", "behind"
            } and reference_xy_raw is not None:
                starting_offset = float(
                    placement_detail.get("offset_m")
                    or placement_detail.get("requested_offset_m")
                    or distance_m or SCENE_PLACE_OFFSET_M
                )
                candidate_centers, search_detail = \
                    _same_side_place_candidates(
                        reference_xy_raw, selected_relation,
                        starting_offset, context, placement_detail,
                        A.ee_rpy(),
                    )
                placement_detail["payload_replan_search"] = search_detail
                log(
                    "stage=collision_aware_place_search status=generated "
                    f"scope={'arm_gripper_payload' if PAYLOAD_COLLISION_ENABLED else 'arm_gripper'} "
                    f"relation={selected_relation} "
                    f"candidates={len(candidate_centers)} "
                    f"radial_step_m={search_detail.get('radial_step_m')} "
                    f"tangent_step_m={search_detail.get('tangent_step_m')} "
                    f"max_radius_m="
                    f"{search_detail.get('max_reference_radius_m')}"
                )
                yaw_step_deg = float(np.clip(float(os.getenv(
                    "REBOT_PLACE_SEARCH_YAW_STEP_DEG", "15.0"
                )), 5.0, 30.0))
                max_yaw_delta_deg = float(np.clip(float(os.getenv(
                    "REBOT_PLACE_SEARCH_MAX_YAW_DELTA_DEG", "30.0"
                )), 0.0, 45.0))
                yaw_levels = int(math.floor(
                    max_yaw_delta_deg / max(yaw_step_deg, 1e-6)
                ))
                yaw_deltas_deg = [0.0]
                # Rotating a release pose only changes collision occupancy when
                # a measured oriented payload is attached. Legacy/manual held
                # contexts have no geometry with which to justify that search.
                if PAYLOAD_COLLISION_ENABLED \
                        and isinstance(context.get("payload_box"), dict):
                    for level in range(1, yaw_levels + 1):
                        yaw_deltas_deg.extend([
                            yaw_step_deg * level, -yaw_step_deg * level,
                        ])
                orientation_cost_m_per_deg = float(np.clip(float(os.getenv(
                    "REBOT_PLACE_SEARCH_YAW_COST_M_PER_DEG", "0.0005"
                )), 0.0, 0.003))
                max_pose_attempts = int(np.clip(int(os.getenv(
                    "REBOT_PLACE_SEARCH_MAX_POSE_ATTEMPTS", "96"
                )), 8, 256))
                spatial_candidates: list[
                    tuple[float, dict[str, Any], float]
                ] = []
                rotated_candidates: list[
                    tuple[float, dict[str, Any], float]
                ] = []
                for candidate in candidate_centers:
                    displacement = float(candidate.get(
                        "requested_displacement_m", 0.0
                    ))
                    for yaw_delta_deg in yaw_deltas_deg:
                        # The exact requested pose at the current orientation
                        # was already checked immediately above.
                        if displacement <= 1e-9 and abs(yaw_delta_deg) <= 1e-9:
                            continue
                        score = displacement + orientation_cost_m_per_deg * abs(
                            yaw_delta_deg
                        )
                        destination = (
                            rotated_candidates
                            if abs(yaw_delta_deg) > 1e-9
                            else spatial_candidates
                        )
                        destination.append((
                            score, candidate, float(yaw_delta_deg)
                        ))
                rank_key = lambda item: (
                    float(item[0]),
                    float(item[1].get("reference_distance_m", 0.0)),
                    abs(float(item[2])),
                )
                spatial_candidates.sort(key=rank_key)
                rotated_candidates.sort(key=rank_key)
                # Guarantee spatial coverage out to the configured radius.
                # Only the remaining budget is shared by yaw variants, so a
                # long payload cannot consume every attempt near the reference.
                pose_candidates = spatial_candidates[:max_pose_attempts]
                remaining_pose_budget = max(
                    0, max_pose_attempts - len(pose_candidates)
                )
                pose_candidates.extend(
                    rotated_candidates[:remaining_pose_budget]
                )
                pose_candidates.sort(key=rank_key)
                search_detail.update({
                    "yaw_step_deg": round(yaw_step_deg, 1),
                    "max_yaw_delta_deg": round(max_yaw_delta_deg, 1),
                    "pose_candidate_count": len(pose_candidates),
                    "spatial_candidates_reserved": min(
                        len(spatial_candidates), max_pose_attempts
                    ),
                    "rotated_candidates_selected": min(
                        len(rotated_candidates), remaining_pose_budget
                    ),
                    "max_pose_attempts": max_pose_attempts,
                })
                log(
                    "stage=collision_aware_place_pose_search status=ranked "
                    f"poses={len(pose_candidates)} "
                    f"spatial_reserved="
                    f"{search_detail['spatial_candidates_reserved']} "
                    f"yaw_deltas_deg={yaw_deltas_deg} "
                    "ranking=closest_requested_destination_first"
                )
                repeated_prefix_failures: dict[str, int] = {}
                for _score, candidate, yaw_delta_deg in pose_candidates:
                    candidate_center_xy = np.asarray(
                        candidate["object_center_xy_m"], dtype=float
                    )
                    radial_offset = float(candidate["radial_offset_m"])
                    tangential_offset = float(
                        candidate["tangential_offset_m"]
                    )
                    candidate_rpy = np.asarray(drop_fixed_rpy, dtype=float)
                    candidate_rpy[2] += math.radians(yaw_delta_deg)
                    candidate_rpy_tuple = tuple(
                        float(value) for value in candidate_rpy
                    )
                    if voxel_navigation:
                        candidate_endpoint = np.array([
                            *candidate_center_xy,
                            float(np.asarray(release_endpoint)[2]),
                        ])
                        candidate_support = {
                            "reason": "voxel planner determines drop Z",
                        }
                    elif support_z is None and not object_box_detail:
                        candidate_endpoint = np.array([
                            *candidate_center_xy,
                            float(np.asarray(release_endpoint)[2]),
                        ])
                        candidate_support = {
                            "reason": "legacy release Z retained",
                        }
                    else:
                        candidate_endpoint, candidate_support = \
                            _support_aware_release_pose(
                            candidate_center_xy, context, candidate_rpy_tuple,
                            support_z_m=float(support_z),
                            release_gap_m=float(os.getenv(
                                "REBOT_PLACE_RELEASE_GAP_M", "0.010"
                            )),
                        )
                    if candidate_endpoint is None:
                        payload_replans.append({
                            "radial_offset_m": radial_offset,
                            "tangential_offset_m": tangential_offset,
                            "yaw_delta_deg": round(yaw_delta_deg, 1),
                            "ok": False,
                            "reason": candidate_support.get("reason"),
                        })
                        continue
                    log(
                        "stage=collision_aware_place_retry "
                        f"relation={selected_relation} "
                        f"radial_m={radial_offset:.3f} "
                        f"tangent_m={tangential_offset:+.3f} "
                        f"yaw_delta_deg={yaw_delta_deg:+.1f} "
                        f"object_xy={candidate_center_xy.round(4).tolist()} "
                        "status=checking"
                    )
                    if voxel_navigation:
                        candidate_route, candidate_endpoint, candidate_voxel, \
                            candidate_reason = _voxel_release_waypoints(
                                candidate_endpoint, candidate_rpy_tuple, log,
                                context,
                            )
                    else:
                        candidate_route, candidate_reason = \
                            _high_release_waypoints(
                                candidate_endpoint,
                                candidate_rpy_tuple, log=log,
                            )
                        candidate_voxel = None
                    payload_replans.append({
                        "radial_offset_m": radial_offset,
                        "tangential_offset_m": tangential_offset,
                        "reference_distance_m": round(float(
                            candidate["reference_distance_m"]
                        ), 4),
                        "object_center_xy_m": \
                            candidate_center_xy.round(4).tolist(),
                        "endpoint_m": candidate_endpoint.round(4).tolist(),
                        "yaw_delta_deg": round(yaw_delta_deg, 1),
                        "release_rpy_rad": candidate_rpy.round(5).tolist(),
                        "ok": candidate_route is not None,
                        "reason": candidate_reason,
                        "voxel_navigation": candidate_voxel,
                    })
                    log(
                        "stage=collision_aware_place_retry "
                        f"relation={selected_relation} "
                        f"radial_m={radial_offset:.3f} "
                        f"tangent_m={tangential_offset:+.3f} "
                        f"yaw_delta_deg={yaw_delta_deg:+.1f} "
                        f"status={'accepted' if candidate_route is not None else 'rejected'} "
                        f"detail={candidate_reason}"
                    )
                    if candidate_route is None:
                        failure_reasons = [str(candidate_reason)]
                        if isinstance(candidate_voxel, dict):
                            failure_reasons.extend(
                                str(attempt.get("reason"))
                                for attempt in candidate_voxel.get(
                                    "arm_footprint_attempts", []
                                )
                                if isinstance(attempt, dict)
                            )
                        signatures = {
                            signature for signature in (
                                _environment_route_collision_signature(reason)
                                for reason in failure_reasons
                            ) if signature is not None
                        }
                        stop_repeated_prefix = False
                        for signature in signatures:
                            repeated_prefix_failures[signature] = \
                                repeated_prefix_failures.get(signature, 0) + 1
                            if repeated_prefix_failures[signature] >= 3:
                                log(
                                    "stage=collision_aware_place_search "
                                    "status=stopped reason=repeated_shared_"
                                    "departure_collision "
                                    f"signature={signature} attempts="
                                    f"{repeated_prefix_failures[signature]}"
                                )
                                stop_repeated_prefix = True
                        if stop_repeated_prefix:
                            break
                    if candidate_route is not None:
                        release_endpoint = candidate_endpoint
                        drop_waypoints = candidate_route
                        drop_fixed_rpy = candidate_rpy_tuple
                        drop_plan_reason = (
                            "collision-aware same-side clearance expansion; "
                            f"{candidate_reason}"
                        )
                        placement_detail.update({
                            "release_endpoint_m": release_endpoint.round(4).tolist(),
                            "support_plane": candidate_support,
                            "offset_m": radial_offset,
                            "radial_offset_m": radial_offset,
                            "tangential_offset_m": tangential_offset,
                            "reference_distance_m": round(float(
                                candidate["reference_distance_m"]
                            ), 4),
                            "desired_object_center_xy_m": \
                                candidate_center_xy.round(4).tolist(),
                            "release_rpy_rad": candidate_rpy.round(5).tolist(),
                            "placement_yaw_delta_deg": round(
                                yaw_delta_deg, 1
                            ),
                            "clearance_expanded_after_pick": True,
                            "payload_replan_reason": (
                                "cached endpoint was invalid under the active "
                                "collision scope; accepted a bounded same-side "
                                "2D alternative"
                            ),
                            "voxel_navigation": candidate_voxel,
                        })
                        break
                placement_detail["payload_replan_attempts"] = payload_replans
                if drop_waypoints is None and payload_replans:
                    last_reason = str(payload_replans[-1].get("reason"))
                    drop_plan_reason = (
                        "no collision-free same-side release pose among "
                        f"{len(payload_replans)} bounded candidates; last "
                        f"refusal: {last_reason}"
                    )
        steps.append(("prevalidate_drop_trajectory", drop_waypoints is not None,
                      drop_plan_reason))
        if drop_waypoints is None:
            return recover(f"drop trajectory is unsafe: {drop_plan_reason}")

        route_count = (
            len(drop_waypoints.get("joint_path", []))
            if isinstance(drop_waypoints, dict) else len(drop_waypoints)
        )
        route_kind = (
            "global_joint" if isinstance(drop_waypoints, dict) else "cartesian"
        )
        log(f"stage=drop_traverse endpoint={release_endpoint.round(4).tolist()} "
            f"route={route_kind} waypoints={route_count}")
        drop_ok, drop_reason = _execute_fixed_approach_trajectory(
            drop_waypoints, log, fixed_rpy=drop_fixed_rpy
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

        if voxel_navigation:
            # Exit the destination column vertically before the ordinary home
            # motion, avoiding a sideways sweep through the released object.
            retreat_start = np.asarray(A.ee_pos(), dtype=float)
            retreat_endpoint = retreat_start.copy()
            retreat_endpoint[2] = min(
                A.Z_MAX - APPROACH_WORKSPACE_MARGIN_M,
                retreat_start[2] + float(np.clip(float(os.getenv(
                    "REBOT_VOXEL_DROP_RETREAT_M", "0.040"
                )), 0.020, 0.080)),
            )
            retreat_waypoints, retreat_reason = \
                _plan_fixed_approach_trajectory(
                    retreat_endpoint, start_m=retreat_start,
                    fixed_rpy=drop_fixed_rpy,
                )
            retreat_ok = retreat_waypoints is not None
            if retreat_waypoints is not None:
                log(
                    "stage=drop_vertical_retreat "
                    f"distance_mm={(retreat_endpoint[2] - retreat_start[2]) * 1000.0:.1f} "
                    f"waypoints={len(retreat_waypoints)}"
                )
                retreat_ok, retreat_reason = \
                    _execute_fixed_approach_trajectory(
                        retreat_waypoints, log, fixed_rpy=drop_fixed_rpy
                    )
            steps.append((
                "drop_vertical_retreat", retreat_ok, retreat_reason
            ))
            if not retreat_ok:
                return recover(
                    "object released, but vertical retreat failed: "
                    f"{retreat_reason}"
                )

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


def _terminal_stop_key(key: str) -> bool:
    """Q and terminal EOF both revoke permission for continued arm motion."""
    return key == "" or key.lower() == "q"


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
                if _terminal_stop_key(key):
                    log("stage=fixed_pick_hold_stop input=q action=freeze_open_home")
                    return abort_fixed_hold(
                        "terminal closed or Q pressed; gripper opened and arm homed"
                    )
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
                    key = sys.stdin.read(1)
                    if key == "":
                        # A closed terminal must never leave a hardware process
                        # orphaned without its operator stop channel.
                        _os.kill(_os.getpid(), signal.SIGINT)
                        return
                    if _terminal_stop_key(key):
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


def run_fixed_pick_place_cli(
    object_name: str, *, verbose: bool = False,
    place_reference: str | None = None,
    place_relation: str = "drop_zone",
    place_distance_m: float | None = None,
) -> dict[str, Any]:
    """Run live pick/place with Q active during planning and every movement."""
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
                    key = sys.stdin.read(1)
                    if _terminal_stop_key(key):
                        _os.kill(_os.getpid(), signal.SIGINT)
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous_tty)
            listener_ready.set()

    listener = threading.Thread(
        target=listen_for_q, name="wrist-pick-place-q-stop", daemon=True
    )
    listener.start()
    listener_ready.wait(timeout=1.0)
    print(
        "[fixed-pick-place] Q or Ctrl+C stops at any stage: freeze, open "
        "gripper, home.",
        flush=True,
    )
    try:
        return fixed_pick_and_place(
            object_name, dry=False, verbose=verbose,
            place_reference=place_reference,
            place_relation=place_relation,
            place_distance_m=place_distance_m,
        )
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
                det_info = (getattr(det, "info", {}) or {}) \
                    if det is not None else {}
                completion_source = str(det_info.get(
                    "segmentation_source", ""
                ))
                completion = ""
                if det_info.get("object_completion_validated"):
                    completion = (
                        " full_object=sam_complete_query"
                        if completion_source.startswith(
                            "complete_object_semantic_query"
                        ) else " full_object=rgbd_completed"
                    )
                latest_detection = (
                    f"{det.source} depth_mm={det.z_mm:.0f} "
                    f"uv=({det.uv[0]:.0f},{det.uv[1]:.0f}){completion}"
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
        out = (
            run_fixed_pick_place_cli(
                args.fixed_pick_place, verbose=verbose,
                place_reference=args.place_reference,
                place_relation=args.place_relation,
                place_distance_m=args.place_distance_m,
            )
            if args.run else
            fixed_pick_and_place(
                args.fixed_pick_place, dry=True, verbose=verbose,
                place_reference=args.place_reference,
                place_relation=args.place_relation,
                place_distance_m=args.place_distance_m,
            )
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
