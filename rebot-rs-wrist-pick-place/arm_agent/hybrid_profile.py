"""One opt-in wrist-only deployment profile for hybrid CLI and MCP.

Apply before importing wrist_servo/arm_safe: those modules snapshot environment
settings at import. Existing launchers remain unchanged. Physical calibration is
read from the existing calib directory, never rewritten by this profile.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "REBOT_ARM_BACKEND": "local",
    "REBOT_WRIST_CAMERA": "realsense",
    "REBOT_DYNAMIC_GRASP": "1",
    "REBOT_SCENE_COLLISION": "1",
    "REBOT_CARRY_PAYLOAD_COLLISION": "1",
    "REBOT_PAYLOAD_COLLISION": "0",
    "REBOT_FIXED_SCENE_TRAJECTORY": "1",
    "REBOT_SIMPLE_GRASP_FLOW": "1",
    "REBOT_SIMPLE_GRASP_MAX_ATTEMPTS": "2",
    "REBOT_OBJECT_GEOMETRY_MODE": "legacy",
    # If semantic whole-object geometry cannot yield a gripper-sized section,
    # repeat the bounded Claw-style view sweep using an exact colour-seeded
    # physical component. This never relaxes calibration or gripper limits.
    "REBOT_REFLEX_COLOR_GRASP_FALLBACK": "1",
    # The Claw adaptation starts only after Hermes has frozen a validated
    # metric object box. It consumes that box directly; no second VLM lookup
    # is used to propose its grasp axes or terminal pose alternatives.
    "REBOT_CLAW_BOX_REFLEX_FALLBACK": "fallback",
    # Grasp safety: the wrist camera decides the exact aperture, but it never
    # turns a depth estimate or an encoder shortfall into extra squeeze force.
    # In particular, do not drive the repaired gripper into its closed stop at
    # startup, undersize a measured object, or retry tighter after a failed
    # visual hold check.
    "REBOT_WRIST_CLOSE_GRIPPER_AT_START": "0",
    "REBOT_GRIPPER_SIZE_COMPRESSION_MM": "0",
    "REBOT_GRIPPER_SIZE_MARGIN_MM": "0",
    "REBOT_GRIPPER_CLAMP_EXTRA_MM": "0",
    "REBOT_GRIPPER_CLOSE_SETTLE_RETRY": "0",
    "REBOT_DYNAMIC_MAX_RETRIES": "0",
    "REBOT_DYNAMIC_RETRY_CLOSE_MM": "0",
    "REBOT_DYNAMIC_RETRY_CLOSE_DEG": "0",
    "REBOT_CAN_BITRATE": "1000000",
    "REBOT_ENABLE_LOCATE_ANYTHING": "1",
    "REBOT_VISION_BACKEND": "sam3_1",
    "REBOT_VISION_TRACKER": "none",
    "REBOT_SAM31_BOOTSTRAP_ONLY": "0",
    "REBOT_VISION_PRIMARY": "1",
    "REBOT_SAM3_MODEL": "facebook/sam3",
    "REBOT_VISION_DEVICE": "cuda",
    "REBOT_VISION_DTYPE": "bfloat16",
    "REBOT_WS_X_MIN": "0.150", "REBOT_WS_X_MAX": "0.540",
    "REBOT_WS_Y_MIN": "-0.300", "REBOT_WS_Y_MAX": "0.300",
    "REBOT_WS_Z_MIN": "0.000", "REBOT_WS_Z_MAX": "0.360",
    "REBOT_WRIST_NEAR_MM": "60", "REBOT_WRIST_FAR_MM": "600",
    "REBOT_WRIST_PICK_CLOSER_MM": "4",
    "REBOT_WRIST_PICK_Y_OFFSET_MM": "8",
    "REBOT_WRIST_PICK_TARGET_TOL_MM": "20",
    "REBOT_WRIST_PICK_MAX_CORRECTIONS": "3",
    "REBOT_WRIST_PICK_MAX_CORRECTION_M": "0.012",
    "REBOT_WRIST_FINAL_ALIGN_GAIN": "0.85",
    "REBOT_WRIST_RIGID_FIT_MAX_ERROR": "0.15",
    "REBOT_WRIST_USE_SAVED_CAMERA_ROTATION": "1",
    "REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG": "22.5",
    "REBOT_WRIST_PICK_OPEN_BEFORE_END_M": "0.040",
    "REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG": "-90",
    "REBOT_WRIST_GRASP_MAX_YAW_DELTA_DEG": "20",
    "REBOT_WRIST_GRASP_MAX_RESIDUAL_YAW_DEG": "6",
    "REBOT_WRIST_GRASP_RPY_STEP_DEG": "2",
    "REBOT_WRIST_GRASP_APERTURE_DERIVED_RESIDUAL": "1",
    "REBOT_WRIST_GRASP_FINAL_APPROACH_MM": "40",
    "REBOT_WRIST_GRASP_MIN_FINAL_APPROACH_MM": "20",
    "REBOT_DYNAMIC_GRIPPER_SLOT_MARGIN_MM": "8",
    "REBOT_DYNAMIC_GRIPPER_CLOSE_STEP_DEG": "18",
    "REBOT_DYNAMIC_MICRO_LIFT_M": "0.015",
    "REBOT_GRASP_HOLD_FRAMES": "6",
    "REBOT_SCENE_COLLISION_VOXEL_M": "0.020",
    "REBOT_SCENE_COLLISION_INFLATION_M": "0.003",
    "REBOT_SCENE_FUSION_FRAMES": "3",
    "REBOT_VOXEL_PAYLOAD_MARGIN_M": "0.010",
    "REBOT_PLACE_RELEASE_GAP_M": "0.010",
    "REBOT_LOCAL_IK_POS_TOL_M": "0.004",
    "REBOT_IK_MAX_ITER": "400",
    "REBOT_ARM_SPEED_M_S": "0.036",
    "REBOT_ARM_MIN_MOVE_S": "0.42",
    "REBOT_ARM_ANGULAR_SPEED_RAD_S": "0.60",
    "REBOT_JOINT_VLIM": "0.60",
    "REBOT_GRIPPER_MOVE_S": "0.95",
    "REBOT_GRIPPER_STEP_S": "0.035",
}


def configure() -> dict[str, str]:
    if "arm_agent.wrist_servo" in sys.modules or "arm_agent.arm_safe" in sys.modules:
        raise RuntimeError("Start hybrid in a fresh process; wrist configuration is already loaded")
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    # Required semantics; do not silently downgrade a hybrid task.
    for key in ("REBOT_DYNAMIC_GRASP", "REBOT_SCENE_COLLISION",
                "REBOT_CARRY_PAYLOAD_COLLISION", "REBOT_FIXED_SCENE_TRAJECTORY",
                "REBOT_SIMPLE_GRASP_FLOW", "REBOT_WRIST_USE_SAVED_CAMERA_ROTATION"):
        if os.environ[key].lower() not in {"1", "true", "yes", "on"}:
            raise ValueError(f"hybrid requires {key}=1")
    if os.environ["REBOT_WRIST_CLOSE_GRIPPER_AT_START"].lower() in {
        "1", "true", "yes", "on"
    }:
        raise ValueError("hybrid forbids an automatic gripper hard-close at startup")
    if os.environ["REBOT_GRIPPER_CLOSE_SETTLE_RETRY"].lower() in {
        "1", "true", "yes", "on"
    }:
        raise ValueError("hybrid forbids a second gripper close after shortfall")
    for key in ("REBOT_GRIPPER_SIZE_COMPRESSION_MM",
                "REBOT_GRIPPER_SIZE_MARGIN_MM",
                "REBOT_GRIPPER_CLAMP_EXTRA_MM",
                "REBOT_DYNAMIC_RETRY_CLOSE_MM",
                "REBOT_DYNAMIC_RETRY_CLOSE_DEG"):
        try:
            value = float(os.environ[key])
        except ValueError as exc:
            raise ValueError(f"hybrid requires numeric {key}=0") from exc
        if abs(value) > 1e-9:
            raise ValueError(f"hybrid forbids {key} other than 0")
    try:
        retry_count = int(os.environ["REBOT_DYNAMIC_MAX_RETRIES"])
    except ValueError as exc:
        raise ValueError("hybrid requires integer REBOT_DYNAMIC_MAX_RETRIES=0") from exc
    if retry_count != 0:
        raise ValueError("hybrid forbids tighter gripper retries")
    if os.environ["REBOT_ARM_BACKEND"] != "local" or os.environ["REBOT_WRIST_CAMERA"] != "realsense":
        raise ValueError("This profile requires the local RS backend and wrist RealSense calibration")
    vendor = ROOT / ".vendor" / "sam3_transformers"
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return {key: os.environ[key] for key in DEFAULTS}
