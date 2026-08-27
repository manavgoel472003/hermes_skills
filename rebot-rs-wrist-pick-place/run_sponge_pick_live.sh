#!/usr/bin/env bash
set -euo pipefail

# Keep exactly one complete console log for the most recent wrist run.  This
# wrapper is the common execution path for scene pick/place, pick-only, dry,
# and watch commands, so failures from the detector, planner, arm, and recovery
# all land in one predictable file while remaining visible in the terminal.
latest_log="${REBOT_LATEST_RUN_LOG:-/home/manav/rebot_lerobot/logs/latest_wrist_run.log}"
mkdir -p "$(dirname "$latest_log")"
exec > >(tee "$latest_log") 2>&1
export PYTHONUNBUFFERED=1
printf '[launcher] latest run log -> %s\n' "$latest_log"
printf '[launcher] command:'
printf ' %q' "$0" "$@"
printf '\n'

cd /home/manav/rebot_lerobot

export PYTHONPATH="/home/manav/rebot_lerobot/.vendor/sam3_transformers${PYTHONPATH:+:$PYTHONPATH}"
export REBOT_DYNAMIC_GRASP=1
# The viewer/export flag does not enable collision checking. Protect the
# arm/gripper route by default; set this explicitly to 0 only for diagnostics.
export REBOT_SCENE_COLLISION="${REBOT_SCENE_COLLISION:-1}"
# Use one geometry authority per grasp attempt.  The visible checkpoint fixes
# subsection, orientation and width; after the jaws open, RGB-D is presence-only
# and cannot steer or resize the grasp.  One failed, fully recovered attempt may
# retry a different measured subsection, then the workflow terminates.
export REBOT_SIMPLE_GRASP_FLOW="${REBOT_SIMPLE_GRASP_FLOW:-1}"
export REBOT_SIMPLE_GRASP_MAX_ATTEMPTS="${REBOT_SIMPLE_GRASP_MAX_ATTEMPTS:-2}"
# Do not spend eight identical frames proving that a clipped ready view is still
# clipped.  After three misses the existing bounded look-around takes over. One
# additional translated discovery sweep is enough to establish/converge the
# hybrid metric box; a second whole sweep only repeated the same evidence.
export REBOT_HYBRID_SEMANTIC_ANCHOR_ATTEMPTS="${REBOT_HYBRID_SEMANTIC_ANCHOR_ATTEMPTS:-3}"
export REBOT_HYBRID_EXTRA_VIEW_SWEEPS="${REBOT_HYBRID_EXTRA_VIEW_SWEEPS:-1}"
# Plan the macro pick route once from the complete initial scene, execute only
# its collision-checked prefix to a dynamically selected visible checkpoint,
# then require fresh stationary RGB-D body consensus before replacing the short
# final leg. The replacement is fully IK/collision validated before motion;
# endpoint vision confirms presence but cannot perform another alignment.
export REBOT_FIXED_SCENE_TRAJECTORY="${REBOT_FIXED_SCENE_TRAJECTORY:-1}"
export REBOT_WRIST_CLOSE_GRIPPER_AT_START=1
export REBOT_CAN_BITRATE=1000000
export REBOT_ENABLE_LOCATE_ANYTHING=1
export REBOT_VISION_BACKEND="${REBOT_VISION_BACKEND:-sam3_1}"
# Full SAM 3.1 video propagation exceeds this machine's 8 GB VRAM. The
# improved 3.1 model supplies each exact mask; the existing multi-view RGB-D
# registration retains temporal identity and metric geometry.
export REBOT_VISION_TRACKER="${REBOT_VISION_TRACKER:-none}"
export REBOT_VISION_PRIMARY=1
export REBOT_SAM3_MODEL=facebook/sam3
export REBOT_SAM31_SCORE_THRESHOLD="${REBOT_SAM31_SCORE_THRESHOLD:-0.30}"
# Keep SAM 3.1 authoritative for every discovery view. Switching to the older
# SAM3 backend after the first proposal allowed broad masks from later views to
# absorb neighbouring objects and inflate the discovery target. Single-frame
# inference remains bounded; full video propagation is still disabled.
export REBOT_SAM31_BOOTSTRAP_ONLY="${REBOT_SAM31_BOOTSTRAP_ONLY:-0}"
export REBOT_VISION_DEVICE=cuda
export REBOT_VISION_DTYPE=bfloat16
export REBOT_SAM3_SCORE_THRESHOLD=0.30
export REBOT_SAM3_MASK_THRESHOLD=0.50
export REBOT_WS_X_MIN=0.150
export REBOT_WS_X_MAX=0.540
export REBOT_WS_Y_MIN=-0.300
export REBOT_WS_Y_MAX=0.300
export REBOT_WS_Z_MIN=0.000
export REBOT_WS_Z_MAX=0.360
export REBOT_WRIST_NEAR_MM=60
export REBOT_WRIST_FAR_MM=600
export REBOT_WRIST_PICK_CLOSER_MM=4
export REBOT_WRIST_PICK_Y_OFFSET_MM=8
export REBOT_WRIST_PICK_TARGET_TOL_MM=20
export REBOT_WRIST_PICK_MAX_CORRECTIONS=3
export REBOT_WRIST_PICK_MAX_CORRECTION_M=0.012
export REBOT_WRIST_FINAL_ALIGN_GAIN=0.85
export REBOT_WRIST_RIGID_FIT_MAX_ERROR=0.15
export REBOT_WRIST_USE_SAVED_CAMERA_ROTATION=1
# Permit the same bounded wrist-yaw envelope used by grasp geometry. The saved
# rigid camera mount is rotated analytically to the current wrist orientation;
# complete IK and collision checks still run before motion.
# The physical grasp yaw is capped at 20deg below. Leave 2.5deg for the small
# roll/pitch already present at the ready pose and for tracking uncertainty.
# The adjusted rigid camera transform, every intermediate orientation waypoint,
# IK, and collision model are still validated before execution.
export REBOT_WRIST_SAVED_ROTATION_MAX_RPY_DELTA_DEG=22.5
export REBOT_WRIST_PICK_OPEN_BEFORE_END_M=0.040
export REBOT_WRIST_JAW_LONG_AXIS_CAMERA_DEG=-90
export REBOT_WRIST_GRASP_MAX_YAW_DELTA_DEG=20
# Some objects, especially a sponge resting at an arbitrary angle, cannot put
# their narrow cross-section between the jaws with yaw alone.  This is an
# opt-in bounded *search* for the sponge/general-object launcher, not a
# relaxation of safety: each candidate and every intermediate orientation is
# still checked with the full IK, workspace, self-collision and scene-collision
# gates before it can move.  The ordinary code defaults remain conservative for
# callers that do not use this launcher.
export REBOT_3D_AXIS_REFINE_MAX_ROLL_DEG="${REBOT_3D_AXIS_REFINE_MAX_ROLL_DEG:-15}"
export REBOT_3D_AXIS_REFINE_MAX_PITCH_DEG="${REBOT_3D_AXIS_REFINE_MAX_PITCH_DEG:-30}"
export REBOT_3D_AXIS_REFINE_MAX_YAW_DEG="${REBOT_3D_AXIS_REFINE_MAX_YAW_DEG:-30}"
export REBOT_3D_AXIS_REFINE_CAMERA_ENVELOPE_DEG="${REBOT_3D_AXIS_REFINE_CAMERA_ENVELOPE_DEG:-45}"
# A marginal object-axis error may be absorbed without exceeding the physical
# yaw command bound.  Grasp geometry projects the residual angle into a larger
# required jaw span and refuses the pick if that span does not fit.
export REBOT_WRIST_GRASP_YAW_MEASUREMENT_SLOP_DEG="${REBOT_WRIST_GRASP_YAW_MEASUREMENT_SLOP_DEG:-1.5}"
# Do not command beyond the wrist limit. If a longer object remains slightly
# skewed, project that residual angle into its required jaw span and proceed
# only when the resulting span still fits the calibrated 50 mm aperture.
export REBOT_WRIST_GRASP_MAX_RESIDUAL_YAW_DEG="${REBOT_WRIST_GRASP_MAX_RESIDUAL_YAW_DEG:-6}"
# Keep the physical wrist command at 20 degrees. If vision requests more, use
# measured object width/length to compute the actual span presented to the jaws;
# proceed only when that conservative projection still fits the calibrated
# aperture. This replaces an object-independent 6 degree rejection with a
# geometry-derived gate, without relaxing IK or collision limits.
export REBOT_WRIST_GRASP_APERTURE_DERIVED_RESIDUAL="${REBOT_WRIST_GRASP_APERTURE_DERIVED_RESIDUAL:-1}"
export REBOT_WRIST_GRASP_RPY_STEP_DEG=2
export REBOT_WRIST_REACH_PITCH_BIAS_DEG=10
# If every measured XYZ/slot grasp is blocked at the terminal configuration,
# search pitch (not grasp yaw) in bounded 5deg increments. The wider camera
# delta applies only to this preplanned escape: the full OBB must still fit the
# 50mm jaws and every pose must pass IK, table, self-, and scene-collision.
export REBOT_GRASP_COLLISION_PITCH_STEP_DEG="${REBOT_GRASP_COLLISION_PITCH_STEP_DEG:-5}"
export REBOT_GRASP_COLLISION_MAX_PITCH_DEG="${REBOT_GRASP_COLLISION_MAX_PITCH_DEG:-20}"
export REBOT_GRASP_COLLISION_CAMERA_DELTA_DEG="${REBOT_GRASP_COLLISION_CAMERA_DELTA_DEG:-40}"
export REBOT_WRIST_GRASP_FINAL_APPROACH_MM=40
export REBOT_WRIST_GRASP_MIN_FINAL_APPROACH_MM=20
export REBOT_WRIST_GRASP_MIN_FEATURE_PROMINENCE_MM=25
export REBOT_DYNAMIC_GRIPPER_SLOT_MARGIN_MM=8
export REBOT_DYNAMIC_GRIPPER_REMEASURE_RADIUS_MM=6
export REBOT_DYNAMIC_GRIPPER_CLOSE_STEP_DEG=18
export REBOT_GRIPPER_SIZE_MARGIN_MM=0
# Position-close 4 mm below the stable image/depth width. Calibrated jaw limits,
# bounded position ramps, and visual lift verification remain mandatory. The
# endpoint camera gate separately requires enough centring for both fingers to
# overlap the measured body before this close begins.
export REBOT_GRIPPER_SIZE_COMPRESSION_MM="${REBOT_GRIPPER_SIZE_COMPRESSION_MM:-4}"
export REBOT_DYNAMIC_MICRO_LIFT_M="${REBOT_DYNAMIC_MICRO_LIFT_M:-0.015}"
export REBOT_GRASP_HOLD_FRAMES=6
# A failed visual hold probe may retry only after returning through the
# validated inverse path and proving the locked grasp section (or its stable
# semantic offset) is still inside the jaw slot. Each retry is at most 1 mm
# tighter with a calibrated jaw curve, or 5 degrees with the single-span model.
export REBOT_DYNAMIC_RETRY_CLOSE_MM="${REBOT_DYNAMIC_RETRY_CLOSE_MM:-1.0}"
export REBOT_DYNAMIC_RETRY_CLOSE_DEG="${REBOT_DYNAMIC_RETRY_CLOSE_DEG:-5.0}"
export REBOT_DYNAMIC_MAX_RETRIES="${REBOT_DYNAMIC_MAX_RETRIES:-3}"
export REBOT_LOCAL_IK_POS_TOL_M=0.004
export REBOT_IK_MAX_ITER=400
# About 20% faster than the previous tested profile. Minimum-jerk timing,
# per-joint velocity enforcement, waypoint collision checks, and endpoint
# settling remain active.
export REBOT_ARM_SPEED_M_S=0.036
export REBOT_ARM_MIN_MOVE_S=0.42
export REBOT_ARM_ANGULAR_SPEED_RAD_S=0.60
export REBOT_JOINT_VLIM=0.60
# Image-derived closure still approaches monotonically without contact
# inference. An 18deg outer step remains below the hard 20deg bound; every
# backend step is itself ramped rather than applied as a target jump.
export REBOT_GRIPPER_MOVE_S=0.95
export REBOT_GRIPPER_STEP_S=0.035

case "${1:---run}" in
  --run) mode_args=(--run --live) ;;
  --dry) mode_args=(--dry --view) ;;
  --pick-only) mode_args=(--run --live); operation=--fixed-pick ;;
  --pick-dry) mode_args=(--dry --view); operation=--fixed-pick ;;
  --watch) mode_args=(--live); operation=--watch ;;
  *) echo "usage: $0 [--watch|--dry|--run|--pick-only|--pick-dry] [object description]" >&2; exit 2 ;;
esac
object_description="${2:-green and yellow rectangular kitchen sponge}"
operation="${operation:---fixed-pick-place}"
extra_args=("${@:3}")

if [[ "$operation" == "--watch" ]]; then
  exec /home/manav/miniconda3/envs/rebot_rs/bin/python \
    -m arm_agent.wrist_servo \
    --watch --object "$object_description" "${mode_args[@]}" "${extra_args[@]}"
fi

exec /home/manav/miniconda3/envs/rebot_rs/bin/python \
  -m arm_agent.wrist_servo \
  "$operation" "$object_description" "${mode_args[@]}" "${extra_args[@]}"
