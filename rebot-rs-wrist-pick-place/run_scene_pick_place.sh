#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry}"
if [[ "$mode" != "--dry" && "$mode" != "--run" ]]; then
  echo 'first argument must be --dry or --run' >&2
  exit 2
fi
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo 'usage: ./run_scene_pick_place.sh [--dry|--run] "pick object" RELATION "reference object" [distance_m]' >&2
  echo 'relations: left right front behind next_to stack' >&2
  exit 2
fi

pick_object="$2"
relation="$3"
reference="$4"
distance_args=()
if [[ $# -eq 5 ]]; then
  distance_args=(--place-distance-m "$5")
fi

cd /home/manav/rebot_lerobot
# Scene-relative runs opt into the fail-closed RGB-D/URDF collision gate. The
# generic object-pick wrapper remains unchanged so this can be validated on the
# scene workflow before becoming the global default.
export REBOT_SCENE_COLLISION=1
# Keep the proven stationary RGB-D object geometry as the default. Set this to
# "hybrid" for pose-aware fusion of the selected object's semantic masks from
# multiple bounded wrist views. The two paths are deliberately explicit: a
# hybrid reconstruction failure stops safely and never changes legacy behavior.
export REBOT_OBJECT_GEOMETRY_MODE="${REBOT_OBJECT_GEOMETRY_MODE:-legacy}"
# Keep the grasp phase free of an attached collision body because the measured
# payload necessarily overlaps the selected pick target while the jaws close.
# The separately configured carry phase below attaches that same measured OBB
# after the target-excluded scene map is restored.
export REBOT_PAYLOAD_COLLISION="${REBOT_PAYLOAD_COLLISION:-0}"
# Pick-target contact is exempted above, but once the object has been lifted its
# measured OBB must participate in full-arm collision validation. A failed
# attachment is now fatal rather than silently falling back to an empty gripper.
export REBOT_CARRY_PAYLOAD_COLLISION="${REBOT_CARRY_PAYLOAD_COLLISION:-1}"
export REBOT_SCENE_COLLISION_VOXEL_M="${REBOT_SCENE_COLLISION_VOXEL_M:-0.020}"
# Preserve 3 mm of uncertainty around each occupied column. With the 20 mm
# coarse voxel this yields a 26 mm protected footprint. Link geometry, the
# measured table, and URDF self-collision checks remain unchanged.
export REBOT_SCENE_COLLISION_INFLATION_M="${REBOT_SCENE_COLLISION_INFLATION_M:-0.003}"
# Keep uncertainty local to the moving payload rather than making every scene
# obstacle look larger. This protects against a few millimetres of RGB-D/FK
# registration error when a carried object passes a mapped obstacle.
export REBOT_VOXEL_PAYLOAD_MARGIN_M="${REBOT_VOXEL_PAYLOAD_MARGIN_M:-0.010}"
export REBOT_SCENE_COLLISION_STRIDE_PX="${REBOT_SCENE_COLLISION_STRIDE_PX:-4}"
# Fuse a short stationary RGB-D burst. A column must appear in a strict majority
# of the frames, so one bad depth sample cannot become a permanent obstacle.
# Points are transformed into the base frame before the 3 mm inflation is applied.
export REBOT_SCENE_FUSION_FRAMES="${REBOT_SCENE_FUSION_FRAMES:-3}"
# Keep inflated neighboring columns from leaking back into the one object the
# gripper is intentionally allowed to contact. This padding applies only around
# the selected pick target's semantic/depth envelope.
export REBOT_SCENE_COLLISION_TARGET_PAD_PX="${REBOT_SCENE_COLLISION_TARGET_PAD_PX:-20}"
# Dense point-cloud export is diagnostic-only and is deliberately off by
# default. The coarse occupied-volume map and URDF collision gate remain on.
export REBOT_SCENE_COLLISION_EXPORT_VIEW="${REBOT_SCENE_COLLISION_EXPORT_VIEW:-0}"
# The default recording shows the one fused cloud and the collision volumes the
# planner actually uses. Historical per-view clouds and 70+ cell labels can be
# enabled explicitly for registration/blocker debugging without changing any
# collision decision.
export REBOT_SCENE_EXPORT_PER_VIEW_CLOUDS="${REBOT_SCENE_EXPORT_PER_VIEW_CLOUDS:-0}"
export REBOT_SCENE_EXPORT_BOX_LABELS="${REBOT_SCENE_EXPORT_BOX_LABELS:-0}"
# A clear direct corridor remains the cheapest option. If it is blocked, the
# carry planner immediately tries full-joint-space RRT-Connect (free yaw and
# bounded roll/pitch) before falling back to inspectable XYZ corridors.
export REBOT_GLOBAL_PLANNER_TIMEOUT_S="${REBOT_GLOBAL_PLANNER_TIMEOUT_S:-4}"
export REBOT_GLOBAL_PLANNER_MAX_NODES="${REBOT_GLOBAL_PLANNER_MAX_NODES:-4000}"
export REBOT_GLOBAL_PLANNER_MAX_TILT_DEG="${REBOT_GLOBAL_PLANNER_MAX_TILT_DEG:-12}"
export REBOT_GLOBAL_PLAN_MAX_AGE_S="${REBOT_GLOBAL_PLAN_MAX_AGE_S:-30}"
# If the exact release pose is occupied after the measured payload is attached,
# search the requested semantic half-plane at the collision-map resolution.
# Candidates are ranked by deviation from the requested point, with bounded yaw
# alternatives for long payloads before the existing Cartesian/RRT planners run.
export REBOT_PLACE_SEARCH_MAX_RADIUS_M="${REBOT_PLACE_SEARCH_MAX_RADIUS_M:-0.280}"
export REBOT_PLACE_SEARCH_MAX_CANDIDATES="${REBOT_PLACE_SEARCH_MAX_CANDIDATES:-64}"
export REBOT_PLACE_SEARCH_MAX_POSE_ATTEMPTS="${REBOT_PLACE_SEARCH_MAX_POSE_ATTEMPTS:-96}"
export REBOT_PLACE_SEARCH_YAW_STEP_DEG="${REBOT_PLACE_SEARCH_YAW_STEP_DEG:-15}"
export REBOT_PLACE_SEARCH_MAX_YAW_DELTA_DEG="${REBOT_PLACE_SEARCH_MAX_YAW_DELTA_DEG:-30}"
# Placement Z is derived from the calibrated support plane and the held
# object's persistent metric 3D box.  Ten millimetres is release clearance,
# not obstacle inflation; the planner checks the complete attached payload.
export REBOT_PLACE_RELEASE_GAP_M="${REBOT_PLACE_RELEASE_GAP_M:-0.010}"
# SAM 3D Objects is an optional completion service (it does not fit the local
# 8 GB control GPU). When REBOT_SAM3D_ENDPOINT is unset, semantic masks plus
# aligned RealSense depth remain the authoritative metric geometry.
exec ./run_sponge_pick_live.sh "$mode" "$pick_object" \
  --place-relation "$relation" \
  --place-reference "$reference" \
  "${distance_args[@]}"
