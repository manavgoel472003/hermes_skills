#!/usr/bin/env bash
set -euo pipefail

mode=--dry
if [[ "${1:-}" == "--dry" || "${1:-}" == "--run" || "${1:-}" == "--watch" ]]; then
  mode="$1"
  shift
fi
if [[ $# -ne 1 || -z "$1" ]]; then
  echo 'usage: run_object_pick.sh [--watch|--dry|--run] "specific object description"' >&2
  exit 2
fi

# This launcher is intentionally pick-only.  Scene-relative placement owns the
# coarse RGB-D obstacle map; letting that map leak into a standalone pick can
# leave a target-overlap column at the exact jaw endpoint and veto intentional
# gripper/object contact.  Pick-only still enforces workspace/table bounds,
# joint limits, URDF self-collision, complete IK prevalidation, bounded visual
# alignment, calibrated jaw limits, and image/depth-based closure and lift
# verification.
export REBOT_SCENE_COLLISION=0

case "$mode" in
  --watch)
    exec /home/manav/rebot_lerobot/run_sponge_pick_live.sh --watch "$1"
    ;;
  --dry)
    exec /home/manav/rebot_lerobot/run_sponge_pick_live.sh --pick-dry "$1"
    ;;
  --run)
    exec /home/manav/rebot_lerobot/run_sponge_pick_live.sh --pick-only "$1"
    ;;
esac
