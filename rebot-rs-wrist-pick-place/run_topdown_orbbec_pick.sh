#!/usr/bin/env bash
# Experimental top-down Orbbec RGB-D path.  It never changes wrist_servo.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo 'usage: run_topdown_orbbec_pick.sh --watch "object" | --calibrate-camera --run | --approach "object" [--run]' >&2
  exit 2
fi

export REBOT_DEPTH_CAMERA=orbbec
export REBOT_TOPDOWN_GRIPPER_MODEL="${REBOT_TOPDOWN_GRIPPER_MODEL:-/home/manav/rebot_lerobot/runs/gripper_segmentation/topdown_yolo11n_orbbec_finetune/weights/best.pt}"
# Colour-only detection selected a neighbouring green/yellow object in the
# overhead view.  Use the same text-conditioned whole-object segmentation
# backend as the validated wrist workflow; it supplies a target mask before
# aligned Orbbec depth establishes the 3-D point.
export PYTHONPATH="/home/manav/rebot_lerobot/.vendor/sam3_transformers${PYTHONPATH:+:$PYTHONPATH}"
export REBOT_ENABLE_LOCATE_ANYTHING=1
export REBOT_VISION_PRIMARY=1
export REBOT_VISION_BACKEND="${REBOT_VISION_BACKEND:-sam3_1}"
export REBOT_VISION_TRACKER=none
export REBOT_SAM3_MODEL=facebook/sam3
export REBOT_SAM31_SCORE_THRESHOLD="${REBOT_SAM31_SCORE_THRESHOLD:-0.30}"
export REBOT_VISION_DEVICE=cuda
export REBOT_VISION_DTYPE=bfloat16

# A watch invocation owns a small loopback-only viewer.  It renders the same
# latest.jpg that the Python process writes, and the HTML cache-busts it so the
# browser is a real live feed rather than a stale JPEG tab.
if [[ "${1:-}" == "--watch" ]]; then
  view_dir="${REBOT_TOPDOWN_DEBUG_DIR:-/tmp/rebot-topdown-orbbec}"
  view_port="${REBOT_TOPDOWN_VIEW_PORT:-8766}"
  mkdir -p "$view_dir"
  cp "$(cd "$(dirname "$0")" && pwd)/tools/topdown_viewer.html" "$view_dir/index.html"
  viewer_pid=""
  if ! ss -ltn 2>/dev/null | grep -q ":${view_port}[[:space:]]"; then
    python3 -m http.server "$view_port" --bind 127.0.0.1 --directory "$view_dir" \
      >/tmp/rebot-topdown-viewer.log 2>&1 &
    viewer_pid="$!"
  fi
  cleanup_viewer() {
    [[ -z "$viewer_pid" ]] || kill "$viewer_pid" 2>/dev/null || true
  }
  trap cleanup_viewer EXIT INT TERM
  (sleep .4; xdg-open "http://127.0.0.1:${view_port}/" >/dev/null 2>&1 || true) &
fi
# Run this branch's module directly but import the hardware backend from the
# robot workspace, where the private RS controller package is installed.
 /home/manav/miniconda3/envs/rebot_rs/bin/python \
  "$(cd "$(dirname "$0")" && pwd)/arm_agent/topdown_orbbec_pick.py" "$@"
