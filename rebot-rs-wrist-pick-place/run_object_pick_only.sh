#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'usage: ./run_object_pick_only.sh "specific object description"' >&2
  exit 2
fi

cd /home/manav/rebot_lerobot
exec ./run_sponge_pick_live.sh --pick-only "$1"
