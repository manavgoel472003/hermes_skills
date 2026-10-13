#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
preview_python="${HYBRID_PYTHON:-/home/manav/miniconda3/envs/rebot_rs/bin/python}"
exec "$preview_python" -m arm_agent.claw_box_preview "$@"
