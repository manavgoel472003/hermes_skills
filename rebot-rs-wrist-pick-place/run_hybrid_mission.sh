#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hybrid_python="${HYBRID_PYTHON:-/home/manav/miniconda3/envs/rebot_rs/bin/python}"
exec "$hybrid_python" -m arm_agent.hybrid_mission "$@"
