#!/usr/bin/env bash
# Direct OBB-reflex executor: initial RGB-D scene/box acquisition, then a
# frozen prevalidated macro and terminal leg.  It deliberately bypasses the
# Hermes visible-checkpoint semantic re-observation/replan stage.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export REBOT_CLAW_DIRECT_REFLEX=1
export REBOT_CLAW_BOX_REFLEX_FALLBACK=force
export REBOT_REFLEX_COLOR_GRASP_FALLBACK=0

hybrid_python="${HYBRID_PYTHON:-/home/manav/miniconda3/envs/rebot_rs/bin/python}"
exec "$hybrid_python" -m arm_agent.hybrid_mission "$@"
