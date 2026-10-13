#!/usr/bin/env bash
# Validate the bundled RS controller source expected by arm_agent.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/reBotArm_control_py_fresh"
if [[ ! -f "$DEST/pyproject.toml" || ! -f "$DEST/config/rebotarm_rs.yaml" ]]; then
  echo "Bundled RS controller source is incomplete: $DEST" >&2
  exit 1
fi
echo "Bundled controller validated at $DEST"
echo "Next: python -m pip install -e $DEST"
