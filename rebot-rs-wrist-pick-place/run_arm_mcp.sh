#!/usr/bin/env bash
# Launches the reBot arm MCP server (stdio) for the hermes agent.
cd /home/manav/rebot_lerobot
export PYTHONPATH=/home/manav/rebot_lerobot
export REBOT_ARM_SERVICE_URL="${REBOT_ARM_SERVICE_URL:-http://127.0.0.1:8000}"
exec /home/manav/miniconda3/envs/rebot_rs/bin/python -m arm_agent.mcp_server
