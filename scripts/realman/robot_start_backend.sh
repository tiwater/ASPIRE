#!/usr/bin/env bash
# One robot-local owner; connections attach through robot_connect_mcp.sh.
set -eo pipefail
ASPIRE_REALMAN_ROOT="${ASPIRE_REALMAN_ROOT:-$HOME/aspire-control}"
source "${ASPIRE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${ASPIRE_ROBOT_SETUP:-$HOME/workspace/rm_robot_ws/install/setup.bash}"
export PYTHONPATH="${ASPIRE_REALMAN_ROOT}/current/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${ASPIRE_REALMAN_ROOT}/.venv/bin/python" -m cap_harness.realman.live_server \
  --policy "${ASPIRE_REALMAN_ROOT}/config/control_policy.json" \
  --artifacts "${ASPIRE_REALMAN_ROOT}/artifacts" \
  --lock-file "${ASPIRE_REALMAN_ROOT}/control.lock" \
  --transport streamable-http --port 8232
