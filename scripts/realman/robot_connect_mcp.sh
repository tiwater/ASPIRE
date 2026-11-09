#!/usr/bin/env bash
# Multiple SSH/MCP connections share one already-running backend. Never restart it.
set -euo pipefail
ASPIRE_REALMAN_ROOT="${ASPIRE_REALMAN_ROOT:-$HOME/aspire-control}"
if ! systemctl --user is-active --quiet aspire-realman-mcp.service; then
  echo 'ASPIRE RealMan backend is not active; start the supervised backend service first.' >&2
  exit 1
fi
export PYTHONPATH="${ASPIRE_REALMAN_ROOT}/current/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${ASPIRE_REALMAN_ROOT}/.venv/bin/python" -m cap_harness.realman.mcp_proxy
