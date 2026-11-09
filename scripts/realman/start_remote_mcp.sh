#!/usr/bin/env bash
# Persistent MCP entrypoint. SSH transports stdio; the server owns ROS clients.
# No inline robot commands, Home, enable, frame changes, or motion on startup.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/connect_shared_mcp.sh"
