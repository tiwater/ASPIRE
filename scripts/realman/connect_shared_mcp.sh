#!/usr/bin/env bash
set -euo pipefail
exec ssh -T -p 7582 -o BatchMode=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
  realman@139.224.22.149 /home/realman/aspire-control/connect_mcp.sh
