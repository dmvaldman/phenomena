#!/bin/bash
# SSH tunnel to the J-lens visualizer on the RunPod pod (localhost:7860).
# Starts the remote uvicorn server if it isn't already running, then holds
# the tunnel open in the foreground.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/remote.env"
SSH_OPTS=(-p "$REMOTE_PORT" -i "$SSH_KEY" -o ConnectTimeout=15)

ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
  'pgrep -f "[u]vicorn lens_server" > /dev/null || (cd /workspace/phenomena/src && export HF_HOME=/workspace/hf && nohup uvicorn lens_server:app --host 0.0.0.0 --port 7860 > /workspace/phenomena/results/lens_server.log 2>&1 & sleep 2; echo "lens server launched")'

echo "tunnel up: http://localhost:7860"
exec ssh "${SSH_OPTS[@]}" -N -L 7860:localhost:7860 "$REMOTE_USER@$REMOTE_HOST"
