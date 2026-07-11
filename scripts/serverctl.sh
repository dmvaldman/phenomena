#!/bin/bash
# Reliable control of the pod-side lens server.
# Usage: scripts/serverctl.sh start|stop|restart|status
# Each phase uses its own ssh session (kill+relaunch in one session dies with it).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/remote.env"
SSH=(ssh -o ConnectTimeout=15 -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST")
LOG=/workspace/phenomena/results/lens_server.log

stop() {
  "${SSH[@]}" 'pkill -f "[u]vicorn lens_server" 2>/dev/null; for i in 1 2 3 4 5; do pgrep -f "[u]vicorn lens_server" >/dev/null || break; sleep 1; done; pgrep -f "[u]vicorn lens_server" >/dev/null && echo "stop: FAILED" || echo "stop: ok"'
}

start() {
  "${SSH[@]}" ": > $LOG; cd /workspace/phenomena/src && export HF_HOME=/workspace/hf && setsid nohup uvicorn lens_server:app --host 0.0.0.0 --port 7860 > $LOG 2>&1 < /dev/null & sleep 2; pgrep -f '[u]vicorn lens_server' >/dev/null && echo 'start: launched' || echo 'start: FAILED'"
  "${SSH[@]}" "for i in \$(seq 1 60); do grep -q 'Uvicorn running' $LOG 2>/dev/null && { echo 'start: ready'; exit 0; }; sleep 5; done; echo 'start: TIMEOUT'; tail -n 5 $LOG; exit 1"
}

case "${1:-status}" in
  stop) stop ;;
  start) start ;;
  restart) stop; start ;;
  status) "${SSH[@]}" "pgrep -f '[u]vicorn lens_server' >/dev/null && echo RUNNING || echo DOWN; tail -n 2 $LOG 2>/dev/null" ;;
  *) echo "usage: serverctl.sh start|stop|restart|status" >&2; exit 1 ;;
esac
