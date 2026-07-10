#!/bin/bash
# Usage:
#   scripts/sync.sh push          # local src/ + research_plan.md -> pod
#   scripts/sync.sh pull          # pod results/ -> local results/
#   scripts/sync.sh run <cmd...>  # push, then run command on pod from REMOTE_DIR with HF_HOME set
#   scripts/sync.sh ssh           # interactive shell on pod
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/remote.env"
SSH=(ssh -p "$REMOTE_PORT" -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST")
RS=(rsync -rltz --no-owner --no-group --no-perms -e "ssh -p $REMOTE_PORT -i $SSH_KEY")

case "${1:-}" in
  push)
    "${SSH[@]}" "mkdir -p $REMOTE_DIR/src $REMOTE_DIR/results"
    "${RS[@]}" --delete "$ROOT/src/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/src/"
    "${RS[@]}" "$ROOT/research_plan.md" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
    echo "pushed."
    ;;
  pull)
    mkdir -p "$ROOT/results"
    "${RS[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/results/" "$ROOT/results/"
    echo "pulled to results/."
    ;;
  run)
    shift
    "$0" push
    "${SSH[@]}" "export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1; cd $REMOTE_DIR && $*"
    ;;
  ssh)
    "${SSH[@]}"
    ;;
  *)
    echo "usage: sync.sh push|pull|run <cmd>|ssh" >&2; exit 1
    ;;
esac
