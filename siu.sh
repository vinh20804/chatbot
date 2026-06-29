#!/bin/bash
cd "$(dirname "$0")"

if tmux has-session -t server 2>/dev/null; then
    echo "[siu] Session 'server' already running. Attach: tmux attach -t server"
    exit 0
fi

tmux new-session -d -s server \
    "source venv/bin/activate && python -u -m lightrag.api.lightrag_server 2>&1 | tee server.log"

echo "[siu] Server started in tmux session 'server'."
echo "  Attach : tmux attach -t server"
echo "  Log    : tail -f server.log"
echo "  Stop   : tmux kill-session -t server"
