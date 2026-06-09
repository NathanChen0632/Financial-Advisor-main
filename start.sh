#!/usr/bin/env bash
# ============================================================
# One command to run the whole stack.
#   ./start.sh            (or:  npm run dev   from the project root)
#
# Starts, together:
#   1. the trading monitor  (executes UI trades, sell approvals, syncs Alpaca)
#   2. a one-off AI recommendations refresh
#   3. the website          (Vite dev server at http://localhost:5173)
#
# Backend processes run in the background with their output sent to log files.
# Press Ctrl-C in this terminal to stop everything at once.
# ============================================================
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/env/bin/python"

# Stream Python output to the log files live instead of block-buffering it,
# so `tail -f monitor.log` shows progress as it happens.
export PYTHONUNBUFFERED=1

# Collect background PIDs so we can shut them down on exit.
pids=""
cleanup() {
  echo ""
  echo "Shutting down backend..."
  for pid in $pids; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

if [ ! -x "$PY" ]; then
  echo "ERROR: Python venv not found at $PY"
  echo "Create it and install deps:  python3 -m venv env && ./env/bin/pip install -r backend/requirements.txt"
  exit 1
fi

echo "============================================================"
echo "  Starting DQN Trading Dashboard (full stack)"
echo "============================================================"

echo "[1/3] Trading monitor   -> logs: monitor.log   (tail -f monitor.log)"
( cd "$ROOT/backend" && "$PY" -m stock_prediction.monitor --alpaca --no-market-check ) \
  > "$ROOT/monitor.log" 2>&1 &
pids="$pids $!"

echo "[2/3] AI recommendations -> logs: research.log  (one-off refresh)"
( cd "$ROOT/backend" && "$PY" -m stock_prediction.recommendation_job ) \
  > "$ROOT/research.log" 2>&1 &
pids="$pids $!"

echo "[3/3] Website            -> http://localhost:5173"
echo "------------------------------------------------------------"
echo "  Backend is starting up (the monitor trains models first —"
echo "  watch progress with:  tail -f monitor.log )."
echo "  Press Ctrl-C here to stop the whole stack."
echo "============================================================"

cd "$ROOT/frontend"
npm run dev
