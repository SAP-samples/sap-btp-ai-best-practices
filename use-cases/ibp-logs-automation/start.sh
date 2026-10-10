#!/usr/bin/env bash
# start.sh — launch Flask backend + Angular dev server

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Backend ────────────────────────────────────────────────────────────────
echo "Starting Flask backend on http://localhost:5001 ..."
cd backend
source .venv/bin/activate
FLASK_PORT=5001 python app.py &
BACKEND_PID=$!
deactivate
cd ..

# ── Frontend ───────────────────────────────────────────────────────────────
echo "Starting Angular dev server on http://localhost:4200 ..."
cd frontend
NG_CLI_ANALYTICS=false NG_COMPLETION_PROMPTED=true npx ng serve --open &
FRONTEND_PID=$!
cd ..

echo ""
echo "Both servers running. Press Ctrl+C to stop."

cleanup() {
  echo ""
  echo "Stopping servers..."
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM
wait
