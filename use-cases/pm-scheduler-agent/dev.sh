#!/usr/bin/env bash
# Start Flask backend + Angular dev server locally.
# Usage: ./dev.sh
# Press Ctrl+C to stop both.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[dev]${NC} $*"; }
ok()   { echo -e "${GREEN}[dev]${NC} $*"; }
warn() { echo -e "${YELLOW}[dev]${NC} $*"; }

# ── Cleanup on exit ──────────────────────────────────────────────────────────
FLASK_PID=""
NG_PID=""
cleanup() {
  echo ""
  log "Shutting down..."
  [ -n "$FLASK_PID" ] && kill "$FLASK_PID" 2>/dev/null && log "Flask stopped (PID $FLASK_PID)"
  [ -n "$NG_PID"    ] && kill "$NG_PID"    2>/dev/null && log "Angular stopped (PID $NG_PID)"
  exit 0
}
trap cleanup INT TERM

# ── 1. Python virtualenv ─────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  log "Creating Python virtualenv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

log "Installing/updating Python dependencies..."
pip install -q -r requirements.txt
ok "Python deps ready."

# ── 2. Angular dependencies ──────────────────────────────────────────────────
# Always run npm install — it is idempotent and re-syncs when package.json changes.
log "Syncing Angular dependencies..."
(cd frontend && npm install --legacy-peer-deps)
ok "Angular deps ready."

# ── 3. Start Flask (background) ──────────────────────────────────────────────
# Kill any stale process on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

log "Starting Flask on http://localhost:5001 ..."
FLASK_APP="backend.app:create_app()" \
  flask run --port 5001 2>&1 | sed $'s/^/\033[33m[flask]\033[0m /' &
FLASK_PID=$!

# Give Flask a moment to bind
sleep 1
if ! kill -0 "$FLASK_PID" 2>/dev/null; then
  echo -e "${RED}[dev] Flask failed to start. Check the output above.${NC}"
  exit 1
fi
ok "Flask running (PID $FLASK_PID)."

# ── 4. Start Angular dev server ──────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}Backend :${NC}  http://localhost:5001"
echo -e "  ${GREEN}Frontend:${NC}  http://localhost:4200"
echo ""
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop both servers."
echo ""

(cd frontend && npm start 2>&1 | sed $'s/^/\033[36m[angular]\033[0m /') &
NG_PID=$!

# Wait for both processes
wait "$NG_PID" "$FLASK_PID"
