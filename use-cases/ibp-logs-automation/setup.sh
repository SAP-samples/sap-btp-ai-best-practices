#!/usr/bin/env bash
# setup.sh — install all dependencies for IBP Optimizer Log Extractor

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Backend: installing Python deps ==="
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
playwright install chromium --with-deps
deactivate
cd ..

echo "=== Frontend: installing npm deps ==="
cd frontend
npm install --legacy-peer-deps
cd ..

echo ""
echo "Setup complete. Run ./start.sh to launch the app."
