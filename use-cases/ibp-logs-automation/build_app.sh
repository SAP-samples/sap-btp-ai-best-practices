#!/usr/bin/env bash
# build_app.sh — builds IBP-Agent.app for macOS
# Run from the project root: ./build_app.sh
set -e

cd "$(dirname "$0")"
source backend/.venv/bin/activate

echo "Installing build dependencies..."
pip install pyinstaller pystray pillow -q

echo "Building IBP-Agent.app..."
pyinstaller \
  --name "IBP-Agent" \
  --onefile \
  --windowed \
  --add-data "local_agent.py:." \
  --add-data "extract.py:." \
  --add-data "backend/.env:." \
  --add-data "backend/session_cookies.json:." \
  --hidden-import flask \
  --hidden-import flask_cors \
  --hidden-import dotenv \
  --hidden-import requests \
  --hidden-import playwright \
  --hidden-import playwright.sync_api \
  --hidden-import pystray \
  --hidden-import PIL \
  --hidden-import PIL.Image \
  --hidden-import PIL.ImageDraw \
  ibp_agent_app.py

echo ""
echo "Done! Output: dist/IBP-Agent"
echo "On macOS: dist/IBP-Agent.app"
echo "Double-click to launch."
