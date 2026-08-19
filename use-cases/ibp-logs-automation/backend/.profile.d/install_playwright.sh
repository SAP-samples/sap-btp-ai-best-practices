#!/usr/bin/env bash
# CF Python buildpack post-compile hook
# Installs Playwright Chromium browser binary after pip install
set -e
echo "-----> Installing Playwright browser (Chromium)..."
playwright install chromium
echo "-----> Playwright Chromium installed."
