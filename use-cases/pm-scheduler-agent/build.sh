#!/usr/bin/env bash
# Build Angular frontend and push to SAP Cloud Foundry.
set -e

# ── 1. Build Angular ──────────────────────────────────────────────────────────
echo ">>> Building Angular frontend..."
cd frontend
npm install
npm run build          # outputs to ../static via angular.json outputPath
cd ..
echo ">>> Angular build complete. Output in ./static/"

# ── 2. Deploy to Cloud Foundry ────────────────────────────────────────────────
echo ">>> Pushing to Cloud Foundry..."
cf push fmi-maintenance-schedule

echo ">>> Deploy complete."
