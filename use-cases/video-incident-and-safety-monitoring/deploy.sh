#!/usr/bin/env bash
#
# Deploy ai-core-video to Cloud Foundry with an ephemeral API key.
# Usage: ./deploy.sh
# Requires: cf CLI logged in and org/space targeted.
#

set -euo pipefail

# Generate a 32-byte API key (hex)
API_KEY="$(openssl rand -hex 32)"
echo "🔑 Generated API key for this deployment."

# Optional: set allowed origin for CORS to the frontend route
FRONTEND_ROUTE="https://ai-core-video.cfapps.eu10-004.hana.ondemand.com"
echo "🌐 Using ALLOWED_ORIGIN: ${FRONTEND_ROUTE}"

# Show reminder for required backend envs (set with cf set-env or adjust manifest.yml)
cat <<'EOF'
Reminder: Ensure backend environment variables are configured (either via manifest.yml or cf set-env):

  AICORE_AUTH_URL
  AICORE_CLIENT_ID
  AICORE_CLIENT_SECRET
  AICORE_BASE_URL
  AICORE_RESOURCE_GROUP (default: "default")

The deploy uses manifest.yml and injects the API_KEY via --var api_key.
EOF

# Push apps with api_key variable for backend
echo "🚀 Deploying apps (frontend + backend) with API key..."
cf push -f manifest.yml --var api_key="${API_KEY}"

echo "✅ Deployment finished."

echo "🔐 To call the backend, include header: X-API-Key: ${API_KEY}"
echo "   Frontend can pass API key via a meta tag:"
echo '   <meta name="backend-api-key" content="'"${API_KEY}"'">'
