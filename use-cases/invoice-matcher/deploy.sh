#!/bin/bash
#
# Deploys the Invoice-Payment Matcher to Cloud Foundry.
# Reads SAP AI Core credentials from api/.env file.
# Requires: cf CLI logged in.
#

set -e

ENV_FILE="api/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Copy api/.env.example to api/.env and fill in credentials."
  exit 1
fi

source "$ENV_FILE"

for var in AICORE_AUTH_URL AICORE_CLIENT_ID AICORE_CLIENT_SECRET AICORE_BASE_URL AICORE_RESOURCE_GROUP; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set in $ENV_FILE"
    exit 1
  fi
done

echo "Deploying application..."
cf push \
  --var aicore_auth_url="$AICORE_AUTH_URL" \
  --var aicore_client_id="$AICORE_CLIENT_ID" \
  --var aicore_client_secret="$AICORE_CLIENT_SECRET" \
  --var aicore_base_url="$AICORE_BASE_URL" \
  --var aicore_resource_group="$AICORE_RESOURCE_GROUP"

echo "Deployment finished."
