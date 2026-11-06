#!/bin/bash
#
# Deploy the API and UI applications to Cloud Foundry.
#
# Examples:
#   ./deploy.sh
#   FETCH_MAX_DAYS=14 ./deploy.sh
#   DOCUMENT_REQUEST_FROM_EMAIL=documents@example.com ./deploy.sh
#   CF_TRACE=true ./deploy.sh
#   CLOUD_LOGGING_SERVICE_NAME="my-cloud-logging" ./deploy.sh
#
# Runtime-only API secrets are read from api/.env and synchronized to the
# Cloud Foundry API app after cf push. They are intentionally not passed as
# manifest variables because cf push prints manifest variable diffs.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"
API_ENV_FILE="$SCRIPT_DIR/api/.env"
API_APP_NAME="shipment-document-request-agent-api"
API_ROUTE="https://shipment-document-request-agent-api.cfapps.eu10-004.hana.ondemand.com"
UI_HOST="shipment-document-request-agent.cfapps.eu10-004.hana.ondemand.com"
FETCH_MAX_DAYS_OVERRIDE="${FETCH_MAX_DAYS-}"
DOCUMENT_REQUEST_FROM_EMAIL_OVERRIDE="${DOCUMENT_REQUEST_FROM_EMAIL-}"
# Cloud Logging service instance that collects the LLM token-usage stdout events.
CLOUD_LOGGING_SERVICE_NAME="${CLOUD_LOGGING_SERVICE_NAME:-Cloud Logging}"

if [[ -f "$API_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$API_ENV_FILE"
else
  echo "Missing API environment file: $API_ENV_FILE" >&2
  echo "Create it from api/.env.example before deploying." >&2
  exit 1
fi

FETCH_MAX_DAYS="${FETCH_MAX_DAYS_OVERRIDE:-${FETCH_MAX_DAYS:-7}}"
DOCUMENT_REQUEST_FROM_EMAIL="${DOCUMENT_REQUEST_FROM_EMAIL_OVERRIDE:-${DOCUMENT_REQUEST_FROM_EMAIL:-}}"
API_RUNTIME_ENV_VARS=(
  HANA_ADDRESS
  HANA_PORT
  HANA_USER
  HANA_PASSWORD
  HANA_ENCRYPT
  AICORE_AUTH_URL
  AICORE_CLIENT_ID
  AICORE_CLIENT_SECRET
  AICORE_BASE_URL
  AICORE_RESOURCE_GROUP
)

require_env_var() {
  local name="$1"
  local value="${!name-}"
  if [[ -z "$value" ]]; then
    echo "Missing required value in api/.env: $name" >&2
    exit 1
  fi
}

set_cf_env_quietly() {
  local app_name="$1"
  local name="$2"
  local value="$3"
  if ! cf set-env "$app_name" "$name" "$value" >/dev/null 2>&1; then
    echo "Failed to set Cloud Foundry env var $name on $app_name." >&2
    echo "Run 'cf target' and confirm you are targeting the expected org and space." >&2
    exit 1
  fi
  echo "   set $name"
}

for env_var in "${API_RUNTIME_ENV_VARS[@]}"; do
  require_env_var "$env_var"
done

API_KEY=$(openssl rand -hex 32)
echo "Generated temporary API key for this deployment."

echo "Building UI for production..."
(
  cd "$UI_DIR"
  VITE_API_BASE_URL="$API_ROUTE" \
  VITE_API_KEY="$API_KEY" \
  npm run build
)

echo "Deploying application..."
cf push \
  --var api_key="$API_KEY" \
  --var api_route="$API_ROUTE" \
  --var ui_host="$UI_HOST" \
  --var fetch_max_days="$FETCH_MAX_DAYS" \
  --var document_request_from_email="$DOCUMENT_REQUEST_FROM_EMAIL"

echo "Synchronizing API runtime environment..."
for env_var in "${API_RUNTIME_ENV_VARS[@]}"; do
  set_cf_env_quietly "$API_APP_NAME" "$env_var" "${!env_var}"
done

echo "Binding $API_APP_NAME to Cloud Logging service '$CLOUD_LOGGING_SERVICE_NAME'..."
cf bind-service "$API_APP_NAME" "$CLOUD_LOGGING_SERVICE_NAME"

echo "Restarting API app to apply runtime environment and service bindings..."
cf restart "$API_APP_NAME"

echo "Deployment finished."
