#!/bin/bash
#
# Deploy the API and UI applications to Cloud Foundry.
#
# Examples:
#   ./deploy.sh
#   FETCH_MAX_DAYS=14 ./deploy.sh
#   PRICE_CHANGE_AGENT_REASONING_EFFORT=medium ./deploy.sh
#   CF_TRACE=true ./deploy.sh
#
# Runtime-only API secrets are read from api/.env and synchronized to the
# Cloud Foundry API app after cf push. They are intentionally not passed as
# manifest variables because cf push prints manifest variable diffs.
#

# --- Main Script ---

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"
API_ENV_FILE="$SCRIPT_DIR/api/.env"
API_APP_NAME="email-price-classifier-api"
EXPECTED_CF_API_ENDPOINT="https://api.cf.eu10-005.hana.ondemand.com"
API_ROUTE="https://email-price-classifier-api.cfapps.eu10-005.hana.ondemand.com"
UI_HOST="email-price-classifier.cfapps.eu10-005.hana.ondemand.com"
FETCH_MAX_DAYS_OVERRIDE="${FETCH_MAX_DAYS-}"

if [[ -z "${PYTHON_BIN-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

load_api_env_from_dotenv() {
  if [[ ! -f "$API_ENV_FILE" ]]; then
    echo "Missing API environment file: $API_ENV_FILE" >&2
    echo "Create it from api/.env.example before deploying." >&2
    exit 1
  fi

  local assignments
  if ! assignments="$("$PYTHON_BIN" - "$API_ENV_FILE" <<'PY'
from __future__ import annotations

import re
import shlex
import sys

try:
    from dotenv import dotenv_values
except ImportError as exc:
    raise SystemExit(
        "python-dotenv is required to parse api/.env safely. "
        "Run the deploy script from the project virtualenv or set PYTHON_BIN."
    ) from exc

name_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
values = dotenv_values(sys.argv[1])
for key, value in values.items():
    if value is None:
        continue
    if not name_pattern.fullmatch(key):
        raise SystemExit(f"Unsupported environment variable name in api/.env: {key}")
    print(f"{key}={shlex.quote(str(value))}")
PY
)"; then
    echo "Failed to parse API environment file with $PYTHON_BIN: $API_ENV_FILE" >&2
    exit 1
  fi

  local assignment
  while IFS= read -r assignment; do
    [[ -z "$assignment" ]] && continue
    eval "export $assignment"
  done <<< "$assignments"
}

load_api_env_from_dotenv

FETCH_MAX_DAYS="${FETCH_MAX_DAYS_OVERRIDE:-${FETCH_MAX_DAYS:-7}}"
PRICE_CHANGE_ATTACHMENT_EXTRACTOR_MODEL="${PRICE_CHANGE_ATTACHMENT_EXTRACTOR_MODEL:-${PRICE_CHANGE_AGENT_MODEL:-gpt-5.4}}"
PRICE_CHANGE_AGENT_REASONING_EFFORT="${PRICE_CHANGE_AGENT_REASONING_EFFORT:-low}"
S4_CONNECTIVITY_MODE="${S4_CONNECTIVITY_MODE:-auto}"
BTP_DEST_DEBUG="${BTP_DEST_DEBUG:-false}"
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
  S4_BASE_URL
  S4_CLIENT
  S4_VERIFY
  S4_USERNAME
  S4_PASSWORD
  S4_DESTINATION_NAME
  DESTINATION_SERVICE_URI
  DESTINATION_TOKEN_BASE_URL
  DESTINATION_CLIENT_ID
  DESTINATION_CLIENT_SECRET
  CONNECTIVITY_PROXY_HOST
  CONNECTIVITY_PROXY_PORT
  CONNECTIVITY_TOKEN_BASE_URL
  CONNECTIVITY_CLIENT_ID
  CONNECTIVITY_CLIENT_SECRET
  S4_CONNECTIVITY_MODE
  BTP_DEST_DEBUG
  GMAIL_MAILBOX_ID
  GMAIL_DESTINATION_NAME
  GMAIL_DESTINATION_LEVEL
  GMAIL_REFRESH_TOKEN_PROPERTY
  DESTINATION_SERVICE_NAME
  PRICE_CHANGE_EXTRACTOR_MODEL
  PRICE_CHANGE_ATTACHMENT_EXTRACTOR_MODEL
  PRICE_CHANGE_AGENT_MODEL
  PRICE_CHANGE_AGENT_REASONING_EFFORT
)
OPTIONAL_API_RUNTIME_ENV_VARS=(
  LOG_USER_HASH_SALT
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

extract_cf_api_endpoint() {
  sed -n 's/^API endpoint:[[:space:]]*//p' | head -n 1
}

assert_expected_cf_target() {
  local current_api_endpoint
  current_api_endpoint="$(cf target | extract_cf_api_endpoint)"
  if [[ "$current_api_endpoint" != "$EXPECTED_CF_API_ENDPOINT" ]]; then
    echo "Expected Cloud Foundry API endpoint $EXPECTED_CF_API_ENDPOINT but current target is ${current_api_endpoint:-unknown}." >&2
    echo "Run 'cf login -a $EXPECTED_CF_API_ENDPOINT' before deploying S/4 Connectivity changes." >&2
    exit 1
  fi
}

for env_var in "${API_RUNTIME_ENV_VARS[@]}"; do
  require_env_var "$env_var"
done

assert_expected_cf_target

# 1. Generate a secure, random API key
# This creates a 32-byte (256-bit) random key and encodes it in hexadecimal.
API_KEY=$(openssl rand -hex 32)
echo "🔑 Generated temporary API key for this deployment."

# 2. Build the UI with the same production values that the deployed app serves.
echo "🏗️ Building UI for production..."
(
  cd "$UI_DIR"
  VITE_API_BASE_URL="$API_ROUTE" \
  VITE_API_KEY="$API_KEY" \
  npm run build
)

# 3. Deploy the application using the generated key
# The `cf push` command reads the manifest.yaml and replaces the ((api_key))
# variable with the value we provide here.
#
# Note: You must be logged into Cloud Foundry before running this script.
echo "🚀 Deploying application..."
cf push \
  --var api_key="$API_KEY" \
  --var api_route="$API_ROUTE" \
  --var ui_host="$UI_HOST" \
  --var fetch_max_days="$FETCH_MAX_DAYS"

echo "🔐 Synchronizing API runtime environment..."
for env_var in "${API_RUNTIME_ENV_VARS[@]}"; do
  set_cf_env_quietly "$API_APP_NAME" "$env_var" "${!env_var}"
done
for env_var in "${OPTIONAL_API_RUNTIME_ENV_VARS[@]}"; do
  if [[ -n "${!env_var-}" ]]; then
    set_cf_env_quietly "$API_APP_NAME" "$env_var" "${!env_var}"
  fi
done

echo "📜 Binding API app to Cloud Logging..."
cf bind-service "$API_APP_NAME" "Cloud Logging"

echo "🔄 Restarting API app to apply runtime environment..."
cf restart "$API_APP_NAME"

echo "✅ Deployment finished."
