#!/bin/bash
#
# Deploy the AI4U Ideation Agent to Cloud Foundry.
#
# All API env vars are read from api/.env.production (git-ignored) and set
# via `cf set-env`. Adding a new variable = adding one line to that file.
#
# Prerequisites:
#   - cf login (to your target org/space)
#   - Service instances exist (XSUAA, HANA, Object Store, AI Core)
#   - api/.env.production exists with production values
#
# First-time setup? Run ./setup.sh instead — it creates services,
# generates configs, and calls this script automatically.
#
set -eo pipefail

BACKEND_ENV_FILE="api/.env.production"

info() { echo "[$(date +'%H:%M:%S')] $*"; }

# --- Preflight checks ---

if ! command -v cf >/dev/null 2>&1; then
  echo "❌ cf CLI not found. Install it and log in first."
  echo "   https://docs.cloudfoundry.org/cf-cli/install-go-cli.html"
  exit 1
fi

if [ ! -f "$BACKEND_ENV_FILE" ]; then
  echo "❌ $BACKEND_ENV_FILE not found."
  echo "   Run ./setup.sh for first-time setup, or create it manually."
  echo "   See api/.env.production.example for the minimal required variables."
  exit 1
fi

if [ ! -f "manifest.yaml" ]; then
  echo "❌ manifest.yaml not found."
  echo "   Run ./setup.sh to generate it from manifest.yaml.template."
  exit 1
fi

# --- Parse app name and XSUAA instance from manifest.yaml ---

APP_NAME=$(grep -E "^[[:space:]]*- name:" manifest.yaml | head -1 | sed 's/.*- name:[[:space:]]*//' | xargs)
if [[ -z "$APP_NAME" ]]; then
  echo "❌ Could not parse API app name from manifest.yaml"
  exit 1
fi

XSUAA_INSTANCE=$(grep -A 20 "services:" manifest.yaml | grep "xsuaa" | head -1 | sed 's/.*- //' | xargs)
XSUAA_KEY_NAME="${XSUAA_INSTANCE}-key"

info "📦 API app: $APP_NAME"
info "🔐 XSUAA:   $XSUAA_INSTANCE"

# --- Parse .env.production to get CDC_API_KEY for UI app ---

cdc_api_key=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  [[ "$line" != *=* ]] && continue
  key="${line%%=*}"
  key=$(echo "$key" | xargs)
  value="${line#*=}"
  value=$(echo "$value" | sed -e "s/^['\"]//;s/['\"]$//")
  [[ "$key" == "CDC_API_KEY" ]] && cdc_api_key="$value"
done < "$BACKEND_ENV_FILE"

# --- 1) Build --var flags (only for unresolved manifest placeholders) ---

PUSH_ARGS=()

if grep -q '((xsuaa_client_id))' manifest.yaml; then
  info "🔑 Reading XSUAA client ID from service key..."
  XSUAA_CLIENT_ID=$(cf service-key "$XSUAA_INSTANCE" "$XSUAA_KEY_NAME" 2>/dev/null | grep '"clientid"' | sed 's/.*: "\(.*\)".*/\1/')
  if [[ -z "$XSUAA_CLIENT_ID" ]]; then
    echo "❌ Could not read XSUAA client ID. Ensure service key exists:"
    echo "   cf create-service-key $XSUAA_INSTANCE $XSUAA_KEY_NAME"
    exit 1
  fi
  info "   Client ID: $XSUAA_CLIENT_ID"
  PUSH_ARGS+=(--var "xsuaa_client_id=$XSUAA_CLIENT_ID")
fi

if grep -q '((cdc_api_key))' manifest.yaml; then
  PUSH_ARGS+=(--var "cdc_api_key=$cdc_api_key")
fi

# --- 2) Push all apps ---

# Copy VERSION file into api/ so the deployed app can read it
cp VERSION api/VERSION 2>/dev/null || true

info "🚀 Pushing all applications..."
cf push "${PUSH_ARGS[@]}"

# --- 3) Set API env vars from .env.production ---
# cf push only sets what's in the manifest. The API app needs all vars from the env file.

info "⚙️  Setting env vars on $APP_NAME from $BACKEND_ENV_FILE..."
var_count=0

while IFS= read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue

  # Must contain = to be a valid assignment
  [[ "$line" != *=* ]] && continue

  # Split on first =
  key="${line%%=*}"
  value="${line#*=}"

  # Trim whitespace from key
  key=$(echo "$key" | xargs)

  # Skip invalid keys (empty, contains spaces)
  [[ -z "$key" || "$key" == *" "* ]] && continue

  # Strip surrounding quotes from value (single or double)
  value=$(echo "$value" | sed -e "s/^['\"]//;s/['\"]$//")

  cf set-env "$APP_NAME" "$key" "$value" > /dev/null
  var_count=$((var_count + 1))
done < "$BACKEND_ENV_FILE"

info "   Set $var_count env vars."

# --- 4) Restage API app to pick up the new env vars ---

info "♻️  Restaging $APP_NAME..."
cf restage "$APP_NAME"

info "✅ Deployment finished."
