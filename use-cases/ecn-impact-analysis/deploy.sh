#!/bin/bash
#
# This script automates the deployment of the application to Cloud Foundry.
# It generates a secure, random API key and reads service credentials from
# the local api/.env file, injecting them into the deployment process.
#
# Prerequisites:
#   - You must be logged into Cloud Foundry before running this script.
#   - The file api/.env must exist with valid credentials.
#   - Install UI dependencies locally with: npm install --prefix ui
#
# Usage:
#   ./deploy.sh
#

# --- Configuration ---

ENV_FILE="api/.env"
APP_NAME=$(grep -m 1 '\- name:' manifest.yaml | awk '{print $3}')

# --- Functions ---

# Reads a variable value from the .env file.
# Strips surrounding quotes (single or double) from the value.
read_env_var() {
  local var_name="$1"
  local value
  value=$(grep "^${var_name}=" "$ENV_FILE" | head -1 | cut -d '=' -f2-)
  # Remove surrounding quotes
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  echo "$value"
}

# --- Main Script ---

# 1. Verify that the .env file exists
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Please create it with the required credentials."
  echo "See api/.env.example for the expected format."
  exit 1
fi

# 2. Generate a secure, random API key
API_KEY=$(openssl rand -hex 32)
echo "Generated temporary API key for this deployment."

# 3. Read credentials from api/.env
echo "Reading credentials from $ENV_FILE..."

AICORE_AUTH_URL=$(read_env_var "AICORE_AUTH_URL")
AICORE_CLIENT_ID=$(read_env_var "AICORE_CLIENT_ID")
AICORE_CLIENT_SECRET=$(read_env_var "AICORE_CLIENT_SECRET")
AICORE_BASE_URL=$(read_env_var "AICORE_BASE_URL")
AICORE_RESOURCE_GROUP=$(read_env_var "AICORE_RESOURCE_GROUP")

HANA_ADDRESS=$(read_env_var "HANA_ADDRESS")
HANA_PORT=$(read_env_var "HANA_PORT")
HANA_USER=$(read_env_var "HANA_USER")
HANA_PASSWORD=$(read_env_var "HANA_PASSWORD")
HANA_ENCRYPT=$(read_env_var "HANA_ENCRYPT")

# 4. Write substitution variables to a temporary vars-file.
#    cf's --var flag rejects empty values and is awkward with JSON strings, so a YAML vars-file is
#    used instead: it accepts empty scalars and embedded double quotes cleanly.
#    mktemp creates the file with owner-only permissions; the trap deletes it
#    on any exit so credentials are not left on disk.
VARS_FILE="$(mktemp)"
trap 'rm -f "$VARS_FILE"' EXIT

# Append one "key: 'value'" YAML line. Single quotes inside the value are
# escaped by doubling them, so any character (quotes inside secrets or an
# empty string) survives YAML parsing unchanged.
emit_var() {
  local key="$1"
  local value="$2"
  value="${value//"'"/"''"}"
  printf "%s: '%s'\n" "$key" "$value" >> "$VARS_FILE"
}

emit_var api_key "$API_KEY"
emit_var aicore_auth_url "$AICORE_AUTH_URL"
emit_var aicore_client_id "$AICORE_CLIENT_ID"
emit_var aicore_client_secret "$AICORE_CLIENT_SECRET"
emit_var aicore_base_url "$AICORE_BASE_URL"
emit_var aicore_resource_group "$AICORE_RESOURCE_GROUP"
emit_var hana_address "$HANA_ADDRESS"
emit_var hana_port "$HANA_PORT"
emit_var hana_user "$HANA_USER"
emit_var hana_password "$HANA_PASSWORD"
emit_var hana_encrypt "$HANA_ENCRYPT"

# 5. Build the UI on this machine with the same API key sent to the API app.
#    The manifest's API_BASE_URL is the deployed API endpoint; Vite embeds both
#    values in the static bundle, so CF must receive the freshly built dist/.
UI_API_BASE_URL=$(awk '$1 == "API_BASE_URL:" { print $2; exit }' manifest.yaml)
if [ -z "$UI_API_BASE_URL" ]; then
  echo "Error: API_BASE_URL is missing from manifest.yaml."
  exit 1
fi
echo "Building UI locally..."
if ! (cd ui && VITE_API_BASE_URL="$UI_API_BASE_URL" VITE_API_KEY="$API_KEY" npm run build); then
  echo "Error: local UI build failed. Install UI dependencies with npm install --prefix ui and retry."
  exit 1
fi
if [ ! -s ui/dist/index.html ] || [ ! -f ui/dist/Staticfile ]; then
  echo "Error: UI build did not produce ui/dist/index.html and ui/dist/Staticfile."
  exit 1
fi

# 6. Deploy the application. Abort if the push fails so the service-binding and
#    restart steps below never run against an app that was not created.
echo "Deploying application..."
if ! cf push --vars-file "$VARS_FILE"; then
  echo "Error: cf push failed. Aborting before service binding and restart."
  exit 1
fi

# 7. Bind the application to SAP Cloud Logging
echo "Binding $APP_NAME to Cloud Logging..."
cf bind-service "$APP_NAME" "Cloud Logging"

# 8. Restart the application to pick up the new service binding
echo "Restarting $APP_NAME..."
cf restart "$APP_NAME"

echo "Deployment finished."
