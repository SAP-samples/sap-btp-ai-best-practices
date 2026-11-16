#!/bin/bash
#
# Deploy the Commodity Code Pipeline (API + UI) to Cloud Foundry.
#
# The script reads the SAP AI Core and SAP HANA credentials from api/.env and
# injects them into manifest.yaml through `cf push --var ...`, so no manual
# `cf set-env` step is needed. It also generates a random API key shared by the
# API and the UI (or reuses API_KEY when it is already exported).
#
# Usage (run from this folder, logged in to the target CF org/space):
#   chmod +x deploy.sh
#   ./deploy.sh
#
#   # Reuse a known API key (e.g. the one stored in the Joule BTP destination):
#   API_KEY="<existing-key>" ./deploy.sh
#
# Prerequisites:
#   - `cf login` done for the target org and space.
#   - api/.env exists with the variables listed in api/.env.example.
#

set -euo pipefail

# --- Configuration ---

# Always resolve paths relative to this script, regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="api/.env"

# --- Functions ---

# Read one variable from $ENV_FILE.
# Takes the first uncommented NAME=value line, strips Windows line endings and
# surrounding single or double quotes. Prints an empty string when not found.
read_env_var() {
  local var_name="$1"
  local value
  value=$(grep -m 1 "^${var_name}=" "$ENV_FILE" | cut -d '=' -f2- || true)
  value="${value%$'\r'}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  echo "$value"
}

# --- Main Script ---

# 1. Verify that the .env file exists.
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Create it from api/.env.example first."
  exit 1
fi

# 2. Use the exported API key if present, otherwise generate a new one.
if [ -n "${API_KEY:-}" ]; then
  echo "Using the API key from the current shell environment."
else
  API_KEY=$(openssl rand -hex 32)
  echo "Generated a new API key for this deployment."
fi

# 3. Read credentials from api/.env.
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
# Defaults matter: the API treats an empty value as "false", so an empty
# HANA_ENCRYPT would silently disable encryption.
HANA_ENCRYPT=$(read_env_var "HANA_ENCRYPT")
HANA_ENCRYPT="${HANA_ENCRYPT:-true}"
HANA_SSL_VALIDATE_CERTIFICATE=$(read_env_var "HANA_SSL_VALIDATE_CERTIFICATE")
HANA_SSL_VALIDATE_CERTIFICATE="${HANA_SSL_VALIDATE_CERTIFICATE:-false}"
# Optional: empty keeps HANA_USER's own schema (cf push accepts an empty --var).
HANA_SCHEMA=$(read_env_var "HANA_SCHEMA")

# 4. Fail early when required values are missing (names only, never values).
missing=()
for name in AICORE_AUTH_URL AICORE_CLIENT_ID AICORE_CLIENT_SECRET AICORE_BASE_URL \
            AICORE_RESOURCE_GROUP HANA_ADDRESS HANA_PORT HANA_USER HANA_PASSWORD; do
  if [ -z "${!name}" ]; then
    missing+=("$name")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Error: missing values in $ENV_FILE: ${missing[*]}"
  exit 1
fi

# 5. Deploy both applications. manifest.yaml maps each ((var)) to an env var.
echo "Deploying application..."
cf push \
  --var api_key="$API_KEY" \
  --var aicore_auth_url="$AICORE_AUTH_URL" \
  --var aicore_client_id="$AICORE_CLIENT_ID" \
  --var aicore_client_secret="$AICORE_CLIENT_SECRET" \
  --var aicore_base_url="$AICORE_BASE_URL" \
  --var aicore_resource_group="$AICORE_RESOURCE_GROUP" \
  --var hana_address="$HANA_ADDRESS" \
  --var hana_port="$HANA_PORT" \
  --var hana_user="$HANA_USER" \
  --var hana_password="$HANA_PASSWORD" \
  --var hana_encrypt="$HANA_ENCRYPT" \
  --var hana_ssl_validate_certificate="$HANA_SSL_VALIDATE_CERTIFICATE" \
  --var hana_schema="$HANA_SCHEMA"

echo "Deployment finished."
