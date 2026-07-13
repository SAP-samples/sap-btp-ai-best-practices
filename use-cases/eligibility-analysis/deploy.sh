#!/bin/bash
#
# This script automates the deployment of the application to Cloud Foundry.
# It generates a secure, random API key and injects it into the deployment
# process, enhancing security and simplifying setup.
#

# --- Optional .env loading for AI Core credentials ---
# Reads api/.env if present to avoid hardcoding secrets in the manifest.
ENV_FILE="api/.env"
if [ -f "$ENV_FILE" ]; then
  eval "$(
    python3 - <<'PY'
import os
import shlex
from pathlib import Path

path = Path("api/.env")
if not path.exists():
    raise SystemExit

allowed = {
    "AICORE_AUTH_URL",
    "AICORE_CLIENT_ID",
    "AICORE_CLIENT_SECRET",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
    "hana_address",
    "hana_port",
    "hana_user",
    "hana_password",
    "hana_encrypt",
}

for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key not in allowed or key in os.environ:
        continue
    value = value.strip()
    if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
        value = value[1:-1]
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
fi

# Default resource group if not set explicitly.
: "${AICORE_RESOURCE_GROUP:=default}"

# HANA vars default to empty (SQLite fallback when hana_address is blank).
: "${hana_address:=}"
: "${hana_port:=443}"
: "${hana_user:=}"
: "${hana_password:=}"
: "${hana_encrypt:=true}"

# --- Main Script ---

# Validate AI Core env vars needed by the agent
missing_vars=()
for var in AICORE_AUTH_URL AICORE_CLIENT_ID AICORE_CLIENT_SECRET AICORE_BASE_URL; do
  if [ -z "${!var}" ]; then
    missing_vars+=("$var")
  fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
  echo "❌ Missing required AI Core env vars: ${missing_vars[*]}" >&2
  echo "Set them in api/.env or export them before running this script." >&2
  exit 1
fi

# 1. Generate a secure, random API key
# This creates a 32-byte (256-bit) random key and encodes it in hexadecimal.
API_KEY=$(openssl rand -hex 32)
echo "🔑 Generated temporary API key for this deployment."

# 2. Deploy the application using the generated key
# The `cf push` command reads the manifest.yaml and replaces the ((api_key))
# variable with the value we provide here.
#
# Note: You must be logged into Cloud Foundry before running this script.
echo "🚀 Deploying application..."
cf push \
  --var api_key="$API_KEY" \
  --var aicore_auth_url="$AICORE_AUTH_URL" \
  --var aicore_client_id="$AICORE_CLIENT_ID" \
  --var aicore_client_secret="$AICORE_CLIENT_SECRET" \
  --var aicore_base_url="$AICORE_BASE_URL" \
  --var aicore_resource_group="$AICORE_RESOURCE_GROUP" \
  --var hana_address="$hana_address" \
  --var hana_port="$hana_port" \
  --var hana_user="$hana_user" \
  --var hana_password="$hana_password" \
  --var hana_encrypt="$hana_encrypt"

echo "✅ Deployment finished." 
