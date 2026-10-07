#!/bin/bash
#
# This script automates the deployment of the application to Cloud Foundry.
# It generates a secure, random API key and reads service credentials from
# the local api/.env file, injecting them into the deployment process.
#
# Prerequisites:
#   - You must be logged into Cloud Foundry before running this script.
#   - The file api/.env must exist with valid credentials.
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

# 4. Deploy the application
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
  --var hana_encrypt="$HANA_ENCRYPT"

# 5. S/4HANA connectivity (BTP Destination + Connectivity). Only the btp-mode
#    variables are copied; direct-mode S/4 credentials never leave the laptop.
#    Values are applied quietly so secrets are not echoed to the terminal.
if [ -n "$(read_env_var "DESTINATION_CLIENT_ID")" ]; then
  echo "Setting S/4 BTP connectivity variables on $APP_NAME..."
  cf set-env "$APP_NAME" S4_CONNECTIVITY_MODE "btp" > /dev/null
  for VAR in S4_CLIENT S4_VERIFY S4_DESTINATION_NAME DESTINATION_SERVICE_URI DESTINATION_TOKEN_BASE_URL \
             DESTINATION_CLIENT_ID DESTINATION_CLIENT_SECRET CONNECTIVITY_PROXY_HOST CONNECTIVITY_PROXY_PORT \
             CONNECTIVITY_TOKEN_BASE_URL CONNECTIVITY_CLIENT_ID CONNECTIVITY_CLIENT_SECRET BTP_DEST_DEBUG \
             S4_PAYMENT_ADVICE_TYPE S4_AUTO_POST S4_COMPANY_CODE_ALIASES S4_REASON_CODE_ALIASES; do
    VALUE=$(read_env_var "$VAR")
    if [ -n "$VALUE" ]; then
      cf set-env "$APP_NAME" "$VAR" "$VALUE" > /dev/null
    fi
  done
else
  echo "No S/4 BTP connectivity variables in $ENV_FILE; S/4 check/post stays unconfigured in Cloud Foundry."
fi

# 5b. Document AI client id and priority-customer seed list. Mirrors api/.env:
#     set when defined there, unset otherwise so the code defaults apply
#     (ai4u_payment_advice and the packaged anonymized seed list).
for VAR in DOCUMENT_AI_CLIENT_ID PAYMENT_ADVICE_SEED_CUSTOMERS_PATH; do
  VALUE=$(read_env_var "$VAR")
  if [ -n "$VALUE" ]; then
    echo "Setting $VAR=$VALUE on $APP_NAME..."
    cf set-env "$APP_NAME" "$VAR" "$VALUE" > /dev/null
  else
    cf unset-env "$APP_NAME" "$VAR" > /dev/null
  fi
done

# 6. Bind the application to SAP Cloud Logging
echo "Binding $APP_NAME to Cloud Logging..."
cf bind-service "$APP_NAME" "Cloud Logging"

# 7. Restart the application to pick up the new service binding and variables
echo "Restarting $APP_NAME..."
cf restart "$APP_NAME"

echo "Deployment finished."
