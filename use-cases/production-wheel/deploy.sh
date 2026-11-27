#!/bin/bash
#
# Deploy the Production Wheel Optimizer to Cloud Foundry.
#
# Two deployments of the same code can be chosen:
#
#   private  manifest.yaml. Users sign in through SAP BTP (XSUAA) and need the
#            role collection ProductionWheel_User. Apps production-wheel-api,
#            production-wheel-worker and production-wheel-ui. This is the
#            deployment for customers and productive use.
#   public   manifest-public.yaml. Open showcase WITHOUT sign-in: anyone with the
#            URL can use the app and change its data. Apps
#            production-wheel-public-api and production-wheel-public-ui. It
#            shares the HANA schema and the worker of the private deployment.
#
# What it does, in order:
#   1. Checks api/.env and, for private, that the apps domain in manifest.yaml
#      matches the redirect URI in xs-security.json (the sign-in fails with
#      "invalid redirect" otherwise).
#   2. Builds the UI (ui/) and copies it into approuter/resources, where the
#      application router serves it.
#   3. Private only: creates the XSUAA service instance production-wheel-xsuaa
#      from xs-security.json, or updates it when it already exists.
#   4. Reads the AI Core and HANA credentials from api/.env and runs cf push
#      with the chosen manifest.
#   5. Binds the API to SAP Cloud Logging and restarts it.
#   6. Prints the URL and, for private, how to grant access (role collection).
#
# Prerequisites:
#   - cf CLI, logged in to the target org and space (cf login).
#   - Node.js 22 and npm (to build the UI locally).
#   - api/.env with valid AI Core and HANA credentials (see api/.env.example).
#   - Private: entitlement for "Authorization and Trust Management Service"
#     (xsuaa), plan "application", in the subaccount.
#   - Public: the private deployment's worker (production-wheel-worker) running
#     in the same space, to execute optimizer runs.
#
# Usage (from the repository root):
#   cf login -a https://api.cf.eu10-005.hana.ondemand.com
#   ./deploy.sh private     # with sign-in (customers, productive use)
#   ./deploy.sh public      # open showcase without sign-in
#   ./deploy.sh             # asks which one to deploy
#
# After the first private deploy, assign the role collection
# "ProductionWheel_User" to the users who may sign in (BTP cockpit > Security >
# Role Collections). See docs/user-authentication.md.
#

set -euo pipefail

# --- Configuration ---

ENV_FILE="api/.env"
XSUAA_INSTANCE="production-wheel-xsuaa"
XSUAA_DESCRIPTOR="xs-security.json"
ROLE_COLLECTION="ProductionWheel_User"
# Worker of the private deployment; the public deployment relies on it.
PRIVATE_WORKER_APP_NAME="production-wheel-worker"

# --- Functions ---

# Reads a variable value from the .env file.
# Strips surrounding quotes (single or double) from the value.
read_env_var() {
  local var_name="$1"
  local value
  value=$(grep "^${var_name}=" "$ENV_FILE" | head -1 | cut -d '=' -f2- || true)
  # Remove surrounding quotes
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  echo "$value"
}

# Prints the apps domain of the UI route in the chosen manifest
# (for example cfapps.eu10-005.hana.ondemand.com).
ui_domain() {
  grep -m 1 "route: ${UI_APP_NAME}\." "$MANIFEST" | sed -E "s/.*route: ${UI_APP_NAME}\.//" | tr -d '[:space:]'
}

# Sets TARGET from the first argument, or asks for it when none is given.
choose_target() {
  TARGET="${1:-}"
  if [ -z "$TARGET" ]; then
    if [ ! -t 0 ]; then
      echo "Error: no deployment given. Use ./deploy.sh private or ./deploy.sh public."
      exit 1
    fi
    echo "Which deployment do you want to deploy?"
    echo "  1) private - with SAP BTP sign-in (customers, productive use)"
    echo "  2) public  - open showcase WITHOUT sign-in"
    read -r -p "Choose 1 or 2: " choice
    case "$choice" in
      1) TARGET="private" ;;
      2) TARGET="public" ;;
      *) echo "Error: unknown choice '$choice'."; exit 1 ;;
    esac
  fi
}

# --- Main Script ---

# 0. Choose the deployment and its manifest and app names.
choose_target "${1:-}"
case "$TARGET" in
  private)
    MANIFEST="manifest.yaml"
    API_APP_NAME="production-wheel-api"
    UI_APP_NAME="production-wheel-ui"
    ;;
  public)
    MANIFEST="manifest-public.yaml"
    API_APP_NAME="production-wheel-public-api"
    UI_APP_NAME="production-wheel-public-ui"
    ;;
  *)
    echo "Error: unknown deployment '$TARGET'. Use private or public."
    exit 1
    ;;
esac
echo "Deploying the $TARGET deployment ($MANIFEST)."
if [ "$TARGET" = "public" ]; then
  echo "WARNING: the public deployment has NO sign-in. Anyone with the URL can use"
  echo "the app, change the shared data and use the AI assistant."
fi

# 1. Verify inputs
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found. Please create it with the required credentials."
  echo "See api/.env.example for the expected format."
  exit 1
fi

DOMAIN=$(ui_domain)
if [ -z "$DOMAIN" ]; then
  echo "Error: could not read the route of $UI_APP_NAME from $MANIFEST."
  exit 1
fi
if [ "$TARGET" = "private" ] && ! grep -q "https://\*\.${DOMAIN}/\*\*" "$XSUAA_DESCRIPTOR"; then
  echo "Error: $XSUAA_DESCRIPTOR has no redirect URI for the apps domain $DOMAIN."
  echo "Set oauth2-configuration.redirect-uris to [\"https://*.${DOMAIN}/**\"]."
  exit 1
fi

# 2. Build the UI and hand it to the approuter.
# An empty VITE_API_BASE_URL makes the UI call /api on its own origin, through
# the approuter (which, in the private deployment, adds the user's token).
echo "Building the UI..."
(cd ui && npm ci && VITE_API_BASE_URL="" npm run build)
rm -rf approuter/resources
cp -R ui/dist approuter/resources

# 3. Private only: create or update the XSUAA service instance (sign-in and the
# access role). The public deployment has no sign-in and does not use it.
if [ "$TARGET" = "private" ]; then
  if cf service "$XSUAA_INSTANCE" > /dev/null 2>&1; then
    echo "Updating XSUAA service $XSUAA_INSTANCE from $XSUAA_DESCRIPTOR..."
    cf update-service "$XSUAA_INSTANCE" -c "$XSUAA_DESCRIPTOR"
  else
    echo "Creating XSUAA service $XSUAA_INSTANCE from $XSUAA_DESCRIPTOR..."
    cf create-service xsuaa application "$XSUAA_INSTANCE" -c "$XSUAA_DESCRIPTOR" --wait
  fi
fi

# 4. Read credentials from api/.env and deploy the applications of the manifest.
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
HANA_SCHEMA=$(read_env_var "HANA_SCHEMA")

echo "Deploying applications from $MANIFEST..."
cf push -f "$MANIFEST" \
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
  --var hana_schema="$HANA_SCHEMA"

# 5. Bind the API to SAP Cloud Logging and restart it to pick up the binding.
echo "Binding $API_APP_NAME to Cloud Logging..."
cf bind-service "$API_APP_NAME" "Cloud Logging"
echo "Restarting $API_APP_NAME..."
cf restart "$API_APP_NAME"

# 6. Next steps.
echo
echo "Deployment finished ($TARGET)."
if [ "$TARGET" = "private" ]; then
  echo "Sign-in URL: https://${UI_APP_NAME}.${DOMAIN}"
  echo "Grant access: BTP cockpit > your subaccount > Security > Role Collections >"
  echo "  ${ROLE_COLLECTION} > Edit > add the users (or map an identity-provider group)."
  echo "Users without this role collection get 403 Forbidden after signing in."
else
  echo "Public URL (no sign-in): https://${UI_APP_NAME}.${DOMAIN}"
  if ! cf app "$PRIVATE_WORKER_APP_NAME" > /dev/null 2>&1; then
    echo "WARNING: $PRIVATE_WORKER_APP_NAME was not found in this space. Optimizer runs"
    echo "launched from the public app stay 'queued' until ./deploy.sh private is run."
  fi
  echo "Stop it after the demo with: cf stop $API_APP_NAME && cf stop $UI_APP_NAME"
fi
