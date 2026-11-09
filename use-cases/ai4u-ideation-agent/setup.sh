#!/bin/bash
#
# AI4U Ideation Agent — First-Time Setup Wizard
#
# Interactive script that sets up everything needed to deploy the app
# on your SAP BTP Cloud Foundry landscape:
#
#   1. Detects your CF environment (region, org, space)
#   2. Creates/verifies required service instances
#   3. Extracts credentials automatically from service keys
#   4. Generates manifest.yaml, xs-security.json, and api/.env.production
#   5. Optionally deploys the app
#
# Prerequisites:
#   - CF CLI installed and logged in: cf login -a <api-endpoint> --sso
#   - AI Core instance already provisioned (via BTP Cockpit)
#   - HANA Cloud instance provisioned with a technical user
#
# Usage:
#   ./setup.sh
#
set -eo pipefail

trap 'echo ""; error "Setup interrupted. Re-run ./setup.sh — existing services will be detected and skipped."' ERR

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

info()    { echo -e "\033[1;34mℹ\033[0m  $*" >&2; }
success() { echo -e "\033[1;32m✓\033[0m  $*" >&2; }
warn()    { echo -e "\033[1;33m⚠\033[0m  $*" >&2; }
error()   { echo -e "\033[1;31m✗\033[0m  $*" >&2; }
ask()     { echo -en "\033[1;36m?\033[0m  $* " >&2; }

# Prompt with default value: prompt_default "Question" "default"
prompt_default() {
  local prompt="$1" default="$2" answer
  ask "$prompt [$default]: "
  read -r answer
  echo "${answer:-$default}"
}

# Prompt required (no default): prompt_required "Question"
prompt_required() {
  local prompt="$1" answer
  while true; do
    ask "$prompt: "
    read -r answer
    [[ -n "$answer" ]] && break
    warn "This field is required."
  done
  echo "$answer"
}

# Prompt yes/no: prompt_yn "Question" "y" → returns 0 (yes) or 1 (no)
prompt_yn() {
  local prompt="$1" default="${2:-y}" answer
  if [[ "$default" == "y" ]]; then
    ask "$prompt [Y/n]: "
  else
    ask "$prompt [y/N]: "
  fi
  read -r answer
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy] ]]
}

# Prompt for password (hidden input)
prompt_secret() {
  local prompt="$1" answer
  while true; do
    ask "$prompt: "
    read -rs answer
    echo "" >&2
    [[ -n "$answer" ]] && break
    warn "This field is required."
  done
  echo "$answer"
}

# Check if a CF service instance exists
service_exists() {
  cf service "$1" &>/dev/null
}

# ═══════════════════════════════════════════════════════════════════════════════
# Preflight
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       AI4U Ideation Agent — First-Time Setup Wizard          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check cf CLI
if ! command -v cf &>/dev/null; then
  error "CF CLI not found. Install it first:"
  echo "   https://docs.cloudfoundry.org/cf-cli/install-go-cli.html"
  exit 1
fi

# Check cf is logged in
if ! cf target &>/dev/null; then
  error "Not logged in to Cloud Foundry. Run:"
  echo "   cf login -a https://api.cf.<region>.hana.ondemand.com --sso"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Detect CF Environment
# ═══════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1: Detect CF Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parse cf target output
CF_API=$(cf target | grep "API endpoint" | awk '{print $3}')
CF_ORG=$(cf target | grep "org:" | awk '{print $2}')
CF_SPACE=$(cf target | grep "space:" | awk '{print $2}')

# Extract region from API URL: https://api.cf.eu10-005.hana.ondemand.com → eu10-005
CF_REGION=$(echo "$CF_API" | sed -n 's|.*api\.cf\.\(.*\)\.hana\.ondemand\.com.*|\1|p')
if [[ -z "$CF_REGION" ]]; then
  CF_REGION=$(prompt_required "Could not detect CF region. Enter it (e.g., eu10-005, us10-001)")
fi

# Derive base region (eu10-005 → eu10, us10-001 → us10)
CF_REGION_BASE=$(echo "$CF_REGION" | sed 's/-[0-9]*$//')

info "Detected environment:"
echo "   API:    $CF_API"
echo "   Org:    $CF_ORG"
echo "   Space:  $CF_SPACE"
echo "   Region: $CF_REGION (base: $CF_REGION_BASE)"
echo ""

# App prefix
APP_PREFIX=$(prompt_default "App name prefix" "ai4u-ideation-agent")
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Service Instances
# ═══════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2: Create / Verify Service Instances"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# --- XSUAA ---
XSUAA_INSTANCE=$(prompt_default "XSUAA instance name" "${APP_PREFIX}-xsuaa")

if service_exists "$XSUAA_INSTANCE"; then
  success "XSUAA instance '$XSUAA_INSTANCE' already exists."
else
  info "Creating XSUAA instance '$XSUAA_INSTANCE'..."
  XSAPPNAME=$(prompt_default "XSUAA xsappname" "${APP_PREFIX}")

  # Generate xs-security.json
  cat > xs-security.json <<EOF
{
  "xsappname": "${XSAPPNAME}",
  "tenant-mode": "dedicated",
  "oauth2-configuration": {
    "redirect-uris": [
      "https://${APP_PREFIX}.cfapps.${CF_REGION}.hana.ondemand.com/**"
    ]
  }
}
EOF
  cf create-service xsuaa application "$XSUAA_INSTANCE" -c xs-security.json
  success "XSUAA instance created."
fi

# Create service key if needed
XSUAA_KEY="${XSUAA_INSTANCE}-key"
if cf service-key "$XSUAA_INSTANCE" "$XSUAA_KEY" &>/dev/null; then
  success "XSUAA service key '$XSUAA_KEY' exists."
else
  info "Creating XSUAA service key..."
  cf create-service-key "$XSUAA_INSTANCE" "$XSUAA_KEY"
  success "XSUAA service key created."
fi
echo ""

# --- Object Store ---
OBJECTSTORE_INSTANCE=$(prompt_default "Object Store instance name" "${APP_PREFIX}-objectstore")

if service_exists "$OBJECTSTORE_INSTANCE"; then
  success "Object Store instance '$OBJECTSTORE_INSTANCE' already exists."
else
  info "Creating Object Store instance '$OBJECTSTORE_INSTANCE'..."
  cf create-service objectstore s3-standard "$OBJECTSTORE_INSTANCE"
  success "Object Store instance created."
fi
echo ""

# --- AI Core ---
info "AI Core must be provisioned via BTP Cockpit (cannot be created via CLI)."
echo ""

# Try to detect existing AI Core instances
AICORE_INSTANCES=$(cf services | grep "aicore" | awk '{print $1}' || true)
if [[ -n "$AICORE_INSTANCES" ]]; then
  info "Found AI Core instance(s) in this space:"
  echo "$AICORE_INSTANCES" | while read -r name; do echo "   • $name"; done
  echo ""
  AICORE_INSTANCE=$(prompt_default "AI Core instance name" "$(echo "$AICORE_INSTANCES" | head -1)")
else
  warn "No AI Core instance found in this space."
  echo "   Provision one via BTP Cockpit → Instances and Subscriptions → AI Core"
  echo "   Guide: https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/initial-setup"
  echo ""
  AICORE_INSTANCE=$(prompt_required "Enter your AI Core instance name (or provision one and re-run)")
fi

if service_exists "$AICORE_INSTANCE"; then
  success "AI Core instance '$AICORE_INSTANCE' verified."
else
  error "AI Core instance '$AICORE_INSTANCE' not found in this space."
  echo "   Provision it first, then re-run this script."
  exit 1
fi
echo ""

# --- HANA Cloud ---
info "HANA Cloud must be provisioned via BTP Cockpit / HANA Cloud Central."
echo ""

HANA_INSTANCE=$(prompt_default "HANA service instance name (bound in manifest)" "${APP_PREFIX}-hana")
if service_exists "$HANA_INSTANCE"; then
  success "HANA instance '$HANA_INSTANCE' found."
else
  warn "HANA instance '$HANA_INSTANCE' not found."
  echo "   If your HANA binding has a different name, enter it."
  echo "   Guide: https://help.sap.com/docs/hana-cloud/sap-hana-cloud-administration-guide/provision-sap-hana-cloud"
  HANA_INSTANCE=$(prompt_required "HANA service instance name")
fi
echo ""

info "HANA technical user credentials (for DDL access):"
echo "   If you haven't created a technical user yet, run this SQL in HANA Cloud Central:"
echo ""
echo "     CREATE USER AI4U_APP PASSWORD \"<password>\" NO FORCE_FIRST_PASSWORD_CHANGE;"
echo "     CREATE SCHEMA AI4U_IDEATION;"
echo "     GRANT ALL ON SCHEMA AI4U_IDEATION TO AI4U_APP;"
echo ""
HANA_HOST=$(prompt_required "HANA host (e.g., <guid>.hana.prod-eu10.hanacloud.ondemand.com)")
HANA_USER=$(prompt_default "HANA user" "AI4U_APP")
HANA_PASSWORD=$(prompt_secret "HANA password")
HANA_SCHEMA=$(prompt_default "HANA schema" "AI4U_IDEATION")
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Extract Credentials
# ═══════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 3: Extract Credentials"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

info "Reading XSUAA service key..."
XSUAA_KEY_JSON=$(cf service-key "$XSUAA_INSTANCE" "$XSUAA_KEY" 2>/dev/null | sed -n '/^{/,/^}/p')

# Extract fields (try jq first, fallback to grep/sed)
if command -v jq &>/dev/null; then
  XSUAA_CLIENT_ID=$(echo "$XSUAA_KEY_JSON" | jq -r '.clientid // empty')
  XSUAA_SUBDOMAIN=$(echo "$XSUAA_KEY_JSON" | jq -r '.identityzone // .subaccountid // empty')
else
  XSUAA_CLIENT_ID=$(echo "$XSUAA_KEY_JSON" | grep '"clientid"' | sed 's/.*: "\(.*\)".*/\1/')
  XSUAA_SUBDOMAIN=$(echo "$XSUAA_KEY_JSON" | grep '"identityzone"' | sed 's/.*: "\(.*\)".*/\1/')
  if [[ -z "$XSUAA_SUBDOMAIN" ]]; then
    XSUAA_SUBDOMAIN=$(echo "$XSUAA_KEY_JSON" | grep '"subaccountid"' | sed 's/.*: "\(.*\)".*/\1/')
  fi
fi

if [[ -z "$XSUAA_CLIENT_ID" ]]; then
  error "Could not extract XSUAA client ID from service key."
  XSUAA_CLIENT_ID=$(prompt_required "Enter XSUAA client ID manually")
fi
if [[ -z "$XSUAA_SUBDOMAIN" ]]; then
  XSUAA_SUBDOMAIN=$(prompt_required "Enter your BTP subdomain (for XSUAA URL)")
fi

success "XSUAA Client ID: $XSUAA_CLIENT_ID"
success "XSUAA Subdomain: $XSUAA_SUBDOMAIN"
echo ""

# Admin emails
ADMIN_EMAILS=$(prompt_default "Admin email(s) (comma-separated)" "")
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Generate Configuration Files
# ═══════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 4: Generate Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# --- manifest.yaml ---
if [[ -f "manifest.yaml" ]]; then
  if prompt_yn "manifest.yaml already exists. Overwrite?" "n"; then
    :
  else
    warn "Skipping manifest.yaml generation. You may need to update it manually."
    SKIP_MANIFEST=true
  fi
fi

if [[ -z "$SKIP_MANIFEST" ]]; then
  # Check if CDC is configured
  CDC_API_KEY=""
  CDC_BASE_URL=""
  if prompt_yn "Do you have SAP CDC (Customer Data Cloud) for external user login?" "n"; then
    CDC_API_KEY=$(prompt_required "CDC API Key")
    CDC_BASE_URL=$(prompt_default "CDC Base URL" "https://accounts.us1.gigya.com")
  fi

  sed -e "s|((app_prefix))|${APP_PREFIX}|g" \
      -e "s|((cf_region))|${CF_REGION}|g" \
      -e "s|((cf_region_base))|${CF_REGION_BASE}|g" \
      -e "s|((xsuaa_instance))|${XSUAA_INSTANCE}|g" \
      -e "s|((hana_instance))|${HANA_INSTANCE}|g" \
      -e "s|((objectstore_instance))|${OBJECTSTORE_INSTANCE}|g" \
      -e "s|((aicore_instance))|${AICORE_INSTANCE}|g" \
      -e "s|((xsuaa_subdomain))|${XSUAA_SUBDOMAIN}|g" \
      -e "s|((xsuaa_client_id))|${XSUAA_CLIENT_ID}|g" \
      -e "s|((cdc_api_key))|${CDC_API_KEY}|g" \
      -e "s|((cdc_base_url))|${CDC_BASE_URL}|g" \
      manifest.yaml.template > manifest.yaml

  success "Generated manifest.yaml"
fi

# --- xs-security.json ---
if [[ -f "xs-security.json" ]] && ! prompt_yn "xs-security.json already exists. Overwrite?" "n"; then
  warn "Skipping xs-security.json generation."
else
  XSAPPNAME=$(prompt_default "XSUAA xsappname" "${APP_PREFIX}")
  sed -e "s|((xsappname))|${XSAPPNAME}|g" \
      -e "s|((app_prefix))|${APP_PREFIX}|g" \
      -e "s|((cf_region))|${CF_REGION}|g" \
      xs-security.json.template > xs-security.json
  success "Generated xs-security.json"
fi

# --- api/.env.production ---
if [[ -f "api/.env.production" ]]; then
  if prompt_yn "api/.env.production already exists. Overwrite?" "n"; then
    :
  else
    warn "Skipping api/.env.production generation."
    SKIP_ENV=true
  fi
fi

if [[ -z "$SKIP_ENV" ]]; then
  (umask 077; cat > api/.env.production <<EOF
# AI4U Ideation Agent — Production Environment (CF deployment)
# Generated by: ./setup.sh
# See api/.env.example for all available variables.

# === REQUIRED ===

# HANA Cloud — technical user with DDL privileges on your schema
HANA_HOST=${HANA_HOST}
HANA_PORT=443
HANA_USER=${HANA_USER}
HANA_PASSWORD=${HANA_PASSWORD}
HANA_ENCRYPT=true
HANA_SCHEMA=${HANA_SCHEMA}

# Admin users (comma-separated emails that get admin access)
ADMIN_EMAILS=${ADMIN_EMAILS}

# === OPTIONAL FEATURES (uncomment to enable) ===

# CDC - External user OTP login
# CDC_API_KEY=""
# CDC_DATA_CENTER="us1.gigya.com"
# CDC_USER_KEY=""
# CDC_SECRET_KEY=""

# Perplexity research (requires Sonar model deployed in AI Core)
# IDEATION_PERPLEXITY_SKILL_ENABLED="true"

# LinkedIn hiring intelligence
# IDEATION_LINKEDIN_SKILL_ENABLED="true"
EOF
  )
  success "Generated api/.env.production"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Summary & Deploy
# ═══════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
info "Configuration summary:"
echo "   Region:         $CF_REGION"
echo "   App prefix:     $APP_PREFIX"
echo "   UI route:       https://${APP_PREFIX}.cfapps.${CF_REGION}.hana.ondemand.com"
echo "   API route:      https://${APP_PREFIX}-api.cfapps.${CF_REGION}.hana.ondemand.com"
echo "   XSUAA:          $XSUAA_INSTANCE"
echo "   Object Store:   $OBJECTSTORE_INSTANCE"
echo "   AI Core:        $AICORE_INSTANCE"
echo "   HANA:           $HANA_INSTANCE"
echo "   HANA user:      $HANA_USER @ $HANA_SCHEMA"
echo ""

if prompt_yn "Deploy now?" "y"; then
  echo ""
  info "Starting deployment..."
  echo ""
  ./deploy.sh
else
  echo ""
  info "Run ./deploy.sh when you're ready to deploy."
  echo "   For subsequent deploys (code updates), just run ./deploy.sh directly."
fi
