#!/bin/bash
#
# """
# Examples:
#   ./deploy.sh
#   DEPLOY_ENV_FILE=api/.env ./deploy.sh
#   chmod +x deploy.sh && ./deploy.sh
# """
#
# This script automates the deployment of the application to Cloud Foundry.
# It generates a secure, random API key and injects it into the deployment
# process, injects required HANA runtime and AI Core configuration from the local
# environment or api/.env, and stages clean deploy directories so Cloud
# Foundry does not upload local outputs, caches, tests, or node_modules.
#

set -euo pipefail

# --- Main Script ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/coa-cf-deploy.XXXXXX")"
TEMP_MANIFEST="$TEMP_DIR/manifest.yaml"
TEMP_VARS_FILE="$TEMP_DIR/vars.yaml"
TEMP_API_DIR="$TEMP_DIR/api"
TEMP_UI_DIR="$TEMP_DIR/ui"
DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-$SCRIPT_DIR/api/.env}"
HANA_ENV_VARS=(
  "hana_address"
  "hana_port"
  "hana_user"
  "hana_password"
  "hana_encrypt"
)
AI_CORE_ENV_VARS=(
  "AICORE_AUTH_URL"
  "AICORE_CLIENT_ID"
  "AICORE_CLIENT_SECRET"
  "AICORE_BASE_URL"
  "AICORE_RESOURCE_GROUP"
)

cleanup() {
  rm -rf "$TEMP_DIR"
}

ensure_cf_cli() {
  if command -v cf >/dev/null 2>&1; then
    return
  fi

  if [[ -x /opt/homebrew/bin/cf ]]; then
    export PATH="/opt/homebrew/bin:$PATH"
    return
  fi

  echo "❌ Cloud Foundry CLI not found on PATH."
  exit 1
}

load_deploy_env_file() {
  if [[ -f "$DEPLOY_ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$DEPLOY_ENV_FILE"
    set +a
    echo "🔧 Loaded deployment variables from $DEPLOY_ENV_FILE."
    return
  fi

  if [[ "$DEPLOY_ENV_FILE" != "$SCRIPT_DIR/api/.env" ]]; then
    echo "❌ DEPLOY_ENV_FILE was set but does not exist: $DEPLOY_ENV_FILE"
    exit 1
  fi
}

validate_hana_deploy_env() {
  local missing=()
  local env_name

  for env_name in "${AI_CORE_ENV_VARS[@]}"; do
    if [[ -z "${!env_name:-}" ]]; then
      missing+=("$env_name")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "❌ Missing required AI Core deployment variables: ${missing[*]}"
    echo "   Set them in your shell or in api/.env before running ./deploy.sh."
    echo "   The chat classifier uses GenAI Hub credentials to map user text to available answer options."
    exit 1
  fi

  local hana_missing=()
  local env_name

  for env_name in "${HANA_ENV_VARS[@]}"; do
    if [[ -z "${!env_name:-}" ]]; then
      hana_missing+=("$env_name")
    fi
  done

  if (( ${#hana_missing[@]} > 0 )); then
    echo "❌ Missing required HANA deployment variables: ${hana_missing[*]}"
    echo "   Set them in your shell or in api/.env before running ./deploy.sh."
    echo "   The API loads HANA-backed runtime data on startup and cannot start in production without them."
    exit 1
  fi

  if ! [[ "$hana_port" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid hana_port value: $hana_port"
    echo "   hana_port must be an integer, for example 443 or 39015."
    exit 1
  fi

  case "$hana_encrypt" in
    1|0|true|false|TRUE|FALSE|True|False|yes|no|YES|NO|Yes|No|y|n|Y|N|t|f|T|F|on|off|ON|OFF|On|Off)
      ;;
    *)
      echo "❌ Invalid hana_encrypt value: $hana_encrypt"
      echo "   Use one of: true, false, 1, 0, yes, no."
      exit 1
      ;;
  esac
}

yaml_single_quote() {
  local escaped
  escaped="$(printf "%s" "$1" | sed "s/'/''/g")"
  printf "'%s'" "$escaped"
}

write_cf_vars_file() {
  umask 077
  {
    printf "api_key: %s\n" "$(yaml_single_quote "$API_KEY")"
    printf "hana_address: %s\n" "$(yaml_single_quote "$hana_address")"
    printf "hana_port: %s\n" "$(yaml_single_quote "$hana_port")"
    printf "hana_user: %s\n" "$(yaml_single_quote "$hana_user")"
    printf "hana_password: %s\n" "$(yaml_single_quote "$hana_password")"
    printf "hana_encrypt: %s\n" "$(yaml_single_quote "$hana_encrypt")"
    printf "AICORE_AUTH_URL: %s\n" "$(yaml_single_quote "$AICORE_AUTH_URL")"
    printf "AICORE_CLIENT_ID: %s\n" "$(yaml_single_quote "$AICORE_CLIENT_ID")"
    printf "AICORE_CLIENT_SECRET: %s\n" "$(yaml_single_quote "$AICORE_CLIENT_SECRET")"
    printf "AICORE_BASE_URL: %s\n" "$(yaml_single_quote "$AICORE_BASE_URL")"
    printf "AICORE_RESOURCE_GROUP: %s\n" "$(yaml_single_quote "$AICORE_RESOURCE_GROUP")"
  } > "$TEMP_VARS_FILE"
}

stage_directory() {
  local source_dir="$1"
  local target_dir="$2"
  shift 2

  mkdir -p "$target_dir"

  local rsync_args=(
    -a
    --delete
  )

  for pattern in "$@"; do
    rsync_args+=("--exclude=$pattern")
  done

  rsync "${rsync_args[@]}" "$source_dir"/ "$target_dir"/
}

trap cleanup EXIT
ensure_cf_cli
load_deploy_env_file
validate_hana_deploy_env

stage_directory \
  "$SCRIPT_DIR/api" \
  "$TEMP_API_DIR" \
  ".env" \
  ".env.example" \
  ".pytest_cache/" \
  "tests/" \
  "output/" \
  "data/" \
  "data_seed/" \
  "chat_threads.sqlite" \
  "__pycache__/" \
  "*.pyc"

stage_directory \
  "$SCRIPT_DIR/ui" \
  "$TEMP_UI_DIR" \
  ".env" \
  ".env.example" \
  "node_modules/" \
  "dist/" \
  "dist-ssr/" \
  "coverage/" \
  "logs/" \
  "tests/" \
  ".vite/"

sed \
  -e "s|path: api|path: $TEMP_API_DIR|" \
  -e "s|path: ui|path: $TEMP_UI_DIR|" \
  "$SCRIPT_DIR/manifest.yaml" > "$TEMP_MANIFEST"

# 1. Generate a secure, random API key
# This creates a 32-byte (256-bit) random key and encodes it in hexadecimal.
API_KEY=$(openssl rand -hex 32)
write_cf_vars_file
echo "🔑 Generated temporary API key for this deployment."
echo "🔐 HANA runtime variables will be injected into the API app from the local deployment environment."
echo "📦 Prepared clean deployment directories in $TEMP_DIR."

# 2. Deploy the application using the generated key
# The `cf push` command reads the manifest.yaml and replaces the variables
# from a temporary vars file that is removed when the script exits.
#
# Note: You must be logged into Cloud Foundry before running this script.
echo "🚀 Deploying application..."
cf push -f "$TEMP_MANIFEST" --vars-file "$TEMP_VARS_FILE"

echo "✅ Deployment finished."
