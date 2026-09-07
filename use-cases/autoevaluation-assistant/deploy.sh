#!/bin/bash
#
# Deploy the Evaluation Assessment Assistant application to Cloud Foundry.
#
# Example commands:
#   cf login -a https://api.cf.eu10-005.hana.ondemand.com -o <org> -s <space>
#   ./deploy.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/api/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing api/.env. Create it from api/.env.example before deploying." >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

required_env_vars=(
  HANA_ADDRESS
  HANA_PORT
  HANA_USER
  HANA_PASSWORD
  AICORE_AUTH_URL
  AICORE_CLIENT_ID
  AICORE_CLIENT_SECRET
  AICORE_BASE_URL
  AICORE_RESOURCE_GROUP
)

for var_name in "${required_env_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required value in api/.env: $var_name" >&2
    exit 1
  fi
done

if [[ -z "${API_KEY:-}" || "${API_KEY:-}" == "your-super-secret-api-key" ]]; then
  API_KEY="$(openssl rand -hex 32)"
  echo "Generated temporary API key for this deployment."
  echo "Use this value for the BTP destination X-API-Key header: $API_KEY"
else
  echo "Using API_KEY from api/.env for this deployment."
fi

ALLOWED_ORIGIN="${ALLOWED_ORIGIN:-https://autoevaluation-assistant.cfapps.eu10-005.hana.ondemand.com}"
API_BASE_URL="${API_BASE_URL:-https://autoevaluation-assistant-api.cfapps.eu10-005.hana.ondemand.com}"
AGENT_PUBLIC_URL="${AGENT_PUBLIC_URL:-$API_BASE_URL}"
HANA_ENCRYPT="${HANA_ENCRYPT:-true}"
GENAI_DEFAULT_MODEL="${GENAI_DEFAULT_MODEL:-gpt-5.4}"
GENAI_REVIEW_MODEL="${GENAI_REVIEW_MODEL:-gpt-5.4}"
GENAI_REASONING_EFFORT="${GENAI_REASONING_EFFORT:-low}"
GENAI_EMBEDDING_MODEL="${GENAI_EMBEDDING_MODEL:-text-embedding-3-small}"
GENAI_TEMPERATURE="${GENAI_TEMPERATURE:-0.2}"
GENAI_MAX_TOKENS="${GENAI_MAX_TOKENS:-2048}"
JOULE_A2A_MODEL_NAME="${JOULE_A2A_MODEL_NAME:-gpt-4.1}"
if [[ -z "${LOG_USER_HASH_SALT:-}" ]]; then
  LOG_USER_HASH_SALT="$(openssl rand -hex 32)"
  echo "Generated temporary LOG_USER_HASH_SALT for this deployment."
  echo "Set LOG_USER_HASH_SALT in api/.env to keep user hashes stable across deployments."
fi

echo "Using GENAI_REVIEW_MODEL=$GENAI_REVIEW_MODEL for deployed review workers."
echo "🚀 Deploying application..."
cf push \
  --var api_key="$API_KEY" \
  --var allowed_origin="$ALLOWED_ORIGIN" \
  --var api_base_url="$API_BASE_URL" \
  --var agent_public_url="$AGENT_PUBLIC_URL" \
  --var hana_address="$HANA_ADDRESS" \
  --var hana_port="$HANA_PORT" \
  --var hana_user="$HANA_USER" \
  --var hana_password="$HANA_PASSWORD" \
  --var hana_encrypt="$HANA_ENCRYPT" \
  --var aicore_auth_url="$AICORE_AUTH_URL" \
  --var aicore_client_id="$AICORE_CLIENT_ID" \
  --var aicore_client_secret="$AICORE_CLIENT_SECRET" \
  --var aicore_base_url="$AICORE_BASE_URL" \
  --var aicore_resource_group="$AICORE_RESOURCE_GROUP" \
  --var genai_default_model="$GENAI_DEFAULT_MODEL" \
  --var genai_review_model="$GENAI_REVIEW_MODEL" \
  --var genai_reasoning_effort="$GENAI_REASONING_EFFORT" \
  --var genai_embedding_model="$GENAI_EMBEDDING_MODEL" \
  --var genai_temperature="$GENAI_TEMPERATURE" \
  --var genai_max_tokens="$GENAI_MAX_TOKENS" \
  --var joule_a2a_model_name="$JOULE_A2A_MODEL_NAME" \
  --var log_user_hash_salt="$LOG_USER_HASH_SALT"

echo "Binding backend applications to Cloud Logging..."
cf bind-service "autoevaluation-assistant-api" "Cloud Logging"
cf bind-service "autoevaluation-assistant-worker" "Cloud Logging"

echo "Restarting backend applications so Cloud Logging bindings are available..."
cf restart "autoevaluation-assistant-api"
cf restart "autoevaluation-assistant-worker"

echo "✅ Deployment finished."
echo "Using X-API-Key header: $API_KEY"
