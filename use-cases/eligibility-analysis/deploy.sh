#!/bin/bash
#
# Deploy the Invoice Workspace to Cloud Foundry and wire the Joule destination.
#
# Examples (repository root):
#   ./deploy.sh                      # reuse the deployed API key (a new one only on first deploy)
#   ./deploy.sh --rotate-api-key     # new API key; UI, destination and CF app are all updated
#   ./deploy.sh --skip-destination   # CF apps only (no BTP destination update)
#   BTP_SUBACCOUNT_ID=<id> ./deploy.sh
#
# Prerequisites:
#   - cf login + cf target (org/space of the apps in manifest.yaml)
#   - btp login + btp target <subaccount> (unless --skip-destination or BTP_SUBACCOUNT_ID is set)
#   - api/.env with AICORE_* and hana_* values (see api/.env.example)
#
# Steps:
#   1. Pre-flight checks (CLI logins, required variables).
#   2. Resolve the API key: reuse the one of the deployed API app unless --rotate-api-key.
#      Rotating also invalidates saved assistant conversations (their access domain
#      is derived from the key) and requires a UI rebuild, which this push performs.
#   3. cf push with secrets in a mode-600 vars file (removed on exit).
#   4. Bind "Cloud Logging" and restart the API.
#   5. Create/update the BTP destination RECEIVABLES_AGENT (URL.headers.X-API-Key).
#   6. Validate the Joule capability locally and print the manual joule deploy commands.
#   7. Smoke-check health, agent card and API-key enforcement.
#
# Joule deployment itself (joule deploy/launch) is always run manually.
#

set -euo pipefail
cd "$(dirname "$0")"

API_APP="eligibility-analysis-api"
DESTINATION_NAME="RECEIVABLES_AGENT"
JOULE_ASSISTANT="receivables_assistant"
ROTATE_API_KEY=0
SKIP_DESTINATION=0

for arg in "$@"; do
  case "$arg" in
    --rotate-api-key) ROTATE_API_KEY=1 ;;
    --skip-destination) SKIP_DESTINATION=1 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"

# --- Load allowed secrets from api/.env (shell exports win) ---
if [ -f api/.env ]; then
  eval "$(
    "$PY" - <<'PY'
import os
import shlex
from pathlib import Path

allowed = {
    "AICORE_AUTH_URL", "AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET", "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP", "LOG_USER_HASH_SALT",
    "hana_address", "hana_port", "hana_user", "hana_password", "hana_encrypt",
}
for line in Path("api/.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key, value = key.strip(), value.strip()
    if key not in allowed or key in os.environ:
        continue
    if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
        value = value[1:-1]
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
fi

: "${AICORE_RESOURCE_GROUP:=default}"
: "${hana_port:=443}"
: "${hana_encrypt:=true}"

# --- 1. Pre-flight ---
echo "==> Pre-flight checks"
cf target >/dev/null 2>&1 || die "cf CLI is not logged in/targeted. Run: cf login && cf target -o <org> -s <space>"
missing=()
for var in AICORE_AUTH_URL AICORE_CLIENT_ID AICORE_CLIENT_SECRET AICORE_BASE_URL hana_address hana_user hana_password; do
  [ -n "${!var:-}" ] || missing+=("$var")
done
[ ${#missing[@]} -eq 0 ] || die "Missing required variables (api/.env or shell): ${missing[*]}. The workspace requires HANA."

API_BASE_URL=$(sed -n 's/^ *API_BASE_URL: *//p' manifest.yaml | head -1)
[ -n "$API_BASE_URL" ] || die "API_BASE_URL not found in manifest.yaml"

SUBACCOUNT="${BTP_SUBACCOUNT_ID:-}"
if [ "$SKIP_DESTINATION" -eq 0 ] && [ -z "$SUBACCOUNT" ]; then
  command -v btp >/dev/null || die "btp CLI not installed. Install it, or use --skip-destination."
  SUBACCOUNT=$(btp --info 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | "$PY" -c 'import re,sys; m=re.findall(r"subaccount, ID: ([0-9a-f-]{36})", sys.stdin.read()); print(m[-1] if m else "")')
  [ -n "$SUBACCOUNT" ] || die "btp is not targeting a subaccount. Run: btp login --sso && btp target (choose the subaccount)"
fi

# --- 2. API key ---
echo "==> Resolving API key"
API_KEY=""
KEY_SOURCE="generated"
if [ "$ROTATE_API_KEY" -eq 0 ] && APP_GUID=$(cf app "$API_APP" --guid 2>/dev/null); then
  API_KEY=$(cf curl "/v3/apps/$APP_GUID/environment_variables" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("var",{}).get("API_KEY",""))')
  [ -n "$API_KEY" ] && KEY_SOURCE="reused from deployed app"
fi
if [ -z "$API_KEY" ]; then
  API_KEY=$(openssl rand -hex 32)
fi
echo "    API key: $KEY_SOURCE"
if [ -z "${LOG_USER_HASH_SALT:-}" ]; then
  LOG_USER_HASH_SALT=$(openssl rand -hex 32)
  echo "    Generated LOG_USER_HASH_SALT (store it in api/.env to keep user hashes stable)"
fi

# --- 3. Push with a private vars file ---
VARS_FILE=$(mktemp)
chmod 600 "$VARS_FILE"
trap 'rm -f "$VARS_FILE"' EXIT
export API_KEY LOG_USER_HASH_SALT AICORE_AUTH_URL AICORE_CLIENT_ID AICORE_CLIENT_SECRET AICORE_BASE_URL \
  AICORE_RESOURCE_GROUP hana_address hana_port hana_user hana_password hana_encrypt
"$PY" - "$VARS_FILE" <<'PY'
import json
import os
import sys

names = {
    "api_key": "API_KEY", "log_user_hash_salt": "LOG_USER_HASH_SALT",
    "aicore_auth_url": "AICORE_AUTH_URL", "aicore_client_id": "AICORE_CLIENT_ID",
    "aicore_client_secret": "AICORE_CLIENT_SECRET", "aicore_base_url": "AICORE_BASE_URL",
    "aicore_resource_group": "AICORE_RESOURCE_GROUP", "hana_address": "hana_address",
    "hana_port": "hana_port", "hana_user": "hana_user", "hana_password": "hana_password",
    "hana_encrypt": "hana_encrypt",
}
# JSON is valid YAML, so cf push --vars-file reads it directly.
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({var: os.environ[env] for var, env in names.items()}, handle)
PY

echo "==> cf push (API and UI)"
cf push --vars-file "$VARS_FILE"

# --- 4. Logging binding ---
echo "==> Binding $API_APP to Cloud Logging"
cf bind-service "$API_APP" "Cloud Logging"
cf restart "$API_APP"

# --- 5. Joule destination ---
if [ "$SKIP_DESTINATION" -eq 0 ]; then
  echo "==> Upserting BTP destination $DESTINATION_NAME in subaccount $SUBACCOUNT"
  JOULE_DESTINATION_API_KEY="$API_KEY" "$PY" api/scripts/upsert_joule_destination.py \
    --name "$DESTINATION_NAME" --url "$API_BASE_URL" --subaccount "$SUBACCOUNT" --apply
else
  echo "==> Skipping destination (--skip-destination). Update $DESTINATION_NAME manually if the key changed."
fi

# --- 6. Joule validation ---
echo "==> Validating the Joule capability"
PYTHONPATH=api "$PY" api/scripts/validate_joule_bridge.py --root .
if command -v joule >/dev/null; then
  joule lint joule/a2a
  if joule status 2>&1 | grep -qi "logged in" && ! joule status 2>&1 | grep -qi "logged out"; then
    COMPILE_DIR=$(mktemp -d)
    joule compile . "$COMPILE_DIR/$JOULE_ASSISTANT.daar"
    rm -rf "$COMPILE_DIR"
  else
    echo "    joule is logged out: compile skipped (run 'joule login' first)."
  fi
else
  echo "    joule CLI not installed: lint/compile skipped (npm i -g @sap/joule-studio-cli)."
fi

# --- 7. Smoke checks ---
echo "==> Smoke checks against $API_BASE_URL"
status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
[ "$(status "$API_BASE_URL/api/health")" = "200" ] && echo "    health: 200" || echo "    WARNING: /api/health did not return 200"
curl -s "$API_BASE_URL/.well-known/agent-card.json" | grep -q "$API_BASE_URL/api/a2a" \
  && echo "    agent card advertises $API_BASE_URL/api/a2a" || echo "    WARNING: agent card does not advertise the deployed endpoint"
[ "$(status -X POST -H 'Content-Type: application/json' -d '{}' "$API_BASE_URL/api/a2a")" = "403" ] \
  && echo "    /api/a2a without key: 403" || echo "    WARNING: /api/a2a without key did not return 403"

cat <<EOF

Deployment finished. Run the Joule steps manually (Node.js 20.12-24, from the repository root):
  joule login
  joule deploy -c -n $JOULE_ASSISTANT
  joule launch $JOULE_ASSISTANT
Then ask Joule: "List my saved offers" and check a second question continues the conversation.
EOF
