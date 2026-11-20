#!/usr/bin/env bash
# Check the local UI build and CF handoff without contacting Cloud Foundry.
# Run from the repository root: bash tests/test_deploy_ui_build.sh

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT
mkdir -p "$fixture/api" "$fixture/ui" "$fixture/bin"
cp "$repo_root/deploy.sh" "$repo_root/manifest.yaml" "$fixture/"
cat > "$fixture/api/.env" <<'EOF'
AICORE_AUTH_URL=https://example.invalid/auth
AICORE_CLIENT_ID=test
AICORE_CLIENT_SECRET=test
AICORE_BASE_URL=https://example.invalid/ai
AICORE_RESOURCE_GROUP=test
HANA_ADDRESS=example.invalid
HANA_PORT=443
HANA_USER=test
HANA_PASSWORD=test
HANA_ENCRYPT=true
EOF
cat > "$fixture/bin/npm" <<'EOF'
#!/usr/bin/env bash
if [ "${FAIL_BUILD:-}" = 1 ]; then exit 1; fi
mkdir -p dist
printf '%s|%s\n' "$VITE_API_BASE_URL" "$VITE_API_KEY" > dist/index.html
touch dist/Staticfile
EOF
cat > "$fixture/bin/cf" <<'EOF'
#!/usr/bin/env bash
set -e
printf '%s\n' "$1" >> "$CF_TEST_LOG"
if [ "$1" = push ]; then
  key=$(sed -n "s/^api_key: '\([^']*\)'$/\1/p" "$3")
  url=$(awk '$1 == "API_BASE_URL:" { print $2; exit }' manifest.yaml)
  [ -n "$key" ]
  grep -Fxq "$url|$key" ui/dist/index.html
  [ -f ui/dist/Staticfile ]
fi
EOF
chmod +x "$fixture/bin/npm" "$fixture/bin/cf"

export PATH="$fixture/bin:$PATH" CF_TEST_LOG="$fixture/cf.log"
cd "$fixture"
bash deploy.sh > "$fixture/deploy.log"
printf 'push\nbind-service\nrestart\n' > "$fixture/expected.log"
cmp "$fixture/expected.log" "$CF_TEST_LOG"

: > "$CF_TEST_LOG"
if FAIL_BUILD=1 bash deploy.sh > "$fixture/failure.log" 2>&1; then
  echo "Expected the failed local build to stop deployment" >&2
  exit 1
fi
[ ! -s "$CF_TEST_LOG" ]
echo "Local UI build deployment checks passed."
