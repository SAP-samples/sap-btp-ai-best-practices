#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.tools.sap/sap-btp-ai-services-coe/Agent-skills-catalog.git"
CLONE_DIR="$(mktemp -d)"
CATALOG_DIR="$CLONE_DIR/skills"
SKILLS_DIR="$HOME/.claude/skills"

cleanup() {
  rm -rf "$CLONE_DIR"
}
trap cleanup EXIT

echo "Cloning skills catalog..."
git clone --depth 1 "$REPO_URL" "$CLONE_DIR"

mkdir -p "$SKILLS_DIR"

added=0
skipped=0

for skill_path in "$CATALOG_DIR"/*/; do
  skill_name="$(basename "$skill_path")"
  dest="$SKILLS_DIR/$skill_name"

  if [[ -d "$dest" ]]; then
    echo "skip  $skill_name (already exists)"
    ((skipped++))
  else
    cp -r "$skill_path" "$dest"
    echo "added $skill_name"
    ((added++))
  fi
done

echo ""
echo "Done. Added: $added  Skipped: $skipped"
