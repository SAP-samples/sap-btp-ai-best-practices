"""Structural tests for the focused Joule/CF lifecycle skill."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
PLUGIN_DIR = SKILL_DIR.parents[1]
REPO_DIR = PLUGIN_DIR.parents[1]


class SkillContractTests(unittest.TestCase):
    """Keep the skill focused, secure by default, and independently documented."""

    def test_skill_delegates_agent_creation_and_preserves_both_modes(self) -> None:
        """The workflow should start from an existing agent and retain mode choice."""

        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("create-langgraph-react-agent", skill)
        self.assertIn("direct", skill)
        self.assertIn("interpretability", skill)
        self.assertIn("OAuth2ClientCredentials", skill)
        self.assertIn("development", skill)
        self.assertNotIn("Phase 3A: Bootstrap From Scratch", skill)

    def test_legacy_full_project_bootstrap_is_removed(self) -> None:
        """The skill must not own application, dependency, or deployment bootstrapping."""

        removed_paths = [
            "scripts/bootstrap_agent.py",
            "templates/app",
            "templates/.env.example",
            "templates/.gitignore",
            "templates/README.md",
            "templates/deploy.sh",
            "templates/pip.conf",
            "templates/requirements.txt",
            "templates/runtime.txt",
        ]
        for relative_path in removed_paths:
            self.assertFalse((SKILL_DIR / relative_path).exists(), relative_path)

    def test_lifecycle_references_cover_separate_operational_domains(self) -> None:
        """Focused references should document A2A, CF, BTP, and Joule lifecycles."""

        reference_expectations = {
            "joule-a2a-contract.md": ["message/send", "60", "contextId", "taskId"],
            "cloud-foundry.md": ["cf push", "cf logs", "VCAP_SERVICES", "401"],
            "btp-destination.md": [
                "OAuth2ClientCredentials",
                "NoAuthentication",
                "connectivity/destination",
                "base route",
            ],
            "joule-cli.md": [
                "@sap/joule-studio-cli",
                "joule lint",
                "joule compile",
                "joule deploy",
                "joule delete",
            ],
        }
        for filename, phrases in reference_expectations.items():
            path = SKILL_DIR / "references" / filename
            self.assertTrue(path.is_file(), filename)
            content = path.read_text(encoding="utf-8")
            for phrase in phrases:
                self.assertIn(phrase, content, f"{filename}: {phrase}")

    def test_feature_is_documented_and_release_is_not_rolled_back(self) -> None:
        """The plugin release and repository docs should retain this redesign."""

        feature_doc = REPO_DIR / "docs" / "joule-cf-lifecycle-skill.md"
        self.assertTrue(feature_doc.is_file())
        feature_text = feature_doc.read_text(encoding="utf-8")
        self.assertIn("Existing agent", feature_text)
        self.assertIn("Inputs", feature_text)
        self.assertIn("Outputs", feature_text)
        self.assertIn("Tests", feature_text)

        manifest_paths = [
            PLUGIN_DIR / "plugin.json",
            PLUGIN_DIR / ".claude-plugin/plugin.json",
            PLUGIN_DIR / ".codex-plugin/plugin.json",
        ]
        for path in manifest_paths:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            version = tuple(int(part) for part in manifest["version"].split("."))
            self.assertGreaterEqual(version, (0, 2, 1), str(path))
        changelog = (PLUGIN_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## 0.2.1 - 2026-09-14", changelog)


if __name__ == "__main__":
    unittest.main()
