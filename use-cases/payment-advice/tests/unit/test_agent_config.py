"""Verify that the UC-02 deduction agent config and methodology skill are wired correctly.

This test loads the real agent.yaml and checks:
  - The interpretation model uses the currently available gpt-5.4 fallback.
  - SkillLoader can discover the deduction-interpretation methodology skill.

It is intentionally narrow: it exercises only the config + skill discovery path,
not LLM connectivity.  Run with::

    PYTHONPATH=api .venv/bin/python -m unittest tests.unit.test_agent_config -v
"""

import os
import sys
import unittest
from pathlib import Path

# Make ``app`` importable and resolve config paths from this file, not the working directory.
_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.deduction_agent.template_agent.config import load_config  # noqa: E402
from app.deduction_agent.template_agent.skills import SkillLoader  # noqa: E402

CONFIG = os.path.join(_API_DIR, "app", "deduction_agent", "config", "agent.yaml")
AUTHORING_CONFIG = os.path.join(_API_DIR, "app", "deduction_agent", "config", "authoring.yaml")


class AgentConfigTest(unittest.TestCase):
    """Verify the UC-02 agent configuration and skill discovery contract."""

    def test_config_loads_and_skill_discovered(self) -> None:
        """Load agent.yaml, assert model name, then confirm the methodology skill is found."""
        cfg = load_config(CONFIG)

        # The live gpt-5.6-luna deployment is regionally rate-limited, so the
        # demonstration runtime uses the same working fallback as authoring.
        self.assertEqual(cfg.model.name, "gpt-5.4")

        # SkillLoader must discover the reusable methodology skill.
        loader = SkillLoader(cfg.skills.directory, cfg.skills.max_loaded_characters)
        names = {s.name for s in loader.scan()}
        self.assertIn("deduction-interpretation", names)

    def test_authoring_config_loads_and_authoring_skill_is_discoverable(self) -> None:
        """The authoring runtime must resolve its config and agent-local skill."""

        cfg = load_config(AUTHORING_CONFIG)
        loader = SkillLoader(cfg.skills.directory, cfg.skills.max_loaded_characters)
        names = {summary.name for summary in loader.scan()}

        self.assertEqual(cfg.model.name, "gpt-5.4")
        self.assertEqual(cfg.model.provider, "openai")
        self.assertIn("deduction-skill-authoring", names)
        self.assertIn("list_rule_sources", cfg.base_prompt)


if __name__ == "__main__":
    unittest.main()
