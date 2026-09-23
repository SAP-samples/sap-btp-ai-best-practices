"""Tests for rendering Joule and Cloud Foundry integration assets."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "prepare_joule_integration.py"


class PrepareJouleIntegrationTests(unittest.TestCase):
    """Verify that the renderer augments, but never bootstraps, an agent."""

    def _make_agent(self, root: Path) -> Path:
        """Create the minimal marker used to represent an existing agent."""

        target = root / "existing-agent"
        config_dir = target / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "agent.yaml").write_text(
            "name: existing-agent\n", encoding="utf-8"
        )
        return target

    def _run_renderer(
        self,
        target: Path,
        *,
        mode: str = "direct",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run the renderer with representative, non-secret arguments."""

        self.assertTrue(SCRIPT.is_file(), "the focused integration renderer must exist")
        command = [
            sys.executable,
            str(SCRIPT),
            "--target",
            str(target),
            "--agent-slug",
            "inventory-agent",
            "--agent-description",
            "Answers inventory availability questions",
            "--public-url",
            "https://inventory-agent.example.test",
            "--start-command",
            "python -m app.main",
            "--auth-service",
            "agent-xsuaa",
            "--mode",
            mode,
        ]
        return subprocess.run(command, check=check, capture_output=True, text=True)

    def test_renders_secure_cf_and_direct_joule_assets(self) -> None:
        """Direct mode should relay the remote answer without embedding secrets."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = self._make_agent(Path(temp_dir))

            result = self._run_renderer(target)

            expected_files = [
                ".cfignore",
                "manifest.yaml",
                "da.sapdas.yaml",
                "joule/a2a/capability.sapdas.yaml",
                "joule/a2a/capability_context.yaml",
                "joule/a2a/functions/inventory_agent_a2a.yaml",
                "joule/a2a/scenarios/agent/inventory_agent.yaml",
            ]
            for relative_path in expected_files:
                self.assertTrue((target / relative_path).is_file(), relative_path)

            self.assertFalse((target / "app").exists())
            self.assertFalse((target / "requirements.txt").exists())

            manifest = (target / "manifest.yaml").read_text(encoding="utf-8")
            self.assertIn("A2A_PUBLIC_URL", manifest)
            self.assertIn("agent-xsuaa", manifest)
            self.assertNotIn("CLIENT_SECRET", manifest)
            self.assertNotIn("API_KEY", manifest)

            cfignore = (target / ".cfignore").read_text(encoding="utf-8")
            self.assertIn(".env", cfignore.splitlines())
            self.assertIn("data/", cfignore.splitlines())

            capability = (target / "joule/a2a/capability.sapdas.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn('schema_version: "3.28.0"', capability)
            self.assertIn("namespace: joule.ext", capability)

            function = (
                target / "joule/a2a/functions/inventory_agent_a2a.yaml"
            ).read_text(encoding="utf-8")
            scenario = (
                target / "joule/a2a/scenarios/agent/inventory_agent.yaml"
            ).read_text(encoding="utf-8")
            self.assertEqual(2, function.count("type: message"))
            self.assertIn('"contextId"', function)
            self.assertIn('"taskId"', function)
            self.assertIn("agent_context_id.isEmpty()", function)
            self.assertIn("? null", function)
            self.assertIn("_agent_response.body.artifacts != null ?", function)
            self.assertNotIn("response_context:", scenario)
            self.assertIn("Rendered Joule integration", result.stdout)

    def test_interpretability_mode_delegates_final_response_to_joule(self) -> None:
        """Interpretability mode should expose compact context and avoid duplicate replies."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = self._make_agent(Path(temp_dir))

            self._run_renderer(target, mode="interpretability")

            function = (
                target / "joule/a2a/functions/inventory_agent_a2a.yaml"
            ).read_text(encoding="utf-8")
            scenario = (
                target / "joule/a2a/scenarios/agent/inventory_agent.yaml"
            ).read_text(encoding="utf-8")
            self.assertEqual(1, function.count("type: message"))
            self.assertIn("input-required", function)
            self.assertIn("response_context:", scenario)
            self.assertIn("agent_result", scenario)

    def test_rejects_non_agent_target_without_partial_writes(self) -> None:
        """An empty directory must be delegated to an agent-creation skill."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "not-an-agent"
            target.mkdir()

            result = self._run_renderer(target, check=False)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("existing LangGraph/A2A agent", result.stderr)
            self.assertFalse((target / "manifest.yaml").exists())
            self.assertFalse((target / "joule").exists())

    def test_collision_fails_before_any_file_is_written(self) -> None:
        """Existing integration files should stop the complete render transaction."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = self._make_agent(Path(temp_dir))
            descriptor = target / "da.sapdas.yaml"
            descriptor.write_text("sentinel: keep\n", encoding="utf-8")

            result = self._run_renderer(target, check=False)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("already exist", result.stderr)
            self.assertEqual("sentinel: keep\n", descriptor.read_text(encoding="utf-8"))
            self.assertFalse((target / "manifest.yaml").exists())
            self.assertFalse((target / "joule").exists())


if __name__ == "__main__":
    unittest.main()
