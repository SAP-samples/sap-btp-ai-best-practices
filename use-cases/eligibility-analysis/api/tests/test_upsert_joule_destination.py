"""Tests for the Joule destination upsert script (BTP CLI calls are mocked).

Run from the repository root:
    cd api && PYTHONPATH=. ../.venv/bin/python -m unittest tests.test_upsert_joule_destination
"""
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upsert_joule_destination.py"
spec = importlib.util.spec_from_file_location("upsert_joule_destination", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

ARGS = ["--name", "RECEIVABLES_AGENT", "--url", "https://api.example.invalid", "--subaccount", "sub-1"]


class FakeBtp:
    """Record BTP CLI calls, capture the written configuration and serve canned JSON."""

    def __init__(self, existing):
        self.existing = existing
        self.calls = []
        self.config = None
        self.config_path = None

    def __call__(self, command, **kwargs):
        """Emulate subprocess.run for the btp commands used by the script."""
        self.calls.append(command[1:])
        assert "JOULE_DESTINATION_API_KEY" not in kwargs["env"]
        if "--configuration" in command:
            self.config_path = Path(command[command.index("--configuration") + 1])
            self.config = json.loads(self.config_path.read_text())
            self.config_mode = self.config_path.stat().st_mode & 0o777
        stdout = ""
        if "list" in command:
            stdout = json.dumps([{"Name": name} for name in self.existing])
        if "get" in command:
            stdout = json.dumps({"destinationConfiguration": self.config})
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


class UpsertJouleDestinationTests(unittest.TestCase):
    """Dry run calls nothing; apply creates or updates with a redacted, cleaned-up secret."""

    def run_main(self, args, fake=None, api_key="secret-key"):
        """Run main() with patched environment and subprocess, returning (code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        environment = {"JOULE_DESTINATION_API_KEY": api_key} if api_key else {}
        with patch.dict("os.environ", environment, clear=False), \
                patch.object(module.subprocess, "run", fake or FakeBtp([])), \
                patch.object(module.shutil, "which", return_value="/usr/bin/btp"), \
                redirect_stdout(out), redirect_stderr(err):
            code = module.main(args)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_prints_redacted_payload_without_calling_btp(self):
        """Without --apply only the preview is printed."""
        fake = FakeBtp([])
        code, out, _ = self.run_main(ARGS, fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.calls, [])
        self.assertIn('"URL.headers.X-API-Key": "[REDACTED]"', out)
        self.assertNotIn("secret-key", out)

    def test_apply_creates_then_updates_and_removes_temp_file(self):
        """A missing destination is created; an existing one is updated; the secret never prints."""
        for existing, operation in (([], "create"), (["RECEIVABLES_AGENT"], "update")):
            fake = FakeBtp(existing)
            code, out, err = self.run_main(ARGS + ["--apply"], fake)
            self.assertEqual(code, 0, err)
            self.assertIn(operation, [call[2] for call in fake.calls if len(call) > 2])
            self.assertEqual(fake.config["URL.headers.X-API-Key"], "secret-key")
            self.assertEqual(fake.config["Authentication"], "NoAuthentication")
            self.assertEqual(fake.config_mode, 0o600)
            self.assertFalse(fake.config_path.exists())
            self.assertIn(f"Destination {operation} completed", out)
            self.assertIn('"api_key_header_present": true', out)
            self.assertNotIn("secret-key", out + err)

    def test_apply_requires_key_and_valid_url(self):
        """Missing key or a URL with a path is rejected before any BTP call."""
        code, _, err = self.run_main(ARGS + ["--apply"], api_key=None)
        self.assertEqual(code, 2)
        self.assertIn("JOULE_DESTINATION_API_KEY", err)
        bad = ["--name", "X", "--url", "https://api.example.invalid/api/a2a", "--subaccount", "s"]
        code, _, err = self.run_main(bad)
        self.assertEqual(code, 2)
        self.assertIn("without a path", err)


if __name__ == "__main__":
    unittest.main()
