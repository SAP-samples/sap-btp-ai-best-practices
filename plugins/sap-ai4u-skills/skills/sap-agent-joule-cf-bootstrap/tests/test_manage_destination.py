"""Tests for safe BTP Destination lifecycle management."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "manage_destination.py"
SECRET = "never-print-this-client-secret"


class ManageDestinationTests(unittest.TestCase):
    """Verify secure defaults, dry runs, and redacted native CLI execution."""

    def _run(
        self,
        *extra_args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the destination manager with representative common arguments."""

        self.assertTrue(SCRIPT.is_file(), "the BTP Destination manager must exist")
        command = [
            sys.executable,
            str(SCRIPT),
            "--name",
            "inventory-agent-a2a",
            "--url",
            "https://inventory-agent.example.test",
            "--subaccount",
            "subaccount-id",
            *extra_args,
        ]
        process_env = os.environ.copy()
        process_env.update(
            {
                "JOULE_AGENT_CLIENT_ID": "destination-client-id",
                "JOULE_AGENT_CLIENT_SECRET": SECRET,
                "JOULE_AGENT_TOKEN_URL": "https://auth.example.test/oauth/token",
            }
        )
        if env:
            process_env.update(env)
        return subprocess.run(
            command, check=False, capture_output=True, text=True, env=process_env
        )

    def test_oauth2_is_the_redacted_dry_run_default(self) -> None:
        """A default invocation should plan OAuth2 without exposing credentials."""

        result = self._run()

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Dry run", result.stdout)
        self.assertIn("OAuth2ClientCredentials", result.stdout)
        self.assertNotIn(SECRET, result.stdout + result.stderr)
        self.assertNotIn("destination-client-id", result.stdout + result.stderr)

    def test_no_auth_requires_an_explicit_development_switch(self) -> None:
        """NoAuthentication should only be available through a clearly named flag."""

        result = self._run("--development-no-auth")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("DEVELOPMENT ONLY", result.stdout)
        self.assertIn("NoAuthentication", result.stdout)

    def test_destination_url_must_be_the_base_route(self) -> None:
        """The destination must not append an A2A or agent-card path."""

        result = self._run(
            "--url",
            "https://inventory-agent.example.test/a2a/",
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("base route", result.stderr)

    def test_apply_uses_native_btp_upsert_without_leaking_secrets(self) -> None:
        """Apply should capability-check, update, verify, and remove its secret file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_btp = temp_path / "btp"
            log_path = temp_path / "btp.log"
            fake_btp.write_text(
                "#!/bin/sh\n"
                'printf \'%s\\n\' "$*" >> "$BTP_LOG"\n'
                'if [ -n "${JOULE_AGENT_CLIENT_SECRET:-}" ]; then\n'
                "  printf '%s\\n' 'secret-env-present' >> \"$BTP_LOG\"\n"
                "fi\n"
                'case " $* " in\n'
                "  *' list connectivity/destination '*)\n"
                "    printf '%s\\n' '{\"value\":[\"inventory-agent-a2a\"]}' ;;\n"
                "  *) printf '%s\\n' '{}' ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_btp.chmod(0o700)
            result = self._run(
                "--apply",
                env={
                    "PATH": f"{temp_path}{os.pathsep}{os.environ.get('PATH', '')}",
                    "BTP_LOG": str(log_path),
                },
            )

            self.assertEqual(0, result.returncode, result.stderr)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("help create connectivity/destination", log)
            self.assertIn("list connectivity/destination", log)
            self.assertIn("update connectivity/destination", log)
            self.assertIn("get connectivity/destination", log)
            self.assertNotIn("--verbose", log)
            self.assertNotIn("secret-env-present", log)
            self.assertNotIn(SECRET, result.stdout + result.stderr + log)

            update_line = next(
                line
                for line in log.splitlines()
                if "update connectivity/destination" in line
            )
            config_path = Path(update_line.split("--configuration ", 1)[1].split()[0])
            self.assertFalse(config_path.exists())


if __name__ == "__main__":
    unittest.main()
