#!/usr/bin/env python3
"""Create or update a Joule A2A destination with the native SAP BTP CLI.

Examples:
    export JOULE_AGENT_CLIENT_ID="..."
    export JOULE_AGENT_CLIENT_SECRET="..."
    export JOULE_AGENT_TOKEN_URL="https://tenant.authentication.example/oauth/token"

    # Safe preview; this does not call BTP.
    python scripts/manage_destination.py \
      --name inventory-agent-a2a \
      --url https://inventory-agent.example.com \
      --subaccount 00000000-0000-0000-0000-000000000000

    # Create or update after reviewing the preview.
    python scripts/manage_destination.py \
      --name inventory-agent-a2a \
      --url https://inventory-agent.example.com \
      --subaccount 00000000-0000-0000-0000-000000000000 \
      --apply

    # Development only; never use this for production ingress.
    python scripts/manage_destination.py \
      --name inventory-agent-a2a-dev \
      --url https://inventory-agent-dev.example.com \
      --subaccount 00000000-0000-0000-0000-000000000000 \
      --development-no-auth

OAuth2 Client Credentials is the default. Credential values are accepted only
through environment variables, written to a mode-0600 temporary JSON file for
the BTP CLI, redacted from errors, and deleted after the command completes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CLIENT_ID_ENV = "JOULE_AGENT_CLIENT_ID"
CLIENT_SECRET_ENV = "JOULE_AGENT_CLIENT_SECRET"
TOKEN_URL_ENV = "JOULE_AGENT_TOKEN_URL"


class DestinationError(ValueError):
    """Represent a safe, user-correctable destination management error."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse destination management command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Dry-run or apply a Joule A2A destination with the BTP CLI."
    )
    parser.add_argument(
        "--name", required=True, help="Destination and system alias name."
    )
    parser.add_argument("--url", required=True, help="Deployed A2A API base route.")
    parser.add_argument("--subaccount", required=True, help="BTP subaccount ID.")
    parser.add_argument(
        "--development-no-auth",
        action="store_true",
        help="Use NoAuthentication for an explicitly development-only destination.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the upsert; omission keeps the command as a local dry run.",
    )
    return parser.parse_args(argv)


def require_single_line(label: str, value: str) -> str:
    """Return a trimmed value or reject empty and multiline input."""

    stripped = value.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise DestinationError(f"{label} must be a non-empty single-line value")
    return stripped


def validate_base_url(label: str, value: str, *, allow_path: bool = False) -> str:
    """Validate an HTTPS URL and optionally require a path-free base route."""

    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise DestinationError(f"{label} must be an absolute HTTPS URL")
    if not allow_path and parsed.path not in ("", "/"):
        raise DestinationError(
            f"{label} must be the deployed API base route without /a2a or agent-card paths"
        )
    return value.rstrip("/")


def build_payload(args: argparse.Namespace, *, include_secrets: bool) -> dict[str, str]:
    """Build a validated Destination configuration payload.

    Args:
        args: Parsed command-line values.
        include_secrets: Whether OAuth credential environment variables are required.

    Returns:
        A complete configuration for apply, or a sanitized configuration for preview.
    """

    name = require_single_line("--name", args.name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise DestinationError(
            "--name must contain only letters, digits, underscores, dots, or hyphens"
        )
    payload = {
        "Name": name,
        "Type": "HTTP",
        "ProxyType": "Internet",
        "URL": validate_base_url("--url", args.url),
        "Authentication": (
            "NoAuthentication"
            if args.development_no_auth
            else "OAuth2ClientCredentials"
        ),
    }
    if args.development_no_auth or not include_secrets:
        return payload

    values = {
        "clientId": os.environ.get(CLIENT_ID_ENV, ""),
        "clientSecret": os.environ.get(CLIENT_SECRET_ENV, ""),
        "tokenServiceURL": os.environ.get(TOKEN_URL_ENV, ""),
    }
    missing = [
        environment_name
        for property_name, environment_name in (
            ("clientId", CLIENT_ID_ENV),
            ("clientSecret", CLIENT_SECRET_ENV),
            ("tokenServiceURL", TOKEN_URL_ENV),
        )
        if not values[property_name]
    ]
    if missing:
        raise DestinationError(
            "OAuth2 apply requires environment variables: " + ", ".join(missing)
        )
    values["tokenServiceURL"] = validate_base_url(
        TOKEN_URL_ENV, values["tokenServiceURL"], allow_path=True
    )
    payload.update(values)
    return payload


def redact(text: str, payload: dict[str, str]) -> str:
    """Replace any known credential value in command output with a marker."""

    redacted = text
    for key in ("clientId", "clientSecret", "tokenServiceURL"):
        value = payload.get(key)
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def run_btp(arguments: list[str], payload: dict[str, str]) -> str:
    """Execute one non-shell BTP CLI command and return its standard output."""

    command_environment = os.environ.copy()
    for credential_name in (CLIENT_ID_ENV, CLIENT_SECRET_ENV, TOKEN_URL_ENV):
        command_environment.pop(credential_name, None)
    result = subprocess.run(
        ["btp", *arguments],
        check=False,
        capture_output=True,
        env=command_environment,
        text=True,
        timeout=60,
    )
    if result.returncode:
        detail = redact(result.stderr.strip() or result.stdout.strip(), payload)
        raise DestinationError(
            f"btp {' '.join(arguments[:2])} failed: {detail or 'no error detail'}"
        )
    return result.stdout


def destination_names(value: Any) -> set[str]:
    """Recursively collect destination names from BTP CLI JSON output."""

    names: set[str] = set()
    if isinstance(value, dict):
        candidate = value.get("Name") or value.get("name")
        if isinstance(candidate, str):
            names.add(candidate)
        for child in value.values():
            names.update(destination_names(child))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, str):
                names.add(child)
            else:
                names.update(destination_names(child))
    return names


def detect_operation(subaccount: str, name: str, payload: dict[str, str]) -> str:
    """Return create or update after listing destinations at subaccount scope."""

    output = run_btp(
        [
            "--format",
            "json",
            "list",
            "connectivity/destination",
            "--subaccount",
            subaccount,
        ],
        payload,
    )
    try:
        existing = destination_names(json.loads(output))
    except json.JSONDecodeError as error:
        raise DestinationError(
            "btp destination list did not return valid JSON"
        ) from error
    return "update" if name in existing else "create"


def write_secret_configuration(payload: dict[str, str]) -> Path:
    """Write the Destination payload to a mode-0600 temporary JSON file."""

    descriptor, raw_path = tempfile.mkstemp(prefix="joule-destination-", suffix=".json")
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file)
        config_file.write("\n")
    return Path(raw_path)


def apply_destination(args: argparse.Namespace, payload: dict[str, str]) -> str:
    """Capability-check, upsert, and read back a destination with the BTP CLI."""

    if shutil.which("btp") is None:
        raise DestinationError(
            "btp CLI is not installed; install it or create the destination in BTP cockpit"
        )
    subaccount = require_single_line("--subaccount", args.subaccount)
    run_btp(["help", "create", "connectivity/destination"], payload)
    operation = detect_operation(subaccount, payload["Name"], payload)
    config_path = write_secret_configuration(payload)
    try:
        run_btp(
            [
                "--format",
                "json",
                operation,
                "connectivity/destination",
                "--configuration",
                str(config_path),
                "--subaccount",
                subaccount,
            ],
            payload,
        )
    finally:
        config_path.unlink(missing_ok=True)
    run_btp(
        [
            "--format",
            "json",
            "get",
            "connectivity/destination",
            "--name",
            payload["Name"],
            "--subaccount",
            subaccount,
        ],
        payload,
    )
    return operation


def print_preview(args: argparse.Namespace, payload: dict[str, str]) -> None:
    """Print only non-secret destination properties and mutation intent."""

    label = "Apply requested" if args.apply else "Dry run"
    print(f"{label}: create or update this subaccount destination")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.development_no_auth:
        print(
            "DEVELOPMENT ONLY: NoAuthentication must not be used for production ingress."
        )


def main(argv: list[str] | None = None) -> int:
    """Preview a destination or apply its native BTP CLI upsert."""

    try:
        args = parse_args(argv)
        require_single_line("--subaccount", args.subaccount)
        preview_payload = build_payload(args, include_secrets=False)
        print_preview(args, preview_payload)
        if not args.apply:
            return 0
        secret_payload = build_payload(args, include_secrets=True)
        operation = apply_destination(args, secret_payload)
    except (DestinationError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(
        f"Destination {operation} completed and readback succeeded: {preview_payload['Name']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
