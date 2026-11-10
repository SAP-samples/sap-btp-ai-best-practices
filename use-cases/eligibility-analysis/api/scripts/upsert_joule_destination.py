#!/usr/bin/env python3
"""Create or update the BTP destination Joule uses to reach the A2A API (API-key header auth).

The deployed API authenticates every A2A message with the shared X-API-Key header. This
script writes an HTTP destination whose additional property ``URL.headers.X-API-Key``
carries that key, so Joule's remote agent requests include it. The key is read only
from the JOULE_DESTINATION_API_KEY environment variable, written to a mode-0600 temporary
JSON file for the BTP CLI, redacted from errors and deleted afterwards.

This destination type is the development path. Production should use
OAuth2ClientCredentials with XSUAA on the API (see docs/deployment-runbook.md).

Examples (repository root, after `btp login` and `btp target`):
    # Dry run: prints the non-secret payload, calls nothing.
    python3 api/scripts/upsert_joule_destination.py --name RECEIVABLES_AGENT \\
        --url https://<api-route> --subaccount <subaccount-id>

    # Create or update, then read back (deploy.sh runs this step).
    JOULE_DESTINATION_API_KEY=... python3 api/scripts/upsert_joule_destination.py \\
        --name RECEIVABLES_AGENT --url https://<api-route> --subaccount <subaccount-id> --apply
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

API_KEY_ENV = "JOULE_DESTINATION_API_KEY"
HEADER_PROPERTY = "URL.headers.X-API-Key"


class DestinationError(ValueError):
    """A user-correctable destination management error."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the destination name, API base URL, subaccount and apply flag."""
    parser = argparse.ArgumentParser(description="Dry-run or apply the Joule A2A destination with the BTP CLI.")
    parser.add_argument("--name", required=True, help="Destination name, equal to the Joule system alias.")
    parser.add_argument("--url", required=True, help="Deployed API base route (no /api/a2a path).")
    parser.add_argument("--subaccount", required=True, help="BTP subaccount ID.")
    parser.add_argument("--apply", action="store_true", help="Create or update; without it the script is a dry run.")
    return parser.parse_args(argv)


def build_payload(name: str, url: str, api_key: str | None) -> dict[str, str]:
    """Return the destination configuration; api_key=None builds the printable preview.

    Args:
        name: Destination name (letters, digits, underscore, dot, hyphen).
        url: HTTPS base route of the deployed API.
        api_key: Shared API key, or None for a redacted preview.

    Returns:
        Destination configuration accepted by ``btp create|update connectivity/destination``.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
        raise DestinationError("--name must contain only letters, digits, underscores, dots or hyphens")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in ("", "/") or parsed.query:
        raise DestinationError("--url must be the HTTPS API base route without a path")
    return {
        "Name": name,
        "Type": "HTTP",
        "ProxyType": "Internet",
        "URL": url.rstrip("/"),
        "Authentication": "NoAuthentication",
        HEADER_PROPERTY: api_key if api_key is not None else "[REDACTED]",
    }


def redact(text: str, secret: str | None) -> str:
    """Remove the API key from any CLI output before it is printed."""
    return text.replace(secret, "[REDACTED]") if secret else text


def run_btp(arguments: list[str], secret: str | None) -> str:
    """Run one BTP CLI command without a shell and return stdout; errors are redacted."""
    environment = {key: value for key, value in os.environ.items() if key != API_KEY_ENV}
    result = subprocess.run(["btp", *arguments], check=False, capture_output=True,
                            env=environment, text=True, timeout=60)
    if result.returncode:
        detail = redact(result.stderr.strip() or result.stdout.strip(), secret)
        raise DestinationError(f"btp {' '.join(arguments[:3])} failed: {detail or 'no error detail'}")
    return result.stdout


def destination_names(value: Any) -> set[str]:
    """Recursively collect destination names from BTP CLI JSON output."""
    names: set[str] = set()
    if isinstance(value, dict):
        candidate = value.get("Name") or value.get("name")
        if isinstance(candidate, str):
            names.add(candidate)
        for child in value.values():
            names |= destination_names(child)
    elif isinstance(value, list):
        for child in value:
            names |= {child} if isinstance(child, str) else destination_names(child)
    return names


def write_configuration(payload: dict[str, str]) -> Path:
    """Write the secret-bearing payload to a mode-0600 temporary JSON file."""
    descriptor, raw_path = tempfile.mkstemp(prefix="joule-destination-", suffix=".json")
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return Path(raw_path)


def apply_destination(name: str, subaccount: str, payload: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """Create or update the destination, then read it back.

    Returns:
        The operation performed ("create" or "update") and the non-secret readback properties.
    """
    if shutil.which("btp") is None:
        raise DestinationError("btp CLI is not installed; create the destination in the BTP cockpit instead")
    secret = payload[HEADER_PROPERTY]
    run_btp(["help", "create", "connectivity/destination"], secret)
    listing = run_btp(["--format", "json", "list", "connectivity/destination", "--subaccount", subaccount], secret)
    try:
        existing = destination_names(json.loads(listing))
    except json.JSONDecodeError as error:
        raise DestinationError("btp destination list did not return valid JSON") from error
    operation = "update" if name in existing else "create"
    config_path = write_configuration(payload)
    try:
        run_btp(["--format", "json", operation, "connectivity/destination",
                 "--configuration", str(config_path), "--subaccount", subaccount], secret)
    finally:
        config_path.unlink(missing_ok=True)
    readback = json.loads(run_btp(["--format", "json", "get", "connectivity/destination",
                                   "--name", name, "--subaccount", subaccount], secret) or "{}")
    return operation, _public_properties(readback)


def _public_properties(value: Any) -> dict[str, Any]:
    """Pick non-secret destination properties from the readback, noting whether the header exists."""
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "Name" in node and "URL" in node:
                found.update({key: node[key] for key in ("Name", "Type", "URL", "Authentication", "ProxyType") if key in node})
                found["api_key_header_present"] = HEADER_PROPERTY in node
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return found


def main(argv: list[str] | None = None) -> int:
    """Preview the destination, or apply it and print the redacted readback."""
    try:
        args = parse_args(argv)
        print(("Apply requested" if args.apply else "Dry run") + ": subaccount destination")
        print(json.dumps(build_payload(args.name, args.url, None), indent=2, sort_keys=True))
        if not args.apply:
            return 0
        api_key = os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise DestinationError(f"Set {API_KEY_ENV} to the API key of the deployed application")
        operation, readback = apply_destination(args.name, args.subaccount,
                                                build_payload(args.name, args.url, api_key))
    except (DestinationError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Destination {operation} completed. Readback: {json.dumps(readback, sort_keys=True)}")
    if not readback.get("api_key_header_present"):
        print(f"Warning: readback does not show {HEADER_PROPERTY}; check the destination in the BTP cockpit.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
