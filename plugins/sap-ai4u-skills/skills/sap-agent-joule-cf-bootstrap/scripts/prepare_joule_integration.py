#!/usr/bin/env python3
"""Render Joule and Cloud Foundry assets around an existing A2A agent.

Examples:
    python scripts/prepare_joule_integration.py \
      --target /path/to/agent \
      --agent-slug inventory-agent \
      --agent-description "Answers inventory questions" \
      --public-url https://inventory-agent.example.com \
      --start-command "python -m app.main" \
      --auth-service inventory-xsuaa

    python scripts/prepare_joule_integration.py \
      --target /path/to/agent \
      --agent-slug inventory-agent \
      --agent-description "Answers inventory questions" \
      --public-url https://inventory-agent-dev.example.com \
      --start-command "python -m app.main" \
      --mode interpretability \
      --development-no-auth

The script never creates application code, installs dependencies, pushes an app,
or deploys a Joule assistant. Existing output files are treated as conflicts so
that an integration render cannot silently overwrite project-owned content.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SKILL_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = SKILL_DIR / "templates"
AGENT_MARKERS = (Path("config/agent.yaml"), Path("config/agent.yml"))
JOULENAME_MAX_LENGTH = 30


class IntegrationError(ValueError):
    """Represent a user-correctable integration rendering error."""


@dataclass(frozen=True)
class RenderConfig:
    """Hold validated inputs used to render integration assets."""

    target: Path
    agent_slug: str
    description: str
    public_url: str
    start_command: str
    mode: str
    auth_service: str | None
    extra_services: tuple[str, ...]
    destination_alias: str
    assistant_name: str
    cf_app_name: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return the raw argparse namespace."""

    parser = argparse.ArgumentParser(
        description="Render Joule/Cloud Foundry integration assets for an existing agent."
    )
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--agent-slug", required=True)
    parser.add_argument("--agent-description", required=True)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--start-command", required=True)
    parser.add_argument(
        "--mode", choices=("direct", "interpretability"), default="direct"
    )
    parser.add_argument("--destination-alias")
    parser.add_argument("--assistant-name")
    parser.add_argument("--cf-app-name")
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help="Additional existing Cloud Foundry service instance; repeat as needed.",
    )
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument(
        "--auth-service",
        help="Existing XSUAA or IAS service instance used by ingress middleware.",
    )
    auth_group.add_argument(
        "--development-no-auth",
        action="store_true",
        help="Explicitly render an unauthenticated development-only manifest.",
    )
    return parser.parse_args(argv)


def normalize_joule_name(value: str, suffix: str = "") -> str:
    """Convert a slug to a valid Joule identifier no longer than 30 characters."""

    normalized = re.sub(r"[^a-z0-9_]", "_", value.lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    available = JOULENAME_MAX_LENGTH - len(suffix)
    if not normalized or available < 1:
        raise IntegrationError("could not derive a valid Joule identifier")
    return f"{normalized[:available].rstrip('_')}{suffix}"


def validate_base_url(value: str) -> str:
    """Validate and normalize an HTTPS deployment base route."""

    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise IntegrationError("--public-url must be an absolute HTTPS URL")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise IntegrationError(
            "--public-url must be the deployed API base route without /a2a or other paths"
        )
    return value.rstrip("/")


def require_single_line(label: str, value: str) -> str:
    """Reject empty or multiline values before inserting them into YAML."""

    stripped = value.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise IntegrationError(f"{label} must be a non-empty single-line value")
    return stripped


def build_config(args: argparse.Namespace) -> RenderConfig:
    """Validate CLI input and construct an immutable rendering configuration."""

    target = args.target.expanduser().resolve()
    if not target.is_dir():
        raise IntegrationError(f"target directory does not exist: {target}")
    if not any((target / marker).is_file() for marker in AGENT_MARKERS):
        raise IntegrationError(
            "target is not an existing LangGraph/A2A agent: expected config/agent.yaml; "
            "create the agent with the dedicated agent-creation skill first"
        )

    slug = require_single_line("--agent-slug", args.agent_slug)
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", slug):
        raise IntegrationError(
            "--agent-slug must use lowercase letters, digits, and internal hyphens"
        )

    assistant_name = args.assistant_name or f"{slug.replace('-', '_')}_assistant"
    if (
        not 3 <= len(assistant_name) <= 50
        or not re.fullmatch(r"[A-Za-z0-9_]+", assistant_name)
        or "__" in assistant_name
    ):
        raise IntegrationError(
            "--assistant-name must be 3-50 letters, digits, or underscores without '__'"
        )

    services = tuple(
        require_single_line("--service", service) for service in args.service
    )
    auth_service = (
        require_single_line("--auth-service", args.auth_service)
        if args.auth_service
        else None
    )
    return RenderConfig(
        target=target,
        agent_slug=slug,
        description=require_single_line("--agent-description", args.agent_description),
        public_url=validate_base_url(args.public_url),
        start_command=require_single_line("--start-command", args.start_command),
        mode=args.mode,
        auth_service=auth_service,
        extra_services=services,
        destination_alias=require_single_line(
            "--destination-alias", args.destination_alias or f"{slug}-a2a"
        ),
        assistant_name=assistant_name,
        cf_app_name=require_single_line("--cf-app-name", args.cf_app_name or slug),
    )


def yaml_string(value: str) -> str:
    """Return a JSON-quoted string, which is also a valid YAML scalar."""

    return json.dumps(value, ensure_ascii=False)


def response_context_block(mode: str) -> str:
    """Return the scenario response context for interpretability mode only."""

    if mode == "direct":
        return ""
    return (
        "response_context:\n"
        "  - description: Remote agent response\n"
        "    value: $target_result.agent_result\n\n"
    )


def completed_message_block(mode: str) -> str:
    """Return direct answer relay actions while avoiding duplicate Joule replies."""

    if mode == "interpretability":
        return ""
    return (
        "  - condition: _agent_response.body != null && "
        "_agent_response.body.artifacts != null && "
        "_agent_response.body.artifacts[0].parts != null\n"
        "    actions:\n"
        "      - type: message\n"
        "        message:\n"
        "          type: text\n"
        "          content: <? _agent_response.body.artifacts[0].parts[0].text ?>\n"
        "          markdown: true\n"
    )


def service_block(config: RenderConfig) -> str:
    """Render unique Cloud Foundry service bindings for the manifest."""

    service_names = ([config.auth_service] if config.auth_service else []) + list(
        config.extra_services
    )
    unique_names = list(dict.fromkeys(service_names))
    if not unique_names:
        return ""
    rendered = "\n".join(f"      - {yaml_string(name)}" for name in unique_names)
    return f"    services:\n{rendered}\n"


def render_template(relative_path: str, replacements: dict[str, str]) -> str:
    """Load one bundled template and replace every declared token."""

    template_path = TEMPLATE_DIR / relative_path
    content = template_path.read_text(encoding="utf-8")
    for token, value in replacements.items():
        content = content.replace(f"{{{{{token}}}}}", value)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", content)
    if unresolved:
        raise IntegrationError(
            f"template {relative_path} has unresolved tokens: {', '.join(unresolved)}"
        )
    return content


def build_outputs(config: RenderConfig) -> dict[Path, str]:
    """Render all output content in memory before touching the target project."""

    base_name = normalize_joule_name(config.agent_slug)
    function_name = normalize_joule_name(config.agent_slug, "_a2a")
    replacements = {
        "AGENT_DESCRIPTION": yaml_string(config.description),
        "ASSISTANT_NAME": config.assistant_name,
        "CAPABILITY_DISPLAY_NAME": yaml_string(
            f"{config.agent_slug.replace('-', ' ').title()} Agent"
        ),
        "CAPABILITY_NAME": base_name,
        "CF_APP_NAME": yaml_string(config.cf_app_name),
        "COMPLETED_MESSAGE_BLOCK": completed_message_block(config.mode),
        "DESTINATION_ALIAS": yaml_string(config.destination_alias),
        "FUNCTION_NAME": function_name,
        "PUBLIC_URL": yaml_string(config.public_url),
        "RESPONSE_CONTEXT_BLOCK": response_context_block(config.mode),
        "SCENARIO_NAME": base_name,
        "SERVICES_BLOCK": service_block(config),
        "START_COMMAND": yaml_string(config.start_command),
    }
    return {
        Path(".cfignore"): render_template(".cfignore", replacements),
        Path("manifest.yaml"): render_template("manifest.yaml", replacements),
        Path("da.sapdas.yaml"): render_template("da.sapdas.yaml", replacements),
        Path("joule/a2a/capability.sapdas.yaml"): render_template(
            "joule/a2a/capability.sapdas.yaml", replacements
        ),
        Path("joule/a2a/capability_context.yaml"): render_template(
            "joule/a2a/capability_context.yaml", replacements
        ),
        Path(f"joule/a2a/functions/{function_name}.yaml"): render_template(
            "joule/a2a/functions/function.yaml", replacements
        ),
        Path(f"joule/a2a/scenarios/agent/{base_name}.yaml"): render_template(
            "joule/a2a/scenarios/agent/scenario.yaml", replacements
        ),
    }


def write_outputs(config: RenderConfig, outputs: dict[Path, str]) -> None:
    """Preflight collisions and then write every rendered integration file."""

    collisions = [path for path in outputs if (config.target / path).exists()]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise IntegrationError(f"integration files already exist: {rendered}")

    for relative_path, content in outputs.items():
        destination = config.target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Validate input, render the asset set, and report the created paths."""

    try:
        config = build_config(parse_args(argv))
        outputs = build_outputs(config)
        write_outputs(config, outputs)
    except (IntegrationError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Rendered Joule integration in {config.target}")
    for relative_path in outputs:
        print(f"  {relative_path}")
    if config.auth_service is None:
        print(
            "Warning: development-only unauthenticated ingress was explicitly selected."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
