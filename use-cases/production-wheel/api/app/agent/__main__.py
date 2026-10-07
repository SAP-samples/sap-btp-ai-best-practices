"""Run the portable agent from the command line.

Examples (run from the api/ directory so the top-level packages resolve):
    python -m app.agent ask "List published datasets and inspect the one I select" --context-id demo
    python -m app.agent chat --context-id demo
    python -m app.agent skills-list
    python -m app.agent mcp-check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.workspace.dependencies import get_service

from .config import load_config
from .mcp import MCPManager
from .models import Attachment
from .runtime import AgentRuntime
from .skills import SkillLoader
from .tools.workspace_tools import workspace_tools

# Packaged config path, resolved relative to this file so the CLI works from any
# working directory (for example run as `python -m app.agent` from api/).
_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config" / "agent.yaml")


def _step_printer(event: dict[str, Any]) -> None:
    """Print one concise agent progress line to stderr, keeping stdout for the answer.

    Surfaces the ReAct cadence live: which tool the model calls and a short preview
    of each tool result (for example a persisted dataset, draft or run ID), so a
    long solve is no longer a silent gap.
    """

    kind = event.get("type")
    if kind == "skills":
        names = event.get("names") or []
        if names:
            print(f"- skills: {', '.join(names)}", file=sys.stderr, flush=True)
    elif kind == "tool_call":
        print(f"-> {event.get('name')}", file=sys.stderr, flush=True)
    elif kind == "tool_result":
        name = event.get("name") or "tool"
        print(f"   {name}: {event.get('preview', '')}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Return the complete standard-library CLI parser."""

    parser = argparse.ArgumentParser(description="Run the portable LangGraph agent")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Agent YAML path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run one agent request")
    ask.add_argument("prompt")
    ask.add_argument("--context-id", default="default")
    ask.add_argument("--attach", action="append", default=[])
    ask.add_argument("--schema", type=Path, help="Optional JSON Schema file")

    chat = subparsers.add_parser("chat", help="Start an interactive conversation")
    chat.add_argument("--context-id", default="default")

    subparsers.add_parser("skills-list", help="List dynamically discovered skills")
    subparsers.add_parser("mcp-check", help="Connect and list configured MCP tools")
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Execute one parsed CLI command."""

    if args.command == "skills-list":
        config = load_config(args.config)
        loader = SkillLoader(
            config.skills.directory, config.skills.max_loaded_characters
        )
        for skill in loader.scan():
            print(f"{skill.name}: {skill.description}")
        return

    if args.command == "mcp-check":
        config = load_config(args.config)
        manager = await MCPManager.create(config.mcp)
        try:
            for name, status in manager.diagnostics.items():
                print(f"{name}: {status}")
            for tool in manager.tools:
                print(f"tool: {tool.name}")
        finally:
            await manager.aclose()
        return

    service = get_service()
    async with await AgentRuntime.create(
        args.config,
        extra_tools=workspace_tools(service, args.context_id),
        model_name=service.get_ai_model_settings()["model"],
    ) as runtime:
        if args.command == "ask":
            schema: dict[str, Any] | None = None
            if args.schema:
                schema = json.loads(args.schema.read_text(encoding="utf-8"))
            result = await runtime.ainvoke(
                args.prompt,
                args.context_id,
                [Attachment(path=Path(path)) for path in args.attach],
                schema,
                on_event=_step_printer,
            )
            print(result.output_text)
            if result.output_parsed is not None:
                value = result.output_parsed
                if hasattr(value, "model_dump"):
                    value = value.model_dump(mode="json")
                print(json.dumps(value, indent=2, ensure_ascii=False))
            return

        session_history: list[dict[str, str]] = []
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                return
            result = await runtime.ainvoke(
                prompt,
                args.context_id,
                on_event=_step_printer,
                session_history=session_history,
            )
            print(f"agent> {result.output_text}")
            session_history.extend(
                (
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result.output_text},
                )
            )
            session_history = session_history[-40:]


def main() -> int:
    """Parse arguments, run the async command, and return a shell exit code."""

    try:
        asyncio.run(_run(build_parser().parse_args()))
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
