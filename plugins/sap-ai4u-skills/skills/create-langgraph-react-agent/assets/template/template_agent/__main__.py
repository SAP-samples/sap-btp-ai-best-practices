"""Run the portable agent from the command line.

Examples:
    .venv/bin/python -m template_agent ask "What can you do?" --context-id demo
    .venv/bin/python -m template_agent ask "Describe this" --attach image.png
    .venv/bin/python -m template_agent chat --context-id demo
    .venv/bin/python -m template_agent skills-list
    .venv/bin/python -m template_agent mcp-check
    .venv/bin/python -m template_agent clear-context demo
    .venv/bin/python -m template_agent serve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .mcp import MCPManager
from .memory import create_conversation_store
from .models import Attachment
from .runtime import AgentRuntime
from .skills import SkillLoader


def build_parser() -> argparse.ArgumentParser:
    """Return the complete standard-library CLI parser."""

    parser = argparse.ArgumentParser(description="Run the portable LangGraph agent")
    parser.add_argument("--config", default="config/agent.yaml", help="Agent YAML path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run one agent request")
    ask.add_argument("prompt")
    ask.add_argument("--context-id", default="default")
    ask.add_argument("--attach", action="append", default=[])
    ask.add_argument("--schema", type=Path, help="Optional JSON Schema file")

    chat = subparsers.add_parser("chat", help="Start an interactive conversation")
    chat.add_argument("--context-id", default="default")

    clear = subparsers.add_parser("clear-context", help="Delete one persisted conversation")
    clear.add_argument("context_id")

    subparsers.add_parser("skills-list", help="List dynamically discovered skills")
    subparsers.add_parser("mcp-check", help="Connect and list configured MCP tools")
    subparsers.add_parser("serve", help="Serve the agent over A2A JSON-RPC")
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Execute one parsed CLI command."""

    if args.command == "serve":
        import uvicorn

        from .a2a_server import create_a2a_app

        config = load_config(args.config)
        server = uvicorn.Server(
            uvicorn.Config(
                create_a2a_app(args.config),
                host=config.a2a.host,
                port=config.a2a.port,
            )
        )
        await server.serve()
        return

    if args.command == "skills-list":
        config = load_config(args.config)
        loader = SkillLoader(config.skills.directory, config.skills.max_loaded_characters)
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

    if args.command == "clear-context":
        config = load_config(args.config)
        store = create_conversation_store(config.memory)
        try:
            await asyncio.to_thread(store.ensure)
            await asyncio.to_thread(store.clear, args.context_id)
        finally:
            store.close()
        print(f"Cleared context {args.context_id!r}")
        return

    async with await AgentRuntime.create(args.config) as runtime:
        if args.command == "ask":
            schema: dict[str, Any] | None = None
            if args.schema:
                schema = json.loads(args.schema.read_text(encoding="utf-8"))
            result = await runtime.ainvoke(
                args.prompt,
                args.context_id,
                [Attachment(path=Path(path)) for path in args.attach],
                schema,
            )
            print(result.output_text)
            if result.output_parsed is not None:
                value = result.output_parsed
                if hasattr(value, "model_dump"):
                    value = value.model_dump(mode="json")
                print(json.dumps(value, indent=2, ensure_ascii=False))
            return

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
            result = await runtime.ainvoke(prompt, args.context_id)
            print(f"agent> {result.output_text}")


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
