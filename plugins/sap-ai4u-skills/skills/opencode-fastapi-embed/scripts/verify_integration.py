#!/usr/bin/env python3
"""Verify embedded OpenCode template integration inside a host FastAPI app."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[FAIL] Could not read {path}: {exc}")
        raise SystemExit(1)


def check(condition: bool, ok_msg: str, fail_msg: str, failures: List[str]) -> None:
    if condition:
        print(f"[PASS] {ok_msg}")
    else:
        print(f"[FAIL] {fail_msg}")
        failures.append(fail_msg)


def check_embedded_files(
    target_app_dir: Path,
    embedded_name: str,
    mount_path: str,
    router_module: str,
    failures: List[str],
) -> None:
    embedded_dir = target_app_dir / embedded_name
    a2a_server_path = embedded_dir / "a2a_server.py"
    env_utils_path = embedded_dir / "env_utils.py"
    agent_path = embedded_dir / "agent.py"
    main_path = target_app_dir / "main.py"
    router_bridge_path = target_app_dir / "routers" / f"{router_module}.py"

    required_paths = [
        embedded_dir,
        a2a_server_path,
        env_utils_path,
        agent_path,
        main_path,
        router_bridge_path,
    ]
    for path in required_paths:
        check(path.exists(), f"Exists: {path}", f"Missing required path: {path}", failures)

    if failures:
        return

    a2a_server_text = read_text(a2a_server_path)
    env_utils_text = read_text(env_utils_path)
    agent_text = read_text(agent_path)
    main_text = read_text(main_path)
    router_bridge_text = read_text(router_bridge_path)

    check(
        "from fastapi import Depends, FastAPI, Request" in a2a_server_text,
        "Embedded a2a_server imports Depends",
        "Embedded a2a_server is missing Depends import",
        failures,
    )
    check(
        "from ..security import get_api_key" in a2a_server_text,
        "Embedded a2a_server imports host security dependency",
        "Embedded a2a_server is missing '..security.get_api_key' import",
        failures,
    )
    check(
        "dependencies = [Depends(get_api_key)] if get_api_key else []" in a2a_server_text,
        "Embedded FastAPI app has shared dependency wiring",
        "Embedded FastAPI app is missing shared dependency wiring",
        failures,
    )
    check(
        '@app.get("/diagnostic")' in a2a_server_text,
        "Diagnostic endpoint exists in embedded a2a_server",
        "Diagnostic endpoint missing in embedded a2a_server",
        failures,
    )

    check(
        "API_ROOT = PROJECT_ROOT.parent.parent" in env_utils_text,
        "env_utils computes API_ROOT",
        "env_utils missing API_ROOT for host-level dotenv loading",
        failures,
    )
    check(
        all(
            token in env_utils_text
            for token in (
                'API_ROOT / ".env"',
                'PROJECT_ROOT / ".env"',
                'API_ROOT / ".env.example"',
                'PROJECT_ROOT / ".env.example"',
            )
        ),
        "env_utils includes host and embedded dotenv search paths",
        "env_utils missing expected dotenv path order for mounted integration",
        failures,
    )

    check(
        "stdin=asyncio.subprocess.DEVNULL" in agent_text,
        "Embedded agent uses stdin=asyncio.subprocess.DEVNULL",
        "Embedded agent missing stdin=asyncio.subprocess.DEVNULL",
        failures,
    )

    expected_bridge_import = f"from ..{embedded_name}.a2a_server import app as app"
    check(
        expected_bridge_import in router_bridge_text,
        "Router bridge imports embedded a2a_server app",
        "Router bridge missing embedded app import",
        failures,
    )

    mount_line = f'app.mount("{mount_path}", {router_module}.app)'
    check(
        mount_line in main_text,
        "Host main.py mounts embedded app",
        f"Host main.py missing mount line: {mount_line}",
        failures,
    )
    import_line_match = re.search(r"^from \.routers import ([^\n]+)$", main_text, flags=re.MULTILINE)
    has_router_import = bool(import_line_match and router_module in import_line_match.group(1))
    check(
        has_router_import,
        "Host main.py imports router bridge module",
        f"Host main.py missing '{router_module}' in .routers import list",
        failures,
    )


def check_local_templates(templates_root: Path, failures: List[str]) -> None:
    template_dirs = sorted(
        path
        for path in templates_root.glob("a2a_opencode_*")
        if path.is_dir() and (path / "agent.py").exists()
    )
    if not template_dirs:
        print(f"[WARN] No local template folders found under: {templates_root}")
        return

    for template_dir in template_dirs:
        agent_path = template_dir / "agent.py"
        agent_text = read_text(agent_path)
        check(
            "stdin=asyncio.subprocess.DEVNULL" in agent_text,
            f"{template_dir.name}/agent.py keeps stdin=DEVNULL",
            f"{template_dir.name}/agent.py missing stdin=asyncio.subprocess.DEVNULL",
            failures,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify mounted OpenCode template integration in a host FastAPI app."
    )
    parser.add_argument("--target-app-dir", required=True, type=Path)
    parser.add_argument("--embedded-name", default="a2a_opencode_template")
    parser.add_argument("--mount-path", default="/api/opencode-agent")
    parser.add_argument("--router-module", default="opencode_agent")
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=None,
        help="Optional path containing local a2a_opencode_* templates to enforce stdin=DEVNULL checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: List[str] = []

    target_app_dir = args.target_app_dir.resolve()
    if not target_app_dir.exists():
        print(f"[FAIL] Target app directory does not exist: {target_app_dir}")
        raise SystemExit(1)

    print(f"Verifying target app: {target_app_dir}")
    check_embedded_files(
        target_app_dir=target_app_dir,
        embedded_name=args.embedded_name,
        mount_path=args.mount_path,
        router_module=args.router_module,
        failures=failures,
    )

    if args.templates_root:
        print(f"\nVerifying template stdin handling under: {args.templates_root.resolve()}")
        check_local_templates(args.templates_root.resolve(), failures)

    if failures:
        print(f"\nSummary: {len(failures)} failure(s)")
        raise SystemExit(1)

    print("\nSummary: all checks passed")


if __name__ == "__main__":
    main()
