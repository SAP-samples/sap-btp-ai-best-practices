#!/usr/bin/env python3
"""Embed and patch an OpenCode A2A template into a host FastAPI app package."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List


DIAGNOSTIC_BLOCK = """

def _truncate(text: str, limit: int = 2000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated {len(text) - limit} chars)"


async def _run_command(
    cmd: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _truncate(stdout.decode(errors="replace") if stdout else ""),
            "stderr": _truncate(stderr.decode(errors="replace") if stderr else ""),
        }
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"Timeout after {timeout} seconds"}
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"Command not found: {exc}"}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


@app.get("/diagnostic")
async def diagnostic(run: bool = False, run_timeout: float = 30.0) -> Dict[str, Any]:
    \"\"\"Diagnostic endpoint to verify OpenCode availability/configuration.\"\"\"
    opencode_command = os.getenv("OPENCODE_COMMAND", "opencode").strip()
    command_parts = shlex.split(opencode_command) if opencode_command else []
    command_bin = command_parts[0] if command_parts else None

    model_env = os.getenv("OPENCODE_MODEL")
    model = model_env or "sap-ai-core/gpt-5"

    env = {**os.environ, "OPENCODE_NONINTERACTIVE": "1"}
    opencode_dir = _BASE_DIR / ".opencode"
    node_bin = _API_ROOT / "node_modules" / ".bin" / "opencode"

    result: Dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "app_dir": str(_BASE_DIR),
        "api_root": str(_API_ROOT),
        "opencode_command": opencode_command,
        "opencode_command_parts": command_parts,
        "opencode_command_available": bool(command_bin and shutil.which(command_bin)),
        "node_bin_exists": node_bin.exists(),
        "opencode_dir_exists": opencode_dir.exists(),
        "opencode_dir_contents": [p.name for p in opencode_dir.iterdir()] if opencode_dir.exists() else [],
        "model": model,
        "model_source": "env" if model_env else "default",
        "agent": os.getenv("OPENCODE_AGENT", "").strip(),
        "timeout_seconds": os.getenv("OPENCODE_TIMEOUT", "180"),
        "aicore_env_vars_present": {
            "AICORE_AUTH_URL": bool(os.getenv("AICORE_AUTH_URL")),
            "AICORE_CLIENT_ID": bool(os.getenv("AICORE_CLIENT_ID")),
            "AICORE_CLIENT_SECRET": bool(os.getenv("AICORE_CLIENT_SECRET")),
            "AICORE_BASE_URL": bool(os.getenv("AICORE_BASE_URL")),
            "AICORE_RESOURCE_GROUP": bool(os.getenv("AICORE_RESOURCE_GROUP")),
            "AICORE_SERVICE_KEY": bool(os.getenv("AICORE_SERVICE_KEY")),
        },
    }

    if command_parts:
        result["version"] = await _run_command(
            [*command_parts, "--version"], cwd=_BASE_DIR, env=env, timeout=10.0
        )
        result["help"] = await _run_command(
            [*command_parts, "--help"], cwd=_BASE_DIR, env=env, timeout=10.0
        )

    if run and command_parts:
        result["run"] = await _run_command(
            [*command_parts, "run", "--format", "json", "-m", model, "Say OK"],
            cwd=_BASE_DIR,
            env=env,
            timeout=run_timeout,
        )

    return result
"""


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        die(f"Failed to read {path}: {exc}")
        raise


def write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        die(f"Failed to write {path}: {exc}")


def copy_template(source_template: Path, embedded_dir: Path, overwrite: bool) -> None:
    if not source_template.exists():
        die(f"Source template does not exist: {source_template}")
    if not source_template.is_dir():
        die(f"Source template is not a directory: {source_template}")

    if embedded_dir.exists():
        if not overwrite:
            die(
                f"Target embedded directory already exists: {embedded_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(embedded_dir)

    shutil.copytree(source_template, embedded_dir)

    sessions_file = embedded_dir / "data" / "sessions.json"
    if sessions_file.exists():
        sessions_file.unlink()


def patch_env_utils(env_utils_path: Path) -> bool:
    text = read_text(env_utils_path)

    if "API_ROOT = PROJECT_ROOT.parent.parent" in text and "env_paths = [" in text:
        return False

    old_block = """PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_dotenv() -> None:
    \"\"\"Load dotenv files from the project root if present.\"\"\"
    env_path = PROJECT_ROOT / ".env"
    env_example_path = PROJECT_ROOT / ".env.example"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    if env_example_path.exists():
        load_dotenv(dotenv_path=env_example_path, override=False)
"""
    new_block = """PROJECT_ROOT = Path(__file__).resolve().parent
API_ROOT = PROJECT_ROOT.parent.parent


def load_project_dotenv() -> None:
    \"\"\"Load dotenv files, preferring api/.env for shared credentials.\"\"\"
    env_paths = [
        API_ROOT / ".env",
        PROJECT_ROOT / ".env",
        API_ROOT / ".env.example",
        PROJECT_ROOT / ".env.example",
    ]

    for path in env_paths:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
"""
    if old_block not in text:
        die(
            "Could not patch env_utils.py automatically. "
            "Unexpected baseline format; patch manually."
        )

    write_text(env_utils_path, text.replace(old_block, new_block, 1))
    return True


def ensure_import(text: str, import_line: str, after_line: str) -> str:
    if import_line in text:
        return text
    if after_line not in text:
        die(f"Could not insert import '{import_line}'. Missing anchor '{after_line}'.")
    return text.replace(after_line, f"{after_line}{import_line}\n", 1)


def patch_a2a_server(a2a_server_path: Path) -> bool:
    text = read_text(a2a_server_path)
    original = text

    text = ensure_import(text, "import asyncio", "from __future__ import annotations\n\n")
    text = ensure_import(text, "import shlex", "import os\n")
    text = ensure_import(text, "import shutil", "import shlex\n")

    if "from fastapi import Depends, FastAPI, Request" not in text:
        if "from fastapi import FastAPI, Request" not in text:
            die("Could not patch FastAPI imports in a2a_server.py.")
        text = text.replace(
            "from fastapi import FastAPI, Request",
            "from fastapi import Depends, FastAPI, Request",
            1,
        )

    if "from ..security import get_api_key" not in text:
        anchor = "    from session_manager import SessionManager\n"
        security_block = """
try:
    from ..security import get_api_key
except Exception:  # pragma: no cover
    get_api_key = None  # type: ignore
"""
        if anchor not in text:
            die("Could not patch security import block in a2a_server.py.")
        text = text.replace(anchor, f"{anchor}{security_block}\n", 1)

    if "dependencies = [Depends(get_api_key)] if get_api_key else []" not in text:
        old_app_line = 'app = FastAPI(title="A2A OpenCode Template", version="0.1.0")'
        new_app_block = (
            "dependencies = [Depends(get_api_key)] if get_api_key else []\n"
            'app = FastAPI(title="A2A OpenCode Template", version="0.1.0", dependencies=dependencies)'
        )
        if old_app_line not in text:
            die("Could not patch FastAPI app creation in a2a_server.py.")
        text = text.replace(old_app_line, new_app_block, 1)

    if "_API_ROOT = _BASE_DIR.parent.parent" not in text:
        base_anchor = '_DEFAULT_SESSIONS_FILE = _BASE_DIR / "data" / "sessions.json"\n'
        if base_anchor not in text:
            die("Could not patch _API_ROOT constant in a2a_server.py.")
        text = text.replace(
            base_anchor,
            f'{base_anchor}_API_ROOT = _BASE_DIR.parent.parent\n',
            1,
        )

    if '@app.get("/diagnostic")' not in text:
        text = text.rstrip() + "\n" + DIAGNOSTIC_BLOCK.rstrip() + "\n"

    if text != original:
        write_text(a2a_server_path, text)
        return True
    return False


def patch_agent_stdin(agent_path: Path) -> bool:
    text = read_text(agent_path)
    original = text

    if "stdin=asyncio.subprocess.DEVNULL" in text:
        return False

    replacement_anchor = "stderr=asyncio.subprocess.PIPE,\n"
    replacement = "stderr=asyncio.subprocess.PIPE,\n            stdin=asyncio.subprocess.DEVNULL,\n"
    if replacement_anchor not in text:
        die("Could not patch stdin=DEVNULL in agent.py.")

    text = text.replace(replacement_anchor, replacement, 1)
    if text != original:
        write_text(agent_path, text)
        return True
    return False


def write_router_bridge(routers_dir: Path, embedded_name: str, router_module: str) -> Path:
    routers_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = routers_dir / f"{router_module}.py"
    bridge_content = (
        '"""Bridge module that exposes the embedded A2A OpenCode app."""\n\n'
        f"from ..{embedded_name}.a2a_server import app as app\n\n"
        '__all__ = ["app"]\n'
    )
    write_text(bridge_path, bridge_content)
    return bridge_path


def patch_main_py(main_py_path: Path, router_module: str, mount_path: str) -> bool:
    text = read_text(main_py_path)
    original = text

    import_re = re.compile(r"^from \.routers import ([^\n]+)$", re.MULTILINE)
    match = import_re.search(text)
    if not match:
        die(f"Could not find '.routers' import line in {main_py_path}.")

    modules = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if router_module not in modules:
        modules.append(router_module)
        replacement = "from .routers import " + ", ".join(modules)
        text = text[: match.start()] + replacement + text[match.end() :]

    mount_line = f'app.mount("{mount_path}", {router_module}.app)'
    if mount_line not in text:
        router_calls = list(
            re.finditer(r"^\s*app\.include_router\([^\n]+\)\s*$", text, re.MULTILINE)
        )
        if router_calls:
            insert_at = router_calls[-1].end()
            text = text[:insert_at] + f"\n{mount_line}" + text[insert_at:]
        else:
            route_marker = '@app.get("/")'
            marker_index = text.find(route_marker)
            if marker_index >= 0:
                text = text[:marker_index] + f"{mount_line}\n\n" + text[marker_index:]
            else:
                text = text.rstrip() + f"\n\n{mount_line}\n"

    if text != original:
        write_text(main_py_path, text)
        return True
    return False


def patch_api_env_example(api_env_example: Path, mount_path: str, embedded_name: str) -> bool:
    if not api_env_example.exists():
        return False

    text = read_text(api_env_example)
    original = text

    base_url_value = f'A2A_BASE_URL="http://127.0.0.1:8000{mount_path}"'
    sessions_value = f'A2A_SESSIONS_FILE="app/{embedded_name}/data/sessions.json"'

    if re.search(r"^A2A_BASE_URL=.*$", text, flags=re.MULTILINE):
        text = re.sub(r'^A2A_BASE_URL=.*$', base_url_value, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n{base_url_value}\n"

    if re.search(r"^A2A_SESSIONS_FILE=.*$", text, flags=re.MULTILINE):
        text = re.sub(r'^A2A_SESSIONS_FILE=.*$', sessions_value, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n{sessions_value}\n"

    if text != original:
        write_text(api_env_example, text)
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an OpenCode A2A template into a host FastAPI app package and "
            "apply the integration patches used in mounted mode."
        )
    )
    parser.add_argument("--source-template", required=True, type=Path)
    parser.add_argument("--target-app-dir", required=True, type=Path)
    parser.add_argument("--embedded-name", default="a2a_opencode_template")
    parser.add_argument("--mount-path", default="/api/opencode-agent")
    parser.add_argument("--router-module", default="opencode_agent")
    parser.add_argument("--api-env-example", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_app_dir = args.target_app_dir.resolve()
    if not target_app_dir.exists():
        die(f"Target app directory does not exist: {target_app_dir}")

    embedded_dir = target_app_dir / args.embedded_name
    copy_template(args.source_template.resolve(), embedded_dir, args.overwrite)

    changed: List[Path] = []

    if patch_env_utils(embedded_dir / "env_utils.py"):
        changed.append(embedded_dir / "env_utils.py")
    if patch_a2a_server(embedded_dir / "a2a_server.py"):
        changed.append(embedded_dir / "a2a_server.py")
    if patch_agent_stdin(embedded_dir / "agent.py"):
        changed.append(embedded_dir / "agent.py")

    bridge_path = write_router_bridge(target_app_dir / "routers", args.embedded_name, args.router_module)
    changed.append(bridge_path)

    if patch_main_py(target_app_dir / "main.py", args.router_module, args.mount_path):
        changed.append(target_app_dir / "main.py")

    if args.api_env_example and patch_api_env_example(
        args.api_env_example.resolve(), args.mount_path, args.embedded_name
    ):
        changed.append(args.api_env_example.resolve())

    print("Embedded template path:", embedded_dir)
    print("Updated files:")
    for path in changed:
        print(f"- {path}")

    print("\nNext checks:")
    print(
        "1) Verify OPENCODE_AGENT matches .opencode/agent/<slug>.md in the embedded template."
    )
    print(
        "2) Verify A2A_BASE_URL includes the mount path and A2A_SESSIONS_FILE is writable."
    )
    print("3) Run verify_integration.py from this skill.")


if __name__ == "__main__":
    main()
