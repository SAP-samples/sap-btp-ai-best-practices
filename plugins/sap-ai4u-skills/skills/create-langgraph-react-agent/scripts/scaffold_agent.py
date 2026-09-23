"""Copy the bundled portable LangGraph agent into an empty target directory.

Examples:
    python scripts/scaffold_agent.py /tmp/customer-agent
    python scripts/scaffold_agent.py ../new-use-case/agent
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Return the guarded scaffold command parser."""

    parser = argparse.ArgumentParser(
        description="Copy the bundled LangGraph agent into an empty target"
    )
    parser.add_argument("target", type=Path, help="New or empty target directory")
    return parser


def scaffold(target: Path) -> Path:
    """Copy the sanitized template and return the resolved target.

    Args:
        target: New directory or an existing empty directory.

    Returns:
        The resolved populated target path.
    """

    source = (Path(__file__).resolve().parent.parent / "assets" / "template").resolve()
    destination = target.expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Target cannot be inside the bundled template")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Target is not empty: {destination}")
    if destination.exists():
        for item in source.iterdir():
            output = destination / item.name
            if item.is_dir():
                shutil.copytree(item, output)
            else:
                shutil.copy2(item, output)
    else:
        shutil.copytree(source, destination)
    return destination


def main() -> int:
    """Scaffold the project and print the conventional pip next steps."""

    destination = scaffold(build_parser().parse_args().target)
    print(f"Created agent template at {destination}")
    print(f"Next: cd {destination}")
    print("python3.11 -m venv .venv  # use python3 when 3.11 is unavailable")
    print(".venv/bin/python -m pip install -r requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
