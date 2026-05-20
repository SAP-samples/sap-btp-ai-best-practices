"""Shared path helpers for the metal composition service."""

from __future__ import annotations

import sys
from pathlib import Path


def _derive_roots(service_dir: Path) -> tuple[Path, Path]:
    """Resolve the API and project roots for local and deployed layouts."""

    api_root = service_dir.parents[2]
    project_root = api_root.parent if api_root.name == "api" else api_root
    return api_root, project_root


SERVICE_DIR = Path(__file__).resolve().parent
API_ROOT, PROJECT_ROOT = _derive_roots(SERVICE_DIR)


def ensure_repo_root_on_path() -> None:
    """Make the repository root importable for shared benchmark modules."""

    repo_root = str(PROJECT_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
