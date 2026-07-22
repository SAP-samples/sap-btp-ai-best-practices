"""Project-level configuration for reusable benchmark projects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .settings import PROJECT_ROOT
from .scenarios import slugify


@dataclass(frozen=True)
class ProjectConfig:
    """User-facing project configuration loaded from project_config.json."""

    project_key: str = "default"
    project_name: str = "Document Extraction Lab"
    project_description: str = "DocAI vs LLM benchmark cockpit."
    data_dir: str = "data"
    runs_dir: str = "runs"
    default_run_name: str = "document_extraction_benchmark"
    schema_prefix: str = "DOC_EXT"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_project_config(project_root: Path | None = None) -> ProjectConfig:
    """Load project config from project_config.json, falling back to defaults."""

    root = project_root or PROJECT_ROOT
    path = root / "project_config.json"
    if not path.exists():
        return ProjectConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ProjectConfig()
        valid_keys = set(ProjectConfig().to_json_dict())
        values = {key: value for key, value in data.items() if key in valid_keys}
        return ProjectConfig(**values)
    except Exception:
        return ProjectConfig()


def save_project_config(config: ProjectConfig, project_root: Path | None = None) -> Path:
    """Persist the active project config."""

    root = project_root or PROJECT_ROOT
    path = root / "project_config.json"
    path.write_text(json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def create_project_config(project_name: str, project_root: Path | None = None) -> ProjectConfig:
    """Create a neutral project layout under projects/<project_key>."""

    root = project_root or PROJECT_ROOT
    key = slugify(project_name or "purchase requisition extraction", max_length=36)
    project_folder = root / "projects" / key
    (project_folder / "data").mkdir(parents=True, exist_ok=True)
    (project_folder / "runs").mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        project_key=key,
        project_name=project_name or "Purchase Requisition Extraction",
        project_description="DocAI vs LLM benchmark cockpit for purchase requisition-ready extraction.",
        data_dir=f"projects/{key}/data",
        runs_dir=f"projects/{key}/runs",
        default_run_name=f"{key}_benchmark",
        schema_prefix="PR_EXT",
    )
    (project_folder / "project_config.json").write_text(
        json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_project_config(config, root)
    return config


def _pdf_files(directory: Path) -> list[Path]:
    files = {path.name.lower(): path for path in [*directory.glob("*.PDF"), *directory.glob("*.pdf")]}
    return sorted(files.values(), key=lambda item: item.name.lower())


def discover_project_configs(project_root: Path | None = None) -> list[ProjectConfig]:
    """Find active and project-folder configs."""

    root = project_root or PROJECT_ROOT
    configs: list[ProjectConfig] = []
    active = load_project_config(root)
    configs.append(active)
    root_data_dir = root / "data"
    if root_data_dir.exists() and _pdf_files(root_data_dir):
        sample_config = ProjectConfig(
            project_key="sample_documents",
            project_name="Sample Documents",
            project_description="Reusable sample workspace with the PDFs stored in the root data folder.",
            data_dir="data",
            runs_dir="runs",
            default_run_name="sample_documents_benchmark",
            schema_prefix="PR_EXT",
        )
        if sample_config.project_key not in {item.project_key for item in configs}:
            configs.append(sample_config)
    projects_dir = root / "projects"
    if projects_dir.exists():
        for path in sorted(projects_dir.glob("*/project_config.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                values = {key: value for key, value in data.items() if key in ProjectConfig().to_json_dict()}
                config = ProjectConfig(**values)
                if config.project_key not in {item.project_key for item in configs}:
                    configs.append(config)
            except Exception:
                continue
    return configs
