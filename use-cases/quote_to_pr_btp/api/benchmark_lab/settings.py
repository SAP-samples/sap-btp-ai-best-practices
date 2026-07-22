"""Application settings and persisted UI configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppPaths:
    """Filesystem locations used by the prototype."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    runs_dir: Path = PROJECT_ROOT / "runs"
    docs_dir: Path = PROJECT_ROOT / "docs"
    docai_service_key: Path = PROJECT_ROOT / "dox_client" / "schemas" / "service_key.json"
    env_file: Path = PROJECT_ROOT / ".env"
    pricing_pdf: Path = PROJECT_ROOT / "3437766_E_20260625.pdf"
    ui_config_file: Path = PROJECT_ROOT / "runs" / "ui_config.json"

    @classmethod
    def for_project(cls, project_root: Path | None = None) -> "AppPaths":
        """Build paths using project_config.json when present."""

        root = project_root or PROJECT_ROOT
        try:
            from .project_config import load_project_config

            config = load_project_config(root)
            return cls(
                project_root=root,
                data_dir=root / config.data_dir,
                runs_dir=root / config.runs_dir,
                docs_dir=root / "docs",
                docai_service_key=root / "dox_client" / "schemas" / "service_key.json",
                env_file=root / ".env",
                pricing_pdf=root / "3437766_E_20260625.pdf",
                ui_config_file=root / config.runs_dir / "ui_config.json",
            )
        except Exception:
            return cls(project_root=root)


@dataclass
class UiConfig:
    """User-facing benchmark configuration persisted between UI sessions."""

    default_judge_model: str = "gpt-5"
    selected_models: list[str] = field(default_factory=lambda: ["gpt-5", "gemini-2.5-pro"])
    selected_llm_scenarios: list[str] = field(
        default_factory=lambda: ["detailed_static_prompt", "dynamic_prompt", "dynamic_prompt_judge_loop"]
    )
    selected_docai_scenarios: list[str] = field(
        default_factory=lambda: ["default", "wide_attributes", "dynamic_attributes", "dynamic_attributes_judge_loop"]
    )
    max_judge_iterations: int = 2
    use_cached_results: bool = True
    force_rerun: bool = False
    require_schema_approval: bool = False
    include_llm_summary: bool = True
    summary_model: str = "gpt-5"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "UiConfig":
        base = cls()
        valid_keys = set(base.to_json_dict())
        values = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**values)


def load_ui_config(paths: AppPaths | None = None) -> UiConfig:
    """Load persisted UI config, falling back to defaults."""

    paths = paths or AppPaths()
    if not paths.ui_config_file.exists():
        return UiConfig()
    try:
        data = json.loads(paths.ui_config_file.read_text(encoding="utf-8"))
        return UiConfig.from_json_dict(data if isinstance(data, dict) else {})
    except Exception:
        return UiConfig()


def save_ui_config(config: UiConfig, paths: AppPaths | None = None) -> Path:
    """Persist UI config and return the saved path."""

    paths = paths or AppPaths()
    paths.ui_config_file.parent.mkdir(parents=True, exist_ok=True)
    paths.ui_config_file.write_text(
        json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths.ui_config_file
