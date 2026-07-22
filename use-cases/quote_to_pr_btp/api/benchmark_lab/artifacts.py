"""Run artifact and cache storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .scenarios import slugify, utc_timestamp_for_run
from .settings import AppPaths


@dataclass
class BenchmarkRunConfig:
    """Configuration that defines a benchmark run."""

    run_name: str
    document_names: list[str]
    docai_scenarios: list[str]
    llm_scenarios: list[str]
    models: list[str]
    judge_model: str = "gpt-5"
    max_judge_iterations: int = 2
    use_cached_results: bool = True
    force_rerun: bool = False
    include_llm_summary: bool = True
    summary_model: str = "gpt-5"
    notes: str = ""

    def normalized(self) -> dict[str, Any]:
        """Return a stable dict for hashing and persistence."""

        data = asdict(self)
        for key in ("document_names", "docai_scenarios", "llm_scenarios", "models"):
            data[key] = sorted(data[key])
        return data

    def config_hash(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class MethodArtifactKey:
    """Smallest cacheable unit for benchmark method output."""

    document_name: str
    method_family: str
    scenario_key: str
    model: str | None = None
    prompt_version: str | None = None
    schema_name: str | None = None

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def folder_name(self) -> str:
        return f"m_{self.cache_key()}"


class RunStore:
    """Filesystem-backed run store."""

    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or AppPaths()
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)

    def make_run_id(self, config: BenchmarkRunConfig) -> str:
        name = slugify(config.run_name, max_length=24)
        return f"{utc_timestamp_for_run()}_{name}_{config.config_hash()}"

    def run_dir(self, run_id: str) -> Path:
        return self.paths.runs_dir / run_id

    def create_run(self, config: BenchmarkRunConfig) -> Path:
        run_id = self.make_run_id(config)
        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=False)
        self.save_json(run_id, "run_config.json", config.normalized())
        self.save_json(
            run_id,
            "run_summary.json",
            {
                "run_id": run_id,
                "status": "created",
                "config_hash": config.config_hash(),
                "documents": config.document_names,
                "models": config.models,
            },
        )
        return path

    def save_json(self, run_id: str, relative_path: str, data: Any) -> Path:
        path = self.run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_json(self, run_id: str, relative_path: str) -> Any:
        path = self.run_dir(run_id) / relative_path
        return json.loads(path.read_text(encoding="utf-8"))

    def method_dir(self, run_id: str, key: MethodArtifactKey) -> Path:
        return self.run_dir(run_id) / "methods" / key.folder_name()

    def method_artifact_exists(self, run_id: str, key: MethodArtifactKey) -> bool:
        return (self.method_dir(run_id, key) / "normalized.json").exists()

    def save_method_artifacts(
        self,
        run_id: str,
        key: MethodArtifactKey,
        *,
        raw: Any | None = None,
        normalized: Any | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        folder = self.method_dir(run_id, key)
        folder.mkdir(parents=True, exist_ok=True)
        self._write_json(folder / "cache_key.json", {"cache_key": key.cache_key(), **asdict(key)})
        if raw is not None:
            self._write_json(folder / "raw.json", raw)
        if normalized is not None:
            self._write_json(folder / "normalized.json", normalized)
        if metrics is not None:
            self._write_json(folder / "metrics.json", metrics)
        if metadata is not None:
            self._write_json(folder / "metadata.json", metadata)
        return folder

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in sorted(self.paths.runs_dir.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            summary_path = path / "run_summary.json"
            config_path = path / "run_config.json"
            summary: dict[str, Any] = {"run_id": path.name, "status": "unknown"}
            if summary_path.exists():
                try:
                    summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            if config_path.exists():
                try:
                    summary["config"] = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            runs.append(summary)
        return runs

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
