"""Runner functions called by the Streamlit UI."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .artifacts import BenchmarkRunConfig, MethodArtifactKey, RunStore
from .diagnostics import classify_llm_failure, inspect_pdf_text_layer
from .llm_adapter import GenAIHubLLMAdapter, LLMCallResult
from .prompts import get_prompt
from .settings import AppPaths
from .summary import (
    OrchestrationSummaryAdapter,
    SummaryAdapter,
    save_summary_artifacts,
    summarize_task_counts,
)


class ExtractionAdapter(Protocol):
    def extract_quote(
        self,
        *,
        pdf_path: Path,
        model: str,
        scenario_key: str,
        route: str = "streamlit.llm_extract",
        user_id: str | None = None,
    ) -> LLMCallResult:
        ...


@dataclass(frozen=True)
class BatchTask:
    """One cacheable LLM extraction unit."""

    document_name: str
    model: str
    scenario_key: str
    prompt_version: str

    def artifact_key(self) -> MethodArtifactKey:
        return MethodArtifactKey(
            document_name=self.document_name,
            method_family="llm",
            scenario_key=self.scenario_key,
            model=self.model,
            prompt_version=self.prompt_version,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchTaskResult:
    """User-facing status for one planned task."""

    task: BatchTask
    status: str
    method_dir: str
    latency_ms: int = 0
    usage: dict[str, int] | None = None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_json_dict(),
            "status": self.status,
            "method_dir": self.method_dir,
            "latency_ms": self.latency_ms,
            "usage": self.usage or {},
            "error": self.error,
        }


@dataclass
class BatchRunResult:
    """Aggregate result returned to Streamlit."""

    run_id: str
    task_results: list[BatchTaskResult]
    started_at: str
    finished_at: str
    summary: dict[str, Any] | None = None

    @property
    def task_counts(self) -> dict[str, int]:
        return summarize_task_counts(self.task_results)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "task_counts": self.task_counts,
            "task_results": [item.to_json_dict() for item in self.task_results],
            "summary": self.summary,
        }


def build_llm_batch_plan(config: BenchmarkRunConfig) -> list[BatchTask]:
    """Create the selected documents x models x LLM scenarios cross-product."""

    tasks: list[BatchTask] = []
    for document_name in config.document_names:
        for model in config.models:
            for scenario_key in config.llm_scenarios:
                prompt = get_prompt(scenario_key)
                tasks.append(
                    BatchTask(
                        document_name=document_name,
                        model=model,
                        scenario_key=scenario_key,
                        prompt_version=prompt.version,
                    )
                )
    return tasks


def _utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _save_llm_result(
    *,
    store: RunStore,
    run_id: str,
    task: BatchTask,
    result: LLMCallResult,
) -> Path:
    diagnostic = (
        classify_llm_failure(model=task.model, error=result.error, document_name=task.document_name).to_json_dict()
        if result.outcome == "error" or result.error
        else None
    )
    saved_dir = store.save_method_artifacts(
        run_id,
        task.artifact_key(),
        raw={"text": result.raw_text, "error": result.error},
        normalized=result.normalized_json,
        metrics={
            "usage": result.usage,
            "latency_ms": result.latency_ms,
            "outcome": result.outcome,
            "provider_family": result.provider_family,
            "correlation_id": result.correlation_id,
            "diagnostic_category": diagnostic["category"] if diagnostic else None,
            "diagnostic_root_cause": diagnostic["root_cause"] if diagnostic else None,
        },
        metadata=result.to_json_dict(),
    )
    if diagnostic:
        diagnostic["pdf_text_layer"] = inspect_pdf_text_layer(store.paths.data_dir / task.document_name)
        (saved_dir / "diagnostics.json").write_text(
            json_dumps(
                {
                    "document_name": task.document_name,
                    "model": task.model,
                    "scenario_key": task.scenario_key,
                    "diagnostic": diagnostic,
                }
            ),
            encoding="utf-8",
        )
    return saved_dir


def run_llm_batch(
    *,
    run_id: str,
    config: BenchmarkRunConfig,
    paths: AppPaths | None = None,
    store: RunStore | None = None,
    adapter: ExtractionAdapter | None = None,
    summary_adapter: SummaryAdapter | None = None,
    progress_callback: Callable[[int, int, BatchTask, str], None] | None = None,
) -> BatchRunResult:
    """Run the LLM portion of a benchmark batch with cache reuse."""

    paths = paths or AppPaths.for_project()
    store = store or RunStore(paths)
    adapter = adapter or GenAIHubLLMAdapter(paths)
    tasks = build_llm_batch_plan(config)
    results: list[BatchTaskResult] = []
    started_at = _utc_now_text()

    for index, task in enumerate(tasks, start=1):
        key = task.artifact_key()
        method_dir = store.method_dir(run_id, key)
        if config.use_cached_results and not config.force_rerun and store.method_artifact_exists(run_id, key):
            status = "cached"
            result_item = BatchTaskResult(task=task, status=status, method_dir=str(method_dir))
            results.append(result_item)
            if progress_callback:
                progress_callback(index, len(tasks), task, status)
            continue

        if progress_callback:
            progress_callback(index, len(tasks), task, "running")
        result = adapter.extract_quote(
            pdf_path=paths.data_dir / task.document_name,
            model=task.model,
            scenario_key=task.scenario_key,
            route="streamlit.llm_batch",
        )
        saved_dir = _save_llm_result(store=store, run_id=run_id, task=task, result=result)
        status = "success" if result.outcome == "success" else "error"
        results.append(
            BatchTaskResult(
                task=task,
                status=status,
                method_dir=str(saved_dir),
                latency_ms=result.latency_ms,
                usage=result.usage,
                error=result.error,
            )
        )
        if progress_callback:
            progress_callback(index, len(tasks), task, status)

    finished_at = _utc_now_text()
    batch_result = BatchRunResult(
        run_id=run_id,
        task_results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    batch_payload = batch_result.to_json_dict()
    store.save_json(run_id, "batch_result.json", batch_payload)

    if config.include_llm_summary:
        chosen_summary_adapter = summary_adapter or OrchestrationSummaryAdapter(paths)
        summary = chosen_summary_adapter.generate_summary(
            config=config,
            batch_payload=batch_payload,
            model=config.summary_model,
        )
        save_summary_artifacts(store, run_id, summary)
        batch_result.summary = summary.to_json_dict()
        store.save_json(run_id, "batch_result.json", batch_result.to_json_dict())
    return batch_result


def run_single_llm_extraction(
    *,
    run_id: str,
    document_name: str,
    model: str,
    scenario_key: str,
    paths: AppPaths | None = None,
    store: RunStore | None = None,
) -> LLMCallResult:
    """Run one document/model/scenario LLM extraction and persist artifacts."""

    paths = paths or AppPaths.for_project()
    store = store or RunStore(paths)
    pdf_path = paths.data_dir / document_name
    result = GenAIHubLLMAdapter(paths).extract_quote(
        pdf_path=pdf_path,
        model=model,
        scenario_key=scenario_key,
    )
    task = BatchTask(
        document_name=document_name,
        model=model,
        scenario_key=scenario_key,
        prompt_version=result.prompt_version,
    )
    _save_llm_result(store=store, run_id=run_id, task=task, result=result)
    return result


def json_dumps(data: Any) -> str:
    import json

    return json.dumps(data, indent=2, ensure_ascii=False)
