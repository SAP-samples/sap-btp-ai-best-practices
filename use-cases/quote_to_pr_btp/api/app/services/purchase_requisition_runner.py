"""Background runner for purchase requisition benchmark sessions."""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from benchmark_lab.artifacts import BenchmarkRunConfig, RunStore
from benchmark_lab.pipeline import run_comparative_pipeline
from benchmark_lab.settings import AppPaths

_RUNNING_THREADS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()

STALE_AFTER_SECONDS = int(os.getenv("PR_RUNNER_STALE_AFTER_SECONDS", "1800"))

SHORTLIST_MODELS = ["gemini-2.5-flash"]
SHORTLIST_LLM_SCENARIOS = ["detailed_static_prompt"]
SHORTLIST_DOCAI_SCENARIOS: list[str] = []

RESEARCH_MODELS = ["gpt-5", "gpt-5-mini", "gemini-2.5-pro", "anthropic--claude-4.7-opus", "anthropic--claude-4.8-opus"]
RESEARCH_LLM_SCENARIOS = ["detailed_static_prompt", "dynamic_prompt", "dynamic_prompt_judge_loop"]
RESEARCH_DOCAI_SCENARIOS = ["default", "wide_attributes", "dynamic_attributes", "dynamic_attributes_judge_loop"]


def live_runner_available() -> bool:
    """Return whether this backend should start the benchmark runner."""

    return os.getenv("PR_LIVE_RUNNER_ENABLED", "true").lower() not in {"0", "false", "no"}


def status_path(run_dir: Path) -> Path:
    return run_dir / "runner_status.json"


def load_runner_status(run_dir: Path) -> dict[str, Any]:
    path = status_path(run_dir)
    if path.exists():
        try:
            return _with_stale_status(run_dir, json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    experiment = _read_json_if_exists(run_dir / "experiment.json") or {}
    if (run_dir / "comparison.json").exists():
        status = "completed"
        stage = "completed"
        step = 6
        message = "Completed result is available."
    else:
        status = str(experiment.get("status") or "prepared")
        stage = "waiting" if status == "prepared" else status
        step = 4 if status in {"prepared", "waiting"} else 1
        message = "Session is saved and waiting for runner."
    return {
        "run_id": run_dir.name,
        "status": status,
        "stage": stage,
        "step": step,
        "total_steps": 6,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _with_stale_status(run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    if (run_dir / "comparison.json").exists() and status.get("status") != "completed":
        completed = dict(status)
        completed.update(
            {
                "status": "completed",
                "stage": "completed",
                "step": 6,
                "message": "Completed result is available.",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return completed

    if status.get("status") not in {"running", "queued"}:
        return status

    existing = _RUNNING_THREADS.get(run_dir.name)
    if existing and existing.is_alive():
        return status

    updated_at = _parse_status_timestamp(status.get("updated_at"))
    if not updated_at:
        return status

    age_seconds = (datetime.now() - updated_at).total_seconds()
    if age_seconds <= STALE_AFTER_SECONDS:
        return status

    stale = dict(status)
    last_update = stale.get("updated_at")
    stale.update(
        {
            "status": "stale",
            "stage": "stale",
            "step": 5,
            "message": (
                f"Runner stopped updating after {int(age_seconds // 60)} minutes. "
                "The background worker is no longer active. Delete this stale session or start a new Lab calculation."
            ),
            "last_runner_update_at": last_update,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _update_manifest(run_dir, {"status": "stale", "stale_detected_at": stale["updated_at"]})
    status_path(run_dir).write_text(json.dumps(stale, indent=2, ensure_ascii=False), encoding="utf-8")
    return stale


def _parse_status_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def start_runner_for_session(run_dir: Path, *, workspace_root: Path) -> dict[str, Any]:
    """Start a background benchmark run for a prepared session."""

    if not live_runner_available():
        status = _write_status(run_dir, "waiting", "waiting", 4, "Runner is disabled by configuration.")
        return status
    if (run_dir / "comparison.json").exists():
        return _write_status(run_dir, "completed", "completed", 6, "Completed result is already available.")

    run_id = run_dir.name
    with _LOCK:
        existing = _RUNNING_THREADS.get(run_id)
        if existing and existing.is_alive():
            return load_runner_status(run_dir)
        thread = threading.Thread(target=_run_session, args=(run_dir, workspace_root), daemon=True, name=f"pr-runner-{run_id}")
        _RUNNING_THREADS[run_id] = thread
        _write_status(run_dir, "queued", "waiting", 4, "Session queued. Benchmark runner will start shortly.")
        thread.start()
    return load_runner_status(run_dir)


def _run_session(run_dir: Path, workspace_root: Path) -> None:
    run_id = run_dir.name
    try:
        manifest = _read_json_if_exists(run_dir / "experiment.json") or {}
        documents = [str(item) for item in manifest.get("documents") or []]
        if not documents:
            raise RuntimeError("No documents are configured for this Lab session.")

        mode = str(manifest.get("mode") or "shortlist")
        customer_fast = manifest.get("approach_profile") == "customer_fast_extraction"
        config = _build_config(run_id=run_id, manifest=manifest, documents=documents, mode=mode)
        _update_manifest(run_dir, {"status": "running", "started_at": datetime.now().isoformat(timespec="seconds")})
        start_message = (
            "Extracting quote fields with the recommended customer model."
            if customer_fast
            else "Runner started. Running extraction, judge scoring, cost and runtime aggregation."
        )
        _write_status(run_dir, "running", "running", 5, start_message)

        api_root = workspace_root.parent.parent.parent
        work_runs_dir = workspace_root / "runs" / "_runner_work"
        work_paths = AppPaths(
            project_root=api_root,
            data_dir=workspace_root / "data",
            runs_dir=work_runs_dir,
            docs_dir=api_root / "docs",
            docai_service_key=api_root / "dox_client" / "schemas" / "service_key.json",
            env_file=api_root / ".env",
            pricing_pdf=api_root / "3437766_E_20260625.pdf",
            ui_config_file=work_runs_dir / "ui_config.json",
        )
        store = RunStore(work_paths)

        def progress(stage: str, message: str) -> None:
            _append_log(run_dir, stage, message)
            stage_name = "running"
            step = 5
            if stage.lower() == "done":
                stage_name = "completed"
                step = 6
            _write_status(run_dir, "running", stage_name, step, f"{stage}: {message}")

        result = run_comparative_pipeline(
            config=config,
            paths=work_paths,
            store=store,
            progress_callback=progress,
            run_quality_judge=not customer_fast,
            run_business_summary=not customer_fast,
        )
        source_dir = Path(result.run_dir)
        _copy_completed_artifacts(source_dir, run_dir)
        _rewrite_run_ids(run_dir, source_run_id=result.run_id, target_run_id=run_id)
        _update_manifest(
            run_dir,
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "completed_run_id": run_id,
                "source_runner_run_id": result.run_id,
            },
        )
        completed_message = (
            "Extraction completed. Required values are ready for validation."
            if customer_fast
            else "Pipeline completed. Charts and comparison table are ready."
        )
        _write_status(run_dir, "completed", "completed", 6, completed_message)
        shutil.rmtree(source_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - persist full diagnostics for the UI.
        _append_log(run_dir, "Error", f"{type(exc).__name__}: {exc}")
        (run_dir / "runner_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _update_manifest(run_dir, {"status": "error", "error": str(exc), "failed_at": datetime.now().isoformat(timespec="seconds")})
        _write_status(run_dir, "error", "error", 4, f"Runner failed: {exc}")


def _build_config(*, run_id: str, manifest: dict[str, Any], documents: list[str], mode: str) -> BenchmarkRunConfig:
    include_docai = bool(manifest.get("include_docai", True))
    include_llm = bool(manifest.get("include_llm", True))
    selected_models = [str(item) for item in manifest.get("selected_llm_models") or []]
    selected_llm_scenarios = [str(item) for item in manifest.get("selected_llm_scenarios") or []]
    selected_docai_scenarios = [str(item) for item in manifest.get("selected_docai_scenarios") or []]
    if selected_models or selected_llm_scenarios or selected_docai_scenarios:
        models = selected_models if include_llm else []
        llm_scenarios = selected_llm_scenarios if include_llm else []
        docai_scenarios = selected_docai_scenarios if include_docai else []
    elif mode == "research":
        models = RESEARCH_MODELS if include_llm else []
        llm_scenarios = RESEARCH_LLM_SCENARIOS if include_llm else []
        docai_scenarios = RESEARCH_DOCAI_SCENARIOS if include_docai else []
    else:
        models = SHORTLIST_MODELS if include_llm else []
        llm_scenarios = SHORTLIST_LLM_SCENARIOS if include_llm else []
        docai_scenarios = SHORTLIST_DOCAI_SCENARIOS if include_docai else []
    return BenchmarkRunConfig(
        run_name=str(manifest.get("experiment_name") or run_id),
        document_names=documents,
        docai_scenarios=docai_scenarios,
        llm_scenarios=llm_scenarios,
        models=models,
        judge_model=str(manifest.get("judge_model") or "gpt-5"),
        max_judge_iterations=2,
        use_cached_results=True,
        force_rerun=False,
        include_llm_summary=False,
        summary_model=str(manifest.get("summary_model") or "gpt-5"),
        notes=f"Started from customer UI runner. Profile: {manifest.get('approach_profile') or 'default'}.",
    )


def _copy_completed_artifacts(source_dir: Path, target_dir: Path) -> None:
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _rewrite_run_ids(run_dir: Path, *, source_run_id: str, target_run_id: str) -> None:
    for relative in ["comparison.json", "pipeline_result.json", "run_summary.json", "summary.json", "batch_result.json", "judge_result.json", "document_report.json"]:
        path = run_dir / relative
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _replace_run_id(data, source_run_id, target_run_id)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            continue


def _replace_run_id(value: Any, source_run_id: str, target_run_id: str) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "run_id" and item == source_run_id:
                value[key] = target_run_id
            else:
                _replace_run_id(item, source_run_id, target_run_id)
    elif isinstance(value, list):
        for item in value:
            _replace_run_id(item, source_run_id, target_run_id)


def _write_status(run_dir: Path, status: str, stage: str, step: int, message: str) -> dict[str, Any]:
    payload = {
        "run_id": run_dir.name,
        "status": status,
        "stage": stage,
        "step": step,
        "total_steps": 6,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "logs": _read_json_if_exists(run_dir / "runner_log.json") or [],
    }
    status_path(run_dir).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _append_log(run_dir: Path, stage: str, message: str) -> None:
    logs = _read_json_if_exists(run_dir / "runner_log.json") or []
    logs.append({"at": datetime.now().isoformat(timespec="seconds"), "stage": stage, "message": message})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runner_log.json").write_text(json.dumps(logs[-200:], indent=2, ensure_ascii=False), encoding="utf-8")


def _update_manifest(run_dir: Path, updates: dict[str, Any]) -> None:
    path = run_dir / "experiment.json"
    data = _read_json_if_exists(path) or {"run_id": run_dir.name}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
