"""LLM-as-a-judge scoring for extraction artifacts."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from dotenv import load_dotenv

from observability.llm_usage_logging import emit_llm_usage_event, extract_token_usage, usage_dict_from_tokens

from .artifacts import BenchmarkRunConfig, RunStore
from .json_utils import parse_json_object
from .settings import AppPaths


@dataclass
class JudgeResult:
    """Structured quality result for one method output."""

    overall_score: float
    field_scores: dict[str, float]
    missing_fields: list[str]
    hallucination_risks: list[str]
    confidence: float
    recommendation: str
    judge_model: str
    outcome: str = "success"
    raw_text: str = ""
    usage: dict[str, int] | None = None
    latency_ms: int = 0
    error: str | None = None
    correlation_id: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage"] = self.usage or {}
        return data


class JudgeAdapter(Protocol):
    def judge_extraction(
        self,
        *,
        normalized: dict[str, Any],
        metadata: dict[str, Any],
        model: str,
    ) -> JudgeResult:
        ...


@dataclass
class JudgeTaskResult:
    method_dir: str
    status: str
    judge_model: str
    overall_score: float | None = None
    confidence: float | None = None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeRunResult:
    run_id: str
    task_results: list[JudgeTaskResult]
    started_at: str
    finished_at: str

    @property
    def task_counts(self) -> dict[str, int]:
        counts = {"total": len(self.task_results), "success": 0, "error": 0, "cached": 0, "skipped": 0}
        for item in self.task_results:
            if item.status in counts:
                counts[item.status] += 1
        return counts

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "task_counts": self.task_counts,
            "task_results": [item.to_json_dict() for item in self.task_results],
        }


def _utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_method_dirs(store: RunStore, run_id: str) -> list[Path]:
    methods_dir = store.run_dir(run_id) / "methods"
    if not methods_dir.exists():
        return []
    return [
        path
        for path in sorted(methods_dir.iterdir())
        if path.is_dir() and (path / "normalized.json").exists()
    ]


def normalize_judge_payload(payload: dict[str, Any], judge_model: str, raw_text: str = "") -> JudgeResult:
    field_scores = payload.get("field_scores") if isinstance(payload.get("field_scores"), dict) else {}
    normalized_scores: dict[str, float] = {}
    for key, value in field_scores.items():
        try:
            normalized_scores[str(key)] = max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            normalized_scores[str(key)] = 0.0

    def score_value(key: str, fallback: float) -> float:
        try:
            return max(0.0, min(100.0, float(payload.get(key, fallback))))
        except (TypeError, ValueError):
            return fallback

    def list_value(key: str) -> list[str]:
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if value:
            return [str(value)]
        return []

    return JudgeResult(
        overall_score=score_value("overall_score", 0.0),
        field_scores=normalized_scores,
        missing_fields=list_value("missing_fields"),
        hallucination_risks=list_value("hallucination_risks"),
        confidence=score_value("confidence", 0.0),
        recommendation=str(payload.get("recommendation") or "Review extraction before business use."),
        judge_model=judge_model,
        raw_text=raw_text,
    )


class OrchestrationJudgeAdapter:
    """Live GenAI Hub judge adapter."""

    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or AppPaths.for_project()
        load_dotenv(dotenv_path=self.paths.env_file)

    def judge_extraction(
        self,
        *,
        normalized: dict[str, Any],
        metadata: dict[str, Any],
        model: str,
    ) -> JudgeResult:
        try:
            from gen_ai_hub.orchestration.models.config import OrchestrationConfig
            from gen_ai_hub.orchestration.models.llm import LLM
            from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
            from gen_ai_hub.orchestration.models.template import Template
            from gen_ai_hub.orchestration.service import OrchestrationService
        except Exception as exc:
            return JudgeResult(
                overall_score=0.0,
                field_scores={},
                missing_fields=[],
                hallucination_risks=["Judge model unavailable."],
                confidence=0.0,
                recommendation="Fix AI Core/model access and rerun judge scoring.",
                judge_model=model,
                outcome="error",
                error=f"Judge model unavailable: {exc}",
            )

        correlation_id = str(uuid.uuid4())
        started = time.perf_counter()
        response_obj: Any = None
        outcome = "success"
        raw = ""
        try:
            template = Template(
                messages=[
                    SystemMessage(
                        "You are an extraction quality judge for purchase requisition automation. "
                        "Score only the provided extraction output. Do not invent source facts."
                    ),
                    UserMessage(
                        "Evaluate this extraction for purchase requisition readiness. "
                        "Return only valid JSON with keys: overall_score, field_scores, missing_fields, "
                        "hallucination_risks, confidence, recommendation. Scores are 0-100. "
                        "field_scores must include quote_header, line_items, pr_mapping, evidence, warnings.\n\n"
                        f"METHOD METADATA:\n{metadata}\n\nEXTRACTION JSON:\n{normalized}"
                    ),
                ]
            )
            llm = LLM(name=model, version="latest")
            response_obj = OrchestrationService(config=OrchestrationConfig(template=template, llm=llm)).run()
            raw = response_obj.orchestration_result.choices[0].message.content
            judged = normalize_judge_payload(parse_json_object(raw), judge_model=model, raw_text=raw)
        except Exception as exc:
            outcome = "error"
            judged = JudgeResult(
                overall_score=0.0,
                field_scores={},
                missing_fields=[],
                hallucination_risks=["Judge generation failed."],
                confidence=0.0,
                recommendation="Review Technical details, then rerun judge scoring.",
                judge_model=model,
                outcome="error",
                raw_text=raw,
                error=str(exc),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage_tokens = extract_token_usage(getattr(response_obj, "orchestration_result", response_obj))
        judged.usage = usage_dict_from_tokens(usage_tokens)
        judged.latency_ms = latency_ms
        judged.outcome = outcome
        judged.correlation_id = correlation_id
        emit_llm_usage_event(
            route="streamlit.judge",
            method="INTERNAL",
            user_id=None,
            provider="sap-ai-core",
            model=model,
            llm_endpoint="orchestration",
            input_tokens=usage_tokens.input_tokens,
            cached_input_tokens=usage_tokens.cached_input_tokens,
            output_tokens=usage_tokens.output_tokens,
            outcome=outcome,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        return judged


def run_judge_for_methods(
    *,
    run_id: str,
    config: BenchmarkRunConfig,
    paths: AppPaths | None = None,
    store: RunStore | None = None,
    adapter: JudgeAdapter | None = None,
    progress_callback: Callable[[int, int, Path, str], None] | None = None,
) -> JudgeRunResult:
    """Score saved method artifacts and persist judge artifacts next to them."""

    paths = paths or AppPaths.for_project()
    store = store or RunStore(paths)
    adapter = adapter or OrchestrationJudgeAdapter(paths)
    method_dirs = list_method_dirs(store, run_id)
    results: list[JudgeTaskResult] = []
    started_at = _utc_now_text()

    for index, method_dir in enumerate(method_dirs, start=1):
        judge_path = method_dir / "judge.json"
        if config.use_cached_results and not config.force_rerun and judge_path.exists():
            cached = _read_json(judge_path)
            results.append(
                JudgeTaskResult(
                    method_dir=str(method_dir),
                    status="cached",
                    judge_model=str(cached.get("judge_model") or config.judge_model),
                    overall_score=cached.get("overall_score"),
                    confidence=cached.get("confidence"),
                )
            )
            if progress_callback:
                progress_callback(index, len(method_dirs), method_dir, "cached")
            continue

        if progress_callback:
            progress_callback(index, len(method_dirs), method_dir, "running")
        normalized = _read_json(method_dir / "normalized.json")
        metadata = {}
        for filename in ("cache_key.json", "metadata.json", "metrics.json"):
            path = method_dir / filename
            if path.exists():
                metadata[filename.replace(".json", "")] = _read_json(path)
        judged = adapter.judge_extraction(normalized=normalized, metadata=metadata, model=config.judge_model)
        _write_json(method_dir / "judge.json", judged.to_json_dict())
        _write_json(method_dir / "judge_raw.json", {"text": judged.raw_text, "error": judged.error})
        _write_json(
            method_dir / "judge_metrics.json",
            {
                "usage": judged.usage or {},
                "latency_ms": judged.latency_ms,
                "outcome": judged.outcome,
                "correlation_id": judged.correlation_id,
            },
        )
        status = "success" if judged.outcome == "success" else "error"
        results.append(
            JudgeTaskResult(
                method_dir=str(method_dir),
                status=status,
                judge_model=judged.judge_model,
                overall_score=judged.overall_score,
                confidence=judged.confidence,
                error=judged.error,
            )
        )
        if progress_callback:
            progress_callback(index, len(method_dirs), method_dir, status)

    finished_at = _utc_now_text()
    judge_result = JudgeRunResult(
        run_id=run_id,
        task_results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    store.save_json(run_id, "judge_result.json", judge_result.to_json_dict())
    return judge_result


def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    import json

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
