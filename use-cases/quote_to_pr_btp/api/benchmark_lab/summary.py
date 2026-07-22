"""Business summaries for completed benchmark runs."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from dotenv import load_dotenv

from observability.llm_usage_logging import emit_llm_usage_event, extract_token_usage

from .artifacts import BenchmarkRunConfig, RunStore
from .json_utils import parse_json_object
from .settings import AppPaths


@dataclass
class RunSummary:
    """Non-technical summary saved next to every run."""

    business_summary: str
    best_approach: str
    risks: list[str]
    recommendations: list[str]
    next_best_action: str
    generated_by_model: str
    used_llm: bool
    task_counts: dict[str, int]
    generated_at: str
    raw_text: str | None = None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class SummaryAdapter(Protocol):
    """Adapter contract used by tests and live LLM summary generation."""

    def generate_summary(
        self,
        *,
        config: BenchmarkRunConfig,
        batch_payload: dict[str, Any],
        model: str,
    ) -> RunSummary:
        ...


def summarize_task_counts(task_results: list[Any]) -> dict[str, int]:
    counts = {"total": len(task_results), "success": 0, "error": 0, "cached": 0, "skipped": 0}
    for item in task_results:
        status = getattr(item, "status", "unknown")
        if status in counts:
            counts[status] += 1
    return counts


def build_default_summary_shell(
    *,
    config: BenchmarkRunConfig,
    batch_payload: dict[str, Any],
    model: str,
    used_llm: bool = False,
    raw_text: str | None = None,
    error: str | None = None,
) -> RunSummary:
    """Create the fallback structure that live LLM output fills in."""

    counts = batch_payload.get("task_counts") or {}
    total = int(counts.get("total", 0))
    success = int(counts.get("success", 0))
    cached = int(counts.get("cached", 0))
    errors = int(counts.get("error", 0))
    completed = success + cached
    completion_rate = round((completed / total) * 100, 1) if total else 0.0

    if errors:
        best_approach = "Review failed LLM calls before selecting a winning extraction approach."
        next_action = "Open Technical details for failed rows, then rerun only the failed model/scenario pairs."
    elif total and cached == total:
        best_approach = "Cached results are ready for review; no new extraction calls were needed."
        next_action = "Move to Compare and inspect quality, latency, and cost tradeoffs."
    elif completed:
        best_approach = "The completed LLM batch is ready for judge scoring and DocAI comparison."
        next_action = "Run the judge step next, then compare the best LLM scenario with DocAI dynamic attributes."
    else:
        best_approach = "No extraction result is available yet."
        next_action = "Select at least one document, model, and LLM scenario, then start the batch."

    risks = []
    if errors:
        risks.append("Some model calls failed, so the benchmark is not yet comparable end to end.")
    if len(config.models) > 4:
        risks.append("Many models increase run time and token cost; cache reuse is important for demos.")
    if "simple_prompt" in config.llm_scenarios:
        risks.append("Simple prompts are useful as a baseline but may miss quote-specific edge cases.")
    if not risks:
        risks.append("Quality still needs judge scoring before the result can be used as evidence.")

    recommendations = [
        "Use detailed static prompt as the first LLM baseline.",
        "Compare the same document set across all selected models before drawing conclusions.",
        "Keep cached results enabled during UI review and stakeholder demos.",
    ]

    return RunSummary(
        business_summary=(
            f"Batch execution covered {total} planned LLM extraction task(s). "
            f"{completed} are available for review ({completion_rate}% complete), with {errors} error(s)."
        ),
        best_approach=best_approach,
        risks=risks,
        recommendations=recommendations,
        next_best_action=next_action,
        generated_by_model=model,
        used_llm=used_llm,
        task_counts=dict(counts),
        generated_at=batch_payload.get("finished_at") or batch_payload.get("started_at") or "",
        raw_text=raw_text,
        error=error,
    )


def build_error_summary(
    *,
    config: BenchmarkRunConfig,
    batch_payload: dict[str, Any],
    model: str,
    error: str,
    raw_text: str | None = None,
) -> RunSummary:
    """Create a transparent failed-summary artifact without pretending it is an LLM result."""

    counts = batch_payload.get("task_counts") or {}
    return RunSummary(
        business_summary="LLM summary was not generated. See Technical details for the model error.",
        best_approach="Not available until summary generation succeeds.",
        risks=["Summary generation failed; do not use this run as final business evidence yet."],
        recommendations=["Check AI Core credentials/model availability and rerun the summary step."],
        next_best_action="Fix the summary model call, then rebuild the comparison summary.",
        generated_by_model=model,
        used_llm=False,
        task_counts=dict(counts),
        generated_at=batch_payload.get("finished_at") or batch_payload.get("started_at") or "",
        raw_text=raw_text,
        error=error,
    )


def summary_to_markdown(summary: RunSummary) -> str:
    risk_lines = "\n".join(f"- {item}" for item in summary.risks)
    recommendation_lines = "\n".join(f"- {item}" for item in summary.recommendations)
    return (
        "# Benchmark Run Summary\n\n"
        f"## Business summary\n{summary.business_summary}\n\n"
        f"## Best approach\n{summary.best_approach}\n\n"
        f"## Risks\n{risk_lines}\n\n"
        f"## Recommendations\n{recommendation_lines}\n\n"
        f"## Next best action\n{summary.next_best_action}\n"
    )


class OrchestrationSummaryAdapter:
    """Generate a business summary through GenAI Hub orchestration."""

    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or AppPaths.for_project()
        load_dotenv(dotenv_path=self.paths.env_file)

    def generate_summary(
        self,
        *,
        config: BenchmarkRunConfig,
        batch_payload: dict[str, Any],
        model: str,
    ) -> RunSummary:
        try:
            from gen_ai_hub.orchestration.models.config import OrchestrationConfig
            from gen_ai_hub.orchestration.models.llm import LLM
            from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
            from gen_ai_hub.orchestration.models.template import Template
            from gen_ai_hub.orchestration.service import OrchestrationService
        except Exception as exc:
            return build_error_summary(
                config=config,
                batch_payload=batch_payload,
                model=model,
                error=f"Summary model unavailable: {exc}",
            )

        correlation_id = str(uuid.uuid4())
        started = time.perf_counter()
        response_obj: Any = None
        outcome = "success"
        raw = ""
        try:
            prompt = (
                "Write a concise business summary for non-technical stakeholders. "
                "Explain what was tested, which approaches should be used, which approaches failed or underperformed, "
                "whether SAP Document AI was included, and what is still missing before purchase requisition creation. "
                "Include: business_summary, best_approach, risks, recommendations, next_best_action. "
                "Return only valid JSON with those exact keys. "
                "risks and recommendations must be arrays of short strings. "
                "Use plain ASCII punctuation only. "
                "Do not expose raw extraction JSON or implementation details."
            )
            template = Template(
                messages=[
                    SystemMessage("You are a senior SAP solution advisor writing for business users."),
                    UserMessage(f"{prompt}\n\nRUN CONFIG:\n{config.normalized()}\n\nBATCH RESULT:\n{batch_payload}"),
                ]
            )
            llm = LLM(name=model, version="latest")
            result = OrchestrationService(config=OrchestrationConfig(template=template, llm=llm)).run()
            response_obj = result.orchestration_result
            raw = response_obj.choices[0].message.content
            summary = build_default_summary_shell(
                config=config,
                batch_payload=batch_payload,
                model=model,
                used_llm=True,
                raw_text=raw,
            )
            parsed = parse_json_object(raw)
            summary.business_summary = str(parsed.get("business_summary") or summary.business_summary)
            summary.best_approach = str(parsed.get("best_approach") or summary.best_approach)
            summary.next_best_action = str(parsed.get("next_best_action") or summary.next_best_action)
            risks = parsed.get("risks")
            recommendations = parsed.get("recommendations")
            if isinstance(risks, list):
                summary.risks = [str(item) for item in risks]
            if isinstance(recommendations, list):
                summary.recommendations = [str(item) for item in recommendations]
            return summary
        except Exception as exc:
            outcome = "error"
            return build_error_summary(
                config=config,
                batch_payload=batch_payload,
                model=model,
                raw_text=raw,
                error=f"Summary generation failed: {exc}",
            )
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            usage_tokens = extract_token_usage(response_obj)
            emit_llm_usage_event(
                route="streamlit.run_summary",
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


def save_summary_artifacts(store: RunStore, run_id: str, summary: RunSummary) -> None:
    summary_json = summary.to_json_dict()
    store.save_json(run_id, "summary.json", summary_json)
    summary_path = store.run_dir(run_id) / "summary.md"
    summary_path.write_text(summary_to_markdown(summary), encoding="utf-8")

    run_summary = store.load_json(run_id, "run_summary.json")
    run_summary.update(
        {
            "status": "completed" if summary.task_counts.get("error", 0) == 0 else "completed_with_errors",
            "business_summary": summary.business_summary,
            "best_approach": summary.best_approach,
            "next_best_action": summary.next_best_action,
            "summary_model": summary.generated_by_model,
            "summary_used_llm": summary.used_llm,
            "summary_json": "summary.json",
            "summary_markdown": "summary.md",
            "task_counts": summary.task_counts,
        }
    )
    store.save_json(run_id, "run_summary.json", run_summary)
