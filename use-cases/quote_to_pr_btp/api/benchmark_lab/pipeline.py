"""One-click comparative pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable

from .artifacts import BenchmarkRunConfig, RunStore
from .comparison import ComparisonResult, build_comparison
from .document_report import append_document_report_to_summary, save_document_report
from .docai_runner import DocumentAIBatchResult, DocumentAITask, run_docai_batch
from .judge import run_judge_for_methods
from .runner import BatchRunResult, BatchTask, run_llm_batch
from .settings import AppPaths
from .summary import OrchestrationSummaryAdapter, save_summary_artifacts


@dataclass
class PipelineResult:
    """End-to-end output from the one-click pipeline."""

    run_id: str
    run_dir: str
    batch: dict[str, Any]
    docai: dict[str, Any] | None
    judge: dict[str, Any]
    comparison: dict[str, Any]
    summary: dict[str, Any] | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_comparative_pipeline(
    *,
    config: BenchmarkRunConfig,
    paths: AppPaths | None = None,
    store: RunStore | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    run_quality_judge: bool = True,
    run_business_summary: bool = True,
) -> PipelineResult:
    """Create a run and execute extraction, judging, comparison, and summary."""

    paths = paths or AppPaths.for_project()
    store = store or RunStore(paths)
    run_dir = store.create_run(config)
    run_id = run_dir.name

    def tell(stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage, message)

    tell("Extraction", "Calling selected models and saving extraction artifacts.")

    def extraction_progress(index: int, total: int, task: BatchTask, status: str) -> None:
        tell("Extraction", f"{index}/{total}: {task.model} on {task.document_name} ({status})")

    batch_result: BatchRunResult = run_llm_batch(
        run_id=run_id,
        config=config,
        paths=paths,
        store=store,
        progress_callback=extraction_progress,
    )

    docai_result: DocumentAIBatchResult | None = None
    if config.docai_scenarios:
        tell("Document AI", "Calling SAP Document AI and saving extraction artifacts.")

        def docai_progress(index: int, total: int, task: DocumentAITask, status: str) -> None:
            tell("Document AI", f"{index}/{total}: SAP Document AI {task.scenario_key} on {task.document_name} ({status})")

        docai_result = run_docai_batch(
            run_id=run_id,
            config=config,
            paths=paths,
            store=store,
            progress_callback=docai_progress,
        )

    judge_result = None
    if run_quality_judge:
        tell("Quality scoring", "Evaluating extraction quality with the selected judge model.")
        judge_result = run_judge_for_methods(
            run_id=run_id,
            config=config,
            paths=paths,
            store=store,
            progress_callback=lambda index, total, _method_dir, status: tell(
                "Quality scoring", f"{index}/{total}: judge scoring {status}"
            ),
        )

    tell("Comparison", "Building comparison charts and cost table.")
    comparison: ComparisonResult = build_comparison(store, run_id)

    summary_payload = {
        "run_id": run_id,
        "task_counts": batch_result.task_counts,
        "docai_task_counts": docai_result.task_counts if docai_result else {},
        "judge_counts": judge_result.task_counts if judge_result else {"skipped": 1},
        "comparison_rows": comparison.rows,
        "field_rows": comparison.field_rows,
        "finished_at": judge_result.finished_at if judge_result else datetime.now().isoformat(timespec="seconds"),
    }
    summary = None
    document_report_artifacts = None
    if run_business_summary:
        tell("Summary", "Generating business summary with the selected summary model.")
        summary = OrchestrationSummaryAdapter(paths).generate_summary(
            config=config,
            batch_payload=summary_payload,
            model=config.summary_model,
        )
        save_summary_artifacts(store, run_id, summary)
        document_report_artifacts = save_document_report(store, run_id, comparison.rows)
        append_document_report_to_summary(store.run_dir(run_id))
    tell("Done", "Extraction completed. The result is ready for validation.")

    result = PipelineResult(
        run_id=run_id,
        run_dir=str(run_dir),
        batch=batch_result.to_json_dict(),
        docai=docai_result.to_json_dict() if docai_result else None,
        judge=judge_result.to_json_dict() if judge_result else {"status": "skipped"},
        comparison=comparison.to_json_dict(),
        summary=summary.to_json_dict() if summary else None,
    )
    pipeline_payload = result.to_json_dict()
    pipeline_payload["document_report"] = document_report_artifacts
    store.save_json(run_id, "pipeline_result.json", pipeline_payload)
    return result
