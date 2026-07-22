"""Failure diagnostics for benchmark runs and method artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureDiagnostic:
    """Business-safe and technical diagnosis for one failed method."""

    category: str
    root_cause: str
    recommended_action: str
    exclusion_candidate: bool = False
    retry_candidate: bool = True
    technical_details: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_llm_failure(*, model: str | None, error: str | None, document_name: str | None = None) -> FailureDiagnostic:
    """Classify known LLM extraction failures into actionable root causes."""

    model_name = model or "unknown model"
    message = str(error or "").strip()
    lowered = message.lower()

    if "nonetype" in lowered and "create" in lowered:
        return FailureDiagnostic(
            category="openai_native_client_unavailable",
            root_cause=(
                "The GenAI Hub native OpenAI Responses client was not available in this runtime, "
                "so the previous code path tried to call responses.create on None."
            ),
            recommended_action=(
                "Use the orchestration text fallback for OpenAI-family models when the native Responses client "
                "is unavailable. Re-run with the current adapter; this is not a model-quality failure."
            ),
            exclusion_candidate=False,
            retry_candidate=True,
            technical_details=message,
        )

    if "requires text fallback" in lowered and "no text layer" in lowered:
        return FailureDiagnostic(
            category="pdf_without_text_layer_for_text_route",
            root_cause=(
                f"{document_name or 'The PDF'} has no extractable text layer, while {model_name} was routed "
                "through the text-only orchestration fallback."
            ),
            recommended_action=(
                "Run OCR before text-only models, use a PDF-native model route, or exclude this model/document "
                "combination from image-only PDFs."
            ),
            exclusion_candidate=model_name.startswith(("anthropic--", "sonar")),
            retry_candidate=True,
            technical_details=message,
        )

    if "404" in lowered and "not_found" in lowered and "publisher model" in lowered:
        return FailureDiagnostic(
            category="model_not_available_in_ai_core",
            root_cause=f"{model_name} is not deployed or not entitled in the current SAP AI Core landscape.",
            recommended_action=(
                "Remove this model from the default benchmark set until AI Core entitlement/deployment is confirmed."
            ),
            exclusion_candidate=True,
            retry_candidate=False,
            technical_details=message,
        )

    if "model name" in lowered and "is not supported" in lowered:
        return FailureDiagnostic(
            category="model_not_supported_by_orchestration",
            root_cause=f"{model_name} is not currently supported by the SAP AI Core Orchestration endpoint.",
            recommended_action="Remove this model from the customer default and retry only after the supported-model list includes it.",
            exclusion_candidate=True,
            retry_candidate=False,
            technical_details=message,
        )

    if "410" in lowered and ("retired" in lowered or "gone" in lowered):
        return FailureDiagnostic(
            category="model_retired_in_ai_core",
            root_cause=f"{model_name} is retired in the current SAP AI Core deployment.",
            recommended_action="Remove this model from benchmark defaults and replace it with a currently deployed successor.",
            exclusion_candidate=True,
            retry_candidate=False,
            technical_details=message,
        )

    if "does not support" in lowered and ("file" in lowered or "pdf" in lowered or "binary" in lowered):
        return FailureDiagnostic(
            category="model_does_not_support_pdf_payload",
            root_cause=f"{model_name} appears not to support direct PDF/binary input through the selected route.",
            recommended_action="Route it through OCR/text extraction or exclude it from PDF-native scenarios.",
            exclusion_candidate=True,
            retry_candidate=True,
            technical_details=message,
        )

    if "server disconnected" in lowered or "connection reset" in lowered or "temporarily unavailable" in lowered:
        return FailureDiagnostic(
            category="llm_transient_transport_error",
            root_cause=f"{model_name} call failed because the upstream service disconnected before returning a response.",
            recommended_action="Retry the same method; if it repeats, reduce payload size or route this scenario through a more stable model.",
            exclusion_candidate=False,
            retry_candidate=True,
            technical_details=message,
        )

    return FailureDiagnostic(
        category="llm_unclassified_error",
        root_cause="The LLM call failed with an error that is not yet mapped to a known diagnostic category.",
        recommended_action="Inspect diagnostics.json/raw error, then add a targeted classifier or adapter fix.",
        exclusion_candidate=False,
        retry_candidate=True,
        technical_details=message,
    )


def classify_docai_failure(*, error: str | None) -> FailureDiagnostic:
    """Classify SAP Document AI failures into actionable root causes."""

    message = str(error or "").strip()
    lowered = message.lower()

    if "invalid extraction fields" in lowered or "e36" in lowered:
        return FailureDiagnostic(
            category="docai_invalid_extraction_fields",
            root_cause=(
                "SAP Document AI rejected the requested field list for the selected document type. "
                "This usually means custom quote fields were sent to a predefined type such as invoice, "
                "or a custom schema should be used instead of ad-hoc extraction fields."
            ),
            recommended_action=(
                "Use only supported predefined invoice fields for invoice fallback, and use managed custom "
                "schemas for quote-specific dynamic attributes."
            ),
            exclusion_candidate=False,
            retry_candidate=True,
            technical_details=message,
        )

    if "custom label provided already exists" in lowered or "e178" in lowered:
        return FailureDiagnostic(
            category="docai_custom_label_collision",
            root_cause="SAP Document AI rejected the upload because the custom label was reused from a previous run.",
            recommended_action="Use unique custom labels per upload attempt, then rerun the failed Document AI method.",
            exclusion_candidate=False,
            retry_candidate=True,
            technical_details=message,
        )

    if "document type" in lowered and "not supported" in lowered:
        return FailureDiagnostic(
            category="docai_document_type_not_supported",
            root_cause="SAP Document AI rejected the selected document type for this tenant/service plan.",
            recommended_action="Query capabilities and switch to a supported predefined type or managed custom schema.",
            exclusion_candidate=False,
            retry_candidate=True,
            technical_details=message,
        )

    if "timed out" in lowered:
        return FailureDiagnostic(
            category="docai_timeout",
            root_cause="SAP Document AI accepted the job but did not finish within the configured timeout.",
            recommended_action="Increase polling timeout or inspect the job status in SAP Document AI.",
            exclusion_candidate=False,
            retry_candidate=True,
            technical_details=message,
        )

    return FailureDiagnostic(
        category="docai_unclassified_error",
        root_cause="SAP Document AI failed with an error that is not yet mapped to a known diagnostic category.",
        recommended_action="Inspect diagnostics.json/raw error, SAP code, and Document AI capabilities.",
        exclusion_candidate=False,
        retry_candidate=True,
        technical_details=message,
    )


def inspect_pdf_text_layer(pdf_path: Path) -> dict[str, Any]:
    """Return a cheap PDF text-layer diagnostic without OCR."""

    try:
        import pypdf  # type: ignore
    except Exception as exc:
        return {
            "path": str(pdf_path),
            "available": False,
            "error": f"pypdf is unavailable: {exc}",
            "has_text_layer": None,
        }

    try:
        page_lengths: list[int] = []
        with pdf_path.open("rb") as handle:
            reader = pypdf.PdfReader(handle)
            for page in reader.pages:
                page_lengths.append(len((page.extract_text() or "").strip()))
        total = sum(page_lengths)
        return {
            "path": str(pdf_path),
            "available": True,
            "page_count": len(page_lengths),
            "page_text_lengths": page_lengths,
            "total_text_length": total,
            "has_text_layer": total > 0,
        }
    except Exception as exc:
        return {
            "path": str(pdf_path),
            "available": False,
            "error": str(exc),
            "has_text_layer": None,
        }


def summarize_run_failures(run_dir: Path) -> dict[str, Any]:
    """Build a run-level diagnostic summary from saved method artifacts."""

    method_dir = run_dir / "methods"
    rows: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    exclusion_candidates: dict[str, int] = {}

    if not method_dir.exists():
        return {"run_id": run_dir.name, "method_count": 0, "failures": [], "categories": {}}

    for folder in sorted(method_dir.iterdir()):
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        cache_path = folder / "cache_key.json"
        if not metadata_path.exists():
            continue
        metadata = _load_json(metadata_path)
        cache = _load_json(cache_path) if cache_path.exists() else {}
        outcome = metadata.get("outcome") or metadata.get("status")
        error = metadata.get("error")
        if outcome != "error" and not error:
            continue
        family = cache.get("method_family") or metadata.get("method_family")
        model = cache.get("model") or metadata.get("model")
        document = cache.get("document_name") or metadata.get("document_name") or metadata.get("document")
        scenario = cache.get("scenario_key") or metadata.get("scenario_key") or metadata.get("scenario")
        diagnostic = (
            classify_docai_failure(error=error)
            if family == "docai"
            else classify_llm_failure(model=model, error=error, document_name=document)
        )
        payload = {
            "method_dir": folder.name,
            "document": document,
            "method_family": family,
            "model": model,
            "scenario": scenario,
            "error": error,
            **diagnostic.to_json_dict(),
        }
        rows.append(payload)
        categories[diagnostic.category] = categories.get(diagnostic.category, 0) + 1
        if diagnostic.exclusion_candidate and model:
            exclusion_candidates[str(model)] = exclusion_candidates.get(str(model), 0) + 1

    return {
        "run_id": run_dir.name,
        "method_count": len([p for p in method_dir.iterdir() if p.is_dir()]) if method_dir.exists() else 0,
        "failure_count": len(rows),
        "categories": categories,
        "exclusion_candidates": exclusion_candidates,
        "failures": rows,
    }


def save_run_diagnostics(run_dir: Path) -> dict[str, Any]:
    """Persist JSON and Markdown diagnostic summaries next to run artifacts."""

    summary = summarize_run_failures(run_dir)
    (run_dir / "run_diagnostics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_diagnostics.md").write_text(_diagnostics_markdown(summary), encoding="utf-8")
    return summary


def _diagnostics_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Run diagnostics: {summary.get('run_id')}",
        "",
        f"Failures: {summary.get('failure_count', 0)} / {summary.get('method_count', 0)} method artifacts",
        "",
        "## Categories",
    ]
    for category, count in sorted((summary.get("categories") or {}).items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Exclusion candidates"])
    for model, count in sorted((summary.get("exclusion_candidates") or {}).items()):
        lines.append(f"- {model}: {count} failing row(s)")
    lines.extend(["", "## Failed rows"])
    for row in summary.get("failures", []):
        lines.append(
            f"- {row.get('document')} | {row.get('method_family')} | {row.get('model')} | "
            f"{row.get('scenario')} | {row.get('category')}: {row.get('root_cause')}"
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
