"""Per-document benchmark report artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import RunStore


def build_document_report_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build UI-safe per-document top approaches and failure summaries."""

    documents: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        document = str(row.get("document") or "unknown")
        grouped[document].append(row)

    for document, doc_rows in sorted(grouped.items()):
        scored = sorted(
            doc_rows,
            key=lambda item: (
                _number(item.get("quality_score")),
                _number(item.get("extraction_quality")),
                _number(item.get("confidence")),
            ),
            reverse=True,
        )
        failures = [row for row in doc_rows if str(row.get("status") or "").lower() == "error"]
        low_quality = [row for row in doc_rows if _number(row.get("quality_score")) < 50]
        top_rows = scored[:5]
        documents[document] = {
            "top_approaches": [_compact_row(row) for row in top_rows],
            "failure_summary": _failure_summary(failures, low_quality),
            "best_business_readout": _best_business_readout(top_rows[0]) if top_rows else "No scored approach is available.",
        }

    return {"documents": documents}


def save_document_report(store: RunStore, run_id: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Persist per-document report as JSON and Markdown."""

    payload = build_document_report_payload(rows)
    store.save_json(run_id, "document_report.json", payload)
    markdown = document_report_to_markdown(payload, store.run_dir(run_id))
    report_path = store.run_dir(run_id) / "document_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return {"json": "document_report.json", "markdown": "document_report.md"}


def append_document_report_to_summary(run_dir: Path) -> None:
    """Append per-document report to summary.md when both artifacts exist."""

    summary_path = run_dir / "summary.md"
    report_path = run_dir / "document_report.md"
    if not summary_path.exists() or not report_path.exists():
        return
    summary_text = summary_path.read_text(encoding="utf-8").rstrip()
    report_text = report_path.read_text(encoding="utf-8").strip()
    marker = "\n\n---\n\n# Document-Level Details\n"
    if "# Document-Level Details" in summary_text:
        summary_text = summary_text.split("\n\n---\n\n# Document-Level Details", 1)[0].rstrip()
    summary_path.write_text(f"{summary_text}{marker}{report_text}\n", encoding="utf-8")


def document_report_to_markdown(payload: dict[str, Any], run_dir: Path | None = None) -> str:
    lines = ["# Document-Level Details", ""]
    documents = payload.get("documents") if isinstance(payload.get("documents"), dict) else {}
    for document, details in documents.items():
        lines.extend([f"## {document}", ""])
        if run_dir:
            lines.append(f"Source file: `{document}`")
            lines.append("")
        lines.append(str(details.get("best_business_readout") or "No best approach available."))
        lines.extend(["", "### Top 5 approaches", ""])
        lines.append("| Rank | Approach | Overall score | Extraction quality | PR readiness | Confidence | Cost | Status | Error code | Recommendation |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
        for index, row in enumerate(details.get("top_approaches") or [], start=1):
            approach = f"{row.get('method_family')} / {row.get('model')} / {row.get('scenario')}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _md(approach),
                        str(row.get("quality_score")),
                        str(row.get("extraction_quality")),
                        str(row.get("pr_readiness")),
                        str(row.get("confidence")),
                        _md(row.get("cost_display") or "not configured"),
                        _md(row.get("status")),
                        _md(row.get("error_code") or ""),
                        _md(_short(row.get("recommendation"), 180)),
                    ]
                )
                + " |"
            )
        lines.extend(["", "### What did not work", ""])
        lines.append(str(details.get("failure_summary") or "No failed or low-quality methods for this document."))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": row.get("document"),
        "method_family": row.get("method_family"),
        "model": row.get("model"),
        "scenario": row.get("scenario"),
        "status": row.get("status"),
        "quality_score": row.get("quality_score"),
        "extraction_quality": row.get("extraction_quality"),
        "pr_readiness": row.get("pr_readiness"),
        "confidence": row.get("confidence"),
        "error_code": row.get("error_code"),
        "error_explanation": row.get("error_explanation"),
        "cost_display": row.get("cost_display"),
        "recommendation": row.get("recommendation"),
        "risks": row.get("risks"),
    }


def _failure_summary(failures: list[dict[str, Any]], low_quality: list[dict[str, Any]]) -> str:
    if not failures and not low_quality:
        return "All tested approaches produced usable scored output. Review the top table for quality trade-offs."
    parts: list[str] = []
    if failures:
        labels = [
            f"{row.get('model')} / {row.get('scenario')} [{row.get('error_code') or 'error'}]: {_short(row.get('error_explanation') or row.get('recommendation'), 140)}"
            for row in failures[:5]
        ]
        parts.append(f"{len(failures)} approach(es) failed. " + " ".join(labels))
    if low_quality:
        parts.append(
            f"{len(low_quality)} approach(es) scored below 50, usually because required PR fields, evidence, or line details were missing."
        )
    return " ".join(parts)


def _best_business_readout(row: dict[str, Any]) -> str:
    return (
        f"Best current approach for this document is {row.get('model')} / {row.get('scenario')} "
        f"with overall score {row.get('quality_score')}, extraction quality {row.get('extraction_quality')}, "
        f"PR readiness {row.get('pr_readiness')}, and confidence {row.get('confidence')}. "
        f"Use it as a candidate extraction, not as an auto-PR record, until missing SAP PR mapping fields are enriched."
    )


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _short(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
