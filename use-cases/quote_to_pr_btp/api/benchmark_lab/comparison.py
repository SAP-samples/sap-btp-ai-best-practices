"""Comparison aggregation for benchmark runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import RunStore
from .judge import list_method_dirs
from .pricing import estimate_document_ai_method_cost, estimate_model_cost


COMPARISON_COLUMNS = [
    "document",
    "method_family",
    "scenario",
    "model",
    "status",
    "quality_score",
    "extraction_quality",
    "pr_readiness",
    "confidence",
    "error_code",
    "error_explanation",
    "latency_s",
    "tokens",
    "estimated_cost",
    "cost_display",
    "cost_currency",
    "cost_basis",
    "risks",
    "recommendation",
]


@dataclass
class ComparisonResult:
    run_id: str
    rows: list[dict[str, Any]]
    field_rows: list[dict[str, Any]]
    artifacts: dict[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rows": self.rows,
            "field_rows": self.field_rows,
            "artifacts": self.artifacts,
        }


def build_comparison(store: RunStore, run_id: str) -> ComparisonResult:
    """Aggregate method, judge, and cost data into UI-safe comparison rows."""

    rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for method_dir in list_method_dirs(store, run_id):
        row, fields = _row_from_method_dir(method_dir)
        rows.append(row)
        field_rows.extend(fields)

    artifacts = save_comparison(store, run_id, rows, field_rows)
    return ComparisonResult(run_id=run_id, rows=rows, field_rows=field_rows, artifacts=artifacts)


def load_comparison(store: RunStore, run_id: str) -> ComparisonResult | None:
    path = store.run_dir(run_id) / "comparison.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ComparisonResult(
        run_id=run_id,
        rows=list(data.get("rows") or []),
        field_rows=list(data.get("field_rows") or []),
        artifacts=dict(data.get("artifacts") or {}),
    )


def save_comparison(
    store: RunStore,
    run_id: str,
    rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
) -> dict[str, str]:
    artifacts = {"json": "comparison.json", "csv": "comparison.csv"}
    payload = {"run_id": run_id, "rows": rows, "field_rows": field_rows, "artifacts": artifacts}
    store.save_json(run_id, "comparison.json", payload)

    csv_path = store.run_dir(run_id) / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return artifacts


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Business-level summary cards for the Compare page."""

    scored = [row for row in rows if isinstance(row.get("quality_score"), (int, float))]
    if not scored:
        return {
            "best_current_approach": "No scored result yet",
            "main_risk": "Run the judge step before comparing approaches.",
            "recommended_next_action": "Build comparison after extraction and judge results are available.",
            "trust_label": "Not ready",
        }

    best = max(scored, key=lambda item: (float(item.get("quality_score") or 0), float(item.get("confidence") or 0)))
    risks = [row.get("risks") for row in rows if row.get("risks") and float(row.get("quality_score") or 0) < 50]
    low_confidence = [row for row in scored if float(row.get("confidence") or 0) < 70]
    failed = [row for row in rows if str(row.get("status") or "").lower() == "error" or float(row.get("quality_score") or 0) <= 5]
    main_risk = "Low confidence on some results." if low_confidence else "No major quality risk flagged by the judge."
    if failed:
        main_risk = f"{len(failed)} result(s) failed or returned empty extraction; do not use those for PR automation."
    if risks:
        main_risk = _short_text(str(risks[0]), 360)
    return {
        "best_current_approach": f"{best.get('model')} / {best.get('scenario')} on {best.get('document')}",
        "main_risk": main_risk,
        "recommended_next_action": str(best.get("recommendation") or "Review the best result with business owners."),
        "trust_label": "Review recommended" if low_confidence else "Good candidate",
        "best_score": best.get("quality_score"),
        "best_confidence": best.get("confidence"),
    }


def summarize_run_findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact business digest for saved comparison results."""

    if not rows:
        return {
            "tested": "No scored methods found yet.",
            "use_models": "Run extraction and quality scoring first.",
            "avoid_models": "Not available.",
            "docai_coverage": "SAP Document AI was not measured in this run.",
        }
    documents = sorted({str(row.get("document")) for row in rows if row.get("document")})
    families = sorted({str(row.get("method_family")) for row in rows if row.get("method_family")})
    scenarios = sorted({str(row.get("scenario")) for row in rows if row.get("scenario")})
    scored = [row for row in rows if isinstance(row.get("quality_score"), (int, float))]
    strong = sorted(scored, key=lambda item: float(item.get("quality_score") or 0), reverse=True)[:5]
    weak = [row for row in scored if float(row.get("quality_score") or 0) < 50]
    docai_rows = [row for row in rows if row.get("method_family") == "docai"]
    return {
        "tested": (
            f"Checked {len(rows)} method result(s) across {len(documents)} document(s), "
            f"{len(families)} approach family/families, and {len(scenarios)} extraction strategy/strategies."
        ),
        "use_models": "; ".join(
            f"{row.get('model')} / {row.get('scenario')} ({row.get('quality_score')})" for row in strong[:3]
        )
        or "No strong candidate yet.",
        "avoid_models": (
            f"{len(weak)} method result(s) scored below 50 and should not be used without remediation."
            if weak
            else "No low-scoring method crossed the rejection threshold."
        ),
        "docai_coverage": (
            f"SAP Document AI included with {len(docai_rows)} result(s)."
            if docai_rows
            else "SAP Document AI was not measured in this run. Run a new pipeline with Document AI scenarios enabled."
        ),
    }


def _row_from_method_dir(method_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_key = _read_json_if_exists(method_dir / "cache_key.json")
    metrics = _read_json_if_exists(method_dir / "metrics.json")
    diagnostics = _read_json_if_exists(method_dir / "diagnostics.json")
    judge = _read_json_if_exists(method_dir / "judge.json")
    usage = metrics.get("usage") if isinstance(metrics.get("usage"), dict) else {}
    model = cache_key.get("model")
    method_family = cache_key.get("method_family")
    cost = estimate_document_ai_method_cost(1) if method_family == "docai" else estimate_model_cost(model, usage)
    extraction_outcome = metrics.get("outcome")
    judge_outcome = judge.get("outcome")
    status = extraction_outcome or judge_outcome or "not_scored"
    if extraction_outcome == "error":
        status = "error"
    risks = judge.get("hallucination_risks") or []
    missing = judge.get("missing_fields") or []
    risk_text = "; ".join(str(item) for item in [*risks, *missing] if item) or ""
    quality_score = _number_or_none(judge.get("overall_score"))
    confidence = _number_or_none(judge.get("confidence"))
    field_scores = judge.get("field_scores") if isinstance(judge.get("field_scores"), dict) else {}
    split_scores = split_quality_scores(field_scores)
    diagnostic = diagnostics.get("diagnostic") if isinstance(diagnostics.get("diagnostic"), dict) else {}
    error_code = metrics.get("diagnostic_category") or diagnostic.get("category") or _infer_error_code(
        metrics=metrics,
        judge=judge,
        risk_text=risk_text,
    )
    error_explanation = (
        metrics.get("diagnostic_root_cause")
        or diagnostic.get("root_cause")
        or _infer_error_explanation(error_code)
        or ""
    )

    row = {
        "document": cache_key.get("document_name"),
        "method_family": method_family,
        "scenario": cache_key.get("scenario_key"),
        "model": model,
        "status": status,
        "quality_score": quality_score,
        "extraction_quality": split_scores.get("extraction_quality"),
        "pr_readiness": split_scores.get("pr_readiness"),
        "confidence": confidence,
        "error_code": error_code,
        "error_explanation": error_explanation,
        "latency_s": round(float(metrics.get("latency_ms") or 0) / 1000, 2),
        "tokens": cost.get("total_tokens", 0),
        "estimated_cost": cost.get("estimated_cost"),
        "cost_display": cost.get("cost_display"),
        "cost_currency": cost.get("currency"),
        "cost_basis": cost.get("cost_basis") or "LLM token estimate",
        "risks": risk_text,
        "recommendation": judge.get("recommendation")
        or metrics.get("error")
        or "Run judge scoring to get a recommendation.",
    }

    field_rows = []
    method_label = f"{model or 'unknown'} / {cache_key.get('scenario_key') or 'unknown'}"
    for field_name, score in field_scores.items():
        field_rows.append(
            {
                "document": cache_key.get("document_name"),
                "method": method_label,
                "field": str(field_name),
                "score": _number_or_none(score),
            }
        )
    return row, field_rows


def _infer_error_code(*, metrics: dict[str, Any], judge: dict[str, Any], risk_text: str) -> str:
    status = str(metrics.get("outcome") or judge.get("outcome") or "").lower()
    combined = " ".join(
        [
            str(metrics.get("error") or ""),
            str(judge.get("recommendation") or ""),
            risk_text,
        ]
    ).lower()
    if "no extractable text layer" in combined or "no text layer" in combined:
        return "pdf_without_text_layer_for_text_route"
    if "no values were extracted" in combined or "all fields null" in combined:
        return "empty_extraction"
    if status == "error":
        return "unclassified_method_error"
    return ""


def _infer_error_explanation(error_code: str | None) -> str:
    if error_code == "pdf_without_text_layer_for_text_route":
        return "The source PDF is image-only or lacks an extractable text layer for the selected text route. Retry with OCR/PDF-native route."
    if error_code == "empty_extraction":
        return "The method returned an empty extraction. Treat this as not usable until the input route or prompt is fixed."
    if error_code == "unclassified_method_error":
        return "The method failed, but no mapped diagnostic category was saved. Inspect diagnostics.json or raw metrics."
    return ""


def split_quality_scores(field_scores: dict[str, Any]) -> dict[str, float | None]:
    """Derive business-friendly score splits from judge field groups.

    Overall judge score intentionally includes PR mapping readiness, so a quote can
    extract well while still scoring around 70-80 when SAP PR fields are missing.
    These derived scores separate those two business questions without rerunning
    extraction or judge calls.
    """

    extraction_quality = _weighted_field_score(
        field_scores,
        {
            "quote_header": 0.35,
            "line_items": 0.35,
            "evidence": 0.20,
            "warnings": 0.10,
        },
    )
    pr_readiness = _weighted_field_score(
        field_scores,
        {
            "pr_mapping": 0.55,
            "quote_header": 0.15,
            "line_items": 0.15,
            "evidence": 0.10,
            "warnings": 0.05,
        },
    )
    return {
        "extraction_quality": extraction_quality,
        "pr_readiness": pr_readiness,
    }


def _weighted_field_score(field_scores: dict[str, Any], weights: dict[str, float]) -> float | None:
    weighted_total = 0.0
    weight_total = 0.0
    for field_name, weight in weights.items():
        value = _number_or_none(field_scores.get(field_name))
        if value is None:
            continue
        weighted_total += float(value) * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return round(weighted_total / weight_total, 2)


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _short_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."
