"""
generate_report.py
------------------
Genera los archivos de output de la evaluacion:
- evaluation.json
- missing_fields.json
- scores.json
- executive_summary.txt
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = Path(__file__).parent.parent.parent / "output" / "evaluation"


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def _save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Saved: %s", path)


def build_executive_summary(
    scores_result: dict[str, Any],
    analysis: dict[str, Any],
    llm_eval: dict[str, Any],
) -> str:
    """Genera el resumen ejecutivo en texto plano."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scores = scores_result.get("scores") or {}
    best   = scores_result.get("best_method") or {}
    missing_in_sap = analysis.get("missing_in_sap") or []
    conflicts      = analysis.get("conflicts") or []
    recommendation = (llm_eval.get("recommendation") or {})
    exec_summary   = llm_eval.get("executive_summary") or ""

    lines = [
        "=" * 70,
        "  EXTRACTION QUALITY EVALUATION REPORT",
        "=" * 70,
        f"  Generated : {ts}",
        "",
        "─" * 70,
        "  BEST METHOD",
        "─" * 70,
        f"  -> {best.get('name', 'N/A')}",
        f"     Score : {best.get('overall_score', 0)}/100",
        f"     Reason: {best.get('reason', '')}",
        "",
        "─" * 70,
        "  OVERALL SCORES",
        "─" * 70,
    ]

    for key in ["sap", "llm_prompting", "llm_structured"]:
        s = scores.get(key) or {}
        lines += [
            f"  {s.get('method_name', key)}:",
            f"    Overall score  : {s.get('overall_score', 0)}/100",
            f"    Completeness   : {s.get('completeness', 0)}%",
            f"    Confidence avg : {s.get('confidence_avg', 0)}%",
            f"    Fields found   : {s.get('fields_found', 0)}/25",
            f"    Missing fields : {s.get('fields_missing', 0)}",
            f"    Line items     : {s.get('line_items_found', 0)}",
            "",
        ]

    if missing_in_sap:
        lines += ["─" * 70, "  MISSING FIELDS IN SAP DOCUMENT AI", "─" * 70]
        for m in missing_in_sap:
            detected = ", ".join(m.get("detected_by") or [])
            lines.append(f"  - {m['field']}  (detected by: {detected})")
        lines.append("")

    if conflicts:
        lines += ["─" * 70, "  CONFLICTS DETECTED", "─" * 70]
        for c in conflicts:
            lines.append(f"  - {c['field']}: {c.get('note', 'values differ')}")
            for method, val in (c.get("values") or {}).items():
                lines.append(f"      {method}: {val}")
        lines.append("")

    # LLM quality assessment
    qa = llm_eval.get("quality_assessment") or {}
    if qa:
        lines += ["─" * 70, "  LLM QUALITY ASSESSMENT", "─" * 70]
        for key in ["sap", "llm_prompting", "llm_structured"]:
            q = qa.get(key) or {}
            name = scores.get(key, {}).get("method_name", key)
            lines.append(f"  {name}: {q.get('quality', 'N/A').upper()}")
            lines.append(f"    {q.get('reasoning', '')}")
        lines.append("")

    if exec_summary:
        lines += ["─" * 70, "  EXECUTIVE SUMMARY", "─" * 70, f"  {exec_summary}", ""]

    rec_reason = recommendation.get("reason") or ""
    if rec_reason:
        lines += [
            "─" * 70,
            "  RECOMMENDATION",
            "─" * 70,
            f"  Use {recommendation.get('best_method', 'N/A')} as primary extraction method.",
            f"  {rec_reason}",
            "",
        ]

    lines += [
        "─" * 70,
        "  OUTPUT FILES",
        "─" * 70,
        f"  {OUTPUT_DIR}/evaluation.json",
        f"  {OUTPUT_DIR}/missing_fields.json",
        f"  {OUTPUT_DIR}/scores.json",
        f"  {OUTPUT_DIR}/executive_summary.txt",
        "=" * 70,
    ]

    return "\n".join(lines)


def save_all(
    analysis: dict[str, Any],
    scores_result: dict[str, Any],
    llm_eval: dict[str, Any],
    summary_text: str,
) -> dict[str, Path]:
    """Guarda todos los archivos de evaluacion."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "llm_evaluation": llm_eval,
    }

    paths = {
        "evaluation":       OUTPUT_DIR / "evaluation.json",
        "missing_fields":   OUTPUT_DIR / "missing_fields.json",
        "scores":           OUTPUT_DIR / "scores.json",
        "executive_summary": OUTPUT_DIR / "executive_summary.txt",
    }

    _save_json(evaluation,                          paths["evaluation"])
    _save_json(analysis.get("missing_in_sap") or [], paths["missing_fields"])
    _save_json(scores_result,                       paths["scores"])
    _save_text(summary_text,                        paths["executive_summary"])

    return paths