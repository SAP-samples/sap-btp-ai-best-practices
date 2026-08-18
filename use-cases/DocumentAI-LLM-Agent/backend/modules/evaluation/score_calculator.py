"""
score_calculator.py
-------------------
Calcula scores numericos por metodo de extraccion:
- completeness score
- confidence score
- consistency score
- line item accuracy
- field coverage
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOTAL_FIELDS = 25  # total de campos INVOICE_FIELDS


def _pct(value: float, total: float) -> int:
    """Convierte a porcentaje entero 0-100."""
    if total == 0:
        return 0
    return min(100, round((value / total) * 100))


def calculate_scores(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Calcula scores para cada metodo basado en el analisis de campos.

    Args:
        analysis: Resultado de field_analyzer.analyze_fields()

    Returns:
        Diccionario con scores por metodo y mejor metodo.
    """
    completeness = analysis.get("completeness") or {}
    conflicts = analysis.get("conflicts") or []
    missing_in_sap = analysis.get("missing_in_sap") or []

    conflict_fields = {c["field"] for c in conflicts}
    scores: dict[str, dict] = {}

    for key, comp in completeness.items():
        found       = comp.get("fields_found", 0)
        conf_avg    = comp.get("confidence_avg", 0.0)
        line_items  = comp.get("line_items_count", 0)
        missing     = comp.get("fields_missing", 0)

        # Completeness: % de campos encontrados
        completeness_score = _pct(found, TOTAL_FIELDS)

        # Confidence: promedio de confianza * 100
        confidence_score = min(100, round(conf_avg * 100))

        # Consistency: penalizar por conflictos en campos de este metodo
        found_list = set(comp.get("found_list") or [])
        conflicts_in_method = len(conflict_fields & found_list)
        consistency_score = max(0, 100 - (conflicts_in_method * 10))

        # Line item accuracy: bonus si tiene line items
        line_item_score = min(100, line_items * 50) if line_items > 0 else 0

        # Field coverage: penalizar campos faltantes
        coverage_score = _pct(found, TOTAL_FIELDS)

        # Overall: promedio ponderado
        overall = round(
            completeness_score * 0.35
            + confidence_score  * 0.30
            + consistency_score * 0.20
            + coverage_score    * 0.15
        )

        scores[key] = {
            "method_name": comp.get("method_name", key),
            "overall_score": overall,
            "completeness": completeness_score,
            "confidence_avg": confidence_score,
            "consistency": consistency_score,
            "line_items_found": line_items,
            "fields_found": found,
            "fields_missing": missing,
            "missing_fields_list": comp.get("missing_list") or [],
        }

    # Determinar mejor metodo
    best_key = max(scores, key=lambda k: scores[k]["overall_score"])
    best = scores[best_key]

    # Razon automatica
    reasons = []
    if best["completeness"] >= 90:
        reasons.append("highest completeness")
    if best["confidence_avg"] >= 80:
        reasons.append("high confidence scores")
    if best["fields_missing"] == 0:
        reasons.append("no missing fields")
    if best["line_items_found"] > 0:
        reasons.append("line items detected")
    reason = ", ".join(reasons) if reasons else "highest overall score"

    logger.info(
        "Scores calculated. Best method: %s (score: %d)",
        best["method_name"], best["overall_score"],
    )

    return {
        "scores": scores,
        "best_method": {
            "key": best_key,
            "name": best["method_name"],
            "overall_score": best["overall_score"],
            "reason": reason,
        },
        "missing_in_sap_count": len(missing_in_sap),
        "conflicts_count": len(conflicts),
    }