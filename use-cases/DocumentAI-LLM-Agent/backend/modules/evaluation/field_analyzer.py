"""
field_analyzer.py
-----------------
Analiza campos extraidos por cada metodo:
- completitud
- campos faltantes
- conflictos entre metodos
- campos vacios
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

INVOICE_FIELDS = [
    "taxAmount", "senderAddress", "senderBankAccount", "grossAmount",
    "receiverName", "purchaseOrderNumber", "senderName", "currencyCode",
    "documentNumber", "documentDate", "receiverAddress", "taxId",
    "netAmount", "deliveryDate", "receiverContact", "taxRate",
    "senderCity", "senderCountryCode", "senderHouseNumber", "senderStreet",
    "senderPostalCode", "receiverCity", "receiverCountryCode",
    "receiverHouseNumber", "receiverStreet",
]

METHOD_NAMES = {
    "sap": "SAP Document AI",
    "llm_prompting": "LLM Prompting",
    "llm_structured": "LLM Structured",
}


def _normalize_sap(sap_raw: dict) -> dict[str, Any]:
    """Extrae campos del formato SAP Document AI."""
    result: dict[str, Any] = {f: None for f in INVOICE_FIELDS}
    result["lineItems"] = []
    result["fieldConfidence"] = {}

    doc = sap_raw.get("extraction") or sap_raw.get("document") or {}
    for field in doc.get("headerFields") or []:
        name = field.get("name") or ""
        val  = field.get("value") if field.get("value") is not None else field.get("rawValue")
        conf = field.get("confidence")
        if name in result:
            result[name] = val
        if conf is not None:
            result["fieldConfidence"][name] = conf

    for item in doc.get("lineItems") or []:
        li: dict[str, Any] = {}
        cols = item if isinstance(item, list) else item.get("columns") or []
        for col in cols:
            if isinstance(col, dict):
                n = col.get("name", "")
                v = col.get("value") if col.get("value") is not None else col.get("rawValue")
                if n:
                    li[n] = v
        result["lineItems"].append(li)

    return result


def _get_conf(data: dict, field: str) -> float:
    fc = data.get("fieldConfidence") or {}
    v = fc.get(field)
    if v is not None:
        return float(v)
    conf = data.get("confidence") or {}
    v2 = (conf.get("fields") or {}).get(field)
    return float(v2) if v2 is not None else 0.0


def analyze_fields(
    sap_raw: dict,
    llm_p1: dict,
    llm_p2: dict,
) -> dict[str, Any]:
    """
    Analiza completitud, missing fields y conflictos entre los tres metodos.

    Returns:
        Diccionario con analisis completo de campos.
    """
    sap = _normalize_sap(sap_raw)
    methods = {"sap": sap, "llm_prompting": llm_p1, "llm_structured": llm_p2}

    # ── Completitud por metodo ───────────────────────────────────────────
    completeness: dict[str, dict] = {}
    for key, data in methods.items():
        found = [f for f in INVOICE_FIELDS if data.get(f) is not None]
        missing = [f for f in INVOICE_FIELDS if data.get(f) is None]
        line_items = data.get("lineItems") or []
        conf_scores = [_get_conf(data, f) for f in found]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        completeness[key] = {
            "method_name": METHOD_NAMES[key],
            "fields_found": len(found),
            "fields_missing": len(missing),
            "found_list": found,
            "missing_list": missing,
            "line_items_count": len(line_items),
            "confidence_avg": round(avg_conf, 3),
        }

    # ── Missing fields (SAP omite, LLM detecta) ─────────────────────────
    missing_in_sap: list[dict] = []
    for field in INVOICE_FIELDS:
        sap_val = sap.get(field)
        p1_val  = llm_p1.get(field)
        p2_val  = llm_p2.get(field)

        if sap_val is None and (p1_val is not None or p2_val is not None):
            detected_by = []
            if p1_val is not None:
                detected_by.append("LLM Prompting")
            if p2_val is not None:
                detected_by.append("LLM Structured")

            missing_in_sap.append({
                "field": field,
                "label": "missing field",
                "source_missing": "SAP Document AI",
                "detected_by": detected_by,
                "llm_prompting_value": p1_val,
                "llm_structured_value": p2_val,
            })

    # ── Conflictos entre metodos ─────────────────────────────────────────
    conflicts: list[dict] = []
    for field in INVOICE_FIELDS:
        vals = {
            k: str(methods[k].get(field)).strip().lower()
            for k in methods
            if methods[k].get(field) is not None
        }
        if len(vals) >= 2:
            unique_vals = set(vals.values())
            if len(unique_vals) > 1:
                conflicts.append({
                    "field": field,
                    "status": "conflict detected",
                    "values": {METHOD_NAMES[k]: methods[k].get(field) for k in vals},
                    "note": "Values differ between extraction methods",
                })

    logger.info(
        "Field analysis complete. missing_in_sap=%d, conflicts=%d",
        len(missing_in_sap), len(conflicts),
    )

    return {
        "completeness": completeness,
        "missing_in_sap": missing_in_sap,
        "conflicts": conflicts,
        "normalized": {k: methods[k] for k in methods},
    }