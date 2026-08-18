"""
compare_results.py
------------------
Compares extraction results: SAP Document AI vs LLM Technique 1 vs LLM Technique 2.
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


def _normalize_sap(sap_raw: dict) -> dict[str, Any]:
    """Normalize the SAP Document AI response to the common format."""
    result: dict[str, Any] = {f: None for f in INVOICE_FIELDS}
    result["lineItems"] = []
    result["fieldConfidence"] = {}

    doc = sap_raw.get("extraction") or sap_raw.get("document") or {}
    header = doc.get("headerFields") or []
    lines = doc.get("lineItems") or []

    for field in header:
        name = field.get("name") or ""
        value = field.get("value") if field.get("value") is not None else field.get("rawValue")
        conf = field.get("confidence")
        if name in result:
            result[name] = value
        if conf is not None:
            result["fieldConfidence"][name] = conf

    # SAP lineItems is a list of lists: [[{name, value, ...}, ...], [...]]
    for item in lines:
        li: dict[str, Any] = {}
        cols = item if isinstance(item, list) else item.get("columns") or []
        for col in cols:
            if isinstance(col, dict):
                name = col.get("name", "")
                val  = col.get("value") if col.get("value") is not None else col.get("rawValue")
                if name:
                    li[name] = val
        result["lineItems"].append(li)

    return result


def _get_confidence(data: dict, field: str) -> float | None:
    """Extract confidence for a field from different response structures."""
    # Techniques 1 and 2 use 'fieldConfidence'
    fc = data.get("fieldConfidence") or {}
    if field in fc and fc[field] is not None:
        return float(fc[field])
    # Fallback: legacy confidence.fields structure
    conf = data.get("confidence") or {}
    fields_conf = conf.get("fields") or {}
    val = fields_conf.get(field)
    return float(val) if val is not None else None


def _avg_confidence(data: dict) -> float:
    """Calculate average confidence for non-null fields."""
    scores = []
    for f in INVOICE_FIELDS:
        if data.get(f) is not None:
            c = _get_confidence(data, f)
            if c is not None:
                scores.append(float(c))
    if not scores:
        oc = (data.get("confidence") or {}).get("overall")
        return float(oc) if oc is not None else 0.0
    return sum(scores) / len(scores)


def _found_fields(data: dict) -> list[str]:
    return [f for f in INVOICE_FIELDS if data.get(f) is not None]


def compare(
    sap_raw: dict,
    llm_prompting: dict,
    llm_structured: dict,
) -> dict[str, Any]:
    """
    Compare the three extraction results and generate a differences report.

    Returns:
        Dictionary with statistics and detected differences.
    """
    sap = _normalize_sap(sap_raw)
    p1 = llm_prompting
    p2 = llm_structured

    sap_fields = set(_found_fields(sap))
    p1_fields  = set(_found_fields(p1))
    p2_fields  = set(_found_fields(p2))
    all_fields = sap_fields | p1_fields | p2_fields

    conflicts: list[dict] = []
    agreements: list[str] = []
    only_sap: list[str] = []
    only_llm: list[str] = []

    for field in sorted(all_fields):
        v_sap = sap.get(field)
        v_p1  = p1.get(field)
        v_p2  = p2.get(field)

        has_sap = v_sap is not None
        has_llm = v_p1 is not None or v_p2 is not None

        if has_sap and not has_llm:
            only_sap.append(field)
        elif has_llm and not has_sap:
            only_llm.append(field)
        elif has_sap and has_llm:
            # Compare values (as strings for tolerance)
            s_sap = str(v_sap).strip().lower()
            s_p1  = str(v_p1).strip().lower() if v_p1 is not None else None
            s_p2  = str(v_p2).strip().lower() if v_p2 is not None else None

            if s_p1 == s_sap or s_p2 == s_sap:
                agreements.append(field)
            else:
                conflicts.append({
                    "field": field,
                    "sap": v_sap,
                    "llm_prompting": v_p1,
                    "llm_structured": v_p2,
                })

    report = {
        "summary": {
            "sap_fields_found": len(sap_fields),
            "sap_confidence_avg": round(_avg_confidence(sap), 3),
            "llm_prompting_fields_found": len(p1_fields),
            "llm_prompting_confidence_avg": round(_avg_confidence(p1), 3),
            "llm_structured_fields_found": len(p2_fields),
            "llm_structured_confidence_avg": round(_avg_confidence(p2), 3),
            "total_unique_fields": len(all_fields),
            "agreements": len(agreements),
            "conflicts": len(conflicts),
            "only_in_sap": len(only_sap),
            "only_in_llm": len(only_llm),
        },
        "agreements": agreements,
        "conflicts": conflicts,
        "only_in_sap": only_sap,
        "only_in_llm": only_llm,
        "sap_normalized": sap,
    }

    logger.info(
        "Comparison completed. Agreements: %d | Conflicts: %d | SAP only: %d | LLM only: %d",
        len(agreements), len(conflicts), len(only_sap), len(only_llm),
    )
    return report