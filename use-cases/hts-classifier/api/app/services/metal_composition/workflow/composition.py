"""Helpers for presenting PDF-derived metal composition results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .normalize import _normalize_float
from .types import MetalCompositionState

TOP_LEVEL_METALS = ("steel", "aluminum", "copper", "cast_iron")
TOP_LEVEL_METAL_LABELS = {
    "steel": "Steel",
    "aluminum": "Aluminum",
    "copper": "Copper",
    "cast_iron": "Cast Iron",
}
STEEL_SUBTYPES = (
    "electrical_steel",
    "cold_rolled_coil_steel",
    "hot_rolled_coil_steel",
    "stainless_steel_304",
    "stainless_steel_316",
    "stainless_steel_bar",
    "duplex_steel",
    "cast_steel",
)
STEEL_SUBTYPE_LABELS = {
    "electrical_steel": "Electrical steel",
    "cold_rolled_coil_steel": "Cold rolled coil steel",
    "hot_rolled_coil_steel": "Hot rolled coil steel",
    "stainless_steel_304": "Stainless steel 304",
    "stainless_steel_316": "Stainless steel 316",
    "stainless_steel_bar": "Stainless steel bar",
    "duplex_steel": "Duplex steel",
    "cast_steel": "Cast steel",
}


def _normalized_top_level(payload: Dict[str, Any]) -> Dict[str, float]:
    return {
        metal: _normalize_float(payload.get(metal, 0.0))
        for metal in TOP_LEVEL_METALS
    }


def _empty_steel_subtypes() -> Dict[str, float]:
    return {subtype: 0.0 for subtype in STEEL_SUBTYPES}


def _normalized_steel_subtypes(payload: Dict[str, Any]) -> Dict[str, float]:
    subtypes = _empty_steel_subtypes()
    for subtype in STEEL_SUBTYPES:
        subtypes[subtype] = _normalize_float(payload.get(subtype, 0.0))
    return subtypes


def _sum_values(payload: Dict[str, float]) -> float:
    return float(sum(_normalize_float(value) for value in payload.values()))


def _normalize_source_status(source: str, *, weight_grams: float) -> str:
    if weight_grams <= 0.0:
        return "none"
    normalized = str(source or "").strip().lower()
    if normalized == "gcc_tracker":
        return "gcc_tracker"
    if normalized == "manual":
        return "manual"
    if normalized in {"human_review_pending", "pdf_only_human_review"}:
        return "needs_review"
    if normalized in {"none", ""}:
        return "estimated"
    return "estimated"


def _format_weight_for_sentence(value_grams: float) -> str:
    grams = _normalize_float(value_grams)
    if grams >= 1000.0:
        kilograms = round(grams / 1000.0, 3)
        if kilograms.is_integer():
            return f"{int(kilograms)} kg"
        return f"{kilograms:g} kg"
    if float(grams).is_integer():
        return f"{int(grams)} g"
    return f"{round(grams, 3):g} g"


def _source_document_entry(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    filename = str(item.get("source_filename") or item.get("filename") or "").strip()
    if not filename:
        return None
    page_number = item.get("page_number")
    try:
        page_number = int(page_number)
    except (TypeError, ValueError):
        page_number = None
    if page_number is not None and page_number <= 0:
        page_number = None
    return {
        "filename": filename,
        "page_number": page_number,
    }


def _dedupe_source_documents(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        entry = _source_document_entry(item)
        if entry is None:
            continue
        key = (entry["filename"], entry.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _explicit_evidence_for_item(values: Any) -> List[Dict[str, Any]]:
    evidence_items: List[Dict[str, Any]] = []
    for item in list(values or []):
        if not isinstance(item, dict):
            continue
        if not item.get("applies_to_current_item"):
            continue
        if not item.get("is_explicit", True):
            continue
        evidence_items.append(dict(item))
    return evidence_items


def _source_documents_for_top_level(
    *,
    metal: str,
    row_weight_grams: float,
    diagram_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    matched_weight_grams = _normalize_float(diagram_output.get("matched_weight_grams") or 0.0)
    matched_metal_type = str(diagram_output.get("matched_metal_type") or "").strip().lower()
    if (
        matched_weight_grams <= 0.0
        or matched_metal_type != metal
        or abs(matched_weight_grams - row_weight_grams) > 1e-3
    ):
        return []

    documents: List[Dict[str, Any]] = []
    material_evidence = _explicit_evidence_for_item(diagram_output.get("material_evidence"))
    for item in material_evidence:
        if str(item.get("normalized_metal") or "").strip().lower() == metal:
            documents.append(item)

    documents.extend(_explicit_evidence_for_item(diagram_output.get("weight_evidence")))

    return _dedupe_source_documents(documents)


def _source_documents_for_steel_subtype(
    *,
    steel_subtype: str,
    row_weight_grams: float,
    diagram_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    matched_weight_grams = _normalize_float(diagram_output.get("matched_weight_grams") or 0.0)
    matched_metal_type = str(diagram_output.get("matched_metal_type") or "").strip().lower()
    matched_steel_subtype = str(diagram_output.get("matched_steel_subtype") or "").strip().lower()
    if (
        row_weight_grams <= 0.0
        or matched_weight_grams <= 0.0
        or matched_metal_type != "steel"
        or matched_steel_subtype != steel_subtype
        or abs(matched_weight_grams - row_weight_grams) > 1e-3
    ):
        return []

    documents: List[Dict[str, Any]] = []
    material_evidence = _explicit_evidence_for_item(diagram_output.get("material_evidence"))
    for item in material_evidence:
        if str(item.get("normalized_steel_subtype") or "").strip().lower() == steel_subtype:
            documents.append(item)
    documents.extend(_explicit_evidence_for_item(diagram_output.get("weight_evidence")))
    return _dedupe_source_documents(documents)


def _material_documents_for_metal(
    *,
    metal: str,
    diagram_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    material_evidence = _explicit_evidence_for_item(diagram_output.get("material_evidence"))
    for item in material_evidence:
        if str(item.get("normalized_metal") or "").strip().lower() == metal:
            documents.append(item)
    return _dedupe_source_documents(documents)


def _build_presentation_rows(
    *,
    final_payload: Dict[str, Any],
    diagram_output: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    provenance = dict(final_payload.get("provenance") or {})
    top_level_sources = dict(provenance.get("top_level_sources") or {})
    steel_subtype_sources = dict(provenance.get("steel_subtype_sources") or {})

    metal_rows: List[Dict[str, Any]] = []
    for metal in TOP_LEVEL_METALS:
        weight_grams = _normalize_float((final_payload.get("top_level_grams", {}) or {}).get(metal, 0.0))
        source_documents = _source_documents_for_top_level(
            metal=metal,
            row_weight_grams=weight_grams,
            diagram_output=diagram_output,
        )
        source_status = "documented" if source_documents else _normalize_source_status(
            str(top_level_sources.get(metal) or "none"),
            weight_grams=weight_grams,
        )
        metal_rows.append(
            {
                "type": TOP_LEVEL_METAL_LABELS[metal],
                "weight_grams": weight_grams,
                "source_documents": source_documents,
                "source_status": source_status,
            }
        )

    steel_subtype_rows: List[Dict[str, Any]] = []
    for steel_subtype in STEEL_SUBTYPES:
        weight_grams = _normalize_float((final_payload.get("steel_subtype_grams", {}) or {}).get(steel_subtype, 0.0))
        source_documents = _source_documents_for_steel_subtype(
            steel_subtype=steel_subtype,
            row_weight_grams=weight_grams,
            diagram_output=diagram_output,
        )
        source_status = "documented" if source_documents else _normalize_source_status(
            str(steel_subtype_sources.get(steel_subtype) or "none"),
            weight_grams=weight_grams,
        )
        steel_subtype_rows.append(
            {
                "type": STEEL_SUBTYPE_LABELS[steel_subtype],
                "weight_grams": weight_grams,
                "source_documents": source_documents,
                "source_status": source_status,
            }
        )

    return metal_rows, steel_subtype_rows


def _augment_final_payload_for_ui(
    *,
    final_payload: Dict[str, Any],
    state: MetalCompositionState,
) -> Dict[str, Any]:
    augmented = dict(final_payload)
    diagram_output = dict(state.get("diagram_output") or {})
    metal_rows, steel_subtype_rows = _build_presentation_rows(
        final_payload=augmented,
        diagram_output=diagram_output,
    )
    augmented["metal_rows"] = metal_rows
    augmented["steel_subtype_rows"] = steel_subtype_rows
    return augmented
