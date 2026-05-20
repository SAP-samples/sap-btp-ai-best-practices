"""Mass-balance validation and repair for final metal composition output."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.metal_composition import CompositionEvidenceProvenance, FinalMetalComposition

from .normalize import _normalize_confidence, _normalize_float


def _clamp_non_negative(payload: Dict[str, Any]) -> Dict[str, float]:
    return {key: max(0.0, _normalize_float(value)) for key, value in payload.items()}


def validate_and_repair_final_composition(
    raw_payload: Dict[str, Any],
    *,
    total_weight_grams: Optional[float],
) -> FinalMetalComposition:
    """Validate the final node output and repair simple mass-balance issues."""

    repair_notes: List[str] = []
    total_weight = None if total_weight_grams is None else max(0.0, float(total_weight_grams))

    top_level = _clamp_non_negative(
        {
            "steel": raw_payload.get("top_level_grams", {}).get("steel", 0.0),
            "aluminum": raw_payload.get("top_level_grams", {}).get("aluminum", 0.0),
            "copper": raw_payload.get("top_level_grams", {}).get("copper", 0.0),
            "cast_iron": raw_payload.get("top_level_grams", {}).get("cast_iron", 0.0),
        }
    )
    subtypes = _clamp_non_negative(
        {
            "electrical_steel": raw_payload.get("steel_subtype_grams", {}).get("electrical_steel", 0.0),
            "cold_rolled_coil_steel": raw_payload.get("steel_subtype_grams", {}).get(
                "cold_rolled_coil_steel", 0.0
            ),
            "hot_rolled_coil_steel": raw_payload.get("steel_subtype_grams", {}).get(
                "hot_rolled_coil_steel", 0.0
            ),
            "stainless_steel_304": raw_payload.get("steel_subtype_grams", {}).get(
                "stainless_steel_304", 0.0
            ),
            "stainless_steel_316": raw_payload.get("steel_subtype_grams", {}).get(
                "stainless_steel_316", 0.0
            ),
            "stainless_steel_bar": raw_payload.get("steel_subtype_grams", {}).get(
                "stainless_steel_bar", 0.0
            ),
            "duplex_steel": raw_payload.get("steel_subtype_grams", {}).get("duplex_steel", 0.0),
            "cast_steel": raw_payload.get("steel_subtype_grams", {}).get("cast_steel", 0.0),
        }
    )

    top_level_total = sum(top_level.values())
    if total_weight is not None and top_level_total > total_weight and top_level_total > 0.0:
        scale = total_weight / top_level_total
        top_level = {key: value * scale for key, value in top_level.items()}
        repair_notes.append(
            f"Scaled top-level grams by {scale:.4f} to keep predicted metal <= total weight."
        )

    subtype_total = sum(subtypes.values())
    steel_total = top_level["steel"]
    if subtype_total > steel_total and subtype_total > 0.0:
        scale = steel_total / subtype_total if steel_total > 0.0 else 0.0
        subtypes = {key: value * scale for key, value in subtypes.items()}
        repair_notes.append(
            f"Scaled steel subtype grams by {scale:.4f} to keep subtype total <= steel grams."
        )

    estimated_total_metal = sum(top_level.values())
    reasoning = str(raw_payload.get("reasoning", "") or "").strip()
    if repair_notes:
        note = "Validation repairs applied: " + " ".join(repair_notes)
        reasoning = f"{reasoning} {note}".strip()
    if not reasoning:
        reasoning = "No reasoning supplied by the final combiner."

    confidence = _normalize_confidence(raw_payload.get("confidence", 0.0))

    return FinalMetalComposition(
        is_metal_item=bool(raw_payload.get("is_metal_item", estimated_total_metal > 0.0)),
        total_weight_grams=total_weight,
        estimated_total_metal_grams=float(estimated_total_metal),
        top_level_grams=top_level,
        steel_subtype_grams=subtypes,
        provenance=CompositionEvidenceProvenance.model_validate(raw_payload.get("provenance") or {}),
        confidence=confidence,
        reasoning=reasoning,
    )
