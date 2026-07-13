"""Diagram / PDF image processing and multi-image vision analysis."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from ..config import MetalCompositionSettings
from ..timing import finish_timing, utc_now_iso
from .llm import LLMClient
from .normalize import _extract_json_payload, _normalize_confidence, _normalize_float
from .page_routing import build_mixed_diagram_batches
from .page_routing import materialize_diagram_sources
from .token_usage import TokenUsageRecorder
from .types import DiagramPayload, MaterializedDiagramSources, MixedDiagramBatchEntry, RenderedDiagramPage, RenderedDiagramTextPage, ZoomedDiagramCrop

logger = logging.getLogger(__name__)

_WEIGHT_PATTERN = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>kg|kgs|kilogram|kilograms|g|gram|grams)\b", re.I)
_VALID_METALS = {"steel", "aluminum", "copper", "cast_iron"}
_VALID_STEEL_SUBTYPES = {
    "electrical_steel",
    "cold_rolled_coil_steel",
    "hot_rolled_coil_steel",
    "stainless_steel_304",
    "stainless_steel_316",
    "stainless_steel_bar",
    "duplex_steel",
    "cast_steel",
}
_VALID_METAL_SHARE_CERTAINTY = {"exact", "estimated"}
_MIN_ZOOM_BOX_FRACTION = 0.01
_MAX_PROVIDER_IMAGES_PER_REQUEST = 50


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_weight_unit(unit: Any) -> Optional[str]:
    text = str(unit or "").strip().lower()
    if not text:
        return None
    if text in {"kg", "kgs", "kilogram", "kilograms"}:
        return "kg"
    if text in {"g", "gram", "grams"}:
        return "g"
    return None


def _normalize_weight_grams(evidence: Dict[str, Any]) -> Optional[float]:
    normalized = _coerce_float(evidence.get("normalized_grams"))
    if normalized is not None and normalized >= 0.0:
        return round(normalized, 3)

    numeric_value = _coerce_float(evidence.get("numeric_value"))
    normalized_unit = _normalize_weight_unit(evidence.get("unit"))
    if numeric_value is not None and normalized_unit is not None:
        factor = 1000.0 if normalized_unit == "kg" else 1.0
        return round(numeric_value * factor, 3)

    raw_weight = str(
        evidence.get("raw_weight")
        or evidence.get("raw_value")
        or evidence.get("source_excerpt")
        or ""
    )
    match = _WEIGHT_PATTERN.search(raw_weight)
    if match is None:
        return None
    numeric_value = _coerce_float(match.group("value"))
    normalized_unit = _normalize_weight_unit(match.group("unit"))
    if numeric_value is None or normalized_unit is None:
        return None
    factor = 1000.0 if normalized_unit == "kg" else 1.0
    return round(numeric_value * factor, 3)


def _normalize_metal_type(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", " ").replace("/", " ")
    if not text:
        return None
    if text in _VALID_METALS:
        return text
    if "cast iron" in text:
        return "cast_iron"
    if "aluminium" in text or "aluminum" in text:
        return "aluminum"
    if "copper" in text:
        return "copper"
    if "steel" in text or "stainless" in text or "duplex" in text:
        return "steel"
    return None


def _normalize_steel_subtype(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", " ").replace("/", " ")
    if not text:
        return None
    enum_like = text.replace(" ", "_")
    if enum_like in _VALID_STEEL_SUBTYPES:
        return enum_like
    if "stainless" in text and "304" in text:
        return "stainless_steel_304"
    if "stainless" in text and "316" in text:
        return "stainless_steel_316"
    if "stainless" in text and "bar" in text:
        return "stainless_steel_bar"
    if "electrical" in text:
        return "electrical_steel"
    if "cold rolled" in text:
        return "cold_rolled_coil_steel"
    if "hot rolled" in text:
        return "hot_rolled_coil_steel"
    if "duplex" in text:
        return "duplex_steel"
    if "cast steel" in text:
        return "cast_steel"
    return None


def _normalized_identifier_variants(product_code: Any) -> List[str]:
    text = str(product_code or "").strip()
    if not text:
        return []
    digits_only = re.sub(r"\D+", "", text)
    variants = [text]
    if digits_only and digits_only != text:
        variants.append(digits_only)
    return variants


def _safe_evidence_items(values: Any) -> List[Dict[str, Any]]:
    return [item for item in list(values or []) if isinstance(item, dict)]


def _safe_dict_items(values: Any) -> List[Dict[str, Any]]:
    return [item for item in list(values or []) if isinstance(item, dict)]


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, list):
        return {}

    merged: Dict[str, Any] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("composition"), dict):
            merged.update(dict(item["composition"]))
            continue
        key = item.get("key")
        if isinstance(key, str) and key.strip():
            merged[key.strip()] = item.get("value")
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            merged[name.strip()] = item.get("value")
            continue
        merged.update(item)
    return merged


def _coerce_named_value_mapping(value: Any, *, allowed_keys: set[str]) -> Dict[str, float]:
    if isinstance(value, dict):
        return {
            key: _normalize_float(raw_value)
            for key, raw_value in value.items()
            if isinstance(key, str) and key in allowed_keys
        }
    if not isinstance(value, list):
        return {}

    normalized: Dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str) or key not in allowed_keys:
            key = item.get("name")
        if not isinstance(key, str) or key not in allowed_keys:
            continue
        normalized[key] = _normalize_float(item.get("value"))
    return normalized


def _normalize_composition_payload(value: Any) -> tuple[Dict[str, Any], bool]:
    normalized = _coerce_mapping(value)
    if not normalized:
        return {}, False

    was_coerced = not isinstance(value, dict)
    raw_top_level = normalized.get("top_level_grams")
    top_level = _coerce_named_value_mapping(
        raw_top_level,
        allowed_keys={"steel", "aluminum", "copper", "cast_iron"},
    )
    if top_level or raw_top_level is not None:
        normalized["top_level_grams"] = top_level
        was_coerced = was_coerced or not isinstance(raw_top_level, dict)

    raw_steel_subtypes = normalized.get("steel_subtype_grams")
    steel_subtypes = _coerce_named_value_mapping(
        raw_steel_subtypes,
        allowed_keys={
            "electrical_steel",
            "cold_rolled_coil_steel",
            "hot_rolled_coil_steel",
            "stainless_steel_304",
            "stainless_steel_316",
            "stainless_steel_bar",
            "duplex_steel",
            "cast_steel",
        },
    )
    if steel_subtypes or raw_steel_subtypes is not None:
        normalized["steel_subtype_grams"] = steel_subtypes
        was_coerced = was_coerced or not isinstance(raw_steel_subtypes, dict)

    return normalized, was_coerced


def _looks_like_material_standard(filename: Any) -> bool:
    name = Path(str(filename or "")).name.strip().lower()
    return bool(name and re.match(r"^m\d{4}", name))


def _weight_evidence_priority(candidate: Dict[str, Any]) -> tuple[int, int]:
    filename = str(candidate.get("source_filename") or candidate.get("filename") or "").strip()
    excerpt = str(
        candidate.get("source_excerpt")
        or candidate.get("raw_weight")
        or candidate.get("raw_value")
        or ""
    ).strip().lower()
    keyword_score = 0
    if "calculated weight" in excerpt:
        keyword_score += 2
    elif "weight" in excerpt or "mass" in excerpt:
        keyword_score += 1
    source_score = 0 if _looks_like_material_standard(filename) else 1
    return source_score, keyword_score


def _derive_matched_diagram_evidence(merged: Dict[str, Any]) -> None:
    weight_candidates: List[Dict[str, Any]] = []
    for evidence in _safe_evidence_items(merged.get("weight_evidence")):
        normalized_grams = _normalize_weight_grams(evidence)
        if normalized_grams is None:
            continue
        candidate = dict(evidence)
        candidate["normalized_grams"] = normalized_grams
        weight_candidates.append(candidate)

    explicit_weight_candidates = [
        candidate
        for candidate in weight_candidates
        if candidate.get("applies_to_current_item") and candidate.get("is_explicit", True)
    ]
    explicit_weight_candidates.sort(
        key=lambda candidate: (
            -_weight_evidence_priority(candidate)[0],
            -_weight_evidence_priority(candidate)[1],
            -_normalize_confidence(candidate.get("match_confidence", 0.0)),
            -_normalize_float(candidate.get("normalized_grams", 0.0)),
        )
    )
    best_weight = explicit_weight_candidates[0] if explicit_weight_candidates else None

    material_candidates: List[Dict[str, Any]] = []
    for evidence in _safe_evidence_items(merged.get("material_evidence")):
        candidate = dict(evidence)
        candidate["normalized_metal"] = _normalize_metal_type(
            evidence.get("normalized_metal") or evidence.get("raw_material")
        )
        candidate["normalized_steel_subtype"] = _normalize_steel_subtype(
            evidence.get("normalized_steel_subtype") or evidence.get("raw_material")
        )
        if candidate["normalized_metal"] is None and candidate["normalized_steel_subtype"] is not None:
            candidate["normalized_metal"] = "steel"
        material_candidates.append(candidate)

    explicit_material_candidates = [
        candidate
        for candidate in material_candidates
        if candidate.get("applies_to_current_item") and candidate.get("is_explicit", True)
    ]
    explicit_material_candidates.sort(
        key=lambda candidate: (
            -_normalize_confidence(candidate.get("match_confidence", 0.0)),
            0 if candidate.get("normalized_steel_subtype") else 1,
        )
    )
    best_material = explicit_material_candidates[0] if explicit_material_candidates else None

    merged["weight_evidence"] = weight_candidates
    merged["material_evidence"] = material_candidates
    merged["matched_weight_grams"] = (
        _normalize_float(best_weight["normalized_grams"]) if best_weight is not None else None
    )
    merged["matched_weight_evidence"] = best_weight or {}
    merged["matched_material_evidence"] = best_material or {}
    merged["matched_metal_type"] = (
        str(best_material.get("normalized_metal")) if best_material is not None and best_material.get("normalized_metal") else None
    )
    merged["matched_steel_subtype"] = (
        str(best_material.get("normalized_steel_subtype"))
        if best_material is not None and best_material.get("normalized_steel_subtype")
        else None
    )
    merged["matched_identifiers"] = [
        str(item)
        for item in list(merged.get("matched_identifiers") or [])
        if str(item or "").strip()
    ]


def _normalize_metal_share_certainty(
    value: Any,
    *,
    metal_share_basis: str,
    matched_weight_evidence: Dict[str, Any] | None,
) -> str:
    text = str(value or "").strip().lower()
    if text in _VALID_METAL_SHARE_CERTAINTY:
        return text

    # Backward-compatible fallback for older model responses that do not emit
    # certainty explicitly. We only infer "exact" for the strongest case:
    # explicit full-metal evidence tied to an item-specific explicit weight.
    if (
        str(metal_share_basis or "").strip().lower() == "explicit_full_metal"
        and _normalize_weight_grams(dict(matched_weight_evidence or {})) is not None
    ):
        return "exact"
    return "estimated"


def _diagram_source_label(diagram_payload: DiagramPayload) -> str:
    source_filename = str(diagram_payload.source_filename or diagram_payload.filename or "").strip()
    page_number = _coerce_int(diagram_payload.page_number)
    if source_filename and page_number and page_number > 0:
        return f"{source_filename} page {page_number}"
    return source_filename or diagram_payload.filename or "document image"


def _rendered_page_source_label(page: RenderedDiagramPage) -> str:
    base_label = _diagram_source_label(
        DiagramPayload(
            filename=page.filename,
            content_type=page.content_type,
            data=b"",
            source_filename=page.source_filename,
            page_number=page.page_number,
        )
    )
    return f"{page.page_ref} | {base_label}"


def _zoom_crop_source_label(crop: ZoomedDiagramCrop) -> str:
    base_label = _diagram_source_label(
        DiagramPayload(
            filename=crop.filename,
            content_type=crop.content_type,
            data=b"",
            source_filename=crop.source_filename,
            page_number=crop.page_number,
        )
    )
    box = crop.normalized_box
    return (
        f"{crop.crop_ref} | zoom from {crop.page_ref} | {base_label} | "
        f"normalized_box=[{box['x0']:.4f}, {box['y0']:.4f}, {box['x1']:.4f}, {box['y1']:.4f}]"
    )


def convert_pdf_to_images(payload: DiagramPayload, settings: MetalCompositionSettings) -> List[DiagramPayload]:
    """Convert a PDF payload to a list of per-page PNG DiagramPayloads.

    Non-PDF files pass through as a single-element list.
    """
    is_pdf = (
        payload.content_type == "application/pdf"
        or (payload.filename or "").lower().endswith(".pdf")
    )
    if not is_pdf:
        return [payload]

    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF diagram processing but is not installed."
        ) from None

    dpi = settings.pdf_render_dpi
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    image_payloads: List[DiagramPayload] = []

    try:
        with fitz.open(stream=payload.data, filetype="pdf") as doc:
            for page_number in range(len(doc)):
                page = doc[page_number]
                pixmap = page.get_pixmap(matrix=matrix)
                png_bytes = pixmap.tobytes("png")
                stem = payload.filename.rsplit(".", 1)[0] if "." in payload.filename else payload.filename
                image_payloads.append(
                    DiagramPayload(
                        filename=f"{stem}_page_{page_number + 1}.png",
                        content_type="image/png",
                        data=png_bytes,
                        source_filename=payload.source_filename or payload.filename,
                        page_number=page_number + 1,
                    )
                )
    except Exception:
        logger.exception("Failed to convert PDF %s to images", payload.filename)
        return [payload]

    return image_payloads if image_payloads else [payload]


def render_diagram_pages(
    diagram_payloads: List[DiagramPayload],
    settings: MetalCompositionSettings,
) -> tuple[List[RenderedDiagramPage], List[Dict[str, Any]]]:
    """Render and preprocess raw diagram payloads into stable page/image objects."""

    rendered_pages: List[RenderedDiagramPage] = []
    preprocess_details_list: List[Dict[str, Any]] = []
    page_counter = 0

    for source_document_index, payload in enumerate(diagram_payloads):
        for image_payload in convert_pdf_to_images(payload, settings):
            processed, details = preprocess_diagram_payload(image_payload, settings)
            page_counter += 1
            processed_size = list(details.get("processed_size") or [])
            rendered_pages.append(
                RenderedDiagramPage(
                    page_ref=f"P{page_counter}",
                    source_document_index=source_document_index,
                    filename=processed.filename,
                    content_type=processed.content_type,
                    data=processed.data,
                    source_filename=processed.source_filename,
                    page_number=processed.page_number,
                    rendered_width=_coerce_int(processed_size[0]) or 0 if len(processed_size) > 0 else 0,
                    rendered_height=_coerce_int(processed_size[1]) or 0 if len(processed_size) > 1 else 0,
                    input_payload=payload,
                )
            )
            preprocess_details_list.append(
                {
                    **details,
                    "page_ref": f"P{page_counter}",
                    "source_document_index": source_document_index,
                    "source_filename": processed.source_filename or payload.filename,
                    "page_number": processed.page_number,
                }
            )

    return rendered_pages, preprocess_details_list


def _preprocess_diagram_payload_with_limits(
    diagram_payload: DiagramPayload,
    *,
    max_dimension: int,
    max_bytes: int,
) -> tuple[DiagramPayload, Dict[str, Any]]:
    details = {
        "original_bytes": len(diagram_payload.data),
        "original_content_type": diagram_payload.content_type,
    }
    try:
        from PIL import Image
    except ModuleNotFoundError:
        details["preprocessing"] = "skipped_missing_pillow"
        return diagram_payload, details

    max_dimension = max(1, int(max_dimension))
    max_bytes = max(1, int(max_bytes))

    with Image.open(BytesIO(diagram_payload.data)) as image:
        image.load()
        details["original_size"] = [int(image.width), int(image.height)]

        if max(image.width, image.height) > max_dimension:
            scale = max_dimension / float(max(image.width, image.height))
            image = image.resize(
                (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
                Image.Resampling.LANCZOS,
            )
            details["resized"] = True
        else:
            details["resized"] = False

        processed_buffer = BytesIO()
        target_content_type = diagram_payload.content_type or "image/png"
        preserve_png = diagram_payload.content_type == "image/png" or image.mode in {"RGBA", "LA"}
        if preserve_png:
            image.save(processed_buffer, format="PNG", optimize=True)
            target_content_type = "image/png"
        else:
            rgb_image = image.convert("RGB") if image.mode not in {"RGB", "L"} else image
            for quality in (85, 75, 65):
                processed_buffer = BytesIO()
                rgb_image.save(processed_buffer, format="JPEG", optimize=True, quality=quality)
                if processed_buffer.tell() <= max_bytes or quality == 65:
                    break
            target_content_type = "image/jpeg"

        if processed_buffer.tell() > max_bytes and preserve_png and image.mode not in {"RGBA", "LA"}:
            rgb_image = image.convert("RGB")
            for quality in (85, 75, 65):
                processed_buffer = BytesIO()
                rgb_image.save(processed_buffer, format="JPEG", optimize=True, quality=quality)
                if processed_buffer.tell() <= max_bytes or quality == 65:
                    break
            target_content_type = "image/jpeg"

        processed_bytes = processed_buffer.getvalue()
        details["processed_bytes"] = len(processed_bytes)
        details["processed_content_type"] = target_content_type
        details["processed_size"] = [int(image.width), int(image.height)]
        return (
            DiagramPayload(
                filename=diagram_payload.filename,
                content_type=target_content_type,
                data=processed_bytes,
                source_filename=diagram_payload.source_filename,
                page_number=diagram_payload.page_number,
            ),
            details,
        )


def preprocess_diagram_payload(
    diagram_payload: DiagramPayload, settings: MetalCompositionSettings
) -> tuple[DiagramPayload, Dict[str, Any]]:
    is_pdf_derived_page = bool(
        str(diagram_payload.source_filename or "").strip().lower().endswith(".pdf")
        and _coerce_int(diagram_payload.page_number)
    )
    max_dimension = max(
        1,
        int(
            settings.pdf_image_max_dimension
            if is_pdf_derived_page
            else settings.image_max_dimension
        ),
    )
    max_bytes = max(
        1,
        int(
            settings.pdf_image_max_bytes
            if is_pdf_derived_page
            else settings.image_max_bytes
        ),
    )
    processed, details = _preprocess_diagram_payload_with_limits(
        diagram_payload,
        max_dimension=max_dimension,
        max_bytes=max_bytes,
    )
    details["pdf_derived_page"] = is_pdf_derived_page
    return processed, details


def _build_visual_batches(
    visuals: List[Any],
    *,
    max_images: int,
    max_bytes: int,
) -> List[List[Any]]:
    max_images = max(1, min(int(max_images), _MAX_PROVIDER_IMAGES_PER_REQUEST))
    batches: List[List[Any]] = []
    current_batch: List[Any] = []
    current_batch_bytes = 0

    for visual in visuals:
        encoded_size = len(visual.data) * 4 // 3 + 4
        if current_batch and (
            len(current_batch) >= max_images
            or current_batch_bytes + encoded_size > max_bytes
        ):
            batches.append(current_batch)
            current_batch = []
            current_batch_bytes = 0
        current_batch.append(visual)
        current_batch_bytes += encoded_size

    if current_batch:
        batches.append(current_batch)

    return batches


def _diagram_output_defaults() -> Dict[str, Any]:
    return {
        "status": "completed",
        "extracted_codes": [],
        "is_likely_metal_item": False,
        "estimated_metal_share": 0.0,
        "metal_share_certainty": "estimated",
        "metal_share_basis": "unknown",
        "non_metal_evidence": [],
        "metal_share_reasoning": "",
        "matched_identifiers": [],
        "weight_evidence": [],
        "material_evidence": [],
        "material_cues": [],
        "material_properties": [],
        "context_of_use": "",
        "hts_hints": [],
        "uncertainty_notes": [],
        "provenance_flags": [],
    }


def _finalize_diagram_result(merged: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(merged)
    for key, value in _diagram_output_defaults().items():
        normalized.setdefault(key, value if not isinstance(value, list) else list(value))
    _derive_matched_diagram_evidence(normalized)
    normalized["metal_share_certainty"] = _normalize_metal_share_certainty(
        normalized.get("metal_share_certainty"),
        metal_share_basis=str(normalized.get("metal_share_basis") or "unknown"),
        matched_weight_evidence=dict(normalized.get("matched_weight_evidence") or {}),
    )
    return normalized


def _extract_item_context(
    *,
    product_code: Optional[str],
    source_summary: Optional[Dict[str, Any]],
    source_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "product_code": product_code,
        "product_code_variants": _normalized_identifier_variants(product_code),
        "pn_revised_standardized": (source_summary or {}).get("pn_revised_standardized")
        or (source_row or {}).get("PN Revised/ Standardized"),
        "part_description": (source_summary or {}).get("part_description")
        or (source_row or {}).get("Part description"),
        "new_part_description": (source_summary or {}).get("new_part_description")
        or (source_row or {}).get("New Part Description"),
        "total_weight_gram": (source_summary or {}).get("total_weight_gram")
        or (source_row or {}).get("Total Weight (Gram)"),
    }


def _build_base_diagram_prompt(item_context: Dict[str, Any]) -> str:
    return (
        "You are analyzing one or more engineering documents (2D drawings, material standards, "
        "material certificates, specifications). These documents may span multiple source pages.\n\n"
        f"Current item context JSON:\n{json.dumps(item_context, default=str)}\n\n"
        "Each source is preceded by a source label that includes a stable page_ref token plus the original "
        "PDF filename and page when available, for example `P12 | file.pdf page 3`.\n"
        "Some sources are images. Some sources are extracted page text and appear as `Source text: ...` blocks.\n"
        "When you cite evidence, copy the source_filename and page_number into the evidence object.\n\n"
        "When matching tables or callouts to the current item, treat product/variant identifiers as equivalent even "
        "if the document inserts spaces, dots, slashes, or labels such as `Variant`. For example, `2797602` should "
        "be treated as the same identifier as `279 76 02`.\n\n"
        "Any explanatory text fields you return must be user-facing. Do not mention internal variable names, enum "
        "tokens, workflow labels, prompt categories, or implementation details. Avoid phrases such as "
        "`explicit_full_metal` or `inferred_full_metal` in narrative text. Do not use "
        "speculative company-practice language such as `typical company practice`, `usually`, or `commonly`.\n\n"
        "Primary goals (material-focused):\n"
        "1. Identify whether any table row, note, or callout explicitly matches the current item context.\n"
        "2. Extract explicit weight evidence for the matched item and normalize kilograms to grams.\n"
        "3. Identify materials and alloys for the matched item (e.g. stainless steel 316L, aluminum 6061, cast iron EN-GJL-250)\n"
        "4. Extract material standard codes (DIN, ASTM, EN, ISO, JIS references)\n"
        "5. Identify material properties where stated (tensile strength, hardness, chemical composition "
        "percentages, alloy grades, heat treatment conditions)\n"
        "6. Determine what the item is (part type, function) and what product/assembly it belongs to\n\n"
        "Secondary goal (soft guidance only):\n"
        "7. Based on the material and article evidence, suggest which HTS chapters might be relevant. "
        "These are hints only, not definitive classifications.\n\n"
        "Rules for estimated_metal_share:\n"
        "1. Return 1.0 when the supplied item is shown as a single-piece metal part and the documents name a "
        "metal/alloy/material standard, with no explicit evidence of non-metal mass in the supplied item.\n"
        "2. Return less than 1.0 only when the documents explicitly show or state non-metal or secondary-material "
        "content in the supplied item: seals, O-rings, overmolds, bonded pads, foams, electronics, glass, rubber, "
        "visible inserts, attached subcomponents, or a multi-part assembly.\n"
        "3. Do not reduce the share for speculative paint, labels, coatings, packaging, or generic conservatism unless "
        "the documents clearly indicate they are part of the shipped item and materially affect mass.\n"
        "4. If you are uncertain but see no explicit non-metal evidence, keep estimated_metal_share at 1.0 and record "
        "the uncertainty in uncertainty_notes instead of shaving mass.\n\n"
        "Rules for metal_share_certainty:\n"
        '1. Return "exact" only when the PDFs give enough item-specific evidence to determine the metal share without estimation.\n'
        '2. Return "estimated" whenever any engineering judgment, proportional allocation, or assumption is needed.\n'
        '3. If the item is obviously fully metallic from the drawings but the exact share is not explicitly proven, return "estimated".\n\n'
        "Rules for weight evidence:\n"
        "1. Record weight evidence only when the page itself explicitly states a numeric weight for the matched current item.\n"
        "2. A material standard, alloy sheet, or internal standard page does not count as weight evidence unless it explicitly "
        "lists the matched item together with a numeric weight.\n"
        "3. When a direct drawing, title block, margin note, or `Calculated weight` callout provides an item-specific weight, "
        "prefer that source over general standards or indirect references.\n"
        "4. If the text is unreadable or ambiguous, do not guess the weight. Leave it unmatched and add a concrete uncertainty note.\n\n"
        "Zoom refinement rules:\n"
        "If tiny or unreadable text is materially blocking matched weight, matched identifier, material designation, or "
        "material-standard extraction, you MUST request at least one targeted zoom region for the relevant page_ref.\n"
        "Only image sources are zoom-eligible. Do not request zoom regions for source pages that were provided as extracted text.\n"
        "Do not both claim that blocking text is too small or unreadable and return zoom_requests as [].\n"
        "Return zoom_requests as [] only when there is no materially blocking tiny text or when the field is already resolved "
        "with sufficient evidence from the full-page views.\n"
        "Only request zooms when they are necessary. At most six requests across the analysis.\n"
        "Each zoom request must reference the stable page_ref and provide normalized_box as [x0, y0, x1, y1] in 0..1 "
        "relative to the full page image you were shown.\n\n"
        "Return STRICT JSON with keys:\n"
        '"status": "completed",\n'
        '"extracted_codes": array of material standard codes found (e.g. "DIN EN 10025-2", "ASTM A36"),\n'
        '"is_likely_metal_item": boolean,\n'
        '"estimated_metal_share": number between 0 and 1,\n'
        '"metal_share_certainty": one of "exact" or "estimated",\n'
        '"metal_share_basis": one of "explicit_full_metal", "explicit_partial_non_metal", '
        '"inferred_full_metal", "inferred_partial_non_metal", "unknown",\n'
        '"non_metal_evidence": array of short concrete strings describing any explicit non-metal evidence; empty if none,\n'
        '"metal_share_reasoning": one short sentence explaining why the share is 1.0 or why it is below 1.0,\n'
        '"matched_identifiers": array of strings for identifiers in the documents that explicitly match the current item,\n'
        '"weight_evidence": array of objects with keys '
        '"identifier", "applies_to_current_item", "raw_weight", "numeric_value", "unit", "normalized_grams", '
        '"source_excerpt", "source_filename", "page_number", "is_explicit", "match_confidence",\n'
        '"material_evidence": array of objects with keys '
        '"identifier", "applies_to_current_item", "raw_material", "normalized_metal", '
        '"normalized_steel_subtype", "source_excerpt", "source_filename", "page_number", "is_explicit", "match_confidence",\n'
        '"material_cues": array of short strings describing materials found,\n'
        '"material_properties": array of objects with "property" and "value" keys '
        '(e.g. {"property": "tensile_strength_mpa", "value": "470-630"}),\n'
        '"context_of_use": string describing the item, its function, and what assembly it belongs to,\n'
        '"hts_hints": array of objects with "chapter" (int) and "rationale" (string), '
        "suggesting potentially relevant HTS chapters. These are non-binding suggestions,\n"
        '"uncertainty_notes": array of short strings noting any ambiguities. Use only concrete document issues such as unreadable '
        'text or conflicting callouts. Do not mention company practice or typical conventions,\n'
        '"provenance_flags": array of short strings describing whether the final cues are explicit PDF facts or inference,\n'
        '"zoom_requests": optional array of objects with keys "page_ref", "reason", "normalized_box". '
        'Return [] when no zoom is needed,\n\n'
        "Metal composition estimate:\n"
        "Using ALL the material evidence you identified above, the item context (especially total_weight_gram), "
        "and the estimated_metal_share, produce a gram-level metal composition breakdown. "
        "If the documents identify multiple metals for the same item (e.g. an aluminum housing with stainless-steel "
        "thread inserts), allocate weight to each metal proportionally based on the engineering context: "
        "a main casting or body receives the bulk of the weight, while small components such as inserts, fasteners, "
        "or fittings receive a reasonable engineering estimate (e.g. a few percent of total weight). "
        "Map steel alloys to their subtypes: A2/A4 stainless = stainless_steel_304/316, "
        "EN 10025 structural = hot_rolled_coil_steel, cold-drawn bar = stainless_steel_bar, "
        "duplex (2205/2507) = duplex_steel, cast steel (e.g. GP240GH) = cast_steel, "
        "electrical steel (grain-oriented/non-oriented) = electrical_steel, "
        "cold-rolled coil = cold_rolled_coil_steel. "
        "If you cannot determine the exact subtype, use the most likely match and note the uncertainty.\n\n"
        'Include these additional keys in the same JSON response:\n'
        '"composition": an object with keys:\n'
        '  "is_metal_item": boolean,\n'
        '  "total_weight_grams": the total_weight_gram from the item context (or the explicit weight from the drawing if found),\n'
        '  "estimated_total_metal_grams": total_weight_grams * estimated_metal_share,\n'
        '  "top_level_grams": object with keys "steel", "aluminum", "copper", "cast_iron" (values in grams, must sum to estimated_total_metal_grams),\n'
        '  "steel_subtype_grams": object with keys "electrical_steel", "cold_rolled_coil_steel", "hot_rolled_coil_steel", '
        '"stainless_steel_304", "stainless_steel_316", "stainless_steel_bar", "duplex_steel", "cast_steel" '
        '(values in grams, must sum to the steel value in top_level_grams),\n'
        '  "confidence": numeric float 0.0-1.0,\n'
        '  "reasoning": one paragraph in plain user-facing language explaining the material allocation.'
    )


def _build_refinement_diagram_prompt(
    *,
    item_context: Dict[str, Any],
    provisional_result: Dict[str, Any],
    zoom_requests: List[Dict[str, Any]],
) -> str:
    return (
        "You are refining a previous engineering-document analysis using targeted zoom crops from the same PDFs/pages.\n\n"
        f"Current item context JSON:\n{json.dumps(item_context, default=str)}\n\n"
        f"Provisional base-pass JSON:\n{json.dumps(provisional_result, default=str)}\n\n"
        f"Executed zoom requests JSON:\n{json.dumps(zoom_requests, default=str)}\n\n"
        "You will see the affected full pages again, plus high-resolution crops labelled with crop refs and the source page_ref. "
        "Use the zoom crops to resolve unreadable or ambiguous text only where they clarify the evidence.\n"
        "The refinement result is authoritative. Keep unchanged evidence when the zoom crops do not alter it.\n"
        "Return the same STRICT JSON schema as the base pass, including `composition`, but set `zoom_requests` to []."
    )


def _build_material_master_focused_diagram_prompt(
    *,
    item_context: Dict[str, Any],
    material_master_material_profile: Dict[str, Any],
) -> str:
    return (
        "You are analyzing one or more engineering documents for HTS-supporting clues only.\n\n"
        f"Current item context JSON:\n{json.dumps(item_context, default=str)}\n\n"
        f"Authoritative Material Master material profile JSON:\n{json.dumps(material_master_material_profile, default=str)}\n\n"
        "The Material Master material profile is the authoritative metal composition source for this item. "
        "Do not infer, estimate, or return any composition breakdown from the diagrams. "
        "Use the PDFs only to identify standards/codes, article identity, context of use, and soft HTS chapter hints.\n\n"
        "Each source is preceded by a source label that includes a stable page_ref token plus the original "
        "PDF filename and page when available.\n\n"
        "Return STRICT JSON with keys:\n"
        '"status": "completed",\n'
        '"extracted_codes": array of material standard codes found (e.g. "DIN EN 10025-2", "ASTM A36"),\n'
        '"context_of_use": string describing the item, its function, and what assembly it belongs to,\n'
        '"hts_hints": array of objects with "chapter" (int) and "rationale" (string),\n'
        '"uncertainty_notes": array of short strings noting concrete document ambiguities.\n'
        "Do not return composition, weight evidence, or material evidence in this focused mode."
    )


def _merge_focused_diagram_batch_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "status": "completed",
        "extracted_codes": [],
        "context_of_use": "",
        "hts_hints": [],
        "uncertainty_notes": [],
    }
    extracted_codes: List[str] = []
    hts_hints: List[Dict[str, Any]] = []
    uncertainty_notes: List[str] = []
    context_parts: List[str] = []
    seen_hint_keys = set()

    for batch_result in batch_results:
        extracted_codes.extend(
            str(code).strip()
            for code in list(batch_result.get("extracted_codes") or [])
            if str(code).strip()
        )
        context_text = str(batch_result.get("context_of_use") or "").strip()
        if context_text:
            context_parts.append(context_text)
        uncertainty_notes.extend(
            str(note).strip()
            for note in list(batch_result.get("uncertainty_notes") or [])
            if str(note).strip()
        )
        for raw_hint in list(batch_result.get("hts_hints") or []):
            if not isinstance(raw_hint, dict):
                continue
            chapter = _coerce_int(raw_hint.get("chapter"))
            rationale = str(raw_hint.get("rationale") or "").strip()
            if chapter is None or chapter <= 0:
                continue
            key = (chapter, rationale)
            if key in seen_hint_keys:
                continue
            seen_hint_keys.add(key)
            hts_hints.append({"chapter": chapter, "rationale": rationale})

    merged["extracted_codes"] = list(dict.fromkeys(extracted_codes))
    merged["context_of_use"] = " ".join(context_parts).strip()
    merged["hts_hints"] = hts_hints
    merged["uncertainty_notes"] = list(dict.fromkeys(uncertainty_notes))
    return merged


def _build_material_master_material_cues(material_master_material_profile: Dict[str, Any]) -> List[str]:
    top_level_labels = {
        "steel": "steel",
        "aluminum": "aluminum",
        "copper": "copper",
        "cast_iron": "cast iron",
    }
    steel_subtype_labels = {
        "electrical_steel": "electrical steel",
        "cold_rolled_coil_steel": "cold rolled coil steel",
        "hot_rolled_coil_steel": "hot rolled coil steel",
        "stainless_steel_304": "stainless steel 304",
        "stainless_steel_316": "stainless steel 316",
        "stainless_steel_bar": "stainless steel bar",
        "duplex_steel": "duplex steel",
        "cast_steel": "cast steel",
    }
    cues: List[str] = []
    for metal, label in top_level_labels.items():
        grams = _normalize_float((material_master_material_profile.get("top_level_grams") or {}).get(metal, 0.0))
        if grams > 0.0:
            cues.append(f"{label} ({round(grams, 3):g} g)")
    for subtype, label in steel_subtype_labels.items():
        grams = _normalize_float((material_master_material_profile.get("steel_subtype_grams") or {}).get(subtype, 0.0))
        if grams > 0.0:
            cues.append(f"{label} ({round(grams, 3):g} g)")
    return cues


def _invoke_diagram_prompt(
    *,
    prompt: str,
    sources: List[MixedDiagramBatchEntry],
    llm: LLMClient,
    settings: MetalCompositionSettings,
    task: str,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for source in sources:
        if source.kind == "image":
            content_blocks.append({"type": "text", "text": f"Source label: {source.label}"})
            encoded = base64.b64encode(source.data).decode("utf-8")
            ct = source.content_type or "application/octet-stream"
            content_blocks.append(
                {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{encoded}"}}
            )
            continue
        content_blocks.append(
            {
                "type": "text",
                "text": f"Source text: {source.label}\n{source.text}",
            }
        )
    response = llm.invoke_native_chat_completion(
        model_name=settings.diagram_model_name,
        messages=[{"role": "user", "content": content_blocks}],
        reasoning_effort="low",
        thinking_type=settings.diagram_reasoning_type,
        thinking_budget=settings.diagram_reasoning_budget,
        phase="diagram",
        task=task,
        usage_recorder=usage_recorder,
    )
    message = response.choices[0].message.content
    raw_text = message if isinstance(message, str) else json.dumps(message)
    return _extract_json_payload(raw_text)


def _run_base_diagram_analysis(
    *,
    image_pages: List[RenderedDiagramPage],
    text_pages: List[RenderedDiagramTextPage],
    item_context: Dict[str, Any],
    llm: LLMClient,
    settings: MetalCompositionSettings,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    prompt = _build_base_diagram_prompt(item_context)
    batches = build_mixed_diagram_batches(
        image_pages=image_pages,
        text_pages=text_pages,
        settings=settings,
    )
    if not batches:
        return {}

    batch_results: List[Optional[Dict[str, Any]]] = [None] * len(batches)
    max_workers = max(1, min(int(settings.diagram_batch_max_concurrency), len(batches)))

    def _run_batch(index: int, batch: List[MixedDiagramBatchEntry]) -> Dict[str, Any]:
        return _invoke_diagram_prompt(
            prompt=prompt,
            sources=batch,
            llm=llm,
            settings=settings,
            task="base_pass",
            usage_recorder=usage_recorder,
        )

    if max_workers == 1:
        for index, batch in enumerate(batches):
            batch_results[index] = _run_batch(index, batch)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_run_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(future_to_index):
                batch_results[future_to_index[future]] = future.result()

    finalized_batch_results = [result for result in batch_results if result is not None]
    if len(finalized_batch_results) == 1:
        return finalized_batch_results[0]
    return _merge_diagram_batch_results(finalized_batch_results)


def _normalize_zoom_box(raw_box: Any) -> Optional[Dict[str, float]]:
    if isinstance(raw_box, dict):
        values = [
            _coerce_float(raw_box.get("x0")),
            _coerce_float(raw_box.get("y0")),
            _coerce_float(raw_box.get("x1")),
            _coerce_float(raw_box.get("y1")),
        ]
    elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        values = [_coerce_float(value) for value in raw_box]
    else:
        return None

    if any(value is None for value in values):
        return None
    x0, y0, x1, y1 = [max(0.0, min(1.0, float(value))) for value in values]
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) < _MIN_ZOOM_BOX_FRACTION or (y1 - y0) < _MIN_ZOOM_BOX_FRACTION:
        return None
    return {
        "x0": round(x0, 4),
        "y0": round(y0, 4),
        "x1": round(x1, 4),
        "y1": round(y1, 4),
    }


def _normalize_zoom_requests(
    raw_zoom_requests: Any,
    *,
    page_lookup: Dict[str, RenderedDiagramPage],
    max_requests: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized_requests: List[Dict[str, Any]] = []
    skipped_requests: List[Dict[str, Any]] = []
    seen = set()

    for raw_request in _safe_dict_items(raw_zoom_requests):
        page_ref = str(raw_request.get("page_ref") or "").strip()
        if not page_ref:
            skipped_requests.append({"reason": "missing_page_ref", "request": raw_request})
            continue
        if page_ref not in page_lookup:
            skipped_requests.append({"reason": "unknown_page_ref", "request": raw_request})
            continue
        normalized_box = _normalize_zoom_box(
            raw_request.get("normalized_box")
            or raw_request.get("box")
            or raw_request.get("coordinates")
        )
        if normalized_box is None:
            skipped_requests.append({"reason": "invalid_box", "request": raw_request, "page_ref": page_ref})
            continue

        dedupe_key = (
            page_ref,
            round(normalized_box["x0"], 2),
            round(normalized_box["y0"], 2),
            round(normalized_box["x1"], 2),
            round(normalized_box["y1"], 2),
        )
        if dedupe_key in seen:
            skipped_requests.append(
                {"reason": "duplicate_box", "request": raw_request, "page_ref": page_ref, "normalized_box": normalized_box}
            )
            continue
        seen.add(dedupe_key)

        normalized_requests.append(
            {
                "page_ref": page_ref,
                "reason": str(raw_request.get("reason") or "").strip() or "Unreadable blocking text.",
                "normalized_box": normalized_box,
            }
        )

    if len(normalized_requests) > max_requests:
        for overflow_request in normalized_requests[max_requests:]:
            skipped_requests.append(
                {
                    "reason": "over_limit",
                    "page_ref": overflow_request["page_ref"],
                    "normalized_box": overflow_request["normalized_box"],
                }
            )
        normalized_requests = normalized_requests[:max_requests]

    return normalized_requests, skipped_requests


def _apply_zoom_padding(
    normalized_box: Dict[str, float],
    *,
    padding_ratio: float,
) -> Dict[str, float]:
    return {
        "x0": round(max(0.0, normalized_box["x0"] - padding_ratio), 4),
        "y0": round(max(0.0, normalized_box["y0"] - padding_ratio), 4),
        "x1": round(min(1.0, normalized_box["x1"] + padding_ratio), 4),
        "y1": round(min(1.0, normalized_box["y1"] + padding_ratio), 4),
    }


def _render_pdf_zoom_crop(
    page: RenderedDiagramPage,
    *,
    normalized_box: Dict[str, float],
    settings: MetalCompositionSettings,
) -> tuple[DiagramPayload, Dict[str, Any]]:
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF zoom refinement but is not installed."
        ) from None

    if page.input_payload is None or page.page_number is None:
        raise ValueError("PDF zoom refinement requires an original PDF payload and page number.")

    with fitz.open(stream=page.input_payload.data, filetype="pdf") as doc:
        pdf_page = doc[page.page_number - 1]
        rect = pdf_page.rect
        clip = fitz.Rect(
            rect.x0 + normalized_box["x0"] * rect.width,
            rect.y0 + normalized_box["y0"] * rect.height,
            rect.x0 + normalized_box["x1"] * rect.width,
            rect.y0 + normalized_box["y1"] * rect.height,
        )
        matrix = fitz.Matrix(settings.diagram_zoom_render_dpi / 72.0, settings.diagram_zoom_render_dpi / 72.0)
        pixmap = pdf_page.get_pixmap(matrix=matrix, clip=clip)
        png_bytes = pixmap.tobytes("png")
        return (
            DiagramPayload(
                filename=f"{page.filename.rsplit('.', 1)[0]}_zoom.png",
                content_type="image/png",
                data=png_bytes,
                source_filename=page.source_filename,
                page_number=page.page_number,
            ),
            {"crop_source": "pdf_clip", "rendered_size": [int(pixmap.width), int(pixmap.height)]},
        )


def _render_image_zoom_crop(
    page: RenderedDiagramPage,
    *,
    normalized_box: Dict[str, float],
    settings: MetalCompositionSettings,
) -> tuple[DiagramPayload, Dict[str, Any]]:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        raise RuntimeError(
            "Pillow is required for image zoom refinement but is not installed."
        ) from None

    source_payload = page.input_payload or DiagramPayload(
        filename=page.filename,
        content_type=page.content_type,
        data=page.data,
        source_filename=page.source_filename,
        page_number=page.page_number,
    )

    with Image.open(BytesIO(source_payload.data)) as image:
        image.load()
        x0 = max(0, min(image.width - 1, int(image.width * normalized_box["x0"])))
        y0 = max(0, min(image.height - 1, int(image.height * normalized_box["y0"])))
        x1 = max(x0 + 1, min(image.width, int(round(image.width * normalized_box["x1"]))))
        y1 = max(y0 + 1, min(image.height, int(round(image.height * normalized_box["y1"]))))
        crop = image.crop((x0, y0, x1, y1))
        if max(crop.width, crop.height) > 0:
            scale = min(
                4.0,
                max(1.0, settings.diagram_zoom_image_max_dimension / float(max(crop.width, crop.height))),
            )
            if scale > 1.0:
                crop = crop.resize(
                    (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                    Image.Resampling.LANCZOS,
                )
        buffer = BytesIO()
        crop.save(buffer, format="PNG", optimize=True)
        return (
            DiagramPayload(
                filename=f"{source_payload.filename.rsplit('.', 1)[0]}_zoom.png",
                content_type="image/png",
                data=buffer.getvalue(),
                source_filename=source_payload.source_filename or page.source_filename,
                page_number=source_payload.page_number or page.page_number,
            ),
            {"crop_source": "image_crop", "rendered_size": [int(crop.width), int(crop.height)]},
        )


def render_zoom_crops(
    rendered_pages: List[RenderedDiagramPage],
    *,
    zoom_requests: List[Dict[str, Any]],
    settings: MetalCompositionSettings,
) -> tuple[List[ZoomedDiagramCrop], List[Dict[str, Any]], List[Dict[str, Any]]]:
    page_lookup = {page.page_ref: page for page in rendered_pages}
    executed_crops: List[ZoomedDiagramCrop] = []
    executed_regions: List[Dict[str, Any]] = []
    skipped_regions: List[Dict[str, Any]] = []

    for index, zoom_request in enumerate(zoom_requests, start=1):
        page = page_lookup.get(str(zoom_request.get("page_ref") or "").strip())
        if page is None:
            skipped_regions.append({"reason": "unknown_page_ref", **zoom_request})
            continue

        normalized_box = _apply_zoom_padding(
            dict(zoom_request["normalized_box"]),
            padding_ratio=float(settings.diagram_zoom_padding_ratio),
        )
        try:
            if page.input_payload and (
                page.input_payload.content_type == "application/pdf"
                or str(page.input_payload.filename or "").strip().lower().endswith(".pdf")
            ):
                raw_crop_payload, render_details = _render_pdf_zoom_crop(
                    page,
                    normalized_box=normalized_box,
                    settings=settings,
                )
            else:
                raw_crop_payload, render_details = _render_image_zoom_crop(
                    page,
                    normalized_box=normalized_box,
                    settings=settings,
                )

            processed_crop_payload, preprocess_details = _preprocess_diagram_payload_with_limits(
                raw_crop_payload,
                max_dimension=settings.diagram_zoom_image_max_dimension,
                max_bytes=settings.diagram_zoom_image_max_bytes,
            )
            processed_size = list(preprocess_details.get("processed_size") or [])
            crop = ZoomedDiagramCrop(
                crop_ref=f"Z{index}",
                page_ref=page.page_ref,
                filename=processed_crop_payload.filename,
                content_type=processed_crop_payload.content_type,
                data=processed_crop_payload.data,
                normalized_box=normalized_box,
                rendered_width=_coerce_int(processed_size[0]) or 0 if len(processed_size) > 0 else 0,
                rendered_height=_coerce_int(processed_size[1]) or 0 if len(processed_size) > 1 else 0,
                source_filename=processed_crop_payload.source_filename or page.source_filename,
                page_number=processed_crop_payload.page_number or page.page_number,
            )
            executed_crops.append(crop)
            executed_regions.append(
                {
                    "crop_ref": crop.crop_ref,
                    "page_ref": crop.page_ref,
                    "reason": zoom_request.get("reason", ""),
                    "normalized_box": normalized_box,
                    "source_filename": crop.source_filename,
                    "page_number": crop.page_number,
                    "crop_size": [crop.rendered_width, crop.rendered_height],
                    "render_details": render_details,
                    "preprocessing": preprocess_details,
                }
            )
        except Exception as exc:  # noqa: BLE001 - zoom is best-effort refinement
            logger.warning("Skipping zoom request for %s: %s", page.page_ref, exc)
            skipped_regions.append(
                {
                    "reason": "render_failed",
                    "page_ref": page.page_ref,
                    "normalized_box": normalized_box,
                    "error": str(exc),
                }
            )

    return executed_crops, executed_regions, skipped_regions


def _run_refinement_diagram_analysis(
    *,
    rendered_pages: List[RenderedDiagramPage],
    zoom_crops: List[ZoomedDiagramCrop],
    executed_regions: List[Dict[str, Any]],
    provisional_result: Dict[str, Any],
    item_context: Dict[str, Any],
    llm: LLMClient,
    settings: MetalCompositionSettings,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    affected_page_refs = {region["page_ref"] for region in executed_regions if region.get("page_ref")}
    refinement_pages = [page for page in rendered_pages if page.page_ref in affected_page_refs]
    prompt = _build_refinement_diagram_prompt(
        item_context=item_context,
        provisional_result=provisional_result,
        zoom_requests=executed_regions,
    )
    sources: List[MixedDiagramBatchEntry] = []
    for page in refinement_pages:
        sources.append(
            MixedDiagramBatchEntry(
                kind="image",
                page_ref=page.page_ref,
                sequence_index=page.sequence_index,
                label=_rendered_page_source_label(page),
                source_filename=page.source_filename,
                page_number=page.page_number,
                content_type=page.content_type,
                data=page.data,
            )
        )
    for crop in zoom_crops:
        sources.append(
            MixedDiagramBatchEntry(
                kind="image",
                page_ref=crop.page_ref,
                sequence_index=10_000 + len(sources),
                label=_zoom_crop_source_label(crop),
                source_filename=crop.source_filename,
                page_number=crop.page_number,
                content_type=crop.content_type,
                data=crop.data,
            )
    )
    return _invoke_diagram_prompt(
        prompt=prompt,
        sources=sources,
        llm=llm,
        settings=settings,
        task="refinement",
        usage_recorder=usage_recorder,
    )


def _run_focused_diagram_analysis(
    *,
    image_pages: List[RenderedDiagramPage],
    text_pages: List[RenderedDiagramTextPage],
    item_context: Dict[str, Any],
    material_master_material_profile: Dict[str, Any],
    llm: LLMClient,
    settings: MetalCompositionSettings,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    prompt = _build_material_master_focused_diagram_prompt(
        item_context=item_context,
        material_master_material_profile=material_master_material_profile,
    )
    batches = build_mixed_diagram_batches(
        image_pages=image_pages,
        text_pages=text_pages,
        settings=settings,
    )
    if not batches:
        return {}

    batch_results: List[Optional[Dict[str, Any]]] = [None] * len(batches)
    max_workers = max(1, min(int(settings.diagram_batch_max_concurrency), len(batches)))

    def _run_batch(index: int, batch: List[MixedDiagramBatchEntry]) -> Dict[str, Any]:
        del index
        return _invoke_diagram_prompt(
            prompt=prompt,
            sources=batch,
            llm=llm,
            settings=settings,
            task="material_master_focused_clues",
            usage_recorder=usage_recorder,
        )

    if max_workers == 1:
        for index, batch in enumerate(batches):
            batch_results[index] = _run_batch(index, batch)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_run_batch, index, batch): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(future_to_index):
                batch_results[future_to_index[future]] = future.result()

    finalized_batch_results = [result for result in batch_results if result is not None]
    if len(finalized_batch_results) == 1:
        return finalized_batch_results[0]
    return _merge_focused_diagram_batch_results(finalized_batch_results)


def _merge_diagram_batch_results(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge results from multiple diagram analysis batches."""
    merged: Dict[str, Any] = {"status": "completed"}

    for key in (
        "extracted_codes",
        "material_cues",
        "material_properties",
        "hts_hints",
        "uncertainty_notes",
        "non_metal_evidence",
        "matched_identifiers",
        "provenance_flags",
    ):
        combined: list = []
        for br in batch_results:
            combined.extend(br.get(key, []))
        merged[key] = combined

    for key in ("weight_evidence", "material_evidence"):
        combined: list = []
        for br in batch_results:
            combined.extend(_safe_evidence_items(br.get(key)))
        merged[key] = combined

    zoom_requests: list = []
    for br in batch_results:
        zoom_requests.extend(_safe_dict_items(br.get("zoom_requests")))
    merged["zoom_requests"] = zoom_requests

    merged["estimated_metal_share"] = max(
        (_normalize_confidence(br.get("estimated_metal_share", 0.0)) for br in batch_results),
        default=0.0,
    )

    merged["is_likely_metal_item"] = any(
        br.get("is_likely_metal_item", False) for br in batch_results
    )

    context_parts = [
        str(br.get("context_of_use", "") or "").strip()
        for br in batch_results
        if str(br.get("context_of_use", "") or "").strip()
    ]
    merged["context_of_use"] = " ".join(context_parts)

    basis_priority = {
        "explicit_partial_non_metal": 4,
        "explicit_full_metal": 3,
        "inferred_partial_non_metal": 2,
        "inferred_full_metal": 1,
        "unknown": 0,
    }
    merged["metal_share_basis"] = "unknown"
    best_score = -1
    for br in batch_results:
        basis = str(br.get("metal_share_basis", "unknown") or "unknown").strip()
        score = basis_priority.get(basis, -1)
        if score > best_score:
            best_score = score
            merged["metal_share_basis"] = basis if basis in basis_priority else "unknown"

    share_reasoning_parts = [
        str(br.get("metal_share_reasoning", "") or "").strip()
        for br in batch_results
        if str(br.get("metal_share_reasoning", "") or "").strip()
    ]
    merged["metal_share_reasoning"] = " ".join(share_reasoning_parts)
    certainty_values = {
        str(br.get("metal_share_certainty") or "").strip().lower()
        for br in batch_results
        if str(br.get("metal_share_certainty") or "").strip()
    }
    merged["metal_share_certainty"] = "exact" if "exact" in certainty_values else "estimated"

    return merged


def analyze_diagrams(
    diagram_payloads: List[DiagramPayload],
    settings: MetalCompositionSettings,
    llm: LLMClient,
    *,
    product_code: Optional[str] = None,
    source_summary: Optional[Dict[str, Any]] = None,
    source_row: Optional[Dict[str, Any]] = None,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    """Analyze one or more diagram/PDF payloads via a multi-image vision call."""
    started_perf = perf_counter()
    started_at = utc_now_iso()

    if settings.diagram_page_routing_enabled:
        materialized = materialize_diagram_sources(
            diagram_payloads,
            settings,
            llm,
            usage_recorder=usage_recorder,
        )
    else:
        rendered_pages, preprocess_details_list = render_diagram_pages(diagram_payloads, settings)
        page_decisions = [
            {
                "page_ref": page.page_ref,
                "source_filename": page.source_filename or page.filename,
                "page_number": page.page_number,
                "final_route": "image_analysis",
                "override_name": "routing_disabled",
                "is_ambiguous": False,
                "fallback_used": False,
                "fallback_resolved": False,
                "fallback_reason": None,
            }
            for page in rendered_pages
        ]
        materialized = MaterializedDiagramSources(
            image_pages=rendered_pages,
            text_pages=[],
            preprocess_details_list=preprocess_details_list,
            routing_summary={
                "page_count": len(rendered_pages),
                "image_analysis_pages": len(rendered_pages),
                "text_analysis_pages": 0,
                "skip_pages": 0,
                "ambiguous_before_fallback_pages": 0,
                "ambiguous_pages": 0,
                "llm_fallback_attempted_pages": 0,
                "llm_fallback_resolved_pages": 0,
                "image_pages_rendered": len(rendered_pages),
                "text_chars_sent": 0,
            },
            page_decisions=page_decisions,
        )
    rendered_pages = materialized.image_pages
    text_pages = materialized.text_pages
    preprocess_details_list = materialized.preprocess_details_list
    item_context = _extract_item_context(
        product_code=product_code,
        source_summary=source_summary,
        source_row=source_row,
    )

    if not rendered_pages and not text_pages:
        merged = _diagram_output_defaults()
        merged["status"] = "omitted"
        merged["zoom_diagnostics"] = {
            "enabled": bool(settings.diagram_zoom_enabled),
            "triggered": False,
            "requested_regions": [],
            "executed_regions": [],
            "skipped_regions": [],
            "affected_pages": [],
            "refinement_applied": False,
            "refinement_changed_output": False,
            "refinement_error": "",
        }
        merged["composition"] = {
            "is_metal_item": False,
            "total_weight_grams": _normalize_float(
                (source_summary or {}).get("total_weight_gram")
                or (source_row or {}).get("Total Weight (Gram)")
            ),
            "estimated_total_metal_grams": 0.0,
            "top_level_grams": {"steel": 0.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {
                "electrical_steel": 0.0,
                "cold_rolled_coil_steel": 0.0,
                "hot_rolled_coil_steel": 0.0,
                "stainless_steel_304": 0.0,
                "stainless_steel_316": 0.0,
                "stainless_steel_bar": 0.0,
                "duplex_steel": 0.0,
                "cast_steel": 0.0,
            },
            "confidence": 0.0,
            "reasoning": "No routable diagram or text pages remained after page routing.",
            "provenance": {
                "dominant_source": "none",
                "top_level_sources": {
                    "steel": "none",
                    "aluminum": "none",
                    "copper": "none",
                    "cast_iron": "none",
                },
                "steel_subtype_sources": {
                    "electrical_steel": "none",
                    "cold_rolled_coil_steel": "none",
                    "hot_rolled_coil_steel": "none",
                    "stainless_steel_304": "none",
                    "stainless_steel_316": "none",
                    "stainless_steel_bar": "none",
                    "duplex_steel": "none",
                    "cast_steel": "none",
                },
                "needs_human_review": False,
                "notes": [],
            },
        }
        merged["timing"] = finish_timing(
            started_perf,
            started_at,
            details={
                "model": settings.diagram_model_name,
                "total_images": 0,
                "total_text_pages": 0,
                "total_batches": 0,
                "source_filenames": [p.filename for p in diagram_payloads],
                "item_context": item_context,
                "preprocessing": preprocess_details_list,
                "routing": {
                    **materialized.routing_summary,
                    "page_decisions": materialized.page_decisions,
                },
                "zoom": {
                    "enabled": bool(settings.diagram_zoom_enabled),
                    "requested_region_count": 0,
                    "executed_region_count": 0,
                    "refinement_applied": False,
                    "refinement_changed_output": False,
                },
            },
        )
        return merged

    base_result = _run_base_diagram_analysis(
        image_pages=rendered_pages,
        text_pages=text_pages,
        item_context=item_context,
        llm=llm,
        settings=settings,
        usage_recorder=usage_recorder,
    )
    raw_zoom_requests = _safe_dict_items(base_result.get("zoom_requests"))
    requested_zoom_regions = list(raw_zoom_requests)
    page_lookup = {page.page_ref: page for page in rendered_pages}
    normalized_zoom_requests, skipped_zoom_regions = _normalize_zoom_requests(
        raw_zoom_requests,
        page_lookup=page_lookup,
        max_requests=int(settings.diagram_zoom_max_requests),
    )

    refinement_applied = False
    refinement_changed_output = False
    refinement_error = ""
    executed_zoom_regions: List[Dict[str, Any]] = []
    zoom_crops: List[ZoomedDiagramCrop] = []
    merged = dict(base_result)

    if bool(settings.diagram_zoom_enabled) and rendered_pages and normalized_zoom_requests:
        zoom_crops, executed_zoom_regions, render_skipped_regions = render_zoom_crops(
            rendered_pages,
            zoom_requests=normalized_zoom_requests,
            settings=settings,
        )
        skipped_zoom_regions.extend(render_skipped_regions)
        if zoom_crops:
            provisional_result = dict(base_result)
            provisional_result.pop("zoom_requests", None)
            try:
                refinement_result = _run_refinement_diagram_analysis(
                    rendered_pages=rendered_pages,
                    zoom_crops=zoom_crops,
                    executed_regions=executed_zoom_regions,
                    provisional_result=provisional_result,
                    item_context=item_context,
                    llm=llm,
                    settings=settings,
                    usage_recorder=usage_recorder,
                )
                refinement_applied = True
                refinement_result.pop("zoom_requests", None)
                refinement_changed_output = json.dumps(
                    _finalize_diagram_result(provisional_result),
                    sort_keys=True,
                    default=str,
                ) != json.dumps(
                    _finalize_diagram_result(refinement_result),
                    sort_keys=True,
                    default=str,
                )
                merged = refinement_result
            except Exception as exc:  # noqa: BLE001 - refinement is best-effort
                logger.warning("Diagram zoom refinement failed: %s", exc)
                refinement_error = str(exc)

    merged.pop("zoom_requests", None)
    merged = _finalize_diagram_result(merged)
    merged["zoom_diagnostics"] = {
        "enabled": bool(settings.diagram_zoom_enabled),
        "triggered": bool(requested_zoom_regions),
        "requested_regions": normalized_zoom_requests if normalized_zoom_requests else requested_zoom_regions,
        "executed_regions": executed_zoom_regions,
        "skipped_regions": skipped_zoom_regions,
        "affected_pages": sorted({region["page_ref"] for region in executed_zoom_regions if region.get("page_ref")}),
        "refinement_applied": refinement_applied,
        "refinement_changed_output": refinement_changed_output,
        "refinement_error": refinement_error,
    }

    # 8. Extract composition from LLM response and build final_composition
    raw_composition_value = merged.pop("composition", None)
    composition_raw, composition_was_coerced = _normalize_composition_payload(raw_composition_value)
    total_weight = _normalize_float(
        (source_summary or {}).get("total_weight_gram")
        or (source_row or {}).get("Total Weight (Gram)")
    )
    metal_share = _normalize_float(merged.get("estimated_metal_share", 1.0))

    if composition_raw.get("top_level_grams"):
        # LLM produced a composition breakdown -- use it directly
        composition_payload = dict(composition_raw)
    else:
        # Fallback: build a basic composition from material evidence
        dominant_metal = None
        for ev in _safe_evidence_items(merged.get("material_evidence")):
            if ev.get("applies_to_current_item") and ev.get("is_explicit", True):
                metal = _normalize_metal_type(ev.get("normalized_metal") or ev.get("raw_material"))
                if metal:
                    dominant_metal = metal
                    break
        metal_grams = (total_weight or 0.0) * metal_share
        top_level = {m: 0.0 for m in ("steel", "aluminum", "copper", "cast_iron")}
        if dominant_metal and dominant_metal in top_level:
            top_level[dominant_metal] = metal_grams
        composition_payload = {
            "is_metal_item": merged.get("is_likely_metal_item", False),
            "total_weight_grams": total_weight,
            "estimated_total_metal_grams": metal_grams,
            "top_level_grams": top_level,
            "steel_subtype_grams": {s: 0.0 for s in (
                "electrical_steel", "cold_rolled_coil_steel", "hot_rolled_coil_steel",
                "stainless_steel_304", "stainless_steel_316", "stainless_steel_bar",
                "duplex_steel", "cast_steel",
            )},
            "confidence": _normalize_confidence(merged.get("estimated_metal_share", 0.0)),
            "reasoning": "Fallback: LLM did not return a composition breakdown.",
        }

    if composition_was_coerced:
        composition_payload["reasoning"] = " ".join(
            part
            for part in (
                str(composition_payload.get("reasoning") or "").strip(),
                "Composition payload was normalized from a list-shaped model response.",
            )
            if part
        )

    # Ensure total_weight_grams and is_metal_item are set
    composition_payload.setdefault("total_weight_grams", total_weight)
    composition_payload.setdefault("is_metal_item", merged.get("is_likely_metal_item", False))

    # Build provenance from composition data
    top_level_grams = composition_payload.get("top_level_grams", {})
    steel_subtype_grams = composition_payload.get("steel_subtype_grams", {})
    composition_payload["provenance"] = {
        "dominant_source": "pdf_diagram",
        "top_level_sources": {
            metal: ("pdf_diagram" if _normalize_float(top_level_grams.get(metal, 0.0)) > 0.0 else "none")
            for metal in ("steel", "aluminum", "copper", "cast_iron")
        },
        "steel_subtype_sources": {
            subtype: ("pdf_diagram" if _normalize_float(steel_subtype_grams.get(subtype, 0.0)) > 0.0 else "none")
            for subtype in (
                "electrical_steel", "cold_rolled_coil_steel", "hot_rolled_coil_steel",
                "stainless_steel_304", "stainless_steel_316", "stainless_steel_bar",
                "duplex_steel", "cast_steel",
            )
        },
        "needs_human_review": False,
        "notes": (
            ["Composition payload was normalized from a list-shaped model response."]
            if composition_was_coerced
            else []
        ),
    }

    merged["composition"] = composition_payload

    merged["timing"] = finish_timing(
        started_perf,
        started_at,
        details={
            "model": settings.diagram_model_name,
            "total_images": len(rendered_pages),
            "total_text_pages": len(text_pages),
            "total_batches": len(
                build_mixed_diagram_batches(
                    image_pages=rendered_pages,
                    text_pages=text_pages,
                    settings=settings,
                )
            ),
            "source_filenames": [p.filename for p in diagram_payloads],
            "item_context": item_context,
            "preprocessing": preprocess_details_list,
            "routing": {
                **materialized.routing_summary,
                "page_decisions": materialized.page_decisions,
            },
            "zoom": {
                "enabled": bool(settings.diagram_zoom_enabled),
                "requested_region_count": len(normalized_zoom_requests),
                "executed_region_count": len(executed_zoom_regions),
                "refinement_applied": refinement_applied,
                "refinement_changed_output": refinement_changed_output,
            },
        },
    )
    return merged


def analyze_diagrams_for_hts_clues(
    diagram_payloads: List[DiagramPayload],
    settings: MetalCompositionSettings,
    llm: LLMClient,
    *,
    material_master_material_profile: Dict[str, Any],
    product_code: Optional[str] = None,
    source_summary: Optional[Dict[str, Any]] = None,
    source_row: Optional[Dict[str, Any]] = None,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    started_perf = perf_counter()
    started_at = utc_now_iso()

    if settings.diagram_page_routing_enabled:
        materialized = materialize_diagram_sources(
            diagram_payloads,
            settings,
            llm,
            usage_recorder=usage_recorder,
        )
    else:
        rendered_pages, preprocess_details_list = render_diagram_pages(diagram_payloads, settings)
        page_decisions = [
            {
                "page_ref": page.page_ref,
                "source_filename": page.source_filename or page.filename,
                "page_number": page.page_number,
                "final_route": "image_analysis",
                "override_name": "routing_disabled",
                "is_ambiguous": False,
                "fallback_used": False,
                "fallback_resolved": False,
                "fallback_reason": None,
            }
            for page in rendered_pages
        ]
        materialized = MaterializedDiagramSources(
            image_pages=rendered_pages,
            text_pages=[],
            preprocess_details_list=preprocess_details_list,
            routing_summary={
                "page_count": len(rendered_pages),
                "image_analysis_pages": len(rendered_pages),
                "text_analysis_pages": 0,
                "skip_pages": 0,
                "ambiguous_before_fallback_pages": 0,
                "ambiguous_pages": 0,
                "llm_fallback_attempted_pages": 0,
                "llm_fallback_resolved_pages": 0,
                "image_pages_rendered": len(rendered_pages),
                "text_chars_sent": 0,
            },
            page_decisions=page_decisions,
        )

    rendered_pages = materialized.image_pages
    text_pages = materialized.text_pages
    preprocess_details_list = materialized.preprocess_details_list
    item_context = _extract_item_context(
        product_code=product_code,
        source_summary=source_summary,
        source_row=source_row,
    )
    total_weight_grams = _normalize_float(
        (source_summary or {}).get("total_weight_gram")
        or (source_row or {}).get("Total Weight (Gram)")
    )
    estimated_total_metal_grams = float(
        sum(
            _normalize_float(value)
            for value in (material_master_material_profile.get("top_level_grams") or {}).values()
        )
    )
    if total_weight_grams and total_weight_grams > 0.0:
        estimated_metal_share = max(0.0, min(1.0, estimated_total_metal_grams / total_weight_grams))
    else:
        estimated_metal_share = 1.0 if estimated_total_metal_grams > 0.0 else 0.0
    material_cues = _build_material_master_material_cues(material_master_material_profile)

    merged = _diagram_output_defaults()
    merged.update(
        {
            "status": "omitted" if not rendered_pages and not text_pages else "completed",
            "is_likely_metal_item": estimated_total_metal_grams > 0.0,
            "estimated_metal_share": estimated_metal_share,
            "metal_share_certainty": "exact",
            "metal_share_basis": (
                "explicit_full_metal"
                if estimated_metal_share >= 0.999 and estimated_total_metal_grams > 0.0
                else "explicit_partial_non_metal"
                if estimated_total_metal_grams > 0.0
                else "unknown"
            ),
            "metal_share_reasoning": (
                "Metal share was computed directly from Material Master prepared gram columns and the item total weight."
            ),
            "material_cues": material_cues,
            "weight_evidence": [],
            "material_evidence": [],
            "material_properties": [],
            "matched_identifiers": [],
            "non_metal_evidence": [],
            "provenance_flags": ["material_master_composition_mode"],
        }
    )

    if rendered_pages or text_pages:
        focused_result = _run_focused_diagram_analysis(
            image_pages=rendered_pages,
            text_pages=text_pages,
            item_context=item_context,
            material_master_material_profile=material_master_material_profile,
            llm=llm,
            settings=settings,
            usage_recorder=usage_recorder,
        )
        merged.update(
            {
                "status": str(focused_result.get("status") or "completed"),
                "extracted_codes": list(focused_result.get("extracted_codes") or []),
                "context_of_use": str(focused_result.get("context_of_use") or "").strip(),
                "hts_hints": list(focused_result.get("hts_hints") or []),
                "uncertainty_notes": list(focused_result.get("uncertainty_notes") or []),
            }
        )

    merged = _finalize_diagram_result(merged)
    merged["zoom_diagnostics"] = {
        "enabled": False,
        "triggered": False,
        "requested_regions": [],
        "executed_regions": [],
        "skipped_regions": [],
        "affected_pages": [],
        "refinement_applied": False,
        "refinement_changed_output": False,
        "refinement_error": "",
    }
    merged["timing"] = finish_timing(
        started_perf,
        started_at,
        details={
            "model": settings.diagram_model_name,
            "analysis_mode": "material_master_focused_clues",
            "total_images": len(rendered_pages),
            "total_text_pages": len(text_pages),
            "total_batches": len(
                build_mixed_diagram_batches(
                    image_pages=rendered_pages,
                    text_pages=text_pages,
                    settings=settings,
                )
            ),
            "source_filenames": [p.filename for p in diagram_payloads],
            "item_context": item_context,
            "material_master_material_profile": material_master_material_profile,
            "preprocessing": preprocess_details_list,
            "routing": {
                **materialized.routing_summary,
                "page_decisions": materialized.page_decisions,
            },
        },
    )
    return merged
