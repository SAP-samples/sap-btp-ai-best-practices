"""Page-level routing and mixed text/image materialization for diagram analysis."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List

import fitz
import numpy as np
from PIL import Image

from ..config import MetalCompositionSettings
from .llm import LLMClient
from .token_usage import TokenUsageRecorder
from .types import DiagramPayload
from .types import MaterializedDiagramSources
from .types import MixedDiagramBatchEntry
from .types import RenderedDiagramPage
from .types import RenderedDiagramTextPage

DRAWING_KEYWORDS = ("dwg. no.", "revision", "scale", "sheet", "section", "detail", "part no.", "qty")
TEXT_KEYWORDS = (
    "scope",
    "requirements",
    "chemical composition",
    "installation",
    "operation",
    "maintenance",
    "material standard",
)
VALID_ROUTES = {"image_analysis", "text_analysis", "skip"}
MIN_FALLBACK_CONFIDENCE = 0.60
MAX_PROVIDER_IMAGES_PER_REQUEST = 50
MAX_FALLBACK_IMAGES_PER_REQUEST = 15
CONFIDENCE_ALIASES = {
    "very high": 0.98,
    "high": 0.90,
    "medium": 0.65,
    "low": 0.35,
    "very low": 0.15,
}


@dataclass(frozen=True)
class PdfNativeSignals:
    page_number: int
    extracted_chars: int
    non_whitespace_chars: int
    word_count: int
    preview: str
    paragraph_block_count: int
    paragraph_text_chars: int
    text_block_coverage_ratio: float
    sentence_density: float
    uppercase_ratio: float
    drawing_keyword_hits: int
    standards_or_manual_keyword_hits: int
    border_coordinate_grid_detected: bool
    title_block_detected: bool
    bom_or_revision_table_detected: bool
    drawing_path_count: int
    embedded_image_count: int
    embedded_image_area_ratio: float


@dataclass(frozen=True)
class PreviewSignals:
    whitespace_ratio: float
    ink_ratio: float
    edge_density: float
    line_art_density: float
    lower_band_density: float
    large_raster_coverage_ratio: float


@dataclass(frozen=True)
class RouteScores:
    image_score: float
    text_score: float
    skip_score: float


@dataclass(frozen=True)
class PageRouteDecision:
    final_route: str
    override_name: str | None
    is_ambiguous: bool
    used_preview: bool
    scores: RouteScores
    fallback_used: bool = False
    fallback_resolved: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class PendingFallbackPage:
    page_ref: str
    sequence_index: int
    source_document_index: int
    payload: DiagramPayload
    source_filename: str
    page_number: int
    page_text: str
    decision: PageRouteDecision
    native: PdfNativeSignals
    preview: PreviewSignals | None
    fallback_png_bytes: bytes


def _normalize_preview_text(text: str) -> str:
    return " ".join((text or "").split())


def _normalize_page_text(text: str) -> str:
    raw_lines = [line.rstrip() for line in (text or "").splitlines()]
    collapsed: list[str] = []
    previous_blank = False
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            if not previous_blank:
                collapsed.append("")
            previous_blank = True
            continue
        collapsed.append(line)
        previous_blank = False
    return "\n".join(collapsed).strip()


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _block_text(block: object) -> str:
    if isinstance(block, (tuple, list)) and len(block) >= 5:
        return str(block[4]).strip()
    return ""


def _word_is_number(token: str) -> bool:
    return token.isdigit()


def _has_edge_coordinate_grid(page) -> bool:
    words = page.get_text("words") or []
    if not words:
        return False
    top_band = min(max(float(page.rect.height) * 0.12, 30.0), 60.0)
    top_numbers = [
        word
        for word in words
        if _word_is_number(str(word[4]).strip()) and float(word[1]) <= top_band
    ]
    if len(top_numbers) < 4:
        return False
    top_numbers = sorted(top_numbers, key=lambda word: float(word[0]))
    return [str(word[4]).strip() for word in top_numbers[:4]] == ["1", "2", "3", "4"]


def extract_pdf_native_signals(page) -> PdfNativeSignals:
    blocks = page.get_text("blocks") or []
    text = (page.get_text() or "").strip()
    lowered = text.lower()
    words = text.split()
    non_whitespace_chars = sum(1 for char in text if not char.isspace())
    block_texts = [_block_text(block) for block in blocks]
    paragraph_blocks = [block_text for block_text in block_texts if len(block_text.split()) >= 6]
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    text_area = 0.0
    for block in blocks:
        if isinstance(block, (tuple, list)) and len(block) >= 4:
            text_area += max(float(block[2] - block[0]) * float(block[3] - block[1]), 0.0)
    image_blocks = [
        block
        for block in (page.get_text("dict") or {}).get("blocks", [])
        if isinstance(block, dict) and block.get("type") == 1 and block.get("bbox")
    ]
    embedded_images = page.get_images(full=True)
    image_area_ratio = 0.0
    for block in image_blocks:
        bbox = block.get("bbox")
        if bbox and len(bbox) == 4:
            image_area_ratio += max(float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1]), 0.0) / page_area

    border_coordinate_grid_detected = _has_edge_coordinate_grid(page)
    title_block_detected = (
        ("dwg. no." in lowered or "scale" in lowered)
        and ("revision" in lowered or "sheet" in lowered)
    )
    return PdfNativeSignals(
        page_number=page.number + 1,
        extracted_chars=len(text),
        non_whitespace_chars=non_whitespace_chars,
        word_count=len(words),
        preview=_normalize_preview_text(text)[:120],
        paragraph_block_count=len(paragraph_blocks),
        paragraph_text_chars=sum(len(block_text) for block_text in paragraph_blocks),
        text_block_coverage_ratio=min(text_area / page_area, 1.0),
        sentence_density=(lowered.count(".") + lowered.count(":")) / max(non_whitespace_chars, 1),
        uppercase_ratio=sum(1 for char in text if char.isupper()) / max(
            sum(1 for char in text if char.isalpha()),
            1,
        ),
        drawing_keyword_hits=_count_keyword_hits(text, DRAWING_KEYWORDS),
        standards_or_manual_keyword_hits=_count_keyword_hits(text, TEXT_KEYWORDS),
        border_coordinate_grid_detected=border_coordinate_grid_detected,
        title_block_detected=title_block_detected,
        bom_or_revision_table_detected=("qty" in lowered and "part no." in lowered) or "revision" in lowered,
        drawing_path_count=len(page.get_drawings()),
        embedded_image_count=len(embedded_images),
        embedded_image_area_ratio=min(image_area_ratio, 1.0),
    )


def extract_preview_signals(page, *, dpi: int) -> PreviewSignals:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
    image = Image.frombytes("L", [pixmap.width, pixmap.height], pixmap.samples)
    pixels = np.asarray(image, dtype=np.uint8)
    pixel_int = pixels.astype(np.int16)
    ink_mask = pixels < 245
    horizontal_edges = np.abs(np.diff(pixel_int, axis=1)) > 20
    vertical_edges = np.abs(np.diff(pixel_int, axis=0)) > 20
    lower_band = ink_mask[int(pixels.shape[0] * 0.75):, :]
    page_area = max(page.rect.width * page.rect.height, 1.0)
    image_rect_area = 0.0
    image_blocks = [
        block
        for block in (page.get_text("dict") or {}).get("blocks", [])
        if isinstance(block, dict) and block.get("type") == 1 and block.get("bbox")
    ]
    for block in image_blocks:
        bbox = block.get("bbox")
        if bbox and len(bbox) == 4:
            image_rect_area += max(float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1]), 0.0) / page_area

    edge_mask = np.zeros_like(pixels, dtype=bool)
    if horizontal_edges.size:
        edge_mask[:, 1:] |= horizontal_edges
        edge_mask[:, :-1] |= horizontal_edges
    if vertical_edges.size:
        edge_mask[1:, :] |= vertical_edges
        edge_mask[:-1, :] |= vertical_edges

    row_activity = float(horizontal_edges.any(axis=1).mean()) if horizontal_edges.size else 0.0
    col_activity = float(vertical_edges.any(axis=0).mean()) if vertical_edges.size else 0.0
    ink_ratio = float(ink_mask.mean())
    return PreviewSignals(
        whitespace_ratio=1.0 - ink_ratio,
        ink_ratio=ink_ratio,
        edge_density=float(edge_mask.mean()) if edge_mask.size else 0.0,
        line_art_density=(row_activity + col_activity) / 2.0,
        lower_band_density=float(lower_band.mean()) if lower_band.size else 0.0,
        large_raster_coverage_ratio=min(image_rect_area, 1.0),
    )


def route_page(native: PdfNativeSignals, preview: PreviewSignals | None) -> PageRouteDecision:
    if native.border_coordinate_grid_detected:
        return PageRouteDecision(
            final_route="image_analysis",
            override_name="border_coordinate_grid_detected",
            is_ambiguous=False,
            used_preview=preview is not None,
            scores=RouteScores(image_score=1.0, text_score=0.0, skip_score=0.0),
        )
    preview_text = native.preview.lower()
    has_drawing_evidence = (
        native.title_block_detected
        or native.bom_or_revision_table_detected
        or native.drawing_keyword_hits > 0
        or native.drawing_path_count >= 40
        or native.embedded_image_count > 0
        or native.embedded_image_area_ratio >= 0.10
        or (
            preview is not None
            and (
                preview.large_raster_coverage_ratio >= 0.20
                or preview.line_art_density >= 0.05
                or preview.edge_density >= 0.03
            )
        )
    )
    if (
        native.paragraph_block_count >= 3
        and native.paragraph_text_chars >= 400
        and native.drawing_path_count < 40
        and not native.title_block_detected
        and not native.border_coordinate_grid_detected
        and not native.bom_or_revision_table_detected
        and native.drawing_keyword_hits == 0
        and native.embedded_image_count == 0
        and native.embedded_image_area_ratio < 0.10
        and not has_drawing_evidence
    ):
        return PageRouteDecision(
            final_route="text_analysis",
            override_name="paragraph_text_guardrails",
            is_ambiguous=False,
            used_preview=preview is not None,
            scores=RouteScores(image_score=0.0, text_score=1.0, skip_score=0.0),
        )
    if (
        native.word_count >= 500
        and native.paragraph_text_chars >= 1500
        and native.text_block_coverage_ratio >= 0.50
        and native.drawing_path_count < 20
        and not native.title_block_detected
        and not native.border_coordinate_grid_detected
        and not native.bom_or_revision_table_detected
        and native.drawing_keyword_hits == 0
    ):
        return PageRouteDecision(
            final_route="text_analysis",
            override_name="dense_prose_guardrails",
            is_ambiguous=False,
            used_preview=preview is not None,
            scores=RouteScores(image_score=0.0, text_score=1.0, skip_score=0.0),
        )
    if "cover page" in preview_text or (native.word_count < 10 and not has_drawing_evidence):
        return PageRouteDecision(
            final_route="skip",
            override_name="low_word_blank_or_cover",
            is_ambiguous=False,
            used_preview=preview is not None,
            scores=RouteScores(image_score=0.0, text_score=0.0, skip_score=1.0),
        )

    image_score = 0.0
    text_score = 0.0
    skip_score = 0.0
    image_score += 0.35 if native.border_coordinate_grid_detected else 0.0
    image_score += 0.25 if native.title_block_detected else 0.0
    image_score += 0.15 if native.bom_or_revision_table_detected else 0.0
    image_score += 0.20 if native.drawing_path_count >= 80 else 0.0
    image_score += 0.15 if native.embedded_image_area_ratio >= 0.35 else 0.0
    image_score += 0.10 if native.drawing_keyword_hits >= 2 else 0.0
    image_score -= 0.25 if native.paragraph_block_count >= 3 else 0.0
    image_score -= 0.20 if native.paragraph_text_chars >= 600 else 0.0

    text_score += 0.30 if native.paragraph_block_count >= 3 else 0.0
    text_score += 0.25 if native.paragraph_text_chars >= 400 else 0.0
    text_score += 0.15 if native.sentence_density >= 0.04 else 0.0
    text_score += 0.15 if native.standards_or_manual_keyword_hits >= 2 else 0.0
    text_score += 0.15 if native.text_block_coverage_ratio >= 0.25 else 0.0
    text_score -= 0.30 if native.title_block_detected else 0.0
    text_score -= 0.30 if native.border_coordinate_grid_detected else 0.0
    text_score -= 0.20 if native.drawing_path_count >= 80 else 0.0

    if preview is not None:
        image_score += 0.15 if preview.large_raster_coverage_ratio >= 0.35 else 0.0
        image_score += 0.10 if preview.edge_density >= 0.02 else 0.0

    scores = RouteScores(
        image_score=max(image_score, 0.0),
        text_score=max(text_score, 0.0),
        skip_score=max(skip_score, 0.0),
    )
    ordered = sorted(
        (("image_analysis", scores.image_score), ("text_analysis", scores.text_score), ("skip", scores.skip_score)),
        key=lambda item: item[1],
        reverse=True,
    )
    top_route, top_score = ordered[0]
    second_score = ordered[1][1]
    if top_score >= 0.55 and (top_score - second_score) >= 0.15:
        return PageRouteDecision(top_route, None, False, preview is not None, scores)
    return PageRouteDecision("image_analysis", None, True, preview is not None, scores)


def _normalize_fallback_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in CONFIDENCE_ALIASES:
            numeric = CONFIDENCE_ALIASES[lowered]
        else:
            try:
                numeric = float(lowered)
            except ValueError:
                return None
    else:
        return None
    return max(0.0, min(1.0, round(numeric, 4)))


def _parse_jsonish(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _build_page_routing_prompt(
    *,
    source_filename: str,
    page_number: int,
    native_signals: dict[str, Any],
    preview_signals: dict[str, Any] | None,
) -> str:
    return (
        "Classify this single engineering-document page as exactly one of: skip, image_analysis, text_analysis.\n\n"
        "Rules:\n"
        "1. skip: only for explicit cover/separator pages with no useful engineering or standards content.\n"
        "2. image_analysis: engineering drawings, schematics, exploded views, section/detail drawings, rasterized drawing pages, or pages whose meaning depends mainly on the visual drawing.\n"
        "3. text_analysis: useful text-heavy manual, standard, specification, certificate, or prose/table page, even if it contains a few warning images or diagrams.\n"
        "4. If uncertain between skip and image_analysis, choose image_analysis.\n"
        "5. If a manual/specification page is mostly readable prose or tables, choose text_analysis.\n\n"
        "Return a JSON object with keys: final_route, confidence, reason.\n"
        "confidence must be a numeric value from 0.0 to 1.0.\n\n"
        f"source_filename: {source_filename}\n"
        f"page_number: {page_number}\n"
        f"native_signals: {native_signals}\n"
        f"preview_signals: {preview_signals}\n"
    )


def _render_page_png_bytes(page, *, dpi: int = 100) -> bytes:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def _build_batched_page_routing_prompt() -> str:
    return (
        "Classify each engineering-document page separately as exactly one of: skip, image_analysis, text_analysis.\n\n"
        "Rules:\n"
        "1. skip: only for explicit cover/separator pages with no useful engineering or standards content.\n"
        "2. image_analysis: engineering drawings, schematics, exploded views, section/detail drawings, rasterized drawing pages, or pages whose meaning depends mainly on the visual drawing.\n"
        "3. text_analysis: useful text-heavy manual, standard, specification, certificate, or prose/table page, even if it contains a few warning images or diagrams.\n"
        "4. If uncertain between skip and image_analysis, choose image_analysis.\n"
        "5. If a manual/specification page is mostly readable prose or tables, choose text_analysis.\n\n"
        'Return JSON in the form {"decisions":[{"page_ref":"P1","final_route":"image_analysis","confidence":0.95,"reason":"..."}]}.\n'
        "confidence must be numeric from 0.0 to 1.0.\n"
        "Include one decision for every provided page_ref and do not omit any page.\n"
    )


def _chunk_pending_fallback_pages(
    pages: List[PendingFallbackPage],
    *,
    max_batch_size: int = MAX_FALLBACK_IMAGES_PER_REQUEST,
) -> List[List[PendingFallbackPage]]:
    max_batch_size = max(1, int(max_batch_size))
    return [pages[index:index + max_batch_size] for index in range(0, len(pages), max_batch_size)]


def _parse_batched_fallback_decisions(payload_json: Any) -> dict[str, dict[str, Any]]:
    raw_decisions: Any
    if isinstance(payload_json, list):
        raw_decisions = payload_json
    elif isinstance(payload_json, dict):
        raw_decisions = payload_json.get("decisions")
        if raw_decisions is None and isinstance(payload_json.get("pages"), list):
            raw_decisions = payload_json.get("pages")
    else:
        raw_decisions = None

    parsed: dict[str, dict[str, Any]] = {}
    for item in list(raw_decisions or []):
        if not isinstance(item, dict):
            continue
        page_ref = str(item.get("page_ref") or "").strip()
        if not page_ref:
            continue
        parsed[page_ref] = item
    return parsed


def _fallback_unresolved(decision: PageRouteDecision, *, reason: str) -> PageRouteDecision:
    return replace(
        decision,
        final_route="image_analysis",
        is_ambiguous=False,
        fallback_used=True,
        fallback_resolved=False,
        fallback_reason=reason,
    )


def _fallback_resolved(
    decision: PageRouteDecision,
    *,
    final_route: str,
    reason: str,
) -> PageRouteDecision:
    return replace(
        decision,
        final_route=final_route,
        is_ambiguous=False,
        fallback_used=True,
        fallback_resolved=True,
        fallback_reason=reason,
    )


def _apply_llm_fallback_batch(
    *,
    pending_pages: List[PendingFallbackPage],
    settings: MetalCompositionSettings,
    llm: LLMClient,
    usage_recorder: TokenUsageRecorder | None = None,
) -> dict[str, PageRouteDecision]:
    if not pending_pages:
        return {}

    prompt = _build_batched_page_routing_prompt()
    resolved: dict[str, PageRouteDecision] = {}

    for batch in _chunk_pending_fallback_pages(pending_pages):
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for page in batch:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Page ref: {page.page_ref}\n"
                        f"Source file: {Path(page.source_filename).name}\n"
                        f"Page number: {page.page_number}\n"
                        f"Native signals JSON: {json.dumps(asdict(page.native), default=str)}\n"
                        f"Preview signals JSON: {json.dumps(asdict(page.preview) if page.preview is not None else None, default=str)}\n"
                        "confidence must be numeric from 0.0 to 1.0"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(page.fallback_png_bytes).decode('utf-8')}"
                    },
                }
            )
        try:
            response = llm.invoke_native_chat_completion(
                model_name=settings.diagram_page_routing_fallback_model_name,
                messages=[{"role": "user", "content": content}],
                phase="diagram",
                task="page_routing_fallback",
                usage_recorder=usage_recorder,
            )
            raw_content = response.choices[0].message.content
            raw_text = raw_content if isinstance(raw_content, str) else json.dumps(raw_content)
            decisions_by_page_ref = _parse_batched_fallback_decisions(_parse_jsonish(raw_text))
        except Exception as exc:  # noqa: BLE001
            for page in batch:
                resolved[page.page_ref] = _fallback_unresolved(
                    page.decision,
                    reason=f"fallback_error:{exc.__class__.__name__}",
                )
            continue

        for page in batch:
            payload_json = decisions_by_page_ref.get(page.page_ref) or {}
            final_route = payload_json.get("final_route")
            confidence = _normalize_fallback_confidence(payload_json.get("confidence"))
            reason = str(payload_json.get("reason") or "").strip()
            if final_route not in VALID_ROUTES or confidence is None or confidence < MIN_FALLBACK_CONFIDENCE or not reason:
                resolved[page.page_ref] = _fallback_unresolved(page.decision, reason="fallback_unresolved")
                continue
            resolved[page.page_ref] = _fallback_resolved(
                page.decision,
                final_route=str(final_route),
                reason=reason,
            )

    return resolved


def _is_pdf_payload(payload: DiagramPayload) -> bool:
    return payload.content_type == "application/pdf" or (payload.filename or "").lower().endswith(".pdf")


def _preprocess_diagram_payload_with_limits(
    diagram_payload: DiagramPayload,
    *,
    max_dimension: int,
    max_bytes: int,
) -> tuple[DiagramPayload, dict[str, Any]]:
    details = {
        "original_bytes": len(diagram_payload.data),
        "original_content_type": diagram_payload.content_type,
    }
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


def _preprocess_diagram_payload(
    diagram_payload: DiagramPayload,
    settings: MetalCompositionSettings,
) -> tuple[DiagramPayload, dict[str, Any]]:
    is_pdf_derived_page = bool(
        str(diagram_payload.source_filename or "").strip().lower().endswith(".pdf")
        and diagram_payload.page_number
    )
    processed, details = _preprocess_diagram_payload_with_limits(
        diagram_payload,
        max_dimension=settings.pdf_image_max_dimension if is_pdf_derived_page else settings.image_max_dimension,
        max_bytes=settings.pdf_image_max_bytes if is_pdf_derived_page else settings.image_max_bytes,
    )
    details["pdf_derived_page"] = is_pdf_derived_page
    return processed, details


def _render_pdf_page_to_image_payload(
    page,
    *,
    payload: DiagramPayload,
) -> DiagramPayload:
    matrix = fitz.Matrix(300 / 72.0, 300 / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    stem = payload.filename.rsplit(".", 1)[0] if "." in payload.filename else payload.filename
    return DiagramPayload(
        filename=f"{stem}_page_{page.number + 1}.png",
        content_type="image/png",
        data=pixmap.tobytes("png"),
        source_filename=payload.source_filename or payload.filename,
        page_number=page.number + 1,
    )


def _render_pdf_page_to_image_payload_by_number(
    *,
    payload: DiagramPayload,
    page_number: int,
) -> DiagramPayload:
    with fitz.open(stream=payload.data, filetype="pdf") as document:
        return _render_pdf_page_to_image_payload(document[page_number - 1], payload=payload)


def _page_ref_sort_key(value: str) -> tuple[int, str]:
    text = str(value or "").strip()
    if text.startswith("P") and text[1:].isdigit():
        return int(text[1:]), text
    return 10**9, text


def _text_source_label(text_page: RenderedDiagramTextPage) -> str:
    base = str(text_page.source_filename or "document text")
    if text_page.page_number:
        base = f"{base} page {text_page.page_number}"
    if text_page.chunk_count > 1:
        return f"{text_page.page_ref} | {base} | text chunk {text_page.chunk_index}/{text_page.chunk_count}"
    return f"{text_page.page_ref} | {base}"


def _rendered_page_source_label(page: RenderedDiagramPage) -> str:
    base_label = str(page.source_filename or page.filename or "document image")
    if page.page_number:
        base_label = f"{base_label} page {page.page_number}"
    return f"{page.page_ref} | {base_label}"


def materialize_diagram_sources(
    diagram_payloads: List[DiagramPayload],
    settings: MetalCompositionSettings,
    llm: LLMClient,
    usage_recorder: TokenUsageRecorder | None = None,
) -> MaterializedDiagramSources:
    if not settings.diagram_page_routing_enabled:
        image_pages: list[RenderedDiagramPage] = []
        preprocess_details_list: list[dict[str, Any]] = []
        page_decisions: list[dict[str, Any]] = []
        page_counter = 0
        for source_document_index, payload in enumerate(diagram_payloads):
            if _is_pdf_payload(payload):
                with fitz.open(stream=payload.data, filetype="pdf") as document:
                    for page in document:
                        page_counter += 1
                        processed, details = _preprocess_diagram_payload(
                            _render_pdf_page_to_image_payload(page, payload=payload),
                            settings,
                        )
                        processed_size = list(details.get("processed_size") or [])
                        image_pages.append(
                            RenderedDiagramPage(
                                page_ref=f"P{page_counter}",
                                source_document_index=source_document_index,
                                filename=processed.filename,
                                content_type=processed.content_type,
                                data=processed.data,
                                sequence_index=page_counter,
                                source_filename=processed.source_filename,
                                page_number=processed.page_number,
                                rendered_width=int(processed_size[0]) if len(processed_size) > 0 else 0,
                                rendered_height=int(processed_size[1]) if len(processed_size) > 1 else 0,
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
                        page_decisions.append(
                            {
                                "page_ref": f"P{page_counter}",
                                "source_filename": processed.source_filename or payload.filename,
                                "page_number": processed.page_number,
                                "final_route": "image_analysis",
                                "override_name": "routing_disabled",
                                "is_ambiguous": False,
                                "fallback_used": False,
                                "fallback_resolved": False,
                                "fallback_reason": None,
                            }
                        )
            else:
                page_counter += 1
                processed, details = _preprocess_diagram_payload(payload, settings)
                processed_size = list(details.get("processed_size") or [])
                image_pages.append(
                    RenderedDiagramPage(
                        page_ref=f"P{page_counter}",
                        source_document_index=source_document_index,
                        filename=processed.filename,
                        content_type=processed.content_type,
                        data=processed.data,
                        sequence_index=page_counter,
                        source_filename=processed.source_filename,
                        page_number=processed.page_number,
                        rendered_width=int(processed_size[0]) if len(processed_size) > 0 else 0,
                        rendered_height=int(processed_size[1]) if len(processed_size) > 1 else 0,
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
                page_decisions.append(
                    {
                        "page_ref": f"P{page_counter}",
                        "source_filename": processed.source_filename or payload.filename,
                        "page_number": processed.page_number,
                        "final_route": "image_analysis",
                        "override_name": "non_pdf_input",
                        "is_ambiguous": False,
                        "fallback_used": False,
                        "fallback_resolved": False,
                        "fallback_reason": None,
                    }
                )
        routing_summary = {
            "page_count": len(page_decisions),
            "image_analysis_pages": len(image_pages),
            "text_analysis_pages": 0,
            "skip_pages": 0,
            "ambiguous_before_fallback_pages": 0,
            "ambiguous_pages": 0,
            "llm_fallback_attempted_pages": 0,
            "llm_fallback_resolved_pages": 0,
            "image_pages_rendered": len(image_pages),
            "text_chars_sent": 0,
        }
        return MaterializedDiagramSources(
            image_pages=image_pages,
            text_pages=[],
            preprocess_details_list=preprocess_details_list,
            routing_summary=routing_summary,
            page_decisions=page_decisions,
        )

    image_pages: list[RenderedDiagramPage] = []
    text_pages: list[RenderedDiagramTextPage] = []
    preprocess_details_list: list[dict[str, Any]] = []
    page_decisions: list[dict[str, Any]] = []
    pending_fallback_pages: list[PendingFallbackPage] = []
    page_counter = 0
    ambiguous_before_fallback_pages = 0

    for source_document_index, payload in enumerate(diagram_payloads):
        if not _is_pdf_payload(payload):
            page_counter += 1
            processed, details = _preprocess_diagram_payload(payload, settings)
            processed_size = list(details.get("processed_size") or [])
            image_pages.append(
                RenderedDiagramPage(
                    page_ref=f"P{page_counter}",
                    source_document_index=source_document_index,
                    filename=processed.filename,
                    content_type=processed.content_type,
                    data=processed.data,
                    sequence_index=page_counter,
                    source_filename=processed.source_filename,
                    page_number=processed.page_number,
                    rendered_width=int(processed_size[0]) if len(processed_size) > 0 else 0,
                    rendered_height=int(processed_size[1]) if len(processed_size) > 1 else 0,
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
            page_decisions.append(
                {
                    "page_ref": f"P{page_counter}",
                    "source_filename": processed.source_filename or payload.filename,
                    "page_number": processed.page_number,
                    "final_route": "image_analysis",
                    "override_name": "non_pdf_input",
                    "is_ambiguous": False,
                    "fallback_used": False,
                    "fallback_resolved": False,
                    "fallback_reason": None,
                }
            )
            continue

        with fitz.open(stream=payload.data, filetype="pdf") as document:
            for page in document:
                page_counter += 1
                page_ref = f"P{page_counter}"
                native = extract_pdf_native_signals(page)
                preview = None
                decision = route_page(native, None)
                if decision.is_ambiguous:
                    ambiguous_before_fallback_pages += 1
                    preview = extract_preview_signals(page, dpi=settings.diagram_page_routing_preview_dpi)
                    decision = route_page(native, preview)
                if decision.is_ambiguous:
                    pending_fallback_pages.append(
                        PendingFallbackPage(
                            page_ref=page_ref,
                            sequence_index=page_counter,
                            source_document_index=source_document_index,
                            payload=payload,
                            source_filename=str(payload.source_filename or payload.filename),
                            page_number=page.number + 1,
                            page_text=_normalize_page_text(page.get_text("text") or page.get_text() or ""),
                            decision=decision,
                            native=native,
                            preview=preview,
                            fallback_png_bytes=_render_page_png_bytes(
                                page,
                                dpi=settings.diagram_page_routing_fallback_render_dpi,
                            ),
                        )
                    )
                    continue
                if decision.final_route == "skip" and not settings.diagram_page_routing_skip_enabled:
                    decision = replace(
                        decision,
                        final_route="image_analysis",
                        override_name="skip_disabled",
                        is_ambiguous=False,
                    )

                if decision.final_route == "text_analysis":
                    text = _normalize_page_text(page.get_text("text") or page.get_text() or "")
                    if text:
                        text_pages.append(
                            RenderedDiagramTextPage(
                                page_ref=page_ref,
                                sequence_index=page_counter,
                                source_document_index=source_document_index,
                                source_filename=payload.source_filename or payload.filename,
                                page_number=page.number + 1,
                                text=text,
                                char_count=len(text),
                                input_payload=payload,
                            )
                        )
                    else:
                        decision = replace(
                            decision,
                            final_route="image_analysis",
                            override_name="text_extraction_empty",
                            is_ambiguous=False,
                        )

                if decision.final_route == "image_analysis":
                    processed, details = _preprocess_diagram_payload(
                        _render_pdf_page_to_image_payload(page, payload=payload),
                        settings,
                    )
                    processed_size = list(details.get("processed_size") or [])
                    image_pages.append(
                        RenderedDiagramPage(
                            page_ref=page_ref,
                            source_document_index=source_document_index,
                            filename=processed.filename,
                            content_type=processed.content_type,
                            data=processed.data,
                            sequence_index=page_counter,
                            source_filename=processed.source_filename,
                            page_number=processed.page_number,
                            rendered_width=int(processed_size[0]) if len(processed_size) > 0 else 0,
                            rendered_height=int(processed_size[1]) if len(processed_size) > 1 else 0,
                            input_payload=payload,
                        )
                    )
                    preprocess_details_list.append(
                        {
                            **details,
                            "page_ref": page_ref,
                            "source_document_index": source_document_index,
                            "source_filename": processed.source_filename or payload.filename,
                            "page_number": processed.page_number,
                        }
                    )

                page_decisions.append(
                    {
                        "page_ref": page_ref,
                        "source_filename": payload.source_filename or payload.filename,
                        "page_number": page.number + 1,
                        "final_route": decision.final_route,
                        "override_name": decision.override_name,
                        "is_ambiguous": decision.is_ambiguous,
                        "fallback_used": decision.fallback_used,
                        "fallback_resolved": decision.fallback_resolved,
                        "fallback_reason": decision.fallback_reason,
                    }
                )

    fallback_decisions = _apply_llm_fallback_batch(
        pending_pages=pending_fallback_pages,
        settings=settings,
        llm=llm,
        usage_recorder=usage_recorder,
    )

    for pending in pending_fallback_pages:
        decision = fallback_decisions.get(pending.page_ref) or _fallback_unresolved(
            pending.decision,
            reason="fallback_unresolved",
        )
        if decision.final_route == "skip" and not settings.diagram_page_routing_skip_enabled:
            decision = replace(
                decision,
                final_route="image_analysis",
                override_name="skip_disabled",
                is_ambiguous=False,
            )

        if decision.final_route == "text_analysis":
            if pending.page_text:
                text_pages.append(
                    RenderedDiagramTextPage(
                        page_ref=pending.page_ref,
                        sequence_index=pending.sequence_index,
                        source_document_index=pending.source_document_index,
                        source_filename=pending.source_filename,
                        page_number=pending.page_number,
                        text=pending.page_text,
                        char_count=len(pending.page_text),
                        input_payload=pending.payload,
                    )
                )
            else:
                decision = replace(
                    decision,
                    final_route="image_analysis",
                    override_name="text_extraction_empty",
                    is_ambiguous=False,
                )

        if decision.final_route == "image_analysis":
            processed, details = _preprocess_diagram_payload(
                _render_pdf_page_to_image_payload_by_number(
                    payload=pending.payload,
                    page_number=pending.page_number,
                ),
                settings,
            )
            processed_size = list(details.get("processed_size") or [])
            image_pages.append(
                RenderedDiagramPage(
                    page_ref=pending.page_ref,
                    source_document_index=pending.source_document_index,
                    filename=processed.filename,
                    content_type=processed.content_type,
                    data=processed.data,
                    sequence_index=pending.sequence_index,
                    source_filename=processed.source_filename,
                    page_number=processed.page_number,
                    rendered_width=int(processed_size[0]) if len(processed_size) > 0 else 0,
                    rendered_height=int(processed_size[1]) if len(processed_size) > 1 else 0,
                    input_payload=pending.payload,
                )
            )
            preprocess_details_list.append(
                {
                    **details,
                    "page_ref": pending.page_ref,
                    "source_document_index": pending.source_document_index,
                    "source_filename": processed.source_filename or pending.payload.filename,
                    "page_number": processed.page_number,
                }
            )

        page_decisions.append(
            {
                "page_ref": pending.page_ref,
                "source_filename": pending.source_filename,
                "page_number": pending.page_number,
                "final_route": decision.final_route,
                "override_name": decision.override_name,
                "is_ambiguous": decision.is_ambiguous,
                "fallback_used": decision.fallback_used,
                "fallback_resolved": decision.fallback_resolved,
                "fallback_reason": decision.fallback_reason,
            }
        )

    image_pages.sort(key=lambda page: (page.sequence_index, page.page_ref))
    text_pages.sort(key=lambda page: (page.sequence_index, page.page_ref))
    preprocess_details_list.sort(key=lambda item: _page_ref_sort_key(str(item.get("page_ref") or "")))
    page_decisions.sort(key=lambda item: _page_ref_sort_key(str(item.get("page_ref") or "")))

    routing_summary = {
        "page_count": len(page_decisions),
        "image_analysis_pages": sum(1 for page in page_decisions if page["final_route"] == "image_analysis"),
        "text_analysis_pages": sum(1 for page in page_decisions if page["final_route"] == "text_analysis"),
        "skip_pages": sum(1 for page in page_decisions if page["final_route"] == "skip"),
        "ambiguous_before_fallback_pages": ambiguous_before_fallback_pages,
        "ambiguous_pages": sum(1 for page in page_decisions if page["is_ambiguous"]),
        "llm_fallback_attempted_pages": sum(1 for page in page_decisions if page["fallback_used"]),
        "llm_fallback_resolved_pages": sum(1 for page in page_decisions if page["fallback_resolved"]),
        "image_pages_rendered": len(image_pages),
        "text_chars_sent": sum(page.char_count for page in text_pages),
    }
    return MaterializedDiagramSources(
        image_pages=image_pages,
        text_pages=text_pages,
        preprocess_details_list=preprocess_details_list,
        routing_summary=routing_summary,
        page_decisions=page_decisions,
    )


def _chunk_text(text: str, *, max_chars: int) -> List[str]:
    cleaned = _normalize_page_text(text)
    if len(cleaned) <= max_chars:
        return [cleaned]
    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [cleaned[index:index + max_chars] for index in range(0, len(cleaned), max_chars)]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph_text = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(paragraph_text) <= max_chars:
            current = paragraph_text
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        for index in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[index:index + max_chars])
    if current:
        chunks.append(current)
    return chunks or [cleaned]


def build_mixed_diagram_batches(
    *,
    image_pages: List[RenderedDiagramPage],
    text_pages: List[RenderedDiagramTextPage],
    settings: MetalCompositionSettings,
) -> List[List[MixedDiagramBatchEntry]]:
    entries: list[MixedDiagramBatchEntry] = []
    for page in image_pages:
        entries.append(
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
    chunk_limit = max(1, int(settings.max_diagram_text_chars_per_page_chunk))
    for page in text_pages:
        chunks = _chunk_text(page.text, max_chars=chunk_limit)
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_page = RenderedDiagramTextPage(
                page_ref=page.page_ref,
                sequence_index=page.sequence_index,
                source_document_index=page.source_document_index,
                source_filename=page.source_filename,
                page_number=page.page_number,
                text=chunk_text,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                char_count=len(chunk_text),
                input_payload=page.input_payload,
            )
            entries.append(
                MixedDiagramBatchEntry(
                    kind="text",
                    page_ref=chunk_page.page_ref,
                    sequence_index=chunk_page.sequence_index,
                    label=_text_source_label(chunk_page),
                    source_filename=chunk_page.source_filename,
                    page_number=chunk_page.page_number,
                    text=chunk_page.text,
                    char_count=chunk_page.char_count,
                )
            )
    entries.sort(key=lambda entry: (entry.sequence_index, 0 if entry.kind == "image" else 1, entry.label))

    batches: list[list[MixedDiagramBatchEntry]] = []
    current_batch: list[MixedDiagramBatchEntry] = []
    current_image_count = 0
    current_image_bytes = 0
    current_text_chars = 0
    max_images = max(1, min(int(settings.max_diagram_images), MAX_PROVIDER_IMAGES_PER_REQUEST))
    max_image_bytes = max(1, int(settings.max_diagram_payload_bytes))
    max_text_chars = max(1, int(settings.max_diagram_text_chars_per_batch))

    def flush() -> None:
        nonlocal current_batch, current_image_count, current_image_bytes, current_text_chars
        if current_batch:
            batches.append(current_batch)
        current_batch = []
        current_image_count = 0
        current_image_bytes = 0
        current_text_chars = 0

    for entry in entries:
        encoded_size = len(entry.data) * 4 // 3 + 4 if entry.kind == "image" else 0
        would_exceed = False
        if current_batch:
            if entry.kind == "image":
                would_exceed = (
                    current_image_count >= max_images
                    or current_image_bytes + encoded_size > max_image_bytes
                )
            else:
                would_exceed = current_text_chars + entry.char_count > max_text_chars
        if would_exceed:
            flush()
        current_batch.append(entry)
        if entry.kind == "image":
            current_image_count += 1
            current_image_bytes += encoded_size
        else:
            current_text_chars += entry.char_count
    flush()
    return batches
