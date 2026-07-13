"""Section 232 source document ingestion, persistence, and retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pypdf import PdfReader

from app.models.metal_composition import Section232MatchEvidence, Section232SourceSummary
from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings
from .hts_catalog import canonicalize_hts_code
from .persistence_common import qualified_table as _qualified_table
from .timing import utc_now_iso, utc_now_iso_after


_HTS_TOKEN_PATTERN = re.compile(
    r"\b\d{4}\.\d{2}(?:\.(?:\d{4}|\d{2}(?:\.\d{2})?))?\b|\b(?:\d{4}|\d{6,10})\b"
)
_FOUR_DIGIT_HTS_PREFIX_RE = re.compile(
    r"(?:\bHTS(?:US)?\b|\bHTS\s+codes?\b|\bHarmonized\s+Tariff\s+Schedule\b|"
    r"\btariff\s+schedule\b|\bheadings?\b|\bsubheadings?\b)[^\n.;:()]{0,80}$",
    re.IGNORECASE,
)
_FOUR_DIGIT_HTS_SUFFIX_RE = re.compile(
    r"^[^\n.;:()]{0,40}(?:\bHTS(?:US)?\b|\bHTS\s+codes?\b|\bheadings?\b|\bsubheadings?\b)",
    re.IGNORECASE,
)
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_TEXT_SOURCE_ORDER = {"plain": 0, "layout": 1}


def _extract_page_text(page: Any, *, layout_aware: bool = False) -> str:
    try:
        if layout_aware:
            return (page.extract_text(extraction_mode="layout") or "").strip()
        return (page.extract_text() or "").strip()
    except TypeError:
        try:
            return (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            return ""


def _page_excerpt(text: str, *, max_chars: int = 280) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    trimmed = normalized[: max_chars - 3].rstrip()
    return f"{trimmed}..."


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _paragraph_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    lower_bound = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(text or ""):
        if match.end() <= start:
            lower_bound = match.end()
            continue
        return lower_bound, match.start()
    return lower_bound, len(text or "")


def _context_window(text: str, start: int, end: int, *, max_chars: int = 420) -> str:
    paragraph_start, paragraph_end = _paragraph_bounds(text, start, end)
    paragraph = _normalize_text(text[paragraph_start:paragraph_end])
    if paragraph and len(paragraph) <= max_chars:
        return paragraph

    snippet_start = max(0, start - (max_chars // 2))
    snippet_end = min(len(text or ""), end + (max_chars // 2))
    snippet = _normalize_text((text or "")[snippet_start:snippet_end])
    if len(snippet) <= max_chars:
        return snippet
    trimmed = snippet[: max_chars - 3].rstrip()
    return f"{trimmed}..."


def _has_four_digit_hts_context(text: str, start: int, end: int) -> bool:
    prefix = (text or "")[max(0, start - 100) : start]
    suffix = (text or "")[end : min(len(text or ""), end + 60)]
    return bool(_FOUR_DIGIT_HTS_PREFIX_RE.search(prefix) or _FOUR_DIGIT_HTS_SUFFIX_RE.search(suffix))


def _extract_hts_occurrences_from_text(
    text: str,
    *,
    page_number: int,
    text_source: str,
) -> List[Dict[str, Any]]:
    occurrences: List[Dict[str, Any]] = []
    seen = set()
    for match in _HTS_TOKEN_PATTERN.finditer(text or ""):
        raw_value = match.group(0)
        if re.fullmatch(r"\d{4}", raw_value) and not _has_four_digit_hts_context(
            text,
            match.start(),
            match.end(),
        ):
            continue
        normalized = canonicalize_hts_code(raw_value)
        if not normalized:
            continue
        context_text = _context_window(text, match.start(), match.end())
        key = (normalized, raw_value, context_text)
        if key in seen:
            continue
        seen.add(key)
        occurrences.append(
            {
                "page_number": int(page_number),
                "matched_text": raw_value,
                "normalized_hts_code": normalized,
                "context_text": context_text,
                "text_sources": [text_source],
            }
        )
    return occurrences


def _merge_hts_occurrences(*occurrence_lists: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for occurrence_list in occurrence_lists:
        for occurrence in occurrence_list:
            key = (
                int(occurrence.get("page_number") or 0),
                str(occurrence.get("normalized_hts_code") or ""),
                str(occurrence.get("matched_text") or ""),
                str(occurrence.get("context_text") or ""),
            )
            current = merged.get(key)
            if current is None:
                merged[key] = {
                    "page_number": key[0],
                    "matched_text": key[2],
                    "normalized_hts_code": key[1],
                    "context_text": key[3],
                    "text_sources": list(occurrence.get("text_sources") or []),
                }
                continue
            text_sources = {
                str(source)
                for source in [*current.get("text_sources", []), *(occurrence.get("text_sources") or [])]
                if str(source) in _TEXT_SOURCE_ORDER
            }
            current["text_sources"] = sorted(text_sources, key=lambda value: _TEXT_SOURCE_ORDER[value])
    return list(merged.values())


@dataclass(frozen=True)
class ExtractedSection232Source:
    page_count: int
    extraction_status: str
    full_text: str
    page_texts: List[Dict[str, Any]]
    hts_mentions: List[str]
    warnings: List[str]


@dataclass(frozen=True)
class PersistedSection232Source:
    source_id: str
    filename: str
    size_bytes: int
    page_count: int
    extraction_status: str
    full_text: str
    page_texts: List[Dict[str, Any]]
    hts_mentions: List[str]
    warnings: List[str]
    content_sha256: str
    uploaded_at: str

    def to_summary(self) -> Section232SourceSummary:
        return Section232SourceSummary(
            source_id=self.source_id,
            filename=self.filename,
            size_bytes=self.size_bytes,
            page_count=self.page_count,
            extraction_status=str(self.extraction_status or "completed"),
            hts_mention_count=len(self.hts_mentions),
            uploaded_at=self.uploaded_at,
            warnings=list(self.warnings),
        )


@dataclass(frozen=True)
class Section232SourceSnippet:
    source_id: str
    filename: str
    page_number: int
    text: str
    matched_codes: List[str]
    score: int


def _safe_json_loads(raw: Any, *, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(str(raw))
    except Exception:
        return default


def _normalize_eligible_hts_codes(codes: Sequence[str]) -> List[str]:
    seen = set()
    normalized_codes: List[str] = []
    for raw_code in codes:
        normalized = canonicalize_hts_code(raw_code)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_codes.append(normalized)
    normalized_codes.sort()
    return normalized_codes


def extract_section_232_pdf(content: bytes) -> ExtractedSection232Source:
    warnings: List[str] = []
    page_texts: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    all_mentions: List[str] = []
    seen_mentions = set()

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        return ExtractedSection232Source(
            page_count=0,
            extraction_status="failed",
            full_text="",
            page_texts=[],
            hts_mentions=[],
            warnings=[f"Failed to read PDF: {exc}"],
        )

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = _extract_page_text(page)
        except Exception as exc:  # noqa: BLE001
            text = ""
            warnings.append(f"Failed to extract text for page {page_index}: {exc}")
        try:
            layout_aware_text = _extract_page_text(page, layout_aware=True)
        except Exception as exc:  # noqa: BLE001
            layout_aware_text = ""
            warnings.append(f"Failed to extract layout-aware text for page {page_index}: {exc}")
        normalized_layout_text = layout_aware_text or text
        plain_occurrences = _extract_hts_occurrences_from_text(
            text,
            page_number=page_index,
            text_source="plain",
        )
        layout_occurrences: List[Dict[str, Any]] = []
        if normalized_layout_text and normalized_layout_text != text:
            layout_occurrences = _extract_hts_occurrences_from_text(
                normalized_layout_text,
                page_number=page_index,
                text_source="layout",
            )
        page_occurrences = _merge_hts_occurrences(plain_occurrences, layout_occurrences)
        page_mentions: List[str] = []
        page_seen_mentions = set()
        for occurrence in page_occurrences:
            normalized = canonicalize_hts_code(occurrence.get("normalized_hts_code"))
            if not normalized or normalized in page_seen_mentions:
                continue
            page_seen_mentions.add(normalized)
            page_mentions.append(normalized)
            if normalized not in seen_mentions:
                seen_mentions.add(normalized)
                all_mentions.append(normalized)
        page_texts.append(
            {
                "page_number": page_index,
                "text": text,
                "layout_aware_text": normalized_layout_text,
                "page_excerpt": _page_excerpt(normalized_layout_text or text),
                "char_count": len(normalized_layout_text or text),
                "hts_mentions": page_mentions,
                "hts_occurrences": page_occurrences,
            }
        )
        if normalized_layout_text:
            full_text_parts.append(f"[Page {page_index}]\n{normalized_layout_text}")

    extraction_status = "completed"
    if warnings and full_text_parts:
        extraction_status = "partial"
    elif warnings and not full_text_parts:
        extraction_status = "failed"

    return ExtractedSection232Source(
        page_count=len(reader.pages),
        extraction_status=extraction_status,
        full_text="\n\n".join(full_text_parts),
        page_texts=page_texts,
        hts_mentions=all_mentions,
        warnings=warnings,
    )


def extract_hts_mentions(text: str) -> List[str]:
    mentions: List[str] = []
    seen = set()
    for occurrence in _extract_hts_occurrences_from_text(text, page_number=0, text_source="plain"):
        normalized = canonicalize_hts_code(occurrence.get("normalized_hts_code"))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        mentions.append(normalized)
    return mentions


def collect_section_232_match_evidence(
    sources: Sequence[PersistedSection232Source],
    *,
    hts_code: str,
    source_ids: Sequence[str] | None = None,
    source_pages: Sequence[int] | None = None,
) -> List[Section232MatchEvidence]:
    normalized_hts_code = canonicalize_hts_code(hts_code)
    if not normalized_hts_code:
        return []

    allowed_source_ids = {str(value) for value in source_ids or [] if str(value)}
    allowed_pages = {
        int(value)
        for value in source_pages or []
        if str(value).strip() and str(value).strip().isdigit()
    }
    evidence_items: List[Section232MatchEvidence] = []
    seen = set()
    for source in sources:
        if allowed_source_ids and source.source_id not in allowed_source_ids:
            continue
        for page in source.page_texts:
            page_number = int(page.get("page_number") or 0)
            if allowed_pages and page_number not in allowed_pages:
                continue
            for occurrence in page.get("hts_occurrences") or []:
                normalized = canonicalize_hts_code(
                    occurrence.get("normalized_hts_code") or occurrence.get("matched_text")
                )
                if normalized != normalized_hts_code:
                    continue
                text_sources = [
                    str(source_name)
                    for source_name in occurrence.get("text_sources") or []
                    if str(source_name) in _TEXT_SOURCE_ORDER
                ]
                dedupe_key = (
                    source.source_id,
                    page_number,
                    str(occurrence.get("matched_text") or ""),
                    str(occurrence.get("context_text") or ""),
                    tuple(text_sources),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                evidence_items.append(
                    Section232MatchEvidence(
                        source_id=source.source_id,
                        source_filename=source.filename,
                        page_number=page_number,
                        matched_text=str(occurrence.get("matched_text") or ""),
                        normalized_hts_code=normalized_hts_code,
                        context_text=str(occurrence.get("context_text") or ""),
                        text_sources=text_sources,
                    )
                )
    return evidence_items


def expand_hts_code_family(code: str) -> List[str]:
    normalized = canonicalize_hts_code(code)
    if not normalized:
        return []
    parts = normalized.split(".")
    out = [normalized]
    if len(parts) >= 3:
        out.append(f"{parts[0]}.{parts[1]}.{parts[2][:2]}")
    if len(parts) >= 2:
        out.append(f"{parts[0]}.{parts[1]}")
    if parts:
        out.append(parts[0])
    seen = set()
    deduped: List[str] = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def build_section_232_query_codes(codes: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    seen = set()
    for code in codes:
        for item in expand_hts_code_family(code):
            if item in seen:
                continue
            seen.add(item)
            expanded.append(item)
    return expanded


def retrieve_section_232_snippets(
    sources: Sequence[PersistedSection232Source],
    *,
    hts_codes: Sequence[str],
    metal_keywords: Sequence[str],
    max_sources: int,
    max_pages_per_source: int,
    max_prompt_chars: int,
) -> List[Section232SourceSnippet]:
    query_codes = set(build_section_232_query_codes(hts_codes))
    metal_tokens = {token.lower() for token in metal_keywords if token}
    ranked: List[Section232SourceSnippet] = []

    for source in sources:
        per_source: List[Section232SourceSnippet] = []
        for page in source.page_texts:
            text = str(page.get("text") or page.get("layout_aware_text") or page.get("page_excerpt") or "").strip()
            if not text:
                continue
            page_mentions = {
                canonicalize_hts_code(str(mention))
                for mention in page.get("hts_mentions", []) or []
                if canonicalize_hts_code(str(mention))
            }
            matched_codes = sorted(query_codes.intersection(page_mentions))
            if page_mentions and not matched_codes:
                continue
            lowered = text.lower()
            metal_hits = sum(1 for token in metal_tokens if token in lowered)
            section_232_hits = int("section 232" in lowered) + int("additional duties" in lowered)
            score = len(matched_codes) * 100 + metal_hits * 10 + section_232_hits * 5
            if score <= 0:
                continue
            per_source.append(
                Section232SourceSnippet(
                    source_id=source.source_id,
                    filename=source.filename,
                    page_number=int(page.get("page_number") or 0),
                    text=text,
                    matched_codes=matched_codes,
                    score=score,
                )
            )
        per_source.sort(key=lambda item: (-item.score, item.page_number))
        ranked.extend(per_source[:max_pages_per_source])

    ranked.sort(key=lambda item: (-item.score, item.filename, item.page_number))
    selected: List[Section232SourceSnippet] = []
    used_sources = set()
    total_chars = 0
    for snippet in ranked:
        if snippet.source_id not in used_sources and len(used_sources) >= max_sources:
            continue
        snippet_chars = len(snippet.text)
        if selected and total_chars + snippet_chars > max_prompt_chars:
            continue
        selected.append(snippet)
        used_sources.add(snippet.source_id)
        total_chars += snippet_chars
    return selected


class InMemorySection232SourceStore:
    """Test double for Section 232 source persistence."""

    def __init__(self) -> None:
        self._sources: List[PersistedSection232Source] = []
        self._eligible_hts_codes: List[str] = []

    def _next_uploaded_at(self) -> str:
        latest_uploaded_at = max((source.uploaded_at for source in self._sources), default="")
        return utc_now_iso_after(latest_uploaded_at or None)

    def save_source(
        self,
        *,
        filename: str,
        size_bytes: int,
        extracted: ExtractedSection232Source,
    ) -> PersistedSection232Source:
        source = PersistedSection232Source(
            source_id=str(uuid.uuid4()),
            filename=filename,
            size_bytes=int(size_bytes),
            page_count=int(extracted.page_count),
            extraction_status=str(extracted.extraction_status or "completed"),
            full_text=extracted.full_text,
            page_texts=list(extracted.page_texts),
            hts_mentions=list(extracted.hts_mentions),
            warnings=list(extracted.warnings),
            content_sha256=hashlib.sha256(extracted.full_text.encode("utf-8")).hexdigest(),
            uploaded_at=self._next_uploaded_at(),
        )
        self._sources.append(source)
        return source

    def list_sources(self) -> List[PersistedSection232Source]:
        return sorted(self._sources, key=lambda item: item.uploaded_at, reverse=True)

    def clear_sources(self) -> None:
        self._sources = []

    def delete_sources(self, source_ids: Sequence[str]) -> int:
        target_ids = {str(source_id or "").strip() for source_id in source_ids if str(source_id or "").strip()}
        if not target_ids:
            return 0
        original_count = len(self._sources)
        self._sources = [source for source in self._sources if source.source_id not in target_ids]
        return original_count - len(self._sources)

    def clear_eligible_hts_codes(self) -> None:
        self._eligible_hts_codes = []

    def has_sources(self) -> bool:
        return bool(self._sources)

    def list_eligible_hts_codes(self) -> List[str]:
        return list(self._eligible_hts_codes)

    def replace_eligible_hts_codes(self, codes: Sequence[str]) -> List[str]:
        self._eligible_hts_codes = _normalize_eligible_hts_codes(codes)
        return self.list_eligible_hts_codes()

    def append_eligible_hts_codes(self, codes: Sequence[str]) -> List[str]:
        merged_codes = [*self._eligible_hts_codes, *codes]
        self._eligible_hts_codes = _normalize_eligible_hts_codes(merged_codes)
        return self.list_eligible_hts_codes()

    def retrieve_snippets(
        self,
        *,
        hts_codes: Sequence[str],
        metal_keywords: Sequence[str],
        settings: MetalCompositionSettings,
    ) -> List[Section232SourceSnippet]:
        return retrieve_section_232_snippets(
            self.list_sources(),
            hts_codes=hts_codes,
            metal_keywords=metal_keywords,
            max_sources=settings.section_232_max_sources,
            max_pages_per_source=settings.section_232_max_pages_per_source,
            max_prompt_chars=settings.section_232_max_prompt_chars,
        )


class Section232SourceStore:
    """Persist Section 232 source documents in SAP HANA."""

    def __init__(
        self,
        settings: MetalCompositionSettings,
        *,
        connection: Optional[HANAConnection] = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or HANAConnection()
        self.schema = settings.section_232_hana_schema or settings.hana_schema or None
        self.sources_table = settings.section_232_sources_table
        self.eligible_hts_codes_table = settings.section_232_hts_codes_table
        self._initialize()

    def _initialize(self) -> None:
        if not self.connection.table_exists(self.sources_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.sources_table, self.schema)} (
                    "SOURCE_ID" NVARCHAR(36) PRIMARY KEY,
                    "FILENAME" NVARCHAR(512) NOT NULL,
                    "SIZE_BYTES" BIGINT NOT NULL,
                    "PAGE_COUNT" BIGINT NOT NULL,
                    "EXTRACTION_STATUS" NVARCHAR(32) NOT NULL,
                    "FULL_TEXT" NCLOB,
                    "PAGE_TEXT_JSON" NCLOB,
                    "HTS_MENTIONS_JSON" NCLOB,
                    "WARNINGS_JSON" NCLOB,
                    "CONTENT_SHA256" NVARCHAR(64) NOT NULL,
                    "UPLOADED_AT" NVARCHAR(64) NOT NULL
                )
                """
            )
            self.connection._ensure_index(  # noqa: SLF001
                self.sources_table,
                schema=self.schema,
                columns=("UPLOADED_AT",),
            )
        if not self.connection.table_exists(self.eligible_hts_codes_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.eligible_hts_codes_table, self.schema)} (
                    "HTS_CODE" NVARCHAR(32) PRIMARY KEY,
                    "UPDATED_AT" NVARCHAR(64) NOT NULL
                )
                """
            )

    def _next_uploaded_at(self) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'''
                SELECT MAX("UPLOADED_AT")
                FROM {_qualified_table(self.sources_table, self.schema)}
                '''
            )
            row = cursor.fetchone()
        latest_uploaded_at = str((row or [None])[0] or "").strip()
        return utc_now_iso_after(latest_uploaded_at or None)

    def save_source(
        self,
        *,
        filename: str,
        size_bytes: int,
        extracted: ExtractedSection232Source,
    ) -> PersistedSection232Source:
        source = PersistedSection232Source(
            source_id=str(uuid.uuid4()),
            filename=filename,
            size_bytes=int(size_bytes),
            page_count=int(extracted.page_count),
            extraction_status=str(extracted.extraction_status or "completed"),
            full_text=extracted.full_text,
            page_texts=list(extracted.page_texts),
            hts_mentions=list(extracted.hts_mentions),
            warnings=list(extracted.warnings),
            content_sha256=hashlib.sha256(extracted.full_text.encode("utf-8")).hexdigest(),
            uploaded_at=self._next_uploaded_at(),
        )
        self.connection.execute(
            f"""
            INSERT INTO {_qualified_table(self.sources_table, self.schema)} (
                "SOURCE_ID",
                "FILENAME",
                "SIZE_BYTES",
                "PAGE_COUNT",
                "EXTRACTION_STATUS",
                "FULL_TEXT",
                "PAGE_TEXT_JSON",
                "HTS_MENTIONS_JSON",
                "WARNINGS_JSON",
                "CONTENT_SHA256",
                "UPLOADED_AT"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                source.source_id,
                source.filename,
                source.size_bytes,
                source.page_count,
                source.extraction_status,
                source.full_text,
                json.dumps(source.page_texts, ensure_ascii=True),
                json.dumps(source.hts_mentions, ensure_ascii=True),
                json.dumps(source.warnings, ensure_ascii=True),
                source.content_sha256,
                source.uploaded_at,
            ],
        )
        return source

    def list_sources(self) -> List[PersistedSection232Source]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    "SOURCE_ID",
                    "FILENAME",
                    "SIZE_BYTES",
                    "PAGE_COUNT",
                    "EXTRACTION_STATUS",
                    "FULL_TEXT",
                    "PAGE_TEXT_JSON",
                    "HTS_MENTIONS_JSON",
                    "WARNINGS_JSON",
                    "CONTENT_SHA256",
                    "UPLOADED_AT"
                FROM {_qualified_table(self.sources_table, self.schema)}
                ORDER BY "UPLOADED_AT" DESC
                """
            )
            rows = cursor.fetchall()
        out: List[PersistedSection232Source] = []
        for row in rows:
            out.append(
                PersistedSection232Source(
                    source_id=str(row[0]),
                    filename=str(row[1]),
                    size_bytes=int(row[2] or 0),
                    page_count=int(row[3] or 0),
                    extraction_status=str(row[4] or "completed"),
                    full_text=str(row[5] or ""),
                    page_texts=_safe_json_loads(row[6], default=[]),
                    hts_mentions=_safe_json_loads(row[7], default=[]),
                    warnings=_safe_json_loads(row[8], default=[]),
                    content_sha256=str(row[9] or ""),
                    uploaded_at=str(row[10] or ""),
                )
            )
        return out

    def clear_sources(self) -> None:
        self.connection.execute(f'DELETE FROM {_qualified_table(self.sources_table, self.schema)}')

    def delete_sources(self, source_ids: Sequence[str]) -> int:
        target_ids = [str(source_id or "").strip() for source_id in source_ids if str(source_id or "").strip()]
        if not target_ids:
            return 0
        placeholders = ", ".join("?" for _ in target_ids)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'''
                DELETE FROM {_qualified_table(self.sources_table, self.schema)}
                WHERE "SOURCE_ID" IN ({placeholders})
                ''',
                target_ids,
            )
            return int(cursor.rowcount or 0)

    def clear_eligible_hts_codes(self) -> None:
        self.connection.execute(
            f'DELETE FROM {_qualified_table(self.eligible_hts_codes_table, self.schema)}'
        )

    def has_sources(self) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'SELECT 1 FROM {_qualified_table(self.sources_table, self.schema)} LIMIT 1'
            )
            return cursor.fetchone() is not None

    def list_eligible_hts_codes(self) -> List[str]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                f'''
                SELECT "HTS_CODE"
                FROM {_qualified_table(self.eligible_hts_codes_table, self.schema)}
                ORDER BY "HTS_CODE" ASC
                '''
            )
            rows = cursor.fetchall()
        return [str(row[0]) for row in rows if row and row[0]]

    def replace_eligible_hts_codes(self, codes: Sequence[str]) -> List[str]:
        normalized_codes = _normalize_eligible_hts_codes(codes)
        self.connection.execute(
            f'DELETE FROM {_qualified_table(self.eligible_hts_codes_table, self.schema)}'
        )
        self._insert_eligible_hts_codes(normalized_codes)
        return normalized_codes

    def append_eligible_hts_codes(self, codes: Sequence[str]) -> List[str]:
        normalized_codes = _normalize_eligible_hts_codes(codes)
        if not normalized_codes:
            return self.list_eligible_hts_codes()
        existing_codes = set(self.list_eligible_hts_codes())
        missing_codes = [code for code in normalized_codes if code not in existing_codes]
        self._insert_eligible_hts_codes(missing_codes)
        return self.list_eligible_hts_codes()

    def _insert_eligible_hts_codes(self, codes: Sequence[str]) -> None:
        if not codes:
            return
        updated_at = utc_now_iso()
        self.connection.executemany(
            f'''
            INSERT INTO {_qualified_table(self.eligible_hts_codes_table, self.schema)} (
                "HTS_CODE",
                "UPDATED_AT"
            ) VALUES (?, ?)
            ''',
            [(code, updated_at) for code in codes],
        )

    def retrieve_snippets(
        self,
        *,
        hts_codes: Sequence[str],
        metal_keywords: Sequence[str],
        settings: MetalCompositionSettings,
    ) -> List[Section232SourceSnippet]:
        return retrieve_section_232_snippets(
            self.list_sources(),
            hts_codes=hts_codes,
            metal_keywords=metal_keywords,
            max_sources=settings.section_232_max_sources,
            max_pages_per_source=settings.section_232_max_pages_per_source,
            max_prompt_chars=settings.section_232_max_prompt_chars,
        )
