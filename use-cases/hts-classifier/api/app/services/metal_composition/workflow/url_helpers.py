"""URL extraction and legal-source detection helpers."""

from __future__ import annotations

import re
from typing import Any, List
from urllib.parse import parse_qs, urlparse

from .normalize import _CHAPTER_99_RE, _normalize_text


def _extract_url(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("http://", "https://")):
            return text
        match = re.search(r"https?://\S+", text)
        return match.group(0).rstrip(").,") if match else ""
    if isinstance(value, dict):
        for key in ("url", "source_url", "link", "href"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _is_cbp_ruling_url(value: Any) -> bool:
    url = _extract_url(value)
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.netloc.lower() == "rulings.cbp.gov"


def _is_usitc_hts_url(value: Any) -> bool:
    url = _extract_url(value)
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.netloc.lower() == "hts.usitc.gov"


def _is_official_legal_source_url(value: Any) -> bool:
    return _is_cbp_ruling_url(value) or _is_usitc_hts_url(value)


def _extract_ruling_number(raw_value: Any, *, fallback_url: str = "") -> str:
    text = _normalize_text(raw_value)
    if text:
        direct = re.search(r"\b(?:NY|HQ|H|N)\s*[A-Z]?\d{5,6}\b", text, flags=re.IGNORECASE)
        if direct:
            return direct.group(0).replace(" ", "").upper()
        fallback = re.search(r"\bN\d{5,6}\b", text, flags=re.IGNORECASE)
        if fallback:
            return fallback.group(0).upper()

    if fallback_url:
        parsed = urlparse(fallback_url)
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts and path_parts[-2:-1] == ["ruling"]:
            return path_parts[-1].upper()
        query_doc_id = parse_qs(parsed.query).get("doc_id", [])
        if query_doc_id:
            candidate = query_doc_id[0].replace("+", " ").strip()
            match = re.search(r"(?:NY|HQ|H|N)\s*[A-Z]?\d{5,6}", candidate, flags=re.IGNORECASE)
            if match:
                return match.group(0).replace(" ", "").upper()
    return ""


def _canonical_cbp_ruling_url(ruling_number: Any) -> str:
    normalized = _extract_ruling_number(ruling_number)
    return f"https://rulings.cbp.gov/ruling/{normalized}" if normalized else ""


def _normalize_chapter_99_codes(values: List[Any]) -> List[str]:
    codes: List[str] = []
    seen = set()
    for value in values:
        for match in _CHAPTER_99_RE.findall(_normalize_text(value)):
            if match in seen:
                continue
            seen.add(match)
            codes.append(match)
    return codes
