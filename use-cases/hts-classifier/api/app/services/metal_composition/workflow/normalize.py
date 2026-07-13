"""Pure normalization utilities and regex constants for LLM response parsing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..hts_catalog import canonicalize_hts_code

_HTS_CODE_RE = re.compile(r"\d{4}(?:\.\d{2}){1,2}\.\d{4}|\d{4}(?:\.\d{2}){0,3}|\d{6,10}")
_CHAPTER_99_RE = re.compile(r"\b99\d{2}(?:\.\d{2}){1,2}\b")
_STANDARD_CUE_RE = re.compile(
    r"\b(?:DIN|ISO|ASTM|SAE|ANSI|ASME|EN|BS)\s*[-A-Z0-9./]+\b",
    flags=re.IGNORECASE,
)


def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty model response")

    candidates = [text]
    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced_matches)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON object from model response: {text[:400]}")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _unique_strings(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _normalize_heading_hypotheses(values: List[Any]) -> List[str]:
    headings: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        normalized = " ".join(str(value).split()).strip()
        if not normalized:
            continue
        match = re.search(r"\b(\d{4})\b", normalized)
        if not match:
            continue
        heading = match.group(1)
        if heading in seen:
            continue
        seen.add(heading)
        headings.append(heading)
    return headings


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def _normalize_candidate_code(value: Any) -> str:
    return canonicalize_hts_code(value)


def _code_digit_count(code: str) -> int:
    return sum(1 for char in str(code) if char.isdigit())


def _normalize_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))

    text = _normalize_text(value).lower()
    if not text:
        return 0.0

    percent_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if percent_match:
        return max(0.0, min(1.0, float(percent_match.group(1)) / 100.0))

    numeric_match = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if numeric_match:
        return max(0.0, min(1.0, float(text)))

    confidence_map = {
        "very low": 0.12,
        "low": 0.28,
        "medium low": 0.4,
        "moderate": 0.55,
        "medium": 0.55,
        "medium high": 0.68,
        "high": 0.82,
        "very high": 0.94,
    }
    if text in confidence_map:
        return confidence_map[text]

    for label, normalized in confidence_map.items():
        if re.search(rf"\b{re.escape(label)}\b", text):
            return normalized

    return 0.0


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = _normalize_text(value)
    if not text:
        return 0
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else 0


def _normalize_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_text(value)
    if not text:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def _summarize_text(value: Any, *, max_chars: int = 480) -> str:
    text = _normalize_text(value)
    if len(text) <= max_chars:
        return text
    trimmed = text[: max_chars - 3].rstrip()
    return f"{trimmed}..."


def _normalize_reasoning_key(value: Any) -> str:
    normalized = _normalize_text(value).lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def _extract_hts_codes_from_text(value: Any) -> List[str]:
    codes: List[str] = []
    seen = set()
    text = _normalize_text(value)
    if not text:
        return codes
    for match in _HTS_CODE_RE.findall(text):
        normalized = _normalize_candidate_code(match)
        if not normalized or normalized in seen:
            continue
        if _code_digit_count(normalized) < 6:
            continue
        if normalized.startswith("99"):
            continue
        seen.add(normalized)
        codes.append(normalized)
    return codes


def _extract_ruling_numbers_from_text(value: Any) -> List[str]:
    text = _normalize_text(value)
    if not text:
        return []
    matches = re.findall(r"\b(?:NY|HQ|H|N)\s*[A-Z]?\d{5,6}\b", text, flags=re.IGNORECASE)
    normalized = [match.replace(" ", "").upper() for match in matches]
    return _unique_strings(normalized)


def _is_generic_function_label(value: Any) -> bool:
    text = _normalize_text(value).lower()
    if not text:
        return True
    generic_patterns = (
        r"\btop\s*\d+\b",
        r"\bpriority\b",
        r"\bpsc\b",
        r"\bbusiness\s+segment\b",
        r"\bsegment\b",
        r"\bsite\b",
        r"\bcommodity\b",
        r"\bclassification\b",
        r"\bcategory\b",
        r"\bspend\b",
        r"\bportfolio\b",
        r"\branking\b",
    )
    return any(re.search(pattern, text) for pattern in generic_patterns)
