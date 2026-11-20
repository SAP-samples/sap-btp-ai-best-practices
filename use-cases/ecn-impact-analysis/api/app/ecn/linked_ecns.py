"""Extract explicit ECN references from source descriptions without model inference."""

from __future__ import annotations

import re


# Keep the numeric identity intact, including leading zeroes. Boundaries prevent
# matching inside part numbers or accepting the numeric prefix of an ECN suffix.
# The same prefix grammar is used by the repository to match stored identifiers.
ECN_PREFIX_PATTERN = r"ECNS?\s*(?:(?:NO\.?|NUMBER)\s*)?[-‐‑‒–—−:#_]?\s*"
_REFERENCE = re.compile(
    rf"(?<!\w){ECN_PREFIX_PATTERN}([0-9]{{1,77}})(?!\w|[.,][0-9])",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(rf"{ECN_PREFIX_PATTERN}([0-9]{{1,77}})", re.IGNORECASE)


def normalize_ecn_number(value: str | None) -> str | None:
    """Normalize an entire explicit identifier, never a substring or bare number."""
    match = _IDENTIFIER.fullmatch(value.strip()) if value else None
    return f"ECN{match[1]}" if match else None


def extract_linked_ecn_numbers(
    description_text: str | None,
    description_summary: str | None = None,
    *,
    current_ecn_number: str | None = None,
) -> list[str]:
    """Return unique non-self references in source order from both raw text fields."""
    current = normalize_ecn_number(current_ecn_number)
    references: dict[str, None] = {}
    for text in (description_text, description_summary):
        for match in _REFERENCE.finditer(text or ""):
            number = f"ECN{match[1]}"
            if number != current:
                references.setdefault(number, None)
    return list(references)
