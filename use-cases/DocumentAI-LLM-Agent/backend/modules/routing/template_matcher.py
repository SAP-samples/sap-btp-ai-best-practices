"""
template_matcher.py
-------------------
Matches a supplier name against available SAP Document AI templates
using fuzzy and semantic similarity scoring.

Responsibilities:
  - Load and normalize template metadata
  - Compare supplier name against template name and description
  - Fuzzy matching (rapidfuzz with difflib fallback)
  - Semantic similarity scoring
  - Confidence scoring and threshold evaluation
  - Return best match above threshold
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Configurable match threshold (0.0 – 1.0)
TEMPLATE_MATCH_THRESHOLD: float = 0.75

# Weight applied to description score vs name score
_NAME_WEIGHT: float = 1.0
_DESC_WEIGHT: float = 0.85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, replace separators, collapse spaces."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[_\-/\\|,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_score(a: str, b: str) -> float:
    """
    Compute fuzzy similarity between two strings.

    Uses rapidfuzz if available (preferred), falls back to difflib.
    Returns a score in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0

    try:
        from rapidfuzz import fuzz

        scores = [
            fuzz.ratio(a, b) / 100.0,
            fuzz.partial_ratio(a, b) / 100.0,
            fuzz.token_sort_ratio(a, b) / 100.0,
            fuzz.token_set_ratio(a, b) / 100.0,
        ]
        return max(scores)

    except ImportError:
        logger.debug("rapidfuzz not available, using difflib fallback.")
        from difflib import SequenceMatcher

        scores = [
            SequenceMatcher(None, a, b).ratio(),
            SequenceMatcher(None, sorted(a.split()), sorted(b.split())).ratio(),
        ]
        return max(scores)


def _substring_bonus(needle: str, haystack: str) -> float:
    """
    Return a bonus score if needle (or its significant words) appear in haystack.
    """
    if not needle or not haystack:
        return 0.0

    # Full substring match
    if needle in haystack:
        return 0.95

    # Word-level match: fraction of supplier words found in haystack
    words = [w for w in needle.split() if len(w) > 2]
    if not words:
        return 0.0

    matched = sum(1 for w in words if w in haystack)
    return (matched / len(words)) * 0.85


def _score_template(
    supplier_normalized: str,
    template: dict[str, Any],
) -> float:
    """
    Compute a composite similarity score between a supplier name and a template.

    Evaluates:
      1. Fuzzy similarity: supplier ↔ template name
      2. Fuzzy similarity: supplier ↔ template description
      3. Substring bonus: supplier words in template name/description

    Returns:
        Best composite score in [0.0, 1.0].
    """
    template_name = _normalize(template.get("name") or "")
    template_desc = _normalize(template.get("description") or "")

    scores: list[float] = []

    # 1. Name similarity
    if template_name:
        name_score = _fuzzy_score(supplier_normalized, template_name) * _NAME_WEIGHT
        scores.append(name_score)
        logger.debug("  name_score='%s' vs '%s': %.3f", supplier_normalized, template_name, name_score)

    # 2. Description similarity
    if template_desc:
        desc_score = _fuzzy_score(supplier_normalized, template_desc) * _DESC_WEIGHT
        scores.append(desc_score)
        logger.debug("  desc_score='%s' vs '%s': %.3f", supplier_normalized, template_desc[:60], desc_score)

    # 3. Substring bonus (name)
    if template_name:
        bonus_name = _substring_bonus(supplier_normalized, template_name)
        if bonus_name > 0:
            scores.append(bonus_name)
            logger.debug("  substring_bonus (name): %.3f", bonus_name)

    # 4. Substring bonus (description)
    if template_desc:
        bonus_desc = _substring_bonus(supplier_normalized, template_desc)
        if bonus_desc > 0:
            scores.append(bonus_desc)
            logger.debug("  substring_bonus (desc): %.3f", bonus_desc)

    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_best_template(
    supplier_normalized: str,
    templates: list[dict[str, Any]],
    threshold: float = TEMPLATE_MATCH_THRESHOLD,
) -> dict[str, Any] | None:
    """
    Find the best matching template for a normalized supplier name.

    Args:
        supplier_normalized: Normalized supplier name string.
        templates: List of template dicts from SAP Document AI.
        threshold: Minimum score to accept a match (default: 0.75).

    Returns:
        Match result dict if a template scores above threshold, else None.
        Match dict keys:
            - template (dict): full template object
            - template_id (str): template ID
            - template_name (str): template name
            - score (float): raw score 0.0–1.0
            - confidence_pct (float): score as percentage
            - matched (bool): True
    """
    if not supplier_normalized:
        logger.warning("Cannot match: supplier_normalized is empty.")
        return None

    if not templates:
        logger.warning("Cannot match: templates list is empty.")
        return None

    logger.info(
        "Matching supplier '%s' against %d templates...",
        supplier_normalized,
        len(templates),
    )

    scored: list[tuple[float, dict]] = []

    for template in templates:
        score = _score_template(supplier_normalized, template)
        scored.append((score, template))
        logger.debug(
            "  Template '%s' → score: %.3f",
            template.get("name", "N/A"),
            score,
        )

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return None

    best_score, best_template = scored[0]

    if best_score >= threshold:
        template_id = (
            best_template.get("id")
            or best_template.get("templateId")
            or best_template.get("template_id")
        )
        template_name = best_template.get("name") or "Unknown Template"

        logger.info(
            "Template match found: '%s' (score: %.3f / %.0f%%)",
            template_name,
            best_score,
            best_score * 100,
        )

        return {
            "template": best_template,
            "template_id": template_id,
            "template_name": template_name,
            "score": round(best_score, 4),
            "confidence_pct": round(best_score * 100, 1),
            "matched": True,
        }

    logger.warning(
        "No template match above threshold. Best: '%s' (score: %.3f, threshold: %.2f)",
        scored[0][1].get("name", "N/A"),
        best_score,
        threshold,
    )
    return None


def match_supplier_to_template(
    supplier_detection: dict[str, Any],
    templates_response: dict[str, Any],
    threshold: float = TEMPLATE_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """
    High-level matching function.

    Args:
        supplier_detection: Output from supplier_detector.extract_supplier_name().
        templates_response: Output from get_templates().
        threshold: Match threshold (default: 0.75).

    Returns:
        Match result dict:
            - matched (bool)
            - supplier_name (str)
            - supplier_name_normalized (str)
            - template / template_id / template_name / score / confidence_pct
              (only when matched=True)
            - reason (str, only when matched=False)
    """
    supplier_raw = supplier_detection.get("supplier_name") or "Unknown"
    supplier_normalized = supplier_detection.get("supplier_name_normalized") or ""

    templates: list[dict] = (
        templates_response.get("templates")
        or templates_response.get("results")
        or templates_response.get("value")
        or []
    )

    logger.info("Loading available templates... (%d found)", len(templates))

    if not supplier_detection.get("detected"):
        logger.warning(
            "Supplier not detected. Skipping template matching."
        )
        return {
            "matched": False,
            "supplier_name": supplier_raw,
            "supplier_name_normalized": supplier_normalized,
            "reason": supplier_detection.get("reason", "Supplier name not detected in SAP result"),
        }

    match = find_best_template(supplier_normalized, templates, threshold)

    if match:
        return {
            "matched": True,
            "supplier_name": supplier_raw,
            "supplier_name_normalized": supplier_normalized,
            **match,
        }

    return {
        "matched": False,
        "supplier_name": supplier_raw,
        "supplier_name_normalized": supplier_normalized,
        "reason": f"No template scored above threshold ({threshold:.0%})",
    }