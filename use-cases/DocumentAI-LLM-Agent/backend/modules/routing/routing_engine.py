"""
routing_engine.py
-----------------
Orchestrates the intelligent routing logic for invoice processing.

Responsibilities:
  - Coordinate supplier detection from initial SAP result
  - Load available templates from SAP Document AI
  - Match supplier against templates
  - Decide: template flow OR GenAI fallback flow
  - Save routing decision artifacts to output/routing/
"""

import json
import logging
from pathlib import Path
from typing import Any

from modules.routing.supplier_detector import extract_supplier_name
from modules.routing.template_matcher import (
    TEMPLATE_MATCH_THRESHOLD,
    match_supplier_to_template,
)
from modules.templates.get_templates import DocumentAIError as TemplateError
from modules.templates.get_templates import get_templates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
OUTPUT_ROUTING_DIR: Path = _PROJECT_ROOT / "output" / "routing"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RoutingError(Exception):
    """Raised when the routing engine encounters a critical failure."""

    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_routing_json(data: dict, filename: str) -> Path:
    """Persist a routing artifact as JSON."""
    OUTPUT_ROUTING_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROUTING_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Routing artifact saved: %s", path)
    return path


def _print_template_found(supplier: dict, match: dict) -> None:
    supplier_name = supplier.get("supplier_name") or "N/A"
    template_name = match.get("template_name") or "N/A"
    confidence = match.get("confidence_pct", 0.0)

    print(f"\n{'='*52}")
    print("  TEMPLATE MATCH FOUND")
    print(f"{'='*52}")
    print(f"\n  Supplier        : {supplier_name}")
    print(f"  Matched Template: {template_name}")
    print(f"  Confidence      : {confidence:.1f}%")
    print(f"\n  Processing invoice using SAP specialized template...")
    print(f"{'='*52}\n")


def _print_no_template(supplier: dict) -> None:
    supplier_name = supplier.get("supplier_name") or "Unknown Supplier"

    print(f"\n{'='*52}")
    print("  NO TEMPLATE MATCH FOUND")
    print(f"{'='*52}")
    print(f"\n  Supplier        : {supplier_name}")
    print(f"  Fallback Strategy: Executing GenAI comparison flow...")
    print(f"{'='*52}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_invoice(
    sap_result: dict[str, Any],
    client_id: str = "default",
    threshold: float = TEMPLATE_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """
    Determine the processing route for an invoice based on supplier detection
    and template matching.

    Flow:
      1. Extract supplier name from initial SAP result
      2. Load available templates from SAP Document AI
      3. Match supplier against templates using fuzzy/semantic scoring
      4. If match found (score >= threshold) → route = "template"
         If no match → route = "genai"
      5. Save all routing artifacts to output/routing/

    Args:
        sap_result: Initial SAP Document AI result (generic processing).
        client_id: SAP Document AI client ID.
        threshold: Template match confidence threshold (default: 0.75).

    Returns:
        Routing decision dict:
            - route (str): "template" | "genai"
            - supplier_detection (dict): supplier detection result
            - template_match (dict | None): template match result
            - decision_reason (str): human-readable explanation
    """
    logger.info("=" * 60)
    logger.info("ROUTING ENGINE: Starting intelligent routing...")
    logger.info("=" * 60)

    # ── Step 1: Detect supplier ──────────────────────────────────────────
    logger.info("Starting supplier detection...")
    supplier_detection = extract_supplier_name(sap_result)
    _save_routing_json(supplier_detection, "supplier_detection.json")

    supplier_name = supplier_detection.get("supplier_name") or "Unknown"
    logger.info(
        "Supplier detection result: detected=%s, name='%s'",
        supplier_detection.get("detected"),
        supplier_name,
    )

    # ── Step 2: Load templates ───────────────────────────────────────────
    logger.info("Loading available templates...")
    try:
        templates_response = get_templates(client_id=client_id)
        template_count = len(
            templates_response.get("templates")
            or templates_response.get("results")
            or []
        )
        logger.info("Templates loaded: %d available", template_count)
    except (TemplateError, Exception) as exc:
        logger.warning(
            "Could not load templates: %s. Falling back to GenAI flow.", exc
        )
        routing_decision = {
            "route": "genai",
            "supplier_detection": supplier_detection,
            "template_match": None,
            "decision_reason": f"Templates API unavailable: {exc}",
        }
        _save_routing_json({}, "template_matches.json")
        _save_routing_json(routing_decision, "routing_decision.json")
        _print_no_template(supplier_detection)
        return routing_decision

    # ── Step 3: Match supplier to template ──────────────────────────────
    logger.info("Matching supplier against templates...")
    template_match = match_supplier_to_template(
        supplier_detection,
        templates_response,
        threshold=threshold,
    )
    _save_routing_json(template_match, "template_matches.json")

    # ── Step 4: Decide route ─────────────────────────────────────────────
    if template_match.get("matched"):
        logger.info(
            "Template match found: '%s' (confidence: %.1f%%)",
            template_match.get("template_name"),
            template_match.get("confidence_pct", 0.0),
        )
        _print_template_found(supplier_detection, template_match)

        routing_decision = {
            "route": "template",
            "supplier_detection": supplier_detection,
            "template_match": template_match,
            "decision_reason": (
                f"Template '{template_match['template_name']}' matched "
                f"with {template_match['confidence_pct']:.1f}% confidence."
            ),
        }
    else:
        logger.warning(
            "No matching template found. Falling back to GenAI flow."
        )
        _print_no_template(supplier_detection)

        routing_decision = {
            "route": "genai",
            "supplier_detection": supplier_detection,
            "template_match": template_match,
            "decision_reason": template_match.get(
                "reason", "No matching template found."
            ),
        }

    _save_routing_json(routing_decision, "routing_decision.json")

    logger.info(
        "Routing decision: route='%s' | reason='%s'",
        routing_decision["route"],
        routing_decision["decision_reason"],
    )

    return routing_decision