"""L14 -- Persona metadata and priority hints.

Hardcoded from the Word doc's embedded 'meet your segments' image
(confirmed in archived/docs/RULES_REVIEW_FINDINGS.md Finding 3).

The seventh segment is called "She's Struggling" in the image but
"Stretched and Struggling" in the workbook.  Both names map to the
same attributes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaAttributes:
    financial_stability: str   # "high" | "medium" | "low"
    tech_connectivity: str     # "high" | "medium" | "low"
    environmental_concern: str # "high" | "medium" | "low"


# Canonical persona matrix (from the Word doc image)
PERSONA_MATRIX: dict[str, PersonaAttributes] = {
    "Successful Money Managers": PersonaAttributes(
        financial_stability="high",
        tech_connectivity="medium",
        environmental_concern="low",
    ),
    "All About Me": PersonaAttributes(
        financial_stability="high",
        tech_connectivity="medium",
        environmental_concern="low",
    ),
    "Eco-Invested Modernists": PersonaAttributes(
        financial_stability="medium",
        tech_connectivity="high",
        environmental_concern="high",
    ),
    "Comfortable Retirees": PersonaAttributes(
        financial_stability="medium",
        tech_connectivity="low",
        environmental_concern="low",
    ),
    "Salt of the Earth": PersonaAttributes(
        financial_stability="low",
        tech_connectivity="medium",
        environmental_concern="high",
    ),
    "Young Unestablished Idealists": PersonaAttributes(
        financial_stability="low",
        tech_connectivity="high",
        environmental_concern="high",
    ),
    # Workbook name for the seventh segment
    "Stretched and Struggling": PersonaAttributes(
        financial_stability="low",
        tech_connectivity="high",
        environmental_concern="medium",
    ),
}

# Alias: the Word doc image uses "She's Struggling"
PERSONA_MATRIX["She's Struggling"] = PERSONA_MATRIX["Stretched and Struggling"]


def get_persona_attributes(segment_name: str | None) -> PersonaAttributes | None:
    """Look up persona attributes by segment name (case-insensitive)."""
    if segment_name is None:
        return None
    # Try exact match first, then case-insensitive
    if segment_name in PERSONA_MATRIX:
        return PERSONA_MATRIX[segment_name]
    for key, val in PERSONA_MATRIX.items():
        if key.lower() == segment_name.lower():
            return val
    return None


def get_persona_hints(segment_name: str | None) -> dict[str, str]:
    """Return priority hints derived from persona attributes.

    Hint values are "boost" or "neutral".  Rules use these to reorder
    offers that share the same priority level.
    """
    attrs = get_persona_attributes(segment_name)
    if attrs is None:
        return {}

    hints: dict[str, str] = {}

    # Low financial stability -> hint toward payment assistance
    if attrs.financial_stability == "low":
        hints["payment_assistance"] = "boost"
    else:
        hints["payment_assistance"] = "neutral"

    # High environmental concern -> hint toward renewable/green programs
    if attrs.environmental_concern == "high":
        hints["renewable_programs"] = "boost"
    else:
        hints["renewable_programs"] = "neutral"

    # High tech connectivity -> hint toward BYOT, EV Smart Charge
    if attrs.tech_connectivity == "high":
        hints["tech_programs"] = "boost"
    else:
        hints["tech_programs"] = "neutral"

    return hints
