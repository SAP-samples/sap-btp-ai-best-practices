"""Shared JSON schema for UC-02 per-line interpretation structured output.

This module provides the single authoritative ``INTERPRETATION_SCHEMA`` dict
that is passed as ``response_model`` to ``AgentRuntime.ainvoke`` by the
interpretation service (``api/app/payment_advice/interpretation_service.py``)
and the FastAPI route (``api/app/routers/payment_advice.py``).

Keeping the schema here prevents the entry points from drifting apart,
and makes it straightforward to version-bump or extend the schema in one place.

Schema shape (UC-02 selective-analysis contract)
-------------------------------------------------
Top-level: object with a single ``interpretations`` array property.

The **caller sends ONLY the anomalous lines** that require analysis (negatives,
chargebacks, lines with a deduction_reason), each tagged with a ``line_index``
that is the position of that line in the original full ``line_items`` list.
Ordinary invoice lines are NOT sent to the LLM.

The model must return EXACTLY ONE entry per provided line, echoing the
``line_index`` it was given.  It must NOT invent entries for lines it was not
given.  Each entry must include a ``rationale`` explaining the classification.

Each array entry is an interpretation object with these fields:
    line_index       – integer; echo the line_index you were given for this line
    document_nature  – "invoice" | "credit_memo" | "chargeback" |
                       "allowance_reversal" | "unknown"
    reason_code      – SAP reason code string, or null when fully resolved
    residual_items   – list of invoice_reference strings for partial-payment links
    flags            – list of diagnostic flag strings (e.g. "linkage-outside-document")
    rationale        – free-text explanation of classification and tools used

The consumer (``enrich_payload``) matches each returned entry to the original
line by ``line_index``, so the echo is mandatory.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Per-entry sub-schema: one interpretation object for a single anomalous line.
# ---------------------------------------------------------------------------
_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["line_index"],
    "properties": {
        "line_index": {
            "type": "integer",
            "description": "Echo the line_index you were given for this line.",
        },
        "document_nature": {
            "type": "string",
            "description": (
                "Nature of the document line: "
                "invoice | credit_memo | chargeback | allowance_reversal | unknown. "
                "Use 'unknown' when the line cannot be classified with confidence."
            ),
        },
        "reason_code": {
            "type": ["string", "null"],
            "description": "SAP reason code, or null when the line is fully resolved.",
        },
        "residual_items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Invoice references for linked residual/partial payments.",
        },
        "flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Diagnostic flags (e.g. 'linkage-outside-document').",
        },
        "rationale": {
            "type": "string",
            "description": (
                "Why this document_nature / reason_code / flags were chosen, "
                "and which tools informed it."
            ),
        },
    },
}

# The schema is intentionally a plain dict rather than a Pydantic class so that
# LangChain's with_structured_output returns a plain Python dict directly,
# which enrich_payload() can consume without any Pydantic-to-dict conversion.
INTERPRETATION_SCHEMA: dict[str, Any] = {
    "title": "Interpretations",
    "type": "object",
    "description": (
        "Selective interpretation result for UC-02.  "
        "The caller provides ONLY the anomalous lines that require analysis "
        "(negatives, chargebacks, lines with a deduction_reason), each tagged "
        "with a line_index that identifies its position in the original full "
        "line_items list.  Return EXACTLY ONE entry per provided line, echoing "
        "its line_index.  Do NOT invent entries for lines you were not given.  "
        "Always include a rationale for each entry."
    ),
    "properties": {
        "interpretations": {
            "type": "array",
            "description": (
                "Exactly one interpretation entry per anomalous line provided by "
                "the caller.  Each entry must echo the line_index it received.  "
                "Do not add entries for lines that were not provided."
            ),
            "items": _ENTRY_SCHEMA,
        },
    },
    "required": ["interpretations"],
}
