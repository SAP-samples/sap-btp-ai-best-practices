"""Generate and semantically evaluate HANA-compatible customer playbooks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from app.payment_advice.customers import normalize_client_key

from .rule_sources import (
    bind_rule_sources,
    list_rule_sources,
    reset_rule_sources,
)
from .trace import extract_tool_calls


_STRING_OR_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}
# Added after playbooks were already stored: the model must return them (strict
# structured output), but stored or golden playbooks without them stay valid.
OPTIONAL_ANCHOR_FIELDS = ("default_company_code", "company_code_rules")

PLAYBOOK_SCHEMA: dict[str, Any] = {
    "title": "DeductionSkillCandidate",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "playbook_text": {"type": "string"},
        "anchors": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_scope": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "document_formats": {"type": "array", "items": {"type": "string"}},
                        "sender_patterns": {"type": "array", "items": {"type": "string"}},
                        "filename_patterns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["document_formats", "sender_patterns", "filename_patterns"],
                },
                "default_customer_account": _STRING_OR_NULL,
                "customer_account_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "priority": {"type": "integer"},
                            "field": {"type": "string"},
                            "operator": {"type": "string"},
                            "value": {"type": "string"},
                            "customer_account": {"type": "string"},
                        },
                        "required": ["priority", "field", "operator", "value", "customer_account"],
                    },
                },
                "default_company_code": _STRING_OR_NULL,
                "company_code_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "priority": {"type": "integer"},
                            "field": {"type": "string"},
                            "operator": {"type": "string"},
                            "value": {"type": "string"},
                            "company_code": {"type": "string"},
                        },
                        "required": ["priority", "field", "operator", "value", "company_code"],
                    },
                },
                "field_mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "target_field": {"type": "string"},
                            "source_field": {"type": "string"},
                            "transformation": {"type": "string"},
                        },
                        "required": ["target_field", "source_field", "transformation"],
                    },
                },
                "deduction_rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "priority": {"type": "integer"},
                            "field": {"type": "string"},
                            "operator": {"type": "string"},
                            "value": {"type": "string"},
                            "reason_code": _STRING_OR_NULL,
                            "action": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": [
                            "priority",
                            "field",
                            "operator",
                            "value",
                            "reason_code",
                            "action",
                            "description",
                        ],
                    },
                },
                "default_rule": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reason_code": _STRING_OR_NULL,
                        "action": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["reason_code", "action", "description"],
                },
                "processing_notes": {"type": "array", "items": {"type": "string"}},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_name": {"type": "string"},
                            "locator": {"type": "string"},
                            "fact": {"type": "string"},
                        },
                        "required": ["source_name", "locator", "fact"],
                    },
                },
                "open_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "source_scope",
                "default_customer_account",
                "customer_account_rules",
                "default_company_code",
                "company_code_rules",
                "field_mappings",
                "deduction_rules",
                "default_rule",
                "processing_notes",
                "evidence",
                "open_questions",
            ],
        },
    },
    "required": ["playbook_text", "anchors"],
}


async def generate_playbook(
    runtime: Any,
    client: str,
    source_paths: Sequence[str | Path],
    golden: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the agent to author one playbook from caller-authorized documents.

    Args:
        runtime: Agent runtime exposing an asynchronous ``ainvoke`` method.
        client: Customer name or key used to normalize the HANA ``CLIENT_KEY``.
        source_paths: Explicit DOCX/XLSX evidence files available to the agent.
        golden: Optional hand-verified playbook used for semantic evaluation.

    Returns:
        A ``(candidate, report)`` tuple. ``candidate`` matches the existing HANA
        playbook fields plus ``client_key``. ``report`` contains source names,
        tool trace, agent narrative, and optional golden comparison.

    Raises:
        ValueError: If the model returns an incomplete candidate.
        RuleSourceError: If source binding or reading fails.
    """

    client_key = normalize_client_key(client)
    token = bind_rule_sources(source_paths)
    try:
        sources = [item["source_name"] for item in list_rule_sources()]
        result = await runtime.ainvoke(
            text=(
                f"Create a HANA customer deduction skill for {client_key!r}. "
                "Load the deduction-skill-authoring agent skill, inspect every bound "
                "Word/Excel source completely with list_rule_sources and "
                "read_rule_source, and return only source-supported rules."
            ),
            context_id=f"authoring_{client_key}",
            response_model=PLAYBOOK_SCHEMA,
        )
    finally:
        reset_rule_sources(token)

    parsed = _validate_candidate(getattr(result, "output_parsed", None))
    candidate = {
        "client_key": client_key,
        "playbook_text": parsed["playbook_text"],
        "anchors": parsed["anchors"],
    }
    report: dict[str, Any] = {
        "client_key": client_key,
        "sources": sources,
        "agent_output": getattr(result, "output_text", ""),
        "tools_called": extract_tool_calls(getattr(result, "messages", []) or []),
    }
    if golden is not None:
        report["comparison"] = compare_playbooks(candidate, golden)
    return candidate, report


def compare_playbooks(
    candidate: dict[str, Any],
    golden: dict[str, Any],
) -> dict[str, Any]:
    """Compare business-significant playbook facts while ignoring prose wording.

    Args:
        candidate: Agent-generated playbook.
        golden: Independently authored expected playbook.

    Returns:
        Match status plus missing and unexpected normalized facts. Evidence,
        descriptions, processing notes, and prose are intentionally not compared.
    """

    candidate_anchors = _validate_candidate(candidate)["anchors"]
    golden_anchors = _validate_candidate(golden)["anchors"]
    missing: dict[str, Any] = {}
    unexpected: dict[str, Any] = {}

    scalar_fields = ("default_customer_account",)
    for field in scalar_fields:
        actual = candidate_anchors[field]
        expected = golden_anchors[field]
        if actual != expected:
            missing[field] = expected
            unexpected[field] = actual

    actual_default = candidate_anchors["default_rule"]["reason_code"]
    expected_default = golden_anchors["default_rule"]["reason_code"]
    if actual_default != expected_default:
        missing["default_reason_code"] = expected_default
        unexpected["default_reason_code"] = actual_default

    comparisons = {
        "source_scope": _source_scope_facts,
        "customer_account_rules": _customer_rule_facts,
        "field_mappings": _field_mapping_facts,
        "deduction_rules": _deduction_rule_facts,
    }
    for field, fact_builder in comparisons.items():
        actual_facts = fact_builder(candidate_anchors[field])
        expected_facts = fact_builder(golden_anchors[field])
        field_missing = sorted(expected_facts - actual_facts)
        field_unexpected = sorted(actual_facts - expected_facts)
        if field_missing:
            missing[field] = field_missing
        if field_unexpected:
            unexpected[field] = field_unexpected

    return {
        "matched": not missing and not unexpected,
        "missing": missing,
        "unexpected": unexpected,
    }


def _validate_candidate(value: Any) -> dict[str, Any]:
    """Return a minimally valid playbook or reject unsafe model output."""

    if not isinstance(value, dict):
        raise ValueError("Agent result must be a playbook object")
    if not isinstance(value.get("playbook_text"), str) or not value["playbook_text"].strip():
        raise ValueError("Agent result must contain non-empty playbook_text")
    anchors = value.get("anchors")
    if not isinstance(anchors, dict):
        raise ValueError("Agent result must contain an anchors object")
    required = [field for field in PLAYBOOK_SCHEMA["properties"]["anchors"]["required"]
                if field not in OPTIONAL_ANCHOR_FIELDS]
    missing = [field for field in required if field not in anchors]
    if missing:
        raise ValueError(f"Agent result anchors missing fields: {', '.join(missing)}")
    return value


def _normalized(value: Any) -> str:
    """Return a stable, whitespace-trimmed string for semantic fact comparison."""

    return "" if value is None else str(value).strip()


def _canonical_label(value: Any) -> str:
    """Normalize equivalent document labels used by different authors or models."""

    label = re.sub(r"[^a-z0-9]+", "_", _normalized(value).lower()).strip("_")
    if label.startswith("column_"):
        label = label.removeprefix("column_")
    return {
        "advice_no": "advice_number",
        "fixed": "constant",
        "invoice_number": "invoice_reference",
        "inv_currency": "invoice_currency",
        "inv_date": "invoice_date",
        "iln_sender": "payer",
        "iln_sender_payer": "payer",
        "iln_recipient": "payee",
        "iln_recipient_payee": "payee",
        "item_characteristic_inv_number": "invoice_reference",
    }.get(label, label)


def _canonical_mapping_source(mapping: dict[str, Any]) -> str:
    """Normalize a mapping source across literal-constant and label representations."""

    source = _canonical_label(mapping["source_field"])
    transformation = _canonical_label(mapping["transformation"])
    if transformation == "constant" or source == "constant":
        return "constant"
    if "current_date" in source:
        return "current_date"
    if source == "rg" or source.startswith("rg_customer_account"):
        return "rg"
    return source


def _source_scope_facts(scope: dict[str, Any]) -> set[tuple[str, str]]:
    """Normalize format, sender, and filename-pattern facts."""

    return {
        (field, _normalized(value).lower())
        for field in ("document_formats", "sender_patterns", "filename_patterns")
        for value in scope.get(field, [])
    }


def _customer_rule_facts(rules: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    """Normalize ordered customer-account selection rules."""

    return {
        (
            int(rule["priority"]),
            _canonical_label(rule["field"]),
            _normalized(rule["operator"]).lower(),
            _normalized(rule["value"]),
            _normalized(rule["customer_account"]),
        )
        for rule in rules
    }


def _field_mapping_facts(mappings: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Normalize source-to-target mappings without comparing narrative transforms."""

    return {
        (
            _canonical_label(mapping["target_field"]),
            _canonical_mapping_source(mapping),
        )
        for mapping in mappings
    }


def _deduction_rule_facts(rules: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    """Normalize ordered condition-to-reason-code deduction rules."""

    return {
        (
            int(rule["priority"]),
            _canonical_label(rule["field"]),
            _normalized(rule["operator"]).lower(),
            _normalized(rule["value"]),
            _normalized(rule["reason_code"]),
        )
        for rule in rules
    }
