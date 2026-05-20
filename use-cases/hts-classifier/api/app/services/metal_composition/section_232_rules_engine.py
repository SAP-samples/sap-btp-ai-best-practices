"""Deterministic Section 232 evaluation against the published ruleset."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from .hts_catalog import canonicalize_hts_code
from .section_232_rulesets import Section232DraftRuleCandidate
from .section_232_sources import expand_hts_code_family

_SECTION_232_AFFECTED_METALS = ("steel", "aluminum", "copper")
_SECTION_232_WEIGHT_EXEMPTION_THRESHOLD = 0.15


def _normalize_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(value: Any) -> str:
    """Return a stripped string representation for rule diagnostics."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_confidence(value: Any) -> float:
    """Return a bounded confidence value between 0 and 1."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric_value))


def _append_sentence(summary: str, sentence: str) -> str:
    """Append a sentence to a summary without duplicating existing text."""
    base = _normalize_text(summary)
    extra = _normalize_text(sentence)
    if not extra:
        return base
    if not base:
        return extra
    if extra in base:
        return base
    return f"{base} {extra}".strip()


def _normalize_metal_share_certainty(diagram_output: Dict[str, Any]) -> str:
    certainty = str(diagram_output.get("metal_share_certainty") or "estimated").strip().lower()
    return certainty if certainty in {"exact", "estimated"} else "estimated"


def _normalized_code_length(code: str) -> int:
    return len(code.replace(".", ""))


def _is_effective_on_date(rule: Section232DraftRuleCandidate, on_date: date) -> bool:
    if rule.effective_from:
        effective_from = date.fromisoformat(rule.effective_from)
        if on_date < effective_from:
            return False
    if rule.effective_to is None:
        return True
    return on_date <= date.fromisoformat(rule.effective_to)


def _rule_priority(
    *,
    family_rank: int,
    rule: Section232DraftRuleCandidate,
) -> tuple[int, int, int, int]:
    return (
        family_rank,
        0 if rule.coverage_effect == "remove" else 1,
        0 if rule.rule_type == "rate_schedule" else 1,
        -_normalized_code_length(rule.hts_code),
    )


def _rules_equivalent(left: Section232DraftRuleCandidate, right: Section232DraftRuleCandidate) -> bool:
    return (
        canonicalize_hts_code(left.hts_code) == canonicalize_hts_code(right.hts_code)
        and left.rule_type == right.rule_type
        and left.coverage_effect == right.coverage_effect
        and (left.metal_scope or "").strip().lower() == (right.metal_scope or "").strip().lower()
        and left.effective_from == right.effective_from
        and left.effective_to == right.effective_to
    )


def _effective_window(*, on_date: date, rule: Optional[Section232DraftRuleCandidate]) -> Dict[str, Any]:
    if rule is None:
        return {
            "evaluation_date": on_date.isoformat(),
            "effective_from": None,
            "effective_to": None,
            "is_active": False,
        }
    return {
        "evaluation_date": on_date.isoformat(),
        "effective_from": rule.effective_from,
        "effective_to": rule.effective_to,
        "is_active": True,
    }


def evaluate_section_232_ruleset(
    *,
    selected_code: str,
    on_date: date,
    ruleset_store: object | None,
    top_level_grams: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate HTS legal coverage against the active published Section 232 ruleset.

    The selected HTS code is matched against active exact and parent-family rules.
    Material composition is intentionally not used at this stage; callers apply
    the deterministic affected-metal weight exemption after legal coverage is
    established. The top-level grams argument is retained for caller compatibility.

    Args:
        selected_code: Current HTS code to evaluate.
        on_date: Date used to filter rule effective windows.
        ruleset_store: Published ruleset store used to load active rules.
        top_level_grams: Deprecated compatibility input; ignored by legal matching.

    Returns:
        A structured legal-coverage result with the decision, matched rule details,
        active ruleset version, effective window, and diagnostic reason.
    """
    _ = top_level_grams
    normalized_selected_code = canonicalize_hts_code(selected_code)
    active_ruleset_version = (
        getattr(ruleset_store, "get_active_ruleset_version", lambda: None)()
        if ruleset_store is not None
        else None
    )
    base_result = {
        "decision": "needs_review",
        "matched_rule_type": None,
        "matched_hts_code": None,
        "effective_window": _effective_window(on_date=on_date, rule=None),
        "matched_rule_scope": None,
        "active_ruleset_version": active_ruleset_version,
        "reason": "",
    }

    if not normalized_selected_code:
        return {**base_result, "reason": "no_selected_code"}

    if ruleset_store is None:
        return {**base_result, "reason": "no_ruleset_store"}

    active_rules = list(getattr(ruleset_store, "list_active_rules", lambda: [])() or [])
    if not active_rules:
        return {**base_result, "reason": "no_active_ruleset"}

    family_codes = expand_hts_code_family(normalized_selected_code)
    family_ranks = {code: index for index, code in enumerate(family_codes)}
    matches: list[tuple[tuple[int, int, int, int], Section232DraftRuleCandidate]] = []
    for rule in active_rules:
        normalized_rule_code = canonicalize_hts_code(rule.hts_code)
        if not normalized_rule_code or normalized_rule_code not in family_ranks:
            continue
        if not _is_effective_on_date(rule, on_date):
            continue
        matches.append(
            (
                _rule_priority(
                    family_rank=family_ranks[normalized_rule_code],
                    rule=rule,
                ),
                rule,
            )
        )

    if not matches:
        return {**base_result, "reason": "no_matching_rule"}

    matches.sort(key=lambda item: item[0])
    winner_priority = matches[0][0]
    winning_rules = [rule for priority, rule in matches if priority == winner_priority]
    winner = winning_rules[0]
    if any(not _rules_equivalent(winner, other) for other in winning_rules[1:]):
        return {
            **base_result,
            "reason": "ambiguous_matching_rules",
            "effective_window": _effective_window(on_date=on_date, rule=winner),
        }

    matched_hts_code = canonicalize_hts_code(winner.hts_code)
    return {
        "decision": "not_subject" if winner.coverage_effect == "remove" else "subject",
        "matched_rule_type": winner.rule_type,
        "matched_hts_code": matched_hts_code,
        "effective_window": _effective_window(on_date=on_date, rule=winner),
        "matched_rule_scope": (
            "exact"
            if matched_hts_code == normalized_selected_code
            else "family"
        ),
        "active_ruleset_version": active_ruleset_version,
        "reason": "matched_rule",
    }


def build_section_232_ruleset_assessment(
    *,
    selected_code: str,
    candidate_codes: list[str],
    legal_result: Dict[str, Any],
) -> Dict[str, Any]:
    decision = str(legal_result.get("decision") or "needs_review")
    reason = str(legal_result.get("reason") or "")
    matched_rule_type = legal_result.get("matched_rule_type")
    matched_hts_code = legal_result.get("matched_hts_code")
    active_ruleset_version = legal_result.get("active_ruleset_version")
    effective_window = dict(legal_result.get("effective_window") or {})

    if reason == "no_active_ruleset":
        basis_summary = (
            "No published Section 232 ruleset is active, so the selected HTS code "
            "could not be evaluated deterministically."
        )
    elif reason == "no_ruleset_store":
        basis_summary = (
            "No published Section 232 ruleset store is available, so the selected HTS code "
            "could not be evaluated deterministically."
        )
    elif reason == "no_selected_code":
        basis_summary = (
            "No stable HTS code was available for deterministic Section 232 evaluation "
            "against the published ruleset."
        )
    elif reason == "no_matching_rule":
        basis_summary = (
            f"The active published Section 232 ruleset does not contain a current match for HTS code "
            f"{selected_code or 'unknown'}."
        )
    elif reason == "ambiguous_matching_rules":
        basis_summary = (
            f"The active published Section 232 ruleset returned conflicting matches for HTS code "
            f"{selected_code or 'unknown'}, so the item requires review."
        )
    elif decision == "not_subject":
        basis_summary = (
            f"The active published Section 232 ruleset excludes HTS code {selected_code or 'unknown'} "
            f"through the matched {matched_rule_type or 'remove'} rule on {matched_hts_code or 'the selected family'}."
        )
    else:
        basis_summary = (
            f"The active published Section 232 ruleset marks HTS code {selected_code or 'unknown'} as subject "
            f"through the matched {matched_rule_type or 'include'} rule on {matched_hts_code or 'the selected family'}"
            "."
        )
    confidence = 0.9 if decision != "needs_review" else 0.4
    if reason == "no_matching_rule":
        confidence = 0.7

    evidence = []
    if active_ruleset_version:
        evidence.append(
            {
                "source": "published_ruleset",
                "summary": (
                    f"Evaluated against published ruleset {active_ruleset_version} "
                    f"for {effective_window.get('evaluation_date') or 'the current date'}."
                ),
                "citations": [matched_hts_code] if matched_hts_code else [],
            }
        )

    return {
        "status": "completed" if selected_code else "unavailable",
        "decision": decision,
        "confidence": confidence,
        "basis_summary": basis_summary,
        "needs_human_review": decision == "needs_review",
        "weight_rule_applied": False,
        "supporting_hts_candidates": list(candidate_codes[:5]),
        "chapter_99_candidates": [],
        "evidence": evidence,
    }


def build_section_232_ruleset_reasoner_output(legal_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "completed",
        "strategy": "published_ruleset",
        "fallback_used": False,
        "reason": legal_result.get("reason"),
        "decision": legal_result.get("decision"),
        "active_ruleset_version": legal_result.get("active_ruleset_version"),
        "matched_rule_type": legal_result.get("matched_rule_type"),
        "matched_hts_code": legal_result.get("matched_hts_code"),
        "matched_rule_scope": legal_result.get("matched_rule_scope"),
        "effective_window": dict(legal_result.get("effective_window") or {}),
    }


def legal_coverage_established(legal_result: Dict[str, Any]) -> bool:
    return (
        str(legal_result.get("reason") or "") == "matched_rule"
        and legal_result.get("matched_rule_type") is not None
        and legal_result.get("matched_hts_code") is not None
    )


def apply_section_232_weight_override(
    *,
    final_composition: Dict[str, Any],
    diagram_output: Dict[str, Any],
    section_payload: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply the deterministic 15% affected-metal override after ruleset matching.

    Inputs are the final composition payload, the diagram certainty metadata, and
    the Section 232 assessment generated from the published ruleset. The output
    is an updated assessment plus diagnostics describing whether the exemption
    changed the decision.
    """
    updated = dict(section_payload or {})
    base_decision = _normalize_text(updated.get("decision")).lower() or "needs_review"
    total_weight_grams = _normalize_float(final_composition.get("total_weight_grams"))
    top_level_grams = dict(final_composition.get("top_level_grams") or {})
    affected_metal_weight_grams = sum(
        _normalize_float(top_level_grams.get(metal, 0.0))
        for metal in _SECTION_232_AFFECTED_METALS
    )
    affected_metal_share = (
        affected_metal_weight_grams / total_weight_grams
        if total_weight_grams > 0.0
        else None
    )
    metal_share_certainty = _normalize_metal_share_certainty(diagram_output)

    diagnostics: Dict[str, Any] = {
        "base_decision": base_decision,
        "final_decision": base_decision,
        "override_applied": False,
        "threshold": _SECTION_232_WEIGHT_EXEMPTION_THRESHOLD,
        "total_weight_grams": total_weight_grams if total_weight_grams > 0.0 else None,
        "affected_metal_weight_grams": affected_metal_weight_grams,
        "affected_metal_share": affected_metal_share,
        "metal_share_certainty": metal_share_certainty,
        "reason": "",
        "needs_human_review": False,
    }

    updated.setdefault("needs_human_review", False)
    updated.setdefault("weight_rule_applied", False)

    if base_decision == "not_subject":
        diagnostics["reason"] = "base_not_subject"
        return updated, diagnostics

    if total_weight_grams <= 0.0:
        diagnostics["reason"] = "missing_total_weight"
        return updated, diagnostics

    if affected_metal_share is None:
        diagnostics["reason"] = "share_unavailable"
        return updated, diagnostics

    if affected_metal_share < _SECTION_232_WEIGHT_EXEMPTION_THRESHOLD:
        diagnostics["override_applied"] = True
        updated["decision"] = "not_subject"
        updated["weight_rule_applied"] = True
        if metal_share_certainty == "exact":
            updated["confidence"] = _normalize_confidence(
                max(_normalize_confidence(updated.get("confidence")), 0.95)
            )
            updated["needs_human_review"] = False
            updated["basis_summary"] = (
                "Material composition explicitly show that the combined steel, aluminum, and copper content "
                "is below 15% of total item weight, so the item is not subject to Section 232."
            )
            diagnostics["reason"] = "exact_below_threshold"
        else:
            updated["confidence"] = 0.55
            updated["needs_human_review"] = True
            updated["basis_summary"] = (
                "Material composition analysis suggest that the combined steel, aluminum, and copper content is below "
                "15% of total item weight, so the item is flagged as not subject to Section 232. "
                "This exemption is estimate-based and should be reviewed by a human."
            )
            diagnostics["reason"] = "estimated_below_threshold"
            diagnostics["needs_human_review"] = True
        diagnostics["final_decision"] = "not_subject"
        return updated, diagnostics

    diagnostics["reason"] = "threshold_not_met"
    diagnostics["final_decision"] = base_decision
    if base_decision == "subject" and metal_share_certainty == "estimated":
        updated["basis_summary"] = _append_sentence(
            updated.get("basis_summary", ""),
            "The affected-metal share is estimated from the assigned documents rather than explicitly stated, "
            "but the estimate remains at or above the 15% exemption threshold.",
        )
    return updated, diagnostics


def build_skipped_weight_override(
    *,
    final_composition: Dict[str, Any],
    diagram_output: Dict[str, Any],
    section_payload: Dict[str, Any],
) -> Dict[str, Any]:
    total_weight_grams = _normalize_float(final_composition.get("total_weight_grams"))
    top_level_grams = dict(final_composition.get("top_level_grams") or {})
    affected_metal_weight_grams = sum(
        _normalize_float(top_level_grams.get(metal, 0.0))
        for metal in _SECTION_232_AFFECTED_METALS
    )
    affected_metal_share = (
        affected_metal_weight_grams / total_weight_grams
        if total_weight_grams > 0.0
        else None
    )
    return {
        "base_decision": str(section_payload.get("decision") or "needs_review"),
        "final_decision": str(section_payload.get("decision") or "needs_review"),
        "override_applied": False,
        "threshold": _SECTION_232_WEIGHT_EXEMPTION_THRESHOLD,
        "total_weight_grams": total_weight_grams if total_weight_grams > 0.0 else None,
        "affected_metal_weight_grams": affected_metal_weight_grams,
        "affected_metal_share": affected_metal_share,
        "metal_share_certainty": _normalize_metal_share_certainty(diagram_output),
        "reason": "legal_coverage_not_established",
        "needs_human_review": bool(section_payload.get("needs_human_review")),
    }
