"""Activation-gated three-valued evaluation for parsed conditional rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.ecn.rules.models import (
    AndExpression,
    EqualsExpression,
    FactSet,
    NotExpression,
    ParsedRule,
    RuleActivationPolicy,
    RuleEvaluation,
    RuleEvaluationState,
    RuleExpression,
    RuleKind,
    RuleParseState,
    TruthValue,
)


@dataclass(frozen=True)
class _ExpressionResult:
    """Hold an internal truth result and variables that limit certainty."""

    truth: TruthValue
    missing: frozenset[str]
    incomplete: frozenset[str]


def _combine_and(values: tuple[TruthValue, ...]) -> TruthValue:
    """Return a false-first three-valued conjunction."""
    if TruthValue.FALSE in values:
        return TruthValue.FALSE
    if TruthValue.UNKNOWN in values:
        return TruthValue.UNKNOWN
    return TruthValue.TRUE


def _combine_or(values: tuple[TruthValue, ...]) -> TruthValue:
    """Return a true-first three-valued disjunction."""
    if TruthValue.TRUE in values:
        return TruthValue.TRUE
    if TruthValue.UNKNOWN in values:
        return TruthValue.UNKNOWN
    return TruthValue.FALSE


def _evaluate_expression(
    expression: RuleExpression, facts: Mapping[str, FactSet]
) -> _ExpressionResult:
    """Evaluate one AST node and retain missing/incomplete variable names."""
    if isinstance(expression, EqualsExpression):
        fact = facts.get(expression.left)
        if fact is None:
            return _ExpressionResult(
                TruthValue.UNKNOWN, frozenset({expression.left}), frozenset()
            )
        if expression.right in fact.values:
            return _ExpressionResult(TruthValue.TRUE, frozenset(), frozenset())
        if fact.complete:
            return _ExpressionResult(TruthValue.FALSE, frozenset(), frozenset())
        return _ExpressionResult(
            TruthValue.UNKNOWN, frozenset(), frozenset({expression.left})
        )
    if isinstance(expression, NotExpression):
        child = _evaluate_expression(expression.operand, facts)
        inverse = {
            TruthValue.TRUE: TruthValue.FALSE,
            TruthValue.FALSE: TruthValue.TRUE,
            TruthValue.UNKNOWN: TruthValue.UNKNOWN,
        }[child.truth]
        return _ExpressionResult(inverse, child.missing, child.incomplete)
    children = tuple(
        _evaluate_expression(item, facts) for item in expression.operands
    )
    values = tuple(item.truth for item in children)
    truth = (
        _combine_and(values) if isinstance(expression, AndExpression) else _combine_or(values)
    )
    uncertain = tuple(item for item in children if item.truth is TruthValue.UNKNOWN)
    if truth is not TruthValue.UNKNOWN:
        uncertain = ()
    return _ExpressionResult(
        truth,
        frozenset().union(*(item.missing for item in uncertain)),
        frozenset().union(*(item.incomplete for item in uncertain)),
    )


def _blocked(
    rule: ParsedRule,
    state: RuleEvaluationState,
    policy_version: str | None,
) -> RuleEvaluation:
    """Build one side-effect-free non-evaluation result."""
    return RuleEvaluation(
        rule_id=rule.rule_id,
        rule_number=rule.rule_number,
        source_row_number=rule.source_row_number,
        parse_state=rule.parse_state,
        parse_reason_code=rule.reason_code,
        parse_reason_offset=rule.reason_offset,
        state=state,
        truth_value=None,
        applied_assignments=(),
        missing_variables=(),
        incomplete_variables=(),
        warnings=rule.warnings,
        activation_policy_version=policy_version,
        dependency_path=(rule.rule_id,),
    )


def evaluate_rule(
    rule: ParsedRule,
    facts: Mapping[str, FactSet],
    activation_policy: RuleActivationPolicy | None,
) -> RuleEvaluation:
    """Evaluate an admitted rule and expose actions only for a true condition.

    Args:
        rule: Parsed rule and its source lifecycle metadata.
        facts: Complete-aware characteristic values used by the condition.
        activation_policy: Explicit policy admitting lifecycle/quarantine states;
            ``None`` keeps every rule non-executable.

    Returns:
        An immutable result with truth, uncertainty, policy, and action provenance.
    """
    if (
        rule.parse_state is not RuleParseState.PARSED
        or rule.condition is None
        or rule.action is None
    ):
        return _blocked(rule, RuleEvaluationState.NOT_EVALUATED_PARSE_STATE, None)
    if activation_policy is None:
        return _blocked(
            rule, RuleEvaluationState.NOT_EVALUATED_ACTIVATION_POLICY_MISSING, None
        )
    if rule.lifecycle_state not in activation_policy.admitted_lifecycle_states:
        return _blocked(
            rule,
            RuleEvaluationState.NOT_EVALUATED_LIFECYCLE_STATE,
            activation_policy.version,
        )
    if rule.review_required and not activation_policy.allow_review_required:
        return _blocked(
            rule,
            RuleEvaluationState.NOT_EVALUATED_REVIEW_REQUIRED,
            activation_policy.version,
        )
    if (
        rule.rule_kind is RuleKind.LARGE_MULTI_ASSIGNMENT_ACTION
        and not activation_policy.allow_large_multi_assignment_actions
    ):
        return _blocked(
            rule,
            RuleEvaluationState.NOT_EVALUATED_LARGE_MULTI_ASSIGNMENT,
            activation_policy.version,
        )
    evaluated = _evaluate_expression(rule.condition, facts)
    return RuleEvaluation(
        rule_id=rule.rule_id,
        rule_number=rule.rule_number,
        source_row_number=rule.source_row_number,
        parse_state=rule.parse_state,
        parse_reason_code=rule.reason_code,
        parse_reason_offset=rule.reason_offset,
        state=RuleEvaluationState.EVALUATED,
        truth_value=evaluated.truth,
        applied_assignments=(
            rule.action.assignments if evaluated.truth is TruthValue.TRUE else ()
        ),
        missing_variables=tuple(sorted(evaluated.missing)),
        incomplete_variables=tuple(sorted(evaluated.incomplete)),
        warnings=rule.warnings,
        activation_policy_version=activation_policy.version,
        dependency_path=(rule.rule_id,),
    )
