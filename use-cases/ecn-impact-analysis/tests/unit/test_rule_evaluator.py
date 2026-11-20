"""Truth-table and activation-boundary tests for rule evaluation.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_rule_evaluator.py
"""

from dataclasses import replace

import pytest
from app.ecn.rules.evaluator import evaluate_rule
from app.ecn.rules.models import (
    FactSet,
    ParsedRule,
    RuleActivationPolicy,
    RuleEvaluationState,
    TruthValue,
)
from app.ecn.rules.parser import parse_rule_definition


def _rule(source: str, *, lifecycle: str = "App For Series") -> ParsedRule:
    """Build one parsed test rule with an explicit source lifecycle state."""
    return replace(
        parse_rule_definition(source, "rule-1", 2), lifecycle_state=lifecycle
    )


@pytest.mark.parametrize(
    ("facts", "truth", "missing", "incomplete"),
    [
        ({"A": FactSet(frozenset({"B"}), True)}, TruthValue.TRUE, (), ()),
        ({"A": FactSet(frozenset({"C"}), True)}, TruthValue.FALSE, (), ()),
        ({}, TruthValue.UNKNOWN, ("A",), ()),
        ({"A": FactSet(frozenset({"C"}), False)}, TruthValue.UNKNOWN, (), ("A",)),
        ({"A": FactSet(frozenset({"B", "C"}), True)}, TruthValue.TRUE, (), ()),
    ],
)
def test_equality_is_complete_aware(
    facts: dict[str, FactSet],
    truth: TruthValue,
    missing: tuple[str, ...],
    incomplete: tuple[str, ...],
) -> None:
    """Treat absent values as false only when that characteristic is complete."""
    result = evaluate_rule(
        _rule("A = B --> Inclure (S = V)"),
        facts,
        RuleActivationPolicy("test-v1", frozenset({"App For Series"})),
    )
    assert result.truth_value is truth
    assert result.missing_variables == missing
    assert result.incomplete_variables == incomplete
    assert bool(result.applied_assignments) is (truth is TruthValue.TRUE)


def test_and_or_not_propagate_unknown_without_boolean_coercion() -> None:
    """Apply Kleene-style truth tables to nested source expressions."""
    policy = RuleActivationPolicy("test-v1", frozenset({"App For Series"}))
    facts = {"A": FactSet(frozenset({"B"}), True)}
    conjunction = evaluate_rule(
        _rule("A = B + C = D --> Inclure (S = V)"), facts, policy
    )
    disjunction = evaluate_rule(
        _rule("A = B / C = D --> Inclure (S = V)"), facts, policy
    )
    assert conjunction.truth_value is TruthValue.UNKNOWN
    assert conjunction.missing_variables == ("C",)
    assert disjunction.truth_value is TruthValue.TRUE
    assert disjunction.missing_variables == ()
    assert (
        evaluate_rule(_rule("-(C = D) --> Inclure (S = V)"), facts, policy).truth_value
        is TruthValue.UNKNOWN
    )


def test_runtime_without_policy_does_not_evaluate_or_apply() -> None:
    """Keep parsed configuration non-executable until lifecycle policy is supplied."""
    result = evaluate_rule(
        _rule("A = B --> Inclure (S = V)"),
        {"A": FactSet(frozenset({"B"}), True)},
        None,
    )
    assert result.state is RuleEvaluationState.NOT_EVALUATED_ACTIVATION_POLICY_MISSING
    assert result.truth_value is None
    assert result.applied_assignments == ()
    assert result.activation_policy_version is None


@pytest.mark.parametrize(
    ("source", "policy", "state"),
    [
        (
            "A = B --> Inclure (S = V)",
            RuleActivationPolicy("p1", frozenset({"Released"})),
            RuleEvaluationState.NOT_EVALUATED_LIFECYCLE_STATE,
        ),
        (
            "-(A = B) / -(C = D) --> Inclure (S = V)",
            RuleActivationPolicy("p1", frozenset({"App For Series"})),
            RuleEvaluationState.NOT_EVALUATED_REVIEW_REQUIRED,
        ),
        (
            "A = B --> Exclure (S = V)",
            RuleActivationPolicy("p1", frozenset({"App For Series"})),
            RuleEvaluationState.NOT_EVALUATED_PARSE_STATE,
        ),
    ],
)
def test_quarantine_states_do_not_apply_actions(
    source: str, policy: RuleActivationPolicy, state: RuleEvaluationState
) -> None:
    """Return an explicit non-evaluation state for every blocked boundary."""
    result = evaluate_rule(_rule(source), {}, policy)
    assert result.state is state
    assert result.truth_value is None
    assert result.applied_assignments == ()


def test_large_action_requires_an_explicit_policy_opt_in() -> None:
    """Do not execute a structurally large action under the default policy."""
    actions = ", ".join(f"S{i} = V{i}" for i in range(101))
    rule = _rule(f"A = B --> Inclure ({actions})")
    denied = RuleActivationPolicy("p1", frozenset({"App For Series"}))
    allowed = RuleActivationPolicy(
        "p2",
        frozenset({"App For Series"}),
        allow_large_multi_assignment_actions=True,
    )
    facts = {"A": FactSet(frozenset({"B"}), True)}
    assert (
        evaluate_rule(rule, facts, denied).state
        is RuleEvaluationState.NOT_EVALUATED_LARGE_MULTI_ASSIGNMENT
    )
    assert len(evaluate_rule(rule, facts, allowed).applied_assignments) == 101
