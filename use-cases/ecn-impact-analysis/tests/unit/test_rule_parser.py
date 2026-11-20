"""Contract tests for reversible deterministic conditional-rule parsing.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_rule_parser.py
"""

import pytest
from app.ecn.rules.models import (
    AndExpression,
    EqualsExpression,
    IncludeAssignment,
    NotExpression,
    OrExpression,
    RuleKind,
    RuleParseState,
)
from app.ecn.rules.parser import (
    canonicalize_rule,
    expression_from_data,
    expression_to_data,
    parse_rule_definition,
    stable_rule_id,
)


@pytest.mark.parametrize(
    ("source", "state"),
    [
        ("A = B --> Inclure (S = V)", RuleParseState.PARSED),
        ("(A = B / A = C) + D = ET --> Inclure (S = V)", RuleParseState.PARSED),
        ("-(A = B / A = C) --> Inclure (S = V)", RuleParseState.PARSED),
        ("A = E_GATEWAY_W/O + C = E_AUTOGUIDE_W/ --> Inclure (S = V)", RuleParseState.PARSED),
        ("A = B + C = D / E = F --> Inclure (S = V)", RuleParseState.AMBIGUOUS),
        ("A = B --> Exclure (S = V)", RuleParseState.UNSUPPORTED),
    ],
)
def test_parse_states_are_deterministic(source: str, state: RuleParseState) -> None:
    """Classify verified, mixed-precedence, and unsupported syntax without guessing."""
    assert parse_rule_definition(source, "r-1", 2).parse_state is state


def test_parser_preserves_et_and_embedded_slashes_as_scalars() -> None:
    """Keep country ET and slash-bearing values out of the operator stream."""
    parsed = parse_rule_definition(
        "A = ET + B = E_GATEWAY_W/O + C = E_AUTOGUIDE_W/ --> Inclure (S = V)",
        "r-2",
        3,
    )
    assert parsed.condition == AndExpression(
        (
            EqualsExpression("A", "ET"),
            EqualsExpression("B", "E_GATEWAY_W/O"),
            EqualsExpression("C", "E_AUTOGUIDE_W/"),
        )
    )


def test_doubled_quotes_export_apostrophe_and_multiple_actions_round_trip() -> None:
    """Normalize observed export wrappers while retaining exact original text."""
    source = "'A = \"\"N.A.\"\" --> Inclure (S = \"\"64348300\"\", X = Y)"
    first = parse_rule_definition(source, "r-3", 4)
    second = parse_rule_definition(canonicalize_rule(first), "r-3", 4)
    assert first.source_text == source
    assert first.parse_text == source[1:]
    assert first.warnings == ("LEADING_EXPORT_APOSTROPHE",)
    assert second.condition == first.condition
    assert second.action == first.action


def test_or_of_negations_is_parsed_but_requires_review() -> None:
    """Do not silently rewrite the suspicious observed negation shape."""
    parsed = parse_rule_definition(
        "-(A = B) / -(C = D) --> Inclure (S = V)", "r-4", 5
    )
    assert parsed.condition == OrExpression(
        (
            NotExpression(EqualsExpression("A", "B")),
            NotExpression(EqualsExpression("C", "D")),
        )
    )
    assert parsed.review_required is True
    assert "SUSPICIOUS_OR_OF_NEGATIONS" in parsed.warnings


def test_more_than_one_hundred_actions_are_structurally_quarantined() -> None:
    """Classify large assignment actions without assigning business semantics."""
    actions = ", ".join(f"S{i} = V{i}" for i in range(101))
    parsed = parse_rule_definition(f"A = B --> Inclure ({actions})", "r-5", 6)
    assert parsed.rule_kind is RuleKind.LARGE_MULTI_ASSIGNMENT_ACTION
    assert parsed.action is not None
    assert len(parsed.action.assignments) == 101


def test_expression_serialization_is_tagged_and_reversible() -> None:
    """Reject an untagged persistence shape and preserve every AST node."""
    expression = NotExpression(
        OrExpression(
            (
                EqualsExpression("A", "B"),
                AndExpression(
                    (EqualsExpression("C", "D"), EqualsExpression("E", "F"))
                ),
            )
        )
    )
    encoded = expression_to_data(expression)

    assert encoded["type"] == "NOT"
    assert expression_from_data(encoded) == expression
    with pytest.raises(ValueError, match="type"):
        expression_from_data({"left": "A", "right": "B"})


def test_stable_rule_id_is_artifact_and_row_scoped() -> None:
    """Changing the source row must change the stable rule identity."""
    artifact_id = "00000000-0000-4000-8000-000000000001"

    assert stable_rule_id(artifact_id, 2) == stable_rule_id(artifact_id, 2)
    assert stable_rule_id(artifact_id, 2) != stable_rule_id(artifact_id, 3)


def test_ambiguous_condition_preserves_safely_parsed_include_action() -> None:
    """Retain action indexing while keeping an ambiguous condition non-executable."""
    parsed = parse_rule_definition(
        "A = B + C = D / E = F --> Inclure (SUB = VC)", "r-ambiguous", 22
    )
    assert parsed.parse_state is RuleParseState.AMBIGUOUS
    assert parsed.condition is None
    assert parsed.action is not None
    assert parsed.action.assignments == (IncludeAssignment("SUB", "VC"),)
    assert parsed.canonical_text is None
