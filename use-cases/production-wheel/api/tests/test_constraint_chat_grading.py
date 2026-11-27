"""Check the grading of the live constraint-chat harness without calling a model.

Run with:
    PYTHONPATH=api .venv/bin/python -m pytest api/tests/test_constraint_chat_grading.py -q
"""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = runpy.run_path(str(ROOT / "scripts/validate_constraint_chat.py"))
CASES = {case["id"]: case for case in json.loads((ROOT / "scripts/constraint_chat_cases.example.json").read_text())}


def expression(op, *args, **values):
    """Build an expression node in the public rule vocabulary."""
    return {"op": op, "args": list(args), **values}


def proposal(rule):
    """Wrap a rule like the propose_profile_constraint event does."""
    return {"change": {"action": "add_constraint", "constraint": rule}}


def test_reference_answers_pass_and_wrong_answers_fail():
    """Reference expressions satisfy the example checks; wrong boundaries and missing proposals fail."""
    grade = HARNESS["grade"]
    field = lambda name: expression("field", field=name)
    literal = lambda value: expression("literal", value=value)
    line_cap = {"kind": "group_rule", "constraint_id": "cap",
        "when": expression("eq", field("group.selected_line"), literal("6")),
        "assertion": expression("lte", field("group.size"), literal(2))}
    assert grade(CASES["line_cap"], [proposal(line_cap)], "P") == []
    eligibility_cap = {**line_cap, "when": expression("contains", field("group.common_lines"), literal("6"))}
    assert grade(CASES["line_cap"], [proposal(eligibility_cap)], "P")
    share = {"kind": "selection_bound", "constraint_id": "share", "lower": 0,
        "measure": expression("mul", field("group.batch_count"), expression("if",
            expression("gte", field("group.pv_lot_litres"), literal(3000)), literal(0.5), literal(-0.5)))}
    assert grade(CASES["disperser_share"], [proposal(share)], "P") == []
    assert grade(CASES["disperser_share"], [], "P") == ["expected a proposal, got none"]
    assert grade(CASES["preferred_line"], [], "P") == []
    assert grade(CASES["preferred_line"], [proposal(line_cap)], "P")
    settings = {"change": {"action": "update_settings", "settings": {"maximum_group_size": 5}}}
    assert grade(CASES["max_finis"], [settings], "P") == []
