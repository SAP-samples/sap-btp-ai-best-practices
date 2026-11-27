"""Planner descriptions compile into complete, validated volume compatibility matrices.

Run from api/:
    ../.venv/bin/python -m pytest tests/test_matrix_description.py -q
"""

import pytest

from app.workspace.matrix_description import (
    dataset_volume_usage,
    expand_matrix_description,
    summarize_matrix_change,
)


def lookup(rows):
    """Index matrix rows by ordered volume pair."""
    return {(row["volume_a"], row["volume_b"]): row["status"] for row in rows}


def test_range_group_keeps_range_together_and_excludes_others():
    """'0.5-1 L together, exclude the rest' forbids crossing pairs, leaves outside pairs alone."""
    rows = lookup(expand_matrix_description({"volumes": ["0.5", "0.75", "1", "5", "10"],
        "default_status": "AVOID", "rules": [{"kind": "range_group", "min_litres": 0.5,
        "max_litres": 1, "within_status": "Y", "outside_status": "N"}]}))
    assert rows["0.5", "1"] == rows["0.75", "1"] == "Y"
    assert rows["0.5", "5"] == rows["5", "1"] == "N"
    assert rows["5", "10"] == "AVOID"
    assert rows["5", "5"] == "Y"


def test_difference_rules_in_litres_and_percent_of_smaller_volume():
    """Absolute and relative difference rules; later rules win over earlier ones."""
    rows = lookup(expand_matrix_description({"volumes": ["2", "2.5", "3", "4"], "default_status": "Y",
        "rules": [{"kind": "max_difference_percent", "percent": 25, "outside_status": "N"}]}))
    assert rows["2", "2.5"] == "Y" and rows["2", "3"] == "N" and rows["3", "4"] == "N"
    rows = lookup(expand_matrix_description({"volumes": ["1", "2", "4"], "default_status": "N",
        "rules": [{"kind": "max_difference_litres", "litres": 1},
                  {"kind": "range_group", "min_litres": 2, "max_litres": 4, "within_status": "AVOID"}]}))
    assert rows["1", "2"] == "Y" and rows["1", "4"] == "N" and rows["2", "4"] == "AVOID"


def test_keep_existing_pairs_only_generates_pairs_with_new_volumes():
    """Adding dataset volumes never changes the reviewed pairs of the current matrix."""
    current = expand_matrix_description({"volumes": ["1", "2"], "default_status": "N"})
    rows = expand_matrix_description({"dataset_id": "d", "keep_existing_pairs": True,
        "default_status": "N", "rules": [{"kind": "max_difference_litres", "litres": 0.5}],
        "overrides": [{"volume_a": "1", "volume_b": "2", "status": "AVOID"}]},
        current, ["1.5", "2.0", "1"])
    table = lookup(rows)
    assert {row["volume_a"] for row in rows} == {"1", "1.5", "2"}
    assert table["1", "1.5"] == table["1.5", "2"] == "Y"
    assert table["1", "2"] == table["2", "1"] == "AVOID"
    summary = summarize_matrix_change(current, rows)
    assert summary["added_volumes"] == ["1.5"] and summary["removed_volumes"] == []
    assert summary["changed_pair_count"] == 1
    assert summary["changed_pairs"] == [{"volume_a": "1", "volume_b": "2", "from": "N", "to": "AVOID"}]
    assert summary["pairs_by_status"] == {"AVOID": 1, "Y": 2}


@pytest.mark.parametrize("description, message", [
    ({"default_status": "N"}, "List the matrix volumes"),
    ({"volumes": ["1"], "default_status": "N", "rules": [{"kind": "max_difference_litres"}]}, "requires litres"),
    ({"volumes": ["1"], "default_status": "N", "rules": [{"kind": "max_difference_percent",
        "percent": 5, "litres": 1}]}, "does not use litres"),
    ({"volumes": ["1"], "default_status": "N", "rules": [{"kind": "range_group",
        "min_litres": 3, "max_litres": 1}]}, "must not exceed"),
    ({"volumes": ["1", "2"], "default_status": "N", "compatible_groups": [["1", "9"]]}, "unlisted volume"),
    ({"volumes": [str(v) for v in range(1, 61)], "dataset_id": "d", "default_status": "N"}, "1–100"),
])
def test_invalid_descriptions_are_rejected(description, message):
    """Missing volumes, wrong rule parameters and oversized combined axes never reach a proposal."""
    with pytest.raises(ValueError, match=message):
        expand_matrix_description(description, dataset_volumes=[str(v) for v in range(61, 121)])


def test_dataset_volume_usage_counts_products_modeled_in_either_mode():
    """Only optimizable rows of the selected plant count, with normalized volumes."""
    rows = [
        {"plant": "P", "package_volume": 0.5, "model_status": "modeled"},
        {"plant": "P", "package_volume": "0.50", "model_status": "excluded", "optimized_pv_model_status": "modeled"},
        {"plant": "P", "package_volume": 5, "model_status": "excluded"},
        {"plant": "Q", "package_volume": 1, "model_status": "modeled"},
        {"plant": "P", "package_volume": None, "model_status": "modeled"},
    ]
    assert dataset_volume_usage(rows, "P") == {"0.5": 2}
    assert dataset_volume_usage(rows) == {"0.5": 2, "1": 1}
