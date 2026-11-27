"""Knee-point classification of frontier points into decision scenarios.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_decision_scenarios.py -q
"""

from __future__ import annotations

import pytest
from production_wheel.decision_scenarios import (
    BALANCED,
    CHANGEOVER_ORIENTED,
    INVENTORY_ORIENTED,
    classify_frontier,
    knee_position,
)


def _curve(xs, scale_x=1.0, scale_y=1.0):
    """Build solutions rows on y = 1/x; point_index follows x."""

    return [
        {
            "point_index": index,
            "j_ch": x * scale_x,
            "demand_weighted_mean_coverage_days": scale_y / x,
            "validation_status": "valid",
        }
        for index, x in enumerate(xs, start=1)
    ]


def _labels(result):
    """Map decision label to point_index for the labelled points only."""

    return {p["decision_label"]: p["point_index"] for p in result["points"] if p["decision_label"]}


def test_convex_curve_knee_and_three_neighbours_per_side() -> None:
    """On y=1/x, x=1..9 the chord slope is -1/9, so the knee is x=3."""

    result = classify_frontier(_curve(range(1, 10)))
    assert result["knee_point_index"] == 3
    assert result["chord_slope"] == pytest.approx(-1 / 9)
    assert _labels(result) == {"CO2": 1, "CO1": 2, "B": 3, "INV1": 4, "INV2": 5, "INV3": 6}
    classes = {p["point_index"]: p["decision_class"] for p in result["points"]}
    assert classes[3] == BALANCED
    assert classes[1] == CHANGEOVER_ORIENTED
    assert classes[6] == INVENTORY_ORIENTED
    assert classes[7] is None


def test_knee_is_independent_of_axis_units() -> None:
    """Rescaling J_CH and coverage changes the slope but not the knee."""

    plain = classify_frontier(_curve(range(1, 10)))
    scaled = classify_frontier(_curve(range(1, 10), scale_x=7.0, scale_y=0.01))
    assert scaled["knee_point_index"] == plain["knee_point_index"]
    assert _labels(scaled) == _labels(plain)


def test_unsorted_string_rows_from_hana_are_ordered_by_j_ch() -> None:
    """HANA returns text; classification sorts by numeric J_CH."""

    rows = [
        {key: str(value) for key, value in row.items()}
        for row in reversed(_curve(range(1, 10)))
    ]
    assert classify_frontier(rows)["knee_point_index"] == 3


def test_side_with_fewer_points_takes_what_exists() -> None:
    """A knee next to the edge yields fewer changeover-oriented points."""

    rows = [
        {"point_index": 1, "j_ch": 0, "demand_weighted_mean_coverage_days": 40},
        {"point_index": 2, "j_ch": 5, "demand_weighted_mean_coverage_days": 15},
        {"point_index": 3, "j_ch": 10, "demand_weighted_mean_coverage_days": 13},
        {"point_index": 4, "j_ch": 20, "demand_weighted_mean_coverage_days": 12},
        {"point_index": 5, "j_ch": 40, "demand_weighted_mean_coverage_days": 11},
    ]
    assert _labels(classify_frontier(rows)) == {"CO1": 1, "B": 2, "INV1": 3, "INV2": 4, "INV3": 5}


def test_single_point_is_balanced_and_invalid_points_are_excluded() -> None:
    """Failed validation and missing axes never become decision scenarios."""

    rows = [
        {"point_index": 1, "j_ch": 3, "demand_weighted_mean_coverage_days": 20},
        {"point_index": 2, "j_ch": 1, "demand_weighted_mean_coverage_days": 30, "validation_status": "invalid"},
        {"point_index": 3, "j_ch": None, "demand_weighted_mean_coverage_days": 25},
        {"point_index": 4, "j_ch": "nan", "demand_weighted_mean_coverage_days": 25},
    ]
    result = classify_frontier(rows)
    assert _labels(result) == {"B": 1}
    assert result["chord_slope"] is None
    assert {item["point_index"]: item["reason"] for item in result["excluded"]} == {
        2: "failed_validation",
        3: "missing_axis",
        4: "missing_axis",
    }


def test_neighbour_count_is_configurable() -> None:
    """One neighbour labels three points; five labels up to eleven."""

    rows = _curve(range(1, 10))
    assert len(_labels(classify_frontier(rows, neighbours=1))) == 3
    assert len(_labels(classify_frontier(rows, neighbours=5))) == 8


def test_knee_ties_go_to_lower_j_ch_and_empty_input_is_rejected() -> None:
    """Collinear points all touch the chord; the lowest J_CH wins."""

    assert knee_position([0.0, 1.0, 2.0], [2.0, 1.0, 0.0]) == (0, -1.0)
    with pytest.raises(ValueError):
        knee_position([], [])
    assert classify_frontier([])["points"] == []
