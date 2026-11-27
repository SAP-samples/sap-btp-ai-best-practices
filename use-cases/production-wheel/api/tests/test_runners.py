"""Shared runner classes and runner multiples (managed-complexity cycle stock).

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_runners.py -q
"""

from __future__ import annotations

import pytest

from production_wheel import runners
from production_wheel.runners import CycleStockPolicy, runner_multiple


def test_module_self_check_passes() -> None:
    """The classifier and multiple self-check (also run by ``python -m``) holds."""

    runners._self_check()


@pytest.mark.parametrize(
    "base,ratio,full,cls,expected",
    [
        (10, 1.0, 12, "high", (1, "reference_runner")),   # worked example P1
        (10, 2.5, 30, "high", (2, "adjusted")),           # P2: high runner still multiplied
        (10, 5.8, 70, "medium", (2, "adjusted")),         # P3: medium minimum 20
        (10, 25.0, 300, "low", (4, "adjusted")),          # P4: low minimum 40
        (10, 1.1, 11, "high", (2, "adjusted")),           # ceil(1.041) = 2
        (10, 3.0, 30, "medium", (2, "adjusted")),         # ceil(20 / 10) = 2 beats floor(1.5) = 1
        (15, 6.3, 25, "medium", (2, "adjusted")),         # ceil(20 / 15) = 2 beats floor(25 * 0.5 / 15) = 0
        (20, 10.0, 60, "medium", (1, "adjusted")),        # floor(60 * 0.5 / 20) = 1 still caps ProposedCycles = 2 above 20
        (60, 25.0, 300, "low", (3, "adjusted")),          # max_days cap floor(200 / 60) = 3
        (250, 2.0, 500, "low", (None, "base_above_max")),
        (10, 0.5, 10, "high", (None, "invalid_input")),
        (10, 2.0, 30, None, (None, "missing_input")),
    ],
)
def test_runner_multiple_branches(base, ratio, full, cls, expected) -> None:
    """Every branch of the shared multiple, on the plan's worked example and edges."""

    assert runner_multiple(base, ratio, full, cls, CycleStockPolicy()) == expected


def test_policy_values_are_editable_and_validated() -> None:
    """A per-site policy changes the result; impossible values are rejected."""

    assert runner_multiple(10, 2.5, 30, "high", CycleStockPolicy(high_min_days=30)) == (3, "adjusted")
    assert runner_multiple(10, 25.0, 300, "low", CycleStockPolicy(max_days=30)) == (3, "adjusted")
    for bad in ({"medium_batch_factor": 1.5}, {"low_min_days": 0}, {"high_min_days": -1}, {"mode": "adjusted"}):
        with pytest.raises(ValueError):
            CycleStockPolicy(**bad)
