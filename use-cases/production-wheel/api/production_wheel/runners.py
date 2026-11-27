"""Single definition of high / medium / low runners, shared by optimizer and SAP settings.

A runner class says how fast a FINI consumes one batch of its SEFI. Two views are
combined and the faster class wins:

- **Absolute**: coverage days of one full batch for the FINI alone.
  ``coverage <= high_max_days`` is high, ``coverage >= low_min_days`` is low,
  anything in between is medium.
- **Relative**: the FINI's coverage divided by the smallest coverage among the
  FINIs of the same plant/SEFI (the SEFI's fastest runner), rounded to one
  decimal. ``ratio <= high_max_ratio`` is high, ``ratio >= low_min_ratio`` is
  low, anything in between is medium.

The thresholds belong to the plant profile and are frozen with every run, so the
optimizer business rules (``member.runner``) and the SAP settings runner audit
classify with the same numbers.

The plant profile also holds the ``CycleStockPolicy``: whether slower runners of a
subgroup keep the subgroup's coverage (``single``) or get a whole multiple of it
(``runner_multiple``, see ``runner_multiple``), and the numbers of that rule.

Self-check::

    python -m production_wheel.runners
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Canonical lowercase values used by business rules and stored results.
RUNNER_CLASSES = ("high", "medium", "low")


class RunnerThresholds(BaseModel):
    """The four cut-offs that split FINIs into high, medium and low runners.

    Attributes:
        high_max_days: Full-batch coverage at or below this is a high runner.
        low_min_days: Full-batch coverage at or above this is a low runner.
        high_max_ratio: Ratio to the SEFI's fastest runner at or below this is high.
        low_min_ratio: Ratio to the SEFI's fastest runner at or above this is low.

    Defaults are the customer template values (10 / 80 days, 3 / 10 ratio).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    high_max_days: float = Field(default=10, gt=0)
    low_min_days: float = Field(default=80, gt=0)
    high_max_ratio: float = Field(default=3, ge=1)
    low_min_ratio: float = Field(default=10, ge=1)

    @model_validator(mode="after")
    def check_order(self):
        """Require a non-empty medium band for both views.

        Returns:
            The validated thresholds.

        Raises:
            ValueError: A high cut-off that is not below its low cut-off.
        """
        if not self.high_max_days < self.low_min_days:
            raise ValueError("high_max_days must be below low_min_days")
        if not self.high_max_ratio < self.low_min_ratio:
            raise ValueError("high_max_ratio must be below low_min_ratio")
        return self


CYCLE_STOCK_MODES = ("single", "runner_multiple", "optimizer_choice")
# Strategy a single subgroup uses (optimizer_choice picks one of these per subgroup).
CYCLE_STOCK_STRATEGIES = ("single", "runner_multiple")


class CycleStockPolicy(BaseModel):
    """How a subgroup's FINIs share the group's coverage (RoC), per plant profile.

    - ``single``: every FINI of a subgroup gets the subgroup's RoC (simple sites).
    - ``runner_multiple``: only the fastest FINI keeps the RoC; every other FINI
      gets a whole multiple of it and is filled every 2nd, 3rd... batch
      (managed-complexity sites). See ``runner_multiple``.
    - ``optimizer_choice``: the optimizer may pick either version per subgroup.

    Attributes:
        mode: One of ``CYCLE_STOCK_MODES``.
        high_min_days: Minimum multiplied coverage of a high runner that is not
            the fastest of its subgroup.
        medium_min_days: Minimum multiplied coverage of a medium runner.
        low_min_days: Minimum multiplied coverage of a low runner.
        medium_batch_factor: A medium runner covers at most this share of its
            full-batch coverage.
        max_days: Cap of cycle-stock days (also the SAP DC lot-size cap).
        mixed_groups_only: In ``optimizer_choice``, offer the multiple version
            only to subgroups whose FINIs span two or more runner classes.

    Defaults are the customer template values; ``single`` keeps today's numbers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mode: Literal["single", "runner_multiple", "optimizer_choice"] = "single"
    # TODO(managed-complexity): open question 3, which number is "the multiplier" of high runners.
    high_min_days: float = Field(default=0, ge=0)
    medium_min_days: float = Field(default=20, gt=0)
    low_min_days: float = Field(default=40, gt=0)
    medium_batch_factor: float = Field(default=0.5, gt=0, le=1)
    max_days: float = Field(default=200, gt=0)
    mixed_groups_only: bool = True


def _band(value: float, high_max: float, low_min: float) -> str:
    """Place one value in the high / medium / low band of a pair of cut-offs."""
    return "high" if value <= high_max else "low" if value >= low_min else "medium"


def classify(coverage: float | None, ratio: float | None, thresholds: RunnerThresholds) -> str | None:
    """Return the runner class of one FINI; the faster of the two views wins.

    Args:
        coverage: Full-batch coverage days of the FINI alone; ``None`` or
            non-positive values skip the absolute view.
        ratio: Coverage divided by the SEFI's fastest coverage (see ``sefi_ratio``);
            ``None`` skips the relative view.
        thresholds: Cut-offs of the plant profile.

    Returns:
        ``"high"``, ``"medium"``, ``"low"``, or ``None`` when neither view has data.
    """
    views = []
    if coverage is not None and coverage > 0:
        views.append(_band(coverage, thresholds.high_max_days, thresholds.low_min_days))
    if ratio is not None:
        views.append(_band(ratio, thresholds.high_max_ratio, thresholds.low_min_ratio))
    return next((name for name in RUNNER_CLASSES if name in views), None)


def block_minimum(coverages: Iterable[float | None]) -> float | None:
    """Return the smallest positive coverage of a plant/SEFI block, or ``None``."""
    positive = [value for value in coverages if value is not None and value > 0]
    return min(positive) if positive else None


def sefi_ratio(coverage: float | None, minimum: float | None) -> float | None:
    """Return coverage / the SEFI's fastest coverage, rounded to one decimal.

    Python ``round`` (half to even) matches the customer's Power Query step.

    Args:
        coverage: Full-batch coverage of the FINI.
        minimum: ``block_minimum`` of its plant/SEFI block.

    Returns:
        The ratio (at least 1.0 for valid inputs), or ``None`` when either input
        is missing or non-positive.
    """
    if coverage is None or minimum is None or coverage <= 0 or minimum <= 0:
        return None
    return round(coverage / minimum, 1)


def excel_round(value: float | None, digits: int = 0) -> float | None:
    """Round like Excel ``ROUND`` (half away from zero); ``None`` stays ``None``.

    Lives here (not in ``sap_settings``) because the optimizer's runner multiples
    and the SAP settings template round the subgroup ratio the same way.
    """

    if value is None:
        return None
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


# Status of one runner_multiple computation; the first two mean "use the multiple".
MULTIPLE_STATUSES = ("reference_runner", "adjusted", "missing_input", "invalid_input", "base_above_max")


def runner_multiple(
    base_days: float | None,
    group_ratio: float | None,
    full_batch_days: float | None,
    runner_class: str | None,
    policy: CycleStockPolicy,
) -> tuple[int | None, str]:
    """Return how many of its subgroup's batches one FINI's fill must cover.

    Port of the customer template column ``Adjusted RoC selected PV``: the
    fastest FINI of a subgroup keeps the subgroup's coverage (multiple 1); every
    other FINI is filled only every ``multiple``-th batch, so its coverage is
    ``base_days * multiple``::

        cycles   = ceil(max(base * (1 + log10(ratio)), class minimum) / base)
        medium:  cycles = max(1, ceil(medium_min_days / base), min(cycles, floor(full_batch * factor / base)))
        multiple = min(cycles, floor(max_days / base))

    Args:
        base_days: Coverage of the subgroup (RoC of the selected PV).
        group_ratio: The FINI's full-batch coverage divided by the smallest one
            in its subgroup, rounded with ``excel_round(..., 1)``.
        full_batch_days: The FINI's full-batch coverage ("RoC 1 full SEFI batch").
        runner_class: ``"high"``, ``"medium"`` or ``"low"`` from ``classify``.
        policy: Class minimums, medium factor and cap of the plant profile.

    Returns:
        ``(multiple, status)``; status is one of ``MULTIPLE_STATUSES``.
        ``reference_runner`` (ratio 1) returns 1; ``adjusted`` returns the
        multiple; unusable inputs return ``None``.
    """

    if base_days is None or group_ratio is None or full_batch_days is None or runner_class is None:
        return None, "missing_input"
    if base_days <= 0 or group_ratio < 1 or full_batch_days <= 0:
        return None, "invalid_input"
    if group_ratio == 1:
        return 1, "reference_runner"
    if base_days > policy.max_days:
        return None, "base_above_max"
    minimum = {"high": policy.high_min_days, "medium": policy.medium_min_days,
               "low": policy.low_min_days}[runner_class]
    cycles = math.ceil(max(base_days * (1 + math.log10(group_ratio)), minimum) / base_days)
    if runner_class == "medium":
        cycles = max(
            1,
            math.ceil(policy.medium_min_days / base_days),
            min(cycles, math.floor(full_batch_days * policy.medium_batch_factor / base_days)),
        )
    return min(cycles, math.floor(policy.max_days / base_days)), "adjusted"


def _self_check() -> None:
    """Assert the boundary behaviour of the classifier; raises on regression."""
    t = RunnerThresholds()
    assert classify(10, None, t) == "high"           # cut-offs are inclusive
    assert classify(10.01, None, t) == "medium"
    assert classify(80, None, t) == "low"
    assert classify(79.9, None, t) == "medium"
    assert classify(50, 3, t) == "high"              # faster view wins
    assert classify(90, 5, t) == "medium"
    assert classify(None, 10, t) == "low"
    assert classify(None, None, t) is None
    assert classify(0, None, t) is None
    assert sefi_ratio(25, block_minimum([None, 10, 0, 40])) == 2.5
    assert sefi_ratio(None, 10) is None
    for bad in ({"high_max_days": 80}, {"high_max_ratio": 10}):
        try:
            RunnerThresholds(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted inverted thresholds {bad}")
    assert math.isclose(RunnerThresholds().low_min_ratio, 10)
    # Worked example of docs/managed-complexity.md: base 10 days.
    p = CycleStockPolicy()
    assert runner_multiple(10, 1.0, 12, "high", p) == (1, "reference_runner")
    assert runner_multiple(10, 2.5, 30, "high", p) == (2, "adjusted")
    assert runner_multiple(10, 5.8, 70, "medium", p) == (2, "adjusted")
    assert runner_multiple(10, 25.0, 300, "low", p) == (4, "adjusted")
    assert runner_multiple(10, 1.1, 11, "high", p) == (2, "adjusted")      # ceil(1.041)
    assert runner_multiple(10, 3.0, 30, "medium", p) == (2, "adjusted")    # ceil(20 / 10) beats floor(30 * 0.5 / 10)
    assert runner_multiple(20, 10.0, 60, "medium", p) == (1, "adjusted")   # floor(60 * 0.5 / 20) still caps above 20
    assert runner_multiple(60, 25.0, 300, "low", p) == (3, "adjusted")     # floor(200 / 60)
    assert runner_multiple(250, 2.0, 500, "low", p) == (None, "base_above_max")
    assert runner_multiple(None, 2.0, 30, "low", p) == (None, "missing_input")
    assert excel_round(2.25, 1) == 2.3 and round(2.25, 1) == 2.2


if __name__ == "__main__":
    _self_check()
    print("runner classification and multiple self-check passed")
