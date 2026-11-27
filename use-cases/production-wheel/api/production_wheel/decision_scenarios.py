"""Knee-point classification of a Pareto frontier into decision scenarios.

Decision makers choose from a short list instead of the full frontier: the
knee point ("Balanced") plus up to ``neighbours`` adjacent points on each side.

- Left of the knee (lower ``J_CH``, more stock) is changeover-oriented:
  ``CO1`` is nearest the knee, ``CO3`` furthest.
- Right of the knee (higher ``J_CH``, less stock) is inventory-oriented:
  ``INV1`` is nearest the knee.

Knee rule: on a smooth convex curve the knee is where the tangent is parallel
to the chord joining the endpoints. On discrete points that is the point that
minimises ``y - s * x``, where ``s`` is the chord slope. Rescaling either axis
multiplies every ``y - s * x`` by the same positive factor, so the knee does
not depend on axis units and needs no normalisation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from production_wheel.extraction.common import as_number

DEFAULT_NEIGHBOURS = 3
BALANCED = "balanced"
CHANGEOVER_ORIENTED = "changeover_oriented"
INVENTORY_ORIENTED = "inventory_oriented"
X_FIELD = "j_ch"
Y_FIELD = "demand_weighted_mean_coverage_days"


def usable_points(
    points: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split published frontier rows into usable points and excluded points.

    Args:
        points: ``solutions`` rows; HANA may return numbers as strings.

    Returns:
        ``(usable, excluded)``. Usable rows are copies sorted by ``J_CH`` then
        coverage. Excluded rows carry ``point_index`` and a ``reason``
        (failed validation or a missing/non-finite axis).
    """

    usable, excluded = [], []
    for point in points:
        if str(point.get("validation_status") or "valid") != "valid":
            excluded.append({"point_index": point.get("point_index"), "reason": "failed_validation"})
        elif as_number(point.get(X_FIELD)) is None or as_number(point.get(Y_FIELD)) is None:
            excluded.append({"point_index": point.get("point_index"), "reason": "missing_axis"})
        else:
            usable.append(dict(point))
    usable.sort(key=lambda row: (as_number(row[X_FIELD]), as_number(row[Y_FIELD]), str(row.get("point_index"))))
    return usable, excluded


def knee_position(xs: Sequence[float], ys: Sequence[float]) -> tuple[int, float | None]:
    """Return the knee index and chord slope for points sorted by ``xs``.

    Args:
        xs: ``J_CH`` values in ascending order.
        ys: Coverage days aligned with ``xs``.

    Returns:
        ``(index, slope)``. Ties go to the lower ``J_CH``. With one point, or
        when all ``xs`` are equal, the slope is ``None`` and the lowest
        coverage point is the knee.
    """

    if not xs:
        raise ValueError("knee requires at least one point")
    if len(xs) == 1 or math.isclose(xs[-1], xs[0]):
        return min(range(len(ys)), key=lambda index: (ys[index], index)), None
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    offsets = [y - slope * x for x, y in zip(xs, ys, strict=True)]
    best = min(offsets)
    tolerance = 1e-9 * max(1.0, abs(best))
    return next(index for index, value in enumerate(offsets) if value - best <= tolerance), slope


def classify_frontier(
    points: Iterable[Mapping[str, Any]],
    neighbours: int = DEFAULT_NEIGHBOURS,
) -> dict[str, Any]:
    """Label the knee and its neighbours on a published frontier.

    Args:
        points: ``solutions`` rows with ``point_index``, ``j_ch`` and
            ``demand_weighted_mean_coverage_days``.
        neighbours: Points to label on each side of the knee; when a side has
            fewer points, all of them are labelled.

    Returns:
        ``knee_point_index``, ``chord_slope``, ``excluded`` and ``points``.
        Every usable point is returned in ``J_CH`` order with ``j_ch``,
        ``coverage_days``, ``decision_class`` and ``decision_label``. The class
        and label are ``None`` for points outside the key set.
    """

    if neighbours < 0:
        raise ValueError("neighbours must be zero or positive")
    usable, excluded = usable_points(points)
    if not usable:
        return {"knee_point_index": None, "chord_slope": None, "points": [], "excluded": excluded}
    xs = [as_number(row[X_FIELD]) for row in usable]
    ys = [as_number(row[Y_FIELD]) for row in usable]
    knee, slope = knee_position(xs, ys)
    classified = []
    for position, (row, x, y) in enumerate(zip(usable, xs, ys, strict=True)):
        offset = position - knee
        if offset == 0:
            decision_class, label = BALANCED, "B"
        elif -neighbours <= offset < 0:
            decision_class, label = CHANGEOVER_ORIENTED, f"CO{-offset}"
        elif 0 < offset <= neighbours:
            decision_class, label = INVENTORY_ORIENTED, f"INV{offset}"
        else:
            decision_class, label = None, None
        classified.append(
            {
                "point_index": int(as_number(row.get("point_index")) or 0),
                "j_ch": x,
                "coverage_days": y,
                "decision_class": decision_class,
                "decision_label": label,
            }
        )
    return {
        "knee_point_index": classified[knee]["point_index"],
        "chord_slope": slope,
        "points": classified,
        "excluded": excluded,
    }
