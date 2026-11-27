"""Volume classes: give each package volume the compatibility of its nearest reference volume.

A plant profile in ``nearest_class`` mode does not list every package volume.
Its volume compatibility matrix lists a few **stops** (reference volumes, for
example 1 / 2.5 / 5 / 10 / 15 L) and two optional **tails**:

* ``"<n"``: every volume below the smallest stop ``n``;
* ``">n"``: every volume above the largest stop ``n``.

Each FINI package volume is classified into one class, and a pair of FINIs
gets the status of the pair of their classes. The customer's workbook does
this with a Power Query step (confirmed on 25 November 2026)::

    if Volume < 1 then "< 1"
    else if Volume > 15 then "> 15"
    else closest of ReferenceVolumes (1, 2.5, 5, 10, 15); on a tie the first one
         (the smaller volume) is kept, because the query replaces the best
         candidate only when the new one is strictly closer.

Examples with the default stops: 0.75 -> <1, 1 -> 1, 1.1 -> 1, 1.75 -> 1 (tie),
2.2 -> 2.5, 4 -> 5, 7.5 -> 5 (tie), 12.5 -> 10 (tie), 15 -> 15, 20 -> >15.

The optimizer never sees classes: when a run is created, ``expand_class_matrix``
turns the class matrix into explicit pairs of the dataset's package volumes, so
the run contract (``RunConfig.matrix_pairs``, numeric volumes only) is unchanged.
The volume -> class map is frozen with the run (``RunConfig.matrix_volume_classes``)
and reported per FINI.

Run the self-check:
    cd api && ../.venv/bin/python -m production_wheel.volume_classes
"""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping

EXACT = "exact"
NEAREST_CLASS = "nearest_class"
VOLUME_MATCHING_MODES = (EXACT, NEAREST_CLASS)
STATUSES = ("Y", "AVOID", "N")

# Default class matrix confirmed by the customer on 25 November 2026. Cells the
# customer marked "NO" are AVOID (a guide the optimizer may override when it
# lowers inventory; HARD mode blocks them). ">15" follows the customer's query:
# a class of its own, compatible only with itself.
DEFAULT_CLASS_LABELS = ("<1", "1", "2.5", "5", "10", "15", ">15")
DEFAULT_COMPATIBLE_PAIRS = (("<1", "1"), ("1", "2.5"), ("1", "5"), ("2.5", "5"), ("5", "10"), ("10", "15"))
DEFAULT_OTHER_STATUS = "AVOID"


def _positive(text: str) -> Decimal:
    """Parse a finite positive decimal or raise ``ValueError``."""
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"volume {text!r} must be a number in litres") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"volume {text!r} must be finite and positive")
    return value


def _text(value: Decimal) -> str:
    """Return the normalized decimal text used as volume identity (2.50 -> 2.5)."""
    return format(value.normalize(), "f")


def parse_label(label: object, allow_classes: bool = True) -> str:
    """Normalize one matrix axis label.

    Args:
        label: A volume in litres (``"2.50"``, ``2.5``) or, when ``allow_classes``,
            a tail label such as ``"< 1"`` or ``">15"``.
        allow_classes: Accept tail labels (``nearest_class`` profiles only).

    Returns:
        ``"2.5"``, ``"<1"``, ``">15"``...

    Raises:
        ValueError: not a positive number, or a tail in an exact matrix.
    """
    text = str(label).strip().replace(" ", "")
    if text[:1] in "<>" and text:
        if not allow_classes:
            raise ValueError(f"{label!r}: '<n' and '>n' classes need volume matching 'nearest_class'")
        return text[0] + _text(_positive(text[1:]))
    return _text(_positive(text))


def label_parts(label: str) -> tuple[int, Decimal]:
    """Return ``(kind, value)``: kind -1 for ``<n``, 0 for a stop, 1 for ``>n``."""
    if label.startswith("<"):
        return -1, Decimal(label[1:])
    if label.startswith(">"):
        return 1, Decimal(label[1:])
    return 0, Decimal(label)


def label_sort_key(label: str) -> tuple[Decimal, int]:
    """Sort key placing ``<1`` before ``1`` and ``>15`` after ``15``."""
    kind, value = label_parts(parse_label(label))
    return value, kind


def sorted_labels(labels: Iterable[object], allow_classes: bool = True) -> list[str]:
    """Return unique normalized labels in axis order."""
    return sorted({parse_label(label, allow_classes) for label in labels}, key=label_sort_key)


def check_labels(labels: Iterable[str]) -> list[str]:
    """Validate the axis of a class matrix and return it sorted.

    Rules: at least one stop when a tail is present; at most one tail of each
    side; ``"<n"`` must name the smallest stop and ``">n"`` the largest one (so
    adding a stop below 1 means renaming ``<1``).

    Raises:
        ValueError: with a planner-readable reason.
    """
    ordered = sorted_labels(labels)
    stops = [label_parts(label)[1] for label in ordered if label_parts(label)[0] == 0]
    below = [label for label in ordered if label.startswith("<")]
    above = [label for label in ordered if label.startswith(">")]
    if (below or above) and not stops:
        raise ValueError("A class matrix needs at least one reference volume besides '<n' and '>n'")
    if len(below) > 1 or len(above) > 1:
        raise ValueError("Use at most one '<n' and one '>n' class")
    if below and label_parts(below[0])[1] != min(stops):
        raise ValueError(f"'{below[0]}' must name the smallest reference volume ({_text(min(stops))}); rename it to '<{_text(min(stops))}'")
    if above and label_parts(above[0])[1] != max(stops):
        raise ValueError(f"'{above[0]}' must name the largest reference volume ({_text(max(stops))}); rename it to '>{_text(max(stops))}'")
    return ordered


def classify_volume(volume: object, labels: Iterable[str]) -> str | None:
    """Return the class label of one package volume, as the customer's query does.

    Args:
        volume: Package volume in litres.
        labels: Axis of the class matrix (stops plus optional tails).

    Returns:
        ``"<n"`` below the smallest stop, ``">n"`` above the largest one, else
        the nearest stop (ties keep the smaller stop). ``None`` when the volume
        falls in a tail the matrix does not have.
    """
    value = Decimal(str(volume))
    ordered = sorted_labels(labels)
    stops = [label_parts(label)[1] for label in ordered if label_parts(label)[0] == 0]
    if not stops:
        return None
    if value < stops[0]:
        tail = f"<{_text(stops[0])}"
        return tail if tail in ordered else None
    if value > stops[-1]:
        tail = f">{_text(stops[-1])}"
        return tail if tail in ordered else None
    best = stops[0]
    for candidate in stops[1:]:
        if abs(value - candidate) < abs(value - best):
            best = candidate
    return _text(best)


def class_pairs(rows: Iterable[Mapping[str, object]]) -> dict[tuple[str, str], str]:
    """Index class matrix rows by ordered (label, label) pair."""
    return {(parse_label(row["volume_a"]), parse_label(row["volume_b"])): str(row["status"]) for row in rows}


def expand_class_matrix(
    class_rows: Iterable[Mapping[str, object]], volumes: Iterable[object]
) -> tuple[list[dict[str, str]], list[list[str]], list[str]]:
    """Turn a class matrix into explicit pairs of concrete package volumes.

    Args:
        class_rows: Complete class matrix (``volume_a``, ``volume_b``, ``status``).
        volumes: Package volumes to classify (any numeric form).

    Returns:
        ``(rows, classes, unclassified)``:
        ``rows`` every ordered pair of classified volumes with the status of their
        classes, in row-major volume order; ``classes`` ``[[volume, class], ...]``
        sorted by volume; ``unclassified`` volumes in a tail the matrix lacks
        (they stay out of ``rows``, so the run's coverage check names them).
    """
    pairs = class_pairs(class_rows)
    labels = sorted({left for left, _ in pairs}, key=label_sort_key)
    concrete = sorted({_text(Decimal(str(v))) for v in volumes if v not in (None, "") and Decimal(str(v)) > 0}, key=Decimal)
    classes, unclassified = [], []
    for volume in concrete:
        label = classify_volume(volume, labels)
        if label is None:
            unclassified.append(volume)
        else:
            classes.append([volume, label])
    rows = [
        {"volume_a": left, "volume_b": right, "status": "Y" if left == right else pairs[left_class, right_class]}
        for left, left_class in classes
        for right, right_class in classes
    ]
    return rows, classes, unclassified


def default_class_matrix() -> list[dict[str, str]]:
    """Return the customer's default class matrix as complete ordered rows."""
    compatible = {frozenset(pair) for pair in DEFAULT_COMPATIBLE_PAIRS}
    return [
        {"volume_a": a, "volume_b": b,
         "status": "Y" if a == b or frozenset((a, b)) in compatible else DEFAULT_OTHER_STATUS}
        for a in DEFAULT_CLASS_LABELS
        for b in DEFAULT_CLASS_LABELS
    ]


def plant_volumes(rows: Iterable[Mapping[str, object]], plant: str | None) -> list[str]:
    """Return the distinct positive package volumes of one plant's ``fini_master`` rows.

    Every row counts (not only modeled ones), so a scenario that admits an
    excluded FINI still finds its volume in the expanded matrix.
    """
    found = set()
    for row in rows:
        if plant and str(row.get("plant")) != str(plant):
            continue
        value = row.get("package_volume")
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError):
            continue
        if number.is_finite() and number > 0:
            found.add(_text(number))
    return sorted(found, key=Decimal)


def with_default_classes(profile: Mapping[str, object]) -> dict:
    """Return a copy of a plant profile switched to the default class matrix.

    Used by Settings ("restore default matrix") and by the one-off migration
    script ``scripts/migrate_profiles_to_volume_classes.py``.

    Args:
        profile: Stored profile (``settings``, ``matrix_rows``, ...).

    Returns:
        A new dict with ``settings.volume_matching = "nearest_class"`` and the
        default class matrix; every other key is unchanged.
    """
    result = copy.deepcopy(dict(profile))
    result["settings"] = {**dict(result.get("settings") or {}), "volume_matching": NEAREST_CLASS}
    result["matrix_rows"] = default_class_matrix()
    return result


def needs_default_classes(profile: Mapping[str, object]) -> bool:
    """True when a profile is not yet in ``nearest_class`` mode (migration filter)."""
    return (profile.get("settings") or {}).get("volume_matching") != NEAREST_CLASS


def _self_check() -> None:
    """Assert the customer's query examples and the default matrix shape."""
    labels = DEFAULT_CLASS_LABELS
    expected = {"0.25": "<1", "0.75": "<1", "1": "1", "1.1": "1", "1.75": "1", "2.2": "2.5",
                "4": "5", "7.5": "5", "12.5": "10", "15": "15", "20": ">15", "1000": ">15"}
    for volume, label in expected.items():
        assert classify_volume(volume, labels) == label, (volume, classify_volume(volume, labels))
    rows, classes, missing = expand_class_matrix(default_class_matrix(), ["0.75", "2.2", "4", "20"])
    lookup = {(r["volume_a"], r["volume_b"]): r["status"] for r in rows}
    assert lookup["2.2", "4"] == "Y" and lookup["0.75", "4"] == "AVOID" and lookup["4", "20"] == "AVOID"
    assert not missing and len(rows) == 16
    print("volume classes self-check passed:", ", ".join(f"{v}->{c}" for v, c in classes))


if __name__ == "__main__":
    _self_check()
