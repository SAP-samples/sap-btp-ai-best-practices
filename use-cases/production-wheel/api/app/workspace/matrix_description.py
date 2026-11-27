"""Compile planner descriptions of volume compatibility into complete profile matrices.

The Workspace Assistant turns sentences such as "volumes 0.5 to 1 L go together
and exclude the rest", "only volumes within 25 % of each other may share a
group" or "add a 2 L stop that behaves like 2.5" into a typed
``MatrixDescription``. This module expands that description into every ordered
pair (Y / AVOID / N) without executing model-written code. The result goes
through the same ``validate_matrix_rows`` check as an Excel upload and is only
saved after the user approves the proposal card.

Axis labels follow the profile's volume matching (``production_wheel.volume_classes``):
    exact          every package volume is a row (numbers only);
    nearest_class  rows are reference volumes (stops) plus optional "<n" / ">n"
                   classes; each product takes the row of its nearest stop.

Expansion order for each pair (later steps win):
    1. ``keep_existing_pairs``: a pair already in the current profile matrix keeps
       its status and skips steps 2-5 (only pairs with a new label are generated).
    2. ``default_status`` (the diagonal is always Y).
    3. ``inherit_from``: a new label copies the row of an existing one
       (``{"2": "2.5"}``: 2 behaves like 2.5, and 2-2.5 is Y).
    4. ``compatible_groups``: every pair inside one listed family becomes Y.
    5. ``rules`` in the listed order (range group, litre difference, % difference).
       Rules compare numbers, so they only set pairs of two numeric labels; pairs
       with a "<n" / ">n" class come from steps 2-4 and 6.
    6. ``overrides``: explicit pairs, applied to existing pairs as well.

Example descriptions:
    all volumes of a dataset added to an exact matrix; new pairs compatible only
    within 25 %, otherwise forbidden:
    {"dataset_id": "...", "keep_existing_pairs": true, "default_status": "N",
     "rules": [{"kind": "max_difference_percent", "percent": 25,
                "within_status": "Y", "outside_status": "N"}]}
    a 2 L stop that behaves like 2.5 in a class matrix:
    {"volumes": ["2"], "keep_existing_pairs": true, "default_status": "AVOID",
     "inherit_from": {"2": "2.5"}}
    move the upper class from ">15" to ">20" with a new 20 L stop:
    {"volumes": ["20", ">20"], "keep_existing_pairs": true, "remove_volumes": [">15"],
     "default_status": "AVOID", "inherit_from": {">20": ">15", "20": "15"}}
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Iterable, Literal, Mapping

from pydantic import Field, model_validator
from production_wheel.rule_models import MatrixPair
from production_wheel.volume_classes import EXACT, NEAREST_CLASS, label_parts, label_sort_key, sorted_labels

from .models import StrictModel
from .profile_matrix import ProfileMatrixPair, validate_matrix_rows

Status = Literal["Y", "AVOID", "N"]


class VolumeRule(StrictModel):
    """One relational compatibility rule applied to every pair of listed volumes.

    Kinds:
        range_group: volumes from ``min_litres`` to ``max_litres`` (inclusive).
            A pair with both volumes inside gets ``within_status``; a pair with
            exactly one inside gets ``outside_status`` (if set). Pairs with both
            volumes outside are unchanged.
            "0.5-1 L go together and exclude the rest" = within Y, outside N.
        max_difference_litres: pairs whose absolute difference is at most
            ``litres`` get ``within_status``; the others get ``outside_status``.
        max_difference_percent: pairs whose difference, as a percentage of the
            smaller volume, is at most ``percent`` get ``within_status``; the
            others get ``outside_status``. 2 L and 2.5 L differ by 25 %.

    A rule never changes the diagonal (a volume with itself is always Y) nor a
    pair with a "<n" / ">n" class.
    ``outside_status`` left empty means the rule only grants ``within_status``.
    """

    kind: Literal["range_group", "max_difference_litres", "max_difference_percent"]
    min_litres: float | None = Field(default=None, ge=0)
    max_litres: float | None = Field(default=None, gt=0)
    litres: float | None = Field(default=None, ge=0)
    percent: float | None = Field(default=None, ge=0)
    within_status: Status = "Y"
    outside_status: Status | None = None

    @model_validator(mode="after")
    def check_parameters(self):
        """Require exactly the parameters of the chosen rule kind."""
        needed = {
            "range_group": ("min_litres", "max_litres"),
            "max_difference_litres": ("litres",),
            "max_difference_percent": ("percent",),
        }[self.kind]
        for name in ("min_litres", "max_litres", "litres", "percent"):
            value = getattr(self, name)
            if name in needed and value is None:
                raise ValueError(f"{self.kind} rule requires {name}")
            if name not in needed and value is not None:
                raise ValueError(f"{self.kind} rule does not use {name}")
        if self.kind == "range_group" and self.min_litres > self.max_litres:
            raise ValueError("range_group min_litres must not exceed max_litres")
        return self

    def status_for(self, left: Decimal, right: Decimal) -> Status | None:
        """Return the status this rule assigns to an unordered pair, or ``None`` to keep it.

        Args:
            left: Smaller (or equal) volume in litres.
            right: Larger (or equal) volume in litres.
        """
        if self.kind == "range_group":
            low, high = Decimal(str(self.min_litres)), Decimal(str(self.max_litres))
            inside = [low <= volume <= high for volume in (left, right)]
            if all(inside):
                return self.within_status
            return self.outside_status if any(inside) else None
        if self.kind == "max_difference_litres":
            within = right - left <= Decimal(str(self.litres))
        else:
            within = (right - left) * 100 <= Decimal(str(self.percent)) * left
        return self.within_status if within else self.outside_status


class MatrixDescription(StrictModel):
    """Typed planner description of a complete volume compatibility matrix.

    Labels come from ``volumes`` plus, optionally, the current profile matrix
    (``include_profile_volumes`` or ``keep_existing_pairs``) and, in an exact
    matrix only, the package volumes of a dataset (``dataset_id``, resolved by
    the caller); ``remove_volumes`` then drops labels. See the module docstring
    for the order in which statuses are applied.
    """

    volumes: list[str] = Field(default_factory=list, max_length=100)
    dataset_id: str | None = Field(default=None, max_length=64)
    include_profile_volumes: bool = False
    keep_existing_pairs: bool = False
    remove_volumes: list[str] = Field(default_factory=list, max_length=100)
    default_status: Status
    inherit_from: dict[str, str] = Field(default_factory=dict, max_length=100)
    compatible_groups: list[list[str]] = Field(default_factory=list, max_length=100)
    rules: list[VolumeRule] = Field(default_factory=list, max_length=50)
    overrides: list[ProfileMatrixPair] = Field(default_factory=list, max_length=10000)


def _normalized(values: Iterable[object], allow_classes: bool = True) -> list[str]:
    """Return unique normalized labels in axis order (duplicates are merged)."""
    return sorted_labels(values, allow_classes)


def expand_matrix_description(
    description: Mapping[str, object] | MatrixDescription,
    current_rows: Iterable[Mapping[str, object]] = (),
    dataset_volumes: Iterable[object] = (),
    volume_matching: str = EXACT,
) -> list[dict[str, str]]:
    """Compile a description into a complete, symmetric, validated matrix.

    Args:
        description: ``MatrixDescription`` or its dict form.
        current_rows: Current profile matrix rows (``volume_a``, ``volume_b``,
            ``status``); used by ``include_profile_volumes``,
            ``keep_existing_pairs``, ``remove_volumes`` and ``inherit_from``.
        dataset_volumes: Package volumes of ``description.dataset_id``, already
            read by the caller. Ignored when the description names no dataset.
        volume_matching: Volume matching of the resulting profile, ``exact`` or
            ``nearest_class``. Class labels ("<1", ">15") need ``nearest_class``;
            ``dataset_id`` needs ``exact`` (class profiles classify volumes).

    Returns:
        Every ordered pair as ``{"volume_a", "volume_b", "status"}``.

    Raises:
        ValueError: no labels, more than 100 labels, a family, inheritance or
            override that names an unlisted label, conflicting overrides, a
            non-Y diagonal or a class axis that breaks the "<n" / ">n" rule.
    """
    spec = MatrixDescription.model_validate(description)
    allow = volume_matching == NEAREST_CLASS
    if spec.dataset_id and allow:
        raise ValueError(
            "In a nearest-class matrix the dataset's package volumes are classified, not added as rows. "
            "List reference volumes (stops) in volumes instead, and preview the classification with "
            "get_dataset_volume_coverage(proposed_stops=...)")
    current = [ProfileMatrixPair.model_validate(row) for row in current_rows]
    existing = {(row.volume_a, row.volume_b): row.status for row in current}
    sources: list[object] = list(spec.volumes)
    if spec.include_profile_volumes or spec.keep_existing_pairs:
        sources += [row.volume_a for row in current]
    if spec.dataset_id:
        sources += list(dataset_volumes)
    if not sources:
        raise ValueError("List the matrix volumes, a dataset_id or include_profile_volumes")
    removed = set(_normalized(spec.remove_volumes, allow)) if spec.remove_volumes else set()
    collected = _normalized(sources, allow)
    if not removed <= set(collected):
        raise ValueError(f"remove_volumes names labels that are not in the matrix: {sorted(removed - set(collected), key=label_sort_key)}")
    volumes = [label for label in collected if label not in removed]
    if not volumes:
        raise ValueError("The matrix needs at least one volume")
    numeric = {volume: label_parts(volume)[1] for volume in volumes if label_parts(volume)[0] == 0}
    kept = set(existing) if spec.keep_existing_pairs else set()

    inherit = {}
    current_labels = {row.volume_a for row in current}
    for new, source in spec.inherit_from.items():
        new_label, source_label = _normalized([new], allow)[0], _normalized([source], allow)[0]
        if new_label not in volumes:
            raise ValueError(f"inherit_from: {new_label} is not a matrix volume")
        if source_label not in current_labels:
            raise ValueError(f"inherit_from: {source_label} is not in the current matrix")
        if new_label in current_labels:
            raise ValueError(f"inherit_from: {new_label} is already in the current matrix; use overrides to change it")
        inherit[new_label] = source_label

    def inherited(left: str, right: str) -> str | None:
        """Status a pair takes from inherited rows, or None when neither side inherits."""
        if left not in inherit and right not in inherit:
            return None
        a, b = inherit.get(left, left), inherit.get(right, right)
        if a == b:
            return "Y"
        return existing.get((a, b))

    families = []
    for family in spec.compatible_groups:
        members = set(_normalized(family, allow))
        if not members <= set(volumes):
            raise ValueError("Compatible family refers to an unlisted volume")
        families.append(members)

    pairs: dict[tuple[str, str], str] = {}
    for index, left in enumerate(volumes):
        pairs[left, left] = existing.get((left, left), "Y") if (left, left) in kept else "Y"
        for right in volumes[index + 1:]:
            if (left, right) in kept and (right, left) in kept:
                status = existing[left, right]
            else:
                status = inherited(left, right) or spec.default_status
                if any(left in family and right in family for family in families):
                    status = "Y"
                if left in numeric and right in numeric:
                    for rule in spec.rules:
                        status = rule.status_for(numeric[left], numeric[right]) or status
            pairs[left, right] = pairs[right, left] = status

    overrides: dict[tuple[str, str], str] = {}
    for row in spec.overrides:
        if row.volume_a not in volumes or row.volume_b not in volumes:
            raise ValueError("Matrix exception refers to an unlisted volume")
        key = tuple(sorted((row.volume_a, row.volume_b), key=label_sort_key))
        if key in overrides and overrides[key] != row.status:
            raise ValueError("Conflicting symmetric matrix exceptions")
        overrides[key] = row.status
        pairs[row.volume_a, row.volume_b] = pairs[row.volume_b, row.volume_a] = row.status
    # Row-major order, the same layout as the Excel template and its preview.
    return validate_matrix_rows(
        [{"volume_a": a, "volume_b": b, "status": pairs[a, b]} for a in volumes for b in volumes],
        volume_matching,
    )


def summarize_matrix_change(
    old_rows: Iterable[Mapping[str, object]], new_rows: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Describe how a proposed matrix differs from the current one, for review.

    Args:
        old_rows: Current profile matrix rows.
        new_rows: Proposed complete matrix rows.

    Returns:
        ``volume_count``, ``added_volumes``, ``removed_volumes``,
        ``pairs_by_status`` (unordered off-diagonal pairs of the new matrix),
        ``changed_pair_count`` (unordered pairs present in both matrices whose
        status changed) and ``changed_pairs`` (first 20, ``{volume_a, volume_b,
        from, to}``).
    """
    def index(rows):
        """Map normalized unordered off-diagonal pairs to their status."""
        result = {}
        for row in rows:
            pair = ProfileMatrixPair.model_validate(row)
            if pair.volume_a != pair.volume_b:
                result[tuple(sorted((pair.volume_a, pair.volume_b), key=label_sort_key))] = pair.status
        return result

    old_rows, new_rows = list(old_rows), list(new_rows)
    old, new = index(old_rows), index(new_rows)
    old_volumes = {ProfileMatrixPair.model_validate(row).volume_a for row in old_rows}
    new_volumes = {ProfileMatrixPair.model_validate(row).volume_a for row in new_rows}
    changed = [
        {"volume_a": a, "volume_b": b, "from": old[a, b], "to": status}
        for (a, b), status in sorted(new.items(), key=lambda item: tuple(map(label_sort_key, item[0])))
        if (a, b) in old and old[a, b] != status
    ]
    return {
        "volume_count": len(new_volumes),
        "added_volumes": sorted(new_volumes - old_volumes, key=label_sort_key),
        "removed_volumes": sorted(old_volumes - new_volumes, key=label_sort_key),
        "pairs_by_status": dict(sorted(Counter(new.values()).items())),
        "changed_pair_count": len(changed),
        "changed_pairs": changed[:20],
    }


def dataset_volume_usage(rows: Iterable[Mapping[str, object]], plant: str | None = None) -> dict[str, int]:
    """Count products per package volume among a dataset's optimizable FINI rows.

    A row counts when it is modeled in either production version mode
    (``model_status`` or ``optimized_pv_model_status`` equals ``modeled``), so a
    matrix built from this list covers both FIXED and OPTIMIZED runs.

    Args:
        rows: ``fini_master`` rows of one dataset.
        plant: Optional plant filter (the profile plant).

    Returns:
        ``{normalized volume: product count}`` sorted by volume.
    """
    usage: Counter[str] = Counter()
    for row in rows:
        if plant and str(row.get("plant")) != plant:
            continue
        if "modeled" not in (row.get("model_status"), row.get("optimized_pv_model_status")):
            continue
        value = row.get("package_volume")
        if value in (None, ""):
            continue
        try:
            volume = MatrixPair(volume_a=str(value), volume_b=str(value), status="Y").volume_a
        except ValueError:
            continue
        usage[volume] += 1
    return dict(sorted(usage.items(), key=lambda item: Decimal(item[0])))
