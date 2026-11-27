"""Plant policy rules for SAP settings generation.

The customer template hard-codes per-site exceptions (minimum lot size and
rounding value at the DC per ABC class, planning-calendar profiles, ...). Here
they are policy rows stored in HANA, so no plant or profile code ever appears in
code. A row is ``(plant, setting, abc_class, value, number)``:

- ``plant``: a plant code, or ``"*"`` for every plant;
- ``abc_class``: ``"A"``, ``"B"``, ``"C"`` or ``"*"`` (only for per-class settings);
- ``value``: text value or rule name;
- ``number``: numeric value, or the parameter of a rule (litres for ``litres``).

Resolution order for a plant: code defaults, then ``"*"`` rows, then the
plant's own rows. Defaults are the template's general rules. Settings with no
generic value (the calendar lot-size profiles) default to ``None``, and the tab
that needs them is reported as blocked.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from production_wheel.extraction.common import as_number, as_text

ABC_CLASSES = ("A", "B", "C")


@dataclass(frozen=True)
class SettingSpec:
    """Default, value kind and allowed rules of one policy setting.

    Attributes:
        default: Scalar default, or ``{abc_class: value}`` for per-class settings.
        kind: ``"number"``, ``"text"`` or ``"rule"`` (rule name plus optional number).
        per_class: Whether rows may target one ABC class.
        choices: Allowed rule names for ``"rule"`` settings.
        description: Business meaning shown in docs and validation errors.
    """

    default: Any
    kind: str
    per_class: bool = False
    choices: frozenset[str] = field(default_factory=frozenset)
    description: str = ""


# Former policy settings now owned by the plant profile's runner_thresholds.
# Kept only so the startup migration can move stored rows (see schema_migrations).
LEGACY_RUNNER_SETTINGS = {
    "runner_high_days": "high_max_days",
    "runner_low_days": "low_min_days",
    "runner_ratio_high": "high_max_ratio",
    "runner_ratio_low": "low_min_ratio",
}



# Former policy settings now owned by the plant profile's cycle_stock_policy.
# Kept only so the startup migration can move stored rows (see schema_migrations).
LEGACY_CYCLE_STOCK_SETTINGS = {
    "runner_medium_min_days": "medium_min_days",
    "runner_low_min_days": "low_min_days",
    "runner_medium_batch_factor": "medium_batch_factor",
    "max_cycle_days": "max_days",
}


def legacy_profile_settings(rows: list[dict[str, Any]], plant: str, mapping: dict[str, str]) -> dict[str, float]:
    """Read former policy rows as plant-profile fields for one plant.

    Same precedence as ``resolve_policy``: ``"*"`` rows, then the plant's rows.

    Args:
        rows: Raw policy rows, possibly holding settings moved to the profile.
        plant: Plant whose values are wanted.
        mapping: ``{former_setting: profile_field}``, e.g. ``LEGACY_RUNNER_SETTINGS``.

    Returns:
        ``{profile_field: number}`` for the settings present (may be empty).
    """

    found: dict[str, float] = {}
    for scope in ("*", plant):
        for row in rows:
            field_name = mapping.get(row.get("setting"))
            number = as_number(row.get("number"))
            if field_name and (as_text(row.get("plant")) or "*") == scope and number is not None:
                found[field_name] = number
    return found


def legacy_runner_thresholds(rows: list[dict[str, Any]], plant: str) -> dict[str, float]:
    """Read former runner cut-off rows as ``RunnerThresholds`` fields for one plant."""

    return legacy_profile_settings(rows, plant, LEGACY_RUNNER_SETTINGS)


def legacy_cycle_stock_policy(rows: list[dict[str, Any]], plant: str) -> dict[str, float]:
    """Read former runner RoC rows as ``CycleStockPolicy`` fields for one plant."""

    return legacy_profile_settings(rows, plant, LEGACY_CYCLE_STOCK_SETTINGS)


MIN_LOT_RULES = frozenset({"pallet", "half_pallet", "one_piece", "litres", "none"})
ROUNDING_RULES = frozenset({"pallet", "layer", "half_pallet", "one_piece", "layer_at_max_days"})

SETTINGS: dict[str, SettingSpec] = {
    "dc_min_lot": SettingSpec({"*": ("pallet", None)}, "rule", True, MIN_LOT_RULES,
        "Minimum lot size at the DC in pieces; 'litres' uses number as litres"),
    "dc_rounding": SettingSpec({"*": ("layer_at_max_days", None)}, "rule", True, ROUNDING_RULES,
        "Rounding value at the DC; layer_at_max_days = layer when days hit the maximum, else pallet"),
    "abc_cycle_days": SettingSpec({"A": 5, "B": 10, "C": 20}, "number", True,
        description="Cycle-stock days for materials outside the wheel, by ABC class at the DC"),
    "calendar_frequency_code": SettingSpec("02W", "text", description="Production frequency code planned every other week"),
    "calendar_odd_profile": SettingSpec(None, "text", description="Lot-size profile for odd weeks (plant customizing code)"),
    "calendar_even_profile": SettingSpec(None, "text", description="Lot-size profile for even weeks (plant customizing code)"),
    "calendar_lot_size": SettingSpec("PK", "text", description="Lot-size procedure for calendar-planned FINIs"),
    "calendar_reset_lot_size": SettingSpec("EX", "text", description="Lot-size procedure for FINIs leaving the calendar"),
    "calendar_reset_text": SettingSpec("Reset via ticket - to be blank", "text",
        description="Calendar/profile text for FINIs leaving the calendar"),
    "recipe_characters": SettingSpec(None, "text",
        description="Fallback list of recipe-combination characters when no dataset list exists"),
    "kpi_line_breakdown": SettingSpec("no", "text", description="'yes' adds a per-filling-line changeover KPI"),
    "apo_default_pv": SettingSpec("PV - 1", "text",
        description="Production version APO already uses by default; its rows are not uploaded"),
    "multi_tier_dc": SettingSpec("no", "text",
        description="'yes' when an intermediate DC sits between plant and selling DCs (central-DC tab)"),
}


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise policy rows and reject unknown, malformed or contradictory ones.

    Args:
        rows: Candidate rows with ``plant``, ``setting``, optional
            ``abc_class``, ``value`` and ``number``.

    Returns:
        Normalised rows, ready to store.

    Raises:
        ValueError: Listing every problem found, so one correction round suffices.
    """

    errors, normalised, seen = [], [], set()
    for index, raw in enumerate(rows, start=1):
        plant = as_text(raw.get("plant")) or "*"
        setting = as_text(raw.get("setting"))
        abc_class = (as_text(raw.get("abc_class")) or "*").upper()
        value, number = as_text(raw.get("value")), as_number(raw.get("number"))
        spec = SETTINGS.get(setting or "")
        where = f"row {index} ({plant}/{setting}/{abc_class})"
        if setting in LEGACY_RUNNER_SETTINGS:
            errors.append(f"{where}: runner classes are now set in the plant profile (runner thresholds)")
            continue
        if setting in LEGACY_CYCLE_STOCK_SETTINGS:
            errors.append(f"{where}: runner RoC multiples and the cycle-stock cap are now set in the plant profile (cycle stock strategy)")
            continue
        if spec is None:
            errors.append(f"{where}: unknown setting")
            continue
        if abc_class != "*" and (not spec.per_class or abc_class not in ABC_CLASSES):
            errors.append(f"{where}: ABC class not allowed for this setting")
        if spec.kind == "number" and (number is None or number <= 0):
            errors.append(f"{where}: a positive number is required")
        if spec.kind == "rule":
            if value not in spec.choices:
                errors.append(f"{where}: rule must be one of {sorted(spec.choices)}")
            elif value == "litres" and (number is None or number <= 0):
                errors.append(f"{where}: the litres rule needs a positive number")
        if spec.kind == "text" and not value:
            errors.append(f"{where}: a text value is required")
        if (plant, setting, abc_class) in seen:
            errors.append(f"{where}: duplicate row")
        seen.add((plant, setting, abc_class))
        normalised.append({"plant": plant, "setting": setting, "abc_class": abc_class, "value": value, "number": number})
    if errors:
        raise ValueError("; ".join(errors))
    return normalised


def resolve_policy(rows: list[dict[str, Any]], plant: str) -> dict[str, Any]:
    """Return the effective settings of one plant.

    Args:
        rows: Validated policy rows.
        plant: Plant whose settings are resolved.

    Returns:
        ``{setting: value}``. Per-class settings map ABC class (or ``"*"``) to a
        value; rule settings use ``(rule, number)`` tuples.
    """

    resolved = {name: copy.deepcopy(spec.default) for name, spec in SETTINGS.items()}
    for scope in ("*", plant):
        for row in rows:
            if row["plant"] != scope or row["setting"] not in SETTINGS:
                continue
            spec = SETTINGS[row["setting"]]
            value = (
                row["number"] if spec.kind == "number"
                else (row["value"], row["number"]) if spec.kind == "rule"
                else row["value"]
            )
            if spec.per_class:
                resolved[row["setting"]][row["abc_class"] or "*"] = value
            else:
                resolved[row["setting"]] = value
    return resolved


def class_value(resolved: dict[str, Any], setting: str, abc_class: str | None) -> Any:
    """Return a per-class setting for one ABC class, falling back to ``"*"``."""

    table = resolved[setting]
    return table.get((abc_class or "").upper(), table.get("*"))
