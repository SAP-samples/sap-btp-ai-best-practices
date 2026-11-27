"""SAP settings rules as a typed section of the plant profile.

Until now these rules were loose policy rows ``(plant, setting, abc_class,
value, number)`` edited as JSON (see ``policy.py``). They now live in the plant
profile under ``settings.sap_settings``, so the Settings page can show them as a
form, the assistant can change them through its approval card, and every run of
the profile reads them without anyone writing JSON.

The generators (``dc_lot_sizes``, ``calendar``, ``recipes``, ``generate``) keep
consuming the *resolved* dict that ``policy.resolve_policy`` returns.
``SapSettingsPolicy.to_resolved`` produces exactly that shape, and
``from_resolved`` reads it back (used by the startup migration of the old rows).

No plant or customizing code appears here: defaults are the template's general
rules, and plant-specific codes (calendar lot-size profiles) default to empty,
which leaves the planning calendar tab blocked until they are set.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .policy import ABC_CLASSES

MinLotName = Literal["pallet", "half_pallet", "one_piece", "litres", "none"]
RoundingName = Literal["pallet", "layer", "half_pallet", "one_piece", "layer_at_max_days"]

# Length limits for free-text settings (SAP field lengths are far shorter; this
# only stops pasted paragraphs).
CODE_MAX_LENGTH = 64
TEXT_MAX_LENGTH = 128


class _Section(BaseModel):
    """Base for every SAP settings section: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MinLotRule(_Section):
    """Minimum lot size at the DC for one ABC class.

    Attributes:
        rule: ``pallet`` (one pallet), ``half_pallet`` (whole layers up to half a
            pallet), ``one_piece``, ``litres`` (``number`` litres converted to
            pieces) or ``none`` (no minimum: the row goes to the "without
            pallet" table).
        number: Litres for the ``litres`` rule; ignored by the other rules.
    """

    rule: MinLotName = "pallet"
    number: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_number(self):
        """The litres rule needs its litres; the other rules carry no number."""
        if self.rule == "litres" and self.number is None:
            raise ValueError("the litres rule needs a positive number of litres")
        if self.rule != "litres":
            self.number = None
        return self


class RoundingRule(_Section):
    """Rounding value at the DC for one ABC class.

    Attributes:
        rule: ``pallet``, ``layer``, ``half_pallet``, ``one_piece`` or
            ``layer_at_max_days`` (one layer when the cycle days reach the
            maximum, otherwise one pallet).
        number: Unused today; kept so the shape matches the minimum lot rule.
    """

    rule: RoundingName = "layer_at_max_days"
    number: float | None = Field(default=None, gt=0)


class PerClassMinLot(_Section):
    """Minimum lot rule for all classes, with optional overrides per ABC class.

    ``None`` for a class means "same as all classes".
    """

    all_classes: MinLotRule = Field(default_factory=MinLotRule)
    A: MinLotRule | None = None
    B: MinLotRule | None = None
    C: MinLotRule | None = None


class PerClassRounding(_Section):
    """Rounding rule for all classes, with optional overrides per ABC class.

    ``None`` for a class means "same as all classes".
    """

    all_classes: RoundingRule = Field(default_factory=RoundingRule)
    A: RoundingRule | None = None
    B: RoundingRule | None = None
    C: RoundingRule | None = None


class AbcCycleDays(_Section):
    """Cycle-stock days for materials outside the wheel, by ABC class at the DC.

    Attributes:
        A, B, C: Days per class (template defaults 5 / 10 / 20).
        unclassified: Days for materials without an ABC class at the DC;
            ``None`` skips them (they are counted as skipped).
    """

    A: float = Field(default=5, gt=0)
    B: float = Field(default=10, gt=0)
    C: float = Field(default=20, gt=0)
    unclassified: float | None = Field(default=None, gt=0)


def _clean_text(value: Any) -> str | None:
    """Strip text; empty text becomes ``None`` (meaning "not set")."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class SapSettingsPolicy(_Section):
    """Every rule SAP settings generation needs, owned by one plant profile.

    Attributes:
        dc_min_lot: Minimum lot size at the DC per ABC class (cycle stock tab).
        dc_rounding: Rounding value at the DC per ABC class (cycle stock tab).
        abc_cycle_days: Cycle-stock days of materials outside the wheel.
        calendar_frequency_code: Production frequency planned every other week.
        calendar_odd_profile / calendar_even_profile: Lot-size profiles (plant
            customizing codes ``<profile>-<calendar>``) for odd and even weeks;
            the planning calendar tab is blocked while either is empty.
        calendar_lot_size: Lot-size procedure of calendar-planned FINIs.
        calendar_reset_lot_size: Lot-size procedure of FINIs leaving the calendar.
        calendar_reset_text: Calendar/profile text of FINIs leaving the calendar.
        recipe_characters: Fallback character list, used only when the dataset
            has no ``RecipeCharacters`` table.
        kpi_line_breakdown: Add per-filling-line changeover KPI rows.
        apo_default_pv: Production version APO already uses; not uploaded.
        multi_tier_dc: The plant ships through an intermediate DC (central-DC tab).
    """

    dc_min_lot: PerClassMinLot = Field(default_factory=PerClassMinLot)
    dc_rounding: PerClassRounding = Field(default_factory=PerClassRounding)
    abc_cycle_days: AbcCycleDays = Field(default_factory=AbcCycleDays)
    calendar_frequency_code: str = Field(default="02W", min_length=1, max_length=CODE_MAX_LENGTH)
    calendar_odd_profile: str | None = Field(default=None, max_length=CODE_MAX_LENGTH)
    calendar_even_profile: str | None = Field(default=None, max_length=CODE_MAX_LENGTH)
    calendar_lot_size: str = Field(default="PK", min_length=1, max_length=CODE_MAX_LENGTH)
    calendar_reset_lot_size: str = Field(default="EX", min_length=1, max_length=CODE_MAX_LENGTH)
    calendar_reset_text: str = Field(default="Reset via ticket - to be blank", min_length=1, max_length=TEXT_MAX_LENGTH)
    recipe_characters: str | None = Field(default=None, max_length=TEXT_MAX_LENGTH)
    kpi_line_breakdown: bool = False
    apo_default_pv: str = Field(default="PV - 1", min_length=1, max_length=CODE_MAX_LENGTH)
    multi_tier_dc: bool = False

    @field_validator("calendar_odd_profile", "calendar_even_profile", "recipe_characters", mode="before")
    @classmethod
    def optional_text(cls, value):
        """Blank optional text means "not set"."""
        return _clean_text(value)

    @field_validator("calendar_frequency_code", "calendar_lot_size", "calendar_reset_lot_size",
                     "calendar_reset_text", "apo_default_pv", mode="before")
    @classmethod
    def required_text(cls, value):
        """Strip required text; blank text is then rejected by ``min_length``."""
        return "" if value is None else str(value).strip()

    @field_validator("calendar_odd_profile", "calendar_even_profile")
    @classmethod
    def calendar_code(cls, value):
        """Calendar lot-size profiles must read ``<profile>-<calendar>``.

        The planning calendar written to SAP is the text after the first ``-``,
        so a code without it would silently upload an empty calendar.
        """
        if value is None:
            return value
        prefix, dash, calendar = value.partition("-")
        if not (prefix.strip() and dash and calendar.strip()):
            raise ValueError("a lot-size profile must look like <profile>-<calendar>, e.g. ODD-W1; "
                             "the planning calendar is the text after '-'")
        return value

    @model_validator(mode="after")
    def distinct_calendar_codes(self):
        """Odd and even weeks need different profiles, or the split is meaningless."""
        if self.calendar_odd_profile and self.calendar_odd_profile == self.calendar_even_profile:
            raise ValueError("the odd- and even-week lot-size profiles must differ")
        return self

    def to_resolved(self) -> dict[str, Any]:
        """Return the settings in the shape of ``policy.resolve_policy``.

        Returns:
            ``{setting: value}``: per-class settings map ``"*"``/A/B/C to a value
            (rules as ``(rule, number)`` tuples), the two switches as ``"yes"`` /
            ``"no"``, and text settings as strings or ``None``.
        """

        def rules(section) -> dict[str, tuple[str, float | None]]:
            """Map a per-class rule section to ``{class: (rule, number)}``."""
            table = {"*": (section.all_classes.rule, section.all_classes.number)}
            for abc_class in ABC_CLASSES:
                override = getattr(section, abc_class)
                if override is not None:
                    table[abc_class] = (override.rule, override.number)
            return table

        days = {abc_class: getattr(self.abc_cycle_days, abc_class) for abc_class in ABC_CLASSES}
        if self.abc_cycle_days.unclassified is not None:
            days["*"] = self.abc_cycle_days.unclassified
        return {
            "dc_min_lot": rules(self.dc_min_lot),
            "dc_rounding": rules(self.dc_rounding),
            "abc_cycle_days": days,
            "calendar_frequency_code": self.calendar_frequency_code,
            "calendar_odd_profile": self.calendar_odd_profile,
            "calendar_even_profile": self.calendar_even_profile,
            "calendar_lot_size": self.calendar_lot_size,
            "calendar_reset_lot_size": self.calendar_reset_lot_size,
            "calendar_reset_text": self.calendar_reset_text,
            "recipe_characters": self.recipe_characters,
            "kpi_line_breakdown": "yes" if self.kpi_line_breakdown else "no",
            "apo_default_pv": self.apo_default_pv,
            "multi_tier_dc": "yes" if self.multi_tier_dc else "no",
        }

    @classmethod
    def from_resolved(cls, resolved: dict[str, Any], *, strict: bool = True) -> tuple["SapSettingsPolicy", list[str]]:
        """Build the profile section from a ``resolve_policy`` result.

        Args:
            resolved: Output of ``policy.resolve_policy`` for one plant.
            strict: When True any invalid value raises. When False each invalid
                field falls back to its default on its own, so one bad value
                never discards the others (used by the startup migration).

        Returns:
            ``(policy, notes)``: notes name every field that fell back to its
            default and why (always empty when ``strict``).

        Raises:
            ValueError: Only when ``strict`` and a value is invalid.
        """

        def rule(value) -> dict[str, Any]:
            """``(rule, number)`` tuple to a rule dict."""
            name, number = value
            return {"rule": name, "number": number}

        def per_class(table: dict[str, Any]) -> dict[str, Any]:
            """``{class: (rule, number)}`` to a per-class rule section."""
            section: dict[str, Any] = {}
            if "*" in table:
                section["all_classes"] = rule(table["*"])
            for abc_class in ABC_CLASSES:
                if abc_class in table:
                    section[abc_class] = rule(table[abc_class])
            return section

        days_table = resolved.get("abc_cycle_days") or {}
        days = {abc_class: days_table[abc_class] for abc_class in ABC_CLASSES if abc_class in days_table}
        if days_table.get("*") is not None:
            days["unclassified"] = days_table["*"]
        candidate = {
            "dc_min_lot": per_class(resolved.get("dc_min_lot") or {}),
            "dc_rounding": per_class(resolved.get("dc_rounding") or {}),
            "abc_cycle_days": days,
            **{name: resolved.get(name) for name in (
                "calendar_frequency_code", "calendar_odd_profile", "calendar_even_profile",
                "calendar_lot_size", "calendar_reset_lot_size", "calendar_reset_text",
                "recipe_characters", "apo_default_pv")},
            "kpi_line_breakdown": str(resolved.get("kpi_line_breakdown") or "no").lower() == "yes",
            "multi_tier_dc": str(resolved.get("multi_tier_dc") or "no").lower() == "yes",
        }
        if strict:
            try:
                return cls.model_validate(candidate), []
            except ValidationError as error:
                raise ValueError(_messages(error)) from error

        # Field by field: each value is tried on top of the defaults alone.
        notes: list[str] = []
        accepted: dict[str, Any] = {}
        for name, value in candidate.items():
            try:
                cls.model_validate({name: value})
                accepted[name] = value
            except ValidationError as error:
                notes.append(f"{name}: {_messages(error)}; default kept")
        try:
            return cls.model_validate(accepted), notes
        except ValidationError as error:
            # Only cross-field checks fail here (equal odd/even codes): drop both codes.
            notes.append(f"calendar profiles: {_messages(error)}; both left empty")
            accepted.pop("calendar_odd_profile", None)
            accepted.pop("calendar_even_profile", None)
            return cls.model_validate(accepted), notes


def _messages(error: ValidationError) -> str:
    """Join pydantic error messages into one readable line."""
    parts = []
    for item in error.errors():
        where = ".".join(str(part) for part in item.get("loc", ()))
        parts.append(f"{where}: {item.get('msg')}" if where else str(item.get("msg")))
    return "; ".join(parts)
