"""Operator-maintained inputs of rule-based plant assignment.

Conditional rules name brands (``GEN_E_BRAND = E_BRAND_JB``), not plants. The
Settings page maintains which brand suffix belongs to which SAP plant, and
which rule lifecycle states may decide a VC group's plants. Both values are
stored in HANA and frozen into every ECN assessment at import time.

Example::

    settings = normalize_plant_settings(
        [{"suffix": "jb", "plant_code": "PL10", "plant_name": "Plant A"}],
        ["To be Obsolete"],
    )
    settings.mappings[0].suffix  # "JB"
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.ecn.errors import SourceValidationError

# Rules being retired should not decide where a VC group is built.
DEFAULT_EXCLUDED_RULE_STATES: tuple[str, ...] = ("To be Obsolete",)

_SUFFIX = re.compile(r"^[A-Z0-9]{1,20}$")


@dataclass(frozen=True)
class PlantMapping:
    """One brand suffix mapped to one SAP plant.

    Attributes:
        suffix: Brand suffix as written after ``E_BRAND_`` in the rules (``JB``).
        plant_code: SAP plant code used by material master, purchasing and
            stock rows (``PL10``).
        plant_name: Display name of the plant (``Plant A``).
    """

    suffix: str
    plant_code: str
    plant_name: str

    def as_json(self) -> dict[str, str]:
        """Return the JSON-safe representation stored in assessments and APIs."""
        return {"suffix": self.suffix, "plant_code": self.plant_code, "plant_name": self.plant_name}


@dataclass(frozen=True)
class PlantSettings:
    """Complete plant-assignment configuration.

    Attributes:
        mappings: Brand suffix to plant rows, ordered by suffix.
        excluded_rule_states: Rule lifecycle states (``Etat``) ignored when
            deriving plants.
        available_rule_states: Lifecycle states seen in uploaded rule
            workbooks, offered as choices in Settings. Not persisted.
    """

    mappings: tuple[PlantMapping, ...]
    excluded_rule_states: tuple[str, ...]
    available_rule_states: tuple[str, ...] = ()

    def frozen_context(self) -> dict[str, list]:
        """Return the value frozen into an ECN assessment at import.

        Returns:
            ``{"mappings": [...], "excluded_rule_states": [...]}``.
        """
        return {
            "mappings": [mapping.as_json() for mapping in self.mappings],
            "excluded_rule_states": list(self.excluded_rule_states),
        }


def _text(value: object, field: str, limit: int) -> str:
    """Return a stripped, bounded text value or raise a validation error."""
    text = str(value or "").strip()
    if not text or len(text) > limit:
        raise SourceValidationError(
            "PLANT_MAPPING_INVALID", f"Each {field} must be 1 to {limit} characters."
        )
    return text


def normalize_plant_settings(
    mappings: Iterable[Mapping[str, object]], excluded_rule_states: Iterable[str]
) -> PlantSettings:
    """Validate and canonicalize user-entered plant settings.

    Suffixes and plant codes are upper-cased because rules and SAP plant codes
    are upper-case. Several suffixes may share a plant, but one plant code must
    always carry the same name.

    Args:
        mappings: Rows with ``suffix``, ``plant_code`` and ``plant_name``.
        excluded_rule_states: Lifecycle states to ignore.

    Returns:
        A ``PlantSettings`` with rows sorted by suffix and unique states.

    Raises:
        SourceValidationError: On an invalid suffix, a duplicate suffix, a
            blank or too long field, or two names for one plant code.
    """
    rows: dict[str, PlantMapping] = {}
    names: dict[str, str] = {}
    for raw in mappings:
        suffix = _text(raw.get("suffix"), "brand suffix", 20).upper()
        if not _SUFFIX.fullmatch(suffix):
            raise SourceValidationError(
                "PLANT_MAPPING_INVALID", "Brand suffixes contain only letters and digits (for example JB)."
            )
        if suffix in rows:
            raise SourceValidationError("PLANT_MAPPING_DUPLICATE", f"Brand suffix {suffix} is mapped twice.")
        code = _text(raw.get("plant_code"), "plant code", 40).upper()
        name = _text(raw.get("plant_name"), "plant name", 200)
        if names.setdefault(code, name) != name:
            raise SourceValidationError(
                "PLANT_NAME_CONFLICT", f"Plant {code} is named both {names[code]} and {name}."
            )
        rows[suffix] = PlantMapping(suffix, code, name)
    states = tuple(sorted({_text(state, "rule state", 200) for state in excluded_rule_states}))
    return PlantSettings(tuple(rows[key] for key in sorted(rows)), states)
