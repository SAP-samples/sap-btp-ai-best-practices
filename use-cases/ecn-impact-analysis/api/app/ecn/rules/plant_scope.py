"""Derive the plants of BOM rows from conditional rules and the Plant Code Mapping.

Plant selection happens at the level-3 VC group of the 150% EBOM. A VC-group
rule looks like::

    (GEN_E_BRAND = E_BRAND_IS / GEN_E_BRAND = E_BRAND_JB) + GEN_E_GEARBOX = ...
        --> Inclure (Q900VS1030 = Q900VV1030207)

``Q900VS1030`` is the level-2 slot and ``Q900VV1030207`` the level-3 VC group.
In the supplied workbook the brand term is always a positive conjunct (only
OR-ed with other brands, never negated), so a rule's brand suffixes are a
necessary condition. A VC group's plants are therefore the union, over every
admitted rule that includes it, of the plants mapped to those suffixes. Other
conjuncts (gearbox, tyres, country...) narrow the configuration, not the plant.

Every function is pure: the caller loads rules and mappings from HANA.

Example::

    index = build_vc_index(rules, excluded_states=("To be Obsolete",))
    resolve_row(row, index, mappings)
    # {"vc_groups": ["Q900VV1030207"], "plants": [{"plant_code": "PL10", ...}],
    #  "reason": None, "unmapped_brands": []}
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.ecn.models import BomKind
from app.ecn.rules.models import (
    AndExpression,
    EqualsExpression,
    OrExpression,
    ParsedRule,
    RuleExpression,
)

BRAND_PREFIX = "E_BRAND_"

# Why a row or material has no plant. Shown to users in the Plants affected card.
REASON_LABELS = {
    "NO_RULES_PINNED": "No conditional rules were available when the ECN was imported",
    "MAPPING_EMPTY": "The Plant Code Mapping was empty when the ECN was imported",
    "NOT_IN_BOM": "Not found in the supplied BOMs",
    "LEVEL_1_COMMON": "Level-1 item common to all plants",
    "NO_RULE_FOR_VC_GROUP": "No conditional rule selects its VC group",
    "NO_BRAND_IN_RULES": "The rules of its VC group name no brand",
    "BRAND_NOT_MAPPED": "The brand of its VC group has no plant in the Plant Code Mapping",
}


@dataclass(frozen=True)
class VcRuleIndex:
    """Brands and rules per VC group, built from one or more rule workbooks.

    Attributes:
        brands: VC group -> brand suffixes named by its admitted rules.
        rule_numbers: VC group -> rule numbers that include it.
        slots: Level-2 slot -> VC groups the rules can select for it.
    """

    brands: Mapping[str, frozenset[str]]
    rule_numbers: Mapping[str, tuple[str, ...]]
    slots: Mapping[str, frozenset[str]]


EMPTY_INDEX = VcRuleIndex({}, {}, {})


def brand_suffixes(expression: RuleExpression | None) -> frozenset[str]:
    """Return the brand suffixes a rule condition requires.

    Negated brands are ignored: ``-(GEN_E_BRAND = E_BRAND_JD)`` does not place
    the VC group in the JD plant.

    Args:
        expression: Parsed rule condition (``None`` for unparsed rules).

    Returns:
        Suffixes such as ``{"JB", "IS"}``.
    """
    if isinstance(expression, EqualsExpression):
        value = expression.right
        return frozenset({value[len(BRAND_PREFIX):]}) if value.startswith(BRAND_PREFIX) else frozenset()
    if isinstance(expression, (AndExpression, OrExpression)):
        return frozenset().union(*(brand_suffixes(operand) for operand in expression.operands))
    return frozenset()  # NotExpression or None


def build_vc_index(rules: Iterable[ParsedRule], excluded_states: Iterable[str] = ()) -> VcRuleIndex:
    """Index every Include assignment of the admitted rules by VC group and slot.

    Args:
        rules: Parsed rules of every pinned rule workbook.
        excluded_states: Lifecycle states (``Etat``) whose rules are ignored.

    Returns:
        A ``VcRuleIndex``. Option-derivation rules (``GEN_E_X = ...``) are
        indexed too; they are simply never looked up as VC groups.
    """
    excluded = set(excluded_states)
    brands: dict[str, set[str]] = {}
    numbers: dict[str, list[str]] = {}
    slots: dict[str, set[str]] = {}
    for rule in rules:
        if rule.action is None or rule.lifecycle_state in excluded:
            continue
        suffixes = brand_suffixes(rule.condition)
        for assignment in rule.action.assignments:
            brands.setdefault(assignment.value, set()).update(suffixes)
            numbers.setdefault(assignment.value, []).append(rule.rule_number or rule.rule_id)
            slots.setdefault(assignment.characteristic, set()).add(assignment.value)
    return VcRuleIndex(
        {vc: frozenset(value) for vc, value in brands.items()},
        {vc: tuple(sorted(set(value))) for vc, value in numbers.items()},
        {slot: frozenset(value) for slot, value in slots.items()},
    )


def row_vc_groups(row, index: VcRuleIndex) -> tuple[str, ...] | None:
    """Return the level-3 VC groups a BOM row belongs to.

    * 150% EBOM, level 3 and deeper: the ``VC`` column.
    * 150% EBOM, level 2 (slot): every VC group the rules select for the slot.
    * 150% EBOM, level 1: ``None`` (common to every plant).
    * 130% EBOM: the root (``.1``) of the row's hierarchy path.
    * MBOM: the ``Item`` without its model suffix (``Q900VV0120800_Q900``).

    Args:
        row: A ``BomRow``.
        index: Rule index providing slot members.

    Returns:
        Sorted VC groups, ``()`` when none is known, ``None`` for level 1.
    """
    if row.bom_kind == BomKind.MBOM_130:
        key = (row.item_value or row.parent_material_number or "").split("_")[0]
    elif row.bom_kind == BomKind.RAW_EBOM_130:
        key = (row.hierarchy_path or "").split("/")[0]
    elif row.dotted_level == "1":
        return None
    elif row.dotted_level == "2":
        return tuple(sorted(index.slots.get(row.sub_value or row.component_number, ())))
    else:
        key = row.vc_value
    return (key,) if key else ()


def resolve_vc_groups(
    vc_groups: tuple[str, ...] | None,
    index: VcRuleIndex,
    mappings: Iterable[Mapping[str, str]],
    rules_pinned: bool = True,
) -> dict:
    """Map VC groups to plants through their rules' brands.

    Args:
        vc_groups: Output of ``row_vc_groups`` (``None`` for level 1).
        index: Rule index of the assessment.
        mappings: Frozen Plant Code Mapping rows (suffix, plant_code, plant_name).
        rules_pinned: Whether any rule workbook was pinned by the assessment.

    Returns:
        JSON-safe ``{"vc_groups", "plants", "reason", "unmapped_brands"}``.
        ``reason`` is ``None`` when at least one plant was found.
    """
    by_suffix = {m["suffix"]: m for m in mappings}
    groups = list(vc_groups or ())
    brands = set().union(*(index.brands.get(vc, ()) for vc in groups)) if groups else set()
    plants = {by_suffix[b]["plant_code"]: by_suffix[b]["plant_name"] for b in brands if b in by_suffix}
    unmapped = sorted(brands - set(by_suffix))
    if plants:
        reason = None
    elif not rules_pinned:
        reason = "NO_RULES_PINNED"
    elif not by_suffix:
        reason = "MAPPING_EMPTY"
    elif vc_groups is None:
        reason = "LEVEL_1_COMMON"
    elif not groups:
        reason = "NOT_IN_BOM"
    elif not any(vc in index.brands for vc in groups):
        reason = "NO_RULE_FOR_VC_GROUP"
    elif not brands:
        reason = "NO_BRAND_IN_RULES"
    else:
        reason = "BRAND_NOT_MAPPED"
    return {
        "vc_groups": groups,
        "plants": [{"plant_code": code, "plant_name": name} for code, name in sorted(plants.items())],
        "reason": reason,
        "unmapped_brands": unmapped,
    }


def resolve_row(row, index: VcRuleIndex, mappings, rules_pinned: bool = True) -> dict:
    """Resolve one BOM row's VC groups and plants (see ``resolve_vc_groups``)."""
    return resolve_vc_groups(row_vc_groups(row, index), index, mappings, rules_pinned)


def merge_resolutions(resolutions: Iterable[dict]) -> dict:
    """Union several row resolutions into one material- or line-level result.

    Args:
        resolutions: Outputs of ``resolve_vc_groups``.

    Returns:
        The same shape; ``reason`` is the first reason seen when no plant was
        found, and ``NOT_IN_BOM`` when there was nothing to merge.
    """
    resolutions = list(resolutions)
    plants = {p["plant_code"]: p for r in resolutions for p in r["plants"]}
    return {
        "vc_groups": sorted({vc for r in resolutions for vc in r["vc_groups"]}),
        "plants": [plants[code] for code in sorted(plants)],
        "reason": None if plants else next((r["reason"] for r in resolutions if r["reason"]), "NOT_IN_BOM"),
        "unmapped_brands": sorted({b for r in resolutions for b in r["unmapped_brands"]}),
    }
