"""Recipe combination characters and APO lot sizes.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_recipes.py -q
"""

from __future__ import annotations

from production_wheel.sap_settings.policy import resolve_policy, validate_rows
from production_wheel.sap_settings.recipes import apo_rows, character_pool, recipe_rows

POLICY = resolve_policy([], "P1")


def _chars(*values):
    """Recipe character rows as stored in HANA."""

    return [{"list_of_characters_allowed_in_bw": value} for value in values]


def _members():
    """SEFI S1 with subgroup G1 (materials 200, 300) and G2 (material 100); S2 with G1."""

    return [
        {"sefi": "S1", "proposed_subgroup_code": "G1", "material": "300", "selected_pv": "PV - 1"},
        {"sefi": "S1", "proposed_subgroup_code": "G1", "material": "200", "selected_pv": "PV - 1"},
        {"sefi": "S1", "proposed_subgroup_code": "G2", "material": "100", "selected_pv": "PV - 2"},
        {"sefi": "S2", "proposed_subgroup_code": "G1", "material": "400", "selected_pv": "PV - 3"},
    ]


def test_character_order_is_digits_letters_symbols() -> None:
    """Character-code order within each family; duplicates and multi-character values drop."""

    pool = character_pool(_chars("b", "A", "1", "0", "#", "é", "AB", "0"), POLICY)
    assert pool == ["0", "1", "A", "b", "#", "é"]


def test_policy_list_is_the_fallback() -> None:
    """Without a dataset list the plant policy string supplies the characters."""

    policy = resolve_policy(validate_rows([{"plant": "P1", "setting": "recipe_characters", "value": "YX"}]), "P1")
    assert character_pool([], policy) == ["X", "Y"]
    assert character_pool([], POLICY) == []


def test_smallest_material_decides_the_character_not_the_subgroup_number() -> None:
    """G2 holds material 100, so it gets the first character within S1."""

    rows, status, _ = recipe_rows(_members(), ["0", "1"], "P1")
    assert status == "ready"
    assigned = {(row["sefi"], row["subgroup"]): row["recipe_combination"] for row in rows}
    assert assigned == {("S1", "G2"): "0", ("S1", "G1"): "1", ("S2", "G1"): "0"}


def test_overflow_and_missing_characters_block_the_tab() -> None:
    """Every SEFI with too many subgroups is listed with its need."""

    rows, status, message = recipe_rows(_members(), ["0"], "P1")
    assert (rows, status) == ([], "blocked")
    assert "S1 needs 2" in message and "S2" not in message
    assert recipe_rows(_members(), [], "P1")[1] == "blocked"


def test_apo_skips_the_default_pv_and_flags_missing_kilograms() -> None:
    """One row per SEFI and recipe; the default PV is not uploaded to APO."""

    recipes, _, _ = recipe_rows(_members(), ["0", "1"], "P1")
    masses = [{"sefi": "S1", "production_version": "PV - 2", "lot_size_kg": 5000}]
    rows = apo_rows(_members(), recipes, masses, POLICY, "26Q4")
    assert [(row["sefi"], row["recipe_combination"], row["lot_size_kg"], row["status"]) for row in rows] == [
        ("S1", "0", 5000, "ready"), ("S2", "0", None, "missing_mass")]
    assert rows[0]["comment"] == "Production Wheel 26Q4"
