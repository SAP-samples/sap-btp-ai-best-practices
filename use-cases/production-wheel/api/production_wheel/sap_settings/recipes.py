"""Recipe combination and APO lot size tabs.

**Recipe combination** (template ``Recipe assignment`` / ``Recipe combination``).
Each subgroup of a SEFI gets one character, which SAP uses to group the SEFI
into FINI subgroups.
- Allowed characters come from the BW character list.
- Order: digits, then letters, then symbols, each in character-code order.
- Within a SEFI, subgroups are ordered by their smallest material code. The
  optimizer's subgroup numbering never decides priority.

**APO lot size** (template ``APO lot size``; uploaded to APO by hand). One row
per SEFI and recipe combination gives the lot size in KG of the selected
production version. Rows using the PV that APO already applies by default are
skipped.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from production_wheel.extraction.common import as_number, as_text


def character_pool(characters: list[dict[str, Any]] | None, policy: dict[str, Any]) -> list[str]:
    """Return allowed recipe characters in assignment order.

    Args:
        characters: ``recipe_characters`` rows (``list_of_characters_allowed_in_bw``).
        policy: Resolved policy; its ``recipe_characters`` string is the
            fallback when the dataset has no list.

    Returns:
        Distinct single characters: digits, then letters, then symbols, each in
        character-code order.
    """

    values = [as_text(row.get("list_of_characters_allowed_in_bw")) for row in characters or []]
    if not any(values) and policy.get("recipe_characters"):
        values = list(policy["recipe_characters"])
    single = {value for value in values if value and len(value) == 1}

    def priority(character: str) -> tuple[int, int]:
        """Sort digits first, letters second, other symbols last."""
        group = 1 if "0" <= character <= "9" else 2 if character.isascii() and character.isalpha() else 3
        return group, ord(character)

    return sorted(single, key=priority)


def recipe_rows(
    wheel_members: list[dict[str, Any]],
    pool: list[str],
    plant: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Assign one character per SEFI subgroup and expand it to every FINI.

    Returns:
        ``(rows, status, message)``. The tab is blocked when there is no
        character list, or when a SEFI has more subgroups than characters; the
        message lists every SEFI with required vs. available counts.
    """

    from production_wheel.sap_settings.requirements import recipe_pool_gap

    gap = recipe_pool_gap(pool)
    if gap:
        return [], gap.status, gap.message
    first_material: dict[tuple[str, str], str] = {}
    for member in wheel_members:
        key = (as_text(member.get("sefi")), as_text(member.get("proposed_subgroup_code")))
        material = as_text(member.get("material")) or ""
        first_material[key] = min(first_material.get(key, material), material)
    by_sefi: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (sefi, subgroup), material in first_material.items():
        by_sefi[sefi].append((material, subgroup))
    overflow = [f"{sefi} needs {len(groups)}" for sefi, groups in sorted(by_sefi.items()) if len(groups) > len(pool)]
    if overflow:
        return [], "blocked", f"Only {len(pool)} recipe characters available: {'; '.join(overflow)}"
    character = {}
    for sefi, groups in by_sefi.items():
        for index, (_material, subgroup) in enumerate(sorted(groups)):
            character[sefi, subgroup] = pool[index]
    rows = [
        {
            "plant": plant,
            "material": as_text(member.get("material")),
            "recipe_combination": character[as_text(member.get("sefi")), as_text(member.get("proposed_subgroup_code"))],
            "sefi": as_text(member.get("sefi")),
            "subgroup": as_text(member.get("proposed_subgroup_code")),
            "sort_material": first_material[as_text(member.get("sefi")), as_text(member.get("proposed_subgroup_code"))],
        }
        for member in wheel_members
    ]
    rows.sort(key=lambda row: (row["sefi"] or "", row["subgroup"] or "", row["material"] or ""))
    return rows, "ready", ""


def apo_rows(
    wheel_members: list[dict[str, Any]],
    recipes: list[dict[str, Any]],
    masses: list[dict[str, Any]],
    policy: dict[str, Any],
    period_label: str,
) -> list[dict[str, Any]]:
    """Build one APO lot-size row per SEFI and recipe combination.

    Args:
        wheel_members: Members with a proposed subgroup.
        recipes: Ready ``recipe_rows`` output.
        masses: ``production_version_masses`` rows (sefi, production_version, lot_size_kg).
        policy: Resolved policy (``apo_default_pv``).
        period_label: Planning period written into the APO comment.

    Returns:
        Distinct rows. ``status`` is ``missing_mass`` when the selected PV has
        no KG lot size.
    """

    recipe = {(row["sefi"], row["subgroup"]): row["recipe_combination"] for row in recipes}
    kilograms = {(as_text(row.get("sefi")), as_text(row.get("production_version"))): as_number(row.get("lot_size_kg"))
                 for row in masses}
    rows, seen = [], set()
    for member in wheel_members:
        sefi, subgroup = as_text(member.get("sefi")), as_text(member.get("proposed_subgroup_code"))
        pv = as_text(member.get("selected_pv"))
        if not pv or pv == policy["apo_default_pv"]:
            continue
        key = (sefi, recipe.get((sefi, subgroup)), pv)
        if key in seen:
            continue
        seen.add(key)
        mass = kilograms.get((sefi, pv))
        rows.append({
            "sefi": sefi, "recipe_combination": key[1], "lot_size_kg": mass,
            "comment": f"Production Wheel {period_label}", "selected_pv": pv,
            "status": "ready" if mass is not None else "missing_mass",
        })
    rows.sort(key=lambda row: (row["sefi"] or "", row["recipe_combination"] or ""))
    return rows
