"""Reconcile the app's SAP settings rules with the customer's own SAP settings tables.

Question answered: given the same wheel (subgroups, selected PVs, RoC and
pallets exactly as the customer's product-groups workbook states them), do the
app's rules produce the same upload values as the customer's Excel tool?

Member rows are built from the workbook's wheel columns, as the customer
template reads them; the optimizer is not involved. Reference tables come from:
- the product-groups workbook itself, when it is of the older format that
  carries its own SAP output tables (default); or
- a refreshed "Wheel settings for SAP" template workbook (``--template``),
  which must have been refreshed from that same product-groups workbook.

The tool reads local files given on the command line, writes nothing to HANA
and prints a markdown report (also written to ``--report`` when given).
The SAP settings rules (calendar profiles, character pool, DC rules) come from
a JSON file, so no site code lives in code: either the ``sap_settings`` section
of a plant profile (``--profile-settings``, the current format) or older policy
rows ``{"rows": [...]}`` (``--policy``).

Examples:
    cd api
    ../.venv/bin/python -m production_wheel.sap_settings.reconcile \\
        --primary "../data/<site> - product groups for wheel.xlsx" \\
        --inputs "../data/<group> - INPUTS.xlsx" \\
        --policy /tmp/policy.json --period 26Q4 --report /tmp/reconcile.md

    # Same, with the plant profile's sap_settings section saved as JSON
    ../.venv/bin/python -m production_wheel.sap_settings.reconcile \\
        --primary ... --inputs ... --profile-settings /tmp/sap_settings.json --period 26Q4

    # Compare with a template refreshed from the same product-groups workbook
    ../.venv/bin/python -m production_wheel.sap_settings.reconcile \\
        --primary ... --inputs ... --policy ... \\
        --template "../data/generation of sap settings/<site> Template - Wheel settings for SAP.xlsx"
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from production_wheel.extraction.common import as_number, as_text, find_table
from production_wheel.sap_settings.generate import generate
from production_wheel.runners import CycleStockPolicy, RunnerThresholds
from production_wheel.sap_settings.policy import (
    LEGACY_CYCLE_STOCK_SETTINGS, LEGACY_RUNNER_SETTINGS, legacy_cycle_stock_policy, legacy_runner_thresholds,
    resolve_policy, validate_rows)

LOOKUP_VIEWS = ("dc_network", "dc_abc_classes", "dc_pallet_quantities", "production_version_masses",
                "recipe_characters")
SAMPLES = 8


def _assigned(row: dict[str, Any]) -> bool:
    """Whether a canonical FINI row belongs to the workbook's current wheel."""

    return str(row.get("baseline_assigned")) in ("1", "True", "true") and bool(row.get("legacy_baseline_group_key"))


def wheel_members(fini_master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return member rows of the workbook's current wheel, in template semantics.

    - Subgroup: recipe and subgroup of the workbook's group key
      (``SEFI/plant/recipe/subgroup``), so subgroups stay distinct per SEFI.
    - Selected PV, RoC selected PV and Pallets as per subgroup: the workbook
      values. Older workbooks have no pallet column; the template formula
      ``share x PV lot / pallet litres`` fills it from the workbook's own lot,
      forecast and pallet values.
    - FINIs outside the wheel keep an empty subgroup (they still get calendar
      resets and ABC-based cycle stock).
    """

    demand: dict[str, float] = defaultdict(float)
    for row in fini_master:
        if _assigned(row):
            demand[row["legacy_baseline_group_key"]] += as_number(row.get("forecast_litres_12m")) or 0.0
    members = []
    for row in fini_master:
        if not _assigned(row):
            members.append({**row, "proposed_subgroup_code": "", "selected_pv": ""})
            continue
        key = row["legacy_baseline_group_key"]
        pallets = as_number(row.get("legacy_pallets_per_subgroup"))
        lot, pallet = as_number(row.get("legacy_selected_pv_lot_litres")), as_number(row.get("pallet_litres_resolved"))
        forecast = as_number(row.get("forecast_litres_12m"))
        if pallets is None and None not in (lot, pallet, forecast) and pallet and demand[key]:
            pallets = lot * forecast / demand[key] / pallet
        members.append({
            **row,
            "proposed_subgroup_code": key.split("/", 2)[2],
            "selected_pv": as_text(row.get("fixed_pv")),
            "base_group_coverage_days": as_number(row.get("legacy_coverage_selected_pv_days")),
            "legacy_nominal_pallet_count": pallets,
        })
    return members


def _flag(value: Any) -> bool:
    """Read a boolean that may be persisted as text."""

    return value is True or str(value).lower() == "true"


# Generated rows per compared tab: (view, row filter, extra fields set on each row).
OURS: dict[str, tuple[str, Callable[[dict], bool], Callable[[dict], dict]]] = {
    "cycle_stock": ("sap_dc_lot_sizes", lambda row: True,
                    lambda row: {"section": "with pallet" if _flag(row.get("min_lot_identified")) else "without pallet"}),
    "planning_calendar": ("sap_planning_calendar", lambda row: True, lambda row: {}),
    "recipe_combination": ("sap_recipe_combinations", lambda row: True, lambda row: {}),
    "apo_lot_size": ("sap_apo_lot_sizes", lambda row: True, lambda row: {}),
    "changeovers": ("sap_changeover_kpis", lambda row: row.get("level") == "site_total", lambda row: {}),
}
KEYS = {"cycle_stock": ("plant", "material"), "planning_calendar": ("material",),
        "recipe_combination": ("plant", "material"), "apo_lot_size": ("sefi", "recipe_combination"),
        "changeovers": ()}
FIELDS = {
    "cycle_stock": ("section", "lot_size_code", "min_lot_size_at_dc", "rounding_value", "defined_in_wheel"),
    "planning_calendar": ("action", "lot_size_procedure", "lot_size_profile"),
    "recipe_combination": ("recipe_combination",),
    "apo_lot_size": ("lot_size_kg",),
    "changeovers": ("batches_per_week", "changeovers_per_week", "pack_changeovers_per_week",
                    "same_pack_changeovers_per_week"),
}
TOLERANCE = {"changeovers": 0.051}  # References round KPIs to 1 decimal.

# Reference sources: (table, {our field: their field}, extra fields) per tab, then optional row filter.
LEGACY = {
    "cycle_stock": [
        ("fini_cycle_stock_at_dc", {"plant": "plant", "material": "material", "lot_size_code": "lot_size",
                                    "min_lot_size_at_dc": "min_lot_size_at_dc", "rounding_value": "rounding_value",
                                    "defined_in_wheel": "defined_in_wheel"}, {"section": "with pallet"}),
        ("fini_cycle_stock_without_pallet_conversion", {"plant": "plant", "material": "material",
                                                        "lot_size_code": "lot_size",
                                                        "defined_in_wheel": "defined_in_wheel"},
         {"section": "without pallet"}),
    ],
    "planning_calendar": [
        ("planning_calendar_inform", {"material": "material", "lot_size_procedure": "lot_size",
                                      "lot_size_profile": "lot_size_profile"}, {"action": "assign"}),
        ("planning_calendar_no_calendar", {"material": "material", "lot_size_procedure": "lot_size"},
         {"action": "reset"}),
    ],
    "recipe_combination": [("fini_new_recipe_combination", {"plant": "plant", "material": "material",
                                                            "recipe_combination": "recipe_combination"}, {})],
    "apo_lot_size": [("apo_sefi_lot_size", {"sefi": "sefi", "recipe_combination": "recipe_combination",
                                            "lot_size_kg": "lot_size_kg"}, {})],
    "changeovers": [("legacy_changeover_estimation", {"batches_per_week": "batches_per_week",
                                                      "changeovers_per_week": "weekly_changes_fini_fini",
                                                      "pack_changeovers_per_week": "weekly_changes_fini_fini_with_pck_change",
                                                      "same_pack_changeovers_per_week": "weekly_fini_fini_with_same_pck"},
                     {})],
}
LEGACY_FILTERS = {"recipe_combination": lambda row: row.get("recipe_combination") not in (None, ""),
                  "changeovers": lambda row: as_text(row.get("plant")).lower() != "total"}
TEMPLATE = {
    "cycle_stock": [
        ("DC_Lot_size_with_pallet", {"plant": "Plant", "material": "Material", "lot_size_code": "Lot size",
                                     "min_lot_size_at_dc": "Min lot size at DC", "rounding_value": "Rounding value",
                                     "defined_in_wheel": "Defined in wheel"}, {"section": "with pallet"}),
        ("DC_Lot_size_without_pallet", {"plant": "Plant", "material": "Material", "lot_size_code": "Lot size",
                                        "defined_in_wheel": "Defined in wheel"}, {"section": "without pallet"}),
    ],
    "planning_calendar": [
        ("FINI_Planning_calendar_at_P_site", {"material": "Material", "lot_size_procedure": "Lot size",
                                              "lot_size_profile": "Lot size Profile"}, {"action": "assign"}),
        ("FINI_Planning_calendar_at_P_site_RESET", {"material": "Material", "lot_size_procedure": "Lot size"},
         {"action": "reset"}),
    ],
    "recipe_combination": [("Recipe_combination", {"plant": "Plant", "material": "Material",
                                                   "recipe_combination": "Recipe combination"}, {})],
    "apo_lot_size": [("APO_lot_size", {"sefi": "SEFI", "recipe_combination": "Recipe combination",
                                       "lot_size_kg": "Lot size KG"}, {})],
    "changeovers": [("Change_over_estimation", {"batches_per_week": "Average batches per week",
                                                "changeovers_per_week": "Average changeovers per week",
                                                "pack_changeovers_per_week": "Average pack-change changeovers per week",
                                                "same_pack_changeovers_per_week": "Average same-pack changeovers per week"},
                     {})],
}
TEMPLATE_FILTERS = {"changeovers": lambda row: row.get("Level") == "Site total"}


def reference_rows(tables: dict[str, list[dict]], spec: dict, filters: dict) -> dict[str, list[dict]]:
    """Map reference tables (legacy views or template Excel tables) onto our field names."""

    result: dict[str, list[dict]] = {}
    for tab, sources in spec.items():
        rows = []
        for table, mapping, extra in sources:
            for row in tables.get(table) or []:
                if tab in filters and not filters[tab](row):
                    continue
                rows.append({**{ours: row.get(theirs) for ours, theirs in mapping.items()}, **extra})
        result[tab] = rows
    return result


def _same(left: Any, right: Any, tolerance: float) -> bool:
    """Compare as numbers when both sides are numeric, else as trimmed text."""

    a, b = as_number(left), as_number(right)
    if a is not None and b is not None:
        return abs(a - b) <= max(tolerance, 1e-9)
    return (as_text(left) or "").strip().upper() == (as_text(right) or "").strip().upper()


def compare(ours: list[dict], reference: list[dict], keys: tuple[str, ...], fields: tuple[str, ...],
            tolerance: float = 0.0) -> dict[str, Any]:
    """Compare two row sets key by key.

    Returns:
        Counts (reference, ours, matched keys, only in reference, only in ours,
        duplicate keys, rows equal on every field), per-field mismatch counts and
        sample rows for each category.
    """

    def index(rows):
        indexed, duplicates = {}, 0
        for row in rows:
            key = tuple(as_text(row.get(name)) for name in keys)
            duplicates += key in indexed
            indexed.setdefault(key, row)
        return indexed, duplicates

    left, left_duplicates = index(ours)
    right, right_duplicates = index(reference)
    common = [key for key in right if key in left]
    mismatches: dict[str, list] = {name: [] for name in fields}
    equal = 0
    for key in common:
        differing = [name for name in fields if not _same(left[key].get(name), right[key].get(name), tolerance)]
        equal += not differing
        for name in differing:
            mismatches[name].append((key, left[key].get(name), right[key].get(name)))
    return {
        "reference": len(reference), "ours": len(ours), "matched": len(common), "equal": equal,
        "only_reference": [key for key in right if key not in left],
        "only_ours": [key for key in left if key not in right],
        "duplicates": {"ours": left_duplicates, "reference": right_duplicates},
        "mismatches": mismatches,
    }


def recipe_partition(ours: list[dict], reference: list[dict]) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    """Compare subgroup partitions per SEFI, ignoring which character labels each subgroup.

    Template versions label subgroups differently, but the FINIs that share a
    recipe combination must be the same. A SEFI has the same partition when
    our characters and the reference characters pair one-to-one.

    Returns:
        ``(summary, translation)``: SEFI counts and the SEFIs that differ, and a
        map ``(sefi, our character) -> reference character`` for SEFIs with the
        same partition (used to compare APO rows).
    """

    theirs = {as_text(row.get("material")): as_text(row.get("recipe_combination")) for row in reference}
    links: dict[str, set] = defaultdict(set)
    for row in ours:
        material = as_text(row.get("material"))
        if material in theirs:
            links[as_text(row.get("sefi"))].add((as_text(row.get("recipe_combination")), theirs[material]))
    translation, differing = {}, []
    for sefi, pairs in links.items():
        if len(pairs) == len({left for left, _ in pairs}) == len({right for _, right in pairs}):
            translation.update({(sefi, left): right for left, right in pairs})
        else:
            differing.append(sefi)
    return {"sefis": len(links), "same_partition": len(links) - len(differing),
            "differing": sorted(differing)}, translation


def report(results: dict[str, dict], summary: dict[str, Any], header: list[str]) -> str:
    """Render the comparison as markdown."""

    lines = ["# SAP settings reconciliation", "", *header, "",
             "| Tab | Reference rows | Our rows | Matched keys | Equal rows | Only reference | Only ours |",
             "|---|---|---|---|---|---|---|"]
    for tab, result in results.items():
        lines.append(f"| {tab} | {result['reference']} | {result['ours']} | {result['matched']} | {result['equal']} "
                     f"| {len(result['only_reference'])} | {len(result['only_ours'])} |")
    lines += ["", "Generation statuses: " + ", ".join(f"{name}={tab['status']}" for name, tab in summary["tabs"].items())]
    for tab, result in results.items():
        lines += ["", f"## {tab}", ""]
        if any(result["duplicates"].values()):
            lines.append(f"- Duplicate keys: {result['duplicates']}")
        for name, rows in result["mismatches"].items():
            if rows:
                lines.append(f"- `{name}` differs on {len(rows)} keys; samples (key: ours vs reference): "
                             + "; ".join(f"{'/'.join(key)}: {ours!r} vs {theirs!r}" for key, ours, theirs in rows[:SAMPLES]))
        for label in ("only_reference", "only_ours"):
            if result[label]:
                lines.append(f"- {label}: {len(result[label])}; samples: "
                             + ", ".join("/".join(key) for key in result[label][:SAMPLES]))
        if lines[-1] == "":
            lines.append("- All matched keys are equal; no unmatched keys.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> str:
    """Run the reconciliation from the command line and return the markdown report."""

    from tqdm import tqdm

    from production_wheel.extraction.datasets import extract_workbooks

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", type=Path, required=True, help="Product-groups workbook with the wheel")
    parser.add_argument("--inputs", type=Path, required=True, help="INPUTS workbook with the SAP lookups")
    parser.add_argument("--policy", type=Path, help="JSON {'rows': [...]} in the former /api/sap-policy format")
    parser.add_argument("--profile-settings", type=Path,
                        help="JSON of a plant profile's settings.sap_settings section (takes precedence over --policy rules)")
    parser.add_argument("--template", type=Path, help="Refreshed SAP settings template to compare with")
    parser.add_argument("--period", default="", help="Period label for the APO comment")
    parser.add_argument("--report", type=Path, help="Also write the markdown report to this file")
    args = parser.parse_args(argv)

    steps = tqdm(total=4, desc="Reconcile SAP settings")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = extract_workbooks(args.primary, args.inputs)["tables"]
    site = tables["site_parameters"][0]
    steps.update()
    raw_rows = json.loads(args.policy.read_text())["rows"] if args.policy else []
    # Former runner cut-off rows in older policy files become the run's runner thresholds.
    thresholds = RunnerThresholds(**legacy_runner_thresholds(raw_rows, site["plant"]))
    # Former runner RoC / cycle-stock cap rows become the run's cycle stock policy (single mode).
    cycle_policy = CycleStockPolicy(**legacy_cycle_stock_policy(raw_rows, site["plant"]))
    moved = LEGACY_RUNNER_SETTINGS.keys() | LEGACY_CYCLE_STOCK_SETTINGS.keys()
    policy_rows = validate_rows([row for row in raw_rows if row.get("setting") not in moved])
    policy = resolve_policy(policy_rows, site["plant"])
    if args.profile_settings:
        from production_wheel.sap_settings.profile_policy import SapSettingsPolicy

        policy = SapSettingsPolicy.model_validate(json.loads(args.profile_settings.read_text())).to_resolved()
    members = wheel_members(tables["fini_master"])
    generated, summary = generate(
        members, {view: tables.get(view) or None for view in LOOKUP_VIEWS}, policy,
        {"point_index": 0}, site["plant"], args.period,
        site["demand_days"] / site["productive_weeks"], runner_thresholds=thresholds,
        runner_thresholds_source="policy file" if args.policy else "defaults",
        cycle_stock_policy=cycle_policy, cycle_stock_policy_source="policy file" if args.policy else "defaults")
    steps.update()
    if args.template:
        from openpyxl import load_workbook

        book = load_workbook(args.template, data_only=True)
        names = [name for sheet in book for name in sheet.tables]
        reference = reference_rows({name: find_table(book, name).records for name in names}, TEMPLATE,
                                   TEMPLATE_FILTERS)
        source = f"template `{args.template.name}`"
    else:
        reference = reference_rows(tables, LEGACY, LEGACY_FILTERS)
        source = f"SAP output tables inside `{args.primary.name}`"
    steps.update()
    partition, translation = recipe_partition(generated["sap_recipe_combinations"], reference["recipe_combination"])
    results = {}
    for tab, (view, keep, extra) in OURS.items():
        ours = [{**row, **extra(row)} for row in generated[view] if keep(row)]
        if tab == "apo_lot_size":  # Compare APO rows through the partition, not the character labels.
            ours = [{**row, "recipe_combination": translation.get(
                (as_text(row.get("sefi")), as_text(row.get("recipe_combination"))), row.get("recipe_combination"))}
                for row in ours]
        results[tab] = compare(ours, reference[tab], KEYS[tab], FIELDS[tab], TOLERANCE.get(tab, 0.0))
    steps.update()
    steps.close()
    wheel = sum(bool(member["proposed_subgroup_code"]) for member in members)
    text = report(results, summary, [
        f"- Wheel: `{args.primary.name}` ({wheel} FINIs in subgroups, {len(members)} rows), plant `{site['plant']}`.",
        f"- Lookups: `{args.inputs.name}`. Reference: {source}.",
        f"- Days per week: {site['demand_days'] / site['productive_weeks']:g}. Policy rows: {len(policy_rows)}.",
        f"- Recipe partition: {partition['same_partition']} of {partition['sefis']} SEFIs group the same FINIs "
        f"(labels aside); differing SEFIs: {', '.join(partition['differing'][:SAMPLES]) or 'none'}. "
        "APO rows are compared through this pairing.",
    ])
    if args.report:
        args.report.write_text(text)
    return text


if __name__ == "__main__":
    print(main())
