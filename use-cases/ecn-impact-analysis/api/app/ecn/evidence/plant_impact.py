"""One row per plant touched by an ECN, for the plant table on the overview.

Plants come from conditional rules, not from the BOM file: each BOM row's
level-3 VC group is mapped to the brands its rules require and each brand to a
plant through the Plant Code Mapping frozen at import (see
``app.ecn.rules.plant_scope``). SAP extracts are not filtered per plant, so a
material master, purchasing or stock row only counts for a plant when the rules
place that material in that plant (matched by SAP plant code).

A material's plants are those of the BOM change lines it takes part in (the
assemblies the ECN changes); a material without lines falls back to all its
current BOM occurrences. A material or line without any plant is listed under
an extra "Plant not determined" row with the reason.

Status:
* ``CONFIRMED`` (green): the plant has dated BOM changes for this ECN.
* ``CHECK`` (amber): a date is missing, the MBOM was not updated, the plant only
  has ECN objects without BOM changes, or its plant could not be determined.
"""

from collections import defaultdict
from decimal import Decimal, InvalidOperation

from app.ecn.rules.plant_scope import REASON_LABELS, merge_resolutions

STATUS_LABELS = {"CONFIRMED": "BOM change confirmed", "CHECK": "Check needed"}
UNDETERMINED_LABEL = "Plant not determined"
BOM_ALIGNMENT_LABELS = {
    "ALIGNED": "EBOM and MBOM aligned",
    "MBOM_PENDING": "Manufacturing BOM update pending",
    "CHECK": "BOM alignment needs review",
}


def _quantity(value):
    """Read a stock quantity as Decimal; unreadable values count as unknown (None)."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _display(value):
    """Return the display text of a source enum such as {"Value": ..., "Display": ...}."""
    return value.get("Display") or value.get("Value") if isinstance(value, dict) else None


def _line_scope(line):
    """Return a BOM change line's plant resolution in ``plant_scope`` shape."""
    return {"vc_groups": line.get("vc_groups", []), "plants": line.get("plants", []),
            "reason": line.get("plant_reason"), "unmapped_brands": line.get("unmapped_brands", [])}


def _assembly_summary(lines):
    """Describe one plant/assembly scope without treating dated rows as alignment proof."""
    qualifiers = {qualifier for line in lines for qualifier in line.get("qualifiers", [])}
    levels = [set(line.get("levels", [])) for line in lines]
    if "MBOM_NOT_UPDATED" in qualifiers:
        alignment = "MBOM_PENDING"
    elif levels and all({"EBOM", "MBOM"} <= level for level in levels):
        alignment = "ALIGNED"
    else:
        alignment = "CHECK"
    dates = sorted({date for line in lines for date in (line.get("end_date"), line.get("manufacturing_date")) if date})
    exemplar = lines[0]
    return {
        "assembly": exemplar.get("assembly"),
        "assembly_description": exemplar.get("assembly_description"),
        "model_range": exemplar.get("model_range"),
        "change_count": len(lines),
        "effective_from": dates[0] if dates else None,
        "effective_to": dates[-1] if dates else None,
        "bom_alignment": alignment,
        "bom_alignment_label": BOM_ALIGNMENT_LABELS[alignment],
    }


def material_scopes(bom_changes, objects=()):
    """Resolve the plants of every material the ECN touches.

    Args:
        bom_changes: Output of ``bom_changes.build_bom_changes``.
        objects: Snapshot objects; technical ones carry enhanced ``locations``
            with ``plant_scope``.

    Returns:
        {number: {"description", "role", "scope"}} where ``role`` is
        ``OUTGOING``, ``INCOMING`` or ``OTHER`` and ``scope`` has the
        ``resolve_vc_groups`` shape.
    """
    roles = {m["number"]: m["role"] for m in bom_changes["materials"]}
    by_line, descriptions = defaultdict(list), {}
    for line in bom_changes["lines"]:
        for side in (line.get("outgoing"), line.get("incoming")):
            if side:
                by_line[side["number"]].append(_line_scope(line))
                descriptions.setdefault(side["number"], side.get("description"))
    located = {}
    for obj in objects:
        if obj.get("technical") and obj.get("number"):
            descriptions[obj["number"]] = obj.get("description") or descriptions.get(obj["number"])
            located[obj["number"]] = [loc["plant_scope"] for loc in obj.get("locations", [])
                                      if loc.get("plant_scope") and not loc.get("comparison")]
    return {
        number: {"description": descriptions.get(number), "role": roles.get(number, "OTHER"),
                 "scope": merge_resolutions(by_line.get(number) or located.get(number) or [])}
        for number in sorted(set(by_line) | set(located))
    }


def build_plant_impact(bom_changes, references, source_json, objects=()):
    """Aggregate BOM changes, materials and master data per rule-derived plant.

    Args:
        bom_changes: Output of ``bom_changes.build_bom_changes`` (lines carry
            ``plants``).
        references: Pinned material master / purchase info / stock rows
            (dicts with dataset_type, material_number, plant_code, comparison,
            unrestricted_quantity).
        source_json: Raw ECN record, read only for its inventory disposition.
        objects: Snapshot objects used to list and locate the ECN's materials.

    Returns:
        {"plants": [...], "inventory_disposition": [...], "priced_new_parts": [...],
        "records_outside_rule_plants": [{plant_code, materials}]}. ``priced_new_parts``
        lists new parts found in the purchase info records of any plant. Each plant has
        plant_code (None for the undetermined row), plant_name, status,
        status_label, changes (count per change type), cutover_from/cutover_to,
        old_part_stock [{number, quantity}], new_parts_with_purchasing,
        new_parts_without_purchasing, assemblies [{assembly,
        assembly_description, model_range, change_count, effective_from,
        effective_to, bom_alignment, bom_alignment_label}], and materials
        [{number, description, role, vc_groups, reason, reason_label,
        unmapped_brands, procurement_types}].
    """
    outgoing = {m["number"] for m in bom_changes["materials"] if m["role"] == "OUTGOING"}
    incoming = {m["number"] for m in bom_changes["materials"] if m["role"] == "INCOMING"}
    touched = outgoing | incoming | {side["number"] for line in bom_changes["lines"]
                                     for side in (line.get("outgoing"), line.get("incoming")) if side}
    materials = material_scopes(bom_changes, objects)
    plants = {}

    def plant(code, name=None):
        """Return the accumulator for one plant (``None`` = undetermined), creating it on first use."""
        entry = plants.setdefault(code, {"plant_code": code, "plant_name": None, "changes": {}, "dates": set(), "assembly_lines": defaultdict(list),
                                            "stock": {}, "master": set(), "purchasing": set(), "needs_check": code is None,
                                           "materials": [], "procurement_types": defaultdict(set)})
        entry["plant_name"] = entry["plant_name"] or name
        return entry

    for line in bom_changes["lines"]:
        required_dates = ([line.get("end_date")] if line["change"] == "REMOVED" else
                          [line.get("manufacturing_date")] if line["change"] == "ADDED" else
                          [line.get("end_date"), line.get("manufacturing_date")])
        needs_check = (not all(d and d != "9999-12-31" for d in required_dates)
                       or bool({"MBOM_NOT_UPDATED", "DECLARED_PHASE_OUT"} & set(line.get("qualifiers", []))))
        # A line valid for several plants (VC group built in both) counts for each.
        for target in line.get("plants") or [{"plant_code": None, "plant_name": UNDETERMINED_LABEL}]:
            entry = plant(target["plant_code"], target["plant_name"])
            entry["changes"][line["change"]] = entry["changes"].get(line["change"], 0) + 1
            entry["needs_check"] |= needs_check
            entry["assembly_lines"][(line.get("assembly"), line.get("model_range"))].append(line)
            if line["manufacturing_date"]:
                entry["dates"].add(line["manufacturing_date"])

    for number, material in materials.items():
        scope = material["scope"]
        row = {"number": number, "description": material["description"], "role": material["role"],
               "vc_groups": scope["vc_groups"], "reason": scope["reason"],
               "reason_label": REASON_LABELS.get(scope["reason"]), "unmapped_brands": scope["unmapped_brands"]}
        for target in scope["plants"] or [{"plant_code": None, "plant_name": UNDETERMINED_LABEL}]:
            plant(target["plant_code"], target["plant_name"])["materials"].append(row)

    outside = defaultdict(set)
    # New parts with a purchase info record in any plant; the record is taken as "we have its price".
    priced = set()
    for ref in references:
        number, code = ref.get("material_number"), ref.get("plant_code")
        if ref.get("comparison") or not code or number not in touched:
            continue
        if ref.get("dataset_type") == "PURCHASE_INFO_RECORDS" and number in incoming:
            priced.add(number)
        rule_plants = {p["plant_code"] for p in materials.get(number, {"scope": {"plants": []}})["scope"]["plants"]}
        if code not in rule_plants:
            outside[code].add(number)
            continue
        entry, kind = plant(code), ref.get("dataset_type")
        if kind == "STOCK" and number in outgoing:
            quantity = _quantity(ref.get("unrestricted_quantity"))
            if quantity is not None:
                entry["stock"][number] = entry["stock"].get(number, Decimal(0)) + quantity
        elif kind == "MATERIAL_MASTER":
            if number in incoming:
                entry["master"].add(number)
            label = ref.get("procurement_type_label")
            if label and label != "Unknown":
                entry["procurement_types"][number].add(label)
        elif kind == "PURCHASE_INFO_RECORDS" and number in incoming:
            entry["purchasing"].add(number)

    rows = []
    # Determined plants in code order, then the undetermined row last.
    for code, entry in sorted(plants.items(), key=lambda item: (item[0] is None, item[0] or "")):
        status = "CONFIRMED" if entry["changes"] and not entry["needs_check"] else "CHECK"
        dates = sorted(entry["dates"])
        rows.append({
            "plant_code": code, "plant_name": entry["plant_name"], "status": status,
            "status_label": UNDETERMINED_LABEL if code is None else STATUS_LABELS[status],
            "changes": entry["changes"], "cutover_from": dates[0] if dates else None, "cutover_to": dates[-1] if dates else None,
            "old_part_stock": [{"number": n, "quantity": str(q.normalize())} for n, q in sorted(entry["stock"].items()) if q],
            "new_parts_with_purchasing": sorted(entry["purchasing"]),
            # Parts set up in this plant but with no purchase info record.
            "new_parts_without_purchasing": sorted(entry["master"] - entry["purchasing"]),
            "assemblies": [
                _assembly_summary(lines)
                for _, lines in sorted(entry["assembly_lines"].items(), key=lambda item: (
                    (item[1][0].get("assembly_description") or item[1][0].get("assembly") or "").lower(),
                    tuple(value or "" for value in item[0]),
                ))
            ],
            "materials": [
                {**material, "procurement_types": sorted(entry["procurement_types"].get(material["number"], set()))}
                for material in sorted(entry["materials"], key=lambda m: (m["role"] != "OUTGOING", m["role"] != "INCOMING", m["number"]))
            ],
        })
    dispositions = sorted({_display(link.get("InventoryDisposition")) for link in source_json.get("CNAffectLinks") or []} - {None})
    return {
        "plants": rows, "inventory_disposition": dispositions, "priced_new_parts": sorted(priced),
        "records_outside_rule_plants": [{"plant_code": code, "materials": sorted(numbers)} for code, numbers in sorted(outside.items())],
    }
