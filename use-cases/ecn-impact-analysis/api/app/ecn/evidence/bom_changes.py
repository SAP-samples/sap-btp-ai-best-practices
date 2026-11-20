"""Structured BOM changes read from dated BOM rows that carry the ECN's change code.

The ECN JSON says which objects are revised or released. The BOM rows say what
actually changed in each assembly: which part stops (and on which date), which
part starts, and in which quantity. This module turns the snapshot BOM
locations (see ``scoped_usage.enhance_snapshot``) into:

* ``lines``: one row per assembly change with the outgoing part, the incoming
  part and how they are linked (same BOM position).
* ``materials``: one row per part with "fully phased out" / "new part" flags.

Rules (deterministic, no text interpretation):

* An MBOM row is *ended by the ECN* when its "to" change code maps to the ECN,
  and *started by the ECN* when its "from" change code maps to the ECN.
  MBOM codes embed the ECN digits (``MPXA3452740A`` for ``ECN345274``).
* Enriched EBOM rows only hold the current state, so they contribute the
  engineering release date and position of the incoming part.
* A 130% EBOM material row with no 150% row at the same position (flagged
  ``raw_only`` by the snapshot) is treated as a 150% row; its lines carry the
  ``EBOM_130_SOURCE`` qualifier. 150% rows always win when both sheets have the
  position.
* Outgoing and incoming parts in the same assembly are linked only when the ECN
  pairs them, their numbers differ by one trailing letter, or they are the only
  part leaving and the only part entering that assembly. Anything else stays
  unlinked and is flagged, never guessed.
* A line's plants are the union of the rule-derived plants of the rows it was
  built from (their level-3 VC groups), so one line may belong to several plants.
"""

import re
from collections import defaultdict

from app.ecn.evidence.bom_pairing import pair_rows
from app.ecn.evidence.raw_ebom import promote_raw_only
from app.ecn.rules.plant_scope import merge_resolutions

# Locations built without rule context (older callers and focused tests).
NO_PLANT_SCOPE = {"vc_groups": [], "plants": [], "reason": "NOT_IN_BOM", "unmapped_brands": []}

DEFAULT_MBOM_PATTERN = r"^[A-Z]{4}(?P<digits>\d{6})[0-9A-Z]{2}$"
ECN_DIGITS = re.compile(r"(?P<digits>\d{6})$")

CHANGE_LABELS = {
    "REPLACED": "Replaced",
    "QUANTITY_CHANGED": "Quantity changed",
    "VALIDITY_CHANGED": "Validity changed",
    "REMOVED": "Removed",
    "ADDED": "Added",
}
PAIRING_LABELS = {
    "ECN": "Paired in the ECN",
    "NUMBER": "Part number revision",
    "ONLY_CHANGE": "Only part out and only part in for this assembly",
}


def change_code_matches(ecn_number, code, bom_kind="MBOM_130", pattern=DEFAULT_MBOM_PATTERN):
    """Return True when a BOM change code belongs to the ECN.

    Args:
        ecn_number: ECN number such as ``ECN345274``.
        code: BOM "from" or "to" change code (EBOM uses the ECN number itself,
            MBOM uses a derived code such as ``MPXA3452740A``).

    Returns:
        True for an exact match or an MBOM code embedding the ECN digits.
    """
    if not ecn_number or not code:
        return False
    if code == ecn_number:
        return True
    if bom_kind not in {"MBOM", "MBOM_130"}:
        return False
    ecn, mbom = ECN_DIGITS.search(ecn_number), re.fullmatch(pattern, code)
    return bool(ecn and mbom and ecn["digits"] == mbom["digits"])


def assembly_key(parent):
    """Return the shared assembly number for EBOM and MBOM parents.

    MBOM parents carry a model suffix (``Q923VV2526401_Q900``); EBOM parents do
    not (``Q923VV2526401``). Both refer to the same VC group.
    """
    return parent.split("_")[0] if parent else None


def _part(location, date_key):
    """Keep only the fields a reader needs for one side of a change line."""
    return {
        "number": location["number"],
        "description": location.get("description"),
        "quantity": location.get("quantity"),
        "unit": location.get("unit"),
        "date": location.get(date_key),
        "file_name": location.get("file_name"),
        "sheet": location.get("sheet"),
        "row": location.get("row"),
    }


def build_bom_changes(ecn_number, locations, lifecycle_changes=(), coverage_complete=False, pattern=DEFAULT_MBOM_PATTERN):
    """Build assembly-level change lines and part-level phase flags.

    Args:
        ecn_number: ECN number of the assessment.
        locations: Snapshot BOM location dicts (current and comparison), each
            with number, kind, parent, change_number, change_number_to,
            valid_from_date, valid_to_date, quantity, item_or_position,
            scope_id, model_range, comparison and ``plant_scope`` (rule-derived
            plants of the row, see ``app.ecn.rules.plant_scope``).
        lifecycle_changes: ``interpret_changes`` output; its counterparts
            are the ECN's own outgoing/incoming pairs.
        coverage_complete: Whether pinned BOMs declare complete coverage;
            without it "fully phased out" is reported as "no remaining use found".

    Returns:
        {"lines": [...], "materials": [...], "counts": {...}} with JSON-safe values.
        Each line carries ``plants`` and ``vc_groups`` merged from its source
        rows, plus ``plant_reason`` when no plant could be derived.
    """
    # 130%-only material rows count as 150% rows from here on; the originals
    # are not needed again, so every later lookup sees one consistent list.
    locations = promote_raw_only(locations)
    assembly_descriptions = {}
    for location in locations:
        number, description = location.get("number"), location.get("description")
        if number and description:
            assembly_descriptions.setdefault(number, description)
    ecn_pairs = {c["number"]: set(c.get("counterparts") or ()) for c in lifecycle_changes}
    phase_out_numbers = {c["number"] for c in lifecycle_changes if c.get("action") == "PHASE_OUT"}
    current = [loc for loc in locations if not loc.get("comparison") and loc.get("number")]
    groups = defaultdict(lambda: {"out": [], "in": []})
    engineering = defaultdict(list)
    engineering_ending = defaultdict(list)
    for loc in current:
        kind = loc["kind"]
        if kind not in {"MBOM_130", "ENRICHED_EBOM"}:
            continue
        # Full parent and alternative stay in the pairing key. The normalized
        # assembly is used only to find matching engineering coordinates.
        key = (loc.get("scope_id"), loc.get("parent"), kind,
               loc.get("alternative"), loc.get("sequence"),
               loc.get("item_or_position") if kind == "ENRICHED_EBOM" else None)
        if change_code_matches(ecn_number, loc.get("change_number_to"), kind, pattern):
            groups[key]["out"].append(loc)
            if kind == "ENRICHED_EBOM":
                engineering_ending[(key[0], assembly_key(key[1]), loc["number"])].append(loc)
        if change_code_matches(ecn_number, loc.get("change_number"), kind, pattern):
            groups[key]["in"].append(loc)
            if kind == "ENRICHED_EBOM":
                engineering[(key[0], assembly_key(key[1]), loc["number"])].append(loc)

    lines, consumed = [], set()

    def append_line(key, change, old, new, basis=None, qualifiers=()):
        """Append an exact change with source rows and separate EBOM/MBOM dates."""
        sample = new or old
        if change == "ADDED" and new["number"] in phase_out_numbers:
            qualifiers = (*qualifiers, "DECLARED_PHASE_OUT")
        is_mbom = sample["kind"] == "MBOM_130"
        matched = engineering.get((key[0], assembly_key(key[1]), new["number"]), []) if new and is_mbom else []
        ending = engineering_ending.get((key[0], assembly_key(key[1]), old["number"]), []) if old and is_mbom else []
        # Merge the old engineering side only at a matched incoming position.
        # Other positions retain their own independently inspectable change.
        if new:
            coordinates = {(r.get("item_or_position"), r.get("alternative"), r.get("sequence")) for r in matched}
            ending = [r for r in ending if (r.get("item_or_position"), r.get("alternative"), r.get("sequence")) in coordinates]
        consumed.update(id(row) for row in [*matched, *ending])
        # Tell the reader when any source row of this line came from the 130% sheet only.
        if any(row.get("source_kind") == "RAW_EBOM_130" for row in (old, new, *matched, *ending) if row):
            qualifiers = (*qualifiers, "EBOM_130_SOURCE")
        engineering_rows = matched or ending or ([sample] if not is_mbom else [])
        plant_scope = merge_resolutions(row.get("plant_scope") or NO_PLANT_SCOPE for row in [old, new, *matched, *ending] if row)
        assembly = assembly_key(key[1])
        lines.append({
            "scope_id": key[0], "assembly": assembly, "assembly_description": assembly_descriptions.get(assembly), "parent": key[1],
            "alternative": key[3], "sequence": key[4],
            "plants": plant_scope["plants"], "vc_groups": plant_scope["vc_groups"],
            "plant_reason": plant_scope["reason"], "unmapped_brands": plant_scope["unmapped_brands"],
            "model_range": sample.get("model_range"),
            "change": change, "change_label": "BOM start; phase-out not confirmed" if "DECLARED_PHASE_OUT" in qualifiers else CHANGE_LABELS[change],
            "outgoing": _part(old, "valid_to_date") if old else None,
            "incoming": _part(new, "valid_from_date") if new else None,
            "pairing": basis, "pairing_label": PAIRING_LABELS.get(basis),
            "position": next((row["item_or_position"] for row in engineering_rows if row.get("item_or_position")), None),
            "engineering_date": min((row["valid_from_date"] for row in engineering_rows if row.get("valid_from_date")), default=None),
            "engineering_end_date": max((row["valid_to_date"] for row in ending if row.get("valid_to_date")), default=None),
            "manufacturing_date": new.get("valid_from_date") if new and is_mbom else None,
            "manufacturing_usage_statuses": [new.get("usage_status", "UNRESOLVED")] if new and is_mbom else [],
            "end_date": old.get("valid_to_date") if old else None,
            "levels": (["EBOM"] if engineering_rows else []) + (["MBOM"] if is_mbom else []),
            "sources": [_part(row, "valid_from_date") | {"valid_from": row.get("valid_from_date"),
                        "valid_to": row.get("valid_to_date"), "kind": row.get("source_kind", row["kind"])}
                        for row in [old, new, *matched, *ending] if row],
            "qualifiers": list(qualifiers),
        })

    # Manufacturing rows consume the corresponding current EBOM release rows,
    # so one replacement is one line, with both dates shown side by side.
    ordered = sorted(groups.items(), key=lambda item: (item[0][2] != "MBOM_130", tuple(str(v) for v in item[0])))
    for key, group in ordered:
        old_rows = [row for row in group["out"] if id(row) not in consumed]
        new_rows = [row for row in group["in"] if id(row) not in consumed]
        pairs, olds, news = pair_rows(old_rows, new_rows, ecn_pairs)
        for old, new, basis in pairs:
            change = ("QUANTITY_CHANGED" if old.get("quantity") != new.get("quantity") else "VALIDITY_CHANGED") if basis == "SAME_PART" else "REPLACED"
            append_line(key, change, old, new, None if basis == "SAME_PART" else basis)
        unlinked = ["REPLACEMENT_NOT_CONFIRMED"] if olds and news else []
        for old in olds:
            append_line(key, "REMOVED", old, None, qualifiers=unlinked)
        for new in news:
            append_line(key, "ADDED", None, new, qualifiers=unlinked)
        for line in lines:
            if line["levels"] != ["EBOM"] or "MBOM_NOT_UPDATED" in line["qualifiers"]:
                continue
            # Released VC groups appear as parents in the manufacturing BOM.
            number = (line["incoming"] or {}).get("number")
            children = [row for group_key, value in groups.items()
                        if group_key[0] == line["scope_id"] and group_key[2] == "MBOM_130"
                        and assembly_key(group_key[1]) == number for row in value["in"]]
            dates = sorted(row["valid_from_date"] for row in children if row.get("valid_from_date"))
            if children:
                line["levels"] = ["EBOM", "MBOM"]
                line["manufacturing_date"] = dates[0] if dates and len(dates) == len(children) else None
                line["manufacturing_usage_statuses"] = sorted({row.get("usage_status", "UNRESOLVED") for row in children})
            else:
                line["qualifiers"].append("MBOM_NOT_UPDATED")

    materials = _material_flags(ecn_number, lines, current, locations, coverage_complete, pattern)
    counts = {change: sum(l["change"] == change for l in lines) for change in CHANGE_LABELS}
    counts |= {
        "fully_phased_out": sum(m["status"] == "FULLY_PHASED_OUT" for m in materials),
        "new_parts": sum(m["status"] == "NEW_PART" for m in materials),
    }
    return {"lines": lines, "materials": materials, "counts": counts, "coverage_complete": bool(coverage_complete)}


def _material_flags(ecn_number, lines, current, locations, coverage_complete, pattern):
    """Flag outgoing parts with no use left and incoming parts never used before.

    Args:
        ecn_number: ECN number used to ignore the ECN's own rows.
        lines: Change lines from ``build_bom_changes``.
        current: Current (non-comparison) BOM locations.
        locations: Current and comparison locations.
        coverage_complete: Declared completeness of the pinned BOMs.

    Returns:
        List of {number, role, status, status_label, date, still_used_in}.
    """
    result = []
    outgoing = {l["outgoing"]["number"]: l for l in lines if l["outgoing"] and l["change"] in {"REPLACED", "REMOVED"}}
    for number, first in sorted(outgoing.items()):
        dates = [l.get("valid_to_date") for l in current if l["number"] == number
                 and l["kind"] in {"MBOM_130", "ENRICHED_EBOM"}
                 and change_code_matches(ecn_number, l.get("change_number_to"), l["kind"], pattern)]
        cutover = max(dates) if dates and all(d and d != "9999-12-31" for d in dates) else None
        # Any other row of this part still valid after the cut-over keeps it alive.
        remaining = [l for l in current if l["number"] == number and l["kind"] != "RAW_EBOM_130"
                     and not change_code_matches(ecn_number, l.get("change_number_to"), l["kind"], pattern)
                     and (l.get("valid_to_date") or "9999-12-31") > (cutover or "")]
        if not cutover or cutover == "9999-12-31":
            status = "TIMING_UNKNOWN"
        elif remaining:
            status = "STILL_USED"
        else:
            status = "FULLY_PHASED_OUT" if coverage_complete else "NO_REMAINING_USE_FOUND"
        result.append({
            "number": number, "description": first["outgoing"]["description"], "role": "OUTGOING",
            "status": status, "date": cutover or None,
            "still_used_in": sorted({assembly_key(l.get("parent")) or "" for l in remaining}),
            "prior_manufacturing_bom_use": any(l["kind"] == "MBOM_130" for l in remaining),
        })
    incoming = {l["incoming"]["number"]: l for l in lines if l["incoming"] and l["change"] in {"REPLACED", "ADDED"}}
    for number, first in sorted(incoming.items()):
        conflict = any("DECLARED_PHASE_OUT" in l["qualifiers"] for l in lines
                       if l["incoming"] and l["incoming"]["number"] == number)
        dates = [l.get("valid_from_date") for l in current if l["number"] == number
                 and l["kind"] in {"MBOM_130", "ENRICHED_EBOM"}
                 and change_code_matches(ecn_number, l.get("change_number"), l["kind"], pattern)]
        start = min(dates) if dates and all(d and d != "9999-12-31" for d in dates) else None
        earlier = [l for l in locations if l.get("number") == number and l["kind"] != "RAW_EBOM_130"
                   and not change_code_matches(ecn_number, l.get("change_number"), l["kind"], pattern)
                   and (l.get("comparison") or (l.get("valid_from_date") or "") < (start or ""))]
        result.append({
            "number": number, "description": first["incoming"]["description"], "role": "OUTGOING" if conflict else "INCOMING",
            "status": "PHASE_OUT_NOT_CONFIRMED" if conflict else "TIMING_UNKNOWN" if not start else "ALREADY_USED" if earlier else "NEW_PART", "date": None if conflict else start,
            "still_used_in": sorted({assembly_key(l.get("parent")) or "" for l in earlier}),
            "prior_manufacturing_bom_use": any(l["kind"] == "MBOM_130" for l in earlier),
        })
    labels = {"PHASE_OUT_NOT_CONFIRMED": "Phase-out not confirmed", "TIMING_UNKNOWN": "Timing needs confirmation", "FULLY_PHASED_OUT": "Fully phased out", "NO_REMAINING_USE_FOUND": "No remaining use found (BOM coverage not declared complete)",
              "STILL_USED": "Still used elsewhere", "NEW_PART": "New part", "ALREADY_USED": "Existing part, already used elsewhere"}
    for item in result:
        item["status_label"] = labels[item["status"]]
    return result
