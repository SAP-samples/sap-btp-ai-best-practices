"""What accepting an ECN changes, one line per part, and whether it can be accepted.

The Engineering review card, the chat assistant and the review save path all
use this plan. It is built only from the assessment snapshot:

* ``bom_changes.materials`` lists the parts the ECN brings in (INCOMING) and
  takes out (OUTGOING), with their date and status.
* ``plant_impact`` gives each part's rule-derived plants, whether a new part is
  already set up in a plant, and ``priced_new_parts``: the new parts that
  appear in the purchasing info record dataset in any plant.

Each part gets one plain-language material master action ("Release in PL10
from 2026-03-13") and a price state:

* ``HAS_PRICE``: a new part found in the purchasing info record dataset. The
  data has no prices, so appearing there is taken as "we have its price".
* ``MISSING``: a new part not found there. It blocks the whole ECN.
* ``NOT_NEEDED``: an old part; phasing out needs no price.

Nothing here writes to SAP. See docs/ecc-material-master-posting.md for how
the real write would work.
"""

def _when(date):
    """Return 'from 2026-03-13', or 'date not known' for a missing or open-ended date."""
    return f"from {date}" if date and date != "9999-12-31" else "date not known"


def _incoming_action(material, plants):
    """Describe how a new part is released, e.g. 'Create and release in PL10 from 2026-03-13'.

    Args:
        material: bom_changes material with number and date.
        plants: plant_impact rows (with a plant_code) the part belongs to.

    Returns:
        One line of text. A part not yet set up in a plant is created there first.
    """
    number = material["number"]
    set_up = [p["plant_code"] for p in plants
              if number in p.get("new_parts_with_purchasing", []) + p.get("new_parts_without_purchasing", [])]
    created = [p["plant_code"] for p in plants if p["plant_code"] not in set_up]
    phrases = ([f"Create and release in {', '.join(created)}"] if created else []) + \
              ([f"Release in {', '.join(set_up)}"] if set_up else [])
    return f"{'; '.join(phrases) or 'Release (plant not known)'} {_when(material.get('date'))}"


def _outgoing_action(material, plants):
    """Describe what happens to an old part, e.g. 'Phase out in PL10 from 2026-05-19'.

    Args:
        material: bom_changes material with number, status, date and still_used_in.
        plants: plant_impact rows (with a plant_code) the part belongs to.

    Returns:
        One line of text. A part still used elsewhere is left unchanged.
    """
    if material["status"] == "PHASE_OUT_NOT_CONFIRMED":
        return "No change: phase-out not confirmed"
    if material["status"] == "STILL_USED":
        used = material.get("still_used_in") or []
        more = f" +{len(used) - 2} more" if len(used) > 2 else ""
        return f"No change: still used in {', '.join(used[:2])}{more}"
    where = "all plants" if material["status"] == "FULLY_PHASED_OUT" else \
        ", ".join(p["plant_code"] for p in plants) or "plant not known"
    return f"Phase out in {where} {_when(material.get('date'))}"


def build_acceptance_plan(snapshot):
    """Return the per-part acceptance plan for one assessment snapshot.

    Args:
        snapshot: Snapshot overview dict. Version 1 snapshots have no
            ``bom_changes``; their plan has no parts and ``available`` False.

    Returns:
        {assessment_id, evidence_hash, available, parts, missing_price,
        blocked, block_reason, bom_item_count}.
        - ``parts``: [{number, description, role ("NEW"/"OLD"), master_data, price}].
        - ``missing_price``: new part numbers with no price; any one blocks.
        - ``block_reason``: one line for the card, the assistant and the API
          error, or None when the ECN can be accepted.
        - ``bom_item_count``: BOM change lines; shown as information only,
          because BOM items are not material master data.
    """
    bom_changes = snapshot.get("bom_changes") or {}
    plant_impact = snapshot.get("plant_impact") or {}
    plants_of = {}
    for plant in plant_impact.get("plants", []):
        if plant.get("plant_code"):
            for member in plant.get("materials", []):
                plants_of.setdefault(member["number"], []).append(plant)
    priced = set(plant_impact.get("priced_new_parts", []))
    parts = []
    for material in bom_changes.get("materials", []):
        plants = plants_of.get(material["number"], [])
        if material["role"] == "INCOMING":
            role, action = "NEW", _incoming_action(material, plants)
            price = "HAS_PRICE" if material["number"] in priced else "MISSING"
        else:
            role, action, price = "OLD", _outgoing_action(material, plants), "NOT_NEEDED"
        parts.append({"number": material["number"], "description": material.get("description"),
                      "role": role, "master_data": action, "price": price})
    # New parts first, so the parts that can block the ECN are seen first.
    parts.sort(key=lambda part: (part["role"] != "NEW", part["number"]))
    missing = [part["number"] for part in parts if part["price"] == "MISSING"]
    count = len(missing)
    return {
        "assessment_id": snapshot.get("assessment_id"), "evidence_hash": snapshot.get("evidence_hash"),
        "available": bool(bom_changes), "parts": parts, "missing_price": missing, "blocked": bool(missing),
        "block_reason": (f"Can't accept: {count} new {'part has' if count == 1 else 'parts have'} no price."
                         if missing else None),
        "bom_item_count": len(bom_changes.get("lines", [])),
    }
