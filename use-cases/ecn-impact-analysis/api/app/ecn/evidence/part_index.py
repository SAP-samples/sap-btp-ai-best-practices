"""Compact per-ECN part index used to match materials across ECNs.

A full snapshot is large and only four are cached at a time, so cross-ECN
questions ("which ECNs touch QKX7702260?", "which ECNs share parts with this
one?") read this small summary instead: for every part number of one ECN, its
role, BOM change types, assemblies, plants and dates.

Example::

    index = build_part_index(snapshot)
    index["parts"]["QKX7702260"]["roles"]   # ["ECN_OBJECT", "INCOMING"]
"""


def _entry(parts: dict, number: str, description: str | None) -> dict:
    """Return (creating on first use) the mutable index entry of one part number."""
    entry = parts.setdefault(number, {
        "number": number, "description": description, "roles": set(), "changes": set(),
        "assemblies": set(), "plants": set(), "engineering_dates": set(),
        "manufacturing_dates": set(), "end_dates": set(),
    })
    entry["description"] = entry["description"] or description
    return entry


def build_part_index(snapshot: dict) -> dict:
    """Summarize which parts one ECN touches and how.

    Args:
        snapshot: Output of ``SnapshotService.get`` (``objects`` and, for
            assessed ECNs, ``bom_changes.lines``).

    Returns:
        ``{"ecn_number", "parts": {number: entry}}``. Each entry has
        ``description`` and sorted lists: ``roles`` (``ECN_OBJECT`` when the
        ECN JSON names the part, ``OUTGOING`` / ``INCOMING`` for its side of a
        BOM change line), ``changes`` (REPLACED, ADDED, REMOVED, ...),
        ``assemblies``, ``plants`` (plant codes), ``engineering_dates``,
        ``manufacturing_dates`` and ``end_dates`` (ISO dates).
    """
    parts: dict[str, dict] = {}
    for obj in snapshot.get("objects", []):
        if obj.get("technical") and obj.get("number") and "ecn" in obj.get("origins", ()):
            _entry(parts, obj["number"], obj.get("description"))["roles"].add("ECN_OBJECT")
    for line in (snapshot.get("bom_changes") or {}).get("lines", []):
        plants = {plant["plant_code"] for plant in line.get("plants") or ()}
        for role, side in (("OUTGOING", line.get("outgoing")), ("INCOMING", line.get("incoming"))):
            if not side or not side.get("number"):
                continue
            entry = _entry(parts, side["number"], side.get("description"))
            entry["roles"].add(role)
            entry["changes"].add(line["change"])
            entry["plants"] |= plants
            if line.get("assembly"):
                entry["assemblies"].add(line["assembly"])
            if role == "INCOMING":
                entry["engineering_dates"].add(line.get("engineering_date"))
                entry["manufacturing_dates"].add(line.get("manufacturing_date"))
            else:
                entry["end_dates"].add(line.get("end_date"))
    return {
        "ecn_number": snapshot.get("ecn_number"),
        "parts": {
            number: {key: sorted(filter(None, value)) if isinstance(value, set) else value
                     for key, value in entry.items()}
            for number, entry in parts.items()
        },
    }
