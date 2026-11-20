"""Deterministic lifecycle and replacement facts shared by every assessment view."""

from collections import defaultdict
import re

from app.ecn.intake.normalization import _walk_resulting

_STATES = {"TOBEOBSOLETE": "PHASE_OUT", "OBSOLETE": "PHASE_OUT", "APPROVEDFORSERIES": "RELEASE"}
_TRANSITIONS = {"OBSOLESCENCE": "PHASE_OUT", "PRODUCTION_RELEASED": "RELEASE"}
PAIR_LABELS = {"SOURCE_RELATION": "Explicit source relationship", "EBOM_POSITION": "EBOM position match", "NUMBER_PATTERN": "Number-pattern match"}


def change_summary(change):
    """Keep business facts and bounded coordinates in model tools; full evidence stays inspectable."""
    result = {k: v for k, v in change.items() if k not in {"pairing_evidence", "lifecycle_evidence", "supporting_objects"}}
    result["supporting_object_count"] = len(change.get("supporting_objects", []))
    result["pairing_reference_count"] = len(change.get("pairing_evidence", []))
    result["pairing_references"] = [
        {side: {key: row.get(key) for key in ("file_name", "sheet", "row", "scope_id", "parent", "item_or_position", "comparison")}
         for side, row in evidence.items() if side in {"before", "after"}}
        for evidence in change.get("pairing_evidence", [])[:3]
    ]
    return result


def _value(value):
    """Read a source enum without coercing malformed values into business facts."""
    return value.get("Value") if isinstance(value, dict) else value if isinstance(value, str) else None


def _one(values):
    """Return a unique nonblank value; conflicting values remain unavailable."""
    unique = {value for value in values if value}
    return next(iter(unique)) if len(unique) == 1 else None


def _task(path):
    """Identify the source implementation task without comparing link ordinals."""
    match = re.search(r"ImplementationPlan\[\d+\]", path)
    return match.group() if match else None


def _number_pattern(old, new):
    """Match a full stem with a single changed/appended letter, never a broad prefix."""
    return bool(old and new and old != new and (
        (len(old) == len(new) and old[:-1] == new[:-1] and new[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        or (len(new) == len(old) + 1 and new[:-1] == old and new[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ))


def _position(row):
    """Return an exact scoped EBOM position, normalizing only numeric zero padding."""
    value = row.get("item_or_position")
    if row.get("kind") not in {"ENRICHED_EBOM", "RAW_EBOM_130"} or value is None or not row.get("parent") or not row.get("scope_id") or "Unknown" in row["scope_id"]:
        return None
    value = str(value).strip()
    if not value:
        return None
    position = str(int(value)) if value.isdigit() else value
    return (row.get("scope_id"), row["parent"], row.get("alternative"), position)


def _before_after(left, right):
    """Require historical-to-current evidence or explicit nonoverlapping validity."""
    if left.get("comparison") and not right.get("comparison"):
        return True
    return bool(not right.get("comparison") and left.get("valid_to_date") and right.get("valid_from_date")
                and left["valid_to_date"] < right["valid_from_date"])


def _entry(number, affected, resulting, links, supporting):
    """Classify one WTPart number, preserving every source state and revision."""
    before = _one(o.revision for o in affected)
    after = _one(o.revision for o in resulting)
    facts, signals, qualifiers = [], set(), []
    for obj in resulting:
        link = links.get(obj.source_path, {})
        target = _value(link.get("TargetTransition"))
        state_signal, target_signal = _STATES.get(obj.state), _TRANSITIONS.get(target)
        if state_signal and target_signal and state_signal != target_signal:
            qualifiers.append("LIFECYCLE_CONFLICT")
        if obj.state and state_signal is None or target and target_signal is None:
            qualifiers.append("LIFECYCLE_UNRECOGNIZED")
        signals.update(s for s in (state_signal, target_signal) if s)
        facts.append({"source_path": obj.source_path, "link_path": obj.link_path, "state": obj.state,
                      "target_transition": target, "revision": obj.revision,
                      "replaces": link.get("OemReplacesAffected")})
    if len(signals) > 1:
        qualifiers.append("LIFECYCLE_CONFLICT")
    for collection in (affected, resulting):
        if len({o.revision for o in collection if o.revision}) > 1:
            qualifiers.append("REVISION_CONFLICT")
    if qualifiers:
        action = "REVIEW"
    elif signals == {"PHASE_OUT"}:
        action = "PHASE_OUT"
    elif signals == {"RELEASE"}:
        if affected and (not before or not after):
            action = "RELEASE"
            qualifiers.append("REVISION_COMPARISON_UNAVAILABLE")
        else:
            action = "REVISION" if before and after and before != after else "UNCHANGED" if affected else "PHASE_IN"
    else:
        action = "UNRESOLVED"
    return {"number": number, "action": action, "revision_from": before, "revision_to": after,
            "counterparts": [], "pairing_basis": None, "pairing_label": None, "pairing_evidence": [],
            "qualifiers": sorted(set(qualifiers)), "lifecycle_evidence": facts,
            "source_paths": sorted({o.source_path for o in (*affected, *resulting)}),
            "supporting_objects": [{"number": o.number, "type": o.object_type, "revision": o.revision,
                                    "state": o.state, "source_path": o.source_path} for o in supporting]}


def interpret_changes(ecn, locations=()):
    """Return deterministic per-WTPart actions and unique, evidence-labelled pairs.

    `locations` contains current and comparison EBOM observations with exact scope,
    parent and position. MBOM observations never establish replacement genealogy.
    Source text is inspected as data only; no LLM or fuzzy join participates.
    """
    links = {path: link for path, _, link, _ in _walk_resulting(ecn.source_json)}
    affected, resulting = defaultdict(list), defaultdict(list)
    documents = []
    for obj in (*ecn.affected_objects, *ecn.resulting_objects):
        if obj.is_technical_wtpart and obj.number:
            (affected if obj.role.value == "AFFECTED" else resulting)[obj.number].append(obj)
        elif obj.role.value == "RESULTING":
            documents.append(obj)
    entries = {}
    for number in sorted(affected.keys() | resulting.keys()):
        supporting = [o for o in documents if o.number in {number, number + ".PRT", number + ".DRW", number + ".ASM"}]
        entries[number] = _entry(number, affected[number], resulting[number], links, supporting)
    indexed = defaultdict(list)
    for row in locations:
        indexed[row.get("number")].append(row)
    outgoing = [e for e in entries.values() if e["action"] == "PHASE_OUT"]
    incoming = [e for e in entries.values() if e["action"] == "PHASE_IN"]
    candidates = []
    for old in outgoing:
        for new in incoming:
            old_number, new_number = old["number"], new["number"]
            # A direct relation must name a real endpoint, not merely contain text.
            identifiers = {old_number, *(o.technical_id for o in affected[old_number]), *(o.technical_id for o in resulting[old_number])}
            explicit = [f for f in new["lifecycle_evidence"] if isinstance(f["replaces"], str) and f["replaces"].strip() in identifiers]
            positions = [(left, right) for left in indexed[old_number] for right in indexed[new_number]
                         if _position(left) is not None and _position(left) == _position(right) and _before_after(left, right)]
            tasks_old = {_task(o.source_path) for o in resulting[old_number]} - {None}
            tasks_new = {_task(o.source_path) for o in resulting[new_number]} - {None}
            if explicit:
                candidates.append((0, old_number, new_number, "SOURCE_RELATION", explicit))
            elif positions:
                candidates.append((1, old_number, new_number, "EBOM_POSITION", [
                    {"before": left, "after": right} for left, right in positions]))
            elif not any(left.get("comparison") and _position(left) for left in indexed[old_number]) and tasks_old & tasks_new and _number_pattern(old_number, new_number):
                candidates.append((2, old_number, new_number, "NUMBER_PATTERN", []))
    # Resolve strongest evidence first. Equal alternatives stay visible for review.
    used = set()
    for priority in (0, 1, 2):
        tier = [c for c in candidates if c[0] == priority and c[1] not in used and c[2] not in used]
        degree = defaultdict(int)
        for _, old, new, _, _ in tier:
            degree[old] += 1
            degree[new] += 1
        for _, old, new, basis, evidence in tier:
            if degree[old] != 1 or degree[new] != 1:
                for number in (old, new):
                    if "AMBIGUOUS_PAIRING" not in entries[number]["qualifiers"]:
                        entries[number]["qualifiers"].append("AMBIGUOUS_PAIRING")
                used.update((old, new))
                continue
            for number, other in ((old, new), (new, old)):
                entries[number].update(counterparts=[other], pairing_basis=basis,
                                       pairing_label=PAIR_LABELS[basis], pairing_evidence=evidence)
            used.update((old, new))
    return list(entries.values())
