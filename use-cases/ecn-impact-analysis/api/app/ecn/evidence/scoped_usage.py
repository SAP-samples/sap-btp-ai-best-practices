"""Dated BOM usage and material changes within declared source coverage."""

def usage_status(location, assessment_date):
    """Classify one dated source occurrence without claiming configuration applicability."""
    if location.get("comparison"):
        return "HISTORICAL"
    start, end = location.get("valid_from_date"), location.get("valid_to_date")
    if not start or not end or location.get("migration_date") or start > end:
        return "UNRESOLVED"
    day = assessment_date.isoformat()
    return "FUTURE" if start > day else "EXPIRED" if end < day else "EFFECTIVE"



def _scope(dataset):
    """Identify exact dataset scope; unknown source dimensions are not guessed.

    Plants are not part of a BOM dataset's scope: they are derived per row from
    conditional rules (see ``app.ecn.rules.plant_scope``).
    """
    return "/".join(str(dataset.get(key) or "Unknown") for key in ("data_group", "environment", "model_range"))


def enhance_snapshot(snapshot, ecn, rows, comparison_rows, datasets, assessment_date, plant_context=None):
    """Attach v2 lifecycle, historical evidence and coverage-bounded usage in place.

    Inputs are immutable HANA-pinned rows and metadata. Outputs are JSON-safe
    facts used identically by the workspace, assistant and synthesis projection.

    Args:
        plant_context: ``{"index": VcRuleIndex, "mappings": [...], "rules_pinned":
            bool}`` built from the assessment's pinned rule workbooks and the
            Plant Code Mapping frozen at import. ``None`` resolves no plant.
            Every location receives ``plant_scope`` (``resolve_row`` output).
    """
    from app.ecn.evidence.snapshot import bom_location
    from app.ecn.evidence.material_changes import interpret_changes
    from app.ecn.evidence.raw_ebom import raw_only_row_ids
    from app.ecn.rules.plant_scope import EMPTY_INDEX, resolve_row

    from app.ecn.settings import load_ecn_settings
    config = load_ecn_settings()
    plant_context = plant_context or {}
    plant_index = plant_context.get("index") or EMPTY_INDEX
    plant_mappings = plant_context.get("mappings") or []
    rules_pinned = bool(plant_context.get("rules_pinned"))
    snapshot["limitations"] = [
        "Findings are limited to pinned datasets and their declared coverage.",
        "Effective BOM validity does not prove configured production use.",
        "Missing master, purchasing or stock records are unavailable evidence, not zero stock.",
    ]
    metadata = {d["artifact_id"]: d for d in datasets if d["dataset_type"] == "BOM"}
    all_rows = (*rows, *comparison_rows)
    by_id = {row.bom_row_id: row for row in all_rows}
    # 130% EBOM material rows that have no 150% twin count as engineering rows.
    raw_only = raw_only_row_ids(all_rows)
    locations = []
    for row in all_rows:
        dataset = metadata.get(row.artifact_id, {})
        location = bom_location(row, by_id)
        location.update(number=row.component_number, comparison=bool(dataset.get("comparison")),
                        scope_id=_scope(dataset), bom_scope=_scope(dataset), model_range=dataset.get("model_range"),
                        plant_scope=resolve_row(row, plant_index, plant_mappings, rules_pinned), file_name=dataset.get("file_name"),
                        snapshot_date=dataset.get("snapshot_date"), environment=dataset.get("environment", "UNKNOWN"))
        location["usage_status"] = usage_status(location, assessment_date)
        location["raw_only"] = row.bom_row_id in raw_only
        locations.append(location)
    changes = interpret_changes(ecn, locations)
    indexed = {change["number"]: change for change in changes}
    current_scopes = [d for d in metadata.values() if not d.get("comparison")]
    coverage_complete = bool(current_scopes) and all(d.get("coverage_complete") and d.get("snapshot_date")
        and d.get("model_range") for d in current_scopes)
    environments = {d.get("environment", "UNKNOWN") for d in current_scopes}
    coverage_complete = coverage_complete and len(environments) == 1 and "UNKNOWN" not in environments
    for obj in snapshot["objects"]:
        if not obj["technical"]:
            continue
        # Raw EBOM rows are shown only when the 150% EBOM has no row at the same
        # position; otherwise they would count one occurrence twice.
        obj["locations"] = [l for l in locations if l["number"] == obj["number"]
                            and (l["kind"] != "RAW_EBOM_130" or l["raw_only"])]
        change = indexed.get(obj["number"])
        obj["change"] = change
        if change is None:
            continue
        observed = [l for l in obj["locations"] if not l["comparison"]]
        effective = [l for l in observed if l["usage_status"] == "EFFECTIVE"]
        unresolved = [l for l in observed if l["usage_status"] == "UNRESOLVED"]
        counterpart_scopes = {l["scope_id"] for l in locations if l["number"] in change["counterparts"] and l.get("change_number") == ecn.ecn_number
                              and l["kind"] in {"ENRICHED_EBOM", "RAW_EBOM_130"} and not l["comparison"]}
        if change["action"] == "PHASE_OUT":
            if effective:
                status = "USED_ELSEWHERE" if counterpart_scopes and any(l["scope_id"] not in counterpart_scopes for l in effective) else "STILL_USED"
            elif unresolved or not coverage_complete:
                status = "UNRESOLVED"
            elif any(l["usage_status"] == "FUTURE" for l in observed):
                status = "FUTURE_USE"
            else:
                status = "NO_REMAINING_USE_IN_SCOPE"
        else:
            status = "OBSERVED_USE" if effective else "UNRESOLVED" if unresolved else "NO_EFFECTIVE_USE_OBSERVED"
        labels = {"USED_ELSEWHERE": "Still used in another model range", "STILL_USED": "Still used in supplied BOMs",
                  "UNRESOLVED": "Usage or coverage needs review", "FUTURE_USE": "Future usage remains",
                  "NO_REMAINING_USE_IN_SCOPE": "No remaining usage within declared coverage",
                  "OBSERVED_USE": "Effective usage observed in supplied BOMs", "NO_EFFECTIVE_USE_OBSERVED": "No effective usage observed"}
        change["usage"] = {"status": status, "label": labels[status], "effective_occurrences": len(effective),
                           "unresolved_occurrences": len(unresolved), "coverage_complete": bool(coverage_complete)}
        change["usage_by_scope"] = [{"scope_id": scope, "effective_occurrences": sum(l["usage_status"] == "EFFECTIVE" for l in observed if l["scope_id"] == scope),
                                    "statuses": sorted({l["usage_status"] for l in observed if l["scope_id"] == scope})}
                                   for scope in sorted({l["scope_id"] for l in observed})]
        change["vc_groups"] = sorted({row.vc_value for row in rows if row.component_number in {obj["number"], *change["counterparts"]} and (row.change_number == ecn.ecn_number or row.component_number == row.vc_value) and row.vc_value})
        change["master_data"] = [{"plant_code": r.get("plant_code"), "plant_status": r.get("plant_status"),
                                  "cross_plant_status": r.get("cross_plant_status"), "comparison": r.get("comparison", False)}
                                 for r in obj["reference_records"] if r.get("dataset_type") == "MATERIAL_MASTER"]
    from app.ecn.evidence.bom_changes import build_bom_changes
    snapshot["bom_changes"] = build_bom_changes(ecn.ecn_number, locations, changes, coverage_complete, config.mbom_change_number_pattern)
    snapshot.update(version="ecn-snapshot-v2", assessment_date=assessment_date.isoformat(), changes=changes,
                    bom_scope="; ".join(sorted({_scope(d) for d in current_scopes})) or "BOM scope not declared",
                    coverage_complete=bool(coverage_complete))
