"""Object-first snapshot facts shared by the workspace, synthesis and read-only tools."""

from collections import Counter, defaultdict
from hashlib import sha256

from app.ecn.models import (
    PROCUREMENT_TYPE_LABELS,
    BomKind,
    BomRowClassification,
    NormalizedEcn,
)

SNAPSHOT_VERSION = "ecn-snapshot-v1"
LEVEL_LABELS = {
    "1": "Main",
    "2": "Sub",
    "3": "VC",
    "4": "Level 4",
    "5": "Level 5",
    "6": "Level 6",
}

BOM_SCOPE = "Q900V only"
"""Single stable BOM scope marker stamped on every snapshot and BOM location.

This value is identical in the snapshot overview, every BOM location, and all
context surfaces (summary, object detail, assistant tools) so they cannot diverge.
"""


def procurement_type_label(value: object) -> str:
    """Map a material-master ProcType code to a human-readable label without inference.

    Labels come from the canonical ``PROCUREMENT_TYPE_LABELS`` map in
    ``app.ecn.models`` (shared with the Settings reference-data legend), so the
    two surfaces cannot diverge on the maintained E/F/X wording. Returns the exact
    label for the three known SAP codes (E, F, X). Any other value — including
    None, empty string, or an unrecognised code — returns "Unknown". The raw code
    is always preserved in the source data; this function only adds the display
    label.

    Args:
        value: A raw ProcType cell value from the material master. May be None,
            empty string, or any other object; only string codes are matched.

    Returns:
        One of "In-house production", "External procurement",
        "In-house and external procurement allowed", or "Unknown".
    """
    if not value:
        return "Unknown"
    return PROCUREMENT_TYPE_LABELS.get(str(value), "Unknown")


def _source_display(value: object) -> str | None:
    """Read a source value or its Windchill display text without inventing one."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("Display", "Value", "Name"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip()
    return None


def ecn_context(source_json: dict) -> dict:
    """Expose concise owner/category/timing facts already carried by the ECN source."""
    return {
        "owner_name": _source_display(
            source_json.get("REQUESTER_NAME")
            or source_json.get("ChangeAdminI")
            or source_json.get("CreatedBy")
        ),
        "category_label": _source_display(source_json.get("CATEGORY2")),
        "business_driver": None,
        "cib_target_date": _source_display(source_json.get("ResolutionDate")),
        "financial_impact": "UNKNOWN",
        "financial_notes": None,
        "risk_level": None,
        "source": "SOURCE",
    }


def bom_location(row, by_id):
    """Describe one observed occurrence and its actual ancestor chain with source coordinates."""
    chain, seen, cursor = [], set(), row
    while cursor and cursor.bom_row_id not in seen:
        seen.add(cursor.bom_row_id)
        chain.append(
            {
                "number": cursor.component_number,
                "level": cursor.dotted_level,
                "label": LEVEL_LABELS.get(cursor.dotted_level, "Object"),
                "description": cursor.description_text,
                "row": cursor.source_row_number,
            }
        )
        parent = by_id.get(cursor.parent_bom_row_id)
        if parent is None and cursor.parent_material_number:
            chain.append(
                {
                    "number": cursor.parent_material_number,
                    "level": None,
                    "label": "Product context"
                    if cursor.dotted_level == "1"
                    else "Parent context",
                    "description": None,
                    "row": None,
                }
            )
        cursor = parent
    return {
        "occurrence_id": row.bom_row_id,
        "kind": row.bom_kind.value,
        "sheet": row.sheet_name,
        "row": row.source_row_number,
        "level": row.dotted_level if row.bom_kind == BomKind.ENRICHED_EBOM else None,
        "parent": row.parent_material_number,
        "path": list(reversed(chain)),
        "description": row.description_text,
        "quantity": str(row.quantity) if row.quantity is not None else None,
        "unit": row.unit,
        "change_number": row.change_number,
        "change_number_to": row.change_number_to,
        "valid_from": row.valid_from_raw,
        "valid_to": row.valid_to_raw,
        "valid_from_quality": "LEGACY_PLACEHOLDER" if row.is_migration_date else "VALID" if row.valid_from_date else "MISSING" if not row.valid_from_raw else "INVALID",
        "valid_to_quality": "VALID" if row.valid_to_date else "MISSING" if not row.valid_to_raw else "INVALID",
        "valid_from_date": row.valid_from_date.isoformat() if row.valid_from_date else None,
        "valid_to_date": row.valid_to_date.isoformat() if row.valid_to_date else None,
        "migration_date": row.is_migration_date,
        "item_or_position": row.position_value or row.item_value,
        "alternative": row.alternative,
        "artifact_id": row.artifact_id,
        "qualifiers": list(row.quality_qualifiers),
        # BOM scope marker: always Q900V only; stamped here so every consumer
        # reads it from the location dict without re-deriving it.
        "bom_scope": BOM_SCOPE,
    }


def build_snapshot(
    ecn: NormalizedEcn, rows, references=(), *, evidence_hash="", datasets=(), assessment_date=None, comparison_rows=(), source_warnings=(),
    plant_context=None,
):
    """Return complete facts for one ECN and pinned BOM/reference snapshots; never infer changes.

    ``plant_context`` (rule index, frozen Plant Code Mapping, rules-pinned flag)
    lets ``enhance_snapshot`` derive each BOM row's plants from the rules.
    """
    by_id = {r.bom_row_id: r for r in rows}
    by_number = defaultdict(list)
    for row in rows:
        if row.bom_kind in {BomKind.ENRICHED_EBOM, BomKind.MBOM_130}:
            by_number[row.component_number].append(row)
    # Enrich material-master reference rows with the ProcType display label so
    # every consumer sees both the raw code and the human-readable label without
    # re-deriving it.  Other dataset types are passed through unchanged.
    reference_index = defaultdict(list)
    for ref in references:
        if ref.get("dataset_type") == "MATERIAL_MASTER":
            ref = {**ref, "procurement_type_label": procurement_type_label(ref.get("procurement_type"))}
        reference_index[ref.get("material_number")].append(ref)
    grouped = {}
    for obj in (*ecn.affected_objects, *ecn.resulting_objects):
        # A CAD object and a technical WTPart sharing a number are different objects.
        key = (
            obj.number or obj.technical_id or obj.source_path,
            obj.is_technical_wtpart,
        )
        entry = grouped.setdefault(
            key,
            {
                "number": obj.number,
                "technical": obj.is_technical_wtpart,
                "types": set(),
                "roles": set(),
                "source_references": [],
                "description": obj.description,
                "locations": [],
                "reference_records": [],
                "origins": {"ecn"},
            },
        )
        entry["types"].add(obj.object_type or "Unspecified")
        entry["roles"].add(obj.role.value)
        entry["source_references"].append(
            {
                "path": f"$[{ecn.source_ordinal}]" + obj.source_path[1:],
                "role": obj.role.value,
                "revision": obj.revision,
                "version": obj.version,
                "state": obj.state,
                "technical_id": obj.technical_id,
            }
        )
    from app.ecn.evidence.bom_changes import change_code_matches
    from app.ecn.settings import load_ecn_settings
    config = load_ecn_settings()
    for row in rows:
        if (
            ecn.ecn_number
            and (change_code_matches(ecn.ecn_number, row.change_number, row.bom_kind.value, config.mbom_change_number_pattern)
                 or change_code_matches(ecn.ecn_number, row.change_number_to, row.bom_kind.value, config.mbom_change_number_pattern))
            and (row.bom_kind != BomKind.RAW_EBOM_130 or row.row_classification == BomRowClassification.MATERIAL_LIKE_OBSERVED)
            and row.component_number
        ):
            key = (row.component_number, True)
            entry = grouped.setdefault(
                key,
                {
                    "number": row.component_number,
                    "technical": True,
                    "types": {"BOM component (source type unavailable)"},
                    "roles": set(),
                    "source_references": [],
                    "description": row.description_text,
                    "locations": [],
                    "reference_records": [],
                    "origins": set(),
                },
            )
            entry["origins"].add("bom")
            entry["roles"].add("BOM_ECN_REFERENCE")
            entry["source_references"].append(
                {
                    "path": f"{row.sheet_name}, row {row.source_row_number}",
                    "role": "BOM_ECN_REFERENCE",
                    "revision": None,
                    "version": None,
                    "state": None,
                    "change_number": row.change_number,
                    "artifact_id": row.artifact_id,
                }
            )
    for key, entry in grouped.items():
        entry["object_key"] = "object-" + sha256(repr(key).encode()).hexdigest()[:32]
        entry["origins"] = sorted(entry["origins"])
        entry["types"] = sorted(entry["types"])
        entry["roles"] = sorted(entry["roles"])
        if entry["technical"]:
            entry["locations"] = [
                bom_location(r, by_id) for r in by_number.get(entry["number"], ())
            ]
            entry["reference_records"] = reference_index.get(entry["number"], [])
        descriptions = [r.get("description") for r in entry["reference_records"]] + [
            r["description"] for r in entry["locations"]
        ]
        entry["description"] = entry["description"] or next(
            (d for d in descriptions if d), None
        )
        entry["observed_levels"] = sorted(
            {r["level"] for r in entry["locations"] if r["level"]}
        )
        entry["coverage"] = (
            "Located in supplied BOM"
            if entry["locations"]
            else (
                "Location unavailable in supplied BOM"
                if entry["technical"]
                else "Supporting document or non-part object"
            )
        )
    objects = sorted(
        grouped.values(),
        key=lambda o: (not o["technical"], o["number"] or "", str(o["types"])),
    )
    technical = [o for o in objects if o["technical"] and "ecn" in o["origins"]]
    located = [o for o in technical if o["locations"]]
    unmatched = len(technical) - len(located)
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "assessment_id": ecn.ecn_row_id,
        "evidence_hash": evidence_hash,
        "ecn_number": ecn.ecn_number,
        "source_position": ecn.source_ordinal + 1,
        "intent": ecn.description_text,
        "source_state": ecn.source_json.get("State"),
        "declared_products": ecn.source_json.get("AFFECTED_PRODUCTS"),
        "ecn_context": ecn_context(ecn.source_json),
        # Single stable BOM scope marker consumed by overview, object detail,
        # summary context, and assistant tools.  All four surfaces read from
        # this field so they cannot diverge.
        "bom_scope": BOM_SCOPE,
        "datasets": list(datasets),
        "counts": {
            "technical_objects": len(technical),
            "supporting_objects": sum(not o["technical"] for o in objects),
            "bom_change_objects": sum("bom" in o["origins"] for o in objects),
            "source_references": sum(
                r["role"] != "BOM_ECN_REFERENCE"
                for o in objects
                for r in o["source_references"]
            ),
            "located_objects": len(located),
            "unlocated_objects": unmatched,
            "enriched_occurrences": len(
                {
                    r["occurrence_id"]
                    for o in technical
                    for r in o["locations"]
                    if r["kind"] == BomKind.ENRICHED_EBOM.value
                }
            ),
            "mbom_occurrences": len(
                {
                    r["occurrence_id"]
                    for o in technical
                    for r in o["locations"]
                    if r["kind"] == BomKind.MBOM_130.value
                }
            ),
            "object_types": dict(Counter(t for o in technical for t in o["types"])),
            "reference_rows": dict(Counter(r["dataset_type"] for r in references)),
        },
        "objects": objects,
        "questions": [],
        "limitations": [
            "Snapshot membership does not establish historical add/remove changes.",
            "Ancestors provide impact context; descendants of a referenced group are not automatically changed.",
            "Product context does not confirm a formal SAP KMAT material type.",
            "Structural presence does not establish active production configuration or a manufacturing plant.",
        ],
    }

    if assessment_date is not None:
        from app.ecn.evidence.scoped_usage import enhance_snapshot
        enhance_snapshot(snapshot, ecn, rows, comparison_rows, datasets, assessment_date, plant_context)
        from app.ecn.evidence.plant_impact import build_plant_impact
        snapshot["plant_impact"] = build_plant_impact(snapshot["bom_changes"], references, ecn.source_json, snapshot["objects"])
    from app.ecn.evidence.review_questions import build_review_questions
    snapshot["questions"] = build_review_questions(snapshot)
    snapshot["programs"] = program_matches(snapshot)
    from app.ecn.evidence.snapshot_views import complete_snapshot_views
    complete_snapshot_views(snapshot, source_warnings)
    return snapshot


def program_matches(snapshot):
    """Group the ECN parts' EBOM and MBOM rows by program (model range).

    Programs are the ECN's declared affected products plus any model range of
    a pinned BOM, so a declared program without an uploaded BOM is listed with
    ``bom_uploaded`` False instead of silently disappearing.

    Args:
        snapshot: Snapshot with ``declared_products`` and technical objects
            whose ``locations`` carry kind, model_range and validity.

    Returns:
        [{"model_range", "declared", "bom_uploaded", "ebom": [...], "mbom": [...]}]
        with one compact dict per BOM row.
    """
    declared = [p.strip() for p in str(snapshot.get("declared_products") or "").split(",") if p.strip()]
    uploaded = {d.get("model_range") for d in snapshot.get("datasets", ()) if d.get("dataset_type") == "BOM" and not d.get("comparison")}
    programs = {name: {"model_range": name, "declared": True, "bom_uploaded": name in uploaded, "ebom": [], "mbom": []} for name in declared}
    for obj in snapshot["objects"]:
        if not obj["technical"]:
            continue
        for location in obj["locations"]:
            # 130% EBOM rows count only when the 150% EBOM has no row at their position.
            if location.get("comparison") or (location["kind"] not in {"ENRICHED_EBOM", "MBOM_130"} and not location.get("raw_only")):
                continue
            name = location.get("model_range") or "Model range not declared"
            program = programs.setdefault(name, {"model_range": name, "declared": False, "bom_uploaded": True, "ebom": [], "mbom": []})
            is_mbom = location["kind"] == "MBOM_130"
            program["mbom" if is_mbom else "ebom"].append({
                "number": obj["number"], "parent": location.get("parent"),
                "position": None if is_mbom else location.get("item_or_position"),
                "quantity": location.get("quantity"), "valid_from": location.get("valid_from_date"),
                "valid_to": location.get("valid_to_date"), "plants": (location.get("plant_scope") or {}).get("plants", []),
                "bom_kind": location["kind"], "file_name": location.get("file_name"), "sheet": location.get("sheet"), "row": location.get("row"),
            })
    # Programs with BOM rows first, then declared-only programs.
    return sorted(programs.values(), key=lambda p: (not (p["ebom"] or p["mbom"]), p["model_range"]))


def snapshot_page(
    snapshot,
    *,
    page=1,
    page_size=25,
    query="",
    technical=True,
    ancestor=None,
    origin="ecn",
):
    """Page referenced objects, optionally restricted to a proven ancestor; never return whole subtrees."""
    objects = [
        o
        for o in snapshot["objects"]
        if o["technical"] == technical
        and origin in o["origins"]
        and query.casefold()
        in f"{o['number']} {o['description']} {o['types']}".casefold()
        and (
            ancestor is None
            or any(n["number"] == ancestor for r in o["locations"] for n in r["path"])
        )
    ]
    start = (page - 1) * page_size
    items = [
        {
            k: v
            for k, v in o.items()
            if k not in {"locations", "source_references", "reference_records"}
        }
        | {
            "occurrence_count": len(o["locations"]),
            "source_reference_count": len(o["source_references"]),
            "reference_record_count": len(o["reference_records"]),
        }
        for o in objects[start : start + page_size]
    ]
    return {
        "assessment_id": snapshot["assessment_id"],
        "total": len(objects),
        "page": page,
        "page_size": page_size,
        "items": items,
    }
