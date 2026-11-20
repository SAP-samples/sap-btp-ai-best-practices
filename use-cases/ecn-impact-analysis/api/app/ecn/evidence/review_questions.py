"""Named, evidence-backed follow-ups shared by the review page and AI synthesis."""


def _subject(obj, details=()):
    """Return a linked object identity and readable evidence from one snapshot object."""
    return {"number": obj["number"], "object_key": obj["object_key"], "details": list(details)}


def _plants_detail(location):
    """Name the rule-derived plants of one BOM occurrence, or say none was derived."""
    plants = (location.get("plant_scope") or {}).get("plants") or []
    return "Plants " + ", ".join(f'{p["plant_code"]} {p["plant_name"]}' for p in plants) if plants else "Plant not determined"


def _location_detail(location):
    """Describe a pinned BOM occurrence without inferring its production applicability."""
    return " · ".join(str(value) for value in (
        location.get("file_name"), location.get("sheet"), f'row {location["row"]}',
        location.get("model_range"), _plants_detail(location),
        f'Parent {location["parent"]}' if location.get("parent") else "Parent not supplied",
        f'Valid from {location.get("valid_from_date") or "unknown"}',
        "No planned end" if location.get("valid_to_date") == "9999-12-31" else f'Valid to {location.get("valid_to_date") or "unknown"}',
    ) if value)


def build_review_questions(snapshot):
    """Build actionable groups from complete pinned objects, dates and stock records.

    Input is the final v1/v2 snapshot, after scoped usage has been attached.
    Output retains the existing question fields and adds subjects, scope and
    related factors. Missing occurrences are never assigned an invented location.
    """
    questions = []
    objects = [o for o in snapshot["objects"] if o["technical"]]
    scopes = [" · ".join(str(d.get(key) or fallback) for key, fallback in (
        ("file_name", "BOM file"), ("model_range", "Model not supplied"),
        ("snapshot_date", "Extraction date not supplied"),
    )) for d in snapshot["datasets"] if d["dataset_type"] == "BOM" and not d.get("comparison")]

    def add(code, title, why, resolution, subjects, factors, checked_scopes=()):
        """Append a nonempty group with complete subjects and its next action; return nothing."""
        if subjects:
            questions.append({"code": code, "question": title, "why": why, "resolution": resolution,
                "affected_count": len(subjects), "subjects": subjects, "factors": factors,
                "scopes": list(checked_scopes)})

    missing, future, expired, unresolved, phaseout, manufacturing, lifecycle, scheduled = [], [], [], [], [], [], [], []
    # New assemblies (VC groups) sit in the MBOM as parents, not components;
    # the BOM change engine marks them MBOM when the ECN starts rows under them.
    manufacturing_lines = [l for l in (snapshot.get("bom_changes") or {}).get("lines", ())
                           if l.get("incoming") and "MBOM" in l.get("levels", ())]
    in_mbom = {l["incoming"]["number"] for l in manufacturing_lines
               if "EFFECTIVE" in l.get("manufacturing_usage_statuses", ())}
    for obj in objects:
        if "ecn" not in obj["origins"]:
            continue
        current = [l for l in obj["locations"] if not l.get("comparison")]
        change = obj.get("change") or {}
        action = change.get("action")
        if not current and change.get("usage", {}).get("status") != "NO_REMAINING_USE_IN_SCOPE":
            historical = any(l.get("comparison") for l in obj["locations"])
            missing.append(_subject(obj, [f'{obj["number"]} was not found as a component in the supplied current EBOMs or MBOMs.'
                + (" Historical occurrences are available in the material evidence." if historical else "")]))
        elif current and snapshot["version"] == "ecn-snapshot-v2":
            statuses = {l["usage_status"] for l in current}
            if "UNRESOLVED" in statuses:
                unresolved.append(_subject(obj, [_location_detail(l) for l in current if l["usage_status"] == "UNRESOLVED"]))
            if "FUTURE" in statuses and "EFFECTIVE" not in statuses:
                future.append(_subject(obj, [_location_detail(l) for l in current if l["usage_status"] == "FUTURE"]))
            if statuses == {"EXPIRED"} and action != "PHASE_OUT":
                expired.append(_subject(obj, [_location_detail(l) for l in current]))
            if action == "PHASE_OUT" and "EFFECTIVE" in statuses:
                phaseout.append(_subject(obj, [_location_detail(l) for l in current if l["usage_status"] == "EFFECTIVE"]))
            if action == "PHASE_IN" and "EFFECTIVE" in statuses and obj["number"] not in in_mbom and not any(
                l["kind"] == "MBOM_130" and l["usage_status"] == "EFFECTIVE" for l in current
            ):
                future_mbom = [l for l in current if l["kind"] == "MBOM_130" and l["usage_status"] == "FUTURE"]
                future_parent = [l for l in manufacturing_lines if l["incoming"]["number"] == obj["number"]
                                 and "FUTURE" in l.get("manufacturing_usage_statuses", ())]
                if future_mbom:
                    scheduled.append(_subject(obj, [_location_detail(l) for l in future_mbom]))
                elif future_parent:
                    scheduled.append(_subject(obj, [f'Manufacturing assembly {obj["number"]}: starts {l["manufacturing_date"]}'
                                                   for l in future_parent]))
                else:
                    manufacturing.append(_subject(obj, [_location_detail(l) for l in current]))
        if action in {"UNRESOLVED", "REVIEW"} or snapshot["version"] == "ecn-snapshot-v1":
            lifecycle.append(_subject(obj))

    add("BOM_LOCATION_UNAVAILABLE", "Material not found in supplied EBOMs or MBOMs",
        "These materials are referenced by the ECN but were not found in the supplied engineering BOMs (EBOMs) or manufacturing BOMs (MBOMs). "
        "Their parent assembly and position in the BOM hierarchy are therefore unknown from these files.",
        "Check each listed material number with engineering and confirm which assembly should contain it, for which model, plant and start date. "
        "Publish the relevant dated EBOM or MBOM in Settings and reassess this ECN. "
        "If a material should not be a BOM component, record the reason for that material in Engineering review.",
        missing, ["manufacturing", "configuration"], scopes or ["BOM model/plant scope was not declared."])
    add("BOM_FUTURE_USAGE", "Confirm the scheduled introduction date",
        "Occurrences exist, but their validity starts after the assessment date.",
        "Check the listed start dates against the planned introduction. Reassess at the intended date, or supply a corrected BOM if the schedule is wrong.",
        future, ["manufacturing"])
    add("BOM_EXPIRED_USAGE", "Confirm use after the recorded end date",
        "Only expired occurrences were found in the assessed current BOMs.",
        "Confirm whether the listed material is still required. If so, supply its current dated BOM and reassess.",
        expired, ["manufacturing"])
    add("BOM_VALIDITY_UNRESOLVED", "Confirm BOM validity dates",
        "The listed occurrences have missing, invalid or placeholder validity dates.",
        "Ask the BOM owner to verify the start and end dates at the listed source rows. Publish the corrected BOM and reassess before concluding current use.",
        unresolved, ["manufacturing"])
    add("PHASE_OUT_STILL_USED", "Confirm phase-out timing and remaining use",
        "The ECN declares phase-out while the listed BOM occurrences are effective on the assessment date.",
        "Confirm when each listed assembly should stop using the outgoing material and when purchasing should stop. Record the disposition in Engineering review; supply updated BOMs when available and reassess.",
        phaseout, ["manufacturing", "procurement"])
    add("MBOM_INTRODUCTION_UNCONFIRMED", "Confirm manufacturing introduction",
        "These incoming materials appear in the engineering BOM (EBOM), but have no manufacturing BOM (MBOM) row valid on the assessment date.",
        "Confirm the intended manufacturing assembly, applicable plant and start date. Supply the corresponding MBOM and reassess; engineering BOM presence alone does not establish manufacturing use.",
        manufacturing, ["manufacturing"])
    add("MBOM_FUTURE_USAGE", "Confirm the scheduled manufacturing rollout",
        "Engineering use is observed, and manufacturing occurrences exist with a later start date.",
        "Confirm that the listed manufacturing start dates match the rollout plan for each shown assembly and plant. "
        "Reassess at that date; provide a corrected MBOM only if the recorded schedule is wrong.", scheduled, ["manufacturing"])
    add("CHANGE_DIRECTION_UNCONFIRMED", "Confirm the intended lifecycle action",
        "Source references do not establish a consistent addition, removal, replacement or revision for the listed objects.",
        "Inspect each material's source evidence. Record the intended action and supporting source in Engineering review; provide corrected source data if required.",
        lifecycle, ["manufacturing"])

    stock_subjects, missing_fields = [], set()
    for obj in objects:
        details = []
        for row in obj["reference_records"]:
            if row.get("dataset_type") != "STOCK" or row.get("comparison"):
                continue
            absent = [label for key, label in (("unit", "unit of measure"), ("snapshot_date", "extraction date")) if not row.get(key)]
            if not absent:
                continue
            missing_fields.update(absent)
            details.append(" · ".join(str(v) for v in (
                row.get("file_name") or "Stock file", row.get("sheet_name") or "Stock",
                f'row {row.get("source_row_number") or "not supplied"}',
                f'Plant {row.get("plant_code") or "not supplied"}',
                f'Storage {row.get("storage_location") or "not supplied"}',
                f'Reported quantity {row.get("unrestricted_quantity", "not supplied")}',
                f'Extracted {row["snapshot_date"]}' if row.get("snapshot_date") else None,
                f'Missing: {", ".join(absent)}',
            ) if v is not None))
        if details:
            stock_subjects.append(_subject(obj, details))
    add("STOCK_CONTEXT_LIMITED", "Confirm missing stock fields",
        "The listed stock records cannot be compared reliably until these specific fields are confirmed. Missing stock records are not zero stock.",
        f'Ask the stock-data owner to confirm the {" and ".join(sorted(missing_fields))} for the listed rows. '
        "Record the confirmation in Engineering review before aggregating stock exposure.", stock_subjects, ["procurement"])
    if snapshot["version"] == "ecn-snapshot-v2" and not snapshot.get("coverage_complete"):
        add("COVERAGE_INCOMPLETE", "Confirm the model and plant coverage",
            "The assessed BOM datasets do not declare complete, dated model/plant coverage.",
            "Confirm which models and plants this ECN applies to. Publish dated BOM datasets with that scope and coverage in Settings, then reassess.",
            [_subject(o) for o in objects if "ecn" in o["origins"]], ["configuration"], scopes)
    return questions
