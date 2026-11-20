"""Read immutable source snapshots from HANA for UI, summary and assistant views."""

import json
from contextlib import nullcontext
from dataclasses import replace
from functools import lru_cache

from sqlalchemy import bindparam, text

from app.ecn.errors import PersistenceNotFoundError
from app.ecn.evidence.snapshot import build_snapshot, snapshot_page
from app.ecn.evidence.part_index import build_part_index
from app.ecn.intake.json_parser import _normalize_ecn
from app.ecn.models import BomKind, BomRowClassification
from app.ecn.reference_data.purchase_info_price_coverage import (
    assess_price_overlay_coverage,
)
from app.ecn.reference_data.purchase_info_reconciliation import material_plant
from app.ecn.rules.importer import load_parsed_rules
from app.ecn.rules.plant_scope import build_vc_index
from app.ecn.services.automated_manufacturing_feasibility import FixtureDrawingResolver
from app.ecn.evidence.business import business_evidence_records, BusinessEvidenceSection
from app.persistence.codec import normalize_row_keys, read_lob_text
from app.persistence.table_contracts import quote_identifier

# Capacity of ECN_IMPACT_ECN.AFFECTED_PLANTS_JSON (NVARCHAR(2000)).
PLANT_LABELS_MAX_LENGTH = 2000


def plant_labels(snapshot):
    """Return the display label of every plant in a snapshot's plant table.

    Labels match the "Plant" column of the ECN overview ("PL10 Plant A", or
    just the code when the name is unknown).

    Args:
        snapshot: Output of ``build_snapshot``; ``plant_impact`` is absent for
            ECNs assessed without an assessment date.

    Returns:
        Labels in plant-code order; ``[]`` when no plant is linked. The
        "Plant not determined" row is not a plant and is left out.
    """
    plants = [p for p in (snapshot.get("plant_impact") or {}).get("plants") or [] if p.get("plant_code")]
    return [f"{p['plant_code']} {p['plant_name']}" if p.get("plant_name") else p["plant_code"] for p in plants]


def plant_labels_json(labels):
    """Serialize plant labels for storage, dropping trailing labels that do not fit.

    Args:
        labels: Plant labels from ``plant_labels``.

    Returns:
        A JSON array string of at most ``PLANT_LABELS_MAX_LENGTH`` characters.
    """
    labels = list(labels)
    # ponytail: ~90 plants fit; past that the last plants are dropped from the
    # list column only (the ECN overview still shows all). Widen the column if hit.
    while len(value := json.dumps(labels, ensure_ascii=False)) > PLANT_LABELS_MAX_LENGTH:
        labels.pop()
    return value



BOM_SOURCE_LABELS = {
    BomKind.ENRICHED_EBOM.value: "Enriched EBOM",
    BomKind.RAW_EBOM_130.value: "130% EBOM",
    BomKind.MBOM_130.value: "130% MBOM",
}


def observed_bom_coverage(obj):
    """Project complete, source-backed BOM occurrence coverage for one material.

    This deliberately counts observations only. A component appearing in both
    engineering and manufacturing BOM sources is not represented as a mapping:
    the supplied data has no handover or occurrence-pairing key.
    """
    occurrences = {}
    for location in obj.get("locations", []):
        if location.get("comparison") or location.get("kind") not in BOM_SOURCE_LABELS:
            continue
        path = location.get("path") or []
        key = location.get("occurrence_id") or (
            location.get("kind"),
            tuple((node.get("number"), node.get("description"), node.get("level")) for node in path),
            location.get("item_or_position"),
            location.get("quantity"),
            location.get("unit"),
            location.get("valid_from_date"),
            location.get("valid_to_date"),
            location.get("change_number"),
            location.get("change_number_to"),
        )
        occurrences[key] = {
            "source": BOM_SOURCE_LABELS[location["kind"]],
            "kind": location["kind"],
            "path": path,
            "item_or_position": location.get("item_or_position"),
            "quantity": location.get("quantity"),
            "unit": location.get("unit"),
            "valid_from_date": location.get("valid_from_date"),
            "valid_from_quality": location.get("valid_from_quality"),
            "valid_from": location.get("valid_from"),
            "valid_to_date": location.get("valid_to_date"),
            "valid_to_quality": location.get("valid_to_quality"),
            "valid_to": location.get("valid_to"),
            "change_number": location.get("change_number"),
            "change_number_to": location.get("change_number_to"),
            # Direct source values only; derived plant scope is intentionally excluded.
            "plant": location.get("plant_code"),
        }

    rows = sorted(occurrences.values(), key=lambda row: (
        (BomKind.ENRICHED_EBOM.value, BomKind.RAW_EBOM_130.value, BomKind.MBOM_130.value).index(row["kind"]),
        "/".join(str(node.get("number") or "") for node in row["path"]),
        str(row["item_or_position"] or ""),
        str(row["valid_from_date"] or ""),
        str(row["change_number"] or ""),
    ))
    engineering = sum(row["kind"] != BomKind.MBOM_130.value for row in rows)
    manufacturing = sum(row["kind"] == BomKind.MBOM_130.value for row in rows)
    coverage = (
        "OBSERVED_IN_BOTH" if engineering and manufacturing
        else "ENGINEERING_ONLY" if engineering
        else "MANUFACTURING_ONLY" if manufacturing
        else "NOT_OBSERVED"
    )
    intervals = {
        (row["valid_from_date"], row["valid_to_date"], row["valid_from_quality"], row["valid_to_quality"])
        for row in rows
    }
    return {
        "engineering_occurrences": engineering,
        "manufacturing_occurrences": manufacturing,
        "coverage": coverage,
        "plants": sorted({row["plant"] for row in rows if row["plant"]}),
        "effective_interval_count": len(intervals),
        "occurrences": rows,
    }


def _source_purchase_price(
    material_number: object,
    plant_code: object,
    price_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Project one plant's price-source state without choosing among records."""
    target = material_plant(
        {"material_number": material_number, "plant_code": plant_code}
    )
    if target is None:
        return {"status": "NOT_SUPPLIED", "matching_source_row_count": 0}

    matches = [row for row in price_rows if material_plant(row) == target]
    coverage = assess_price_overlay_coverage(matches, [(str(material_number), str(plant_code))])
    entry = coverage.entries[0]
    result: dict[str, object] = {
        "status": entry.status,
        "matching_source_row_count": entry.matching_source_row_count,
    }
    if entry.status == "SOURCE_PRICE_AVAILABLE":
        source = matches[0]
        result.update(
            {
                "net_price": source.get("net_price"),
                "currency": source.get("currency"),
                "price_per": source.get("price_per"),
                "order_price_unit": source.get("order_price_unit"),
            }
        )
    return result


def material_brief(snapshot, obj, *, price_rows: list[dict[str, object]] | None = None):
    """Return concise relationship, permission, and source-price facts for one material dialog."""
    plant_names = {
        plant["plant_code"]: plant.get("plant_name")
        for plant in (snapshot.get("plant_impact") or {}).get("plants") or []
        if plant.get("plant_code")
    }
    links = {}
    for location in obj.get("locations", []):
        if location.get("comparison"):
            continue
        path = location.get("path") or []
        parent = next((node for node in reversed(path) if node.get("number") == location.get("parent")), None)
        parent = parent or (path[-2] if len(path) > 1 else None) or {}
        plants = tuple(sorted(
            (plant.get("plant_code"), plant.get("plant_name"))
            for plant in (location.get("plant_scope") or {}).get("plants") or []
            if plant.get("plant_code")
        ))
        link = {
            "assembly": parent.get("number") or location.get("parent"),
            "assembly_description": parent.get("description"),
            "model_range": location.get("model_range"),
            # A plant is shown only when it is present on the source occurrence;
            # rule-derived scope remains internal evidence, not a claimed source value.
            "plant": location.get("plant_code"),
            "kind": location.get("kind"),
            "path": path,
            "plants": [
                {"plant_code": code, "plant_name": name or plant_names.get(code)}
                for code, name in plants
            ],
            "bom": "EBOM" if location.get("kind") in {"ENRICHED_EBOM", "RAW_EBOM_130"} else "MBOM",
            "usage_status": location.get("usage_status"),
            "valid_from_date": location.get("valid_from_date"),
            "valid_to_date": location.get("valid_to_date"),
            "quantity": location.get("quantity"),
            "unit": location.get("unit"),
        }
        key = (
            link["assembly"], link["assembly_description"], link["model_range"], link["plant"],
            link["kind"], tuple((node.get("number"), node.get("level"), node.get("label")) for node in path), plants,
            link["bom"], link["usage_status"], link["valid_from_date"], link["valid_to_date"],
            link["quantity"], link["unit"],
        )
        links[key] = link

    sourcing = {}
    price_rows = price_rows if price_rows is not None else [
        reference
        for reference in obj.get("reference_records", [])
        if reference.get("dataset_type") == "PURCHASE_INFO_PRICES"
        and not reference.get("comparison")
    ]
    for reference in obj.get("reference_records", []):
        if reference.get("dataset_type") != "MATERIAL_MASTER" or reference.get("comparison"):
            continue
        code = reference.get("plant_code")
        entry = sourcing.setdefault(code, {
            "plant_code": code,
            "plant_name": plant_names.get(code),
            "procurement_type_labels": set(),
        })
        if label := reference.get("procurement_type_label"):
            entry["procurement_type_labels"].add(label)

    # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: mock procurement enrichment.
    # The price/recommendation branches below are deliberately disabled to validate
    # that material-master procurement permissions and observed BOM evidence stand
    # alone without the prototype overlay.
    # for price in snapshot.get("purchase_prices") or []:
    #     if price.get("material_number") != obj.get("number"):
    #         continue
    #     code = price.get("plant_code")
    #     entry = sourcing.setdefault(code, {
    #         "plant_code": code,
    #         "plant_name": plant_names.get(code),
    #         "procurement_type_labels": set(),
    #         "purchase_prices": [],
    #     })
    #     entry["purchase_prices"].append({
    #         "net_price": price.get("net_price"),
    #         "currency": price.get("currency"),
    #         "price_per": price.get("price_per"),
    #         "unit_of_measure": price.get("unit_of_measure"),
    #         "status": price.get("status"),
    #     })
    # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: mock procurement enrichment.

    sourcing_rows = [
        {
            **entry,
            "procurement_type_labels": sorted(entry["procurement_type_labels"]),
            "source_lead_time": {"status": "NOT_SUPPLIED"},
            "source_purchase_price": _source_purchase_price(
                obj.get("number"), entry["plant_code"], price_rows
            ),
        }
        for _, entry in sorted(sourcing.items(), key=lambda item: (item[0] is None, item[0] or ""))
    ]

    # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: curated make-or-buy assessments.
    # sourcing_by_plant = {entry["plant_code"]: entry for entry in sourcing_rows}
    # impacted_plants = {
    #     plant["plant_code"]
    #     for link in links.values()
    #     for plant in link["plants"]
    #     if plant.get("plant_code")
    # }
    # make_buy_assessments = []
    # for recommendation in snapshot.get("make_buy_recommendations") or []:
    #     if recommendation.get("material_number") != obj.get("number"):
    #         continue
    #     plant_code = recommendation.get("plant_code")
    #     prices = (sourcing_by_plant.get(plant_code) or {}).get("purchase_prices") or []
    #     buy_cost = next((price for price in prices if price.get("status") == "ACTIVE" and price.get("net_price")), None)
    #     buy_cost = buy_cost or next((price for price in prices if price.get("net_price")), None)
    #     scoped_plants = sorted(impacted_plants | ({plant_code} if plant_code else set()))
    #     make_buy_assessments.append({
    #         "plant_code": plant_code,
    #         "plant_name": plant_names.get(plant_code),
    #         "recommendation": recommendation.get("recommendation"),
    #         "recommendation_summary": recommendation.get("recommendation_summary"),
    #         "owner": recommendation.get("owner"),
    #         "make_cost": recommendation.get("make_cost"),
    #         "currency": recommendation.get("currency"),
    #         "price_per": recommendation.get("price_per"),
    #         "unit_of_measure": recommendation.get("unit_of_measure"),
    #         "buy_cost": buy_cost,
    #         "impacted_plant_count": len(scoped_plants),
    #         "comparable_history": recommendation.get("comparable_history"),
    #         "tooling": recommendation.get("tooling"),
    #         "capacity": recommendation.get("capacity"),
    #         "raw_materials": recommendation.get("raw_materials"),
    #         "source": recommendation.get("source"),
    #     })
    # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: curated make-or-buy assessments.

    return {
        "bom_coverage": observed_bom_coverage(obj),
        "linked_use": sorted(links.values(), key=lambda link: (
            (link["assembly_description"] or link["assembly"] or link["model_range"] or "").lower(),
            link["bom"], tuple(plant["plant_code"] for plant in link["plants"]),
        )),
        "sourcing": sourcing_rows,
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: "make_buy_assessments": make_buy_assessments,
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: "cost_assumption": (snapshot.get("demo_assumptions") or [None])[0],
    }


class SnapshotService:
    """Resolve only an ECN's pinned datasets, with small caches of immutable source versions."""

    def __init__(
        self,
        ecn_repository,
        artifact_repository,
        reference_repository,
        engine,
        table_names,
        manufacturing_feasibility_repository=None,
        drawing_resolver=None,
    ):
        """Bind existing read repositories and their exact HANA namespace."""
        self.ecns = ecn_repository
        self.artifacts = artifact_repository
        self.references = reference_repository
        self.engine = engine
        self.tables = table_names
        self.manufacturing_feasibility = manufacturing_feasibility_repository
        self.drawings = drawing_resolver or FixtureDrawingResolver()

    def pinned_references(self, upload_id, numbers, pinned_datasets=None):
        """Return complete matching reference rows and provenance from pinned HANA datasets."""
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        f'SELECT D."DATASET_ID", D."DATASET_TYPE", D."ARTIFACT_ID", A."FILE_NAME", A."SHA256" '
                        f"FROM {quote_identifier(self.tables.upload_dataset)} P "
                        f'JOIN {quote_identifier(self.tables.reference_dataset)} D ON P."DATASET_ID" = D."DATASET_ID" '
                        f'JOIN {quote_identifier(self.tables.source_artifact)} A ON D."ARTIFACT_ID" = A."ARTIFACT_ID" '
                        'WHERE P."UPLOAD_ID" = :id ORDER BY D."DATASET_TYPE", D."DATASET_ID"'
                    ),
                    {"id": upload_id},
                )
                .mappings()
                .all()
            )
            datasets = [
                {k.lower(): v for k, v in normalize_row_keys(r).items()} for r in rows
            ]
            if pinned_datasets is not None:
                datasets = list(pinned_datasets)
            references = []
            for dataset in datasets:
                key = {
                    "MATERIAL_MASTER": "material_master_row",
                    "PURCHASE_INFO_RECORDS": "purchase_info_row",
                    "PURCHASE_INFO_PRICES": "purchase_info_price_row",
                    "STOCK": "stock_row",
                }.get(dataset["dataset_type"])
                if key is None:
                    continue
                statement = text(
                    f"SELECT * FROM {quote_identifier(getattr(self.tables, key))} "
                    'WHERE "DATASET_ID" = :dataset AND "MATERIAL_NUMBER" IN :numbers ORDER BY "SOURCE_ROW_NUMBER"'
                ).bindparams(bindparam("numbers", expanding=True))
                for offset in range(0, len(numbers), 500):
                    found = (
                        connection.execute(
                            statement,
                            {
                                "dataset": dataset["dataset_id"],
                                "numbers": numbers[offset : offset + 500],
                            },
                        )
                        .mappings()
                        .all()
                    )
                    references.extend(
                        {k.lower(): v for k, v in normalize_row_keys(r).items()}
                        | {
                            "dataset_type": dataset["dataset_type"],
                            "file_name": dataset["file_name"],
                            "artifact_id": dataset["artifact_id"],
                            "comparison": dataset.get("comparison", False),
                            "snapshot_date": dataset.get("snapshot_date"),
                        }
                        for r in found
                    )
        return references, datasets

    @lru_cache(maxsize=4)  # noqa: B019 - bounded singleton assessment cache.
    def get(self, ecn_row_id):
        """Build complete deterministic facts from the occurrence's immutable source pins."""
        record = self.ecns.get_ecn_detail(ecn_row_id)
        if record is None:
            raise PersistenceNotFoundError(
                "ECN_NOT_FOUND",
                "The requested ECN occurrence was not found.",
                status_code=404,
            )
        # The existing detail reader deliberately returns metadata only. Load
        # this occurrence's source NCLOB explicitly, never the whole export BLOB.
        with self.engine.connect() as connection:
            raw = connection.execute(
                text(
                    f'SELECT "SOURCE_JSON" FROM {quote_identifier(self.tables.ecn)} WHERE "ECN_ROW_ID" = :id'
                ),
                {"id": ecn_row_id},
            ).scalar_one()
            source_json = json.loads(read_lob_text(raw))
        ecn = replace(
            _normalize_ecn(
                source_json,
                record.ecn_artifact.artifact_id,
                record.ecn.source_ordinal,
            ),
            ecn_row_id=ecn_row_id,
        )
        source_numbers = sorted(
            {
                obj.number
                for obj in (*ecn.affected_objects, *ecn.resulting_objects)
                if obj.is_technical_wtpart and obj.number
            }
        )
        bundle = self.ecns.get_evidence_bundle(ecn_row_id)
        context = bundle.context if bundle is not None else None
        modern = context is not None and context.deterministic_logic_version == "ecn-evidence-v3"
        pins = context.dataset_context if modern else ()
        bom_pins = [d for d in pins if d["dataset_type"] == "BOM"]
        rows, comparison_rows = [], []
        pins_or_upload = bom_pins or [{"artifact_id": record.bom_artifact.artifact_id}]
        # Parts the ECN starts or ends in any BOM (e.g. MBOM removals that are
        # not ECN objects) are loaded with all their rows, so remaining use and
        # earlier use can be checked.
        touched = sorted({n for pin in pins_or_upload for n in self.artifacts.change_touched_components(pin["artifact_id"], ecn.ecn_number)})
        load_numbers = sorted(set(source_numbers) | set(touched))
        for pin in pins_or_upload:
            target = comparison_rows if pin.get("comparison") else rows
            target.extend(self.artifacts.snapshot_bom_rows(pin["artifact_id"], load_numbers, ecn.ecn_number))
        numbers = sorted(
            set(load_numbers)
            | {
                r.component_number
                for r in rows
                if r.component_number
                and ecn.ecn_number
                and r.change_number == ecn.ecn_number
                and r.bom_kind == BomKind.RAW_EBOM_130
                and r.row_classification == BomRowClassification.MATERIAL_LIKE_OBSERVED
            }
        )
        refs, datasets = self.pinned_references(record.upload.upload_id, numbers, pins if modern and pins else None)
        warnings = []
        for warning in business_evidence_records(bundle) if bundle else ():
            if warning.section != BusinessEvidenceSection.DATA_QUALITY:
                continue
            raw = warning.technical_details.raw_values
            code = next((value.get("code") for value in raw if value.get("code")), None)
            code = code or next(iter(warning.technical_details.quality_qualifiers), "QUALITY_ISSUE")
            warnings.append({"code": code, "summary": warning.summary, "count": warning.matched_row_count,
                             "sources": [" · ".join(str(v) for v in (source.sheet_name, source.source_row_number, source.json_path) if v)
                                         for source in warning.sources]})
        snapshot = build_snapshot(
            ecn, rows, refs, evidence_hash=record.ecn.evidence_hash, datasets=datasets,
            assessment_date=context.analysis_as_of_date if modern else None, comparison_rows=comparison_rows, source_warnings=warnings,
            plant_context=self._plant_context(pins, context.plant_assignment if modern else {}),
        )
        self._store_plant_labels(ecn_row_id, snapshot)
        return snapshot

    def _plant_context(self, pins, plant_assignment):
        """Combine the assessment's pinned rule workbooks with its frozen Plant Code Mapping.

        Args:
            pins: Dataset pins of the assessment (``EvidenceContext.dataset_context``).
            plant_assignment: Mapping and excluded rule states frozen at import.

        Returns:
            ``{"index", "mappings", "rules_pinned"}`` for ``build_snapshot``.
        """
        rule_ids = tuple(sorted(pin["artifact_id"] for pin in pins
                                if pin["dataset_type"] == "CONDITIONAL_RULES" and not pin.get("comparison")))
        excluded = tuple(sorted(plant_assignment.get("excluded_rule_states") or ()))
        return {"index": self._vc_index(rule_ids, excluded), "mappings": list(plant_assignment.get("mappings") or ()),
                "rules_pinned": bool(rule_ids)}

    @lru_cache(maxsize=4)  # noqa: B019 - rule artifacts are immutable; a few versions are ever pinned.
    def _vc_index(self, rule_artifact_ids, excluded_states):
        """Build (once per rule-version set and state filter) the VC group index of pinned rules.

        Args:
            rule_artifact_ids: Sorted artifact IDs of the pinned rule workbooks.
            excluded_states: Sorted rule lifecycle states to ignore.

        Returns:
            A ``VcRuleIndex`` over every rule of those workbooks.
        """
        rules = []
        for artifact_id in rule_artifact_ids:
            artifact = self.artifacts.get_artifact(artifact_id)
            if artifact is not None and artifact.normalized_content:
                rules.extend(load_parsed_rules(artifact.normalized_content))
        return build_vc_index(rules, excluded_states)

    def _store_plant_labels(self, ecn_row_id, snapshot):
        """Save the snapshot's plant labels on the ECN row for the list page.

        The snapshot of one ECN occurrence is immutable, so the value is
        written only while still NULL and never rewritten.

        Args:
            ecn_row_id: ECN occurrence identifier.
            snapshot: The freshly built snapshot of that occurrence.
        """
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f'UPDATE {quote_identifier(self.tables.ecn)} SET "AFFECTED_PLANTS_JSON" = :plants '
                    'WHERE "ECN_ROW_ID" = :id AND "AFFECTED_PLANTS_JSON" IS NULL'
                ),
                {"plants": plant_labels_json(plant_labels(snapshot)), "id": ecn_row_id},
            )

    def affected_plants(self, ecn_row_id):
        """Return an ECN's plant labels, building (and saving) its snapshot if needed.

        Args:
            ecn_row_id: ECN occurrence identifier.

        Returns:
            Plant labels as stored for the list page.
        """
        return json.loads(plant_labels_json(plant_labels(self.get(ecn_row_id))))

    def overview(self, ecn_row_id):
        """Return complete aggregate facts without sending large object graphs to clients."""
        full_snapshot = self.get(ecn_row_id)
        snapshot = {k: v for k, v in full_snapshot.items() if k not in {"objects", "changes"}}
        # The decision brief needs part names even when a referenced object has no
        # BOM placement, but the full object graph remains behind the evidence API.
        snapshot["part_descriptions"] = [
            {"number": obj["number"], "description": obj.get("description")}
            for obj in full_snapshot.get("objects", [])
            if obj.get("technical") and obj.get("number") and obj.get("description")
        ]
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: runtime mock overlay.
        # from app.ecn.mock_overlay import merge_mock_data
        # return merge_mock_data(snapshot)
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: runtime mock overlay.
        return snapshot

    def changes(self, ecn_row_id, *, page=1, page_size=25):
        """Return a bounded page of shared material changes for this assessment."""
        items = self.get(ecn_row_id).get("changes", [])
        start = (page - 1) * page_size
        return {"assessment_id": ecn_row_id, "total": len(items), "page": page,
                "page_size": page_size, "items": items[start:start + page_size]}

    def objects(self, ecn_row_id, **filters):
        """Return a bounded page of referenced objects at their observed levels."""
        return snapshot_page(self.get(ecn_row_id), **filters)

    def _active_price_overlay_rows(self, snapshot, material_number: str, *, connection=None) -> list[dict[str, object]]:
        """Read the active same-scope Price Overlay for a display-only material lookup.

        Procurement permission remains part of the ECN's immutable assessment.
        Source purchase price is an operational display field, so it intentionally
        follows the active overlay in the same model-range/data-group/environment
        scope rather than requiring reassessment of historic ECNs.
        """
        scope_source = next(
            (
                dataset
                for dataset in snapshot.get("datasets", [])
                if not dataset.get("comparison")
                and dataset.get("dataset_type")
                in {"PURCHASE_INFO_RECORDS", "MATERIAL_MASTER"}
            ),
            None,
        )
        if scope_source is None:
            return []
        context = {
            key: scope_source.get(key)
            for key in ("model_range", "data_group", "environment")
        }
        overlay = next(
            (
                record
                for record in self.references.list(connection=connection)
                if record.active
                and str(record.dataset_type) == "PURCHASE_INFO_PRICES"
                and all(record.context.get(key) == value for key, value in context.items())
            ),
            None,
        )
        if overlay is None:
            return []
        with (self.engine.connect() if connection is None else nullcontext(connection)) as connection:
            rows = connection.execute(
                text(
                    f"SELECT * FROM {quote_identifier(self.tables.purchase_info_price_row)} "
                    'WHERE "DATASET_ID" = :dataset_id AND "MATERIAL_NUMBER" = :material_number '
                    'ORDER BY "SOURCE_ROW_NUMBER"'
                ),
                {"dataset_id": overlay.dataset_id, "material_number": material_number},
            ).mappings().all()
        return [
            {key.lower(): value for key, value in normalize_row_keys(row).items()}
            | {
                "dataset_type": "PURCHASE_INFO_PRICES",
                "dataset_id": overlay.dataset_id,
                "active_overlay": True,
            }
            for row in rows
        ]

    def _member_object(self, snapshot, number, technical=True):
        """Require exact membership in the selected ECN before reading material content."""
        obj = next(
            (
                o
                for o in snapshot["objects"]
                if (o["number"] == number or o["object_key"] == number)
                and o["technical"] == technical
            ),
            None,
        )
        if obj is None:
            raise PersistenceNotFoundError(
                "OBJECT_NOT_FOUND",
                "This object is not referenced by the selected ECN.",
                status_code=404,
            )
        return obj

    def drawing(self, ecn_row_id, number):
        """Read an allowlisted sample PDF for a technical member of this ECN."""
        obj = self._member_object(self.get(ecn_row_id), number)
        resolved = self.drawings.resolve(obj["number"])
        if resolved is None:
            raise PersistenceNotFoundError(
                "DRAWING_NOT_FOUND", "A drawing is not available for this material.",
                status_code=404,
            )
        return resolved

    def object_details(
        self, ecn_row_id, number, *, technical=True, page=1, page_size=25
    ):
        """Return one member object's paginated occurrences, references and source coordinates."""
        snapshot = self.get(ecn_row_id)
        obj = self._member_object(snapshot, number, technical)
        drawing = self.drawings.available(obj["number"]) if technical else None
        # Reuse one short-lived connection for mutable operational reads. Each
        # opening still reads the active price publication and latest report;
        # no mutable result is added to the immutable snapshot cache.
        with self.engine.connect() as connection:
            price_rows = self._active_price_overlay_rows(snapshot, obj["number"], connection=connection)
            manufacturing_feasibility = (
                self.manufacturing_feasibility.get_completed(ecn_row_id, obj["number"], connection=connection)
                if self.manufacturing_feasibility is not None else None
            )
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: object-detail mock overlay.
        # from app.ecn.mock_overlay import merge_mock_data
        # overlay = merge_mock_data({"ecn_number": snapshot.get("ecn_number")})
        # MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: object-detail mock overlay.
        counterpart_descriptions = {
            other.get("number"): other.get("description")
            for other in snapshot["objects"]
            if other.get("number") and other.get("description")
        }
        change = obj.get("change")
        if change:
            change = {
                **change,
                "counterpart_details": [
                    {"number": counterpart, "description": counterpart_descriptions.get(counterpart)}
                    for counterpart in change.get("counterparts") or []
                ],
            }
        start = (page - 1) * page_size
        return {
            k: v
            for k, v in obj.items()
            if k not in {"locations", "reference_records", "source_references", "change"}
        } | {
            "change": change,
            "material_brief": material_brief(
                snapshot,
                obj,
                price_rows=price_rows,
            ),
            "assessment_id": ecn_row_id,
            "assessment_date": snapshot.get("assessment_date"),
            "page": page,
            "page_size": page_size,
            "locations": obj["locations"][start : start + page_size],
            "location_total": len(obj["locations"]),
            "reference_records": obj["reference_records"][start : start + page_size],
            "reference_total": len(obj["reference_records"]),
            "source_references": obj["source_references"][start : start + page_size],
            "source_reference_total": len(obj["source_references"]),
            "engineering_drawing": (
                {"file_name": drawing.file_name, "media_type": "application/pdf"}
                if drawing else None
            ),
            "manufacturing_feasibility": manufacturing_feasibility,
        }

    @lru_cache(maxsize=1024)  # noqa: B019 - small immutable per-occurrence summary.
    def part_index(self, ecn_row_id):
        """Return the compact part summary of one ECN for cross-ECN matching.

        Args:
            ecn_row_id: ECN occurrence identifier.

        Returns:
            ``build_part_index`` output (ECN number and per-part roles,
            changes, assemblies, plants and dates).
        """
        # ponytail: the first cross-ECN part search builds every ECN's snapshot
        # once (the same cost as the list's first plant fill). If that becomes
        # slow, persist this summary on snapshot build like AFFECTED_PLANTS_JSON
        # and query it with SQL.
        return build_part_index(self.get(ecn_row_id))

    def clear(self):
        """Drop ephemeral cached facts when history is explicitly reset."""
        self.get.cache_clear()
        self.part_index.cache_clear()
