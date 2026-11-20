"""Deterministic test-data builders for the ECN impact domain.

This module provides both domain-record builders (make_bom_row, etc.) and
in-memory workbook builders (bom_workbook_bytes, enriched_headers, etc.)
used by test_bom_parser.py and test_bom_hierarchy.py.  No file I/O is
performed; all builders return values directly from in-memory structures.

Every builder returns a valid frozen record with sensible defaults; all
parameters are keyword overrides so tests only need to specify the fields
they care about.  Fixed UUID strings and UTC datetimes are used throughout
to keep tests reproducible without reading from ``data/``.

Example usage::

    from tests.fixtures.builders import make_source_artifact, make_bom_row

    artifact = make_source_artifact(file_name="custom.json")
    row = make_bom_row(material_number="MAT-001")
"""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from uuid import NAMESPACE_URL, uuid5

from app.ecn.api_models import ImpactCounts
from app.ecn.models import (
    DETERMINISTIC_LOGIC_VERSION,
    PARSER_VERSION,
    RULE_EVALUATOR_VERSION,
    AnalysisSectionKey,
    ArtifactKind,
    BomKind,
    BomRow,
    BomRowClassification,
    EvidenceBundle,
    EvidenceBundleSections,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceTier,
    NormalizedEcn,
    NormalizedEffectivity,
    NormalizedSourceObject,
    ParseState,
    SourceArtifact,
    SourceObjectRole,
)

# analysis_models is imported lazily inside the builder functions so that
# the builders module itself does not create a hard import-time dependency on
# Task 2 modules when only Task 1 builders are used in isolation.
#
# If analysis_models is not yet available (e.g. during early Task 1 tests),
# the Task 2 builders will raise ImportError at call time, not at import time.

# ---------------------------------------------------------------------------
# Fixed UUID pool — all version-4-shaped UUIDs reserved for test use only.
# These share a common prefix to make test data easy to recognise in logs.
# ---------------------------------------------------------------------------

_UUID_ARTIFACT = "00000000-0000-4000-8000-000000000001"
_UUID_UPLOAD = "00000000-0000-4000-8000-000000000002"
_UUID_BOM_ROW = "00000000-0000-4000-8000-000000000003"
_UUID_ECN_ROW = "00000000-0000-4000-8000-000000000004"
_UUID_SOURCE_OBJ = "00000000-0000-4000-8000-000000000005"
_UUID_BOM_ARTIFACT = "00000000-0000-4000-8000-000000000006"

# Artifact UUID reserved for bom/rule-artifact row builder (index 0x10 → 16)
_UUID_BOM_ARTIFACT_ROW = "00000000-0000-4000-8000-000000000010"

# UUIDs reserved for Task-6 upload-publication builders.
_UUID_REQUEST = "00000000-0000-4000-8000-000000000007"
_UUID_JOB = "00000000-0000-4000-8000-000000000008"

# Fixed timestamp used as the default created_at for all artifact builders.
_CREATED_AT = datetime(2026, 9, 14, tzinfo=timezone.utc)


def synthetic_uuid(label: str) -> str:
    """Return a stable canonical UUID for one human-readable synthetic label."""
    return str(uuid5(NAMESPACE_URL, f"ecn-impact-test:{label}"))


def make_source_artifact(**kwargs: object) -> SourceArtifact:
    """Build a minimal valid ``SourceArtifact`` with keyword overrides.

    ``byte_size`` is derived from ``content_bytes`` unless the caller
    explicitly overrides it.

    Args:
        **kwargs: Any ``SourceArtifact`` field values that should differ from
            the built-in defaults.

    Returns:
        A frozen ``SourceArtifact`` instance.
    """
    # Pop content_bytes early so we can derive byte_size from the actual bytes.
    content_bytes: bytes = kwargs.pop("content_bytes", b"[]")  # type: ignore[assignment]
    defaults: dict[str, object] = {
        "artifact_id": _UUID_ARTIFACT,
        "artifact_kind": ArtifactKind.ECN_JSON,
        "file_name": "ecn.json",
        "media_type": "application/json",
        "byte_size": len(content_bytes),
        "sha256": hashlib.sha256(content_bytes).hexdigest(),
        "content_bytes": content_bytes,
        "normalized_content": "[]",
        "parse_state": ParseState.VALID,
        "parse_error_code": None,
        "created_at": _CREATED_AT,
    }
    defaults.update(kwargs)
    # Recompute byte_size from content_bytes if the caller did not override it.
    if "byte_size" not in kwargs:
        defaults["byte_size"] = len(defaults["content_bytes"])  # type: ignore[arg-type]
    return SourceArtifact(**defaults)  # type: ignore[arg-type]


def make_bom_row(**kwargs: object) -> BomRow:
    """Build a minimal valid ``BomRow`` with all nullable columns set to ``None``.

    ``source_row_number`` defaults to ``2`` (row 1 is typically a header).

    Args:
        **kwargs: Any ``BomRow`` field values that should differ from the
            built-in defaults.

    Returns:
        A frozen ``BomRow`` instance.
    """
    defaults: dict[str, object] = {
        "bom_row_id": _UUID_BOM_ROW,
        "artifact_id": _UUID_BOM_ARTIFACT,
        "sheet_name": "EBOM",
        "bom_kind": BomKind.ENRICHED_EBOM,
        "source_row_number": 2,
        "parent_bom_row_id": None,
        "dotted_level": None,
        "hierarchy_path": None,
        "main_value": None,
        "sub_value": None,
        "vc_value": None,
        "level_4_value": None,
        "level_5_value": None,
        "level_6_value": None,
        "item_value": None,
        "position_value": None,
        "material_number": None,
        "parent_material_number": None,
        "component_number": None,
        "change_number": None,
        "change_number_to": None,
        "description_text": None,
        "unit": None,
        "plant": None,
        "alternative": None,
        "valid_from_raw": None,
        "valid_to_raw": None,
        "quantity": None,
        "valid_from_date": None,
        "valid_to_date": None,
        "row_classification": BomRowClassification.NOT_APPLICABLE,
        "is_documentation": False,
        "is_migration_date": False,
        "row_json": (),
        "quality_qualifiers": (),
        "quality_details": {},
    }
    defaults.update(kwargs)
    return BomRow(**defaults)  # type: ignore[arg-type]


def make_source_object(**overrides: object) -> NormalizedSourceObject:
    """Return one complete synthetic technical WTPart source object."""
    base = NormalizedSourceObject(
        source_object_id=synthetic_uuid("object-1"),
        source_path="$.ImplementationPlan[0].ResultingLinks[0].ResultingObjects[0]",
        link_path="$.ImplementationPlan[0].ResultingLinks[0]",
        role=SourceObjectRole.RESULTING,
        source_ordinal=0,
        technical_id="OR:wt.part.WTPart:1",
        version_id="VR:wt.part.WTPart:11",
        number="PART-A",
        name="Part A",
        description=None,
        object_type="WTPart",
        revision="1",
        version="1.1",
        state="Approved",
        created_on=None,
        last_modified=None,
        relationship_values={},
        raw_object={},
        is_technical_wtpart=True,
        source_subtype="part",
        quality_qualifiers=(),
    )
    return replace(base, **overrides)


def make_ecn(**overrides: object) -> NormalizedEcn:
    """Return one complete synthetic normalized ECN occurrence."""
    base = NormalizedEcn(
        ecn_row_id=synthetic_uuid("ecn-row-1"),
        artifact_id=synthetic_uuid("ecn-artifact-1"),
        source_ordinal=0,
        ecn_number="ECN1",
        title="Synthetic ECN",
        description_summary="Synthetic change",
        description_text="DescriptionSummary: Synthetic change",
        affected_objects=(),
        resulting_objects=(),
        source_json={},
        normalization_qualifiers=(),
    )
    return replace(base, **overrides)


def make_normalized_ecn(**kwargs: object) -> NormalizedEcn:
    """Build a minimal ``NormalizedEcn`` with ``ecn_number=None`` by default.

    The nullable ``ecn_number`` defaults to ``None`` to exercise the code path
    encountered when a source ECN record carries no explicit number field.

    Args:
        **kwargs: Any ``NormalizedEcn`` field values that should differ from
            the built-in defaults.

    Returns:
        A frozen ``NormalizedEcn`` instance.
    """
    defaults: dict[str, object] = {
        "ecn_row_id": _UUID_ECN_ROW,
        "artifact_id": _UUID_ARTIFACT,
        "source_ordinal": 0,
        "ecn_number": None,
        "title": None,
        "description_summary": None,
        "description_text": None,
        "affected_objects": (),
        "resulting_objects": (),
        "source_json": {},
        "normalization_qualifiers": (),
    }
    defaults.update(kwargs)
    return NormalizedEcn(**defaults)  # type: ignore[arg-type]


def make_evidence_bundle_sections(**kwargs: object) -> EvidenceBundleSections:
    """Build an ``EvidenceBundleSections`` with all six sections empty by default.

    Args:
        **kwargs: Any section field values that should differ from the
            built-in empty tuple defaults.

    Returns:
        A frozen ``EvidenceBundleSections`` instance.
    """
    defaults: dict[str, object] = {
        "executive_summary": (),
        "change_intent": (),
        "affected_materials": (),
        "ebom_mbom_impact": (),
        "configuration_scope": (),
        "uncertainties_and_missing_evidence": (),
    }
    defaults.update(kwargs)
    return EvidenceBundleSections(**defaults)  # type: ignore[arg-type]


def make_evidence_context(**kwargs: object) -> EvidenceContext:
    """Build a minimal ``EvidenceContext`` with current version constants.

    Args:
        **kwargs: Any ``EvidenceContext`` field values that should differ from
            the built-in defaults.

    Returns:
        A frozen ``EvidenceContext`` instance.
    """
    defaults: dict[str, object] = {
        "upload_id": _UUID_UPLOAD,
        "ecn_artifact_id": _UUID_ARTIFACT,
        "bom_artifact_id": _UUID_BOM_ARTIFACT,
        "rule_artifact_id": None,
        "parser_version": PARSER_VERSION,
        "deterministic_logic_version": DETERMINISTIC_LOGIC_VERSION,
        "rule_evaluator_version": RULE_EVALUATOR_VERSION,
        "analysis_as_of_date": None,
    }
    defaults.update(kwargs)
    return EvidenceContext(**defaults)  # type: ignore[arg-type]


def make_evidence_bundle(**kwargs: object) -> EvidenceBundle:
    """Build a minimal ``EvidenceBundle`` with empty sections and current versions.

    ``context`` and ``sections`` are constructed from their own builders when
    not overridden, so callers only need to override the fields they care about.

    Args:
        **kwargs: Any ``EvidenceBundle`` field values that should differ from
            the built-in defaults.  Pass a pre-built ``EvidenceContext`` or
            ``EvidenceBundleSections`` to override those sub-records.

    Returns:
        A frozen ``EvidenceBundle`` instance.
    """
    defaults: dict[str, object] = {
        "ecn_row_id": _UUID_ECN_ROW,
        "context": make_evidence_context(),
        "sections": make_evidence_bundle_sections(),
        "evidence_hash": "0" * 64,
    }
    defaults.update(kwargs)
    return EvidenceBundle(**defaults)  # type: ignore[arg-type]


def minimal_evidence_bundle() -> EvidenceBundle:
    """Return one provider-ready bundle with a cited fact in every section."""
    items = {
        key.value: (
            EvidenceItem(
                evidence_id=f"E-{index:016x}",
                section_key=key,
                kind=EvidenceKind.SOURCE_FACT,
                tier=EvidenceTier.DIRECT_ECN_RAW,
                summary=f"Synthetic source fact for {key.value}.",
                sources=(),
                values={"section": key.value},
                related_ids=(),
                quality_qualifiers=(),
            ),
        )
        for index, key in enumerate(AnalysisSectionKey, start=1)
    }
    return make_evidence_bundle(
        sections=make_evidence_bundle_sections(**items),
        evidence_hash=hashlib.sha256(b"minimal-evidence-bundle").hexdigest(),
    )


# ---------------------------------------------------------------------------
# Analysis-response payload builders (Task 2)
# These return raw dicts suitable for AnalysisResponse.model_validate() and
# json.dumps() / validate_analysis_response().  Each section's default
# evidence_id is f"{key_value.upper()}-1" so that the standard allowed-set
# {key: {f"{key.value.upper()}-1"} for key in AnalysisSectionKey} accepts it.
# ---------------------------------------------------------------------------

_SECTION_KEYS = [
    "executive_summary",
    "change_intent",
    "affected_materials",
    "ebom_mbom_impact",
    "configuration_scope",
    "uncertainties_and_missing_evidence",
]


def _build_section(key_value: str, status: object) -> dict[str, object]:
    """Return a single valid section dict for the given status.

    The ``status`` value is passed through as-is so that callers may supply
    either a ``SectionStatus`` enum instance (required for
    ``AnalysisResponse.model_validate`` with ``strict=True``) or a plain
    string (accepted by ``model_validate_json`` through JSON round-trip).

    ``AnalysisSection`` uses ``strict=True`` in its model config, which means
    ``model_validate`` requires an actual ``SectionStatus`` instance for the
    ``status`` field; plain strings are only accepted via ``model_validate_json``
    (the LLM output path).  The ``analysis_payload`` builder therefore resolves
    string aliases to enum members so that ``model_validate(payload)`` works.

    Args:
        key_value: The snake_case section key (e.g. ``"change_intent"``).
        status: A ``SectionStatus`` enum instance (or string alias for
            ``"INSUFFICIENT_EVIDENCE"``).

    Returns:
        A dict acceptable to ``AnalysisSection.model_validate``.
    """
    # Import here to avoid a hard import-time dependency on analysis_models
    # when builders.py is loaded during Task 1-only test runs.
    from app.ecn.analysis_models import SectionStatus

    status_enum: SectionStatus = (
        SectionStatus(status) if isinstance(status, str) else status  # type: ignore[assignment]
    )
    if status_enum is SectionStatus.INSUFFICIENT_EVIDENCE:
        return {
            "status": status_enum,
            "assessment_markdown": f"Assessment for {key_value}: data is incomplete.",
            "evidence_ids": [],
            "missing_evidence": [f"{key_value} data is incomplete"],
        }
    # SUFFICIENT and NOT_APPLICABLE both require evidence_ids and empty missing_evidence.
    return {
        "status": status_enum,
        "assessment_markdown": f"Assessment for {key_value}.",
        "evidence_ids": [f"{key_value.upper()}-1"],
        "missing_evidence": [],
    }


# ---------------------------------------------------------------------------
# Task-11 builders: effectivity normalization result helper.
# ---------------------------------------------------------------------------


def make_normalized_effectivity(**kwargs: object) -> NormalizedEffectivity:
    """Build a minimal ``NormalizedEffectivity`` with all fields null by default.

    Useful for constructing expected values in effectivity and BOM-row tests
    without repeating the full positional constructor in every test.

    Args:
        **kwargs: Any ``NormalizedEffectivity`` field values that should differ
            from the built-in defaults (all null / False).

    Returns:
        A frozen ``NormalizedEffectivity`` instance.
    """
    defaults: dict[str, object] = {
        "raw_text": None,
        "parsed_date": None,
        "is_migration_placeholder": False,
        "parse_warning": None,
    }
    defaults.update(kwargs)
    return NormalizedEffectivity(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Task-5 builders: short-hand artifact constructor and DB-row helpers.
# These return either domain records or column-keyed dicts that the codec
# row-mapper functions can consume directly.
# ---------------------------------------------------------------------------


def source_artifact(**kwargs: object) -> SourceArtifact:
    """Build a minimal valid ``SourceArtifact``; shorthand for make_source_artifact.

    All ``SourceArtifact`` fields default to sensible test values.
    Use keyword arguments to override only the fields under test.

    Args:
        **kwargs: Any ``SourceArtifact`` field values to override.

    Returns:
        A frozen ``SourceArtifact`` instance.
    """
    return make_source_artifact(**kwargs)


def source_artifact_row(**kwargs: object) -> dict[str, object]:
    """Return a column-keyed row dict matching the source-artifact table layout.

    The row uses the same defaults as ``make_source_artifact`` but with
    *column names* (uppercase) instead of Python field names.  The
    ``CREATED_AT`` value is a *naive* UTC datetime as the HANA driver would
    return it; ``artifact_from_row`` restores ``tzinfo=utc``.

    Args:
        **kwargs: Column name → value overrides.

    Returns:
        A dict of 11 column-keyed entries ready for ``artifact_from_row``.
    """
    content_bytes: bytes = b"[]"
    defaults: dict[str, object] = {
        "ARTIFACT_ID": _UUID_ARTIFACT,
        "ARTIFACT_KIND": "ECN_JSON",
        "FILE_NAME": "ecn.json",
        "MEDIA_TYPE": "application/json",
        "BYTE_SIZE": len(content_bytes),
        "SHA256": hashlib.sha256(content_bytes).hexdigest(),
        "CONTENT_BYTES": content_bytes,
        "NORMALIZED_CONTENT": "[]",
        "PARSE_STATE": "VALID",
        "PARSE_ERROR_CODE": None,
        # Naive UTC — mirrors what the HANA TIMESTAMP column returns.
        "CREATED_AT": datetime(2026, 9, 14),  # noqa: DTZ001 - HANA is naive UTC.
    }
    defaults.update(kwargs)
    return defaults


def bom_row_mapping(n: int, **kwargs: object) -> dict[str, object]:
    """Return a column-keyed row dict matching the BOM-row table layout.

    Each call with a different ``n`` produces a distinct row UUID and
    source row number so that multiple rows can be queued into
    ``RecordingEngine`` in sequence without UUID collisions.

    The ``ROW_JSON`` and ``QUALITY_JSON`` columns use pre-serialized empty
    values so that ``bom_row_from_row`` can reconstruct a valid ``BomRow``.

    Args:
        n: A positive integer used to derive a unique UUID and row number.
        **kwargs: Column name → value overrides.

    Returns:
        A dict of 35 column-keyed entries ready for ``bom_row_from_row``.
    """
    row_uuid = f"00000000-0000-4000-8000-{n:012x}"
    row_json_str: str = json.dumps([])
    quality_json_str: str = json.dumps({"details": {}, "qualifiers": []})
    defaults: dict[str, object] = {
        "BOM_ROW_ID": row_uuid,
        "ARTIFACT_ID": _UUID_BOM_ARTIFACT_ROW,
        "BOM_KIND": "ENRICHED_EBOM",
        "SHEET_NAME": "EBOM",
        "SOURCE_ROW_NUMBER": n + 1,
        "PARENT_BOM_ROW_ID": None,
        "DOTTED_LEVEL": None,
        "HIERARCHY_PATH": None,
        "MAIN_VALUE": None,
        "SUB_VALUE": None,
        "VC_VALUE": None,
        "LEVEL_4_VALUE": None,
        "LEVEL_5_VALUE": None,
        "LEVEL_6_VALUE": None,
        "ITEM_VALUE": None,
        "POSITION_VALUE": None,
        "MATERIAL_NUMBER": None,
        "PARENT_MATERIAL_NUMBER": None,
        "COMPONENT_NUMBER": None,
        "CHANGE_NUMBER": None,
        "CHANGE_NUMBER_TO": None,
        "DESCRIPTION_TEXT": None,
        "QUANTITY": None,
        "UNIT": None,
        "PLANT": None,
        "ALTERNATIVE": None,
        "VALID_FROM_RAW": None,
        "VALID_FROM_DATE": None,
        "VALID_TO_RAW": None,
        "VALID_TO_DATE": None,
        "ROW_CLASSIFICATION": "NOT_APPLICABLE",
        "IS_DOCUMENTATION": 0,
        "IS_MIGRATION_DATE": 0,
        "ROW_JSON": row_json_str,
        "QUALITY_JSON": quality_json_str,
    }
    defaults.update(kwargs)
    return defaults


def analysis_payload(status: str = "SUFFICIENT") -> dict[str, object]:
    """Build a valid six-section ``AnalysisResponse`` payload dict.

    All six sections use the same ``status`` by default.  Override individual
    sections by mutating the returned dict before passing it to
    ``AnalysisResponse.model_validate`` or ``json.dumps``.

    The ``status`` argument is a string alias that is resolved to a
    ``SectionStatus`` enum instance so that the payload is compatible with
    ``AnalysisSection``'s ``strict=True`` model config when calling
    ``AnalysisResponse.model_validate``.  Because ``SectionStatus`` is a
    ``StrEnum`` (a ``str`` subclass), ``json.dumps`` serialises the enum
    members as plain strings, preserving JSON round-trip compatibility.

    The default evidence ID for section ``key`` is ``f"{key.upper()}-1"``,
    which matches the standard test allowed-set produced by::

        {key: {f"{key.value.upper()}-1"} for key in AnalysisSectionKey}

    Args:
        status: A ``SectionStatus`` string alias (``"SUFFICIENT"``,
            ``"NOT_APPLICABLE"``, or ``"INSUFFICIENT_EVIDENCE"``).
            Defaults to ``"SUFFICIENT"``.

    Returns:
        A dict with a ``"sections"`` key containing six section dicts, each
        with a ``SectionStatus`` enum instance as the ``status`` value.
    """
    return {
        "sections": {
            key: _build_section(key, status)
            for key in _SECTION_KEYS
        }
    }


# ---------------------------------------------------------------------------
# Task-6 builders: upload publication, ECN publication, and upload row helpers.
# These provide complete, internally-consistent UploadPublication records
# so that UploadRepository unit tests only specify the fields under test.
# ---------------------------------------------------------------------------


def upload_row(**kwargs: object) -> dict[str, object]:
    """Return a column-keyed row dict matching the upload table read projection.

    Only the nine columns that ``upload_record_from_row`` reads are included;
    write-only columns such as ``PARSER_VERSION`` are omitted because the read
    projection does not expose them.  ``CREATED_AT`` is a naive UTC datetime as
    the HANA driver would return it; ``upload_record_from_row`` restores UTC.

    Args:
        **kwargs: Column name → value overrides.

    Returns:
        A dict of nine column-keyed entries ready for ``upload_record_from_row``.
    """
    defaults: dict[str, object] = {
        "UPLOAD_ID": _UUID_UPLOAD,
        "REQUEST_ID": _UUID_REQUEST,
        "PUBLICATION_ORDER": 1,
        "ECN_ARTIFACT_ID": _UUID_ARTIFACT,
        "BOM_ARTIFACT_ID": _UUID_BOM_ARTIFACT,
        "RULE_ARTIFACT_ID": None,
        "BOM_REUSED": 0,
        "IMPORTED_ECN_COUNT": 1,
        # Naive UTC — mirrors what the HANA TIMESTAMP column returns.
        "CREATED_AT": datetime(2026, 9, 14),  # noqa: DTZ001 - HANA is naive UTC.
    }
    defaults.update(kwargs)
    return defaults


def _make_ecn_publication_item(ordinal: int, upload_id: str, ecn_artifact_id: str, bom_artifact_id: str) -> object:
    """Build one ``EcnPublication`` for the given zero-based source ordinal.

    All UUIDs are derived from the ordinal so that multi-ECN publications have
    no UUID collisions.  The evidence context is wired to the supplied upload
    and artifact IDs so that validation invariants are satisfied.

    Args:
        ordinal: Zero-based position of this ECN in the source file.
        upload_id: The owning upload's canonical UUID.
        ecn_artifact_id: The ECN source artifact's canonical UUID.
        bom_artifact_id: The BOM source artifact's canonical UUID.

    Returns:
        A frozen ``EcnPublication`` instance.
    """
    from app.persistence.records import EcnPublication

    ecn_row_id = f"00000000-0000-4000-8000-{0x20 + ordinal:012x}"
    job_id = f"00000000-0000-4000-8000-{0x30 + ordinal:012x}"

    ecn = make_normalized_ecn(
        ecn_row_id=ecn_row_id,
        artifact_id=ecn_artifact_id,
        source_ordinal=ordinal,
    )
    context = make_evidence_context(
        upload_id=upload_id,
        ecn_artifact_id=ecn_artifact_id,
        bom_artifact_id=bom_artifact_id,
    )
    evidence = make_evidence_bundle(
        ecn_row_id=ecn_row_id,
        context=context,
    )
    counts = ImpactCounts(
        affected_item_count=0,
        raw_ecn_match_count=0,
        ebom_match_count=0,
        ebom_context_count=0,
        mbom_match_count=0,
        warning_count=0,
    )
    return EcnPublication(
        ecn=ecn,
        change_candidates=(),
        evidence=evidence,
        counts=counts,
        job_id=job_id,
        prompt_version="v1",
        output_schema_version="v1",
        model_name="gpt-4",
        model_config_json="{}",
    )


# ---------------------------------------------------------------------------
# Task-12 builders: in-memory XLSX workbook and raw/enriched hierarchy helpers.
# ---------------------------------------------------------------------------

# ISO-format strings accepted by the EXCEL_TYPED_OR_ISO effectivity parser.
EFFECTIVE_DATE = "2020-06-01"
"""Default validity-start ISO string for MBOM and enriched EBOM rows."""

EXPIRY_DATE = "9999-12-31"
"""Default validity-end ISO string representing an open-ended interval."""


def enriched_headers() -> list:
    """Return the 18 enriched EBOM column headers in the standard order.

    Matches the ENRICHED_EBOM WorkbookSheetContract.required_headers exactly
    so that role detection succeeds for worksheets built from this list.

    Returns:
        A list of 18 normalized header strings.
    """
    return [
        "Level", "Post number", "Component", "Item designation",
        "Component quantity", "Quantity unit", "N° of modification",
        "Start of validity", "Up to modif. Number", "Expiration date",
        "Parent", "Main", "Sub", "VC", "Level 4", "Level 5",
        "Level 6", "Alternative",
    ]


def enriched_row(
    *,
    level: int | str,
    component: str,
    parent: str,
    main: str,
    sub: str | None = None,
    vc: str | None = None,
    level_4: str | None = None,
    level_5: str | None = None,
    level_6: str | None = None,
    change_number: str = "ECN1",
    quantity: int = 1,
    post_number: str | None = None,
    valid_from: str = EFFECTIVE_DATE,
    valid_to: str = EXPIRY_DATE,
) -> list:
    """Return one enriched EBOM data row aligned to the 18-column header layout.

    Column positions match enriched_headers() exactly so that the parser can
    locate each field by its normalized header index.

    Args:
        level: Integer level 1–6 (mapped to dotted_level as its string form).
        component: Component number and the slot value at this level.
        parent: Parent material number (the row's Parent column value).
        main: Main-level value (Main column).
        sub: Sub-level value or None.
        vc: VC-level value or None.
        level_4: Level-4 value or None.
        level_5: Level-5 value or None.
        level_6: Level-6 value or None.
        change_number: ECN change number.
        quantity: Numeric quantity.
        post_number: Post number (position); defaults to None.
        valid_from: Start of validity ISO string.
        valid_to: Expiration date ISO string.

    Returns:
        A 18-element list in enriched_headers() column order.
    """
    return [
        level,           # Level
        post_number,     # Post number
        component,       # Component
        f"{component} description",  # Item designation
        quantity,        # Component quantity
        "EA",            # Quantity unit
        change_number,   # N° of modification
        valid_from,      # Start of validity
        "",              # Up to modif. Number
        valid_to,        # Expiration date
        parent,          # Parent
        main,            # Main
        sub,             # Sub
        vc,              # VC
        level_4,         # Level 4
        level_5,         # Level 5
        level_6,         # Level 6
        None,            # Alternative
    ]


def mbom_headers() -> list:
    """Return the 11 MBOM column headers in the standard order.

    Matches the MBOM_130 WorkbookSheetContract.required_headers exactly.

    Returns:
        A list of 11 normalized header strings.
    """
    return [
        "Item", "Component", "Component name", "Quantity",
        "N° of modification", "Start of validity", "at the number of modif.",
        "Expiration date", "alternative postal group", "Sequence",
        "Employment probability",
    ]


def bom_workbook_bytes(
    remove_role: str | None = None,
    duplicate_role: str | None = None,
    duplicate_required_header: str | None = None,
    extra_raw_rows: int = 0,
    add_unmatched_sheet: bool = False,
) -> bytes:
    """Return a synthetic three-role workbook with hierarchy and quality edge cases.

    The default workbook contains:
    - One raw EBOM sheet with a dotted hierarchy including a D-prefixed document,
      a year-1900 migration date, and two duplicate MS header columns.
    - One enriched EBOM sheet with a two-level hierarchy.
    - One MBOM sheet with a single component row.

    Optional modifications allow creating contract-violation scenarios for
    negative tests:

    Args:
        remove_role: BomKind name whose sheet is omitted (e.g. ``"MBOM_130"``).
        duplicate_role: BomKind name for which a second matching sheet is added
            so that role detection finds two candidates.
        duplicate_required_header: A required header string to duplicate in the
            raw sheet (e.g. ``"Typ"``).  The duplicate is inserted as an extra
            column before the original.
        extra_raw_rows: Number of additional data rows appended to the raw sheet;
            useful for row/cell limit tests.
        add_unmatched_sheet: When True, adds an extra sheet with no recognized
            required headers so that role detection must ignore it.

    Returns:
        OOXML-formatted bytes of the completed synthetic workbook.
    """
    # Import here to avoid making openpyxl a hard import-time dependency of
    # the builders module when only non-workbook builders are used.
    from openpyxl import Workbook as _Workbook

    workbook = _Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    # ------------------------------------------------------------------
    # Raw 130-percent EBOM sheet
    # ------------------------------------------------------------------
    raw_headers: list = [
        None, " Level no. ", "Item", "Component number", "Object description",
        "Quantity", "Un", "Valid from", "Valid to", "AltItm", "MS", "MS",
        "Typ", "Change No.", "Chg No. To",
    ]
    if duplicate_required_header is not None:
        raw_headers = [duplicate_required_header] + raw_headers

    raw_data = [
        [None, ".1",   "0",   "ROOT001",                     "Root",      1, "EA", "01.01.2000", "31.12.9999", "",  "a", "b", "ND", "ECN1", ""],
        [None, "..2",  "150", "0000123",                     "Part",      2, "EA", "02.01.1900", "31.12.9999", "",  "c", "d", "ND", "ECN2", ""],
        [None, "...3", "D001","0000123 ZVD 000 01",          "Document",  1, "EA", "02.01.1900", "31.12.9999", "",  "e", "f", "",   "ECN2", ""],
        [None, "..2",  "160", "0000456",                     "Unresolved",1, "EA", "05.06.2025", "31.12.9999", "",  "g", "h", "",   "ECN3", ""],
    ]
    if duplicate_required_header is not None:
        # Pad each data row with an extra leading None to align with duplicated header.
        raw_data = [[None] + row for row in raw_data]

    if remove_role != "RAW_EBOM_130":
        raw = workbook.create_sheet("Renamed raw evidence")
        raw.append(raw_headers)
        for row in raw_data:
            raw.append(row)
        for _ in range(extra_raw_rows):
            raw.append([None, ".1", "X", "EXTRA", "extra", 1, "EA", "01.01.2020", "31.12.9999", "", "", "", "ND", "ECN9", ""])

    # ------------------------------------------------------------------
    # Enriched EBOM sheet
    # ------------------------------------------------------------------
    if remove_role != "ENRICHED_EBOM":
        enriched = workbook.create_sheet("Renamed enriched evidence")
        enriched.append(enriched_headers())
        enriched.append(enriched_row(level=1, component="MAIN", parent="ROOT", main="MAIN"))
        enriched.append(enriched_row(level=2, component="SUB", parent="MAIN", main="MAIN", sub="SUB"))

    # ------------------------------------------------------------------
    # MBOM sheet
    # ------------------------------------------------------------------
    if remove_role != "MBOM_130":
        mbom = workbook.create_sheet("Renamed manufacturing evidence")
        mbom.append(mbom_headers())
        mbom.append(["SUB", "0000123", "Part", 2, "ECN2", EFFECTIVE_DATE, "", EXPIRY_DATE, "A", "0", 100])

    # ------------------------------------------------------------------
    # Optional duplicate-role sheet
    # ------------------------------------------------------------------
    if duplicate_role == "RAW_EBOM_130":
        extra_raw = workbook.create_sheet("Duplicate raw evidence")
        extra_raw.append(raw_headers)
    elif duplicate_role == "ENRICHED_EBOM":
        extra_enriched = workbook.create_sheet("Duplicate enriched evidence")
        extra_enriched.append(enriched_headers())
    elif duplicate_role == "MBOM_130":
        extra_mbom = workbook.create_sheet("Duplicate manufacturing evidence")
        extra_mbom.append(mbom_headers())

    # ------------------------------------------------------------------
    # Optional unrecognized sheet
    # ------------------------------------------------------------------
    if add_unmatched_sheet:
        extra = workbook.create_sheet("Unrecognized sheet")
        extra.append(["Some", "Unrelated", "Headers"])

    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def raw_hierarchy_rows(
    specs: list[tuple[str, str, str, str]],
) -> tuple["BomRow", ...]:
    """Build pre-classified RAW_EBOM_130 BomRow objects for raw hierarchy tests.

    Each spec tuple is ``(dotted_level, item_value, component_number, type_value)``.
    Row classification is computed via classify_raw_row.  TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED
    is added for MATERIAL_LIKE_OBSERVED rows.  No parent or path links are set —
    those are assigned by reconstruct_raw_hierarchy.

    Args:
        specs: List of ``(dotted_level, item_value, component_number, type_value)``
            tuples.

    Returns:
        A tuple of frozen BomRow instances in the same order as specs.
    """
    from app.ecn.intake.normalization import classify_raw_row
    from app.ecn.models import BomKind, BomRowClassification

    rows: list[BomRow] = []
    for index, (dotted_level, item_value, component_number, type_value) in enumerate(specs):
        classification = classify_raw_row(item_value, type_value, component_number)
        is_doc = classification is BomRowClassification.CONFIRMED_DOCUMENT
        qualifiers: tuple[str, ...] = ()
        if classification is BomRowClassification.MATERIAL_LIKE_OBSERVED:
            qualifiers = ("TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED",)
        row_id = f"10000000-0000-4000-8000-{index:012x}"
        rows.append(make_bom_row(
            bom_row_id=row_id,
            artifact_id=_UUID_BOM_ARTIFACT,
            bom_kind=BomKind.RAW_EBOM_130,
            sheet_name="Test raw sheet",
            source_row_number=index + 2,
            dotted_level=dotted_level,
            item_value=item_value,
            position_value=item_value,
            component_number=component_number,
            material_number=component_number,
            row_classification=classification,
            is_documentation=is_doc,
            quality_qualifiers=qualifiers,
        ))
    return tuple(rows)


def enriched_hierarchy_rows(
    specs: list[dict],
) -> tuple["BomRow", ...]:
    """Build ENRICHED_EBOM BomRow objects for enriched hierarchy tests.

    Each spec dict may contain: ``level`` (str/int), ``component``, ``parent``,
    ``main``, ``sub``, ``vc``, ``level_4``, ``level_5``, ``level_6``.
    No parent or path links are set — those are assigned by
    reconstruct_enriched_hierarchy.

    Args:
        specs: List of dicts describing each enriched row.

    Returns:
        A tuple of frozen BomRow instances in the same order as specs.
    """
    from app.ecn.models import BomKind, BomRowClassification

    rows: list[BomRow] = []
    for index, spec in enumerate(specs):
        row_id = f"20000000-0000-4000-8000-{index:012x}"
        rows.append(make_bom_row(
            bom_row_id=row_id,
            artifact_id=_UUID_BOM_ARTIFACT,
            bom_kind=BomKind.ENRICHED_EBOM,
            sheet_name="Test enriched sheet",
            source_row_number=index + 2,
            dotted_level=str(spec["level"]),
            component_number=spec.get("component"),
            material_number=spec.get("component"),
            parent_material_number=spec.get("parent"),
            main_value=spec.get("main"),
            sub_value=spec.get("sub"),
            vc_value=spec.get("vc"),
            level_4_value=spec.get("level_4"),
            level_5_value=spec.get("level_5"),
            level_6_value=spec.get("level_6"),
            row_classification=BomRowClassification.NOT_APPLICABLE,
        ))
    return tuple(rows)


def upload_publication(
    *,
    bom_reused: bool = False,
    ecn_count: int = 1,
    **kwargs: object,
) -> object:
    """Build a valid ``UploadPublication`` with internally consistent IDs.

    All sub-records reference the same upload and artifact UUIDs so that
    ``_validate_publication`` accepts the result without modification.

    Args:
        bom_reused: When ``True``, ``new_bom_rows`` is empty and the BOM
            artifact is flagged as reused.  When ``False``, one ``BomRow``
            is included and the BOM artifact is new.
        ecn_count: Number of ``EcnPublication`` items to include.
        **kwargs: Any ``UploadPublication`` field values to override.

    Returns:
        A frozen ``UploadPublication`` instance.
    """
    from app.persistence.records import UploadPublication

    upload_id: str = str(kwargs.pop("upload_id", _UUID_UPLOAD))
    request_id: str = str(kwargs.pop("request_id", _UUID_REQUEST))
    ecn_artifact_id: str = _UUID_ARTIFACT
    bom_artifact_id: str = _UUID_BOM_ARTIFACT

    ecn_artifact = make_source_artifact(
        artifact_id=ecn_artifact_id,
        artifact_kind=ArtifactKind.ECN_JSON,
        parse_state=ParseState.VALID,
    )
    bom_artifact = make_source_artifact(
        artifact_id=bom_artifact_id,
        artifact_kind=ArtifactKind.BOM_WORKBOOK,
        parse_state=ParseState.VALID,
    )

    if bom_reused:
        new_bom_rows: tuple = ()
    else:
        new_bom_rows = (
            make_bom_row(artifact_id=bom_artifact_id),
        )

    ecns = tuple(
        _make_ecn_publication_item(i, upload_id, ecn_artifact_id, bom_artifact_id)
        for i in range(ecn_count)
    )

    defaults: dict[str, object] = {
        "upload_id": upload_id,
        "request_id": request_id,
        "ecn_artifact": ecn_artifact,
        "bom_artifact": bom_artifact,
        "new_bom_rows": new_bom_rows,
        "rule_artifact": None,
        "bom_reused": bom_reused,
        "ecns": ecns,
        "actor_id": None,
        "parser_version": PARSER_VERSION,
        "deterministic_logic_version": DETERMINISTIC_LOGIC_VERSION,
        "rule_evaluator_version": RULE_EVALUATOR_VERSION,
        "created_at": _CREATED_AT,
    }
    defaults.update(kwargs)
    return UploadPublication(**defaults)  # type: ignore[arg-type]


def minimal_ecn_json_bytes() -> bytes:
    """Return two ECNs with stable WTPart affected/resulting objects."""
    records = []
    for index, number in enumerate(("ECN100", "ECN200"), start=1):
        records.append(
            {
                "Number": number,
                "Name": f"Synthetic change {index}",
                "DescriptionSummary": f"Change component P{index}00",
                "Description": f"Synthetic evidence for {number}",
                "CNAffectLinks": [
                    {
                        "AffectedObjects": [
                            {
                                "ID": f"OR:wt.part.WTPart:{index}",
                                "VersionID": f"VR:wt.part.WTPart:{index}",
                                "Number": f"P{index}00",
                                "Name": f"Affected part {index}",
                                "ObjectType": "Part",
                                "Revision": "A",
                                "Version": "1",
                            }
                        ]
                    }
                ],
                "ImplementationPlan": [
                    {
                        "ResultingLinks": [
                            {
                                "ResultingObjects": [
                                    {
                                        "ID": f"OR:wt.part.WTPart:{index}",
                                        "VersionID": f"VR:wt.part.WTPart:{index}",
                                        "Number": f"P{index}00",
                                        "Name": f"Resulting part {index}",
                                        "ObjectType": "Part",
                                        "Revision": "B",
                                        "Version": "2",
                                    }
                                ]
                            }
                        ]
                    }
                ],
            }
        )
    return json.dumps(records).encode("utf-8")


def duplicate_id_json_bytes() -> bytes:
    """Return two repeated business IDs plus one missing business ID."""
    records = json.loads(minimal_ecn_json_bytes())
    records[1]["Number"] = records[0]["Number"]
    missing = json.loads(json.dumps(records[0]))
    missing.pop("Number")
    missing["Name"] = "Missing identifier occurrence"
    records.append(missing)
    return json.dumps(records).encode("utf-8")


def minimal_bom_workbook_bytes(component_suffix: str = "") -> bytes:
    """Return one valid workbook containing the three required role signatures."""
    from datetime import date

    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    enriched = workbook.create_sheet("Enriched")
    enriched.append(
        [
            "Level",
            "Post number",
            "Component",
            "Item designation",
            "Component quantity",
            "Quantity unit",
            "N° of modification",
            "Start of validity",
            "Up to modif. Number",
            "Expiration date",
            "Parent",
            "Main",
            "Sub",
            "VC",
            "Level 4",
            "Level 5",
            "Level 6",
            "Alternative",
        ]
    )
    raw = workbook.create_sheet("Raw 130")
    raw.append(
        [
            "Level no.",
            "Item",
            "Component number",
            "Object description",
            "Quantity",
            "Un",
            "Valid from",
            "Valid to",
            "Change No.",
            "Chg No. To",
            "Typ",
            "AltItm",
        ]
    )
    manufacturing = workbook.create_sheet("Manufacturing 130")
    manufacturing.append(
        [
            "Item",
            "Component",
            "Component name",
            "Quantity",
            "N° of modification",
            "Start of validity",
            "at the number of modif.",
            "Expiration date",
            "alternative postal group",
            "Sequence",
            "Employment probability",
        ]
    )

    for index, number in enumerate(("ECN100", "ECN200"), start=1):
        component = f"P{index}00{component_suffix}"
        enriched.append(
            [
                "1",
                str(index * 10),
                component,
                f"Component {index}",
                1,
                "EA",
                number,
                date(2026, 9, 14),
                None,
                None,
                "ROOT",
                "ROOT",
                None,
                None,
                None,
                None,
                None,
                "01",
            ]
        )
        raw.append(
            [
                ".1",
                "150" if index == 1 else "160",
                component,
                f"Component {index}",
                1,
                "EA",
                "14.09.2026",
                None,
                number,
                None,
                "ND",
                "01",
            ]
        )
        manufacturing.append(
            [
                "ROOT",
                component,
                f"Component {index}",
                1,
                number,
                date(2026, 9, 14),
                None,
                None,
                "01",
                index,
                100,
            ]
        )

    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def replacement_bom_workbook_bytes() -> bytes:
    """Return a valid replacement snapshot with distinct component identities."""
    return minimal_bom_workbook_bytes(component_suffix="-R")
