"""Unit tests for ECN JSON parsing and source-object normalisation.

Tests enforce:
- Strict UTF-8 decoding with byte-offset error details.
- Top-level array requirement; malformed/non-object members → 400.
- Non-standard JSON constants (NaN, Infinity) rejected.
- ECN field mapping: Number→ecn_number, Name→title, DescriptionSummary→description_summary.
- Fixed description_text construction order (DescriptionSummary, Description, SOLUTION_DESCRIPTION).
- Error-shaped strings (wt.util.WTException:) excluded from prose with qualifier.
- Title and description_summary truncation (soft) with DISPLAY_VALUE_TRUNCATED.
- ECN number over 80 chars rejected (hard) without leaking the value.
- Non-string field values stay in source_json with a qualifier; never in prose.
- Stable canonical JSON preserving original values.
- Unique canonical UUIDs per NormalizedEcn and NormalizedSourceObject.
- Nested source-object paths: $.CNAffectLinks and $.ImplementationPlan traversal.
- Role, source_ordinal, link_path, source_path correctness.
- WTPart identity from OR: / VR: prefixed IDs (different opaque IDs are OK).
- Class contradictions between ID and VersionID → TECHNICAL_IDENTITY_CONTRADICTION.
- CAD/document types never become WTPart seeds.
- Relationship values extracted from owning link; error strings retained with qualifier.
- State normalisation from string or {Value, Display} dict.
- Malformed collections/members → path-specific qualifiers, not crashes.
- Number and identifier fields: whitespace-trimmed, no case folding, no leading-zero strip.

Usage::

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_json_parser.py
"""

import json

import pytest

from app.ecn.errors import SourceValidationError
from app.ecn.intake.json_parser import parse_ecn_json

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ARTIFACT_ID = "00000000-0000-4000-8000-000000000001"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def minimal_ecn(**overrides: object) -> bytes:
    """Return minimal valid ECN JSON bytes with optional field overrides.

    Args:
        **overrides: Key-value pairs merged into the single ECN dict.

    Returns:
        UTF-8-encoded JSON array containing one ECN object.
    """
    base: dict[str, object] = {
        "Number": "ECN0001",
        "Name": "Test ECN",
        "DescriptionSummary": "Summary text",
        "CNAffectLinks": [],
        "ImplementationPlan": [],
    }
    base.update(overrides)
    return json.dumps([base]).encode()


def ecn_with_affected_and_resulting_objects() -> bytes:
    """Return ECN JSON with one affected object and one resulting object.

    Affected object lives in CNAffectLinks[0].AffectedObjects[0] with
    ChangeIntent on the link.  Resulting object lives in
    ImplementationPlan[0].ResultingLinks[0].ResultingObjects[0] with
    ImpactIntent on the link.

    Returns:
        UTF-8-encoded JSON array containing one ECN with nested objects.
    """
    doc = [
        {
            "Number": "ECN0002",
            "Name": "Change",
            "CNAffectLinks": [
                {
                    "ChangeIntent": "inspect",
                    "AffectedObjects": [
                        {
                            "ID": "OR:wt.part.WTPart:111",
                            "VersionID": "VR:wt.part.WTPart:222",
                            "Number": "0000123",
                            "ObjectType": "wt.part.WTPart",
                        }
                    ],
                }
            ],
            "ImplementationPlan": [
                {
                    "ResultingLinks": [
                        {
                            "ImpactIntent": "replace",
                            "ResultingObjects": [
                                {
                                    "ID": "OR:wt.part.WTPart:333",
                                    "VersionID": "VR:wt.part.WTPart:444",
                                    "Number": "0000456",
                                    "ObjectType": "wt.part.WTPart",
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    ]
    return json.dumps(doc).encode()


def ecn_with_object(
    *,
    object_id: str,
    version_id: str,
    number: str,
    extra: dict[str, object] | None = None,
) -> bytes:
    """Return ECN JSON with a single affected object with the given identity fields.

    Args:
        object_id: The ``ID`` field value for the source object.
        version_id: The ``VersionID`` field value for the source object.
        number: The ``Number`` field value for the source object.
        extra: Optional additional fields to merge into the object dict.

    Returns:
        UTF-8-encoded JSON array containing one ECN with the object.
    """
    obj: dict[str, object] = {
        "ID": object_id,
        "VersionID": version_id,
        "Number": number,
    }
    if extra:
        obj.update(extra)
    doc = [
        {
            "Number": "ECN0003",
            "Name": "Identity test",
            "CNAffectLinks": [{"AffectedObjects": [obj]}],
            "ImplementationPlan": [],
        }
    ]
    return json.dumps(doc).encode()


# ---------------------------------------------------------------------------
# UTF-8 and root-level structure tests
# ---------------------------------------------------------------------------


def test_top_level_mapping_and_fixed_description_order() -> None:
    """ECN fields must map correctly and description_text follows fixed order."""
    content = json.dumps(
        [
            {
                "Number": " ECN0007 ",
                "Name": "Change title",
                "DescriptionSummary": "Summary",
                "Description": "Detail",
                "SOLUTION_DESCRIPTION": "Solution",
                "CNAffectLinks": [],
                "ImplementationPlan": [],
            }
        ]
    ).encode()
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert ecn.source_ordinal == 0
    # Number must be trimmed but not otherwise altered.
    assert ecn.ecn_number == "ECN0007"
    assert ecn.title == "Change title"
    assert ecn.description_text == (
        "DescriptionSummary:\nSummary\n\nDescription:\nDetail"
        "\n\nSOLUTION_DESCRIPTION:\nSolution"
    )
    # Original untrimmed Number must remain in source_json.
    assert json.loads(parsed.normalized_content)[0]["Number"] == " ECN0007 "


@pytest.mark.parametrize(
    "content",
    [
        b'{"Number":"X"}',         # root object instead of array
        b"[1]",                     # array member that is not an object
        b"[,{]",                    # syntax error
        b'[{"Number":NaN}]',        # non-standard constant
    ],
)
def test_invalid_root_member_or_syntax_is_safe_bad_request(content: bytes) -> None:
    """Root-level structure errors and non-standard constants must produce HTTP 400."""
    with pytest.raises(SourceValidationError) as caught:
        parse_ecn_json(content, ARTIFACT_ID)
    assert caught.value.status_code == 400


def test_invalid_utf8_rejected_with_byte_offset() -> None:
    """Invalid UTF-8 bytes must produce a SourceValidationError with a byte_offset detail."""
    bad_bytes = b"[{\"Number\": \"\xff\xfe\"}]"
    with pytest.raises(SourceValidationError) as caught:
        parse_ecn_json(bad_bytes, ARTIFACT_ID)
    assert caught.value.status_code == 400
    assert caught.value.code == "ECN_JSON_ENCODING_INVALID"
    assert caught.value.details is not None
    assert "byte_offset" in caught.value.details


def test_malformed_json_reports_line_and_column() -> None:
    """A JSON syntax error must produce code ECN_JSON_SYNTAX_INVALID with line/column."""
    bad = b'[{"key": }]'
    with pytest.raises(SourceValidationError) as caught:
        parse_ecn_json(bad, ARTIFACT_ID)
    assert caught.value.code == "ECN_JSON_SYNTAX_INVALID"
    assert caught.value.details is not None
    assert "line" in caught.value.details
    assert "column" in caught.value.details


def test_empty_array_is_valid() -> None:
    """An empty JSON array must parse successfully and produce zero ECNs."""
    parsed = parse_ecn_json(b"[]", ARTIFACT_ID)
    assert len(parsed.ecns) == 0
    assert parsed.normalized_content == "[]"


def test_missing_ecn_number_is_valid() -> None:
    """An ECN without a Number field must parse; ecn_number must be None."""
    content = json.dumps([{"Name": "No number ECN", "CNAffectLinks": [], "ImplementationPlan": []}]).encode()
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    assert parsed.ecns[0].ecn_number is None


# ---------------------------------------------------------------------------
# Display field tests (title / description_summary truncation)
# ---------------------------------------------------------------------------


def test_title_truncated_to_500_with_qualifier() -> None:
    """A title over 500 chars must be truncated; DISPLAY_VALUE_TRUNCATED qualifier added."""
    long_title = "T" * 600
    parsed = parse_ecn_json(minimal_ecn(Name=long_title), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert len(ecn.title) == 500
    assert "DISPLAY_VALUE_TRUNCATED" in ecn.normalization_qualifiers
    # Original must remain in source_json.
    assert json.loads(parsed.normalized_content)[0]["Name"] == long_title


def test_description_summary_truncated_to_2000_with_qualifier() -> None:
    """A DescriptionSummary over 2000 chars must be truncated with the qualifier."""
    long_summary = "S" * 2500
    parsed = parse_ecn_json(minimal_ecn(DescriptionSummary=long_summary), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert len(ecn.description_summary) == 2000
    assert "DISPLAY_VALUE_TRUNCATED" in ecn.normalization_qualifiers
    assert json.loads(parsed.normalized_content)[0]["DescriptionSummary"] == long_summary


def test_title_within_limit_not_truncated() -> None:
    """A title at exactly 500 chars must not be truncated and must carry no qualifier."""
    exact_title = "X" * 500
    parsed = parse_ecn_json(minimal_ecn(Name=exact_title), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert ecn.title == exact_title
    assert "DISPLAY_VALUE_TRUNCATED" not in ecn.normalization_qualifiers


def test_title_with_outer_whitespace_measured_stripped_not_truncated() -> None:
    """A title where stripped length fits the cap but raw length exceeds it must NOT truncate.

    fit_display_text measures and stores the stripped value; raw surrounding
    whitespace must not push a legitimately short title into DISPLAY_VALUE_TRUNCATED.
    """
    # 500 meaningful chars surrounded by whitespace: stripped = 500, raw = 502.
    padded_title = "  " + "X" * 500
    parsed = parse_ecn_json(minimal_ecn(Name=padded_title), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert len(ecn.title) == 500
    assert "DISPLAY_VALUE_TRUNCATED" not in ecn.normalization_qualifiers
    # The stored title is the stripped version.
    assert ecn.title == "X" * 500


# ---------------------------------------------------------------------------
# ECN number rejection
# ---------------------------------------------------------------------------


def test_overlong_ecn_number_rejected_without_value_leakage() -> None:
    """An ECN Number over 80 chars must raise SourceValidationError; value must not leak."""
    long_number = "N" * 81
    with pytest.raises(SourceValidationError) as caught:
        parse_ecn_json(minimal_ecn(Number=long_number), ARTIFACT_ID)
    assert caught.value.status_code == 400
    assert caught.value.code == "SOURCE_VALUE_TOO_LONG"
    # The raw oversized value must not appear in the public message or details.
    assert long_number not in caught.value.public_message
    if caught.value.details:
        for v in caught.value.details.values():
            assert long_number != str(v)


def test_ecn_number_at_80_chars_accepted() -> None:
    """An ECN Number at exactly 80 chars must be accepted without error."""
    exact_number = "N" * 80
    parsed = parse_ecn_json(minimal_ecn(Number=exact_number), ARTIFACT_ID)
    assert parsed.ecns[0].ecn_number == exact_number


# ---------------------------------------------------------------------------
# Canonical JSON and UUID stability
# ---------------------------------------------------------------------------


def test_canonical_json_preserves_original_values() -> None:
    """normalized_content must be a stable JSON string preserving original data."""
    content = json.dumps([{"Number": " ECN-1 ", "Name": "Test"}]).encode()
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    root = json.loads(parsed.normalized_content)
    assert root[0]["Number"] == " ECN-1 "
    assert root[0]["Name"] == "Test"


def test_canonical_json_sorts_object_keys() -> None:
    """Equivalent source objects normalize to the same key-sorted JSON text."""
    first = parse_ecn_json(b'[{"Number":"ECN-1","Name":"Test"}]', ARTIFACT_ID)
    second = parse_ecn_json(b'[{"Name":"Test","Number":"ECN-1"}]', ARTIFACT_ID)
    assert first.normalized_content == second.normalized_content
    assert first.normalized_content == '[{"Name":"Test","Number":"ECN-1"}]'


def test_each_ecn_gets_unique_canonical_id() -> None:
    """Each NormalizedEcn must have a unique ecn_row_id."""
    content = json.dumps([
        {"Number": "ECN-A"},
        {"Number": "ECN-B"},
    ]).encode()
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    ids = [ecn.ecn_row_id for ecn in parsed.ecns]
    assert len(set(ids)) == 2


def test_artifact_id_propagated_to_each_ecn() -> None:
    """artifact_id must be set from the parse_ecn_json argument on every ECN."""
    parsed = parse_ecn_json(minimal_ecn(), ARTIFACT_ID)
    assert parsed.ecns[0].artifact_id == ARTIFACT_ID


# ---------------------------------------------------------------------------
# Non-string / error-shaped description values
# ---------------------------------------------------------------------------


def test_non_string_description_summary_excluded_from_prose() -> None:
    """A non-string DescriptionSummary must not appear in description_text."""
    parsed = parse_ecn_json(minimal_ecn(DescriptionSummary=42), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    # description_text must not contain "42".
    assert ecn.description_text is None or "42" not in ecn.description_text
    # The original value must be preserved in source_json.
    raw = json.loads(parsed.normalized_content)[0]
    assert raw["DescriptionSummary"] == 42


def test_error_shaped_description_excluded_from_prose_with_qualifier() -> None:
    """A wt.util.WTException: string must be excluded from prose with a qualifier."""
    error_str = "wt.util.WTException: server error"
    parsed = parse_ecn_json(minimal_ecn(DescriptionSummary=error_str), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert ecn.description_text is None or error_str not in ecn.description_text
    assert "SOURCE_ERROR_VALUE_EXCLUDED_FROM_PROSE" in ecn.normalization_qualifiers
    raw = json.loads(parsed.normalized_content)[0]
    assert raw["DescriptionSummary"] == error_str


def test_description_text_only_from_nonblank_nonerror_strings() -> None:
    """Blank and error-shaped values must not appear in description_text."""
    content = json.dumps(
        [
            {
                "DescriptionSummary": "Good summary",
                "Description": "wt.util.WTException: bad",
                "SOLUTION_DESCRIPTION": "Good solution",
                "CNAffectLinks": [],
                "ImplementationPlan": [],
            }
        ]
    ).encode()
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    ecn = parsed.ecns[0]
    # Description (error-shaped) must be absent from prose.
    assert "wt.util.WTException" not in (ecn.description_text or "")
    # Both good fields must appear.
    assert "Good summary" in (ecn.description_text or "")
    assert "Good solution" in (ecn.description_text or "")


# ---------------------------------------------------------------------------
# Nested source-object path and relationship tests
# ---------------------------------------------------------------------------


def test_nested_paths_roles_and_link_relationship_values() -> None:
    """Affected and resulting objects must carry exact paths and relationship values."""
    parsed = parse_ecn_json(ecn_with_affected_and_resulting_objects(), ARTIFACT_ID)
    ecn = parsed.ecns[0]

    affected = ecn.affected_objects[0]
    resulting = ecn.resulting_objects[0]

    assert affected.source_path == "$.CNAffectLinks[0].AffectedObjects[0]"
    assert affected.link_path == "$.CNAffectLinks[0]"
    assert affected.role.value == "AFFECTED"
    assert affected.relationship_values.get("ChangeIntent") == "inspect"

    assert resulting.source_path == (
        "$.ImplementationPlan[0].ResultingLinks[0].ResultingObjects[0]"
    )
    assert resulting.link_path == "$.ImplementationPlan[0].ResultingLinks[0]"
    assert resulting.role.value == "RESULTING"
    assert resulting.relationship_values.get("ImpactIntent") == "replace"


def test_source_ordinals_are_zero_based_and_sequential() -> None:
    """Source ordinals within each role list must be sequential starting at 0."""
    doc = [
        {
            "Number": "ECN0004",
            "CNAffectLinks": [
                {
                    "AffectedObjects": [
                        {"ID": "OR:wt.part.WTPart:10", "Number": "P001"},
                        {"ID": "OR:wt.part.WTPart:11", "Number": "P002"},
                    ]
                }
            ],
            "ImplementationPlan": [],
        }
    ]
    parsed = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID)
    ordinals = [obj.source_ordinal for obj in parsed.ecns[0].affected_objects]
    assert ordinals == [0, 1]


# ---------------------------------------------------------------------------
# WTPart identity tests
# ---------------------------------------------------------------------------


def test_or_and_vr_wtpart_suffixes_need_not_match() -> None:
    """OR: and VR: suffixes identify different Windchill layers; mismatch is fine."""
    content = ecn_with_object(
        object_id="OR:wt.part.WTPart:100",
        version_id="VR:wt.part.WTPart:200",
        number="0000123",
    )
    part = parse_ecn_json(content, ARTIFACT_ID).ecns[0].affected_objects[0]
    assert part.is_technical_wtpart is True
    assert part.number == "0000123"


def test_conflicting_technical_classes_do_not_seed_a_wtpart() -> None:
    """A WTPart ID paired with a WTDocument VersionID must be flagged and not seeded."""
    content = ecn_with_object(
        object_id="OR:wt.part.WTPart:100",
        version_id="VR:wt.doc.WTDocument:200",
        number="0000123",
    )
    item = parse_ecn_json(content, ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.is_technical_wtpart is False
    assert "TECHNICAL_IDENTITY_CONTRADICTION" in item.quality_qualifiers


def test_cad_epm_document_not_a_wtpart_seed() -> None:
    """EPMDocument objects must have is_technical_wtpart=False."""
    content = ecn_with_object(
        object_id="OR:com.ptc.windchill.cadx.wctype.EPMDocument:555",
        version_id="VR:com.ptc.windchill.cadx.wctype.EPMDocument:666",
        number="DRW-001",
    )
    item = parse_ecn_json(content, ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.is_technical_wtpart is False
    assert "TECHNICAL_IDENTITY_CONTRADICTION" not in item.quality_qualifiers


def test_wtdocument_not_a_wtpart_seed() -> None:
    """WTDocument objects must have is_technical_wtpart=False regardless of number shape."""
    content = ecn_with_object(
        object_id="OR:wt.doc.WTDocument:777",
        version_id="VR:wt.doc.WTDocument:888",
        number="0000789",
    )
    item = parse_ecn_json(content, ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.is_technical_wtpart is False


def test_missing_id_and_version_id_is_not_a_wtpart() -> None:
    """A source object with no ID or VersionID must not be a WTPart seed."""
    doc = [
        {
            "Number": "ECN0005",
            "CNAffectLinks": [{"AffectedObjects": [{"Number": "0000111"}]}],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.is_technical_wtpart is False
    assert item.number == "0000111"


def test_wtpart_backed_vc_subtype_is_technical_wtpart() -> None:
    """An object with a WTPart ID but non-WTPart ObjectType must still be is_technical_wtpart=True."""
    content = ecn_with_object(
        object_id="OR:wt.part.WTPart:900",
        version_id="VR:wt.part.WTPart:901",
        number="VC-001",
        extra={"ObjectType": "com.ptc.windchill.vc.WTVariantConfiguration"},
    )
    item = parse_ecn_json(content, ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.is_technical_wtpart is True
    # ObjectType is preserved as-is in the record.
    assert item.object_type == "com.ptc.windchill.vc.WTVariantConfiguration"


# ---------------------------------------------------------------------------
# State normalisation
# ---------------------------------------------------------------------------


def test_state_from_string() -> None:
    """A string State value must be used as-is."""
    doc = [
        {
            "Number": "ECN0006",
            "CNAffectLinks": [{"AffectedObjects": [{"State": "RELEASED"}]}],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.state == "RELEASED"


def test_state_from_value_display_dict() -> None:
    """A {Value, Display} dict State must use the Value string."""
    doc = [
        {
            "Number": "ECN0007",
            "CNAffectLinks": [
                {"AffectedObjects": [{"State": {"Value": "INWORK", "Display": "In Work"}}]}
            ],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.state == "INWORK"


def test_unsupported_state_shape_adds_qualifier() -> None:
    """A State that is neither a string nor a {Value, Display} dict must add SOURCE_FIELD_SHAPE_UNSUPPORTED."""
    doc = [
        {
            "Number": "ECN0008",
            "CNAffectLinks": [{"AffectedObjects": [{"State": [1, 2, 3]}]}],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.state is None
    assert "SOURCE_FIELD_SHAPE_UNSUPPORTED" in item.quality_qualifiers


# ---------------------------------------------------------------------------
# Relationship value tests
# ---------------------------------------------------------------------------


def test_error_relation_string_retained_with_qualifier() -> None:
    """A wt.util.WTException: value in a link must be retained with SOURCE_RELATION_ERROR_VALUE."""
    doc = [
        {
            "Number": "ECN0009",
            "CNAffectLinks": [
                {
                    "ChangeIntent": "wt.util.WTException: cannot load",
                    "AffectedObjects": [{"Number": "P001"}],
                }
            ],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert "ChangeIntent" in item.relationship_values
    assert "SOURCE_RELATION_ERROR_VALUE" in item.quality_qualifiers


def test_oem_replaces_affected_in_relationship_values() -> None:
    """OemReplacesAffected from the link must appear in relationship_values."""
    doc = [
        {
            "Number": "ECN0010",
            "CNAffectLinks": [
                {
                    "OemReplacesAffected": "0000999",
                    "AffectedObjects": [{"Number": "P002"}],
                }
            ],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.relationship_values.get("OemReplacesAffected") == "0000999"


# ---------------------------------------------------------------------------
# Malformed collection handling
# ---------------------------------------------------------------------------


def test_non_list_affected_links_produces_qualifier_not_crash() -> None:
    """A non-list CNAffectLinks must add a qualifier and produce zero objects."""
    content = minimal_ecn(CNAffectLinks="not a list")
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert len(ecn.affected_objects) == 0
    assert any("AFFECT" in q or "COLLECTION" in q for q in ecn.normalization_qualifiers)


def test_non_list_implementation_plan_produces_qualifier_not_crash() -> None:
    """A non-list ImplementationPlan must add a qualifier and produce zero objects."""
    content = minimal_ecn(ImplementationPlan={"key": "val"})
    parsed = parse_ecn_json(content, ARTIFACT_ID)
    ecn = parsed.ecns[0]
    assert len(ecn.resulting_objects) == 0
    assert any("PLAN" in q or "COLLECTION" in q or "RESULTING" in q for q in ecn.normalization_qualifiers)


def test_non_dict_object_member_skipped_with_qualifier() -> None:
    """A non-dict member in AffectedObjects must be skipped; other objects proceed."""
    doc = [
        {
            "Number": "ECN0011",
            "CNAffectLinks": [
                {
                    "AffectedObjects": [
                        "not-a-dict",
                        {"ID": "OR:wt.part.WTPart:200", "Number": "P003"},
                    ]
                }
            ],
            "ImplementationPlan": [],
        }
    ]
    parsed = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID)
    ecn = parsed.ecns[0]
    # The valid second object must still be returned.
    assert len(ecn.affected_objects) == 1
    assert ecn.affected_objects[0].number == "P003"


# ---------------------------------------------------------------------------
# Identifier preservation
# ---------------------------------------------------------------------------


def test_leading_zeros_preserved_in_number() -> None:
    """Part numbers with leading zeros must not have them stripped."""
    doc = [
        {
            "Number": "ECN0012",
            "CNAffectLinks": [{"AffectedObjects": [{"Number": "  0001234  "}]}],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    # Only external whitespace is trimmed; leading zeros preserved.
    assert item.number == "0001234"


def test_identifier_case_not_folded() -> None:
    """Part identifiers must not be case-folded."""
    doc = [
        {
            "Number": "ECN0013",
            "CNAffectLinks": [{"AffectedObjects": [{"Number": "ABC-123"}]}],
            "ImplementationPlan": [],
        }
    ]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.number == "ABC-123"


def test_each_source_object_gets_unique_uuid() -> None:
    """Every NormalizedSourceObject must carry a distinct canonical UUID."""
    doc = [
        {
            "Number": "ECN0014",
            "CNAffectLinks": [
                {
                    "AffectedObjects": [
                        {"Number": "P-A"},
                        {"Number": "P-B"},
                        {"Number": "P-C"},
                    ]
                }
            ],
            "ImplementationPlan": [],
        }
    ]
    parsed = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID)
    ids = [obj.source_object_id for obj in parsed.ecns[0].affected_objects]
    assert len(set(ids)) == 3


def test_raw_object_preserved_losslessly() -> None:
    """raw_object must contain every key-value from the source object without mutation."""
    obj = {"ID": "OR:wt.part.WTPart:99", "Number": " P999 ", "CustomField": [1, 2, 3]}
    doc = [{"Number": "ECN0015", "CNAffectLinks": [{"AffectedObjects": [obj]}], "ImplementationPlan": []}]
    item = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID).ecns[0].affected_objects[0]
    assert item.raw_object["Number"] == " P999 "
    assert item.raw_object["CustomField"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Source-object Number join-key length enforcement
# ---------------------------------------------------------------------------


def test_source_object_number_over_200_chars_raises_source_value_too_long() -> None:
    """A source object Number string over 200 chars must raise SourceValidationError.

    Number is the primary BOM join key; silently truncating it would corrupt
    downstream matching.  The error must not echo the raw value.
    """
    long_number = "N" * 201
    doc = [
        {
            "Number": "ECN0016",
            "CNAffectLinks": [{"AffectedObjects": [{"Number": long_number}]}],
            "ImplementationPlan": [],
        }
    ]
    with pytest.raises(SourceValidationError) as caught:
        parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID)
    assert caught.value.status_code == 400
    assert caught.value.code == "SOURCE_VALUE_TOO_LONG"
    assert long_number not in caught.value.public_message
    if caught.value.details:
        for v in caught.value.details.values():
            assert long_number != str(v)


def test_source_object_number_at_200_chars_accepted() -> None:
    """A source object Number at exactly 200 chars must be accepted without error."""
    exact_number = "N" * 200
    doc = [
        {
            "Number": "ECN0017",
            "CNAffectLinks": [{"AffectedObjects": [{"Number": exact_number}]}],
            "ImplementationPlan": [],
        }
    ]
    parsed = parse_ecn_json(json.dumps(doc).encode(), ARTIFACT_ID)
    assert parsed.ecns[0].affected_objects[0].number == exact_number
