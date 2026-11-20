"""Source parsing and target resolution for outgoing ECN navigation."""

from dataclasses import replace
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from app.ecn.linked_ecns import extract_linked_ecn_numbers, normalize_ecn_number
from app.persistence.ecn_repository import EcnRepository
from tests.fixtures.persistence import RecordingEngine
from tests.unit.test_ecn_repository import ecn_detail_record


@pytest.mark.parametrize("reference", [
    "ECN12345", "ecn12345", "EcN12345", "ECN 12345", "ECN\t12345",
    "ECN\n12345", "ECN\u00a012345", "ECN-12345", "ECN – 12345",
    "ECN—12345", "ECN‑12345", "ECN_12345", "ECN: 12345", "ECN #12345",
    "ECNs 12345", "ecns12345", "ECN No. 12345", "ECN no 12345",
    "ECN number: 12345", "ECN0012345",
])
def test_extracts_explicit_case_spacing_and_punctuation_variants(reference):
    expected = "ECN0012345" if "0012345" in reference else "ECN12345"
    assert extract_linked_ecn_numbers(f"Linked to ({reference}).") == [expected]
    assert normalize_ecn_number(f" {reference} ") == expected


@pytest.mark.parametrize("text", [
    None, "", "No linked ECN.", "Part 12345.", "ECN", "ECN-",
    "XECN12345", "part_ECN12345", "ECN12345A", "ECN12345_ABC",
    "ECN12345abc678", "ECN12.5", "ECN12,500", "ECN" + "1" * 78,
])
def test_does_not_invent_links_or_accept_partial_identifiers(text):
    assert extract_linked_ecn_numbers(text) == []


def test_deduplicates_sources_excludes_self_and_preserves_leading_zeroes():
    assert extract_linked_ecn_numbers(
        "ECN999 links to ECN12345, ecn-12345 and ECN 00123.",
        "Follow ECNs: 12345 / ECN67890.",
        current_ecn_number="ecn-999",
    ) == ["ECN12345", "ECN00123", "ECN67890"]


def test_a_reference_on_the_new_ecn_does_not_create_a_reverse_link():
    assert extract_linked_ecn_numbers("Original change", current_ecn_number="ECN12345") == []
    assert extract_linked_ecn_numbers("Linked to ECN12345", current_ecn_number="ECN67890") == ["ECN12345"]


def test_resolves_all_references_with_one_query_and_keeps_missing_targets():
    engine = RecordingEngine()
    base = ecn_detail_record(ecn_number="ECN999")
    record = replace(base, ecn=replace(base.ecn,
        description_text="Linked to ECN12345, ECN888 and ECN00123; see ecn-12345.",
        description_summary="ECN999",))
    preferred = "00000000-0000-4000-8000-000000000041"
    duplicate = "00000000-0000-4000-8000-000000000042"
    zeroes = "00000000-0000-4000-8000-000000000043"
    engine.queue_mappings([
        {"ECN_NUMBER": "ecn - 12345", "ECN_ROW_ID": preferred},
        {"ECN_NUMBER": "ECN12345", "ECN_ROW_ID": duplicate},
        {"ECN_NUMBER": "ECN-00123", "ECN_ROW_ID": zeroes},
        {"ECN_NUMBER": "ECN0012345", "ECN_ROW_ID": duplicate},
    ])
    references = EcnRepository(engine).get_linked_ecns(record)
    assert [(item.ecn_number, item.ecn_row_id) for item in references] == [
        ("ECN12345", UUID(preferred)), ("ECN888", None), ("ECN00123", UUID(zeroes)),
    ]
    assert engine.connect_count == 1
    sql, parameters = engine.executions[0]
    assert 'EVIDENCE_JSON' not in sql
    assert "ECN12345" not in sql
    assert parameters["linked_0"] == "ECN%12345"
    assert parameters["current_row_id"] == record.ecn.ecn_row_id
    assert parameters["current_upload_id"] == record.upload.upload_id
    assert 'CASE WHEN E."UPLOAD_ID" = :current_upload_id' in sql
    assert 'U."PUBLICATION_ORDER" DESC' in sql


def test_no_reference_needs_no_target_query():
    engine = RecordingEngine()
    assert EcnRepository(engine).get_linked_ecns(ecn_detail_record()) == []
    assert engine.connect_count == 0


def test_target_query_selects_same_upload_then_latest_and_rejects_similar_numbers():
    """Execute the actual query across duplicates and formatting variants."""
    engine = create_engine("sqlite://")
    base = ecn_detail_record(ecn_number="ECN999")
    record = replace(base, ecn=replace(base.ecn,
        description_text="ECN12345, ECN00123, ECN67890 and ECN888",))
    older = "00000000-0000-4000-8000-000000000021"
    newer = "00000000-0000-4000-8000-000000000022"
    rows = [
        ("00000000-0000-4000-8000-000000000041", record.upload.upload_id, 2, "ecn - 12345"),
        ("00000000-0000-4000-8000-000000000042", newer, 0, "ECN12345"),
        ("00000000-0000-4000-8000-000000000043", older, 0, "ECN00123"),
        ("00000000-0000-4000-8000-000000000044", newer, 1, "ECN00123"),
        ("00000000-0000-4000-8000-000000000045", newer, 2, "ECNs No. #67890"),
        ("00000000-0000-4000-8000-000000000046", newer, 3, "ECN0012345"),
        ("00000000-0000-4000-8000-000000000047", newer, 4, "ECNPART888"),
        ("00000000-0000-4000-8000-000000000048", newer, 5, "ECN8888"),
    ]
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE ECN_IMPACT_UPLOAD (UPLOAD_ID TEXT, PUBLICATION_ORDER INTEGER)'))
        connection.execute(text('CREATE TABLE ECN_IMPACT_ECN (ECN_ROW_ID TEXT, UPLOAD_ID TEXT, SOURCE_ORDINAL INTEGER, ECN_NUMBER TEXT)'))
        connection.execute(text('INSERT INTO ECN_IMPACT_UPLOAD VALUES (:id, :publication)'), [
            {"id": record.upload.upload_id, "publication": 1},
            {"id": older, "publication": 2}, {"id": newer, "publication": 3},
        ])
        connection.execute(text('INSERT INTO ECN_IMPACT_ECN VALUES (:id, :upload, :ordinal, :number)'), [
            {"id": row_id, "upload": upload, "ordinal": ordinal, "number": number}
            for row_id, upload, ordinal, number in rows
        ])
    references = EcnRepository(engine).get_linked_ecns(record)
    assert [(item.ecn_number, str(item.ecn_row_id) if item.ecn_row_id else None) for item in references] == [
        ("ECN12345", rows[0][0]), ("ECN00123", rows[3][0]),
        ("ECN67890", rows[4][0]), ("ECN888", None),
    ]
    engine.dispose()
