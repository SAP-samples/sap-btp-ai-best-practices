"""Repository tests for atomic reference publication and active selection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.ecn.errors import PersistenceStateError
from app.ecn.reference_data.models import ReferenceDatasetType
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository

from tests.fixtures.persistence import RecordingEngine


def _row(
    dataset_id: str,
    dataset_type: str,
    publication_order: int,
    scope_code: str | None,
) -> dict[str, object]:
    """Build one joined metadata row returned by the repository list query."""
    return {
        "DATASET_ID": dataset_id,
        "DATASET_TYPE": dataset_type,
        "ARTIFACT_ID": "00000000-0000-4000-8000-000000000299",
        "PUBLICATION_ORDER": publication_order,
        "SCOPE_CODE": scope_code,
        "SCOPE_NAME": None,
        "ROW_COUNT": 10,
        "WARNING_COUNT": 0,
        "WARNING_JSON": "{}",
        "PARSER_VERSION": "ecn-reference-v1",
        "CREATED_BY": None,
        "CREATED_AT": datetime(2026, 9, 15, tzinfo=timezone.utc).replace(tzinfo=None),
        "FILE_NAME": "source.xlsx",
        "SHA256": "a" * 64,
    }


def test_list_marks_latest_publication_active_per_type_and_scope() -> None:
    """Keep separate scope publications eligible while superseding only exact scope."""
    engine = RecordingEngine()
    engine.queue_mappings(
        [
            _row("00000000-0000-4000-8000-000000000301", "MATERIAL_MASTER", 3, "DE10"),
            _row("00000000-0000-4000-8000-000000000302", "MATERIAL_MASTER", 2, "PL10"),
            _row("00000000-0000-4000-8000-000000000303", "MATERIAL_MASTER", 1, "PL10"),
        ]
    )
    records = ReferenceDatasetRepository(engine).list()  # type: ignore[arg-type]
    assert [record.active for record in records] == [True, True, False]
    assert records[0].dataset_type is ReferenceDatasetType.MATERIAL_MASTER
    assert "CONTENT_BYTES" not in engine.executions[0][0]


def test_list_tolerates_lowercase_hana_result_keys() -> None:
    """Decode the lowercase mapping keys returned by sqlalchemy-hana."""
    engine = RecordingEngine()
    engine.queue_mappings(
        [
            {
                key.lower(): value
                for key, value in _row(
                    "00000000-0000-4000-8000-000000000304",
                    "MATERIAL_MASTER",
                    1,
                    None,
                ).items()
            }
        ]
    )

    records = ReferenceDatasetRepository(engine).list()  # type: ignore[arg-type]

    assert len(records) == 1
    assert records[0].dataset_type is ReferenceDatasetType.MATERIAL_MASTER
    assert records[0].row_count == 10
    assert records[0].active is True


@pytest.mark.parametrize(
    ("dataset_type", "row_table"),
    [
        ("BOM", "BOM_ROW"),
        ("CONDITIONAL_RULES", "CONDITIONAL_RULE_ROW"),
        ("STOCK", "STOCK_ROW"),
        ("MATERIAL_MASTER", "MATERIAL_MASTER_ROW"),
        ("PURCHASE_INFO_RECORDS", "PURCHASE_INFO_ROW"),
    ],
)
def test_delete_removes_rows_publication_and_file_in_one_transaction(
    dataset_type: str, row_table: str
) -> None:
    """Delete every supported workbook type in foreign-key-safe order."""
    engine = RecordingEngine()
    dataset_id = "00000000-0000-4000-8000-000000000305"
    engine.queue_mappings([{"DATASET_TYPE": dataset_type, "ARTIFACT_ID": "00000000-0000-4000-8000-000000000299"}])
    engine.queue_scalar(0)
    engine.queue_scalar(0)
    engine.queue_mappings([])

    assert ReferenceDatasetRepository(engine).delete(dataset_id) is True  # type: ignore[arg-type]

    assert engine.begin_count == 1
    deletes = [statement for statement, _ in engine.executions if statement.startswith("DELETE")]
    expected_tables = [row_table, "REFERENCE_DATASET", "SOURCE_ARTIFACT"]
    assert expected_tables == [
        next(name for name in expected_tables if name in statement)
        for statement in deletes
    ]
    assert all(params.get("dataset_id") == dataset_id for statement, params in engine.executions if ":dataset_id" in statement)


@pytest.mark.parametrize(("pin_count", "direct_count"), [(1, 0), (0, 1)])
def test_delete_rejects_dataset_used_by_an_upload_before_any_write(
    pin_count: int, direct_count: int
) -> None:
    """Keep workbook rows for both explicit pins and direct artifact use."""
    engine = RecordingEngine()
    engine.queue_mappings([{"DATASET_TYPE": "STOCK", "ARTIFACT_ID": "00000000-0000-4000-8000-000000000299"}])
    engine.queue_scalar(pin_count)
    engine.queue_scalar(direct_count)

    with pytest.raises(PersistenceStateError) as error:
        ReferenceDatasetRepository(engine).delete("00000000-0000-4000-8000-000000000305")  # type: ignore[arg-type]

    assert error.value.code == "REFERENCE_DATASET_IN_USE"
    assert engine.rollback_count == 1
    assert not any(statement.startswith("DELETE") for statement, _ in engine.executions)


def test_delete_rejects_comparison_baseline_before_any_write() -> None:
    """Preserve a dataset referenced by another version's comparison context."""
    engine = RecordingEngine()
    dataset_id = "00000000-0000-4000-8000-000000000305"
    engine.queue_mappings([{"DATASET_TYPE": "MATERIAL_MASTER", "ARTIFACT_ID": "00000000-0000-4000-8000-000000000299"}])
    engine.queue_scalar(0)
    engine.queue_scalar(0)
    engine.queue_mappings([{"DATASET_ID": "00000000-0000-4000-8000-000000000306", "CONTEXT_JSON": f'{{"comparison_dataset_id":"{dataset_id}"}}'}])

    with pytest.raises(PersistenceStateError) as error:
        ReferenceDatasetRepository(engine).delete(dataset_id)  # type: ignore[arg-type]

    assert error.value.code == "REFERENCE_DATASET_COMPARISON_IN_USE"
    assert not any(statement.startswith("DELETE") for statement, _ in engine.executions)
