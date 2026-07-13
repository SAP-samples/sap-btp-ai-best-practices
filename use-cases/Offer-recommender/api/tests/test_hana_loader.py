from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.nbo.config import COL_BIZ_OFFERING_NAME, COL_RATE_PLAN, TABLE_ACTIVE_OFFERING
from app.nbo.hana import DATASET_TABLES
from app.nbo.hana_loader import load_seed_datasets, recreate_and_load_tables
from synthetic_runtime import synthetic_runtime_datasets


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        self.executemany_calls.append((sql, rows))

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True


def _write_seed_workbooks(tmp_path: Path) -> tuple[Path, Path]:
    """Write explicit anonymized seed workbooks for loader tests."""
    datasets = synthetic_runtime_datasets()
    customer_path = tmp_path / "customer_seed.xlsx"
    program_path = tmp_path / "program_seed.xlsx"

    with pd.ExcelWriter(customer_path) as writer:
        datasets["residential"].to_excel(writer, sheet_name="Residential", index=False)
        datasets["res_segment"].to_excel(
            writer,
            sheet_name="Residential_Residential_Segment",
            index=False,
        )
        datasets["commercial"].to_excel(writer, sheet_name="Commercial", index=False)
        datasets["comm_segment"].to_excel(
            writer,
            sheet_name="Commercial_Residential_Segment",
            index=False,
        )
        datasets["active_offering"].to_excel(
            writer,
            sheet_name="Active business offering",
            index=False,
        )

    with pd.ExcelWriter(program_path) as writer:
        datasets["program_contract"].to_excel(
            writer,
            sheet_name="Program Contract",
            index=False,
            startrow=1,
        )
        datasets["program_samples"].to_excel(
            writer,
            sheet_name="Sample Accounts",
            index=False,
        )

    return customer_path, program_path


def test_load_seed_datasets_preserves_sheet_mapping_and_contract_header(tmp_path: Path) -> None:
    customer_path, program_path = _write_seed_workbooks(tmp_path)
    datasets = load_seed_datasets(
        customer_workbook=customer_path,
        program_codes_workbook=program_path,
    )

    assert set(datasets) == set(DATASET_TABLES)
    assert "Program/Contract" in datasets["program_contract"].columns
    assert len(datasets["residential"]) > 0
    assert len(datasets["commercial"]) > 0
    assert DATASET_TABLES["active_offering"] == TABLE_ACTIVE_OFFERING
    assert COL_RATE_PLAN in datasets["active_offering"].columns
    assert COL_BIZ_OFFERING_NAME in datasets["active_offering"].columns
    assert len(datasets["active_offering"]) > 0
    assert len(datasets["program_samples"]) > 0


def test_recreate_and_load_tables_reports_expected_row_counts() -> None:
    datasets = synthetic_runtime_datasets()
    connection = FakeConnection()

    row_counts = recreate_and_load_tables(connection, datasets)

    assert connection.committed is True
    assert row_counts == {
        table_name: len(datasets[dataset_name])
        for dataset_name, table_name in DATASET_TABLES.items()
    }
    assert len(connection.cursor_instance.executemany_calls) == sum(
        1 for dataset_name in DATASET_TABLES if len(datasets[dataset_name]) > 0
    )
