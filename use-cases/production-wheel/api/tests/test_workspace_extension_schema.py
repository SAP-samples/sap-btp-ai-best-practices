"""Regression checks for imports into HANA tables created by earlier extractors."""

import pytest

from app.workspace.repository import HanaRepository, _sql_value


class SchemaConnection:
    """Record schema commands and supply existing column metadata without HANA."""

    def __init__(self, columns):
        """Accept catalog rows and initialize an SQL command log."""
        self.columns = columns
        self.commands = []

    def cursor(self):
        """Return this fixture as its own cursor."""
        return self

    def execute(self, sql, parameters=None):
        """Record commands and optional bound parameters."""
        self.commands.append((sql, parameters))

    def fetchall(self):
        """Return the configured existing schema."""
        return self.columns

    def commit(self):
        """Complete a simulated schema transaction."""

    def close(self):
        """Close a simulated cursor or connection."""


@pytest.mark.parametrize("sqltype,length", [("DOUBLE", 8), ("INTEGER", 4), ("NVARCHAR", 5000), ("NCLOB", 0)])
def test_existing_extension_types_survive_import_and_restart(sqltype, length):
    """Uncatalogued columns retain physical types without destructive ALTER statements."""
    connection = SchemaConnection([
        ("OWNER_ID", "NVARCHAR", 64), ("ROW_ID", "INTEGER", 4),
        ("TOTAL_VOLUME_PER_WEEK", sqltype, length),
    ])
    for _ in range(2):
        repo = HanaRepository(lambda: connection)
        spec = repo._row_spec("legacy_changeover_estimation", [{"total_volume_per_week": 12}])
        expected = f"NVARCHAR({length})" if sqltype == "NVARCHAR" else sqltype
        assert spec["total_volume_per_week"] == expected
        assert _sql_value(12, expected) == (12 if sqltype in ("DOUBLE", "INTEGER") else "12")
        before = len(connection.commands)
        assert repo._row_spec("legacy_changeover_estimation", [{"total_volume_per_week": None}]) == spec
        assert len(connection.commands) == before
    assert all(sql.startswith("SELECT") for sql, _ in connection.commands)


def test_new_extensions_remain_stable_text_and_catalog_mismatches_fail():
    """New unknown evidence stays NCLOB; declared field contracts are not relaxed."""
    connection = SchemaConnection([])
    repo = HanaRepository(lambda: connection)
    assert repo._row_spec("legacy_changeover_estimation", [{"new_evidence": None}])["new_evidence"] == "NCLOB"
    connection = SchemaConnection([("OWNER_ID", "NCLOB", 0)])
    with pytest.raises(RuntimeError, match="incompatible HANA column"):
        HanaRepository(lambda: connection)._row_spec("legacy_changeover_estimation", [])


@pytest.mark.parametrize("warm_cache", [False, True])
def test_excel_errors_widen_numeric_extensions_without_losing_source_text(warm_cache):
    """Mixed source evidence widens safely even after an earlier numeric import."""
    connection = SchemaConnection([
        ("OWNER_ID", "NVARCHAR", 64), ("ROW_ID", "INTEGER", 4),
        ("SEFIS", "DOUBLE", 15),
    ])
    repo = HanaRepository(lambda: connection)
    if warm_cache:
        repo._row_spec("legacy_sales_forecast_summary", [{"sefis": 12}])
    spec = repo._row_spec("legacy_sales_forecast_summary", [{"sefis": 12}, {"sefis": "#REF!"}])
    assert spec["sefis"] == "NVARCHAR(5000)"
    assert _sql_value("#REF!", spec["sefis"]) == "#REF!"
    assert _sql_value(12, spec["sefis"]) == "12"
    assert any('ALTER ("SEFIS" NVARCHAR(5000))' in sql for sql, _ in connection.commands)
