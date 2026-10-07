"""Verify bounded repository reads use owner and point predicates before transfer."""

from contextlib import contextmanager

import pytest

from app.workspace.repository import HanaRepository, MemoryRepository


def test_memory_reads_filter_original_point_indices_before_returning_rows():
    """The memory contract matches typed HANA indices and historical CSV strings."""
    repo = MemoryRepository()
    repo.replace_tables(
        "run",
        {
            "members": [
                {"point_index": 1, "material": "A"},
                {"point_index": "2", "material": "B"},
                {"point_index": 2.0, "material": "C"},
            ]
        },
    )
    assert [
        row["material"] for row in repo.rows("run", "members", {"point_index": 2})
    ] == ["B", "C"]
    assert len(repo.rows("run", "members")) == 3
    with pytest.raises(ValueError):
        repo.rows("run", "members", {"owner_id": "other"})
    with pytest.raises(ValueError):
        repo.rows("run", "members", {"point_index": "1 OR 1=1"})
    with pytest.raises(ValueError):
        repo.rows("dataset", "fini_master", {"point_index": 1})


def test_hana_point_read_is_parameterized_and_owner_scoped(monkeypatch):
    """Only one point is selected in SQL, with caller values bound as parameters."""
    statements = []

    class Cursor:
        """Record executed SQL and return the database's filtered fixture rows."""

        def execute(self, sql, parameters):
            """Capture the parameterized command without connecting to HANA."""
            statements.append((sql, parameters))

        def fetchall(self):
            """Return table columns or the single row matching the SQL predicate."""
            if "SYS.TABLE_COLUMNS" in statements[-1][0]:
                return [("OWNER_ID",), ("ROW_ID",), ("POINT_INDEX",), ("MATERIAL",)]
            return [("run", 1, 7, "F7")]

    @contextmanager
    def cursor():
        """Provide an isolated fake cursor for the repository read path."""
        yield Cursor()

    repo = HanaRepository()
    monkeypatch.setattr(repo, "cursor", cursor)
    assert repo.rows("run", "members", {"point_index": 7}) == [
        {"point_index": 7, "material": "F7"}
    ]
    sql, parameters = statements[-1]
    assert '"OWNER_ID"=? AND "POINT_INDEX"=?' in sql
    assert parameters == ("run", 7)
    assert "7" not in sql
