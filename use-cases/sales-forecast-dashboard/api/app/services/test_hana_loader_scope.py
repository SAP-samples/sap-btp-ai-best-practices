"""Regression tests for memory-safe MODEL_B HANA loading."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd


def install_hana_loader_import_fakes(monkeypatch) -> None:
    """Install the minimal dependency modules needed to load hana_loader.py."""
    hana_ml_module = types.ModuleType("hana_ml")
    hana_ml_module.ConnectionContext = object

    app_package = types.ModuleType("app")
    app_package.__path__ = []
    services_package = types.ModuleType("app.services")
    services_package.__path__ = []
    memory_module = types.ModuleType("app.services.memory_monitor")
    memory_module.collect_dataframe_with_memory_logging = (
        lambda label, collect_func: collect_func()
    )

    monkeypatch.setitem(sys.modules, "hana_ml", hana_ml_module)
    monkeypatch.setitem(sys.modules, "app", app_package)
    monkeypatch.setitem(sys.modules, "app.services", services_package)
    monkeypatch.setitem(sys.modules, "app.services.memory_monitor", memory_module)


def load_hana_loader_module(monkeypatch):
    """Load hana_loader.py directly so tests do not import the full agent package."""
    install_hana_loader_import_fakes(monkeypatch)
    root = Path(__file__).parents[1]
    module_path = root / "agent" / "hana_loader.py"
    spec = importlib.util.spec_from_file_location("hana_loader_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_model_b_columns_renames_input_in_place(monkeypatch) -> None:
    """MODEL_B normalization must not allocate a second full DataFrame."""
    module = load_hana_loader_module(monkeypatch)
    df = pd.DataFrame(
        {
            "CONVERSIONRATE_LAG_1": [0.1, 0.2],
            "AOV_ROLL_MEAN_8": [40.0, 42.0],
            "HORIZON": [1, 2],
        }
    )

    normalized = module._normalize_model_b_columns(df)

    assert normalized is df
    assert normalized.columns.tolist() == [
        "ConversionRate_lag_1",
        "AOV_roll_mean_8",
        "horizon",
    ]


def test_origin_summary_query_contains_complete_scope(monkeypatch) -> None:
    """Origin metadata must be aggregated in HANA using the requested scope."""
    module = load_hana_loader_module(monkeypatch)
    queries: list[str] = []
    labels: list[str | None] = []

    def fake_query_to_dataframe(
        query: str,
        label: str | None = None,
    ) -> pd.DataFrame:
        """Record aggregate SQL and return one HANA-shaped result row."""
        queries.append(query)
        labels.append(label)
        return pd.DataFrame(
            {
                "ORIGIN_WEEK_DATE": ["2025-07-07"],
                "MAX_TARGET_WEEK_DATE": ["2025-09-29"],
                "HORIZON_COUNT": [13],
            }
        )

    monkeypatch.setattr(module, "query_to_dataframe", fake_query_to_dataframe)

    result = module.load_model_b_origin_summary(
        profit_center_nbrs=[63, 62],
        channel="B&M",
        max_horizon=13,
    )

    sql = " ".join(queries[0].split()).upper()
    assert 'FROM "AICOE"."MODEL_B"' in sql
    assert "PROFIT_CENTER_NBR IN (63,62)" in sql
    assert "CHANNEL = 'B&M'" in sql
    assert "HORIZON >= 1" in sql
    assert "HORIZON <= 13" in sql
    assert "COUNT(DISTINCT HORIZON)" in sql
    assert "GROUP BY ORIGIN_WEEK_DATE" in sql
    assert labels == [
        "MODEL_B_ORIGIN_SUMMARY scope=store_ids;channel;max_horizon"
    ]
    assert result.columns.tolist() == [
        "origin_week_date",
        "max_target_week_date",
        "horizon_count",
    ]


def test_store_recency_query_groups_requested_stores(monkeypatch) -> None:
    """Store staleness must use a two-column aggregate rather than feature history."""
    module = load_hana_loader_module(monkeypatch)
    queries: list[str] = []
    labels: list[str | None] = []

    def fake_query_to_dataframe(
        query: str,
        label: str | None = None,
    ) -> pd.DataFrame:
        """Record aggregate SQL and return store-recency rows."""
        queries.append(query)
        labels.append(label)
        return pd.DataFrame(
            {
                "PROFIT_CENTER_NBR": [63],
                "LAST_ORIGIN_WEEK_DATE": ["2025-07-14"],
            }
        )

    monkeypatch.setattr(module, "query_to_dataframe", fake_query_to_dataframe)

    result = module.load_model_b_last_origin_by_store(
        profit_center_nbrs=[63],
        channel="B&M",
    )

    sql = " ".join(queries[0].split()).upper()
    assert "SELECT PROFIT_CENTER_NBR" in sql
    assert "MAX(ORIGIN_WEEK_DATE) AS LAST_ORIGIN_WEEK_DATE" in sql
    assert "PROFIT_CENTER_NBR IN (63)" in sql
    assert "GROUP BY PROFIT_CENTER_NBR" in sql
    assert labels == ["MODEL_B_STORE_RECENCY scope=store_ids;channel"]
    assert result.to_dict(orient="records") == [
        {
            "profit_center_nbr": 63,
            "last_origin_week_date": "2025-07-14",
        }
    ]


def test_load_table_logs_bounded_filter_scope(monkeypatch) -> None:
    """A filtered collect label must identify its scope without logging raw data."""
    module = load_hana_loader_module(monkeypatch)
    labels: list[str] = []

    class FakeHanaFrame:
        """Small HANA DataFrame stand-in."""

        def collect(self) -> pd.DataFrame:
            """Return one synthetic row."""
            return pd.DataFrame({"VALUE": [1]})

    class FakeConnection:
        """Capture SQL passed to the HANA connection."""

        def sql(self, query: str) -> FakeHanaFrame:
            """Return a fake collectable HANA frame."""
            return FakeHanaFrame()

    def fake_collect(label: str, collect_func: Any) -> pd.DataFrame:
        """Record the memory-log label and execute the fake collection."""
        labels.append(label)
        return collect_func()

    monkeypatch.setattr(module, "get_hana_connection", lambda: FakeConnection())
    monkeypatch.setattr(module, "collect_dataframe_with_memory_logging", fake_collect)

    module.load_table(
        "MODEL_B",
        where_clause=(
            "PROFIT_CENTER_NBR IN (63) AND CHANNEL = 'B&M' "
            "AND ORIGIN_WEEK_DATE = '2025-07-07' AND HORIZON <= 13"
        ),
    )

    assert labels == [
        "MODEL_B scope=store_ids;channel;origin_week_date;max_horizon"
    ]
