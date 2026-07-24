"""Regression tests for bounded forecast initialization."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest


class FakeScenarioData:
    """Small ScenarioData replacement used by initialization tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Store scenario fields provided by the context tool."""
        self.__dict__.update(kwargs)


class FakeSession:
    """Capture state mutations and scenarios created by initialization."""

    def __init__(self) -> None:
        """Initialize empty captured state."""
        self.origin_date: Optional[str] = None
        self.horizon_weeks: Optional[int] = None
        self.channel: Optional[str] = None
        self.store_filter: List[int] = []
        self.dma_filter: List[str] = []
        self.scenarios: Dict[str, FakeScenarioData] = {}
        self.active_scenario: Optional[str] = None

    def set_origin_date(self, value: str) -> None:
        """Capture the origin date."""
        self.origin_date = value

    def set_horizon_weeks(self, value: int) -> None:
        """Capture the forecast horizon."""
        self.horizon_weeks = value

    def set_channel(self, value: str) -> None:
        """Capture the active channel."""
        self.channel = value

    def set_store_filter(self, value: List[int]) -> None:
        """Capture the requested store scope."""
        self.store_filter = value

    def set_dma_filter(self, value: List[str]) -> None:
        """Capture the requested DMA scope."""
        self.dma_filter = value

    def add_scenario(self, scenario: FakeScenarioData) -> None:
        """Store a created scenario by name."""
        self.scenarios[scenario.name] = scenario

    def set_active_scenario(self, name: str) -> None:
        """Capture the active scenario name."""
        self.active_scenario = name


def placeholder_session() -> None:
    """Return no session before the module function is monkeypatched."""
    return None


def empty_dataframe(**kwargs: Any) -> pd.DataFrame:
    """Return an empty DataFrame for import-time loader placeholders."""
    return pd.DataFrame()


def install_context_import_fakes(monkeypatch) -> None:
    """Install minimal app.agent modules needed to load context.py directly."""
    agent_package = types.ModuleType("app.agent")
    agent_package.__path__ = []

    session_module = types.ModuleType("app.agent.session")
    session_module.get_session = placeholder_session

    state_module = types.ModuleType("app.agent.state")
    state_module.ScenarioData = FakeScenarioData

    hana_module = types.ModuleType("app.agent.hana_loader")
    hana_module.load_model_b_filtered = empty_dataframe
    hana_module.load_model_b_origin_summary = empty_dataframe
    hana_module.load_model_b_last_origin_by_store = empty_dataframe
    hana_module.load_store_master = empty_dataframe

    monkeypatch.setitem(sys.modules, "app.agent", agent_package)
    monkeypatch.setitem(sys.modules, "app.agent.session", session_module)
    monkeypatch.setitem(sys.modules, "app.agent.state", state_module)
    monkeypatch.setitem(sys.modules, "app.agent.hana_loader", hana_module)


def load_context_module(monkeypatch):
    """Load context.py with focused dependency fakes."""
    install_context_import_fakes(monkeypatch)
    root = Path(__file__).parents[1]
    module_path = root / "agent" / "tools" / "context.py"
    spec = importlib.util.spec_from_file_location("context_tools_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def get_tool_function(tool_object: Any):
    """Return the callable wrapped by a LangChain tool."""
    return getattr(tool_object, "func", tool_object)


def make_origin_summary() -> pd.DataFrame:
    """Return aggregate metadata with historical and full-coverage origins."""
    return pd.DataFrame(
        {
            "origin_week_date": ["2025-06-30", "2025-07-07", "2025-07-14"],
            "max_target_week_date": ["2025-09-22", "2025-09-29", "2025-10-06"],
            "horizon_count": [13, 13, 3],
        }
    )


@pytest.mark.parametrize(
    ("requested_origin", "horizon_weeks", "expected_mode", "expected_origin"),
    [
        ("2025-07-10", 12, "backtesting", "2025-07-07"),
        ("2025-06-01", 13, "backtesting", "2025-06-30"),
        ("2026-07-22", 13, "forecasting", "2025-07-07"),
    ],
)
def test_select_model_b_origin_preserves_mode_rules(
    monkeypatch,
    requested_origin: str,
    horizon_weeks: int,
    expected_mode: str,
    expected_origin: str,
) -> None:
    """Origin selection floors history and uses latest complete future origin."""
    module = load_context_module(monkeypatch)

    mode, selected = module._select_model_b_origin(
        make_origin_summary(),
        pd.Timestamp(requested_origin),
        horizon_weeks,
    )

    assert mode == expected_mode
    assert selected == expected_origin


def test_select_model_b_origin_falls_back_to_latest_incomplete_origin(
    monkeypatch,
) -> None:
    """Future mode uses the latest origin when no origin has full coverage."""
    module = load_context_module(monkeypatch)
    summary = make_origin_summary()
    summary["horizon_count"] = [3, 5, 7]

    mode, selected = module._select_model_b_origin(
        summary,
        pd.Timestamp("2026-07-22"),
        13,
    )

    assert mode == "forecasting"
    assert selected == "2025-07-14"


def test_initialize_rejects_implicit_all_store_scope(monkeypatch) -> None:
    """Omitted scope must fail before session mutation or HANA collection."""
    module = load_context_module(monkeypatch)
    session = FakeSession()
    monkeypatch.setattr(module, "get_session", lambda: session)
    monkeypatch.setattr(
        module,
        "load_model_b_origin_summary",
        lambda **kwargs: pytest.fail("HANA must not be queried"),
        raising=False,
    )

    result = get_tool_function(module.initialize_forecast_simulation)(
        origin_date="2025-07-10",
    )

    assert result == {
        "error": "Forecast scope is required to protect application memory.",
        "hint": "Provide store_ids or dmas. Set allow_all_stores=true only for an explicitly requested whole-portfolio forecast.",
    }
    assert session.origin_date is None


def test_initialize_collects_one_selected_origin_for_store(monkeypatch) -> None:
    """Scoped initialization must collect full MODEL_B columns exactly once."""
    module = load_context_module(monkeypatch)
    session = FakeSession()
    full_load_calls: list[Dict[str, Any]] = []

    def fake_full_load(**kwargs: Any) -> pd.DataFrame:
        """Record the selected full-column query and return its baseline rows."""
        full_load_calls.append(kwargs)
        return pd.DataFrame(
            {
                "profit_center_nbr": [63, 63],
                "dma": ["BERLIN OST", "BERLIN OST"],
                "channel": ["B&M", "B&M"],
                "origin_week_date": ["2025-07-07", "2025-07-07"],
                "target_week_date": ["2025-07-14", "2025-07-21"],
                "horizon": [1, 2],
            }
        )

    monkeypatch.setattr(module, "get_session", lambda: session)
    monkeypatch.setattr(module, "_get_fiscal_context", lambda value: {})
    monkeypatch.setattr(module, "load_model_b_origin_summary", lambda **kwargs: make_origin_summary(), raising=False)
    monkeypatch.setattr(
        module,
        "load_model_b_last_origin_by_store",
        lambda **kwargs: pd.DataFrame(
            {
                "profit_center_nbr": [63],
                "last_origin_week_date": ["2025-07-14"],
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "load_model_b_filtered", fake_full_load)

    result = get_tool_function(module.initialize_forecast_simulation)(
        origin_date="2025-07-10",
        horizon_weeks=12,
        store_ids=[63],
    )

    assert result["status"] == "initialized"
    assert result["mode"] == "backtesting"
    assert result["actual_origin_date"] == "2025-07-07"
    assert len(full_load_calls) == 1
    assert full_load_calls[0] == {
        "profit_center_nbrs": [63],
        "channel": "B&M",
        "origin_week_date": "2025-07-07",
        "max_horizon": 12,
    }


def test_initialize_allows_explicit_portfolio_opt_in(monkeypatch) -> None:
    """A deliberate portfolio request remains possible with one bounded origin."""
    module = load_context_module(monkeypatch)
    session = FakeSession()
    session.store_filter = [46]
    session.dma_filter = ["OLD DMA"]
    full_load_calls: list[Dict[str, Any]] = []

    def fake_full_load(**kwargs: Any) -> pd.DataFrame:
        """Return one selected-origin row for the portfolio request."""
        full_load_calls.append(kwargs)
        return pd.DataFrame(
            {
                "profit_center_nbr": [63],
                "origin_week_date": ["2025-07-07"],
                "target_week_date": ["2025-07-14"],
                "horizon": [1],
            }
        )

    monkeypatch.setattr(module, "get_session", lambda: session)
    monkeypatch.setattr(module, "_get_fiscal_context", lambda value: {})
    monkeypatch.setattr(module, "load_model_b_origin_summary", lambda **kwargs: make_origin_summary(), raising=False)
    monkeypatch.setattr(module, "load_model_b_filtered", fake_full_load)

    result = get_tool_function(module.initialize_forecast_simulation)(
        origin_date="2026-07-22",
        horizon_weeks=13,
        allow_all_stores=True,
    )

    assert result["status"] == "initialized"
    assert result["mode"] == "forecasting"
    assert full_load_calls == [
        {
            "profit_center_nbrs": None,
            "channel": "B&M",
            "origin_week_date": "2025-07-07",
            "max_horizon": 13,
        }
    ]
    assert session.store_filter == []
    assert session.dma_filter == []


def test_initialize_rejects_store_with_stale_aggregate_data(monkeypatch) -> None:
    """A stale requested store must fail before origin or full-data collection."""
    module = load_context_module(monkeypatch)
    session = FakeSession()
    monkeypatch.setattr(module, "get_session", lambda: session)
    monkeypatch.setattr(
        module,
        "load_model_b_last_origin_by_store",
        lambda **kwargs: pd.DataFrame(
            {
                "profit_center_nbr": [63],
                "last_origin_week_date": ["2024-01-01"],
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "load_model_b_origin_summary",
        lambda **kwargs: pytest.fail("origin summary must not run"),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "load_model_b_filtered",
        lambda **kwargs: pytest.fail("full collect must not run"),
    )

    result = get_tool_function(module.initialize_forecast_simulation)(
        origin_date="2025-07-10",
        store_ids=[63],
    )

    assert result["error"] == (
        "All requested stores have stale or missing data for the specified date."
    )
    assert "Store 63: Last data from 2024-01-01" in result["details"][0]


def test_initialize_intersects_store_and_dma_scope_consistently(monkeypatch) -> None:
    """Combined store and DMA filters persist only their effective intersection."""
    module = load_context_module(monkeypatch)
    session = FakeSession()
    aggregate_scopes: list[List[int]] = []
    full_load_calls: list[Dict[str, Any]] = []

    def fake_recency(**kwargs: Any) -> pd.DataFrame:
        """Assert staleness is checked only for the effective DMA intersection."""
        aggregate_scopes.append(kwargs["profit_center_nbrs"])
        return pd.DataFrame(
            {
                "profit_center_nbr": [63],
                "last_origin_week_date": ["2025-07-14"],
            }
        )

    def fake_summary(**kwargs: Any) -> pd.DataFrame:
        """Capture the store scope used by the origin aggregate."""
        aggregate_scopes.append(kwargs["profit_center_nbrs"])
        return make_origin_summary()

    def fake_full_load(**kwargs: Any) -> pd.DataFrame:
        """Capture the final selected-origin query."""
        full_load_calls.append(kwargs)
        return pd.DataFrame(
            {
                "profit_center_nbr": [63],
                "origin_week_date": ["2025-07-07"],
                "target_week_date": ["2025-07-14"],
                "horizon": [1],
            }
        )

    monkeypatch.setattr(module, "get_session", lambda: session)
    monkeypatch.setattr(module, "_get_fiscal_context", lambda value: {})
    monkeypatch.setattr(
        module,
        "load_store_master",
        lambda: pd.DataFrame(
            {
                "profit_center_nbr": [63, 62],
                "market_city": ["Berlin Ost", "Dresden Ost"],
            }
        ),
    )
    monkeypatch.setattr(module, "load_model_b_last_origin_by_store", fake_recency, raising=False)
    monkeypatch.setattr(module, "load_model_b_origin_summary", fake_summary, raising=False)
    monkeypatch.setattr(module, "load_model_b_filtered", fake_full_load)

    result = get_tool_function(module.initialize_forecast_simulation)(
        origin_date="2025-07-10",
        horizon_weeks=12,
        store_ids=[63, 62],
        dmas=["BERLIN OST"],
    )

    assert result["status"] == "initialized"
    assert aggregate_scopes == [[63], [63]]
    assert full_load_calls[0]["profit_center_nbrs"] == [63]
    assert result["store_filter"] == [63]
    assert session.store_filter == [63]
