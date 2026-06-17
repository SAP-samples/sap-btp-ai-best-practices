"""Tests for execution tools using the global inference cache."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import pytest


class FakePredictionResult:
    """Small PredictionResult replacement for execution tool tests."""

    def __init__(
        self,
        scenario_name: str,
        predictions_df: pd.DataFrame,
        shap_df: Any = None,
        generated_at: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Store prediction result fields used by the execution tools."""
        self.scenario_name = scenario_name
        self.predictions_df = predictions_df
        self.shap_df = shap_df
        self.generated_at = generated_at
        self.metadata = metadata or {}


class FakeScenario:
    """Small scenario replacement with a DataFrame and channel."""

    def __init__(self, name: str, df: pd.DataFrame, channel: str = "B&M") -> None:
        """Initialize a fake scenario."""
        self.name = name
        self.df = df
        self.channel = channel
        self.modifications = []


class FakeForecastSession:
    """Session fake for run_forecast_model tests."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize the fake session with one active scenario."""
        self.checkpoint_dir = checkpoint_dir
        self.predictions: Dict[str, FakePredictionResult] = {}
        self.stored_forecast = None
        self.scenario = FakeScenario(
            "scenario",
            pd.DataFrame(
                {
                    "channel": ["B&M"],
                    "profit_center_nbr": [46],
                    "horizon": [1],
                    "target_week_date": ["2025-07-07"],
                }
            ),
        )

    def get_origin_date(self) -> str:
        """Return an initialized origin date."""
        return "2025-07-01"

    def get_active_scenario_name(self) -> str:
        """Return the active scenario name."""
        return "scenario"

    def get_scenario(self, scenario_name: str | None = None) -> FakeScenario:
        """Return the fake scenario."""
        return self.scenario

    def get_state(self) -> Dict[str, Any]:
        """Return state with the fake scenario."""
        return {"scenarios": {"scenario": self.scenario}}

    def get_channel(self) -> str:
        """Return B&M channel."""
        return "B&M"

    def get_checkpoint_dir(self) -> Path:
        """Return the fake checkpoint directory."""
        return self.checkpoint_dir

    def add_prediction(self, prediction: FakePredictionResult) -> None:
        """Cache a prediction result."""
        self.predictions[prediction.scenario_name] = prediction

    def store_forecast_results(self, value: str) -> None:
        """Store serialized forecast results."""
        self.stored_forecast = value


class FakeSensitivitySession(FakeForecastSession):
    """Session fake for analyze_sensitivity tests."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize the fake session with baseline data and predictions."""
        super().__init__(checkpoint_dir)
        baseline_df = pd.DataFrame(
            {
                "channel": ["B&M", "B&M"],
                "feature_one": [10.0, 10.0],
                "feature_two": [20.0, 20.0],
            }
        )
        self.baseline = FakeScenario("baseline_bm", baseline_df)
        self.sensitivity = None
        self.predictions["baseline_bm"] = FakePredictionResult(
            scenario_name="baseline_bm",
            predictions_df=pd.DataFrame({"pred_sales_p50": [100.0]}),
        )

    def get_scenario(self, scenario_name: str | None = None) -> FakeScenario:
        """Return the baseline scenario for sensitivity analysis."""
        return self.baseline

    def has_prediction(self, scenario_name: str) -> bool:
        """Return whether a fake prediction exists."""
        return scenario_name in self.predictions

    def get_prediction(self, scenario_name: str) -> FakePredictionResult:
        """Return a fake cached prediction."""
        return self.predictions[scenario_name]

    def store_sensitivity(self, value: str) -> None:
        """Store serialized sensitivity results."""
        self.sensitivity = value


def create_checkpoint_files(checkpoint_dir: Path) -> None:
    """Create the checkpoint files expected by B&M validation."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("bm_multi.cbm", "bm_conversion.cbm", "residual_stats.json"):
        (checkpoint_dir / file_name).write_text("{}")


def placeholder_session() -> None:
    """Return no session before tests monkeypatch module-level get_session."""
    return None


def passthrough_json(value: Any) -> Any:
    """Return a value unchanged for sanitize_for_json tests."""
    return value


def resolve_feature_name(value: str) -> str:
    """Return the input as the resolved feature name."""
    return value


def get_feature_metadata(value: str) -> SimpleNamespace:
    """Return generic fake metadata for a feature."""
    return SimpleNamespace(category="test", min_value=None, max_value=None)


def get_modifiable_features(channel: str) -> list[str]:
    """Return two fake modifiable features."""
    return ["feature_one", "feature_two"]


def get_features_by_category(category: str) -> list[str]:
    """Return two fake category features."""
    return ["feature_one", "feature_two"]


def load_empty_model_b(**kwargs: Any) -> pd.DataFrame:
    """Return an empty fake model_b DataFrame."""
    return pd.DataFrame()


def install_execution_import_fakes(monkeypatch) -> None:
    """Install fake modules required to load execution.py directly."""
    agent_package = types.ModuleType("app.agent")
    agent_package.__path__ = []

    session_module = types.ModuleType("app.agent.session")
    session_module.get_session = placeholder_session

    state_module = types.ModuleType("app.agent.state")
    state_module.PredictionResult = FakePredictionResult

    feature_module = types.ModuleType("app.agent.feature_mapping")
    feature_module.resolve_feature_name = resolve_feature_name
    feature_module.get_feature_metadata = get_feature_metadata
    feature_module.get_modifiable_features = get_modifiable_features
    feature_module.get_features_by_category = get_features_by_category
    feature_module.FEATURE_CATEGORIES = {"test": ["feature_one", "feature_two"]}

    hana_module = types.ModuleType("app.agent.hana_loader")
    hana_module.load_model_b_filtered = load_empty_model_b

    tools_package = types.ModuleType("app.agent.tools")
    tools_package.__path__ = []

    utils_module = types.ModuleType("app.agent.tools.utils")
    utils_module.sanitize_for_json = passthrough_json

    monkeypatch.setitem(sys.modules, "app.agent", agent_package)
    monkeypatch.setitem(sys.modules, "app.agent.session", session_module)
    monkeypatch.setitem(sys.modules, "app.agent.state", state_module)
    monkeypatch.setitem(sys.modules, "app.agent.feature_mapping", feature_module)
    monkeypatch.setitem(sys.modules, "app.agent.hana_loader", hana_module)
    monkeypatch.setitem(sys.modules, "app.agent.tools", tools_package)
    monkeypatch.setitem(sys.modules, "app.agent.tools.utils", utils_module)


def load_execution_module(monkeypatch):
    """Load execution.py with fake imports for focused tool tests."""
    install_execution_import_fakes(monkeypatch)
    root = Path(__file__).parents[1]
    module_path = root / "agent" / "tools" / "execution.py"
    spec = importlib.util.spec_from_file_location("execution_tools_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def get_tool_function(tool_object: Any):
    """Return the underlying callable for a LangChain tool or plain function."""
    return getattr(tool_object, "func", tool_object)


def test_run_forecast_model_uses_cached_inference(monkeypatch, tmp_path: Path) -> None:
    """run_forecast_model runs through the global cache without saving CSV outputs."""
    module = load_execution_module(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    create_checkpoint_files(checkpoint_dir)
    session = FakeForecastSession(checkpoint_dir)
    calls = []

    def fake_run_cached_inference(**kwargs: Any) -> SimpleNamespace:
        """Record cached inference arguments and return fake B&M predictions."""
        calls.append(kwargs)
        return SimpleNamespace(
            bm_predictions=pd.DataFrame(
                {
                    "profit_center_nbr": [46],
                    "horizon": [1],
                    "target_week_date": ["2025-07-07"],
                    "pred_sales_p50": [123.0],
                    "pred_sales_p90": [150.0],
                    "pred_aov_p50": [42.0],
                    "pred_traffic_p50": [20.0],
                    "pred_traffic_p90": [25.0],
                }
            ),
            web_predictions=None,
        )

    def get_fake_session() -> FakeForecastSession:
        """Return the fake forecast session."""
        return session

    monkeypatch.setattr(module, "get_session", get_fake_session)
    monkeypatch.setattr(module, "run_cached_inference", fake_run_cached_inference, raising=False)

    result = get_tool_function(module.run_forecast_model)(scenario_names=["scenario"])

    assert result["status"] == "predicted"
    assert len(calls) == 1
    assert calls[0]["save_outputs"] is False
    assert calls[0]["estimate_traffic"] is True
    assert calls[0]["channels"] == ["B&M"]


@pytest.mark.parametrize("include_traffic", [False, True])
def test_analyze_sensitivity_reuses_cached_pipeline_and_controls_traffic(
    monkeypatch,
    tmp_path: Path,
    include_traffic: bool,
) -> None:
    """analyze_sensitivity reuses one cached pipeline and controls traffic estimation."""
    module = load_execution_module(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    create_checkpoint_files(checkpoint_dir)
    session = FakeSensitivitySession(checkpoint_dir)
    cache_calls = []
    run_calls = []
    pipeline = object()

    def fake_get_cached_inference_pipeline(**kwargs: Any) -> object:
        """Record cache lookup calls and return one fake pipeline object."""
        cache_calls.append(kwargs)
        return pipeline

    def fake_run_cached_inference(**kwargs: Any) -> SimpleNamespace:
        """Record cached inference calls and return sales based on perturbed input."""
        run_calls.append(kwargs)
        df = kwargs["model_b_data"]
        sales = float(df["feature_one"].mean() + df["feature_two"].mean())
        return SimpleNamespace(
            bm_predictions=pd.DataFrame({"pred_sales_p50": [sales]}),
            web_predictions=None,
        )

    def get_fake_session() -> FakeSensitivitySession:
        """Return the fake sensitivity session."""
        return session

    monkeypatch.setattr(module, "get_session", get_fake_session)
    monkeypatch.setattr(
        module,
        "get_cached_inference_pipeline",
        fake_get_cached_inference_pipeline,
        raising=False,
    )
    monkeypatch.setattr(module, "run_cached_inference", fake_run_cached_inference, raising=False)

    result = get_tool_function(module.analyze_sensitivity)(include_traffic=include_traffic)

    assert result["status"] == "analyzed"
    assert len(cache_calls) == 1
    assert len(run_calls) == 4
    assert {call["pipeline"] for call in run_calls} == {pipeline}
    assert {call["estimate_traffic"] for call in run_calls} == {include_traffic}
    assert {call["save_outputs"] for call in run_calls} == {False}
