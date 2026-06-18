"""Tests for inference pipeline runtime options."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


class FakePredictor:
    """Predictor fake that records whether traffic estimation was requested."""

    def __init__(self) -> None:
        """Initialize the fake predictor."""
        self.estimate_traffic_values = []

    def predict(self, df: pd.DataFrame, estimate_traffic: bool = True) -> SimpleNamespace:
        """Return deterministic fake prediction arrays."""
        self.estimate_traffic_values.append(estimate_traffic)
        row_count = len(df)
        return SimpleNamespace(
            log_sales=np.zeros(row_count),
            log_aov=np.zeros(row_count),
            log_orders=np.zeros(row_count),
            logit_conversion=np.zeros(row_count),
            traffic=None,
        )


class FakeBiasCorrection:
    """Bias correction fake with B&M correction disabled."""

    def should_correct_bm(self) -> bool:
        """Return False for B&M correction."""
        return False


def load_inference_module(monkeypatch):
    """Load inference.py with fake model dependencies."""
    models_module = types.ModuleType("app.regressor.models")
    models_module.BMPredictor = object
    models_module.WEBPredictor = object
    models_module.SurrogateExplainer = object
    models_module.TrafficEstimator = object

    configs_module = types.ModuleType("app.regressor.configs")
    configs_module.InferenceConfig = object
    configs_module.BiasCorrection = FakeBiasCorrection

    monkeypatch.setitem(sys.modules, "app.regressor.models", models_module)
    monkeypatch.setitem(sys.modules, "app.regressor.configs", configs_module)

    root = Path(__file__).parents[1]
    module_path = root / "regressor" / "pipelines" / "inference.py"
    spec = importlib.util.spec_from_file_location("inference_pipeline_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_can_skip_csv_output_and_traffic_estimation(monkeypatch, tmp_path: Path) -> None:
    """InferencePipeline.run can avoid CSV writes and B&M traffic estimation."""
    module = load_inference_module(monkeypatch)
    predictor = FakePredictor()
    save_calls = []

    def record_save(self: Any, output_dir: Any = None) -> None:
        """Record unexpected save calls."""
        save_calls.append(output_dir)

    monkeypatch.setattr(module.InferenceResult, "save", record_save)

    config = SimpleNamespace(
        checkpoint_dir=tmp_path,
        output_dir=tmp_path / "infer",
        channels=["B&M"],
        run_explainability=False,
        bias_correction=FakeBiasCorrection(),
    )
    pipeline = module.InferencePipeline(config)
    pipeline.bm_predictor = predictor
    pipeline.residual_stats = {"bm": {"rmse_sales": 0.0, "rmse_aov": 0.0}}

    result = pipeline.run(
        pd.DataFrame({"channel": ["B&M"], "origin_week_date": ["2025-07-01"]}),
        channels=["B&M"],
        save_outputs=False,
        estimate_traffic=False,
    )

    assert result.bm_predictions is not None
    assert save_calls == []
    assert predictor.estimate_traffic_values == [False]
