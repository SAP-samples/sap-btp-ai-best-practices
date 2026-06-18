"""Tests for the process-level inference cache service."""

from __future__ import annotations

import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List

from app.services import inference_cache


class FakeInferenceConfig:
    """Small stand-in for the production InferenceConfig."""

    def __init__(self, **kwargs: Any) -> None:
        """Store provided configuration fields as attributes."""
        self.__dict__.update(kwargs)


class FakeBiasCorrection:
    """Small stand-in for the production BiasCorrection config."""

    def __init__(self, **kwargs: Any) -> None:
        """Store provided correction flags as attributes."""
        self.__dict__.update(kwargs)


class FakePipeline:
    """Fake inference pipeline that tracks load and run calls."""

    created: int = 0
    loaded: int = 0
    run_events: List[str] = []
    active_runs: int = 0
    max_active_runs: int = 0

    def __init__(self, config: FakeInferenceConfig) -> None:
        """Initialize the fake pipeline with a config object."""
        type(self).created += 1
        self.config = config

    def _load_models(self) -> None:
        """Track explicit model loading during cache warmup."""
        time.sleep(0.01)
        type(self).loaded += 1

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Track serialized inference calls."""
        type(self).run_events.append("start")
        type(self).active_runs += 1
        type(self).max_active_runs = max(type(self).max_active_runs, type(self).active_runs)
        time.sleep(0.02)
        type(self).active_runs -= 1
        type(self).run_events.append("end")
        return "result"

    @classmethod
    def reset(cls) -> None:
        """Reset fake pipeline counters."""
        cls.created = 0
        cls.loaded = 0
        cls.run_events = []
        cls.active_runs = 0
        cls.max_active_runs = 0


def install_fake_regressor_modules(monkeypatch) -> None:
    """Install fake regressor modules used by inference_cache imports."""
    pipelines_module = types.ModuleType("app.regressor.pipelines")
    pipelines_module.InferencePipeline = FakePipeline

    configs_module = types.ModuleType("app.regressor.configs")
    configs_module.InferenceConfig = FakeInferenceConfig
    configs_module.BiasCorrection = FakeBiasCorrection

    monkeypatch.setitem(sys.modules, "app.regressor.pipelines", pipelines_module)
    monkeypatch.setitem(sys.modules, "app.regressor.configs", configs_module)


def create_checkpoint_files(checkpoint_dir: Path) -> None:
    """Create the checkpoint files required for B&M warmup."""
    for file_name in ("bm_multi.cbm", "bm_conversion.cbm", "residual_stats.json"):
        (checkpoint_dir / file_name).write_text("{}")


def map_to_get_pipeline(_: int):
    """Adapter used to avoid anonymous functions in executor.map."""
    return _GET_PIPELINE_FOR_TEST()


def map_to_run_prediction(_: int):
    """Adapter used to avoid anonymous functions in executor.map."""
    return _RUN_PREDICTION_FOR_TEST()


def _missing_test_callable() -> None:
    """Raise when a test executor adapter is used before setup."""
    raise RuntimeError("test callable was not configured")


_GET_PIPELINE_FOR_TEST = _missing_test_callable
_RUN_PREDICTION_FOR_TEST = _missing_test_callable


def test_warm_inference_cache_loads_pipeline_once(monkeypatch, tmp_path: Path) -> None:
    """Warmup loads one pipeline and future lookups return the same object."""
    install_fake_regressor_modules(monkeypatch)
    create_checkpoint_files(tmp_path)
    FakePipeline.reset()
    inference_cache.reset_inference_cache_for_tests()

    pipeline = inference_cache.warm_inference_cache(
        checkpoint_dir=tmp_path,
        channels=["B&M"],
    )
    same_pipeline = inference_cache.get_cached_inference_pipeline(
        checkpoint_dir=tmp_path,
        channels=["B&M"],
    )

    assert pipeline is same_pipeline
    assert FakePipeline.created == 1
    assert FakePipeline.loaded == 1


def test_concurrent_cold_start_initializes_cache_once(monkeypatch, tmp_path: Path) -> None:
    """Concurrent first access uses double-checked locking around model loading."""
    install_fake_regressor_modules(monkeypatch)
    create_checkpoint_files(tmp_path)
    FakePipeline.reset()
    inference_cache.reset_inference_cache_for_tests()
    global _GET_PIPELINE_FOR_TEST

    def get_pipeline() -> FakePipeline:
        """Return the cached fake pipeline."""
        return inference_cache.get_cached_inference_pipeline(
            checkpoint_dir=tmp_path,
            channels=["B&M"],
        )

    _GET_PIPELINE_FOR_TEST = get_pipeline
    with ThreadPoolExecutor(max_workers=8) as executor:
        pipelines = list(executor.map(map_to_get_pipeline, range(20)))

    assert len({id(pipeline) for pipeline in pipelines}) == 1
    assert FakePipeline.created == 1
    assert FakePipeline.loaded == 1


def test_run_cached_inference_serializes_pipeline_runs(monkeypatch, tmp_path: Path) -> None:
    """Cached inference calls are serialized by the global prediction lock."""
    install_fake_regressor_modules(monkeypatch)
    create_checkpoint_files(tmp_path)
    FakePipeline.reset()
    inference_cache.reset_inference_cache_for_tests()
    global _RUN_PREDICTION_FOR_TEST
    pipeline = inference_cache.warm_inference_cache(
        checkpoint_dir=tmp_path,
        channels=["B&M"],
    )

    def run_prediction() -> str:
        """Run one cached fake prediction."""
        return inference_cache.run_cached_inference(
            model_b_data=None,
            channels=["B&M"],
            pipeline=pipeline,
            save_outputs=False,
            estimate_traffic=False,
        )

    _RUN_PREDICTION_FOR_TEST = run_prediction
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(map_to_run_prediction, range(2)))

    assert results == ["result", "result"]
    assert FakePipeline.max_active_runs == 1
    assert FakePipeline.run_events == ["start", "end", "start", "end"]
