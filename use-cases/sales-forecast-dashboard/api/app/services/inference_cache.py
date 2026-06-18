"""Process-level inference pipeline cache for forecast model reuse.

The API expects one forecast model configuration per Python process. This
module owns that cached pipeline, protects cold-start loading with a lock, and
serializes prediction calls until predictor thread-safety is proven.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from app.services.memory_monitor import get_process_memory

logger = logging.getLogger("app.inference_cache")

CacheKey = Tuple[str, Tuple[str, ...], bool]

_cache_lock = threading.Lock()
_prediction_lock = threading.Lock()
_cached_pipeline: Any = None
_cached_key: Optional[CacheKey] = None


def _default_checkpoint_dir() -> Path:
    """Return the default deployed agent checkpoint directory."""
    default_input_dir = Path(__file__).resolve().parents[1] / "agent" / "input"
    agent_input_dir = Path(os.getenv("AGENT_INPUT_DIR", str(default_input_dir)))
    return Path(os.getenv("CHECKPOINT_DIR", str(agent_input_dir / "checkpoints")))


def _normalize_channels(channels: Optional[Iterable[str]] = None) -> List[str]:
    """Normalize inference channel names from input values or environment.

    Parameters
    ----------
    channels : Optional[Iterable[str]]
        Channel names to normalize. When omitted, INFERENCE_PRELOAD_CHANNELS is
        read from the environment and defaults to B&M.

    Returns
    -------
    List[str]
        Normalized unique channel names in request order.
    """
    if channels is None:
        raw_channels = os.getenv("INFERENCE_PRELOAD_CHANNELS", "B&M").split(",")
    else:
        raw_channels = list(channels)

    normalized: List[str] = []
    for raw_channel in raw_channels:
        channel = str(raw_channel).strip()
        if not channel:
            continue
        upper = channel.upper()
        if upper in {"BM", "B&M"}:
            value = "B&M"
        elif upper == "WEB":
            value = "WEB"
        else:
            raise ValueError(f"Unsupported inference channel: {channel}")
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        raise ValueError("At least one inference channel is required")
    return normalized


def _make_cache_key(
    checkpoint_dir: Optional[Path] = None,
    channels: Optional[Iterable[str]] = None,
    run_explainability: bool = False,
) -> CacheKey:
    """Build the process-level cache key for the configured model bundle."""
    resolved_checkpoint_dir = Path(checkpoint_dir or _default_checkpoint_dir()).resolve()
    return (
        str(resolved_checkpoint_dir),
        tuple(_normalize_channels(channels)),
        bool(run_explainability),
    )


def _required_checkpoint_files(channels: Iterable[str]) -> List[str]:
    """Return checkpoint files required by the requested channels."""
    required = {"residual_stats.json"}
    channel_set = set(channels)
    if "B&M" in channel_set:
        required.update({"bm_multi.cbm", "bm_conversion.cbm"})
    if "WEB" in channel_set:
        required.add("web_multi.cbm")
    return sorted(required)


def _validate_checkpoint_files(checkpoint_dir: Path, channels: Iterable[str]) -> None:
    """Raise FileNotFoundError when required checkpoint files are missing."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    missing = [
        file_name
        for file_name in _required_checkpoint_files(channels)
        if not (checkpoint_dir / file_name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint files in {checkpoint_dir}: {missing}"
        )


def _memory_delta(before: dict[str, Any], after: dict[str, Any], field: str) -> Optional[float]:
    """Calculate a rounded memory delta for one field."""
    before_value = before.get(field)
    after_value = after.get(field)
    if before_value is None or after_value is None:
        return None
    return round(float(after_value) - float(before_value), 2)


def _format_delta(value: Optional[float]) -> str:
    """Format a memory delta for log output."""
    if value is None:
        return "None"
    return f"{value:.2f}"


def _build_pipeline(key: CacheKey) -> Any:
    """Create and load an InferencePipeline for a cache key."""
    from app.regressor.configs import InferenceConfig
    from app.regressor.pipelines import InferencePipeline

    checkpoint_dir = Path(key[0])
    channels = list(key[1])
    run_explainability = key[2]
    _validate_checkpoint_files(checkpoint_dir, channels)

    config = InferenceConfig(
        checkpoint_dir=checkpoint_dir,
        output_dir=checkpoint_dir.parent / "infer",
        channels=channels,
        run_explainability=run_explainability,
    )
    pipeline = InferencePipeline(config)
    load_models = getattr(pipeline, "_load_models", None)
    if callable(load_models):
        load_models()
    return pipeline


def get_cached_inference_pipeline(
    checkpoint_dir: Optional[Path] = None,
    channels: Optional[Iterable[str]] = None,
    run_explainability: bool = False,
) -> Any:
    """Return the process-level cached inference pipeline.

    Parameters
    ----------
    checkpoint_dir : Optional[Path]
        Checkpoint directory. Defaults to CHECKPOINT_DIR or agent input
        checkpoints.
    channels : Optional[Iterable[str]]
        Channels to configure. Defaults to INFERENCE_PRELOAD_CHANNELS.
    run_explainability : bool
        Whether the cached bundle should include surrogate explainability models.

    Returns
    -------
    Any
        Loaded InferencePipeline instance.

    Raises
    ------
    RuntimeError
        If a different model configuration is requested after the process cache
        has already been initialized.
    """
    global _cached_key, _cached_pipeline

    key = _make_cache_key(checkpoint_dir, channels, run_explainability)
    if _cached_pipeline is not None:
        if _cached_key != key:
            raise RuntimeError(
                "Inference cache already initialized for "
                f"{_cached_key}; requested {key}. Restart the API process or "
                "configure INFERENCE_PRELOAD_CHANNELS to include the required model."
            )
        logger.info("inference_cache event=hit key=%s", key)
        return _cached_pipeline

    with _cache_lock:
        if _cached_pipeline is not None:
            if _cached_key != key:
                raise RuntimeError(
                    "Inference cache already initialized for "
                    f"{_cached_key}; requested {key}."
                )
            logger.info("inference_cache event=hit_after_lock key=%s", key)
            return _cached_pipeline

        before = get_process_memory()
        start = time.perf_counter()
        logger.info("inference_cache event=load_start key=%s", key)
        pipeline = _build_pipeline(key)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        after = get_process_memory()
        _cached_pipeline = pipeline
        _cached_key = key
        logger.info(
            "inference_cache event=load_complete key=%s elapsed_ms=%.2f "
            "rss_delta_mb=%s uss_delta_mb=%s rss_mb=%s uss_mb=%s",
            key,
            elapsed_ms,
            _format_delta(_memory_delta(before, after, "rss_mb")),
            _format_delta(_memory_delta(before, after, "uss_mb")),
            after.get("rss_mb"),
            after.get("uss_mb"),
        )
        return _cached_pipeline


def warm_inference_cache(
    checkpoint_dir: Optional[Path] = None,
    channels: Optional[Iterable[str]] = None,
    run_explainability: bool = False,
) -> Any:
    """Preload the global inference cache and return the loaded pipeline."""
    return get_cached_inference_pipeline(
        checkpoint_dir=checkpoint_dir,
        channels=channels,
        run_explainability=run_explainability,
    )


def run_cached_inference(
    model_b_data: Any,
    model_a_data: Any = None,
    channels: Optional[Iterable[str]] = None,
    checkpoint_dir: Optional[Path] = None,
    run_explainability: bool = False,
    save_outputs: bool = False,
    estimate_traffic: bool = True,
    pipeline: Any = None,
    label: str = "forecast_inference",
) -> Any:
    """Run inference through the process cache under a prediction lock.

    Parameters
    ----------
    model_b_data : Any
        Model B input data passed to InferencePipeline.run.
    model_a_data : Any
        Optional Model A input data passed to InferencePipeline.run.
    channels : Optional[Iterable[str]]
        Channels to score.
    checkpoint_dir : Optional[Path]
        Checkpoint directory for cache lookup when pipeline is not provided.
    run_explainability : bool
        Whether the cached pipeline must include explainability models.
    save_outputs : bool
        Whether to write prediction CSV files.
    estimate_traffic : bool
        Whether B&M traffic Monte Carlo estimates should be computed.
    pipeline : Any
        Optional already-resolved cached pipeline.
    label : str
        Log label for the inference call.

    Returns
    -------
    Any
        InferenceResult returned by InferencePipeline.run.
    """
    active_pipeline = pipeline or get_cached_inference_pipeline(
        checkpoint_dir=checkpoint_dir,
        channels=channels,
        run_explainability=run_explainability,
    )
    run_channels = _normalize_channels(channels) if channels is not None else None

    before = get_process_memory()
    start = time.perf_counter()
    logger.info(
        "inference_cache event=run_start label=%s channels=%s save_outputs=%s "
        "estimate_traffic=%s",
        label,
        run_channels,
        save_outputs,
        estimate_traffic,
    )
    with _prediction_lock:
        result = active_pipeline.run(
            model_b_data=model_b_data,
            model_a_data=model_a_data,
            channels=run_channels,
            save_outputs=save_outputs,
            estimate_traffic=estimate_traffic,
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    after = get_process_memory()
    logger.info(
        "inference_cache event=run_complete label=%s elapsed_ms=%.2f "
        "rss_delta_mb=%s uss_delta_mb=%s rss_mb=%s uss_mb=%s",
        label,
        elapsed_ms,
        _format_delta(_memory_delta(before, after, "rss_mb")),
        _format_delta(_memory_delta(before, after, "uss_mb")),
        after.get("rss_mb"),
        after.get("uss_mb"),
    )
    return result


def reset_inference_cache_for_tests() -> None:
    """Reset process cache state for focused unit tests."""
    global _cached_key, _cached_pipeline
    with _cache_lock:
        _cached_pipeline = None
        _cached_key = None
