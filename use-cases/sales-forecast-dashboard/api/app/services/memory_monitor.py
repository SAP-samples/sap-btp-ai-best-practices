"""Memory instrumentation utilities for API diagnostics.

The helpers in this module are intentionally lightweight and safe to call from
request handlers, LangChain tools, and HANA collection boundaries. They emit
structured log lines that can be read locally or in Cloud Foundry logs.
"""

from __future__ import annotations

import functools
import gc
import inspect
import logging
import os
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd

LOGGER_NAME = "app.memory"
logger = logging.getLogger(LOGGER_NAME)

F = TypeVar("F", bound=Callable[..., Any])
MemorySnapshot = Dict[str, Any]

_FILE_HANDLER_PATH: Optional[str] = None


def _bytes_to_mb(value: Optional[float]) -> Optional[float]:
    """Convert bytes to megabytes, preserving None values.

    Parameters
    ----------
    value : Optional[float]
        Byte value to convert.

    Returns
    -------
    Optional[float]
        Value in MiB rounded to two decimals, or None when value is None.
    """
    if value is None:
        return None
    return round(float(value) / (1024 * 1024), 2)


def _get_peak_rss_mb() -> Optional[float]:
    """Return peak RSS from the standard resource module.

    Returns
    -------
    Optional[float]
        Peak resident set size in MiB when available.
    """
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    # Linux reports KiB; macOS reports bytes.
    if sys.platform == "darwin":
        return _bytes_to_mb(peak)
    return round(float(peak) / 1024, 2)


def _read_proc_status_memory() -> MemorySnapshot:
    """Read Linux /proc memory fields when psutil is unavailable.

    Returns
    -------
    Dict[str, Optional[float]]
        Memory fields in MiB read from /proc/self/status.
    """
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return {}

    values: MemorySnapshot = {}
    field_map = {
        "VmRSS": "rss_mb",
        "VmHWM": "peak_rss_mb",
        "VmSize": "vms_mb",
    }

    try:
        for line in status_path.read_text().splitlines():
            key, _, raw_value = line.partition(":")
            if key not in field_map:
                continue
            parts = raw_value.strip().split()
            if not parts:
                continue
            # /proc/self/status reports KiB.
            values[field_map[key]] = round(float(parts[0]) / 1024, 2)
    except Exception:
        return {}

    values["source"] = "procfs"
    return values


def _get_psutil_memory() -> MemorySnapshot:
    """Read process memory via psutil when installed.

    Returns
    -------
    Dict[str, Optional[float]]
        Current process memory fields in MiB.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return {}

    try:
        process = psutil.Process(os.getpid())
        info = process.memory_full_info()
    except Exception:
        return {}

    return {
        "rss_mb": _bytes_to_mb(getattr(info, "rss", None)),
        "uss_mb": _bytes_to_mb(getattr(info, "uss", None)),
        "vms_mb": _bytes_to_mb(getattr(info, "vms", None)),
        "peak_rss_mb": _get_peak_rss_mb(),
        "source": "psutil",
    }


def get_process_memory() -> MemorySnapshot:
    """Return a snapshot of current process memory.

    Returns
    -------
    Dict[str, Optional[float]]
        JSON-safe memory snapshot with current RSS, optional USS, VMS, peak RSS,
        and the source used to collect the values.
    """
    memory = _get_psutil_memory()
    if not memory:
        memory = _read_proc_status_memory()

    if not memory:
        memory = {
            "rss_mb": None,
            "uss_mb": None,
            "vms_mb": None,
            "peak_rss_mb": _get_peak_rss_mb(),
            "source": "resource",
        }

    memory.setdefault("peak_rss_mb", _get_peak_rss_mb())
    return memory


def configure_memory_file_logging() -> Optional[str]:
    """Attach an optional file handler for memory logs.

    The file destination is controlled by the ``MEMORY_LOG_FILE`` environment
    variable. When unset, memory logs continue to flow through normal app logs.

    Returns
    -------
    Optional[str]
        Configured log file path, or None when file logging is disabled.
    """
    global _FILE_HANDLER_PATH

    log_path = os.getenv("MEMORY_LOG_FILE")
    if not log_path:
        return None

    if _FILE_HANDLER_PATH == log_path:
        return log_path

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    _FILE_HANDLER_PATH = log_path
    return log_path


def summarize_dataframe(name: str, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Summarize a DataFrame's shape and memory usage.

    Parameters
    ----------
    name : str
        Human-readable label for the DataFrame.
    df : Optional[pd.DataFrame]
        DataFrame to inspect.

    Returns
    -------
    Dict[str, Any]
        JSON-safe summary with row count, column count, and memory in MiB.
    """
    if df is None:
        return {"name": name, "rows": 0, "columns": 0, "memory_mb": 0.0}

    memory_bytes = int(df.memory_usage(deep=True).sum())
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_mb": round(memory_bytes / (1024 * 1024), 4),
    }


def _format_memory_for_log(memory: MemorySnapshot) -> str:
    """Format memory snapshot fields for compact log lines.

    Parameters
    ----------
    memory : Dict[str, Optional[float]]
        Process memory snapshot.

    Returns
    -------
    str
        Space-separated key-value fields.
    """
    return (
        f"rss_mb={memory.get('rss_mb')} "
        f"uss_mb={memory.get('uss_mb')} "
        f"vms_mb={memory.get('vms_mb')} "
        f"peak_rss_mb={memory.get('peak_rss_mb')} "
        f"source={memory.get('source')}"
    )


def _memory_delta(
    before: MemorySnapshot,
    after: MemorySnapshot,
    field: str,
) -> Optional[float]:
    """Calculate a rounded memory delta for one memory field.

    Parameters
    ----------
    before : Dict[str, Optional[float]]
        Snapshot before work.
    after : Dict[str, Optional[float]]
        Snapshot after work.
    field : str
        Field to subtract.

    Returns
    -------
    Optional[float]
        Delta in MiB, or None when either side is missing.
    """
    before_value = before.get(field)
    after_value = after.get(field)
    if before_value is None or after_value is None:
        return None
    return round(float(after_value) - float(before_value), 2)


def _format_delta(value: Optional[float]) -> str:
    """Format a memory delta for stable log output.

    Parameters
    ----------
    value : Optional[float]
        Delta value in MiB.

    Returns
    -------
    str
        Delta formatted with two decimals, or ``None`` when missing.
    """
    if value is None:
        return "None"
    return f"{value:.2f}"


def collect_dataframe_with_memory_logging(
    label: str,
    collect_func: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Run a DataFrame collect operation and log its memory impact.

    Parameters
    ----------
    label : str
        Label identifying the HANA query or table.
    collect_func : Callable[[], pd.DataFrame]
        Function that performs the collection and returns a DataFrame.

    Returns
    -------
    pd.DataFrame
        The collected DataFrame.
    """
    configure_memory_file_logging()
    before = get_process_memory()
    df = collect_func()
    after = get_process_memory()
    summary = summarize_dataframe(label, df)

    logger.info(
        "hana_collect label=%s rows=%s columns=%s dataframe_mb=%.2f "
        "rss_delta_mb=%s uss_delta_mb=%s before_%s after_%s",
        label,
        summary["rows"],
        summary["columns"],
        summary["memory_mb"],
        _format_delta(_memory_delta(before, after, "rss_mb")),
        _format_delta(_memory_delta(before, after, "uss_mb")),
        _format_memory_for_log(before),
        _format_memory_for_log(after),
    )
    return df


def memory_logged_tool(tool_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorate an agent tool function with before/after memory logging.

    Parameters
    ----------
    tool_name : Optional[str]
        Explicit tool name for logs. Defaults to the wrapped function name.

    Returns
    -------
    Callable[[F], F]
        Decorator that preserves the wrapped function signature for LangChain.
    """
    def decorator(func: F) -> F:
        name = tool_name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                configure_memory_file_logging()
                before = get_process_memory()
                logger.info(
                    "tool_memory stage=before tool=%s %s",
                    name,
                    _format_memory_for_log(before),
                )
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    error_snapshot = get_process_memory()
                    logger.exception(
                        "tool_memory stage=error tool=%s rss_delta_mb=%s uss_delta_mb=%s %s",
                        name,
                        _format_delta(_memory_delta(before, error_snapshot, "rss_mb")),
                        _format_delta(_memory_delta(before, error_snapshot, "uss_mb")),
                        _format_memory_for_log(error_snapshot),
                    )
                    raise
                finally:
                    gc.collect()
                    after = get_process_memory()
                    logger.info(
                        "tool_memory stage=after tool=%s rss_delta_mb=%s uss_delta_mb=%s %s",
                        name,
                        _format_delta(_memory_delta(before, after, "rss_mb")),
                        _format_delta(_memory_delta(before, after, "uss_mb")),
                        _format_memory_for_log(after),
                    )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            configure_memory_file_logging()
            before = get_process_memory()
            logger.info(
                "tool_memory stage=before tool=%s %s",
                name,
                _format_memory_for_log(before),
            )
            try:
                return func(*args, **kwargs)
            except Exception:
                error_snapshot = get_process_memory()
                logger.exception(
                    "tool_memory stage=error tool=%s rss_delta_mb=%s uss_delta_mb=%s %s",
                    name,
                    _format_delta(_memory_delta(before, error_snapshot, "rss_mb")),
                    _format_delta(_memory_delta(before, error_snapshot, "uss_mb")),
                    _format_memory_for_log(error_snapshot),
                )
                raise
            finally:
                gc.collect()
                after = get_process_memory()
                logger.info(
                    "tool_memory stage=after tool=%s rss_delta_mb=%s uss_delta_mb=%s %s",
                    name,
                    _format_delta(_memory_delta(before, after, "rss_mb")),
                    _format_delta(_memory_delta(before, after, "uss_mb")),
                    _format_memory_for_log(after),
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def instrument_langchain_tool(tool: Any) -> Any:
    """Wrap a LangChain tool object's callable with memory logging.

    Parameters
    ----------
    tool : Any
        LangChain ``StructuredTool``-like object with a ``name`` and ``func``.

    Returns
    -------
    Any
        The same tool object with its callable instrumented.
    """
    if getattr(tool, "_memory_logging_enabled", False):
        return tool

    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown_tool")
    func = getattr(tool, "func", None)
    if callable(func):
        object.__setattr__(tool, "func", memory_logged_tool(str(name))(func))

    coroutine = getattr(tool, "coroutine", None)
    if callable(coroutine):
        object.__setattr__(tool, "coroutine", memory_logged_tool(str(name))(coroutine))

    object.__setattr__(tool, "_memory_logging_enabled", True)
    return tool


def _stored_result_size_bytes(session_manager: Any) -> int:
    """Calculate bytes retained by stored JSON result strings.

    Parameters
    ----------
    session_manager : Any
        SessionManager-like object.

    Returns
    -------
    int
        Total UTF-8 encoded bytes for stored report result strings.
    """
    total = 0
    for attr in (
        "_last_forecast_results",
        "_last_scenario_comparison",
        "_last_explanation",
        "_last_sensitivity",
        "_last_yoy_actuals",
    ):
        value = getattr(session_manager, attr, None)
        if isinstance(value, str):
            total += len(value.encode("utf-8"))
    return total


def _summarize_session(session: Any) -> Dict[str, Any]:
    """Build a JSON-safe memory summary for one user session.

    Parameters
    ----------
    session : Any
        UserSession-like object from SessionStore.

    Returns
    -------
    Dict[str, Any]
        Per-session memory and state summary.
    """
    manager = session.session_manager
    state = getattr(manager, "_state", None) or {}
    scenarios = state.get("scenarios", {})
    predictions = state.get("predictions", {})

    scenario_frames = [
        summarize_dataframe(name, getattr(scenario, "df", None))
        for name, scenario in scenarios.items()
    ]
    prediction_frames = [
        summarize_dataframe(name, getattr(prediction, "predictions_df", None))
        for name, prediction in predictions.items()
    ]

    stored_bytes = _stored_result_size_bytes(manager)
    messages = state.get("messages", [])

    return {
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "models_loaded": getattr(manager, "_models_loaded", False),
        "num_scenarios": len(scenarios),
        "scenario_rows": sum(item["rows"] for item in scenario_frames),
        "scenario_memory_mb": round(sum(item["memory_mb"] for item in scenario_frames), 2),
        "num_predictions": len(predictions),
        "prediction_rows": sum(item["rows"] for item in prediction_frames),
        "prediction_memory_mb": round(sum(item["memory_mb"] for item in prediction_frames), 2),
        "stored_result_mb": _bytes_to_mb(stored_bytes) or 0.0,
        "messages_count": len(messages),
        "session_plot_files": len(getattr(manager, "_session_plot_files", [])),
        "session_export_files": len(getattr(manager, "_session_export_files", [])),
        "scenario_frames": scenario_frames,
        "prediction_frames": prediction_frames,
    }


def summarize_sessions() -> Dict[str, Any]:
    """Summarize active chatbot sessions and their retained data.

    Returns
    -------
    Dict[str, Any]
        Session store settings plus per-session memory summaries.
    """
    from app.agent.session_store import get_session_store

    store = get_session_store()
    with store._lock:  # Diagnostic endpoint needs a consistent snapshot.
        sessions = [_summarize_session(session) for session in store._sessions.values()]

    return {
        "active_sessions": len(sessions),
        "max_sessions": store.max_sessions,
        "session_ttl_hours": store.session_ttl_hours,
        "total_scenario_memory_mb": round(
            sum(session["scenario_memory_mb"] for session in sessions), 2
        ),
        "total_prediction_memory_mb": round(
            sum(session["prediction_memory_mb"] for session in sessions), 2
        ),
        "total_stored_result_mb": round(
            sum(session["stored_result_mb"] for session in sessions), 2
        ),
        "sessions": sessions,
    }


def build_memory_snapshot() -> Dict[str, Any]:
    """Build a complete API memory diagnostics snapshot.

    Returns
    -------
    Dict[str, Any]
        Process, session, and garbage collector diagnostics suitable for the
        ``/api/admin/memory`` endpoint.
    """
    configure_memory_file_logging()
    return {
        "timestamp": datetime.now().isoformat(),
        "process": get_process_memory(),
        "sessions": summarize_sessions(),
        "gc": {
            "counts": list(gc.get_count()),
            "thresholds": list(gc.get_threshold()),
        },
        "memory_log_file": os.getenv("MEMORY_LOG_FILE"),
    }
