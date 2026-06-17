"""Tests for API memory instrumentation helpers."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from app.services import memory_monitor


def test_summarize_dataframe_reports_shape_and_memory() -> None:
    """A DataFrame summary includes shape and positive memory usage."""
    df = pd.DataFrame({"store": [101, 102], "dma": ["NYC", "LA"]})

    summary = memory_monitor.summarize_dataframe("sample", df)

    assert summary["name"] == "sample"
    assert summary["rows"] == 2
    assert summary["columns"] == 2
    assert summary["memory_mb"] > 0


def test_collect_dataframe_with_memory_logging_logs_result(
    monkeypatch,
    caplog,
) -> None:
    """HANA collect wrapper logs memory and DataFrame shape after collection."""
    snapshots = iter(
        [
            {"rss_mb": 100.0, "uss_mb": 80.0, "source": "test"},
            {"rss_mb": 125.5, "uss_mb": 101.0, "source": "test"},
        ]
    )
    monkeypatch.setattr(memory_monitor, "get_process_memory", lambda: next(snapshots))

    caplog.set_level(logging.INFO, logger=memory_monitor.LOGGER_NAME)

    df = memory_monitor.collect_dataframe_with_memory_logging(
        "MODEL_B",
        lambda: pd.DataFrame({"a": [1, 2, 3]}),
    )

    assert len(df) == 3
    assert "hana_collect label=MODEL_B" in caplog.text
    assert "rows=3" in caplog.text
    assert "rss_delta_mb=25.50" in caplog.text


def test_memory_logged_tool_logs_before_and_after(monkeypatch, caplog) -> None:
    """Tool decorator logs before and after snapshots around wrapped functions."""
    snapshots = iter(
        [
            {"rss_mb": 42.0, "uss_mb": 21.0, "source": "test"},
            {"rss_mb": 45.25, "uss_mb": 22.5, "source": "test"},
        ]
    )
    monkeypatch.setattr(memory_monitor, "get_process_memory", lambda: next(snapshots))
    caplog.set_level(logging.INFO, logger=memory_monitor.LOGGER_NAME)

    @memory_monitor.memory_logged_tool("demo_tool")
    def demo_tool(value: int) -> int:
        """Return twice the input value."""
        return value * 2

    assert demo_tool(5) == 10
    assert "tool_memory stage=before tool=demo_tool" in caplog.text
    assert "tool_memory stage=after tool=demo_tool" in caplog.text
    assert "rss_delta_mb=3.25" in caplog.text


@pytest.mark.asyncio
async def test_memory_logged_tool_logs_async_tools(monkeypatch, caplog) -> None:
    """Tool decorator logs around coroutine tool functions after they await."""
    snapshots = iter(
        [
            {"rss_mb": 70.0, "uss_mb": 35.0, "source": "test"},
            {"rss_mb": 74.5, "uss_mb": 37.0, "source": "test"},
        ]
    )
    monkeypatch.setattr(memory_monitor, "get_process_memory", lambda: next(snapshots))
    caplog.set_level(logging.INFO, logger=memory_monitor.LOGGER_NAME)

    @memory_monitor.memory_logged_tool("async_tool")
    async def async_tool(value: int) -> int:
        """Return one more than the input value."""
        return value + 1

    assert await async_tool(7) == 8
    assert "tool_memory stage=before tool=async_tool" in caplog.text
    assert "tool_memory stage=after tool=async_tool" in caplog.text
    assert "rss_delta_mb=4.50" in caplog.text


def test_instrument_langchain_tool_wraps_tool_func(monkeypatch, caplog) -> None:
    """Tool object instrumentation wraps the func callable once."""
    snapshots = iter(
        [
            {"rss_mb": 10.0, "uss_mb": 5.0, "source": "test"},
            {"rss_mb": 12.0, "uss_mb": 6.0, "source": "test"},
        ]
    )
    monkeypatch.setattr(memory_monitor, "get_process_memory", lambda: next(snapshots))
    caplog.set_level(logging.INFO, logger=memory_monitor.LOGGER_NAME)

    class FakeTool:
        """Small stand-in for a LangChain StructuredTool."""

        name = "fake_tool"

        def __init__(self) -> None:
            """Initialize the fake tool with a callable func."""
            self.func = increment_value

    def increment_value(value: int) -> int:
        """Return one more than the input value."""
        return value + 1

    tool = FakeTool()
    wrapped = memory_monitor.instrument_langchain_tool(tool)

    assert wrapped is tool
    assert tool.func(2) == 3
    assert memory_monitor.instrument_langchain_tool(tool) is tool
    assert "tool_memory stage=after tool=fake_tool" in caplog.text
    assert "rss_delta_mb=2.00" in caplog.text
