"""Tests for admin maintenance routes."""

from __future__ import annotations

import pytest

from app.routers import admin


@pytest.mark.asyncio
async def test_get_memory_snapshot_returns_diagnostics(monkeypatch) -> None:
    """Memory admin handler returns process and session diagnostics."""
    monkeypatch.setattr(
        admin,
        "build_memory_snapshot",
        lambda: {
            "process": {"rss_mb": 123.4},
            "sessions": {"active_sessions": 2},
        },
    )

    result = await admin.get_memory_snapshot()

    assert result["process"]["rss_mb"] == 123.4
    assert result["sessions"]["active_sessions"] == 2
