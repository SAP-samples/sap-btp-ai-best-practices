"""Timing helpers for metal composition workflow instrumentation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now_iso_after(minimum_exclusive: str | None = None) -> str:
    current = utc_now_iso()
    if minimum_exclusive and current <= minimum_exclusive:
        floor = _parse_utc_iso(minimum_exclusive)
        if floor is not None:
            return (floor + timedelta(milliseconds=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return current


def _parse_utc_iso(value: str | None) -> datetime | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _round_ms(seconds: float) -> float:
    return round(max(0.0, float(seconds)) * 1000.0, 2)


def finish_timing(
    started_perf: float,
    started_at: str,
    *,
    status: str = "completed",
    details: Dict[str, Any] | None = None,
    substeps: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
        "duration_ms": _round_ms(perf_counter() - started_perf),
    }
    if details:
        payload["details"] = details
    if substeps:
        payload["substeps"] = substeps
    return payload


def rank_timings(spans: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    ranked = []
    for name, span in spans.items():
        ranked.append(
            {
                "name": name,
                "duration_ms": round(float(span.get("duration_ms", 0.0) or 0.0), 2),
                "status": str(span.get("status", "unknown")),
            }
        )
    ranked.sort(key=lambda item: item["duration_ms"], reverse=True)
    return ranked


def summarize_parallel_timings(
    spans: Dict[str, Dict[str, Any]],
    *,
    wall_clock_duration_ms: float,
) -> Dict[str, Any]:
    ranked = rank_timings(spans)
    sequential_equivalent_ms = round(sum(item["duration_ms"] for item in ranked), 2)
    bottleneck = ranked[0]["name"] if ranked else None
    return {
        "parallel_wall_clock_duration_ms": round(float(wall_clock_duration_ms), 2),
        "sequential_equivalent_duration_ms": sequential_equivalent_ms,
        "estimated_parallel_time_saved_ms": round(
            max(0.0, sequential_equivalent_ms - float(wall_clock_duration_ms)),
            2,
        ),
        "bottleneck": bottleneck,
        "ranked_by_duration": ranked,
    }
