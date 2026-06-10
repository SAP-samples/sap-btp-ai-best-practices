"""LangGraph tools for querying eligibility pattern analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from ...models.patterns import PatternFilters
from ...services.eligibility.pattern_analyzer import PatternAnalyzer
from .custom import to_json

_analyzer: Optional[PatternAnalyzer] = None


def _get_analyzer() -> PatternAnalyzer:
    """Lazy singleton for PatternAnalyzer (matches router pattern)."""
    global _analyzer
    if _analyzer is None:
        _analyzer = PatternAnalyzer()
    return _analyzer


@tool("get_eligibility_patterns")
def get_eligibility_patterns(
    seller_id: Optional[str] = None,
    debtor_id: Optional[str] = None,
    programa: Optional[str] = None,
    insurer_id: Optional[str] = None,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Detect recurring non-eligibility patterns across historical invoice data.

    Runs all pattern detection algorithms (chronic rule failures, trending
    increases, repeat offenders, rule concentration, amount at risk) and
    returns a severity-sorted summary. Use this to identify systemic
    eligibility issues for a seller, debtor, program, insurer, or across all.
    """
    analyzer = _get_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=int(lookback_days),
    )
    summary = analyzer.analyze_all(filters=filters)
    return to_json(summary)


@tool("get_debtor_rejection_profile")
def get_debtor_rejection_profile(
    debtor_id: Optional[str] = None,
    debtor_name: Optional[str] = None,
    seller_id: Optional[str] = None,
    programa: Optional[str] = None,
    insurer_id: Optional[str] = None,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Get per-debtor non-eligibility profiles with dominant rule and amount data.

    Returns one profile per (seller, debtor) pair, sorted by non-eligible
    invoice count descending. Each profile includes the dominant non-eligibility
    rule, rule breakdown, total amount at risk, and batch-level statistics.

    Filter by debtor_id (exact match) or debtor_name (case-insensitive
    partial match). When neither is provided, returns all profiles.
    Optionally filter by programa or insurer_id.
    """
    analyzer = _get_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=int(lookback_days),
    )
    profiles = analyzer.get_debtor_profiles(filters=filters)
    if debtor_id:
        profiles = [p for p in profiles if p.debtor_id == debtor_id]
    elif debtor_name:
        needle = debtor_name.lower()
        profiles = [
            p for p in profiles
            if p.debtor_name and needle in p.debtor_name.lower()
        ]
    return {
        "count": len(profiles),
        "profiles": to_json(profiles),
    }


@tool("get_eligibility_trend")
def get_eligibility_trend(
    seller_id: Optional[str] = None,
    debtor_id: Optional[str] = None,
    programa: Optional[str] = None,
    insurer_id: Optional[str] = None,
    granularity: str = "week",
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Get non-eligibility rate time-series for charting eligibility trends.

    Returns data points bucketed by the chosen granularity (day, week, month,
    or quarter) over the lookback window. Each point includes total invoices,
    non-eligible invoices, non-eligibility rate, and per-rule counts.
    Optionally filter by seller, debtor, program, and/or insurer.
    """
    analyzer = _get_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=int(lookback_days),
    )
    points = analyzer.get_rejection_trend(
        granularity=granularity,
        filters=filters,
    )
    return {
        "count": len(points),
        "trend": to_json(points),
    }


@tool("get_eligibility_pattern_insights")
def get_eligibility_pattern_insights(
    seller_id: Optional[str] = None,
    debtor_id: Optional[str] = None,
    programa: Optional[str] = None,
    insurer_id: Optional[str] = None,
    lookback_days: int = 180,
) -> Dict[str, Any]:
    """Get combined pattern analysis outputs used by the eligibility insights view.

    Args:
        seller_id: Optional seller filter.
        debtor_id: Optional debtor filter.
        programa: Optional program filter.
        insurer_id: Optional insurer filter.
        lookback_days: Number of days to analyze. Defaults to 180.

    Returns:
        A compact envelope containing:
            - normalized filters metadata,
            - `patterns`: combined pattern alerts,
            - `debtor_profiles`: per-debtor non-eligibility summaries,
            - `trend`: non-eligibility trend points (weekly).
    """
    analyzer = _get_analyzer()
    filters = PatternFilters(
        seller_id=seller_id,
        debtor_id=debtor_id,
        programa=programa,
        insurer_id=insurer_id,
        lookback_days=lookback_days,
        min_invoices=3,
    )

    summary = analyzer.analyze_all(filters=filters)
    profiles = analyzer.get_debtor_profiles(filters=filters)
    trend = analyzer.get_rejection_trend(granularity="week", filters=filters)

    return {
        "filters": {
            "seller_id": seller_id,
            "debtor_id": debtor_id,
            "programa": programa,
            "insurer_id": insurer_id,
            "lookback_days": lookback_days,
        },
        "patterns": to_json(summary),
        "debtor_profiles": to_json(profiles),
        "trend": to_json(trend),
    }


ELIGIBILITY_PATTERN_TOOLS = [
    get_eligibility_patterns,
    get_debtor_rejection_profile,
    get_eligibility_trend,
    get_eligibility_pattern_insights,
]
