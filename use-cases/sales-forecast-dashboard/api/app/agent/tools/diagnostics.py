"""
Diagnostic Tools for Sales Forecast Agent.

These tools provide high-level business health analysis by examining pre-computed
data from DMA (market) level down to individual profit centers (stores).

Tools:
- get_business_health_overview: Entry point for "What's wrong with my business?"
- get_dma_diagnostic: Drill down into specific DMAs with store breakdown
- get_underperforming_stores: Find stores below performance thresholds
- get_store_diagnostic: Deep dive with SHAP driver analysis
- get_performance_ranking: Rank stores/DMAs by various metrics
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

# Data directory for pre-computed JSON files
DATA_DIR = Path(__file__).parent.parent.parent / "data"


# =============================================================================
# Helper Functions
# =============================================================================

def _load_dma_summary() -> List[dict]:
    """Load DMA summary data from JSON file."""
    with open(DATA_DIR / "dma_summary.json") as f:
        return json.load(f)


def _load_stores() -> List[dict]:
    """Load stores data from JSON file."""
    with open(DATA_DIR / "stores.json") as f:
        return json.load(f)


def _load_store_timeseries(store_id: int) -> Optional[dict]:
    """Load timeseries data for a specific store."""
    path = DATA_DIR / "timeseries" / f"store_{store_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_dma_timeseries(dma_name: str) -> Optional[dict]:
    """Load timeseries data for a DMA."""
    safe_name = dma_name.replace("/", "_").replace(" ", "_")
    path = DATA_DIR / "timeseries" / f"dma_{safe_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_store_yoy(store: dict, channel: str = "B&M") -> Optional[float]:
    """
    Get YoY change for a store based on channel.

    Args:
        store: Store data dictionary
        channel: "B&M" or "WEB"

    Returns:
        YoY percentage change for the specified channel, or None if not available
    """
    if channel == "WEB":
        return store.get("web_yoy_auv_change")
    # Default to B&M (includes "B&M" and any other value)
    return store.get("bm_yoy_auv_change")


def _get_dma_yoy(dma: dict, channel: str = "B&M") -> Optional[float]:
    """
    Get YoY change for a DMA based on channel.

    Args:
        dma: DMA data dictionary
        channel: "B&M" or "WEB"

    Returns:
        YoY percentage change for the specified channel, or None if not available
    """
    if channel == "WEB":
        return dma.get("web_yoy_auv_change_pct")
    # Default to B&M
    return dma.get("bm_yoy_auv_change_pct")


def _classify_yoy(
    yoy_change: Optional[float],
    red_threshold: float = -5.0,
    green_threshold: float = 5.0
) -> str:
    """
    Classify YoY change into performance category.

    Args:
        yoy_change: Year-over-year percentage change (None for new entities)
        red_threshold: YoY below this is "declining"
        green_threshold: YoY above this is "growing"

    Returns:
        One of: "declining", "stable", "growing", "no_baseline"
    """
    if yoy_change is None:
        return "no_baseline"
    elif yoy_change < red_threshold:
        return "declining"
    elif yoy_change > green_threshold:
        return "growing"
    else:
        return "stable"


def _aggregate_shap_features(timeseries: List[dict], top_n: int = 5) -> Dict[str, Any]:
    """
    Aggregate SHAP features across all weeks in timeseries.

    Computes average impact per feature across all time points.

    Args:
        timeseries: List of weekly data points with shap_features
        top_n: Number of top contributors to return

    Returns:
        Dict with top_positive_drivers and top_negative_drivers lists
    """
    feature_impacts = defaultdict(list)

    for point in timeseries:
        shap_features = point.get("shap_features", [])
        for sf in shap_features:
            feature = sf.get("feature")
            impact = sf.get("impact", 0)
            if feature:
                feature_impacts[feature].append(impact)

    if not feature_impacts:
        return {
            "top_positive_drivers": [],
            "top_negative_drivers": [],
        }

    # Compute average impact per feature
    avg_impacts = {
        f: sum(impacts) / len(impacts)
        for f, impacts in feature_impacts.items()
    }

    # Split into positive and negative, sort by magnitude
    positive = [(f, v) for f, v in avg_impacts.items() if v > 0]
    negative = [(f, v) for f, v in avg_impacts.items() if v < 0]

    positive.sort(key=lambda x: -x[1])  # Descending by impact
    negative.sort(key=lambda x: x[1])   # Ascending (most negative first)

    top_positive = [
        {
            "feature": f,
            "avg_impact": round(v, 4),
            "weeks_present": len(feature_impacts[f]),
            "direction": "positive"
        }
        for f, v in positive[:top_n]
    ]

    top_negative = [
        {
            "feature": f,
            "avg_impact": round(v, 4),
            "weeks_present": len(feature_impacts[f]),
            "direction": "negative"
        }
        for f, v in negative[:top_n]
    ]

    return {
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
    }


def _generate_shap_interpretation(shap_analysis: Dict[str, Any]) -> str:
    """
    Generate natural language interpretation of SHAP analysis.

    Args:
        shap_analysis: Output from _aggregate_shap_features

    Returns:
        Human-readable interpretation string
    """
    positive = shap_analysis.get("top_positive_drivers", [])
    negative = shap_analysis.get("top_negative_drivers", [])

    parts = []

    if positive:
        top_pos = positive[0]
        parts.append(
            f"{top_pos['feature']} is the strongest positive contributor "
            f"(avg impact: {top_pos['avg_impact']:+.3f})"
        )

    if negative:
        top_neg = negative[0]
        parts.append(
            f"{top_neg['feature']} is the strongest negative contributor "
            f"(avg impact: {top_neg['avg_impact']:+.3f})"
        )

    if not parts:
        return "No significant SHAP drivers identified."

    return ". ".join(parts) + "."


def _compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std = variance ** 0.5

    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "min": round(min(sorted_vals), 2),
        "max": round(max(sorted_vals), 2),
    }


# =============================================================================
# Tool 1: get_business_health_overview
# =============================================================================

@tool
def get_business_health_overview(
    channel: str = "B&M",
    include_details: bool = True,
    yoy_threshold_red: float = -5.0,
    yoy_threshold_green: float = 5.0,
) -> Dict[str, Any]:
    """
    Get a high-level snapshot of business health across all DMAs and stores.

    This is the primary entry point for answering questions like
    "What is wrong with my business?" It provides:
    - Overall health distribution (declining/stable/growing)
    - Natural language concerns and highlights
    - Top declining and growing DMAs

    Note: The average YoY change is calculated as a sales-weighted average,
    where larger stores have proportionally more impact on the overall metric.
    This matches the dashboard calculation for consistency.

    Args:
        channel: Sales channel to analyze - "B&M" (default) or "WEB"
        include_details: If True, include lists of best/worst performing DMAs
        yoy_threshold_red: YoY percentage below this is considered "declining" (default -5%)
        yoy_threshold_green: YoY percentage above this is considered "growing" (default 5%)

    Returns:
        Dict with business health overview including:
        - summary: Total counts, aggregate metrics, and weighted_avg_yoy_change_pct
        - health_distribution: Counts by performance category for DMAs and stores
        - concerns: List of issues requiring attention
        - highlights: List of positive trends
        - top_declining_dmas: Worst performing markets (if include_details=True)
        - top_growing_dmas: Best performing markets (if include_details=True)

    Example:
        >>> get_business_health_overview(channel="B&M")
        {"status": "success", "summary": {"total_dmas": 56, ...}, ...}
    """
    try:
        dmas = _load_dma_summary()
        stores = _load_stores()
    except FileNotFoundError as e:
        return {
            "error": f"Data file not found: {e}",
            "hint": "Run regenerate_dashboard_data.py to create the data files."
        }

    # Classify DMAs using channel-specific YoY
    dma_categories = {"declining": [], "stable": [], "growing": [], "no_baseline": []}
    for dma in dmas:
        category = _classify_yoy(
            _get_dma_yoy(dma, channel),
            yoy_threshold_red,
            yoy_threshold_green
        )
        dma_categories[category].append(dma)

    # Classify stores using channel-specific YoY
    store_categories = {"declining": [], "stable": [], "growing": [], "no_baseline": []}
    for store in stores:
        category = _classify_yoy(
            _get_store_yoy(store, channel),
            yoy_threshold_red,
            yoy_threshold_green
        )
        store_categories[category].append(store)

    # Compute aggregate metrics using channel-specific AUV
    auv_field = "bm_auv_p50" if channel == "B&M" else "web_auv_p50"
    total_auv = sum(s.get(auv_field, 0) for s in stores)

    # Weighted average YoY (weighted by AUV volume)
    # This ensures larger stores have proportional impact on the overall metric,
    # matching the dashboard calculation for consistency
    stores_with_yoy = [
        s for s in stores
        if _get_store_yoy(s, channel) is not None and s.get(auv_field, 0) > 0
    ]
    if stores_with_yoy:
        weighted_yoy_sum = sum(
            _get_store_yoy(s, channel) * s.get(auv_field, 0)
            for s in stores_with_yoy
        )
        total_yoy_auv = sum(s.get(auv_field, 0) for s in stores_with_yoy)
        avg_yoy = weighted_yoy_sum / total_yoy_auv if total_yoy_auv > 0 else None
    else:
        avg_yoy = None

    # Generate concerns
    concerns = []
    declining_dma_count = len(dma_categories["declining"])
    declining_store_count = len(store_categories["declining"])

    if declining_dma_count > 0:
        concerns.append(f"{declining_dma_count} DMAs are declining (>{abs(yoy_threshold_red)}% YoY decrease) in {channel}")
        worst_dma = min(dma_categories["declining"], key=lambda x: _get_dma_yoy(x, channel) or 0)
        worst_yoy = _get_dma_yoy(worst_dma, channel)
        concerns.append(
            f"{worst_dma['dma']} has the worst YoY change at {worst_yoy:.2f}%" if worst_yoy else f"{worst_dma['dma']} is declining"
        )

    if declining_store_count > 0:
        concerns.append(f"{declining_store_count} stores are underperforming their {channel} baseline")

    # Generate highlights
    highlights = []
    growing_dma_count = len(dma_categories["growing"])
    growing_store_count = len(store_categories["growing"])

    if growing_dma_count > 0:
        highlights.append(f"{growing_dma_count} DMAs are growing (>{yoy_threshold_green}% YoY increase) in {channel}")
        best_dma = max(dma_categories["growing"], key=lambda x: _get_dma_yoy(x, channel) or 0)
        best_yoy = _get_dma_yoy(best_dma, channel)
        highlights.append(
            f"{best_dma['dma']} has the best YoY improvement at +{best_yoy:.2f}%" if best_yoy else f"{best_dma['dma']} is growing"
        )

    if growing_store_count > 0:
        highlights.append(f"{growing_store_count} stores are outperforming their {channel} baseline")

    result = {
        "status": "success",
        "channel": channel,
        "summary": {
            "total_dmas": len(dmas),
            "total_stores": len(stores),
            "total_auv": round(total_auv, 2),
            "weighted_avg_yoy_change_pct": round(avg_yoy, 2) if avg_yoy is not None else None,
        },
        "health_distribution": {
            "dmas": {
                "declining": len(dma_categories["declining"]),
                "stable": len(dma_categories["stable"]),
                "growing": len(dma_categories["growing"]),
                "no_baseline": len(dma_categories["no_baseline"]),
            },
            "stores": {
                "declining": len(store_categories["declining"]),
                "stable": len(store_categories["stable"]),
                "growing": len(store_categories["growing"]),
                "no_baseline": len(store_categories["no_baseline"]),
            },
        },
        "concerns": concerns if concerns else ["No significant concerns identified."],
        "highlights": highlights if highlights else ["No significant growth highlights."],
        "hint": "Use get_dma_diagnostic for detailed DMA analysis or get_underperforming_stores for store-level issues."
    }

    if include_details:
        # Get channel-specific AUV field for DMA
        dma_auv_field = "bm_total_auv_p50" if channel == "B&M" else "web_total_auv_p50"

        # Top declining DMAs (sorted by YoY ascending)
        declining_sorted = sorted(
            dma_categories["declining"],
            key=lambda x: _get_dma_yoy(x, channel) or 0
        )
        result["top_declining_dmas"] = [
            {
                "dma": d["dma"],
                "yoy_change_pct": _get_dma_yoy(d, channel),
                "store_count": d.get("store_count"),
                "total_auv": d.get(dma_auv_field),
            }
            for d in declining_sorted[:5]
        ]

        # Top growing DMAs (sorted by YoY descending)
        growing_sorted = sorted(
            dma_categories["growing"],
            key=lambda x: _get_dma_yoy(x, channel) or 0,
            reverse=True
        )
        result["top_growing_dmas"] = [
            {
                "dma": d["dma"],
                "yoy_change_pct": _get_dma_yoy(d, channel),
                "store_count": d.get("store_count"),
                "total_auv": d.get(dma_auv_field),
            }
            for d in growing_sorted[:5]
        ]

    return result


# =============================================================================
# Tool 2: get_dma_diagnostic
# =============================================================================

@tool
def get_dma_diagnostic(
    channel: str = "B&M",
    dma_names: Optional[List[str]] = None,
    include_stores: bool = True,
    include_timeseries_summary: bool = False,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Drill down into specific DMAs with store-level breakdown.

    Provides detailed analysis of market performance including:
    - DMA-level metrics and YoY status
    - Store breakdown by performance category
    - Top declining and growing stores within each DMA
    - Generated insights

    Args:
        channel: Sales channel to analyze - "B&M" (default) or "WEB"
        dma_names: List of DMA names to analyze. If None, analyzes all declining DMAs.
        include_stores: If True, include store-level breakdown for each DMA
        include_timeseries_summary: If True, include weekly trend summary
        top_n: Number of stores per category to return (default 5)

    Returns:
        Dict with detailed DMA diagnostics including:
        - dmas: List of DMA analyses with overview and store breakdowns
        - comparison: Summary if multiple DMAs requested

    Example:
        >>> get_dma_diagnostic(channel="B&M", dma_names=["BOSTON/NH", "NEW YORK"])
        {"status": "success", "dmas": [...], "comparison": {...}}
    """
    try:
        dmas = _load_dma_summary()
        stores = _load_stores()
    except FileNotFoundError as e:
        return {
            "error": f"Data file not found: {e}",
            "hint": "Run regenerate_dashboard_data.py to create the data files."
        }

    # Build store lookup by DMA
    stores_by_dma = defaultdict(list)
    for store in stores:
        stores_by_dma[store.get("dma")].append(store)

    # Determine which DMAs to analyze using channel-specific YoY
    if dma_names is None:
        # Default to all declining DMAs for the specified channel
        target_dmas = [
            d for d in dmas
            if _classify_yoy(_get_dma_yoy(d, channel)) == "declining"
        ]
        if not target_dmas:
            return {
                "status": "success",
                "message": "No declining DMAs found.",
                "dmas": [],
                "hint": "All DMAs are stable or growing. Use dma_names parameter to analyze specific DMAs."
            }
    else:
        # Find requested DMAs
        dma_lookup = {d["dma"]: d for d in dmas}
        target_dmas = []
        missing = []
        for name in dma_names:
            if name in dma_lookup:
                target_dmas.append(dma_lookup[name])
            else:
                missing.append(name)

        if missing:
            available = list(dma_lookup.keys())[:10]
            return {
                "error": f"DMAs not found: {missing}",
                "hint": f"Available DMAs include: {available}...",
                "available_count": len(dma_lookup),
            }

    # Analyze each DMA
    dma_results = []
    for dma_data in target_dmas:
        dma_name = dma_data["dma"]
        dma_stores = stores_by_dma.get(dma_name, [])

        # Classify stores within this DMA using channel-specific YoY
        store_cats = {"declining": [], "stable": [], "growing": [], "no_baseline": []}
        for store in dma_stores:
            cat = _classify_yoy(_get_store_yoy(store, channel))
            store_cats[cat].append(store)

        # Get channel-specific fields
        dma_auv_field = "bm_total_auv_p50" if channel == "B&M" else "web_total_auv_p50"
        dma_auv_p90_field = "bm_total_auv_p90" if channel == "B&M" else "web_total_auv_p90"
        dma_status_field = "bm_yoy_status" if channel == "B&M" else "web_yoy_status"

        dma_result = {
            "dma": dma_name,
            "channel": channel,
            "overview": {
                "store_count": dma_data.get("store_count"),
                "total_auv": dma_data.get(dma_auv_field),
                "total_auv_p90": dma_data.get(dma_auv_p90_field),
                "yoy_status": dma_data.get(dma_status_field),
                "yoy_change_pct": _get_dma_yoy(dma_data, channel),
                "lat": dma_data.get("lat"),
                "lng": dma_data.get("lng"),
            },
            "store_breakdown": {
                "declining": len(store_cats["declining"]),
                "stable": len(store_cats["stable"]),
                "growing": len(store_cats["growing"]),
                "no_baseline": len(store_cats["no_baseline"]),
            },
        }

        if include_stores:
            # Get store-level AUV field
            store_auv_field = "bm_auv_p50" if channel == "B&M" else "web_auv_p50"

            # Top declining stores using channel-specific YoY
            declining_sorted = sorted(
                store_cats["declining"],
                key=lambda x: _get_store_yoy(x, channel) or 0
            )
            dma_result["top_declining_stores"] = [
                {
                    "id": s["id"],
                    "name": s.get("name"),
                    "auv_p50": s.get(store_auv_field),
                    "yoy_change": _get_store_yoy(s, channel),
                    "is_outlet": s.get("is_outlet"),
                    "city": s.get("city"),
                    "state": s.get("state"),
                }
                for s in declining_sorted[:top_n]
            ]

            # Top growing stores using channel-specific YoY
            growing_sorted = sorted(
                store_cats["growing"],
                key=lambda x: _get_store_yoy(x, channel) or 0,
                reverse=True
            )
            dma_result["top_growing_stores"] = [
                {
                    "id": s["id"],
                    "name": s.get("name"),
                    "auv_p50": s.get(store_auv_field),
                    "yoy_change": _get_store_yoy(s, channel),
                    "is_outlet": s.get("is_outlet"),
                    "city": s.get("city"),
                    "state": s.get("state"),
                }
                for s in growing_sorted[:top_n]
            ]

        if include_timeseries_summary:
            ts_data = _load_dma_timeseries(dma_name)
            if ts_data and ts_data.get("timeseries"):
                ts = ts_data["timeseries"]
                sales_values = [p.get("pred_sales_p50", 0) for p in ts]
                if sales_values:
                    peak_idx = sales_values.index(max(sales_values))
                    trough_idx = sales_values.index(min(sales_values))

                    # Determine trend: compare first half avg to second half avg
                    mid = len(sales_values) // 2
                    first_half_avg = sum(sales_values[:mid]) / mid if mid > 0 else 0
                    second_half_avg = sum(sales_values[mid:]) / (len(sales_values) - mid) if len(sales_values) > mid else 0

                    if second_half_avg > first_half_avg * 1.05:
                        trend = "improving"
                    elif second_half_avg < first_half_avg * 0.95:
                        trend = "declining"
                    else:
                        trend = "stable"

                    dma_result["timeseries_summary"] = {
                        "trend": trend,
                        "weeks_count": len(ts),
                        "recent_week_sales": sales_values[-1] if sales_values else None,
                        "peak_week": ts[peak_idx].get("date"),
                        "peak_sales": max(sales_values),
                        "trough_week": ts[trough_idx].get("date"),
                        "trough_sales": min(sales_values),
                    }

        # Generate insights using channel-specific YoY
        insights = []
        total_stores = len(dma_stores)
        declining_count = len(store_cats["declining"])

        if declining_count > 0:
            pct_declining = (declining_count / total_stores * 100) if total_stores > 0 else 0
            insights.append(f"{declining_count} of {total_stores} stores are declining in {channel} ({pct_declining:.0f}%)")

            if store_cats["declining"]:
                worst = min(store_cats["declining"], key=lambda x: _get_store_yoy(x, channel) or 0)
                worst_yoy = _get_store_yoy(worst, channel)
                insights.append(
                    f"Store #{worst['id']} ({worst.get('name', 'Unknown')}) has the largest decline "
                    f"at {worst_yoy:.1f}%" if worst_yoy else f"Store #{worst['id']} is declining"
                )

        # Check if outlets vs non-outlets differ
        outlet_stores = [s for s in dma_stores if s.get("is_outlet")]
        non_outlet_stores = [s for s in dma_stores if not s.get("is_outlet")]

        if outlet_stores and non_outlet_stores:
            outlet_yoys = [_get_store_yoy(s, channel) for s in outlet_stores if _get_store_yoy(s, channel) is not None]
            non_outlet_yoys = [_get_store_yoy(s, channel) for s in non_outlet_stores if _get_store_yoy(s, channel) is not None]

            if outlet_yoys and non_outlet_yoys:
                outlet_avg = sum(outlet_yoys) / len(outlet_yoys)
                non_outlet_avg = sum(non_outlet_yoys) / len(non_outlet_yoys)

                if abs(outlet_avg - non_outlet_avg) > 5:
                    if outlet_avg > non_outlet_avg:
                        insights.append(
                            f"Outlets outperforming non-outlets in {channel} (avg YoY: {outlet_avg:.1f}% vs {non_outlet_avg:.1f}%)"
                        )
                    else:
                        insights.append(
                            f"Non-outlets outperforming outlets in {channel} (avg YoY: {non_outlet_avg:.1f}% vs {outlet_avg:.1f}%)"
                        )

        dma_result["insights"] = insights if insights else ["No specific issues identified in this DMA."]
        dma_results.append(dma_result)

    result = {
        "status": "success",
        "channel": channel,
        "dmas": dma_results,
        "hint": "Use get_store_diagnostic to analyze SHAP drivers for specific stores."
    }

    # Add comparison if multiple DMAs
    if len(dma_results) > 1:
        yoy_values = [d["overview"].get("yoy_change_pct") for d in dma_results if d["overview"].get("yoy_change_pct") is not None]
        best = max(dma_results, key=lambda x: x["overview"].get("yoy_change_pct") or float("-inf"))
        worst = min(dma_results, key=lambda x: x["overview"].get("yoy_change_pct") or float("inf"))

        result["comparison"] = {
            "best_performer": {
                "dma": best["dma"],
                "yoy_change_pct": best["overview"].get("yoy_change_pct"),
            },
            "worst_performer": {
                "dma": worst["dma"],
                "yoy_change_pct": worst["overview"].get("yoy_change_pct"),
            },
            "avg_yoy_across_selected": round(sum(yoy_values) / len(yoy_values), 2) if yoy_values else None,
        }

    return result


# =============================================================================
# Tool 3: get_underperforming_stores
# =============================================================================

@tool
def get_underperforming_stores(
    channel: str = "B&M",
    dma_filter: Optional[List[str]] = None,
    store_filter: Optional[List[int]] = None,
    yoy_threshold: float = -5.0,
    include_outlets: bool = True,
    include_new_stores: bool = False,
    sort_by: str = "yoy_change",
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Identify stores that are performing below threshold.

    Filters stores by YoY performance and returns detailed information
    for underperformers, with summaries by DMA and store type.

    Args:
        channel: Sales channel to analyze - "B&M" (default) or "WEB"
        dma_filter: List of DMA names to filter by (None = all DMAs)
        store_filter: List of store IDs to filter by (None = all stores)
        yoy_threshold: YoY percentage below this is "underperforming" (default -5%)
        include_outlets: If True, include outlet stores in results
        include_new_stores: If True, include new stores (null YoY) in results (default False)
        sort_by: Sort field - "yoy_change" or "auv_p50" (default "yoy_change")
        limit: Maximum number of stores to return (default 20)

    Returns:
        Dict with underperforming stores including:
        - total_underperforming: Count of stores matching criteria
        - stores: List of store details with channel-specific YoY
        - summary_by_dma: Aggregation by DMA
        - summary_by_type: Aggregation by outlet/non-outlet

    Example:
        >>> get_underperforming_stores(channel="B&M", dma_filter=["BOSTON/NH"], yoy_threshold=-3.0)
        {"status": "success", "total_underperforming": 5, "stores": [...]}
    """
    try:
        stores = _load_stores()
    except FileNotFoundError as e:
        return {
            "error": f"Data file not found: {e}",
            "hint": "Run regenerate_dashboard_data.py to create the data files."
        }

    # Apply filters
    filtered = stores

    if dma_filter:
        filtered = [s for s in filtered if s.get("dma") in dma_filter]

    if store_filter:
        filtered = [s for s in filtered if s.get("id") in store_filter]

    if not include_outlets:
        filtered = [s for s in filtered if not s.get("is_outlet")]

    if not include_new_stores:
        filtered = [s for s in filtered if _get_store_yoy(s, channel) is not None]

    # Filter by YoY threshold using channel-specific YoY
    underperforming = []
    for store in filtered:
        yoy = _get_store_yoy(store, channel)
        if yoy is not None and yoy < yoy_threshold:
            underperforming.append(store)
        elif yoy is None and include_new_stores:
            # New stores without baseline - include but mark
            underperforming.append(store)

    # Sort using channel-specific fields
    if sort_by == "yoy_change":
        # Put None values last, sort by YoY ascending (worst first)
        underperforming.sort(key=lambda x: (_get_store_yoy(x, channel) is None, _get_store_yoy(x, channel) or 0))
    elif sort_by == "auv_p50":
        # Sort by AUV descending (highest sales first)
        auv_field = "bm_auv_p50" if channel == "B&M" else "web_auv_p50"
        underperforming.sort(key=lambda x: x.get(auv_field, 0), reverse=True)

    # Limit results
    limited = underperforming[:limit]

    # Build store details with channel-specific fields
    auv_field = "bm_auv_p50" if channel == "B&M" else "web_auv_p50"
    auv_p90_field = "bm_auv_p90" if channel == "B&M" else "web_auv_p90"
    store_details = [
        {
            "id": s["id"],
            "name": s.get("name"),
            "dma": s.get("dma"),
            "city": s.get("city"),
            "state": s.get("state"),
            "channel": channel,
            "auv_p50": s.get(auv_field),
            "auv_p90": s.get(auv_p90_field),
            "yoy_change": _get_store_yoy(s, channel),
            "is_outlet": s.get("is_outlet"),
            "is_new_store": s.get("is_new_store"),
            "bcg_category": s.get("bcg_category"),
            "merchandising_sf": s.get("merchandising_sf"),
            "store_design_sf": s.get("store_design_sf"),
        }
        for s in limited
    ]

    # Summary by DMA using channel-specific YoY
    dma_summary = defaultdict(lambda: {"count": 0, "yoy_values": []})
    for s in underperforming:
        dma = s.get("dma", "Unknown")
        dma_summary[dma]["count"] += 1
        yoy = _get_store_yoy(s, channel)
        if yoy is not None:
            dma_summary[dma]["yoy_values"].append(yoy)

    summary_by_dma = {
        dma: {
            "count": data["count"],
            "avg_yoy": round(sum(data["yoy_values"]) / len(data["yoy_values"]), 2) if data["yoy_values"] else None
        }
        for dma, data in dma_summary.items()
    }

    # Summary by store type using channel-specific YoY
    outlet_stores = [s for s in underperforming if s.get("is_outlet")]
    non_outlet_stores = [s for s in underperforming if not s.get("is_outlet")]

    outlet_yoys = [_get_store_yoy(s, channel) for s in outlet_stores if _get_store_yoy(s, channel) is not None]
    non_outlet_yoys = [_get_store_yoy(s, channel) for s in non_outlet_stores if _get_store_yoy(s, channel) is not None]

    summary_by_type = {
        "outlet": {
            "count": len(outlet_stores),
            "avg_yoy": round(sum(outlet_yoys) / len(outlet_yoys), 2) if outlet_yoys else None
        },
        "non_outlet": {
            "count": len(non_outlet_stores),
            "avg_yoy": round(sum(non_outlet_yoys) / len(non_outlet_yoys), 2) if non_outlet_yoys else None
        },
    }

    # Build filter description
    yoy_field_name = "bm_yoy_auv_change" if channel == "B&M" else "web_yoy_auv_change"
    filters_applied = [f"channel={channel}", f"{yoy_field_name} < {yoy_threshold}%"]
    if dma_filter:
        filters_applied.append(f"dma in {dma_filter}")
    if store_filter:
        filters_applied.append(f"store_id in {store_filter}")
    if not include_outlets:
        filters_applied.append("exclude outlets")
    if include_new_stores:
        filters_applied.append("include new stores (no YoY baseline)")

    return {
        "status": "success",
        "channel": channel,
        "total_underperforming": len(underperforming),
        "showing": len(limited),
        "filters_applied": filters_applied,
        "stores": store_details,
        "summary_by_dma": dict(summary_by_dma),
        "summary_by_type": summary_by_type,
        "hint": "Use get_store_diagnostic to analyze SHAP drivers for specific stores."
    }


# =============================================================================
# Tool 4: get_store_diagnostic
# =============================================================================

@tool
def get_store_diagnostic(
    store_ids: List[int],
    channel: str = "B&M",
    include_shap: bool = True,
    include_timeseries: bool = False,
    shap_top_n: int = 5,
) -> Dict[str, Any]:
    """
    Deep dive into specific stores with SHAP driver analysis.

    Provides comprehensive analysis including:
    - Store metadata and performance classification
    - SHAP feature analysis (what's driving the forecast up/down vs baseline)
    - Natural language interpretation of drivers

    Args:
        store_ids: List of store IDs to analyze (max 10)
        channel: Sales channel to analyze - "B&M" (default) or "WEB"
        include_shap: If True, include SHAP feature analysis (default True)
        include_timeseries: If True, include full weekly timeseries data
        shap_top_n: Number of top SHAP contributors to return (default 5)

    Returns:
        Dict with store diagnostics including:
        - stores: List of store analyses with overview and SHAP drivers
        - comparison: Summary if multiple stores requested

    Example:
        >>> get_store_diagnostic(store_ids=[2, 123], channel="B&M")
        {"status": "success", "stores": [...], "comparison": {...}}
    """
    if len(store_ids) > 10:
        return {
            "error": "Maximum 10 stores can be analyzed at once.",
            "hint": "Split your request into smaller batches."
        }

    if not store_ids:
        return {
            "error": "At least one store_id must be provided.",
            "hint": "Use get_underperforming_stores to find store IDs."
        }

    try:
        stores = _load_stores()
    except FileNotFoundError as e:
        return {
            "error": f"Data file not found: {e}",
            "hint": "Run regenerate_dashboard_data.py to create the data files."
        }

    # Build store lookup
    store_lookup = {s["id"]: s for s in stores}

    # Process each store
    store_results = []
    missing = []

    for store_id in store_ids:
        if store_id not in store_lookup:
            missing.append(store_id)
            continue

        store_data = store_lookup[store_id]

        # Classify performance using channel-specific YoY
        yoy = _get_store_yoy(store_data, channel)
        if store_data.get("is_new_store") or yoy is None:
            performance_status = "new"
        else:
            performance_status = _classify_yoy(yoy)

        # Get channel-specific AUV fields
        auv_field = "bm_auv_p50" if channel == "B&M" else "web_auv_p50"
        auv_p90_field = "bm_auv_p90" if channel == "B&M" else "web_auv_p90"

        store_result = {
            "id": store_id,
            "name": store_data.get("name"),
            "dma": store_data.get("dma"),
            "channel": channel,
            "overview": {
                "auv_p50": store_data.get(auv_field),
                "auv_p90": store_data.get(auv_p90_field),
                "yoy_change": yoy,
                "is_outlet": store_data.get("is_outlet"),
                "is_new_store": store_data.get("is_new_store"),
                "is_comp_store": store_data.get("is_comp_store"),
                "bcg_category": store_data.get("bcg_category"),
                "store_design_sf": store_data.get("store_design_sf"),
                "merchandising_sf": store_data.get("merchandising_sf"),
                "city": store_data.get("city"),
                "state": store_data.get("state"),
                "lat": store_data.get("lat"),
                "lng": store_data.get("lng"),
            },
            "performance_status": performance_status,
        }

        # Load timeseries for SHAP analysis
        ts_data = _load_store_timeseries(store_id)

        if include_shap and ts_data:
            timeseries = ts_data.get("timeseries", [])
            shap_analysis = _aggregate_shap_features(timeseries, top_n=shap_top_n)
            interpretation = _generate_shap_interpretation(shap_analysis)

            store_result["shap_analysis"] = {
                "top_positive_drivers": shap_analysis["top_positive_drivers"],
                "top_negative_drivers": shap_analysis["top_negative_drivers"],
                "interpretation": interpretation,
                "note": "SHAP impacts represent YoY change vs 2024 baseline. "
                        "Positive = contributes to higher forecast, Negative = drags forecast down."
            }

        if include_timeseries and ts_data:
            timeseries = ts_data.get("timeseries", [])
            store_result["timeseries"] = [
                {
                    "date": p.get("date"),
                    "pred_sales_p50": p.get("pred_sales_p50"),
                    "pred_sales_p90": p.get("pred_sales_p90"),
                    "pred_aov_p50": p.get("pred_aov_p50"),
                    "pred_traffic_p50": p.get("pred_traffic_p50"),
                }
                for p in timeseries
            ]

        store_results.append(store_result)

    result = {
        "status": "success",
        "channel": channel,
        "stores": store_results,
    }

    if missing:
        result["warning"] = f"Store IDs not found: {missing}"

    # Add comparison if multiple stores
    if len(store_results) > 1:
        yoy_values = [
            (s["id"], s["overview"].get("yoy_change"))
            for s in store_results
            if s["overview"].get("yoy_change") is not None
        ]

        if yoy_values:
            best = max(yoy_values, key=lambda x: x[1])
            worst = min(yoy_values, key=lambda x: x[1])

            result["comparison"] = {
                "best_yoy": {"id": best[0], "yoy": best[1]},
                "worst_yoy": {"id": worst[0], "yoy": worst[1]},
            }

    result["hint"] = (
        "SHAP drivers show what factors are contributing to forecast changes vs baseline. "
        "Use modify_business_lever to simulate changes to these drivers."
    )

    return result


# =============================================================================
# Tool 5: get_performance_ranking
# =============================================================================

@tool
def get_performance_ranking(
    channel: str = "B&M",
    level: str = "store",
    metric: str = "yoy_change",
    dma_filter: Optional[List[str]] = None,
    exclude_new: bool = False,
    exclude_outlets: bool = False,
    top_n: int = 10,
    bottom_n: int = 10,
) -> Dict[str, Any]:
    """
    Rank stores or DMAs by various metrics.

    Useful for identifying best practices (what's working) and areas needing attention.

    Args:
        channel: Sales channel to analyze - "B&M" (default) or "WEB"
        level: Ranking level - "store" or "dma" (default "store")
        metric: Metric to rank by - "yoy_change" or "auv_p50" (default "yoy_change")
        dma_filter: Filter to specific DMAs (store level only)
        exclude_new: If True, exclude entities without baseline YoY data
        exclude_outlets: If True, exclude outlet stores (store level only)
        top_n: Number of top performers to return (default 10)
        bottom_n: Number of bottom performers to return (default 10)

    Returns:
        Dict with performance rankings including:
        - top_performers: Best performing entities
        - bottom_performers: Worst performing entities
        - statistics: Summary stats (median, mean, std, min, max)

    Example:
        >>> get_performance_ranking(channel="B&M", level="dma", metric="yoy_change", top_n=5)
        {"status": "success", "top_performers": [...], "bottom_performers": [...]}
    """
    if level not in ("store", "dma"):
        return {
            "error": f"Invalid level: {level}",
            "hint": "Use 'store' or 'dma'"
        }

    if metric not in ("yoy_change", "auv_p50"):
        return {
            "error": f"Invalid metric: {metric}",
            "hint": "Use 'yoy_change' or 'auv_p50'"
        }

    try:
        if level == "store":
            data = _load_stores()
        else:
            data = _load_dma_summary()
    except FileNotFoundError as e:
        return {
            "error": f"Data file not found: {e}",
            "hint": "Run regenerate_dashboard_data.py to create the data files."
        }

    # Apply filters
    filtered = data

    if level == "store":
        if dma_filter:
            filtered = [s for s in filtered if s.get("dma") in dma_filter]
        if exclude_outlets:
            filtered = [s for s in filtered if not s.get("is_outlet")]

    # Determine the metric field name based on channel and level
    if level == "store":
        if metric == "yoy_change":
            metric_field = "bm_yoy_auv_change" if channel == "B&M" else "web_yoy_auv_change"
        else:  # auv_p50
            metric_field = "bm_auv_p50" if channel == "B&M" else "web_auv_p50"
    else:
        # DMA level
        if metric == "yoy_change":
            metric_field = "bm_yoy_auv_change_pct" if channel == "B&M" else "web_yoy_auv_change_pct"
        else:  # auv_p50
            metric_field = "bm_total_auv_p50" if channel == "B&M" else "web_total_auv_p50"

    # Filter by baseline availability
    if exclude_new:
        filtered = [e for e in filtered if e.get(metric_field) is not None]

    # Separate into entities with and without metric data
    with_data = [e for e in filtered if e.get(metric_field) is not None]
    without_data = [e for e in filtered if e.get(metric_field) is None]

    # Sort by metric
    sorted_data = sorted(with_data, key=lambda x: x.get(metric_field, 0), reverse=True)

    # Get top and bottom performers
    top_performers = sorted_data[:top_n]
    bottom_performers = sorted_data[-bottom_n:] if len(sorted_data) >= bottom_n else sorted_data

    # Reverse bottom performers to show worst first
    bottom_performers = list(reversed(bottom_performers))

    # Build output based on level
    if level == "store":
        top_list = [
            {
                "rank": i + 1,
                "id": s["id"],
                "name": s.get("name"),
                "dma": s.get("dma"),
                metric: s.get(metric_field),
                "is_outlet": s.get("is_outlet"),
                "city": s.get("city"),
                "state": s.get("state"),
            }
            for i, s in enumerate(top_performers)
        ]
        bottom_list = [
            {
                "rank": len(sorted_data) - len(bottom_performers) + i + 1,
                "id": s["id"],
                "name": s.get("name"),
                "dma": s.get("dma"),
                metric: s.get(metric_field),
                "is_outlet": s.get("is_outlet"),
                "city": s.get("city"),
                "state": s.get("state"),
            }
            for i, s in enumerate(bottom_performers)
        ]
    else:
        top_list = [
            {
                "rank": i + 1,
                "dma": d["dma"],
                metric: d.get(metric_field),
                "store_count": d.get("store_count"),
            }
            for i, d in enumerate(top_performers)
        ]
        bottom_list = [
            {
                "rank": len(sorted_data) - len(bottom_performers) + i + 1,
                "dma": d["dma"],
                metric: d.get(metric_field),
                "store_count": d.get("store_count"),
            }
            for i, d in enumerate(bottom_performers)
        ]

    # Compute statistics
    metric_values = [e.get(metric_field) for e in with_data]
    statistics = _compute_statistics(metric_values)
    statistics["count_with_data"] = len(with_data)
    statistics["count_no_baseline"] = len(without_data)

    # Build filter description
    filters_applied = []
    if dma_filter:
        filters_applied.append(f"dma in {dma_filter}")
    if exclude_new:
        filters_applied.append("exclude entities without baseline")
    if exclude_outlets and level == "store":
        filters_applied.append("exclude outlets")

    return {
        "status": "success",
        "channel": channel,
        "ranking_config": {
            "level": level,
            "channel": channel,
            "metric": metric,
            "metric_field": metric_field,
            "filters": filters_applied if filters_applied else ["none"],
        },
        "top_performers": top_list,
        "bottom_performers": bottom_list,
        "statistics": statistics,
        "hint": "Top performers can reveal best practices. Use get_store_diagnostic to understand what drives their success."
    }
