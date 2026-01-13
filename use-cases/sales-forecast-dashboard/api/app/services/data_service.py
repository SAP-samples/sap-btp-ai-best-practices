"""Data service for loading and caching JSON data files."""

import json
from pathlib import Path
from functools import lru_cache
from typing import List, Optional

from ..models.forecast import (
    DMA, Store, StoreTimeseries, DMATimeseries, TimeseriesPoint,
    ShapFeature, Explanation, ExplanationDriver
)
from ..agent.hana_loader import (
    load_calendar,
    parse_fiscal_week,
    parse_fiscal_quarter,
    parse_fiscal_int,
)

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def _get_fiscal_calendar_lookup() -> dict:
    """Load fiscal calendar from HANA and create date-keyed lookup."""
    calendar_df = load_calendar()
    lookup = {}
    for _, row in calendar_df.iterrows():
        # Handle date format (could be datetime or string)
        date_val = row.get('date')
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y-%m-%d')
        else:
            date_str = str(date_val)[:10]

        lookup[date_str] = {
            'fiscal_year': parse_fiscal_int(row.get('fiscal_year')),
            'fiscal_quarter': parse_fiscal_quarter(row.get('fiscal_quarter')),
            'fiscal_month': parse_fiscal_int(row.get('fiscal_month')),
            'fiscal_week': parse_fiscal_week(row.get('fiscal_week'))
        }
    return lookup


def _get_fiscal_fields(date_str: str) -> dict:
    """Get fiscal fields for a date string."""
    lookup = _get_fiscal_calendar_lookup()
    return lookup.get(date_str, {
        'fiscal_year': None, 'fiscal_quarter': None,
        'fiscal_month': None, 'fiscal_week': None
    })


@lru_cache(maxsize=1)
def _load_dma_summary() -> List[dict]:
    """Load and cache DMA summary data."""
    with open(DATA_DIR / "dma_summary.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_stores() -> List[dict]:
    """Load and cache stores data."""
    with open(DATA_DIR / "stores.json") as f:
        return json.load(f)


def get_dma_data() -> List[DMA]:
    """Get all DMA data as models."""
    data = _load_dma_summary()
    return [DMA(**d) for d in data]


def get_dma_by_name(dma_name: str) -> Optional[DMA]:
    """Get a specific DMA by name."""
    data = _load_dma_summary()
    for d in data:
        if d.get("dma") == dma_name:
            return DMA(**d)
    return None


def get_stores_data(
    dma: Optional[str] = None,
    include_non_comp: bool = False
) -> List[Store]:
    """
    Get all stores, optionally filtered by DMA and comp store status.

    By default, only comp stores (is_comp_store=True) are returned.
    Use include_non_comp=True to include all stores.

    Note: The underlying stores.json may only contain comp stores if
    it was generated with comp_stores_only=True. In that case,
    include_non_comp=True has no effect.

    Args:
        dma: Filter by DMA name
        include_non_comp: If True, include non-comp stores (default: False)

    Returns:
        List of Store models
    """
    data = _load_stores()

    # Filter by DMA if specified
    if dma:
        data = [s for s in data if s.get("dma") == dma]

    # Filter to comp stores only unless include_non_comp=True
    if not include_non_comp:
        data = [s for s in data if s.get("is_comp_store", False)]

    return [Store(**s) for s in data]


def get_store_by_id(store_id: int) -> Optional[Store]:
    """Get a specific store by ID."""
    data = _load_stores()
    for s in data:
        if s.get("id") == store_id:
            return Store(**s)
    return None


def get_store_timeseries(store_id: int, channel: str = "B&M") -> Optional[StoreTimeseries]:
    """
    Get timeseries data for a specific store and channel.

    Args:
        store_id: The store ID
        channel: Channel name ("B&M" or "WEB"). Defaults to "B&M".

    Returns:
        StoreTimeseries model or None if not found
    """
    # Determine file suffix based on channel
    channel_suffix = "bm" if channel == "B&M" else "web"
    path = DATA_DIR / "timeseries" / f"store_{store_id}_{channel_suffix}.json"

    # Fallback to legacy filename without channel suffix for backward compatibility
    if not path.exists():
        legacy_path = DATA_DIR / "timeseries" / f"store_{store_id}.json"
        if legacy_path.exists():
            path = legacy_path
        else:
            return None

    with open(path) as f:
        data = json.load(f)

    timeseries = []
    for point in data.get("timeseries", []):
        # Parse SHAP features
        shap_features = None
        if point.get("shap_features"):
            shap_features = [ShapFeature(**sf) for sf in point["shap_features"]]

        # Parse explanation if present
        explanation = None
        if point.get("explanation"):
            exp_data = point["explanation"]
            drivers = [ExplanationDriver(**d) for d in exp_data.get("drivers", [])]
            offsets = [ExplanationDriver(**o) for o in exp_data.get("offsets", [])]
            explanation = Explanation(
                summary=exp_data.get("summary", ""),
                drivers=drivers,
                offsets=offsets,
                other_factors_impact=exp_data.get("other_factors_impact"),
                is_estimated=exp_data.get("is_estimated", True),
                no_baseline=exp_data.get("no_baseline", False),
                generated_by=exp_data.get("generated_by", "static")
            )

        # Get fiscal fields for this date
        fiscal = _get_fiscal_fields(point["date"])

        timeseries.append(TimeseriesPoint(
            date=point["date"],
            fiscal_year=fiscal.get('fiscal_year'),
            fiscal_quarter=fiscal.get('fiscal_quarter'),
            fiscal_month=fiscal.get('fiscal_month'),
            fiscal_week=fiscal.get('fiscal_week'),
            pred_sales_p50=point.get("pred_sales_p50"),
            pred_sales_p90=point.get("pred_sales_p90"),
            baseline_sales_p50=point.get("baseline_sales_p50"),
            yoy_change_pct=point.get("yoy_change_pct"),
            pred_aov_mean=point.get("pred_aov_mean"),
            pred_aov_p50=point.get("pred_aov_p50"),
            pred_aov_p90=point.get("pred_aov_p90"),
            pred_traffic_p10=point.get("pred_traffic_p10"),
            pred_traffic_p50=point.get("pred_traffic_p50"),
            pred_traffic_p90=point.get("pred_traffic_p90"),
            shap_features=shap_features,
            explanation=explanation
        ))

    # Get channel from file data if present, otherwise use parameter
    file_channel = data.get("channel", channel)

    return StoreTimeseries(store_id=store_id, channel=file_channel, timeseries=timeseries)


def get_dma_timeseries(dma_name: str) -> Optional[DMATimeseries]:
    """Get aggregated timeseries data for a DMA."""
    safe_name = dma_name.replace("/", "_").replace(" ", "_")
    path = DATA_DIR / "timeseries" / f"dma_{safe_name}.json"

    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    timeseries = []
    for point in data.get("timeseries", []):
        # Get fiscal fields for this date
        fiscal = _get_fiscal_fields(point["date"])

        timeseries.append(TimeseriesPoint(
            date=point["date"],
            fiscal_year=fiscal.get('fiscal_year'),
            fiscal_quarter=fiscal.get('fiscal_quarter'),
            fiscal_month=fiscal.get('fiscal_month'),
            fiscal_week=fiscal.get('fiscal_week'),
            pred_sales_p50=point["pred_sales_p50"],
            pred_sales_p90=point["pred_sales_p90"],
            pred_aov_p50=point.get("pred_aov_p50"),
            pred_aov_p90=point.get("pred_aov_p90"),
            shap_features=None
        ))

    return DMATimeseries(dma=dma_name, timeseries=timeseries)
