"""
Context and setup tools for the forecasting agent.

These tools handle session initialization and store discovery:
- lookup_store_metadata: Find stores by criteria
- initialize_forecast_simulation: Set origin date and create baseline scenario
- get_session_state: Return current session state

Per Agent_plan.md Section 3.1: Context & Setup Tools
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.state import ScenarioData
from app.agent.hana_loader import load_model_b_filtered, load_store_master


def _get_fiscal_context(origin_date: str) -> Dict[str, Any]:
    """Get fiscal context for origin date."""
    try:
        from app.services.data_service import _get_fiscal_fields
        return _get_fiscal_fields(origin_date)
    except Exception:
        return {}


def _check_store_staleness(
    store_ids: List[int],
    as_of_date: str,
    staleness_threshold_weeks: int = 13,
) -> Dict[int, Dict[str, Any]]:
    """
    Check staleness of stores based on model_b.csv data recency.

    Args:
        store_ids: List of profit_center_nbr to check
        as_of_date: Reference date (YYYY-MM-DD format)
        staleness_threshold_weeks: Weeks without data to flag as inactive (default 13)

    Returns:
        Dict mapping store_id to activity status:
        - last_data_date: str or None
        - weeks_stale: float or None
        - is_active: bool
        - status: str description
    """
    session = get_session()

    # Load data from HANA with server-side filtering for better performance
    try:
        historical_df = load_model_b_filtered(profit_center_nbrs=store_ids)
        # Select only needed columns (columns are already lowercase from HANA loader)
        historical_df = historical_df[["profit_center_nbr", "origin_week_date"]].copy()
        historical_df["origin_week_date"] = pd.to_datetime(historical_df["origin_week_date"])
    except Exception as e:
        return {sid: {
            "last_data_date": None,
            "weeks_stale": None,
            "is_active": None,
            "status": f"Cannot determine - HANA query failed: {e}"
        } for sid in store_ids}
    as_of_dt = pd.to_datetime(as_of_date)

    results = {}
    for store_id in store_ids:
        store_data = historical_df[historical_df["profit_center_nbr"] == store_id]

        if store_data.empty:
            results[store_id] = {
                "last_data_date": None,
                "weeks_stale": None,
                "is_active": False,
                "status": "No historical data found - store may not exist"
            }
        else:
            last_date = store_data["origin_week_date"].max()
            weeks_stale = (as_of_dt - last_date).days / 7
            is_active = weeks_stale <= staleness_threshold_weeks

            # Build status message
            if is_active:
                if weeks_stale <= 0:
                    status_msg = f"Active (data available through {last_date.strftime('%Y-%m-%d')})"
                else:
                    status_msg = f"Active (last data {int(weeks_stale)} weeks before {as_of_date})"
            else:
                status_msg = f"Possibly closed (last data {last_date.strftime('%Y-%m-%d')}, {int(weeks_stale)} weeks ago)"

            results[store_id] = {
                "last_data_date": last_date.strftime("%Y-%m-%d"),
                "weeks_stale": round(weeks_stale, 1),
                "is_active": is_active,
                "status": status_msg
            }

    return results


@tool
def lookup_store_metadata(
    store_ids: Optional[List[int]] = None,
    dma: Optional[str] = None,
    is_outlet: Optional[bool] = None,
    min_weeks_open: Optional[int] = None,
    max_weeks_open: Optional[int] = None,
    check_active: bool = False,
    as_of_date: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Find stores based on criteria to populate store filters.

    Use this tool to discover stores in the system before initializing a simulation.
    Results can be used with initialize_forecast_simulation to scope the analysis.

    Args:
        store_ids: Filter by specific profit_center_nbr values. Use this to look up
                   metadata for specific stores (e.g., [160] to find store 160's DMA).
        dma: Filter by DMA name (e.g., "CHICAGO", "NEW YORK"). Case-insensitive partial match.
        is_outlet: Filter by outlet status. True = outlets only, False = non-outlets only.
        min_weeks_open: Minimum weeks since store opened (exclude newer stores).
        max_weeks_open: Maximum weeks since store opened (find newer stores).
        check_active: If True, check data recency in model_b.csv to determine if stores
                      are still active. Stores with no data for >13 weeks are flagged as
                      potentially closed. Default False.
        as_of_date: Reference date for activity check (YYYY-MM-DD). Only used when
                    check_active=True. Defaults to today's date.
        limit: Maximum number of stores to return (default 50).

    Returns:
        Dictionary containing:
        - stores: List of matching store records with profit_center_nbr, dma, lat/lon, etc.
        - total_count: Total number of matching stores
        - filter_applied: Summary of filters applied
        - available_dmas: List of unique DMAs in results (for further filtering)
        - activity_status: (only if check_active=True) Dict mapping store_id to activity info
          including last_data_date, weeks_stale, is_active, and status message
        - as_of_date: (only if check_active=True) The reference date used for activity check

    Example:
        >>> lookup_store_metadata(store_ids=[160])
        {"stores": [{"profit_center_nbr": 160, "dma": "CHICAGO", ...}], ...}

        >>> lookup_store_metadata(dma="chicago", is_outlet=False)
        {"stores": [...], "total_count": 15, ...}

        >>> lookup_store_metadata(store_ids=[158, 234], check_active=True, as_of_date="2025-06-06")
        {"stores": [...], "activity_status": {158: {"is_active": False, ...}, 234: {...}}}
    """
    session = get_session()

    # Load store master
    store_master = session.get_store_master()
    df = store_master.copy()

    # Compute weeks_since_open from date_opened if not present
    if "weeks_since_open" not in df.columns and "date_opened" in df.columns:
        today = pd.Timestamp.now()
        df["date_opened"] = pd.to_datetime(df["date_opened"], errors="coerce")
        df["weeks_since_open"] = ((today - df["date_opened"]).dt.days / 7).fillna(0).astype(int)

    # Track applied filters
    filters_applied = []

    # Filter by store_ids if specified (allows looking up specific stores)
    if store_ids:
        df = df[df["profit_center_nbr"].isin(store_ids)]
        filters_applied.append(f"store_ids in {store_ids}")

    # Filter by DMA (market_city column from store master is the DMA-like field)
    if dma:
        dma_lower = dma.lower()
        df = df[df["market_city"].str.lower().str.contains(dma_lower, na=False)]
        filters_applied.append(f"dma contains '{dma}'")

    # Filter by outlet status (is_outlet is boolean in store master)
    if is_outlet is not None:
        if "is_outlet" in df.columns:
            df = df[df["is_outlet"] == is_outlet]
            filters_applied.append(f"is_outlet = {is_outlet}")

    # Filter by weeks_since_open
    if min_weeks_open is not None and "weeks_since_open" in df.columns:
        df = df[df["weeks_since_open"] >= min_weeks_open]
        filters_applied.append(f"weeks_since_open >= {min_weeks_open}")

    if max_weeks_open is not None and "weeks_since_open" in df.columns:
        df = df[df["weeks_since_open"] <= max_weeks_open]
        filters_applied.append(f"weeks_since_open <= {max_weeks_open}")

    total_count = len(df)

    # Limit results
    df = df.head(limit)

    # Select relevant columns (market_city is the DMA field in store master)
    columns_to_include = [
        "profit_center_nbr",
        "profit_center_name",
        "store_address",
        "market_city",
        "latitude",
        "longitude",
        "is_outlet",
        "merchandising_sf",
        "weeks_since_open",
        "date_opened",
    ]
    columns_available = [c for c in columns_to_include if c in df.columns]
    df_subset = df[columns_available].copy()

    # Rename market_city to dma for clearer output to agent
    if "market_city" in df_subset.columns:
        df_subset = df_subset.rename(columns={"market_city": "dma"})

    # Convert to records
    stores = df_subset.to_dict(orient="records")

    # Get unique DMAs for context (filter out NaN values to avoid sorting TypeError)
    if "market_city" in df.columns and not df.empty:
        available_dmas = [
            dma for dma in df["market_city"].unique().tolist()
            if pd.notna(dma) and isinstance(dma, str)
        ]
    else:
        available_dmas = []

    # Check store activity if requested
    activity_status = {}
    if check_active:
        # Get all store IDs from results (not just filtered store_ids param)
        result_store_ids = [s["profit_center_nbr"] for s in stores]
        if result_store_ids:
            ref_date = as_of_date or pd.Timestamp.now().strftime("%Y-%m-%d")
            activity_status = _check_store_staleness(result_store_ids, ref_date)

    # Build result dict
    result = {
        "stores": stores,
        "total_count": total_count,
        "returned_count": len(stores),
        "filters_applied": filters_applied if filters_applied else ["none"],
        "available_dmas": sorted(available_dmas),
        "hint": "Use store profit_center_nbr values with initialize_forecast_simulation to scope your analysis.",
    }

    if check_active:
        result["activity_status"] = activity_status
        result["as_of_date"] = as_of_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    return result


@tool
def initialize_forecast_simulation(
    origin_date: str,
    horizon_weeks: int = 13,
    channel: str = "B&M",
    store_ids: Optional[List[int]] = None,
    dmas: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Initialize a forecast simulation and create the baseline scenario.

    This is the first step in any what-if analysis. It sets up the simulation context
    (origin date, horizon, scope) and creates a "baseline" scenario representing
    "business as usual" forecasts.

    Per Agent_plan.md Section 3.1:
    - Backtesting Mode: If origin_date + horizon is in the past (data exists), loads ACTUAL
      lever values. Enables "What if we had done X instead?" counterfactual analysis.
    - Pure Forecasting Mode: If horizon is in the future, uses Seasonal Naive (lag-52)
      to populate "business as usual" values.

    Args:
        origin_date: The "today" of the simulation (t0) in YYYY-MM-DD format.
                     This is when the forecast is made.
        horizon_weeks: How far ahead to forecast (1-52 weeks). Default 13 (one quarter).
        channel: Channel to analyze: "B&M" (brick & mortar) or "WEB". Default "B&M".
        store_ids: Optional list of profit_center_nbr to include. If None, uses all stores.
        dmas: Optional list of DMAs to include (e.g., ["CHICAGO", "NEW YORK"]).

    Returns:
        Dictionary containing:
        - status: "initialized" on success
        - origin_date: The t0 date set
        - horizon_weeks: Number of weeks forecast
        - channel: Active channel
        - baseline_scenario: Summary of the baseline scenario created
        - store_count: Number of stores in scope
        - mode: "backtesting" or "forecasting" based on data availability

    Example:
        >>> initialize_forecast_simulation("2024-01-15", horizon_weeks=13, channel="B&M")
        {"status": "initialized", "origin_date": "2024-01-15", ...}
    """
    session = get_session()

    # Validate inputs
    if horizon_weeks < 1 or horizon_weeks > 52:
        return {"error": f"horizon_weeks must be 1-52, got {horizon_weeks}"}

    if channel not in ("B&M", "WEB"):
        return {"error": f"channel must be 'B&M' or 'WEB', got {channel}"}

    # Parse origin date
    try:
        origin_dt = pd.to_datetime(origin_date)
    except Exception as e:
        return {"error": f"Invalid origin_date format: {origin_date}. Use YYYY-MM-DD. Error: {e}"}

    # Update session state
    session.set_origin_date(origin_date)
    session.set_horizon_weeks(horizon_weeks)
    session.set_channel(channel)

    if store_ids:
        session.set_store_filter(store_ids)
    if dmas:
        session.set_dma_filter(dmas)

    # Determine mode: backtesting vs forecasting
    # Check if we have historical data for the requested horizon
    mode = "forecasting"  # Default to forecasting mode
    baseline_df = None
    stale_store_warnings = []  # Track potentially closed stores
    stale_store_ids = []  # Store IDs that are stale/closed
    valid_store_ids = []  # Store IDs with recent data
    actual_origin_date = None  # Track actual origin date used in backtesting mode

    # Resolve DMAs to store IDs for efficient server-side filtering
    # This prevents loading ALL stores when user only specifies DMAs
    effective_store_ids = store_ids
    if dmas and not store_ids:
        try:
            store_master_df = load_store_master()
            dma_stores = store_master_df[store_master_df["market_city"].isin(dmas)]
            effective_store_ids = dma_stores["profit_center_nbr"].tolist()
            if not effective_store_ids:
                return {"error": f"No stores found in DMAs: {dmas}"}
        except Exception as e:
            return {"error": f"Failed to resolve DMAs to store IDs: {e}"}

    # Load historical data from HANA to check coverage (server-side filtering for performance)
    try:
        historical_df = load_model_b_filtered(
            profit_center_nbrs=effective_store_ids,
            channel=channel,
            max_horizon=horizon_weeks,
        )
        historical_df["origin_week_date"] = pd.to_datetime(historical_df["origin_week_date"])
        historical_df["target_week_date"] = pd.to_datetime(historical_df["target_week_date"])

        # Filter by dmas if specified and we have store_ids (both filters active)
        # Skip if dmas was already resolved to effective_store_ids above
        if dmas and store_ids:
            historical_df = historical_df[historical_df["dma"].isin(dmas)]

        # Check for potentially closed stores (last data >13 weeks before origin)
        # Separate into valid vs stale stores
        if store_ids:
            for store_id in store_ids:
                store_data = historical_df[historical_df["profit_center_nbr"] == store_id]
                if store_data.empty:
                    stale_store_warnings.append(
                        f"Store {store_id}: No historical data found in dataset"
                    )
                    stale_store_ids.append(store_id)
                else:
                    last_date = store_data["origin_week_date"].max()
                    weeks_stale = (origin_dt - last_date).days / 7
                    if weeks_stale > 13:
                        stale_store_warnings.append(
                            f"Store {store_id}: Last data from {last_date.strftime('%Y-%m-%d')} "
                            f"({int(weeks_stale)} weeks ago) - store may be closed"
                        )
                        stale_store_ids.append(store_id)
                    else:
                        valid_store_ids.append(store_id)

            # If ALL requested stores are stale/closed, return an error
            if len(valid_store_ids) == 0:
                return {
                    "error": "All requested stores have stale or missing data for the specified date.",
                    "details": stale_store_warnings,
                    "hint": "The requested store(s) may be closed or not yet open. "
                    "Try a different store or an earlier origin_date within the store's operating period.",
                }

            # Filter historical data to only valid stores
            if stale_store_ids:
                historical_df = historical_df[
                    ~historical_df["profit_center_nbr"].isin(stale_store_ids)
                ]

        # Check if target dates fall within historical data
        target_end = origin_dt + pd.Timedelta(weeks=horizon_weeks)
        max_historical_date = historical_df["target_week_date"].max()

        if target_end <= max_historical_date:
            # Backtesting mode - we have actual data
            mode = "backtesting"

            # Find the closest prior origin_week_date (floor to week start)
            # This handles cases where data might not always be on Mondays
            unique_origins = historical_df["origin_week_date"].unique()

            # Filter to origins <= requested date, then take the max (most recent prior)
            prior_origins = [o for o in unique_origins if pd.Timestamp(o) <= origin_dt]

            if prior_origins:
                # Floor: use the most recent origin that is <= requested date
                closest_origin = max(prior_origins, key=lambda x: pd.Timestamp(x))
            else:
                # If no prior origins exist, use the earliest available
                closest_origin = min(unique_origins, key=lambda x: pd.Timestamp(x))

            actual_origin_date = pd.Timestamp(closest_origin).strftime("%Y-%m-%d")

            # Load actual data for the closest origin and requested horizons
            mask = (
                (historical_df["origin_week_date"] == closest_origin)
                & (historical_df["horizon"] >= 1)
                & (historical_df["horizon"] <= horizon_weeks)
            )
            baseline_df = historical_df[mask].copy()

        else:
            # Forecasting mode - generate baselines using generator
            # Use valid_store_ids to exclude stale/closed stores when store_ids was provided
            # Otherwise use DMA-resolved effective_store_ids from earlier
            mode = "forecasting"
            forecasting_store_ids = valid_store_ids if store_ids else effective_store_ids
            baseline_df = _generate_baseline_from_hana(
                origin_date, horizon_weeks, channel, forecasting_store_ids, dmas
            )

    except Exception as e:
        # Fall back to forecasting mode with error message
        return {
            "error": f"Failed to load historical data from HANA: {e}. "
            "Please ensure HANA connection is available."
        }

    if baseline_df is None or baseline_df.empty:
        return {
            "error": "Could not create baseline scenario. No data available for the specified parameters."
        }

    # Create baseline scenario with channel-specific name to allow both
    # B&M and WEB baselines to coexist in the same session
    baseline_name = f"baseline_{channel.lower().replace('&', '')}"  # "baseline_bm" or "baseline_web"
    baseline = ScenarioData(
        name=baseline_name,
        df=baseline_df,
        parent_scenario=None,
        created_at=datetime.now().isoformat(),
        modifications=[],
        channel=channel,  # Track which channel created this scenario
    )

    # Add to session
    session.add_scenario(baseline)
    session.set_active_scenario(baseline_name)

    # Get fiscal context for origin date
    fiscal_context = _get_fiscal_context(origin_date)

    response = {
        "status": "initialized",
        "origin_date": origin_date,
        "fiscal_year": fiscal_context.get('fiscal_year'),
        "fiscal_quarter": fiscal_context.get('fiscal_quarter'),
        "fiscal_week": fiscal_context.get('fiscal_week'),
        "horizon_weeks": horizon_weeks,
        "channel": channel,
        "mode": mode,
        "store_count": baseline_df["profit_center_nbr"].nunique() if "profit_center_nbr" in baseline_df.columns else 0,
        "store_filter": store_ids or "all",
        "dma_filter": dmas or "all",
        "baseline_scenario": {
            "name": baseline_name,
            "rows": len(baseline_df),
            "horizons": sorted(baseline_df["horizon"].unique().tolist()) if "horizon" in baseline_df.columns else [],
        },
        "next_step": "Use create_scenario to fork the baseline for what-if analysis, "
        "or modify_business_lever to change the baseline directly.",
    }

    # Add actual origin date used if different from requested (backtesting mode)
    if actual_origin_date and actual_origin_date != origin_date:
        response["actual_origin_date"] = actual_origin_date
        response["note"] = f"Using closest available origin date: {actual_origin_date}"

    # Add warnings for potentially closed stores
    if stale_store_warnings:
        response["warnings"] = stale_store_warnings

    return response


def _generate_baseline_from_hana(
    origin_date: str,
    horizon_weeks: int,
    channel: str,
    store_ids: Optional[List[int]],
    dmas: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """
    Generate baseline from HANA model_b data for forecasting mode.

    Uses the most recent available origin date from HANA data that has
    ALL requested horizons (to avoid data edge effects where recent dates
    may have incomplete horizon coverage).
    Server-side filtering is applied for channel, store_ids, and max_horizon.
    """
    try:
        # Use server-side filtering for better performance
        historical_df = load_model_b_filtered(
            profit_center_nbrs=store_ids,
            channel=channel,
            max_horizon=horizon_weeks,
        )
        historical_df["origin_week_date"] = pd.to_datetime(historical_df["origin_week_date"])

        # Filter by dmas if specified (still client-side as DMA filter is less common)
        if dmas:
            historical_df = historical_df[historical_df["dma"].isin(dmas)]

        if historical_df.empty:
            return None

        # Find the most recent origin date that has ALL requested horizons.
        # Due to data edge effects, the most recent origin dates may have
        # incomplete horizon coverage (e.g., only horizon 1-3 instead of 1-13).
        horizon_counts = historical_df.groupby("origin_week_date")["horizon"].nunique()
        full_coverage_origins = horizon_counts[horizon_counts >= horizon_weeks].index

        if len(full_coverage_origins) > 0:
            # Use the most recent origin with full horizon coverage
            best_origin = max(full_coverage_origins)
        else:
            # Fall back to most recent origin if none have full coverage
            best_origin = historical_df["origin_week_date"].max()

        # Filter to that origin and the requested horizons
        mask = (
            (historical_df["origin_week_date"] == best_origin)
            & (historical_df["horizon"] >= 1)
            & (historical_df["horizon"] <= horizon_weeks)
        )
        baseline_df = historical_df[mask].copy()

        return baseline_df if not baseline_df.empty else None

    except Exception as e:
        print(f"Warning: HANA baseline generation failed: {e}")
        return None


def _generate_baseline_from_generator(
    model_b_path: Path,
    origin_date: str,
    horizon_weeks: int,
    channel: str,
    store_ids: Optional[List[int]],
    dmas: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """
    Generate baseline using BaselineGenerator for forecasting mode.

    Uses seasonal naive approach (lag-52) for future horizons.
    Now passes store_ids and dmas to the generator for early filtering
    to avoid generating rows for all stores then filtering.
    """
    import warnings as warn_module

    try:
        from app.regressor.baseline_generator import BaselineGenerator, BaselineConfig

        config = BaselineConfig(
            model_b_path=model_b_path,
            horizons=range(1, horizon_weeks + 1),
            channels=[channel],
            origin_date=origin_date,
            store_ids=store_ids,  # Early filtering in generator
            dmas=dmas,            # Early filtering in generator
        )

        generator = BaselineGenerator(config)

        # Suppress expected warnings about optional CRM features
        with warn_module.catch_warnings():
            warn_module.filterwarnings(
                "ignore",
                message="Model B:.*expected features not found",
                category=UserWarning
            )
            model_a_df, model_b_df = generator.generate()

        return model_b_df

    except Exception as e:
        print(f"Warning: BaselineGenerator failed: {e}")
        return None


@tool
def get_session_state() -> Dict[str, Any]:
    """
    Get the current session state summary.

    Returns an overview of the current simulation context including:
    - Origin date and horizon
    - Active channel and filters
    - List of scenarios and their status
    - List of predictions available
    - Model loading status

    Use this to understand the current state before making decisions about
    which tools to use next.

    Returns:
        Dictionary containing:
        - origin_date: Current simulation t0
        - horizon_weeks: Forecast horizon
        - channel: Active channel
        - active_scenario: Name of currently active scenario
        - scenarios: List of scenario names with summaries
        - predictions: List of scenarios with cached predictions
        - models_loaded: Whether inference models are loaded
        - checkpoint_dir: Path to model checkpoints
        - store_metadata: List of metadata dicts for initialized stores (dma, lat/lon, is_outlet)

    Example:
        >>> get_session_state()
        {"origin_date": "2024-01-15", "horizon_weeks": 13, "active_scenario": "baseline_bm",
         "store_metadata": [{"profit_center_nbr": 160, "dma": "CHICAGO", ...}], ...}
    """
    session = get_session()
    summary = session.get_session_summary()

    # Add scenario details
    scenarios_detail = []
    for name, scenario in session.get_state().get("scenarios", {}).items():
        scenario_info = scenario.get_summary()
        scenario_info["has_predictions"] = session.has_prediction(name)
        scenarios_detail.append(scenario_info)

    summary["scenarios_detail"] = scenarios_detail

    # Add store metadata for initialized stores (so agent knows DMA, location, etc.)
    store_filter = session.get_store_filter()
    if store_filter:
        try:
            store_master = session.get_store_master()
            # Select columns that exist in store master
            metadata_cols = ["profit_center_nbr", "profit_center_name", "store_address", "market_city", "is_outlet", "latitude", "longitude"]
            available_cols = [c for c in metadata_cols if c in store_master.columns]
            store_metadata_df = store_master[
                store_master["profit_center_nbr"].isin(store_filter)
            ][available_cols].copy()
            store_metadata_df = store_metadata_df.rename(columns={"market_city": "dma"})
            summary["store_metadata"] = store_metadata_df.to_dict(orient="records")
        except Exception:
            summary["store_metadata"] = []
    else:
        summary["store_metadata"] = []

    # Add hint about next steps
    if not summary.get("origin_date"):
        summary["hint"] = "Session not initialized. Use initialize_forecast_simulation first."
    elif summary.get("num_scenarios", 0) == 1:
        summary["hint"] = "Only baseline exists. Use create_scenario to fork a what-if scenario."
    elif summary.get("num_predictions", 0) == 0:
        summary["hint"] = "No predictions yet. Use run_forecast_model to generate predictions."
    else:
        summary["hint"] = "Ready for analysis. Use compare_scenarios or explain_forecast_change."

    return summary


# Export all tools
__all__ = [
    "lookup_store_metadata",
    "initialize_forecast_simulation",
    "get_session_state",
]
