"""
Budget-to-Awareness conversion tools for the forecasting agent.

Provides estimation of:
- Marketing budget needed to achieve awareness targets
- Awareness changes expected from budget investments
- Model quality metrics and available markets

Data Sources (from data/ directory):
- budget_marketing.xlsx: Monthly budget by DMA (2024, 2025 sheets)
- Awareness_Consideration_2022-2025.xlsx: Weekly awareness by Market
- BDF Data Model Master Tables.xlsx (YOUGOV_DMA_MAP sheet): Market mappings

Model: LogLinearControl
    awareness_change = beta_0 + beta_1 * log(1 + budget) + beta_2 * awareness_current
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.paths import AGENT_INPUT_DIR
from app.agent.hana_loader import (
    load_yougov_dma_map,
    load_awareness_consideration,
    load_budget_data,
)

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketCoefficients:
    """Fitted coefficients for a market's LogLinearControl model."""
    market: str
    intercept: float
    beta_log_budget: float
    beta_awareness_lag: float
    r_squared: float
    adj_r_squared: float
    rmse: float
    n_observations: int
    p_value_intercept: float
    p_value_log_budget: float
    p_value_awareness_lag: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketCoefficients":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# Path and Cache Configuration
# =============================================================================

def _get_data_dir() -> Path:
    """Get path to agent input data directory."""
    return AGENT_INPUT_DIR


def _get_cache_dir() -> Path:
    """Get path to cache directory for coefficients."""
    module_path = Path(__file__).parent.parent.parent.parent
    cache_dir = module_path / ".cache" / "budget_awareness"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_path() -> Path:
    """Get path to coefficients cache file."""
    return _get_cache_dir() / "coefficients.json"


# =============================================================================
# Data Preparation and Panel Creation
# =============================================================================

def _prepare_monthly_panel() -> pd.DataFrame:
    """
    Prepare monthly panel data merging budget and awareness.

    Returns DataFrame with columns:
    - market: Market name
    - year, month: Time period
    - budget: Marketing budget (dollars)
    - awareness_roll: Rolling awareness (0-100)
    - awareness_roll_lag_1: Lagged awareness
    - awareness_change: Target variable
    - log_budget: Log-transformed budget
    """
    # Load data sources from HANA
    yougov_map = load_yougov_dma_map()
    budget_df = load_budget_data()
    awareness_df = load_awareness_consideration()

    # Log and validate data loaded successfully
    logger.info(f"Loaded yougov_dma_map: {len(yougov_map) if yougov_map is not None else 0} rows")
    logger.info(f"Loaded budget_data: {len(budget_df) if budget_df is not None else 0} rows")
    logger.info(f"Loaded awareness_consideration: {len(awareness_df) if awareness_df is not None else 0} rows")

    if yougov_map is None or yougov_map.empty:
        raise ValueError("Failed to load YOUGOV_DMA_MAP from HANA - no data returned")
    if budget_df is None or budget_df.empty:
        raise ValueError("Failed to load BUDGET_MARKETING from HANA - no data returned")
    if awareness_df is None or awareness_df.empty:
        raise ValueError("Failed to load AWARENESS_CONSIDERATION from HANA - no data returned")

    # Find company column (handle case variations from HANA)
    company_col = None
    for col in awareness_df.columns:
        if col.lower() == 'company':
            company_col = col
            break

    # Filter target company if Company column exists
    if company_col:
        awareness_df = awareness_df[awareness_df[company_col] == "Company X"].copy()

    logger.info(f"After company filter: {len(awareness_df)} rows")

    if awareness_df.empty:
        raise ValueError(
            "No awareness data found for Company X. "
            "Check AWARENESS_CONSIDERATION table has matching Company records."
        )

    # Map budget to Market via market_city
    logger.info(f"budget_df columns: {list(budget_df.columns)}")
    # Clean market_city: strip trailing commas and normalize to uppercase
    budget_df['market_city_upper'] = budget_df['market_city'].str.rstrip(',').str.upper()

    # Manual mapping for budget market_city names that don't match YouGov naming conventions
    budget_to_yougov_mapping = {
        'BOSTON': 'BOSTON/NH',
        'DAVENPORT': 'DAVENPORT - ROCK ISLAND - MOLINE',
        'DES-MOINES-AMES': 'DES MOINES',
        'FRESNO-VISALIA': 'FRESNO',
        'GREEN BAY-APPLETON': 'GREEN BAY',
        'GREENSBORO': 'GREENSBORO-HIGH POINT-WINSTON SALEM',
        'HARTFORD & NEW HAVEN': 'HARTFORD/NEW HAVEN',
        'NORFOLK': 'NORFOLK / NEWPORT NEWS',
        'PORTLAND - AUBURN': 'PORTLAND',
        'SPRINGFIELD-HOLYOKE': 'SPRINGFIELD',
        'WASHINGTON': 'WASHINGTON DC',
    }
    budget_df['market_city_upper'] = budget_df['market_city_upper'].replace(budget_to_yougov_mapping)

    yougov_lookup = yougov_map[['market_city', 'market']].copy()
    yougov_lookup['market_city_upper'] = yougov_lookup['market_city'].str.upper()
    yougov_lookup = yougov_lookup[['market_city_upper', 'market']].drop_duplicates()

    logger.info(f"yougov_lookup has {len(yougov_lookup)} unique market_city mappings")
    logger.info(f"Sample budget market_cities: {budget_df['market_city_upper'].head(5).tolist()}")
    logger.info(f"Sample yougov market_cities: {yougov_lookup['market_city_upper'].head(5).tolist()}")

    budget_df = budget_df.merge(yougov_lookup, on='market_city_upper', how='left')

    matched_count = budget_df['market'].notna().sum()
    logger.info(f"After merge: {matched_count}/{len(budget_df)} rows matched to a market")

    # Filter to valid markets
    budget_valid = budget_df[budget_df['market'].notna()].copy()
    budget_valid = budget_valid[~budget_valid['market'].str.contains('Out of Footprint', na=False)]

    logger.info(f"After filtering valid markets: {len(budget_valid)} rows")

    # Aggregate budget by Market and year-month
    budget_monthly = budget_valid.groupby(['market', 'year', 'month']).agg({
        'budget': 'sum'
    }).reset_index()

    # Aggregate awareness to monthly (4-week rolling mean)
    awareness_df['year'] = awareness_df['week_start'].dt.year
    awareness_df['month'] = awareness_df['week_start'].dt.month

    # Use 4-week rolling mean within each market
    awareness_df = awareness_df.sort_values(['market', 'week_start'])
    awareness_df['awareness_roll'] = awareness_df.groupby('market')['awareness'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )

    # Monthly aggregation
    awareness_monthly = awareness_df.groupby(['market', 'year', 'month']).agg({
        'awareness_roll': 'mean'
    }).reset_index()

    # Merge budget with awareness
    panel = budget_monthly.merge(
        awareness_monthly,
        on=['market', 'year', 'month'],
        how='inner'
    )

    # Log and validate merge produced results
    logger.info(f"budget_monthly: {len(budget_monthly)} rows, awareness_monthly: {len(awareness_monthly)} rows")
    logger.info(f"After merge: {len(panel)} rows, {panel['market'].nunique() if not panel.empty else 0} unique markets")

    if panel.empty:
        raise ValueError(
            "No matching budget-awareness data after merging. "
            "This may indicate a market name mismatch between budget and awareness tables."
        )

    # Sort and add features
    panel = panel.sort_values(['market', 'year', 'month']).reset_index(drop=True)

    # Add lagged awareness for model
    panel['awareness_roll_lag_1'] = panel.groupby('market')['awareness_roll'].shift(1)

    # Calculate awareness change (target variable)
    panel['awareness_change'] = panel['awareness_roll'] - panel['awareness_roll_lag_1']

    # Add log-transformed budget
    panel['log_budget'] = np.log1p(panel['budget'])

    return panel


# =============================================================================
# Model Fitting Functions
# =============================================================================

def _fit_market_model(market_df: pd.DataFrame, market: str) -> Optional[MarketCoefficients]:
    """
    Fit LogLinearControl model to a single market's data.

    Model: awareness_change = beta_0 + beta_1 * log(1 + budget) + beta_2 * awareness_lag_1
    """
    # Prepare data
    data = market_df.dropna(subset=['log_budget', 'awareness_roll_lag_1', 'awareness_change'])

    if len(data) < 5:
        return None

    X = data[['log_budget', 'awareness_roll_lag_1']]
    X = sm.add_constant(X)
    y = data['awareness_change']

    # Fit OLS
    model = sm.OLS(y, X).fit()

    # Calculate RMSE
    residuals = model.resid
    rmse = np.sqrt(np.mean(residuals ** 2))

    return MarketCoefficients(
        market=market,
        intercept=float(model.params.iloc[0]),
        beta_log_budget=float(model.params.iloc[1]),
        beta_awareness_lag=float(model.params.iloc[2]),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        rmse=float(rmse),
        n_observations=len(data),
        p_value_intercept=float(model.pvalues.iloc[0]),
        p_value_log_budget=float(model.pvalues.iloc[1]),
        p_value_awareness_lag=float(model.pvalues.iloc[2]),
    )


def _fit_all_markets() -> Dict[str, MarketCoefficients]:
    """Fit models for all markets and return coefficients dict."""
    panel = _prepare_monthly_panel()

    coefficients = {}
    for market in panel['market'].unique():
        market_df = panel[panel['market'] == market].copy()
        coefs = _fit_market_model(market_df, market)
        if coefs is not None:
            coefficients[market] = coefs

    return coefficients


# =============================================================================
# Caching Functions
# =============================================================================

def _compute_data_hash() -> str:
    """Compute hash of source data files to detect changes."""
    data_dir = _get_data_dir()
    files = [
        data_dir / "budget_marketing.xlsx",
        data_dir / "Awareness_Consideration_2022-2025.xlsx",
        data_dir / "BDF Data Model Master Tables.xlsx"
    ]

    hasher = hashlib.md5()
    for f in files:
        if f.exists():
            hasher.update(f.stat().st_mtime_ns.to_bytes(8, 'little'))
            hasher.update(f.stat().st_size.to_bytes(8, 'little'))

    return hasher.hexdigest()


def _load_cached_coefficients() -> Optional[Dict[str, MarketCoefficients]]:
    """Load coefficients from cache if valid."""
    cache_path = _get_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Check hash matches
        current_hash = _compute_data_hash()
        if cache_data.get('data_hash') != current_hash:
            return None

        # Reconstruct coefficients
        coefficients = {}
        for market, coef_dict in cache_data.get('coefficients', {}).items():
            coefficients[market] = MarketCoefficients.from_dict(coef_dict)

        return coefficients

    except Exception:
        return None


def _save_coefficients_to_cache(coefficients: Dict[str, MarketCoefficients]) -> None:
    """Save coefficients to cache file."""
    cache_path = _get_cache_path()

    cache_data = {
        'data_hash': _compute_data_hash(),
        'generated_at': datetime.now().isoformat(),
        'coefficients': {
            market: coef.to_dict() for market, coef in coefficients.items()
        }
    }

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)


# Module-level cache
_coefficients_cache: Optional[Dict[str, MarketCoefficients]] = None


def _get_coefficients(force_refit: bool = False) -> Dict[str, MarketCoefficients]:
    """Get coefficients, using cache if available."""
    global _coefficients_cache

    if not force_refit and _coefficients_cache is not None:
        return _coefficients_cache

    if not force_refit:
        cached = _load_cached_coefficients()
        if cached is not None:
            _coefficients_cache = cached
            return cached

    # Fit models with error handling for data loading issues
    try:
        coefficients = _fit_all_markets()
    except ValueError as e:
        # Log error and return empty dict (tools will handle gracefully)
        logger.error(f"Failed to fit budget-awareness models: {e}")
        return {}

    # Save to cache
    _save_coefficients_to_cache(coefficients)
    _coefficients_cache = coefficients

    return coefficients


# =============================================================================
# Prediction Helper Functions
# =============================================================================

def _get_market_for_dma(dma: str) -> Optional[str]:
    """
    Map a DMA (market_city) to its aggregate Market name.

    The forecasting agent uses DMA values like "CHICAGO", "BALTIMORE".
    We need to map these to aggregate Market names used in awareness data.
    """
    coefficients = _get_coefficients()

    # Direct match on market name
    dma_upper = dma.upper()
    for market in coefficients.keys():
        if market.upper() == dma_upper:
            return market

    # Try loading YOUGOV map for lookup from HANA
    try:
        yougov_map = load_yougov_dma_map()
        match = yougov_map[yougov_map['market_city'].str.upper() == dma_upper]
        if not match.empty:
            market = match['market'].iloc[0]
            if market in coefficients:
                return market
    except Exception:
        pass

    # Partial match
    for market in coefficients.keys():
        if dma_upper in market.upper() or market.upper() in dma_upper:
            return market

    return None


def _classify_confidence(coefs: MarketCoefficients) -> str:
    """Classify confidence level based on model quality."""
    if coefs.r_squared > 0.15 and coefs.n_observations >= 18:
        return "high"
    elif coefs.r_squared > 0.05 and coefs.n_observations >= 12:
        return "medium"
    else:
        return "low"


def _predict_awareness_change(
    budget: float,
    current_awareness: float,
    coefs: MarketCoefficients,
) -> float:
    """
    Predict awareness change using LogLinearControl model.

    awareness_change = intercept + beta_log_budget * log(1 + budget) + beta_awareness_lag * current_awareness
    """
    log_budget = np.log1p(budget)
    change = (
        coefs.intercept
        + coefs.beta_log_budget * log_budget
        + coefs.beta_awareness_lag * current_awareness
    )
    # Floor at zero (spending cannot decrease awareness) and cap at headroom
    headroom = 100 - current_awareness
    return max(0, min(change, headroom))


# =============================================================================
# Tool Functions
# =============================================================================

@tool
def estimate_budget_for_awareness(
    target_awareness: float,
    current_awareness: float,
    dma: Optional[str] = None,
    market: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate the marketing budget needed to achieve a target awareness level.

    Uses the LogLinearControl model fitted on historical budget-awareness data:
        awareness_change = beta_0 + beta_1 * log(1 + budget) + beta_2 * awareness_current

    The model inverts this to solve for budget given a target awareness change.

    Args:
        target_awareness: Desired awareness level (0-100 scale).
        current_awareness: Current awareness level (0-100 scale).
        dma: DMA name (e.g., "BALTIMORE", "CHICAGO"). Uses active session's DMA if not provided.
        market: Aggregate market name (e.g., "Baltimore"). Alternative to dma parameter.
                Takes precedence over dma if both provided.

    Returns:
        Dictionary containing:
        - status: "estimated" on success
        - current_awareness: Input current awareness
        - target_awareness: Input target awareness
        - awareness_change_needed: Delta required
        - estimated_budget: Estimated monthly budget in dollars
        - budget_range: (low, high) tuple representing uncertainty
        - confidence: "high", "medium", or "low" based on model fit
        - feasibility: "likely", "uncertain", "unlikely", or "impossible"
        - model_quality: R-squared and observation count
        - interpretation: Human-readable summary
        - hint: Next steps guidance

    Example:
        >>> estimate_budget_for_awareness(70.0, 60.0, dma="BALTIMORE")
        {"status": "estimated", "estimated_budget": 185000, ...}
    """
    session = get_session()

    # Resolve market
    if market:
        resolved_market = market
    elif dma:
        resolved_market = _get_market_for_dma(dma)
    else:
        # Try to get DMA from session
        dma_filter = session.get_dma_filter()
        if dma_filter:
            resolved_market = _get_market_for_dma(dma_filter[0])
        else:
            return {
                "error": "No DMA or market specified. Provide dma or market parameter, "
                "or initialize session with a DMA filter."
            }

    if resolved_market is None:
        available = list(_get_coefficients().keys())
        return {
            "error": f"Could not map DMA '{dma or market}' to a market with model data. "
            f"Available markets: {available[:10]}... (use get_budget_awareness_info for full list)"
        }

    coefficients = _get_coefficients()
    if resolved_market not in coefficients:
        return {
            "error": f"No model available for market '{resolved_market}'. "
            f"Use get_budget_awareness_info to see available markets."
        }

    coefs = coefficients[resolved_market]

    # Validate inputs
    if not 0 <= current_awareness <= 100:
        return {"error": f"current_awareness must be 0-100, got {current_awareness}"}
    if not 0 <= target_awareness <= 100:
        return {"error": f"target_awareness must be 0-100, got {target_awareness}"}

    target_change = target_awareness - current_awareness
    headroom_factor = (100 - current_awareness) / 100

    # Invert the model to estimate budget
    # log(1 + budget) = (change - intercept - beta_aware * awareness) / beta_log
    if coefs.beta_log_budget > 0:
        adjusted_target = target_change - coefs.intercept - coefs.beta_awareness_lag * current_awareness
        log_budget = adjusted_target / coefs.beta_log_budget
        estimated_budget = max(0, np.expm1(log_budget))  # exp(x) - 1
    else:
        estimated_budget = float('inf')

    # Calculate budget range using uncertainty
    confidence_level = _classify_confidence(coefs)
    if np.isfinite(estimated_budget) and estimated_budget > 0:
        uncertainty_factor = 0.5 if confidence_level == "low" else 0.3
        budget_low = estimated_budget * (1 - uncertainty_factor)
        budget_high = estimated_budget * (1 + uncertainty_factor)
    else:
        budget_low = budget_high = float('inf')

    # Assess feasibility
    if target_awareness > 100:
        feasibility = "impossible"
    elif headroom_factor < 0.10 and target_change > 5:
        feasibility = "unlikely"
    elif np.isinf(estimated_budget) or estimated_budget > 10_000_000:
        feasibility = "uncertain"
    elif confidence_level == "low":
        feasibility = "uncertain"
    else:
        feasibility = "likely"

    # Add warning for unrealistically high estimates
    warnings = []
    if estimated_budget > 1_000_000 and abs(target_change) < 10:
        warnings.append(
            f"Budget estimate (${estimated_budget/1000:.0f}K) seems high for a "
            f"{abs(target_change):.0f}-point change. Model coefficients for this market "
            "may not be reliable for planning purposes."
        )
    if coefs.p_value_log_budget > 0.1:
        warnings.append(
            f"Budget coefficient not statistically significant (p={coefs.p_value_log_budget:.2f})."
        )

    # Generate interpretation
    if np.isfinite(estimated_budget) and estimated_budget > 0:
        if abs(target_change) < 0.5:
            # Maintenance scenario (target ~= current)
            interpretation = (
                f"To maintain awareness at {current_awareness:.0f}% in {resolved_market}, "
                f"invest approximately ${estimated_budget/1000:.0f}K monthly "
                f"(range: ${budget_low/1000:.0f}K-${budget_high/1000:.0f}K). "
                f"This offsets natural decay at this awareness level."
            )
        else:
            interpretation = (
                f"To {'increase' if target_change > 0 else 'decrease'} awareness "
                f"from {current_awareness:.0f}% to {target_awareness:.0f}% "
                f"in {resolved_market} ({target_change:+.1f} points), invest approximately "
                f"${estimated_budget/1000:.0f}K monthly (range: ${budget_low/1000:.0f}K-${budget_high/1000:.0f}K). "
                f"Headroom: {headroom_factor:.0%}."
            )
    else:
        interpretation = (
            f"Cannot reliably estimate budget for {resolved_market}. "
            f"The model does not show a positive relationship between budget and awareness."
        )

    # Classify scenario type
    scenario_type = "maintenance" if abs(target_change) < 0.5 else ("increase" if target_change > 0 else "decrease")

    return {
        "status": "estimated",
        "scenario_type": scenario_type,
        "market": resolved_market,
        "dma_input": dma,
        "current_awareness": current_awareness,
        "target_awareness": target_awareness,
        "awareness_change_needed": round(target_change, 2),
        "estimated_budget": round(estimated_budget, 0) if np.isfinite(estimated_budget) else None,
        "budget_range": (round(budget_low, 0), round(budget_high, 0)) if np.isfinite(budget_low) else (None, None),
        "confidence": confidence_level,
        "feasibility": feasibility,
        "headroom_factor": round(headroom_factor, 2),
        "model_quality": {
            "r_squared": round(coefs.r_squared, 3),
            "n_observations": coefs.n_observations,
            "p_value_budget": round(coefs.p_value_log_budget, 3),
        },
        "warnings": warnings if warnings else None,
        "interpretation": interpretation,
        "hint": "Use modify_business_lever to apply the awareness change to your scenario.",
    }


@tool
def estimate_awareness_from_budget(
    budget_amount: float,
    current_awareness: float,
    dma: Optional[str] = None,
    market: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate the awareness change from a marketing budget investment.

    Uses the LogLinearControl model:
        awareness_change = beta_0 + beta_1 * log(1 + budget) + beta_2 * awareness_current

    This captures:
    - Diminishing returns on budget (via log transformation)
    - Saturation effects from current awareness level

    Args:
        budget_amount: Marketing budget in dollars (monthly).
        current_awareness: Current awareness level (0-100 scale).
        dma: DMA name (e.g., "BALTIMORE", "CHICAGO"). Uses active session's DMA if not provided.
        market: Aggregate market name (e.g., "Baltimore"). Alternative to dma parameter.

    Returns:
        Dictionary containing:
        - status: "estimated" on success
        - budget_input: Input budget amount
        - current_awareness: Input current awareness
        - estimated_awareness_change: Predicted change in awareness points
        - estimated_new_awareness: Predicted new awareness level
        - confidence_interval: (low, high) tuple for awareness change
        - confidence: "high", "medium", or "low"
        - headroom_factor: Room for improvement (1.0 = starting from 0)
        - model_quality: R-squared and observation count
        - interpretation: Human-readable summary

    Example:
        >>> estimate_awareness_from_budget(200000, 60.0, dma="BALTIMORE")
        {"status": "estimated", "estimated_awareness_change": 3.5, ...}
    """
    session = get_session()

    # Resolve market (same logic as estimate_budget_for_awareness)
    if market:
        resolved_market = market
    elif dma:
        resolved_market = _get_market_for_dma(dma)
    else:
        dma_filter = session.get_dma_filter()
        if dma_filter:
            resolved_market = _get_market_for_dma(dma_filter[0])
        else:
            return {
                "error": "No DMA or market specified. Provide dma or market parameter."
            }

    if resolved_market is None:
        return {
            "error": f"Could not map DMA '{dma or market}' to a market with model data."
        }

    coefficients = _get_coefficients()
    if resolved_market not in coefficients:
        return {
            "error": f"No model available for market '{resolved_market}'."
        }

    coefs = coefficients[resolved_market]

    # Validate inputs
    if budget_amount < 0:
        return {"error": f"budget_amount must be non-negative, got {budget_amount}"}
    if not 0 <= current_awareness <= 100:
        return {"error": f"current_awareness must be 0-100, got {current_awareness}"}

    # Calculate headroom
    headroom_factor = (100 - current_awareness) / 100

    # Predict awareness change
    estimated_change = _predict_awareness_change(budget_amount, current_awareness, coefs)
    estimated_new = current_awareness + estimated_change

    # Calculate confidence interval
    log_budget = np.log1p(budget_amount)
    se = abs(coefs.p_value_log_budget) * coefs.beta_log_budget * log_budget * 0.5  # Approximate SE
    ci_low = max(0, estimated_change - 1.645 * se)
    ci_high = min(100 - current_awareness, estimated_change + 1.645 * se)

    confidence_level = _classify_confidence(coefs)

    # Generate interpretation
    if estimated_change > 0:
        interpretation = (
            f"A ${budget_amount/1000:.0f}K monthly marketing investment in {resolved_market} "
            f"(current awareness: {current_awareness:.0f}%) is estimated to increase "
            f"brand awareness by ~{estimated_change:.1f} points to {estimated_new:.1f}% "
            f"(range: {current_awareness + ci_low:.1f}%-{current_awareness + ci_high:.1f}%). "
            f"Headroom: {headroom_factor:.0%}."
        )
    else:
        interpretation = (
            f"A ${budget_amount/1000:.0f}K investment in {resolved_market} "
            f"may not produce measurable awareness gains at awareness level {current_awareness:.0f}%. "
            f"Consider higher investment or different market."
        )

    return {
        "status": "estimated",
        "market": resolved_market,
        "dma_input": dma,
        "budget_input": budget_amount,
        "current_awareness": current_awareness,
        "estimated_awareness_change": round(estimated_change, 2),
        "estimated_new_awareness": round(estimated_new, 2),
        "confidence_interval": (round(ci_low, 2), round(ci_high, 2)),
        "confidence": confidence_level,
        "headroom_factor": round(headroom_factor, 2),
        "model_quality": {
            "r_squared": round(coefs.r_squared, 3),
            "n_observations": coefs.n_observations,
        },
        "interpretation": interpretation,
        "hint": "Use modify_business_lever to apply this awareness change to a scenario.",
    }


@tool
def refit_budget_awareness_model(
    force: bool = False,
) -> Dict[str, Any]:
    """
    Refit the budget-to-awareness model using latest data.

    Clears the cache and refits all market models. Use this if source data
    (budget_marketing.xlsx or Awareness_Consideration_2022-2025.xlsx) has been updated.

    Args:
        force: If True, always refit even if cache is valid. Default False.

    Returns:
        Dictionary containing:
        - status: "refitted" on success
        - markets_fitted: Number of markets with successful model fits
        - cache_updated: Whether cache was updated
        - summary: Model quality summary across markets

    Example:
        >>> refit_budget_awareness_model(force=True)
        {"status": "refitted", "markets_fitted": 35, ...}
    """
    global _coefficients_cache

    try:
        # Clear in-memory cache
        _coefficients_cache = None

        # Fit all markets
        coefficients = _fit_all_markets()

        # Save to cache
        _save_coefficients_to_cache(coefficients)
        _coefficients_cache = coefficients

        # Guard against empty results
        if not coefficients:
            return {
                "error": "No markets could be fitted. Check that budget and awareness data exists in HANA.",
                "markets_fitted": 0,
                "cache_updated": True,
                "hint": "Verify HANA connection and that BUDGET_MARKETING and AWARENESS_CONSIDERATION tables have data."
            }

        # Compute summary statistics
        r_squared_vals = [c.r_squared for c in coefficients.values()]
        n_obs_vals = [c.n_observations for c in coefficients.values()]

        return {
            "status": "refitted",
            "markets_fitted": len(coefficients),
            "cache_updated": True,
            "summary": {
                "mean_r_squared": round(np.mean(r_squared_vals), 3),
                "median_r_squared": round(np.median(r_squared_vals), 3),
                "markets_r2_above_0.1": sum(1 for r in r_squared_vals if r > 0.1),
                "mean_observations": round(np.mean(n_obs_vals), 1),
            },
            "hint": "Model refitted. Use estimate_budget_for_awareness or estimate_awareness_from_budget.",
        }

    except Exception as e:
        return {
            "error": f"Failed to refit model: {str(e)}",
            "hint": "Check that data files exist in data/ directory."
        }


@tool
def get_budget_awareness_info(
    market: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get information about available markets and model quality for budget-awareness conversion.

    Lists all markets with fitted models and their quality metrics. Use this to understand
    which markets have reliable budget-to-awareness relationships.

    Args:
        market: Optional specific market to get detailed info for.
                If not provided, returns summary of all markets.

    Returns:
        Dictionary containing:
        - status: "info" on success
        - If market specified:
            - market_info: Detailed coefficients and quality metrics
        - If no market:
            - available_markets: List of market names
            - market_count: Total count
            - quality_summary: Aggregate quality metrics
            - top_markets: Top 10 markets by R-squared

    Example:
        >>> get_budget_awareness_info()
        {"status": "info", "available_markets": [...], "market_count": 35, ...}

        >>> get_budget_awareness_info(market="Baltimore")
        {"status": "info", "market_info": {"r_squared": 0.25, ...}}
    """
    coefficients = _get_coefficients()

    # Guard against empty coefficients
    if not coefficients:
        return {
            "error": "No budget-awareness models available. Data may not have loaded from HANA.",
            "available_markets": [],
            "market_count": 0,
            "hint": "Use refit_budget_awareness_model() to diagnose data loading issues."
        }

    if market:
        # Try to resolve market name
        resolved = None
        market_upper = market.upper()

        for m in coefficients.keys():
            if m.upper() == market_upper:
                resolved = m
                break
            if market_upper in m.upper() or m.upper() in market_upper:
                resolved = m
                break

        if resolved is None:
            return {
                "error": f"Market '{market}' not found. Use get_budget_awareness_info() to see available markets."
            }

        coefs = coefficients[resolved]

        return {
            "status": "info",
            "market": resolved,
            "market_info": {
                "intercept": round(coefs.intercept, 4),
                "beta_log_budget": round(coefs.beta_log_budget, 4),
                "beta_awareness_lag": round(coefs.beta_awareness_lag, 4),
                "r_squared": round(coefs.r_squared, 3),
                "adj_r_squared": round(coefs.adj_r_squared, 3),
                "rmse": round(coefs.rmse, 2),
                "n_observations": coefs.n_observations,
                "p_value_log_budget": round(coefs.p_value_log_budget, 3),
                "p_value_awareness_lag": round(coefs.p_value_awareness_lag, 3),
                "confidence": _classify_confidence(coefs),
            },
            "interpretation": (
                f"For {resolved}: A 1% increase in log(budget) is associated with "
                f"a {coefs.beta_log_budget:.2f} point change in awareness. "
                f"Model explains {coefs.r_squared*100:.1f}% of variance."
            ),
        }

    # Summary for all markets
    r_squared_vals = [c.r_squared for c in coefficients.values()]
    n_obs_vals = [c.n_observations for c in coefficients.values()]

    # Top 10 by R-squared
    sorted_markets = sorted(coefficients.items(), key=lambda x: x[1].r_squared, reverse=True)
    top_10 = [
        {
            "market": m,
            "r_squared": round(c.r_squared, 3),
            "n_observations": c.n_observations,
            "confidence": _classify_confidence(c),
        }
        for m, c in sorted_markets[:10]
    ]

    return {
        "status": "info",
        "available_markets": sorted(coefficients.keys()),
        "market_count": len(coefficients),
        "quality_summary": {
            "mean_r_squared": round(np.mean(r_squared_vals), 3),
            "median_r_squared": round(np.median(r_squared_vals), 3),
            "min_r_squared": round(min(r_squared_vals), 3),
            "max_r_squared": round(max(r_squared_vals), 3),
            "markets_high_confidence": sum(1 for c in coefficients.values() if _classify_confidence(c) == "high"),
            "markets_medium_confidence": sum(1 for c in coefficients.values() if _classify_confidence(c) == "medium"),
            "markets_low_confidence": sum(1 for c in coefficients.values() if _classify_confidence(c) == "low"),
        },
        "top_markets": top_10,
        "hint": "Use estimate_budget_for_awareness or estimate_awareness_from_budget with these market names.",
    }


# Export all tools
__all__ = [
    "estimate_budget_for_awareness",
    "estimate_awareness_from_budget",
    "refit_budget_awareness_model",
    "get_budget_awareness_info",
]
