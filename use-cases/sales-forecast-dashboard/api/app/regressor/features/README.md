# Feature Engineering Module

**Location:** `api/app/regressor/features/`

**Purpose:** Feature engineering primitives for Company X sales forecasting system. This module implements a dual-architecture approach with Model A (actionable/explainability) and Model B (production/full accuracy).

---

## Table of Contents

- [Overview](#overview)
- [File Inventory](#file-inventory)
- [Feature Categories](#feature-categories)
- [Dual Model Architecture](#dual-model-architecture)
- [Data Flow and Leakage Prevention](#data-flow-and-leakage-prevention)
- [Usage Examples](#usage-examples)
- [Feature Tiers](#feature-tiers)

---

## Overview

This folder contains modular feature engineering components that support weekly sales forecasting across multiple targets:

### Prediction Targets
- **Sales** (B&M and WEB channels)
- **AOV** (Average Order Value)
- **Orders** (derived from Sales / AOV)
- **Conversion** (B&M only, requires traffic data)
- **Traffic** (derived via Monte Carlo from Sales, AOV, Conversion)

### Key Principles
1. **Leakage Prevention**: All features use data strictly at or before `origin_week_date` (t0)
2. **Dual Architecture**: Model A (explainability) vs Model B (production accuracy)
3. **Channel-Specific**: B&M and WEB channels have different optimal features
4. **Empirical Validation**: Features validated against driver screening benchmarks

---

## File Inventory

### Core Transformation Utilities

| File | Purpose | Key Functions |
|------|---------|---------------|
| `transforms.py` | Reusable transformation primitives | `compute_lag()`, `compute_rolling_mean()`, `winsorize_mad()`, `encode_cyclical()`, `safe_log()`, `safe_logit()`, `compute_volatility()` |

**transforms.py** provides the foundational building blocks for all feature engineering:
- **Lags**: Time-shifted features with leakage prevention (e.g., lag_1 uses t0-1 week)
- **Rolling Means**: Window-based aggregations with MAD winsorization for outlier robustness
- **Winsorization**: MAD-based (3.5 MAD ≈ 3 standard deviations) to handle heavy-tailed distributions
- **Cyclical Encoding**: Sin/cos transformation for week-of-year (preserves week 1 ≈ week 52)
- **Safe Transformations**: Log and logit with floor/ceiling to prevent -inf/+inf
- **Volatility**: Rolling MAD or standard deviation for uncertainty quantification

### Validation and Quality

| File | Purpose | Key Functions |
|------|---------|---------------|
| `validation.py` | Feature validation against benchmarks | `validate_feature_correlations()`, `validate_feature_coverage()`, `validate_leakage_prevention()`, `validate_channel_specific_features()`, `validate_feature_ranges()` |

**validation.py** ensures feature quality and correctness:
- **Correlation Validation**: Compares actual correlations against driver screening benchmarks (threshold: ±0.02)
- **Coverage Validation**: Checks NaN rates (max 5%) and feature presence
- **Leakage Detection**: Verifies lag/roll features are identical across horizons for same origin
- **Channel Validation**: Ensures WEB-only features are NaN for B&M rows and vice versa
- **Range Validation**: Checks features fall within expected bounds (e.g., percentages in [0,1])

### Feature Categories

| File | Purpose | Feature Tier | Key Features |
|------|---------|--------------|--------------|
| `time_varying_features.py` | Known-in-advance features (seasonality, holidays) | TIER 1-3 | `dma_seasonal_weight` (FE ρ=+0.42 B&M, +0.21 WEB), cyclical week encoding, holiday indicators |
| `dynamics_features.py` | Autoregressive features (sales, AOV, conversion) | TIER 1-2 | Sales lags/rolls (Pooled ρ=0.78-0.83), conversion rolls (FE ρ=+0.37), omnichannel % (FE ρ=+0.26) |
| `static_features.py` | Store DNA and maturity features | TIER 1-2 | `proforma_annual_sales` (Pooled ρ=+0.84), `is_outlet` (ρ=-0.54), `weeks_since_open` (FE ρ=+0.30) |
| `awareness_features.py` | Brand awareness and consideration | TIER 1-3 | `brand_awareness_dma_score` (Pooled ρ=+0.31 WEB, +0.18 B&M), market-level signals |
| `cannibalization.py` | Competitive pressure from new stores | TIER 2-3 | `cannibalization_pressure` (validated -13.8% to -22% impact <10 miles) |
| `crm_features.py` | Customer demographic mix | TIER 3 | CRM percentages (Pooled ρ=0.15-0.21, weak FE signal) |

**Feature Categories Explained:**

#### FE-01: Time-Varying Known-in-Advance Features (`time_varying_features.py`)
Features that depend on `target_week_date` (t0+h) but are deterministic/knowable at forecast time:
- **Seasonality**: DMA-specific seasonal weights using N-1 logic (TIER 1 for B&M: FE ρ=+0.42)
- **Calendar**: Week-of-year (cyclical encoding), fiscal periods, month, quarter
- **Holidays**: Binary indicators, window flags (Black Friday ±1 week, Christmas ±2 weeks), proximity features

#### FE-02: Sales & AOV Dynamics - B&M (`dynamics_features.py`)
Autoregressive features observed at `origin_week_date` (t0):
- **Sales Lags**: 1, 4, 13, 52 weeks (YoY persistence Pooled ρ=+0.78)
- **Sales Rolling Means**: 4, 8, 13 weeks with MAD winsorization (13-week optimal: Pooled ρ=+0.82, FE ρ=+0.15)
- **AOV Rolling Means**: 8, 13 weeks (Pooled ρ=+0.56)
- **Volatility**: 13-week rolling MAD

#### FE-03: Web-Specific Dynamics (`dynamics_features.py`)
WEB channel features (NaN for B&M):
- **Web Traffic**: Allocated web traffic lags/rolls (PRIMARY driver: Pooled ρ=+0.58, log-transformed 0.749)
- **Web Sales**: From Written Sales (SOURCE OF TRUTH) - lags {1,4,13}, rolls {4,8,13}, volatility
- **Web AOV**: Rolling means {4,8}
- **Optimization**: 4-week windows optimal for WEB (vs 13-week for B&M)

#### FE-04: Conversion & Omnichannel (`dynamics_features.py`)
- **Conversion (B&M only)**: Highly autoregressive (FE ρ=+0.37, Pooled ρ=0.81-0.85)
- **Omnichannel %**: TOP actionable lever (FE ρ=+0.26), lags {1,4}, roll_mean_4
- **Product Mix**: Value/premium product percentages (FE ρ=±0.12-0.18)
- **Service Blend**: White glove percentage (FE ρ=+0.13, Pooled ρ=+0.46)

#### FE-05: Static Store DNA (`static_features.py`)
Time-invariant or slowly-changing features:
- **TIER 1 Cross-Sectional**: `proforma_annual_sales` (Pooled ρ=+0.84 STRONGEST), `is_outlet` (ρ=-0.54)
- **TIER 2 Maturity**: `weeks_since_open` at target_week_date (FE ρ=+0.30 for conversion, CRITICAL for new stores)
- **Store Characteristics**: Square footage, region, format, DMA (categorical for CatBoost)

#### FE-06: Cannibalization (`cannibalization.py`)
Competitive pressure from nearby store openings:
- **Formula**: `Σ exp(-dist/8km) × (1 + weeks_since_open/13)` over stores opened <52 weeks, <20 miles
- **Validated Impact**: -13.8% to -22.0% for stores <10 miles from new openings
- **Features**: Pressure metric, distance to nearest new store, count within 10/20 miles

#### FE-07: Brand Awareness (`awareness_features.py`)
Market-level brand metrics from YouGov (39 aggregate markets, 96.7% store coverage):
- **Awareness**: TIER 1 for WEB (Pooled ρ=+0.31), TIER 2 for B&M (ρ=+0.18)
- **Consideration**: TIER 3 (Pooled ρ≤+0.17)
- **Cascading**: market_city → YOUGOV_DMA_MAP → Market, with sister-DMA fallback for unmapped stores

#### FE-08: CRM Demographics (`crm_features.py`)
Customer composition percentages (TIER 3, cross-sectional):
- **Top Features**: Single-family dwelling (ρ=+0.21 WEB AOV), homeowner (ρ=+0.21), income 150k+ (ρ=+0.20)
- **Treatment**: Static/cross-sectional (FE ρ<0.05), NOT actionable operational levers
- **Coverage**: 23 demographic percentages (dwelling, ownership, income, education, age bands)

### Model Architecture

| File | Purpose | Key Components |
|------|---------|----------------|
| `model_views.py` | Dual architecture builder (Model A vs Model B) | `build_model_a_features()`, `build_model_b_features()`, `get_feature_importance_comparison()` |

**model_views.py** implements the dual model architecture:
- **Model A (Actionable/Explainability)**: Business levers + known-in-advance/static ONLY, NO target lags/rolls
- **Model B (Production/Full)**: Complete feature set including autoregressive lags/rolls
- **Rationale**: Autoregressive lags overshadow operational levers in SHAP analysis

---

## Feature Categories

### By Time-Awareness

1. **Known-in-Advance (vary by horizon)**
   - Seasonality: DMA seasonal weights, cyclical week encoding
   - Calendar: Week-of-year, month, quarter, fiscal periods
   - Holidays: Binary indicators, windows, proximity

2. **Observed at t0 (same across all horizons)**
   - Sales/AOV/Conversion lags and rolling means
   - Omnichannel, product mix, service blend
   - Web traffic lags and rolling means

3. **Static/Cross-Sectional (constant)**
   - Store DNA: Proforma sales, outlet flag, square footage
   - CRM demographics: Income, education, age distributions
   - Brand awareness (slow-moving market signal)

4. **Time-Aware at t0+h**
   - `weeks_since_open`: Computed at target_week_date for accurate maturity at forecast horizon

### By Actionability

1. **TIER 1 Operational Levers (Actionable)**
   - Omnichannel percentage (FE ρ=+0.26)
   - Product mix (value/premium %)
   - Service blend (white glove %)
   - Web traffic allocation

2. **TIER 1 Autoregressive (Strong Signal, Not Actionable)**
   - Sales lags/rolls (Pooled ρ=0.78-0.83)
   - Conversion rolls (FE ρ=+0.37)
   - Seasonality (FE ρ=+0.42 B&M)

3. **TIER 2-3 Context Features**
   - Store DNA (proforma, maturity)
   - Cannibalization pressure
   - Brand awareness
   - CRM demographics

### By Channel

1. **Both B&M and WEB**
   - Calendar and seasonality
   - Static store DNA
   - Operational levers (omnichannel, product mix)
   - Brand awareness

2. **B&M Only**
   - Conversion features (requires foot traffic)
   - Cannibalization pressure
   - Store merchandising square footage

3. **WEB Only**
   - Allocated web traffic lags/rolls
   - WEB-specific sales/AOV features
   - DMA web penetration percentage

---

## Dual Model Architecture

### Model A: Actionable/Explainability Model

**Purpose:** Surrogate model for SHAP-based explainability and what-if analysis

**Features Included (16 total):**
- Business levers: Omnichannel, product mix, service blend (NO lags/rolls to avoid dominance)
- Known-in-advance: Seasonality, holidays, calendar (EXCLUDED from explainability to focus on levers)
- Static context: Store DNA, maturity, brand awareness
- Cannibalization pressure

**Features Excluded (58 total):**
- ALL autoregressive lags/rolls of targets (sales, AOV, conversion)
- Web traffic lags/rolls
- Calendar/seasonality (to focus on business levers)
- Operational lever lags/rolls (use snapshot only)

**Rationale:** Autoregressive lags overshadow operational levers in SHAP analysis. Example: Week-1 omnichannel push shows up in lag_1 next week, stealing credit from the operational lever.

**Use Cases:**
- SHAP explanations for stakeholders
- What-if scenario analysis
- Identifying actionable drivers
- NOT for production forecasts (lower accuracy)

### Model B: Production/Full Accuracy Model

**Purpose:** Production forecasting with maximum accuracy

**Features Included:**
- **B&M Channel (57 features):**
  - All Model A features
  - Sales/AOV/conversion autoregressive features
  - B&M-specific features (conversion, cannibalization)

- **WEB Channel (57 features):**
  - All Model A features (excluding B&M-only)
  - Web traffic autoregressive features (allocated web traffic)
  - WEB-specific sales/AOV features (from Written Sales - SOURCE OF TRUTH)
  - DMA web penetration

**Channel-Specific Exclusions:**
- B&M excludes: Web traffic features, WEB sales/AOV features, DMA web penetration
- WEB excludes: Conversion features, cannibalization, merchandising square footage

**Use Cases:**
- Production forecasts
- Accuracy benchmarking
- Model monitoring

### Architecture Comparison

| Aspect | Model A (Explainability) | Model B (Production) |
|--------|-------------------------|----------------------|
| Feature Count | 16 | 57 (B&M), 57 (WEB) |
| Autoregressive Features | NO | YES |
| Calendar/Seasonality | NO (excluded for focus) | YES |
| Primary Use | SHAP explanations, what-if | Production forecasts |
| Accuracy | Lower | Higher |
| Interpretability | High (levers only) | Lower (lag dominance) |

---

## Data Flow and Leakage Prevention

### Critical Principle: Time-Awareness

All features respect the forecast timeline to prevent information leakage:

```
origin_week_date (t0) ──────> horizon ──────> target_week_date (t0 + h)
        │                                             │
        │                                             │
   Observed at t0:                            Known-in-advance at t0:
   - Sales lags/rolls                         - Seasonality
   - Conversion lags/rolls                    - Holidays
   - Omnichannel %                            - Calendar
   - Product mix                              - weeks_since_open (at t0+h)
   - Web traffic
```

### Leakage Prevention Rules

1. **Observed-at-t0 Features**: Use ONLY data ≤ `origin_week_date`
   - `lag_1`: Uses week at (origin_week_date - 1 week)
   - `roll_mean_13`: Uses 13 weeks ending AT origin_week_date (inclusive)
   - Join on: `(store, channel, origin_week_date)`

2. **Known-in-Advance Features**: Use `target_week_date` (t0 + h)
   - Seasonality, holidays, calendar
   - `weeks_since_open`: Computed at target_week_date for accurate maturity at forecast horizon
   - Join on: `(store, target_week_date)` or `(DMA, target_week_date)`

3. **Static Features**: Time-invariant (constant across all periods)
   - Store DNA, proforma sales, outlet flag
   - CRM demographics (slow-moving, treated as static)
   - Join on: `(store)` only

### Validation Checks

The `validation.py` module includes leakage detection:
- **Test**: Lag/roll features should be IDENTICAL across all horizons for same (store, origin_week_date)
- **Logic**: If feature varies by horizon, it's using future data (LEAKAGE)
- **Function**: `validate_leakage_prevention()`

---

## Usage Examples

### 1. Basic Feature Engineering Pipeline

```python
from forecasting.regressor.features.time_varying_features import attach_time_varying_features
from forecasting.regressor.features.dynamics_features import (
    attach_sales_aov_dynamics_bm,
    attach_web_dynamics,
    attach_conversion_omnichannel_features,
    attach_product_mix_features
)
from forecasting.regressor.features.static_features import attach_static_store_features
from forecasting.regressor.features.awareness_features import attach_awareness_features
from forecasting.regressor.features.cannibalization import compute_cannibalization_pressure
from forecasting.regressor.features.crm_features import attach_crm_features

# Load data
canonical_df = load_canonical_table()  # (store, origin, horizon, target) rows
sales_history = load_sales_history()
ecomm_history = load_ecomm_history()
store_master = load_store_master()
dma_seasonality = load_dma_seasonality()
holiday_calendar = load_holiday_calendar()
awareness_data = load_awareness_data()
yougov_map = load_yougov_dma_map()
crm_data = load_crm_mix()

# 1. Time-varying features (known in advance at t0+h)
df = attach_time_varying_features(
    canonical_df,
    dma_seasonality,
    holiday_calendar,
    date_col='target_week_date'
)

# 2. Static store DNA features
df = attach_static_store_features(
    df,
    store_master,
    target_col='target_week_date'
)

# 3. Dynamics features (observed at t0)
df = attach_sales_aov_dynamics_bm(df, sales_history)
df = attach_web_dynamics(df, ecomm_history)
df = attach_conversion_omnichannel_features(df, sales_history)
df = attach_product_mix_features(df, sales_history)

# 4. Awareness features
df = attach_awareness_features(
    df,
    awareness_data,
    yougov_map,
    store_master,  # uses origin_week_date by default to avoid t0+h leakage
)

# 5. Cannibalization features
df = compute_cannibalization_pressure(df, store_master)

# 6. CRM features (optional - TIER 3)
df = attach_crm_features(df, crm_data, include_lags=False, include_rolls=False)
```

### 2. Building Model Views

```python
from forecasting.regressor.features.model_views import (
    build_model_a_features,
    build_model_b_features
)

# Categorical features for CatBoost
categorical_features = ['region', 'format', 'profit_center_nbr', 'dma', 'channel']

# Model B (Production) - Full feature set
model_b_df = build_model_b_features(df, categorical_features)

# Model A (Explainability) - Actionable only
model_a_df = build_model_a_features(df, categorical_features)

# Verify no autoregressive features in Model A
lag_features = [c for c in model_a_df.columns if 'lag_' in c or 'roll_mean_' in c]
target_lags = [f for f in lag_features
               if any(x in f for x in ['log_sales', 'AOV', 'ConversionRate'])]
assert len(target_lags) == 0, f"Model A contains target lags: {target_lags}"
```

### 3. Feature Validation

```python
from forecasting.regressor.features.validation import (
    validate_feature_correlations,
    validate_feature_coverage,
    validate_leakage_prevention,
    validate_channel_specific_features
)

# Validate correlations against driver screening benchmarks
benchmarks = {
    'dma_seasonal_weight': {'label_log_sales_bm': 0.42},
    'log_sales_roll_mean_13': {'label_log_sales_bm': 0.82},
    'pct_omni_channel_roll_mean_4': {'label_logit_conversion': 0.26}
}

corr_results = validate_feature_correlations(
    df, df, benchmarks, threshold=0.02
)
failures = corr_results[corr_results['status'] == 'FAIL']

# Validate coverage (max 5% NaN)
expected_features = [
    'log_sales_lag_1', 'log_sales_roll_mean_13',
    'dma_seasonal_weight', 'proforma_annual_sales'
]
coverage_results = validate_feature_coverage(
    df, expected_features, max_nan_rate=0.05
)

# Detect leakage
leakage_results = validate_leakage_prevention(
    df,
    origin_col='origin_week_date',
    target_col='target_week_date'
)

if leakage_results['status'] == 'FAIL':
    print(f"LEAKAGE DETECTED in {leakage_results['n_violations']} features!")
    print(leakage_results['violating_features'])

# Validate channel-specific features
channel_map = {
    'WEB': ['allocated_web_traffic_roll_mean_4'],
    'B&M': ['ConversionRate_roll_mean_13']
}
channel_results = validate_channel_specific_features(df, 'channel', channel_map)
```

### 4. Transform Utilities

```python
from forecasting.regressor.features.transforms import (
    compute_lag,
    compute_rolling_mean,
    encode_cyclical,
    safe_log,
    winsorize_mad
)

# Compute custom lag feature
df = compute_lag(
    df,
    group_cols=['profit_center_nbr', 'channel'],
    date_col='week_date',
    value_col='sales',
    lag_weeks=4
)

# Compute rolling mean with winsorization
df = compute_rolling_mean(
    df,
    group_cols=['profit_center_nbr', 'channel'],
    date_col='week_date',
    value_col='sales',
    window_weeks=13,
    winsorize=True,
    n_mad=3.5
)

# Cyclical encoding for week-of-year
df['sin_woy'], df['cos_woy'] = encode_cyclical(df['woy'], period=52)

# Safe log transformation
df['log_sales'] = safe_log(df['sales'], floor=1e-6)
```

---

## Feature Tiers

Features are organized into tiers based on empirical correlation strength with targets:

### TIER 1: Strongest Predictors (FE ρ > 0.25 OR Pooled ρ > 0.50)

**Cross-Sectional (Pooled ρ):**
- `proforma_annual_sales`: Pooled ρ=+0.84 (STRONGEST cross-sectional predictor)
- `log_sales_roll_mean_{4,8,13}`: Pooled ρ=0.78-0.83 (autoregressive)
- `ConversionRate_roll_mean_{4,8,13}`: Pooled ρ=0.81-0.85 (B&M)
- `is_outlet`: Pooled ρ=-0.54 (negative effect)
- `AOV_roll_mean_{8,13}`: Pooled ρ=+0.56
- `allocated_web_traffic_roll_mean_4`: Pooled ρ=+0.58 (WEB, log-transformed: 0.749)

**Fixed-Effects (FE ρ - Within-Store Over Time):**
- `dma_seasonal_weight`: FE ρ=+0.42 (B&M), +0.21 (WEB)
- `ConversionRate_roll_mean_4`: FE ρ=+0.37 (B&M)
- `brand_awareness_dma_score`: Pooled ρ=+0.31 (WEB, top-15 driver)
- `weeks_since_open`: FE ρ=+0.30 (conversion, CRITICAL for new stores)
- `pct_omni_channel_roll_mean_4`: FE ρ=+0.26 (TOP actionable lever)
- `ConversionRate_lag_1`: FE ρ=+0.26-0.31 (B&M)

### TIER 2: Moderate Predictors (FE ρ 0.10-0.25 OR Pooled ρ 0.20-0.50)

- `log_sales_roll_mean_13`: FE ρ=+0.15 (B&M optimal window)
- `log_sales_lag_1`: FE ρ=+0.14
- `pct_white_glove_roll_mean_4`: FE ρ=+0.13 (Pooled ρ=+0.46)
- `AOV_roll_mean_13`: FE ρ=+0.17
- `brand_awareness_dma_score`: Pooled ρ=+0.18 (B&M), +0.21 (conversion)
- `pct_value_product_roll_mean_4`: FE ρ=-0.12 (negative - actionable)
- `pct_premium_product_roll_mean_4`: FE ρ=-0.18 (negative)

### TIER 3: Weak Predictors (FE ρ < 0.10, Pooled ρ < 0.20)

- `weeks_to_holiday`: FE ρ=+0.10
- `is_pre_holiday_1wk`: FE ρ=-0.09
- `is_pre_holiday_2wk`: FE ρ=-0.07
- `is_pre_holiday_3wk`: FE ρ=-0.04
- `brand_consideration_dma_score`: Pooled ρ≤+0.17
- **All CRM features**: Pooled ρ=0.15-0.21, FE ρ<0.05 (cross-sectional only)
- **Cannibalization**: Event-specific (5-10% improvement for affected stores)

### Correlation Types

- **Pooled ρ**: Cross-sectional + temporal correlation (captures baseline differences between stores)
- **FE ρ**: Within-store correlation over time (captures operational dynamics, controls for store fixed effects)

---

## Key Design Decisions

### 1. MAD Winsorization (n_mad=3.5)
- **Why**: Robust to outliers in heavy-tailed distributions (vs standard deviation)
- **Formula**: Clip to median ± 3.5 × MAD, where MAD = median(|x - median(x)|)
- **Equivalence**: 3.5 MAD ≈ 3 standard deviations for normal distributions
- **Applied to**: All rolling means of sales, AOV, conversion, product mix

### 2. Channel-Specific Window Sizes
- **B&M**: 13-week windows optimal (seasonality FE ρ=+0.42)
- **WEB**: 4-week windows optimal (seasonality FE ρ=+0.21, faster dynamics)
- **Rationale**: WEB channel has faster-moving trends, shorter memory

### 3. Data Source Hierarchy for WEB
1. **Sales/AOV**: Written Sales WEB channel (SOURCE OF TRUTH)
   - Validation showed 12.6% median difference vs Ecomm MerchAmt
2. **Traffic**: Ecomm Traffic AllocatedWebTraffic (validated predictive, ρ=0.749)
3. **EXCLUDED**: UnallocatedWebTraffic (measures web sessions, not comparable to Store_Traffic)

### 4. Awareness Cascading Logic
- **Map**: market_city (store master) → Market (YOUGOV_DMA_MAP) → awareness scores
- **Coverage**: 96.7% of operational stores
- **Fallback**: Sister-DMA mapping for 11 unmapped stores
- **Frequency**: Weekly data with forward-fill for gaps

### 5. Cannibalization Formula Parameters
- **λ = 8 km**: Primary impact zone (distance decay)
- **τ = 13 weeks**: Effects INTENSIFY (not decay) over first 13 weeks
- **Max Distance**: 20 miles (32 km) cutoff
- **Lookback**: 52 weeks (only recent openings)
- **Validated**: -13.8% to -22.0% impact for stores <10 miles from new openings

---

## References

- **Driver Screening Results**: Empirical correlations for all features
- **YOUGOV_DMA_MAP**: Sheet 25 in Master Tables workbook
- **Cannibalization Validation**: Event study analysis results
- **Model A Features**: `model_a_features.md`
- **Model B Features**: `model_b_features.md`

---

## Notes

- All features computed at store-week-channel granularity
- CatBoost handles categorical features natively (no one-hot encoding needed)
- NaN handling: Features allow NaN for missing data (CatBoost handles gracefully)
- Feature importance: Available via SHAP (Model A) and CatBoost native (Model B)
- Validation thresholds: Correlation ±0.02, NaN rate ≤5%, coverage ≥96.7%
