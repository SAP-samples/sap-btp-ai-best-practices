# Canonical Training Table (ETL) Module

**Status**: EPIC 3 + EPIC 4 Complete ✅
**Schema Version**: 1.0.0
**Last Updated**: November 2025

## Overview

This module provides the **single source of truth** for building the canonical training table used for Company X sales forecasting models. It implements both Epic 3 (ETL pipeline) and Epic 4 (Feature Engineering) requirements from `PROJECT_PLAN.md` and follows specifications in `ENGINEERING_INSTRUCTIONS.md`.

The ETL module serves as the foundation for two model variants:
- **Model A**: Actionable business levers only (for SHAP explainability)
- **Model B**: Full autoregressive features including lags/rolls (for production forecasts)

Both models predict 5 targets across 2 channels (B&M and WEB):
- **Targets**: Sales, AOV, Orders, Conversion, Traffic
- **Channels**: B&M (Brick-and-Mortar), WEB (E-commerce)

### Scope

**Epic 3 - ETL Pipeline:**
- **Canonical schema definition** (`schema.py`) - Strict typing and validation rules
- **History exploder** - Generates (store, channel, origin, horizon, target) prediction tasks
- **Target attachment** - Channel-aware calculation of log(sales), log(AOV), logit(conversion)
- **Data quality** - Traffic flag filtering, missing value handling
- **Leakage prevention** - Strict temporal guards for train/validation splits

**Epic 4 - Feature Engineering:**
- **12-step orchestrated pipeline** - Automated feature generation in dependency order
- **~100-120 features** - From time-varying seasonality to static store DNA
- **Model variant filtering** - Separate feature sets for Model A vs Model B
- **Multi-source integration** - Awareness, CRM, cannibalization, dynamics, omnichannel

---

## Files in This Folder

| File | Lines | Purpose |
|------|-------|---------|
| **`canonical_table.py`** | 670 | Main ETL orchestrator. Implements data loading, history explosion, target calculation, and Epic 4 feature engineering pipeline. Contains `build_canonical_training_table()` - the primary public API. |
| **`schema.py`** | 314 | Schema definition and validation framework. Defines `CANONICAL_TABLE_SCHEMA` with column specs (name, dtype, description, required, channel-specific, value ranges). Implements `validate_canonical_table()` with 7 validation rules. |
| **`test_canonical_table.py`** | 380 | Comprehensive unit tests for Epic 3 completion. Tests AOV calculation (Sales/Orders, not AUR), WEB sales source (Ecomm Traffic), conversion filtering (B&M only), traffic flag application, and schema validation. |
| **`__init__.py`** | 24 | Public API exports. Exposes `build_canonical_training_table()`, `explode_history()`, `attach_targets()`, `attach_features()`, schema utilities. |
| **`README.md`** | This file | Comprehensive documentation for the ETL module. |

---

## ETL in the Forecasting Model Context

### Position in the Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES (Epic 1)                         │
├─────────────────────────────────────────────────────────────────────┤
│ • Written Sales Data.csv (B&M sales, orders, traffic)               │
│ • Ecomm Traffic.csv (WEB sales - authoritative)                     │
│ • Store Master (DMA, format, proforma sales)                        │
│ • Awareness & Consideration (DMA-level brand metrics)               │
│ • CRM Demographics Mix (23 customer segment percentages)            │
│ • Holiday Calendar (known-in-advance events)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   LAYER 0 ARTIFACTS (Epic 2)                         │
├─────────────────────────────────────────────────────────────────────┤
│ • DMA Seasonality Weights (detrended, normalized)                   │
│ • WEB Global Seasonality (N-1 logic)                                │
│ • Sister-DMA Mappings (fallback for new stores)                     │
│ • YouGov DMA Map (awareness → store mapping)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│           ★ CANONICAL TRAINING TABLE (Epic 3 + 4 - THIS MODULE) ★   │
├─────────────────────────────────────────────────────────────────────┤
│ EPIC 3 - ETL Pipeline:                                              │
│   1. Load & clean data from multiple sources                        │
│   2. Explode history: (store, channel, origin, horizon) → rows      │
│   3. Attach targets: log(sales), log(AOV), logit(conversion)        │
│   4. Validate schema: 7 rules, channel-specific logic               │
│                                                                      │
│ EPIC 4 - Feature Engineering (12 steps):                            │
│   [FE-01] Time-varying known-in-advance (seasonality, holidays)     │
│   [FE-02] Sales & AOV dynamics (B&M: lags 1/4/13/52, rolls 4/8/13)  │
│   [FE-03] Web dynamics (sales rolls, allocated traffic)             │
│   [FE-04] Conversion & omnichannel (logit conv, pct_omni_channel)   │
│   [FE-05] Static store DNA (proforma sales, is_outlet, weeks open)  │
│   [FE-06] Cannibalization pressure (time-varying distance decay)    │
│   [FE-07] DMA awareness & consideration (96.7% store coverage)      │
│   [FE-08] CRM demographics (23 customer segments, optional)         │
│   + Product mix & service blend features                            │
│   + Model variant filtering (A vs B)                                │
│                                                                      │
│ OUTPUT: ~100-120 features × N prediction tasks                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING (Epic 5)                           │
├─────────────────────────────────────────────────────────────────────┤
│ • Model A: Actionable levers (SHAP/explainability)                  │
│ • Model B: Full features (production forecasts)                     │
│ • 5 CatBoost models: Sales, AOV, Orders, Conversion, Traffic        │
│ • Multi-target coherence prioritized over per-target quantiles      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                PREDICTION & DEPLOYMENT (Epic 6+)                     │
├─────────────────────────────────────────────────────────────────────┤
│ • Rolling accuracy validation (h=1-52 weeks)                        │
│ • SHAP feature importance (Model A only)                            │
│ • Business recommendations (actionable insights)                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Why the Canonical Table is Critical

1. **Single Source of Truth**: All models train on identical data structure, ensuring consistency
2. **Leakage Prevention**: Strict separation of features at t0 (observed) vs t0+h (known-in-advance)
3. **Channel-Aware Logic**: Different rules for B&M vs WEB (e.g., conversion only for B&M)
4. **Temporal Integrity**: Prevents using future information in historical prediction tasks
5. **Reproducibility**: Schema validation ensures data quality across experiments

---

## Data Sources

### B&M (Brick-and-Mortar) Channel

| Metric | Source | Notes |
|--------|--------|-------|
| **Sales** | `Written Sales Data.csv` | Total weekly revenue |
| **Orders** | `Written Sales Data.csv` | Transaction count |
| **Traffic** | `Written Sales Data.csv` (`store_traffic` column) | Physical visitor count |
| **has_traffic_data** | Computed from traffic missingness | Binary flag for reliable traffic |

### WEB (E-commerce) Channel

| Metric | Source | Notes |
|--------|--------|-------|
| **Sales** | **`Ecomm Traffic.csv` (`merch_amt`)** | **AUTHORITATIVE** WEB sales source |
| **Orders** | `Written Sales Data.csv` (WEB rows) | For AOV calculation only |
| **Traffic** | N/A | Set to NaN (no physical traffic concept) |
| **has_traffic_data** | Always 0 | WEB has no traffic data |

**Critical**: Do NOT use WEB rows from `Written Sales Data.csv` for sales amounts. Use `Ecomm Traffic.csv` as the source of truth per ENGINEERING_INSTRUCTIONS.md lines 32-33.

---

## Target Variables

### Primary Targets (Transformed)

| Target | Formula | Channels | Description |
|--------|---------|----------|-------------|
| **label_log_sales** | `log(total_sales)` | B&M, WEB | Natural log of weekly sales |
| **label_log_aov** | `log(total_sales / order_count)` | B&M, WEB | Natural log of Average Order Value |
| **label_logit_conversion** | `logit(orders / traffic)` | **B&M only** | Logit-transformed conversion rate |

### Target Calculation Rules

#### Conversion (Channel-Specific)

```python
# WEB channel
label_logit_conversion = NaN  # No physical traffic

# B&M channel with has_traffic_data = 0
label_logit_conversion = NaN  # Unreliable traffic data

# B&M channel with has_traffic_data = 1
conversion = order_count / store_traffic
label_logit_conversion = logit(conversion)  # Valid conversion
```

**Rationale**:
- WEB has no physical traffic (Features.md line 89)
- B&M stores without reliable traffic data excluded from Conversion Model training
- B&M stores kept for Sales/AOV training even without traffic

---

## Data Transformations & Key ETL Functions

### Epic 3 Functions (Core ETL)

#### 1. `_prepare_sales_with_dma()` - Data Loading & Unification

**Purpose**: Load sales data from correct sources and attach DMA info.

**Transformations**:
- Load B&M sales from `Written Sales Data.csv` (channel != 'WEB')
- Load WEB sales from `Ecomm Traffic.csv` (merch_amt → total_sales)
- Load WEB orders from `Written Sales Data.csv` WEB rows (for AOV only)
- Merge store master for DMA mapping (profit_center_nbr → market_city)
- Compute `has_traffic_data` flag (1 if reliable traffic, 0 otherwise)
- Combine B&M and WEB into unified dataframe
- Impute missing DMAs using sister-store fallback

**Output**: DataFrame with columns: `profit_center_nbr`, `dma`, `channel`, `origin_week_date`, `total_sales`, `order_count`, `store_traffic`, `aur`, `has_traffic_data`

#### 2. `explode_history()` - History Explosion

**Purpose**: Create (store, dma, channel, origin_week_date, horizon, target_week_date) rows for all prediction tasks.

**Transformations**:
- Cross-join sales data with horizons (default: 1-52 weeks)
- Calculate `target_week_date = origin_week_date + horizon weeks`
- Drop rows with missing origin_week_date
- Each origin week generates 52 prediction tasks (one per horizon)

**Example**:
```
Input:  1 store × 1 channel × 100 weeks of history
Output: 1 store × 1 channel × 100 origin weeks × 52 horizons = 5,200 rows
```

#### 3. `attach_targets()` - Target Calculation

**Purpose**: Attach target values and labels at target_week_date with channel-aware logic.

**Transformations**:
1. **Sales Target** (all channels):
   - `label_log_sales = log(total_sales)` with floor of 1e-6

2. **AOV Target** (all channels):
   - `aov = total_sales / order_count`
   - `label_log_aov = log(aov)` with floor of 1e-6
   - **CRITICAL**: Uses orders, NOT AUR (sales/units)

3. **Conversion Target** (B&M only with valid traffic):
   - Initialize all as NaN
   - Calculate only for `channel != 'WEB'` AND `has_traffic_data == 1`
   - `conversion = order_count / store_traffic`
   - `label_logit_conversion = logit(conversion)` with epsilon clipping (1e-6, 1-1e-6)

4. **Drop Missing Targets** (if enabled):
   - Always drop if Sales or AOV missing
   - Drop B&M rows with `has_traffic_data=1` but missing Conversion
   - Keep WEB rows (conversion expected to be NaN)
   - Keep B&M rows with `has_traffic_data=0` (for Sales/AOV training only)

#### 4. `_safe_log()` & `_safe_logit()` - Safe Transformations

**Purpose**: Apply log/logit transformations with numerical stability.

**Transformations**:
- `_safe_log()`: Clips values to floor (1e-6) before log to prevent -inf
- `_safe_logit()`: Clips probabilities to (1e-6, 1-1e-6) before logit to prevent inf

### Epic 4 Functions (Feature Engineering)

#### 5. `attach_all_features()` - Feature Engineering Orchestrator

**Purpose**: Coordinate all 12 feature engineering steps in priority order.

**Steps**:

1. **[FE-01] Time-Varying Known-in-Advance Features**
   - Seasonality: `dma_seasonal_weight` (B&M), `web_seasonal_weight` (WEB)
   - Calendar: `woy_sin`, `woy_cos`, `month`, `quarter`, `is_month_start/end`
   - Holidays: `is_holiday`, `weeks_to_holiday`, `weeks_since_holiday`
   - **Temporal Availability**: Known at target_week_date (t0+h)

2. **[FE-02] Sales & AOV Dynamics (B&M)**
   - Lags: `log_sales_lag_{1,4,13,52}`, `log_aov_lag_{1,4,13,52}`
   - Rolling means: `log_sales_roll_mean_{4,8,13}`, `log_aov_roll_mean_{4,8,13}`
   - Winsorization: MAD-based outlier handling on rolling windows
   - **Temporal Availability**: Observed at origin_week_date (t0)
   - **Model Variant**: Model B only (excluded from Model A for explainability)

3. **[FE-03] Web-Specific Dynamics**
   - Source: `Ecomm Traffic.csv` (NOT Written Sales)
   - Lags: `log_web_sales_lag_{1,4}`, `allocated_web_traffic_lag_{1,4}`
   - Rolling means: `log_web_sales_roll_mean_4`, `allocated_web_traffic_roll_mean_4`
   - **Temporal Availability**: Observed at t0
   - **Model Variant**: Model B only

4. **[FE-04] Conversion & Omnichannel Features**
   - Conversion dynamics: `logit_conversion_lag_{1,4,13}`, `logit_conversion_roll_mean_{4,8,13}`
   - Omnichannel: `pct_omni_channel_lag_{1,4}`, `pct_omni_channel_roll_mean_{4,8}`
   - **Filtering**: Only calculated for B&M stores with `has_traffic_data=1`
   - **Temporal Availability**: Observed at t0
   - **Model Variant**: Mixed (omnichannel in Model A, lags/rolls in Model B)

5. **[FE-05] Static Store DNA**
   - `proforma_annual_sales`: Expected annual revenue (from store master)
   - `is_outlet`: Binary flag for outlet vs regular store
   - `weeks_since_open`: Store age at target_week_date (t0+h)
   - `region`, `format`: Categorical store attributes
   - **Temporal Availability**: Static or known at t0+h
   - **Model Variant**: Both Model A and B (actionable context)

6. **[FE-06] Cannibalization Pressure**
   - Formula: `Pressure_i = Σ(j≠i) exp(-dist_ij/8) × (1 + weeks_open_j/13)`
   - Time-varying: Updates as new stores open
   - Uses pairwise store distances and opening dates
   - **Temporal Availability**: Known at t0+h
   - **Model Variant**: Both Model A and B (market context)

7. **[FE-07] DMA Awareness & Consideration**
   - Source: `Awareness_Consideration_2022-2025.xlsx`
   - Mapping: Store DMA → YouGov Market (96.7% coverage)
   - Features: `brand_awareness_dma_score`, `brand_consideration_dma_score`
   - Fallback: Sister-DMA for 7 unmapped stores
   - **Temporal Availability**: Known at target_week_date (forward-filled)
   - **Model Variant**: Both Model A and B (Tier 1 for WEB, Tier 2 for B&M)
   - **Importance**: Top-15 WEB driver (Pooled ρ=+0.31)

8. **[FE-08] CRM Demographics (Optional)**
   - 23 customer segment percentages (e.g., homeowners, single-family, high income)
   - Lags: 1, 4 weeks
   - Rolling windows: 4, 8 weeks
   - **Temporal Availability**: Observed at t0 (lags/rolls)
   - **Model Variant**: Both Model A and B (Tier 3 cross-sectional)
   - **Privacy**: PII stripped, percentages normalized to ≤1

9. **Product Mix & Service Blend**
   - Category mix percentages (apparel, accessories, home)
   - Service blend (alterations, personal styling)
   - **Temporal Availability**: Observed at t0
   - **Model Variant**: Model B only

10. **Model Variant Filtering**
    - **Model A** (actionable levers): Exclude lags, rolls, autoregressive features
    - **Model B** (production): Include all features (~100-120 total)
    - Categorical features: `profit_center_nbr`, `dma`, `channel`, `region`, `format`

#### 6. `build_canonical_training_table()` - Main Public API

**Purpose**: Single entry point to build the complete canonical training table.

**Parameters**:
- `horizons`: Forecast horizons (default: 1-52 weeks)
- `drop_missing_targets`: Drop rows with missing required targets (default: True)
- `validate`: Validate output against schema (default: True)
- `include_features`: Run Epic 4 feature engineering (default: False for backward compatibility)
- `model_variant`: 'A' (explainability) or 'B' (production, default)
- `include_crm`: Include CRM demographics (default: False, Tier 3 optional)

**Workflow**:
1. Load and prepare data with DMA mapping
2. Explode history to create prediction tasks
3. Attach targets with channel-aware logic
4. (Optional) Attach all features via 12-step pipeline
5. Order columns for readability
6. Validate schema if enabled
7. Return canonical table

**Output**: DataFrame with 15-130 columns depending on `include_features`:
- Keys (6): `profit_center_nbr`, `dma`, `channel`, `origin_week_date`, `horizon`, `target_week_date`
- Targets (3): `label_log_sales`, `label_log_aov`, `label_logit_conversion`
- Metadata (1): `has_traffic_data`
- Raw (4, optional): `total_sales`, `order_count`, `store_traffic`, `aur`
- Features (~100-120, if `include_features=True`)

---

## Schema

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `profit_center_nbr` | int64 | Store identifier |
| `dma` | object | Designated Market Area |
| `channel` | object | 'B&M' or 'WEB' |
| `origin_week_date` | datetime64 | Origin week (t0) - last known data |
| `horizon` | int64 | Forecast horizon in weeks (1-52) |
| `target_week_date` | datetime64 | Target week (t0 + h) - being forecasted |

### Target Columns

| Column | Type | Required | Channels |
|--------|------|----------|----------|
| `label_log_sales` | float64 | Yes | B&M, WEB |
| `label_log_aov` | float64 | Yes | B&M, WEB |
| `label_logit_conversion` | float64 | B&M with traffic only | B&M (NaN for WEB) |

### Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `has_traffic_data` | int64 | 1 if reliable traffic (B&M), 0 otherwise |

### Raw Columns (Optional, for debugging)

| Column | Type | Description |
|--------|------|-------------|
| `total_sales` | float64 | Raw sales before log transform |
| `order_count` | float64 | Number of transactions |
| `store_traffic` | float64 | Physical visitors (B&M only) |
| `aur` | float64 | Average Unit Retail (NOT used for AOV) |

**Full schema**: See `schema.py` for complete specification and validation rules.

---

## Usage

### Basic Usage (Epic 3 Only - Keys + Targets)

```python
from forecasting.regressor.etl import build_canonical_training_table

# Build table for all 52 horizons (keys + targets only, no features)
df = build_canonical_training_table()

# Inspect
print(df[["profit_center_nbr", "channel", "horizon", "label_log_sales"]].head())
print(f"Shape: {df.shape}")  # e.g., (500000, 14) rows × columns
```

### Custom Horizons

```python
# Short-term forecasting only (h=1-13)
df_short = build_canonical_training_table(horizons=range(1, 14))

# Specific horizons
df_custom = build_canonical_training_table(horizons=[1, 4, 13, 26, 52])
```

### Epic 4 - Full Feature Engineering (Recommended)

```python
# Build Model B (production forecasts with all features)
df_model_b = build_canonical_training_table(
    horizons=range(1, 53),
    include_features=True,      # Enable Epic 4 pipeline
    model_variant='B',           # Full features (lags, rolls, dynamics)
    include_crm=False            # Exclude CRM (optional Tier 3)
)

print(f"Model B Shape: {df_model_b.shape}")  # e.g., (500000, 125)
```

```python
# Build Model A (explainability with actionable levers only)
df_model_a = build_canonical_training_table(
    horizons=range(1, 53),
    include_features=True,
    model_variant='A',           # Exclude lags/rolls for SHAP
    include_crm=False
)

print(f"Model A Shape: {df_model_a.shape}")  # e.g., (500000, 45)
```

### Include CRM Demographics (Tier 3)

```python
# Build with CRM demographic features
df_with_crm = build_canonical_training_table(
    include_features=True,
    model_variant='B',
    include_crm=True             # Include 23 customer segment features
)
```

### Legacy Usage (Deprecated - Manual Feature Attachment)

```python
# Prepare features observed at t0 (lags, rolling means, etc.)
features_t0 = pd.DataFrame({
    'profit_center_nbr': [101, 101],
    'channel': ['B&M', 'B&M'],
    'origin_week_date': ['2024-01-01', '2024-01-08'],
    'log_sales_lag_1': [11.2, 11.3],
    'log_sales_roll_mean_13': [11.1, 11.15],
})

# Prepare known-in-advance features at t0+h (seasonality, holidays, etc.)
features_tfuture = pd.DataFrame({
    'profit_center_nbr': [101],
    'channel': ['B&M'],
    'target_week_date': ['2024-02-01'],
    'dma_seasonal_weight': [1.15],
    'is_holiday': [1],
})

# Build table with manual features (Epic 3 only)
df = build_canonical_training_table(
    features_t0=features_t0,
    features_tfuture=features_tfuture,
    include_features=False       # Use legacy manual attachment
)
```

**Note**: The legacy approach is deprecated. Use `include_features=True` with `model_variant='A'/'B'` for automated feature engineering.

### Disable Validation (for speed)

```python
# Skip schema validation if you're confident data is clean
df = build_canonical_training_table(
    include_features=True,
    validate=False               # Skip validation for faster execution
)
```

---

## Validation

### Schema Validation

```python
from forecasting.regressor.etl import validate_canonical_table

# Validate dataframe
errors = validate_canonical_table(df, strict=False, check_ranges=True)

if errors:
    for err in errors:
        print(f"  - {err}")
else:
    print("✓ Schema validation passed")
```

### Validation Rules

1. **WEB channel**: `label_logit_conversion` must be NaN
2. **B&M with has_traffic_data=0**: `label_logit_conversion` must be NaN
3. **B&M with has_traffic_data=1**: `label_logit_conversion` should be valid
4. **horizon**: Must be in range [1, 52]
5. **has_traffic_data**: Must be 0 or 1
6. **No duplicates**: Unique (store, channel, origin_week_date, horizon) combinations
7. **No infinites**: Target columns cannot contain inf/-inf values

---

## Architecture

### Module Components

```
etl/
├── canonical_table.py    # Main ETL + Feature Engineering orchestrator (670 lines)
├── schema.py             # Schema definition & validation (ETL-01) (314 lines)
├── test_canonical_table.py  # Unit tests for Epic 3 (380 lines)
├── README.md             # This file
└── __init__.py           # Public API exports
```

### Function Flow

```
build_canonical_training_table()
  │
  ├─> _prepare_sales_with_dma()
  │    ├─> load_written_sales_with_flags()  [B&M]
  │    ├─> load_ecomm_traffic()             [WEB sales]
  │    ├─> Merge WEB sales + WEB orders
  │    └─> _impute_dma()                    [Sister-store fallback]
  │
  ├─> explode_history()                     [ETL-02]
  │    └─> Cross-join stores × horizons
  │
  ├─> attach_targets()                      [ETL-03]
  │    ├─> Calculate log(sales), log(AOV)
  │    ├─> Calculate logit(conversion) [B&M only, with traffic]
  │    └─> Drop missing targets [channel-aware]
  │
  ├─> attach_all_features()  [EPIC 4 - if include_features=True]
  │    │
  │    ├─> [1/12] Load Layer 0 artifacts (seasonality, holidays)
  │    ├─> [2/12] [FE-01] Time-varying known-in-advance
  │    ├─> [3/12] [FE-05] Static store DNA
  │    ├─> [4/12] [FE-02] B&M sales/AOV dynamics
  │    ├─> [5/12] [FE-03] Web-specific dynamics
  │    ├─> [6/12] [FE-04] Conversion & omnichannel
  │    ├─> [7/12] Product mix & service blend
  │    ├─> [8/12] [FE-06] Cannibalization pressure
  │    ├─> [9/12] [FE-07] DMA awareness & consideration
  │    ├─> [10/12] [FE-08] CRM demographics [optional]
  │    ├─> [11/12] Build model variant view (A or B)
  │    └─> [12/12] Complete (return ~100-120 features)
  │
  └─> validate_canonical_table()            [Optional]
       ├─> Check required columns
       ├─> Verify data types
       ├─> Validate value ranges
       ├─> Check channel-specific rules
       ├─> Detect duplicates
       └─> Find infinite values
```

### Leakage Prevention

**Critical**: Features are strictly separated by temporal availability:

- **features_t0**: Observed at `origin_week_date` (t0)
  - Sales lags, rolling means, conversion dynamics
  - Product mix, CRM demographics
  - NO access to data from t0+1 onwards

- **features_tfuture**: Known at `target_week_date` (t0+h)
  - DMA seasonal weights, holiday indicators, fiscal calendar
  - Weather normals (climate averages)
  - Store age (weeks_since_open at target week)
  - Cannibalization pressure (updates as stores open)

**Never** mix these! Ensure `features_t0` is computed using ONLY data available at or before t0.

### Model Variant Comparison

| Feature Category | Model A (Explainability) | Model B (Production) |
|-----------------|-------------------------|---------------------|
| **Keys** | ✅ | ✅ |
| **Targets** | ✅ | ✅ |
| **Time-varying (FE-01)** | ✅ Seasonality, holidays, calendar | ✅ Same |
| **Sales/AOV Dynamics (FE-02)** | ❌ No lags/rolls | ✅ Lags 1/4/13/52, Rolls 4/8/13 |
| **Web Dynamics (FE-03)** | ❌ No lags/rolls | ✅ Lags 1/4, Rolls 4 |
| **Conversion (FE-04)** | ⚠️ Omnichannel % only | ✅ Full lags/rolls |
| **Store DNA (FE-05)** | ✅ Proforma, outlet, age | ✅ Same |
| **Cannibalization (FE-06)** | ✅ Pressure score | ✅ Same |
| **Awareness (FE-07)** | ✅ DMA scores | ✅ Same |
| **CRM (FE-08)** | ✅ If enabled | ✅ If enabled |
| **Product Mix** | ❌ Excluded | ✅ Category %s |
| **Categorical Encoding** | ✅ Store, DMA, channel | ✅ Same |
| **Total Features** | ~35-50 | ~100-120 |
| **Use Case** | SHAP analysis, business insights | Production forecasts |

---

## Channel-Specific Behavior

### B&M (Brick-and-Mortar)

- Sales from `Written Sales Data.csv`
- Traffic from `store_traffic` column
- Conversion calculated for stores with `has_traffic_data=1`
- Stores without traffic excluded from Conversion Model training but kept for Sales/AOV
- DMA seasonality used (detrended, normalized)

### WEB (E-commerce)

- **Sales from `Ecomm Traffic.csv` (merch_amt)**
- Orders from `Written Sales Data.csv` WEB rows
- Traffic = NaN (no physical traffic concept)
- Conversion = NaN (always)
- `has_traffic_data = 0` (always)
- Global WEB seasonality used (N-1 logic)

---

## Examples

### Check Data Sources

```python
df = build_canonical_training_table(horizons=[1], include_features=False)

# Verify WEB sales come from Ecomm Traffic
web_df = df[df['channel'] == 'WEB']
print(f"WEB rows: {len(web_df)}")
print(f"WEB conversion (should be all NaN): {web_df['label_logit_conversion'].isna().all()}")

# Verify B&M conversion filtering
bm_df = df[df['channel'] != 'WEB']
bm_with_traffic = bm_df[bm_df['has_traffic_data'] == 1]
bm_no_traffic = bm_df[bm_df['has_traffic_data'] == 0]

print(f"B&M with traffic: {len(bm_with_traffic)}")
print(f"B&M without traffic (conv should be NaN): {bm_no_traffic['label_logit_conversion'].isna().all()}")
```

### View Schema

```python
from forecasting.regressor.etl.schema import print_schema_documentation

# Print full schema to console
print_schema_documentation()
```

### Feature Engineering Progress

```python
# Monitor feature engineering progress
df = build_canonical_training_table(
    horizons=range(1, 53),
    include_features=True,
    model_variant='B'
)

# Output will show:
# EPIC 4 Feature Engineering: Starting feature attachment...
# [1/12] Loading Layer 0 artifacts...
# [2/12] [FE-01] Attaching time-varying features...
#         Added 15 features
# [3/12] [FE-05] Attaching static store DNA features...
#         Added 8 features
# ...
# [12/12] Feature engineering complete!
#         Total features attached: 112
#         Final dataframe: 487234 rows × 125 columns
```

### Compare Model A vs Model B

```python
# Build both variants
df_a = build_canonical_training_table(
    horizons=[1, 13, 52],
    include_features=True,
    model_variant='A'
)

df_b = build_canonical_training_table(
    horizons=[1, 13, 52],
    include_features=True,
    model_variant='B'
)

# Compare column counts
key_cols = ['profit_center_nbr', 'dma', 'channel', 'origin_week_date', 'horizon', 'target_week_date']
target_cols = ['label_log_sales', 'label_log_aov', 'label_logit_conversion']

features_a = [c for c in df_a.columns if c not in key_cols + target_cols + ['has_traffic_data']]
features_b = [c for c in df_b.columns if c not in key_cols + target_cols + ['has_traffic_data']]

print(f"Model A features: {len(features_a)}")  # e.g., 35
print(f"Model B features: {len(features_b)}")  # e.g., 112

# Find Model B exclusive features (lags/rolls)
exclusive_b = set(features_b) - set(features_a)
print(f"\nModel B exclusive (autoregressive): {len(exclusive_b)}")
print(sorted(exclusive_b)[:10])  # e.g., log_sales_lag_1, log_sales_roll_mean_4, ...
```

---

## Testing

Run unit tests to verify correctness:

```bash
pytest forecasting/regressor/etl/test_canonical_table.py -v
```

Key test coverage:
- ✅ AOV calculation (Sales / Orders, not AUR)
- ✅ WEB sales from Ecomm Traffic
- ✅ Conversion filtering (B&M with traffic only)
- ✅ Traffic flag application
- ✅ Schema validation
- ✅ Channel-specific rules
- ✅ Safe log/logit transformations
- ✅ History explosion logic
- ✅ Drop missing targets (channel-aware)

---

## Troubleshooting

### Common Issues

1. **"Missing required columns" validation error**
   - Ensure all data sources are available (Written Sales, Ecomm Traffic, Store Master)
   - Check column names match expected schema (case-sensitive)

2. **"WEB has non-NaN conversion" validation error**
   - Check data loading: WEB sales should come from Ecomm Traffic, not Written Sales
   - Verify `attach_targets()` logic sets WEB conversion to NaN

3. **Feature engineering fails on awareness/CRM**
   - These are optional (Tier 3) and will be skipped if data not available
   - Check `include_crm=False` to disable CRM features
   - Awareness should auto-skip with warning if mapping fails

4. **Memory issues with large datasets**
   - Use smaller horizon ranges (e.g., `horizons=range(1, 14)` instead of 1-52)
   - Disable validation (`validate=False`)
   - Filter to specific stores/DMAs before building

5. **Slow execution**
   - Feature engineering adds ~2-3 minutes for full 52-horizon table
   - Use `include_features=False` for quick iteration on Epic 3 only
   - Consider parallelizing across stores/DMAs (not implemented yet)

---

## Performance Notes

Typical execution times on standard hardware (M1 Mac, 16GB RAM):

| Configuration | Rows | Columns | Time |
|--------------|------|---------|------|
| Epic 3 only (h=1-52) | ~500K | 14 | ~30 seconds |
| Epic 4 Model A (h=1-52) | ~500K | 45 | ~3 minutes |
| Epic 4 Model B (h=1-52) | ~500K | 125 | ~4 minutes |
| Epic 4 Model B + CRM (h=1-52) | ~500K | 150 | ~5 minutes |

Memory usage peaks at ~2-3GB for full 52-horizon table with all features.

---

## See Also

- **`ENGINEERING_INSTRUCTIONS.md`**: Section 4 (Canonical Training Table), Section 5 (Features)
- **`PROJECT_PLAN.md`**: Epic 3 (ETL-01, ETL-02, ETL-03), Epic 4 (FE-01 through FE-08)
- **`Features.md`**: Target variable specifications (lines 77-111)
- **`Model.md`**: Step 2 - Canonical Training Table (lines 12-31)
- **`schema.py`**: Full schema specification and validation functions
- **`../features/`**: Feature engineering modules (dynamics, awareness, CRM, etc.)
- **`../seasonality.py`**: DMA and WEB seasonality computation (Layer 0)
- **`../data_ingestion/`**: Raw data loaders (Epic 1)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Nov 2025 | Initial release - Epic 3 + Epic 4 complete. Added full feature engineering pipeline, model variant system (A/B), 12-step orchestrator, comprehensive documentation. |

---

## Contributing

When modifying this module:

1. **Maintain schema compatibility**: Don't break existing column contracts
2. **Add tests**: Update `test_canonical_table.py` for new logic
3. **Document transformations**: Explain any new data cleaning rules
4. **Preserve leakage prevention**: Never access future data in t0 features
5. **Update this README**: Keep examples and architecture diagrams current

---

**For questions or issues**, refer to:
- Epic 3 completion checklist in `PROJECT_PLAN.md` (lines 42-58)
- Epic 4 feature requirements (lines 60-96)
- Schema validation rules in `schema.py` (lines 166-309)
