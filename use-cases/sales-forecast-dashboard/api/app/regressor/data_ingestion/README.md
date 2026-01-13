# Data Ingestion Module

## Overview

The `data_ingestion` module provides the foundational data loading layer for Company X Sales Forecasting System. It establishes a clean, validated interface between raw data sources and downstream feature engineering pipelines.

This module implements **Epic 1: Data Ingestion & Cleaning** from the project plan, ensuring all raw inputs are properly cleaned, typed, and augmented with necessary flags and metadata before entering the canonical training table.

## Purpose in Model Architecture

The forecasting system maintains two model variants for different purposes:

- **Model A**: Business-actionable levers + known-in-advance/static context (for SHAP explainability)
- **Model B**: Full feature set including autoregressive lags/rolls (for production forecasts)

Data ingestion serves both models by providing:
1. **Clean, validated raw data** with standardized schemas
2. **Enrichment logic** (DMA mappings, traffic flags, demographic pivots)
3. **Quality gates** to handle missing values, NULL strings, and data type coercion

## Module Files

| File | Purpose | Key Functions | Data Source |
|------|---------|---------------|-------------|
| **`__init__.py`** | Module exports and public API | Exports all loader functions for downstream import | N/A |
| **`written_sales.py`** | Load historical sales data with traffic quality flags | `load_written_sales_with_flags()`<br>`compute_traffic_missingness_flag()` | `Written Sales Data.csv` |
| **`ecomm_traffic.py`** | Load e-commerce traffic allocation data | `load_ecomm_traffic()` | `Ecomm Traffic.csv` |
| **`awareness.py`** | Load brand awareness/consideration with DMA mapping | `load_awareness_with_mapping()`<br>`load_awareness_data()` (alias) | `Awareness_Consideration_2022-2025.xlsx`<br>`YOUGOV_DMA_MAP` (master tables) |
| **`crm_mix.py`** | Load CRM demographics and customer mix features | `load_demographics_with_typing()`<br>`load_crm_mix()` (alias) | `Demographics (CRM + Real Estate).csv` |
| **`store_master.py`** | Load store metadata and geographic mappings | `load_store_master()`<br>`load_market_region_map()`<br>`load_yougov_dma_map()` | `BDF Data Model Master Tables.xlsx` |

## Data Sources

All raw data files are loaded from `DATA_DIR` (configured in `forecasting.regressor.paths`):

### Primary Data Files

| File | Description | Key Columns | Granularity |
|------|-------------|-------------|-------------|
| **Written Sales Data.csv** | Historical sales actuals with operational metrics | `profit_center_nbr`, `channel`, `fiscal_start_date_week`, `total_sales`, `store_traffic`, `order_qty`, `order_count`, `aur`, staffing metrics, delivery mix, product mix, financing mix | Store × Channel × Week |
| **Ecomm Traffic.csv** | Web traffic allocation by store DMA | `profit_center_nbr`, `market_city`, `fiscal_start_date_week`, `merch_amt`, `perc_sales`, `unallocated_web_traffic`, `allocated_web_traffic` | Store × Week |
| **Awareness_Consideration_2022-2025.xlsx** | Weekly brand awareness/consideration by market | `market`, `week_start`, `awareness`, `consideration` | 39 Markets × 190 Weeks (7,404 rows) |
| **Demographics (CRM + Real Estate).csv** | Static real estate + time-varying customer mix | Static: population, income, households, drive times<br>CRM: age/income/dwelling/education mix percentages by week | Store (static) + Store × Channel × Week (CRM) |
| **BDF Data Model Master Tables.xlsx** | Master reference tables | **PROFIT_CENTER**: Store metadata, lat/lon, open/close dates, square footage, proforma sales<br>**MARKET**: Market to Region mapping<br>**YOUGOV_DMA_MAP**: DMA to awareness market cascade | Store (PROFIT_CENTER)<br>Market (MARKET)<br>DMA (YOUGOV_DMA_MAP) |

## Key Functions

### Written Sales

```python
from forecasting.regressor.data_ingestion import load_written_sales_with_flags

# Load sales data with traffic quality flag
sales_df = load_written_sales_with_flags(
    null_threshold=0.2,  # Max 20% missing traffic allowed
    min_weeks=8          # Min 8 non-null traffic weeks required
)

# Returns: profit_center_nbr, channel, fiscal_year, fiscal_month, fiscal_week,
#          fiscal_start_date_week, total_sales, store_traffic, order_qty, order_count,
#          aur, employee_hours, unique_associates, avg_tenure_days, delivery metrics,
#          product mix, financing mix, has_traffic_data (flag)
```

**Traffic Missingness Flag Logic** (for Conversion Model exclusion):
- Compute per-store missing rate: `(# NULL traffic weeks) / (total weeks)`
- Flag `has_traffic_data = 1` if:
  - Missing rate ≤ `null_threshold` (default 20%)
  - AND non-null weeks ≥ `min_weeks` (default 8)
- Otherwise flag = 0 (exclude from Conversion training)

### E-commerce Traffic

```python
from forecasting.regressor.data_ingestion import load_ecomm_traffic

traffic_df = load_ecomm_traffic()

# Returns: profit_center_nbr, market_city, fiscal_start_date_week,
#          merch_amt, perc_sales, unallocated_web_traffic, allocated_web_traffic
```

**Note**: Use this data for Web-specific dynamics (Epic 4 [FE-03]), NOT the traffic column from Written Sales.

### Brand Awareness/Consideration

```python
from forecasting.regressor.data_ingestion import load_awareness_with_mapping

awareness_df = load_awareness_with_mapping()

# Returns: market, week_start, awareness, consideration,
#          yougov_dma, region, market_city (from DMA mapping)
```

**DMA Cascade Mechanism**:
- Awareness data = 39 aggregate market groupings (e.g., "Boston", "Midwest Single DMAs")
- YOUGOV_DMA_MAP provides: YouGov DMA → Market → Market City
- Coverage: 96.7% of operational stores (325 of 336 stores)
- Missing stores: Use "Single DMAs" groupings or sister-DMA fallback

**Feature Importance** (from driver screening):
- Web Sales: Pooled ρ = +0.31 (top-15 driver, Tier 1)
- B&M Sales: Pooled ρ = +0.18 (strong cross-sectional, Tier 2)
- B&M Conversion: Pooled ρ = +0.21 (moderate lift)

### CRM Demographics & Customer Mix

```python
from forecasting.regressor.data_ingestion import load_demographics_with_typing

static_df, crm_df = load_demographics_with_typing()

# static_df (one row per store):
#   Keys: profit_center_nbr
#   Columns: population_20min, population_30min, median_income_20min,
#            total_households, drive_time_70pct, internal_stores_nearby, etc.

# crm_df (time-varying customer mix):
#   Keys: profit_center_nbr, channel_norm, week_start
#   Columns: crm_age_25_34_pct, crm_income_150k_plus_pct,
#            crm_owner_renter_owner_pct, crm_children_y_pct,
#            crm_dwelling_single_family_dwelling_unit_pct, etc.
```

**Feature Design**:
- **Static Real Estate** (CV=0): Population, income, households - constant over time
- **Time-Varying CRM** (CV>0.6): Customer demographics - varies weekly
- **Percentage Normalization**: Categories are converted to percentages summing to 100%
- **Category Name Cleaning**: Standardized column suffixes (e.g., "25 to 34" → "25_34")

**Feature Importance** (Tier 3 cross-sectional):
- Sales/AOV: Pooled ρ = 0.15-0.21
- Even with weak FE correlations, provides cross-sectional lift in pooled models

### Store Master & Mappings

```python
from forecasting.regressor.data_ingestion import (
    load_store_master,
    load_market_region_map,
    load_yougov_dma_map
)

stores = load_store_master()
# Returns: profit_center_nbr, market, market_city, ga_dma, latitude, longitude,
#          date_opened, date_closed, location_type, dc_location,
#          merchandising_sf, store_design_sf, is_outlet, proforma_annual_sales

market_map = load_market_region_map()
# Returns: market, region

dma_map = load_yougov_dma_map()
# Returns: yougov_dma, region, market, market_city
```

## Data Flow

```
Raw Data Sources (DATA_DIR)
    │
    ├─> Written Sales Data.csv
    │       └─> load_written_sales_with_flags()
    │               └─> Compute traffic missingness flag
    │                       └─> [sales_df with has_traffic_data]
    │
    ├─> Ecomm Traffic.csv
    │       └─> load_ecomm_traffic()
    │               └─> [traffic_df for Web features]
    │
    ├─> Awareness_Consideration_2022-2025.xlsx
    │       └─> load_awareness_with_mapping()
    │               └─> Join YOUGOV_DMA_MAP
    │                       └─> [awareness_df with DMA cascade]
    │
    ├─> Demographics (CRM + Real Estate).csv
    │       └─> load_demographics_with_typing()
    │               └─> Split static vs. time-varying
    │                   └─> Parse categorical → percentage columns
    │                       └─> [static_df, crm_df]
    │
    └─> BDF Data Model Master Tables.xlsx
            └─> load_store_master()
            └─> load_market_region_map()
            └─> load_yougov_dma_map()
                    └─> [store metadata, mappings]

                            ↓
                    DOWNSTREAM USAGE
                            ↓
        Feature Engineering (Epic 4)
                            ↓
        Canonical Training Table (Epic 3)
                            ↓
            Model A (Explainability) + Model B (Production)
```

## Architecture Notes

### Wrapper Pattern

All ingestion modules follow a consistent wrapper pattern:
- **Core loaders** in `forecasting.regressor.io_utils` handle raw file I/O, type coercion, and validation
- **Ingestion wrappers** in this module add domain-specific enrichment (flags, mappings, pivots)
- This separation allows testing of raw loaders independently from business logic

### Backward Compatibility

Several functions maintain legacy aliases for existing feature pipelines:
- `load_awareness_data()` → `load_awareness_with_mapping()`
- `load_crm_mix()` → `load_demographics_with_typing()[1]` (returns only CRM DataFrame)

### Data Quality Handling

**NULL String Cleaning**: `store_traffic` in Written Sales contains literal string "NULL" values
- Converted to `pd.NA` before numeric coercion
- Prevents silent `0` conversions that would corrupt traffic data

**Percentage Validation**: All `pct_*` columns in Written Sales are clipped to [0, 100]
- Negative values logged and set to 0
- Ensures valid percentages for delivery/product/financing mix features

**Date Parsing**: All date columns use `errors='coerce'` to handle malformed dates gracefully
- Invalid dates become `NaT` rather than failing the entire load

**Type Safety**: Integer columns use pandas `Int64` (nullable) rather than `int64`
- Allows missing values without forcing float conversion
- Preserves data semantics for store IDs and fiscal periods

## Validation & Coverage

### Awareness Data Coverage (Epic 1 [DATA-03])

From `AWARENESS_COVERAGE_REPORT.md`:
- **Total Active Stores**: 356
- **Stores WITH Awareness**: 325 (91.3%)
- **Operational Stores** (excl. CORPORATE): 336
- **Operational WITH Awareness**: 325 (96.7%)

**Status**: VALIDATED - READY FOR EPIC 4

**Unmapped Stores** (7 total):
- CHAMPAIGN/SPRINGFIELD (2 stores) → Map to "Midwest Single DMAs"
- SPARTANBURG/ASHEVILLE (2 stores) → Map to "Southeast Single DMAs"
- DAVENPORT (1 store) → Map to "Midwest Single DMAs"
- PEORIA (1 store) → Map to "Midwest Single DMAs"
- Plus 1 additional store

**Fallback Strategy**: Sister-DMA or regional average for unmapped stores

### Traffic Quality (Epic 1 [DATA-02])

Traffic missingness flag enables:
- Excluding low-quality stores from Conversion Model training
- Preventing bias from systematic traffic data gaps
- Maintaining data quality standards across store cohorts

**Default Thresholds**:
- `null_threshold = 0.2` (max 20% missing weeks)
- `min_weeks = 8` (min 8 non-null traffic observations)

## Integration Points

### Epic 3: Canonical Training Table

All ingestion functions output DataFrames that feed into the "History Exploder" ETL:
1. **Written Sales** provides targets (Sales, AOV, Orders, Conversion) and operational metrics
2. **Ecomm Traffic** provides Web-specific traffic features
3. **Awareness** provides DMA-level brand strength signals
4. **CRM Demographics** provides static real estate + time-varying customer mix
5. **Store Master** provides store DNA (lat/lon, open dates, proforma, outlet flag)

### Epic 4: Feature Engineering

Data ingestion outputs are transformed into features:
- **[FE-02]** Sales & AOV Dynamics: Uses Written Sales actuals
- **[FE-03]** Web Dynamics: Uses Ecomm Traffic data
- **[FE-04]** Conversion Features: Uses Written Sales with traffic flag filter
- **[FE-05]** Static Store DNA: Uses Store Master metadata
- **[FE-07]** Awareness Features: Uses Awareness with DMA mapping
- **[FE-08]** CRM Mix: Uses demographics static + time-varying

## Usage Example

```python
from forecasting.regressor.data_ingestion import (
    load_written_sales_with_flags,
    load_ecomm_traffic,
    load_awareness_with_mapping,
    load_demographics_with_typing,
    load_store_master,
)

# Load all data sources for canonical table generation
sales = load_written_sales_with_flags()
traffic = load_ecomm_traffic()
awareness = load_awareness_with_mapping()
static_demo, crm_mix = load_demographics_with_typing()
stores = load_store_master()

# Join to create feature views
# ... (ETL logic in Epic 3)

# Feed to feature engineering
# ... (transforms in Epic 4)

# Train Model A (actionable levers) + Model B (full features)
# ... (training in Epic 5)
```

## Related Documentation

- **Project Plan**: `api/app/regressor/PROJECT_PLAN.md`
- **Awareness Coverage Report**: `api/app/regressor/data_ingestion/AWARENESS_COVERAGE_REPORT.md`
- **Core I/O Functions**: `api/app/regressor/io_utils.py`

---

**Module Owner**: Data Engineering Team
**Epic Status**: Epic 1 Complete
**Last Updated**: 2024-11-28
