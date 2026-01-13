# Sales Forecast Dashboard Data

This directory contains JSON data files generated from SAP HANA predictions used by the Sales Forecast Dashboard API.

## Data Generation Pipeline

### Overview

The JSON files in this directory are automatically generated from a SAP HANA instance through the following workflow:

```
SAP HANA Database
     ↓
  (Data Connection via HANA Loader)
     ↓
regenerate_dashboard_data.py
     ↓
JSON Output Files (stores.json, dma_summary.json, timeseries/)
```

### Prerequisites

1. **HANA Instance Configuration**: Data must be pre-loaded in the SAP HANA database with the following tables:
   - `PREDICTIONS_BM` - B&M channel predictions
   - `PREDICTIONS_WEB` - Web channel predictions
   - `PROFIT_CENTER` - Store master data (locations, attributes)

2. **Environment Variables**: Configure the following in your `.env` file:
   - `HANA_HOST` - HANA database host
   - `HANA_PORT` - HANA database port
   - `HANA_USER` - HANA database user
   - `HANA_PASSWORD` - HANA database password
   - `HANA_SCHEMA` - HANA schema name (default: `AICOE`)

### Data Regeneration

To regenerate the JSON files from HANA, run the data regeneration script:

```bash
# From the project root
python -m use_cases.sales_forecast_dashboard.api.app.scripts.regenerate_dashboard_data

# With options
python -m use_cases.sales_forecast_dashboard.api.app.scripts.regenerate_dashboard_data --verbose
python -m use_cases.sales_forecast_dashboard.api.app.scripts.regenerate_dashboard_data --skip-timeseries  # Faster, excludes timeseries files
python -m use_cases.sales_forecast_dashboard.api.app.scripts.regenerate_dashboard_data --dry-run  # Test connection only
```

## Data Files

### 1. stores.json

Contains aggregated store-level data with predictions and year-over-year metrics.

**Key Fields:**
- `id` - Store ID (profit center number)
- `name` - Store name
- `dma` - Designated Market Area
- `lat`, `lng` - Geographic coordinates
- `city`, `state` - Location details
- `bm_auv_p50`, `bm_auv_p90` - B&M channel Annualized Unit Volume (52-week forecast sum)
- `bm_yoy_auv_change` - B&M year-over-year AUV change percentage
- `web_auv_p50`, `web_auv_p90` - Web channel AUV
- `web_yoy_auv_change` - Web channel YoY change percentage
- `auv_p50`, `auv_p90` - Combined (B&M + Web) AUV
- `yoy_auv_change` - Combined YoY change percentage
- Store attributes: `is_outlet`, `is_new_store`, `is_comp_store`, `store_design_sf`, `merchandising_sf`, `bcg_category`

**Example:**
```json
{
  "id": 1001,
  "name": "Downtown Store",
  "dma": "New York",
  "lat": 40.7128,
  "lng": -74.0060,
  "city": "New York",
  "state": "NY",
  "bm_auv_p50": 2450000.00,
  "bm_auv_p90": 2650000.00,
  "bm_yoy_auv_change": 8.5,
  "web_auv_p50": 850000.00,
  "web_auv_p90": 920000.00,
  "web_yoy_auv_change": 12.3,
  "auv_p50": 3300000.00,
  "auv_p90": 3570000.00,
  "yoy_auv_change": 9.2,
  "is_comp_store": true
}
```

### 2. dma_summary.json

Contains aggregated market-level data across all stores within each DMA.

**Key Fields:**
- `dma` - Market name
- `lat`, `lng` - Market centroid coordinates
- `store_count` - Number of stores in the DMA
- `bm_total_auv_p50`, `bm_total_auv_p90` - B&M total AUV across all stores
- `bm_avg_auv_p50` - B&M average AUV per store
- `bm_yoy_auv_change_pct` - B&M weighted average YoY change
- `bm_yoy_status` - Traffic light status ("increase", "stable", "decrease")
- `web_total_auv_p50`, `web_total_auv_p90` - Web channel totals
- `web_avg_auv_p50` - Web average per store
- `web_yoy_auv_change_pct` - Web weighted average YoY change
- `web_yoy_status` - Web traffic light status
- Similar combined fields for B&M + Web aggregate

**Example:**
```json
{
  "dma": "New York",
  "lat": 40.7128,
  "lng": -74.0060,
  "store_count": 25,
  "bm_total_auv_p50": 65200000.00,
  "bm_total_auv_p90": 71500000.00,
  "bm_avg_auv_p50": 2608000.00,
  "bm_yoy_auv_change_pct": 7.8,
  "bm_yoy_status": "increase",
  "web_total_auv_p50": 18500000.00,
  "web_avg_auv_p50": 740000.00,
  "web_yoy_auv_change_pct": 11.2,
  "web_yoy_status": "increase"
}
```

### 3. timeseries/

Individual time series files for each store and DMA.

#### 3a. store_{id}_{channel}.json

Contains weekly-level predictions with SHAP feature importance and explanations for individual stores.

**File Pattern:** `store_1001_bm.json`, `store_1001_web.json`, etc.

**Key Fields:**
- `store_id` - Store ID
- `channel` - "B&M" or "WEB"
- `timeseries` - Array of weekly data points:
  - `date` - Target week date (ISO 8601 format)
  - `pred_sales_p50`, `pred_sales_p90` - Weekly sales forecast (probabilistic percentiles)
  - `baseline_sales_p50` - 2024 sales for the same week (year-over-year baseline)
  - `yoy_change_pct` - Year-over-year change percentage
  - `pred_aov_mean`, `pred_aov_p50`, `pred_aov_p90` - Average Order Value forecasts
  - `pred_traffic_p10/p50/p90` - Traffic (visitor count) forecasts (B&M only)
  - `shap_features` - Array of top influential features with their impact on sales:
    - `feature` - Feature name
    - `value` - Current feature value
    - `baseline_value` - 2024 feature value for comparison
    - `impact` - SHAP delta value (change in log-sales contribution)
    - `has_baseline` - Whether 2024 data was available for comparison
  - `explanation` - Natural language description of key drivers

**Example:**
See `store_1001_bm.json` for a complete example with realistic data structure.

#### 3b. dma_{name}.json

Aggregated weekly sales by DMA (sum of all stores in the market).

**File Pattern:** `dma_new_york.json`, `dma_los_angeles.json`, etc.
(Filenames use URL-safe substitutions: `/` → `_`, spaces → `_`)

**Key Fields:**
- `dma` - Market name
- `timeseries` - Array of weekly aggregates:
  - `date` - Target week date
  - `pred_sales_p50`, `pred_sales_p90` - Total market sales forecast

**Example:**
```json
{
  "dma": "New York",
  "timeseries": [
    {
      "date": "2025-01-06",
      "pred_sales_p50": 1245000.50,
      "pred_sales_p90": 1450000.00
    }
  ]
}
```

## Data Processing Details

### AUV (Annualized Unit Volume)

AUV represents a 52-week forecast sum of weekly sales predictions. This metric allows comparison of store performance over a full year horizon.

**Calculation:**
```
AUV_P50 = SUM(weekly pred_sales_p50) across 52 weeks
AUV_P90 = SUM(weekly pred_sales_p90) across 52 weeks
```

### Year-Over-Year (YoY) Changes

YoY metrics compare current year (2025) forecasts against 2024 actuals for the same calendar week.

**Calculation:**
```
YoY% = ((AUV_2025 - AUV_2024) / AUV_2024) × 100
```

### YoY Status ("Traffic Light" Colors)

Status is determined by worst-case logic across all stores in a market:
- **"increase" (Green)**: All stores have YoY > 5%
- **"stable" (Yellow)**: Any store has -5% ≤ YoY ≤ 5%
- **"decrease" (Red)**: Any store has YoY < -5%

### SHAP Features (Feature Importance)

SHAP values represent each feature's contribution to the sales forecast. The delta between 2025 and 2024 SHAP values shows how feature impacts have changed year-over-year.

**Matching Logic:**
- Features are matched by store, channel, and ISO week number (not horizon)
- This enables YoY comparisons even when forecast horizons differ between years

## File Updates

Files are not automatically updated in this repository. To regenerate:

1. Ensure HANA data is current and pre-loaded with latest predictions
2. Run the regeneration script from the project root
3. Commit the regenerated JSON files to version control

## Development Notes

- The `timeseries/` directory is tracked in git to preserve the data structure
- Mock example files are included for reference and testing
- The regeneration script can be run with `--dry-run` to test HANA connectivity
- Use `--skip-timeseries` to quickly regenerate aggregate files (stores.json, dma_summary.json) without the detailed timeseries data
- For performance, the script processes data in batches with memory optimization
