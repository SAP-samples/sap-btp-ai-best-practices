# Model B Feature List

**Purpose:** Production forecasting model

**Channel-Specific Models:** B&M and WEB are trained separately with different feature sets.

---

## Prediction Targets

Model B predicts **actual values**:

| Target | Description |
|--------|-------------|
| `label_log_sales` | Actual log sales |
| `label_log_aov` | Actual log AOV |
| `label_log_orders` | Actual log orders |
| `label_logit_conversion` | Actual logit conversion (B&M only) |
| Traffic | Derived via Monte Carlo from Sales, AOV, Conversion |

---

## Feature Architecture

Model B = Model A Features (shared) + Additional Autoregressive & Calendar Features

### Shared Features (from Model A)

Model A features are fully included in Model B. These are the actionable business lever features used for explainability.

#### Categorical Features (5)
- `is_outlet` - Outlet store flag
- `is_comp_store` - Comparable store flag
- `is_new_store` - New store flag
- `is_xmas_window` - Christmas window flag
- `is_black_friday_window` - Black Friday window flag

#### Store DNA (2)
- `weeks_since_open` - Weeks since store opening
- `merchandising_sf` - Merchandising square footage (B&M only)

#### Market Signals (2)
- `brand_awareness_dma_roll_mean_4` - Brand awareness DMA score (4-week rolling mean)
- `brand_consideration_dma_roll_mean_4` - Brand consideration DMA score (4-week rolling mean)

#### Cannibalization (4) - B&M only
- `cannibalization_pressure` - Cannibalization pressure metric
- `min_dist_new_store_km` - Minimum distance to new store (km)
- `num_new_stores_within_10mi_last_52wk` - New stores within 10 miles (last 52 weeks)
- `num_new_stores_within_20mi_last_52wk` - New stores within 20 miles (last 52 weeks)

#### Seasonality/Calendar (4)
- `dma_seasonal_weight` - DMA seasonal weight
- `sin_woy` - Sine transformation of week of year
- `cos_woy` - Cosine transformation of week of year
- `is_holiday` - Holiday week flag

#### Staffing Levers (2) - B&M only
- `staffing_unique_associates_roll_mean_4` - 4-week rolling mean of unique associates
- `staffing_hours_roll_mean_4` - 4-week rolling mean of employee hours

#### Operational Levers (4)
- `pct_omni_channel_roll_mean_4` - 4-week rolling mean of omnichannel %
- `pct_value_product_roll_mean_4` - 4-week rolling mean of value product %
- `pct_premium_product_roll_mean_4` - 4-week rolling mean of premium product %
- `pct_white_glove_roll_mean_4` - 4-week rolling mean of white glove service %

#### Financing Levers (3)
- `pct_primary_financing_roll_mean_4` - 4-week rolling mean of primary financing %
- `pct_secondary_financing_roll_mean_4` - 4-week rolling mean of secondary financing %
- `pct_tertiary_financing_roll_mean_4` - 4-week rolling mean of tertiary financing %

#### Other (1)
- `horizon` - Forecast horizon (weeks ahead)

**Total Shared Features: 27 for B&M, 20 for WEB** (WEB excludes merchandising_sf, cannibalization, staffing)

---

## B&M Channel Features

### Model A Features (Shared - 27)

The following 27 features from Model A are included in B&M Model B:

| Category | Count | Features |
|----------|-------|----------|
| Categorical | 5 | is_outlet, is_comp_store, is_new_store, is_xmas_window, is_black_friday_window |
| Store DNA | 2 | weeks_since_open, merchandising_sf |
| Market Signals | 2 | brand_awareness_dma_roll_mean_4, brand_consideration_dma_roll_mean_4 |
| Cannibalization | 4 | cannibalization_pressure, min_dist_new_store_km, num_new_stores_within_10mi_last_52wk, num_new_stores_within_20mi_last_52wk |
| Seasonality | 4 | dma_seasonal_weight, sin_woy, cos_woy, is_holiday |
| Staffing | 2 | staffing_unique_associates_roll_mean_4, staffing_hours_roll_mean_4 |
| Operational | 4 | pct_omni_channel_roll_mean_4, pct_value_product_roll_mean_4, pct_premium_product_roll_mean_4, pct_white_glove_roll_mean_4 |
| Financing | 3 | pct_primary_financing_roll_mean_4, pct_secondary_financing_roll_mean_4, pct_tertiary_financing_roll_mean_4 |
| Other | 1 | horizon |

### Categorical Features (handled natively by CatBoost)
- `profit_center_nbr` - Store identifier (categorical)
- `dma` - Designated Market Area

Note: `is_pre_holiday_*` features removed to prevent spurious interactions with other features.

### Additional Calendar/Seasonality (Model B only - 7)
Extended calendar features beyond Model A's core seasonality:
- `sin_woy` - Sine transformation of week of year
- `cos_woy` - Cosine transformation of week of year
- `month` - Month (1-12)
- `quarter` - Quarter (1-4)
- `fiscal_year` - Fiscal year
- `fiscal_period` - Fiscal period
- `weeks_to_holiday` - Weeks until next holiday

Note: Core seasonality (dma_seasonal_weight, woy, is_holiday) now in Model A (shared).

### Additional Store DNA (Model B only - 2)
- `sq_ft` - Total square footage
- `store_design_sf` - Store design square footage

### CRM Demographic Features (Tier 3 Cross-Sectional - Optional 12)

12 validated CRM features based on driver screening analysis (pooled |rho| >= 0.15). These are cross-sectional predictors providing baseline lift - strong pooled correlations but weak FE correlations (<0.10), indicating they capture store-level characteristics rather than time-varying dynamics.

**Note:** Only included when `--include-crm` flag is used during data generation.

| Feature | B&M AOV | B&M Sales | WEB AOV | WEB Sales |
|---------|---------|-----------|---------|-----------|
| `crm_dwelling_single_family_dwelling_unit_pct` | +0.19 | -0.10 | +0.21 | -0.08 |
| `crm_dwelling_multi_family_dwelling_unit_pct` | -0.15 | +0.15 | -0.20 | +0.13 |
| `crm_owner_renter_owner_pct` | +0.18 | -0.12 | +0.21 | -0.09 |
| `crm_income_150k_plus_pct` | +0.13 | +0.02 | +0.19 | +0.20 |
| `crm_income_under_50k_pct` | -0.18 | +0.04 | -0.18 | -0.10 |
| `crm_marital_married_pct` | +0.17 | -0.05 | +0.16 | +0.04 |
| `crm_age_25_34_pct` | -0.09 | -0.19 | -0.05 | -0.03 |
| `crm_age_55_64_pct` | +0.06 | +0.08 | +0.01 | -0.14 |
| `crm_age_65_80_pct` | -0.03 | -0.18 | +0.04 | -0.21 |
| `crm_education_college_pct` | -0.01 | -0.02 | -0.01 | +0.19 |
| `crm_education_high_school_pct` | -0.01 | -0.03 | +0.02 | -0.18 |
| `crm_children_y_pct` | +0.12 | -0.05 | -0.01 | -0.04 |

### Sales Autoregressive (B&M - 8)
- `log_sales_lag_1` - 1-week lag of log sales
- `log_sales_lag_4` - 4-week lag of log sales
- `log_sales_lag_13` - 13-week lag of log sales
- `log_sales_lag_52` - 52-week (YoY) lag of log sales
- `log_sales_roll_mean_4` - 4-week rolling mean of log sales
- `log_sales_roll_mean_8` - 8-week rolling mean of log sales
- `log_sales_roll_mean_13` - 13-week rolling mean of log sales
- `vol_log_sales_13` - 13-week sales volatility

### AOV Autoregressive (B&M - 2)
- `AOV_roll_mean_8` - 8-week rolling mean of AOV
- `AOV_roll_mean_13` - 13-week rolling mean of AOV

### Conversion Autoregressive (B&M-ONLY - 5)
- `ConversionRate_lag_1` - 1-week lag of conversion rate
- `ConversionRate_lag_4` - 4-week lag of conversion rate
- `ConversionRate_roll_mean_4` - 4-week rolling mean of conversion rate
- `ConversionRate_roll_mean_8` - 8-week rolling mean of conversion rate
- `ConversionRate_roll_mean_13` - 13-week rolling mean of conversion rate

---

## WEB Channel Features

### Model A Features (Shared - 20)

The following 20 features from Model A are included in WEB Model B (excludes B&M-only features):

| Category | Count | Features |
|----------|-------|----------|
| Categorical | 5 | is_outlet, is_comp_store, is_new_store, is_xmas_window, is_black_friday_window |
| Store DNA | 1 | weeks_since_open |
| Market Signals | 2 | brand_awareness_dma_roll_mean_4, brand_consideration_dma_roll_mean_4 |
| Seasonality | 4 | dma_seasonal_weight, sin_woy, cos_woy, is_holiday |
| Operational | 4 | pct_omni_channel_roll_mean_4, pct_value_product_roll_mean_4, pct_premium_product_roll_mean_4, pct_white_glove_roll_mean_4 |
| Financing | 3 | pct_primary_financing_roll_mean_4, pct_secondary_financing_roll_mean_4, pct_tertiary_financing_roll_mean_4 |
| Other | 1 | horizon |

**Excluded from WEB:** merchandising_sf, cannibalization (4 features), staffing (2 features)

### Categorical Features (handled natively by CatBoost)
- `profit_center_nbr` - Store identifier (categorical)
- `dma` - Designated Market Area

### Additional Calendar/Seasonality (Model B only - 7)
- `sin_woy` - Sine transformation of week of year
- `cos_woy` - Cosine transformation of week of year
- `month` - Month (1-12)
- `quarter` - Quarter (1-4)
- `fiscal_year` - Fiscal year
- `fiscal_period` - Fiscal period
- `weeks_to_holiday` - Weeks until next holiday

### Additional Store DNA (Model B only - 2)
- `sq_ft` - Total square footage
- `store_design_sf` - Store design square footage

### Market Signals (additional WEB-specific)
- `dma_web_penetration_pct` - DMA-level WEB sales penetration

### Sales Autoregressive (WEB - 8)
- `log_sales_lag_1` - 1-week lag of log sales
- `log_sales_lag_4` - 4-week lag of log sales
- `log_sales_lag_13` - 13-week lag of log sales
- `log_sales_lag_52` - 52-week (YoY) lag of log sales
- `log_sales_roll_mean_4` - 4-week rolling mean of log sales
- `log_sales_roll_mean_8` - 8-week rolling mean of log sales
- `log_sales_roll_mean_13` - 13-week rolling mean of log sales
- `vol_log_sales_13` - 13-week sales volatility

### AOV Autoregressive (WEB - 2)
- `AOV_roll_mean_8` - 8-week rolling mean of AOV
- `AOV_roll_mean_13` - 13-week rolling mean of AOV

### Web Traffic (WEB-ONLY - 6)
- `allocated_web_traffic_lag_1` - 1-week lag of allocated web traffic
- `allocated_web_traffic_lag_4` - 4-week lag of allocated web traffic
- `allocated_web_traffic_lag_13` - 13-week lag of allocated web traffic
- `allocated_web_traffic_roll_mean_4` - 4-week rolling mean (OPTIMAL for WEB)
- `allocated_web_traffic_roll_mean_8` - 8-week rolling mean
- `allocated_web_traffic_roll_mean_13` - 13-week rolling mean

### WEB Sales Autoregressive (WEB-ONLY)
- `log_web_sales_lag_1` - 1-week lag of WEB log sales
- `log_web_sales_lag_4` - 4-week lag of WEB log sales
- `log_web_sales_lag_13` - 13-week lag of WEB log sales
- `log_web_sales_roll_mean_4` - 4-week rolling mean
- `log_web_sales_roll_mean_8` - 8-week rolling mean
- `log_web_sales_roll_mean_13` - 13-week rolling mean
- `vol_log_web_sales_13` - 13-week volatility
- `log_web_sales_roll_mean_web_4` - 4-week rolling mean (legacy)

### WEB AOV (WEB-ONLY)
- `web_aov_roll_mean_4` - 4-week rolling mean of WEB AOV
- `web_aov_roll_mean_8` - 8-week rolling mean of WEB AOV

---

## Pruned Features (Removed from Pipeline)

The following business lever variants have been pruned to reduce collinearity. They are **no longer generated** by the feature engineering pipeline:

### Omnichannel (removed)
- `pct_omni_channel_lag_1`
- `pct_omni_channel_lag_4`

### Product Mix (removed)
- `pct_value_product_roll_mean_8`

### Financing (removed)
- `pct_primary_financing_lag_1`, `pct_primary_financing_lag_4`, `pct_primary_financing_roll_mean_8`
- `pct_secondary_financing_lag_1`, `pct_secondary_financing_lag_4`, `pct_secondary_financing_roll_mean_8`
- `pct_tertiary_financing_lag_1`, `pct_tertiary_financing_lag_4`, `pct_tertiary_financing_roll_mean_8`

---

## Channel-Specific Feature Exclusions

### B&M Model Excludes (WEB-ONLY Features):
- All `allocated_web_traffic_*` features (web sessions, not foot traffic)
- All `log_web_sales_*` features (WEB channel specific)
- All `web_aov_roll_mean_*` features (WEB channel specific)
- `dma_web_penetration_pct` (WEB market signal)

### WEB Model Excludes (B&M-ONLY Features):
- All `ConversionRate_*` features (requires foot traffic, NaN for WEB)
- `cannibalization_pressure` (physical store competition)
- `min_dist_new_store_km` (physical store proximity)
- `num_new_stores_within_10mi_last_52wk` (physical store competition)
- `num_new_stores_within_20mi_last_52wk` (physical store competition)
- `merchandising_sf` (physical space irrelevant for digital channel)
- `staffing_unique_associates_roll_mean_4` (physical store staffing)
- `staffing_hours_roll_mean_4` (physical store staffing)

---

## Excluded Columns (Both Channels)

These columns are explicitly excluded from training:

### Metadata/Keys
- `channel` - Channel identifier (B&M/WEB)
- `origin_week_date` - Origin week date
- `target_week_date` - Target week date

### Target Variables
- `label_log_sales` - Target variable (actual sales)
- `label_log_aov` - Target variable (actual AOV)
- `label_logit_conversion` - Target variable (actual conversion)
- `label_log_orders` - Target variable (actual orders)

### Raw Values / Metadata
- `has_traffic_data` - Metadata flag
- `total_sales` - Raw target (use log version)
- `order_count` - Derived target
- `store_traffic` - Raw metric

### Prediction Outputs (if present)
- `predicted_log_sales` - Model prediction output
- `predicted_log_aov` - Model prediction output
- `predicted_logit_conversion` - Model prediction output
- `predicted_log_orders` - Model prediction output

---

## Data Source Hierarchy

For WEB channel features:
1. **Sales/AOV**: Written Sales Data (WEB channel) - SOURCE OF TRUTH
2. **Web Traffic**: Ecomm Traffic (AllocatedWebTraffic) - validated predictive (rho=0.749 log-transformed)
3. **EXCLUDED**: UnallocatedWebTraffic - measures web sessions, not comparable to Store_Traffic

---

## Time Alignment

Model B uses both origin-aligned and target-aligned features:

| Feature Category | Time Alignment | Rationale |
|-----------------|----------------|-----------|
| **Autoregressive** (Sales/AOV/Conversion Lags) | Origin | Historical actuals only known at t0 |
| **Business Levers** (Staffing, Financing, etc.) | Target | Enable What-If scenarios |
| **Market Signals** (Awareness, Cannibalization) | Target | State at forecast target |
| **Seasonality/Calendar** | Target | Future holidays/seasons known |
| **Store DNA** | Static | Fundamental store properties |

This dual alignment allows:
- Autoregressive features to capture momentum from historical data
- Business levers to be adjusted for scenario analysis at target dates

---

## Feature Engineering Notes

### WEB-Specific Optimizations
- B&M optimal window: 13-week rolling means (seasonality FE rho=+0.42)
- WEB optimal window: 4-week rolling means (seasonality FE rho=+0.21, faster dynamics)
- WEB sales features computed from Written Sales (12.6% median difference vs Ecomm MerchAmt)
- AllocatedWebTraffic correlation with WEB sales: 0.446 (log-transformed: 0.749)
