# Explainability Model (Model A) Feature List

**Purpose:** Surrogate model for explainability (SHAP analysis of forecasting model predictions)

---

## Prediction Targets

Explainability model predicts **forecasting model outputs** (not actuals) using only business lever features:

| Target | Description |
|--------|-------------|
| `pred_log_sales` | Forecasting model's log sales prediction |
| `pred_log_aov` | Forecasting model's log AOV prediction |
| `pred_log_orders` | Forecasting model's log orders prediction |
| `pred_logit_conversion` | Forecasting model's logit conversion prediction (B&M only) |
| Traffic | Derived via Monte Carlo from Sales, AOV, Conversion |

---

## Included Features (27 total for B&M, 20 for WEB)

### Categorical Features (5)
- `is_outlet` - Outlet store flag
- `is_comp_store` - Comparable store flag
- `is_new_store` - New store flag
- `is_xmas_window` - Christmas window flag
- `is_black_friday_window` - Black Friday window flag

### Store DNA (2)
- `weeks_since_open` - Weeks since store opening
- `merchandising_sf` - Merchandising square footage (B&M only)

### Market Signals (2) - 4-week rolling mean for stability
- `brand_awareness_dma_roll_mean_4` - Brand awareness DMA score (4-week rolling mean)
- `brand_consideration_dma_roll_mean_4` - Brand consideration DMA score (4-week rolling mean)

### Cannibalization (4) - B&M only
- `cannibalization_pressure` - Cannibalization pressure metric
- `min_dist_new_store_km` - Minimum distance to new store (km)
- `num_new_stores_within_10mi_last_52wk` - New stores within 10 miles (last 52 weeks)
- `num_new_stores_within_20mi_last_52wk` - New stores within 20 miles (last 52 weeks)

### Seasonality/Calendar (4) - Prevent feature aliasing
Required to prevent the model from falsely attributing seasonal demand to business levers.
These features cancel out in Differential SHAP (Scenario - Baseline). The goal is to be able to explain the difference in behavior form a given baseline to the target scenario.

- `dma_seasonal_weight` - DMA seasonal weight
- `sin_woy` - Sine transformation of week of year (cyclical encoding)
- `cos_woy` - Cosine transformation of week of year (cyclical encoding)
- `is_holiday` - Holiday week flag

**Note:** Using sin/cos encoding for week-of-year preserves the cyclical nature (week 52 is adjacent to week 1).

### Staffing Levers - B&M ONLY (2)
Key actionable levers affecting conversion. Physical store concept.

- `staffing_unique_associates_roll_mean_4` - 4-week rolling mean of unique associates
- `staffing_hours_roll_mean_4` - 4-week rolling mean of employee hours

### Operational Levers - ACTIONABLE (4)

- `pct_omni_channel_roll_mean_4` - 4-week rolling mean of omnichannel %
- `pct_value_product_roll_mean_4` - 4-week rolling mean of value product %
- `pct_premium_product_roll_mean_4` - 4-week rolling mean of premium product %
- `pct_white_glove_roll_mean_4` - 4-week rolling mean of white glove service %

### Financing Levers - ACTIONABLE (3)

- `pct_primary_financing_roll_mean_4` - 4-week rolling mean of primary financing %
- `pct_secondary_financing_roll_mean_4` - 4-week rolling mean of secondary financing %
- `pct_tertiary_financing_roll_mean_4` - 4-week rolling mean of tertiary financing %

### Other (1)
- `horizon` - Forecast horizon (weeks ahead)

---

## Excluded Features

These columns are explicitly excluded to focus on actionable business levers:

### Metadata/Keys (preserved separately, not as features)
- `channel` - Channel identifier (B&M/WEB)
- `origin_week_date` - Origin week date
- `target_week_date` - Target week date
- `profit_center_nbr` - Store identifier
- `dma` - Designated Market Area

### Target Variables (used as labels, not features)
- `label_log_sales` - Target variable (actual sales)
- `label_log_aov` - Target variable (actual AOV)
- `label_logit_conversion` - Target variable (actual conversion)
- `label_log_orders` - Target variable (actual orders)

### Raw Values / Metadata
- `has_traffic_data` - Metadata flag
- `total_sales` - Raw target
- `order_count` - Derived target
- `store_traffic` - Raw metric

### Store DNA (excluded)
- `store_design_sf` - Store design square footage

### Prediction Outputs (these are what we're trying to explain)
- `pred_log_sales`, `pred_log_aov`, `pred_log_orders`, `pred_logit_conversion`

### Additional Calendar/Seasonality (Model B only)
- `woy` - Raw week of year
- `month`, `quarter`, `fiscal_year`, `fiscal_period`
- `weeks_to_holiday` - Weeks until next holiday

### Sales Autoregressive (excluded to prevent lag dominance)
- `log_sales_lag_1`, `log_sales_lag_4`, `log_sales_lag_13`, `log_sales_lag_52`
- `log_sales_roll_mean_4`, `log_sales_roll_mean_8`, `log_sales_roll_mean_13`
- `vol_log_sales_13` - 13-week sales volatility

### AOV Autoregressive (excluded)
- `AOV_roll_mean_8`, `AOV_roll_mean_13`

### Conversion Autoregressive (excluded)
- `ConversionRate_lag_1`, `ConversionRate_lag_4`
- `ConversionRate_roll_mean_4`, `ConversionRate_roll_mean_8`, `ConversionRate_roll_mean_13`

### Web Traffic Autoregressive (excluded)
- `allocated_web_traffic_lag_1`, `allocated_web_traffic_lag_4`, `allocated_web_traffic_lag_13`
- `allocated_web_traffic_roll_mean_4`, `allocated_web_traffic_roll_mean_8`, `allocated_web_traffic_roll_mean_13`
- `log_web_sales_roll_mean_web_4`

---

## Channel-Specific Features

The explainability model (Model A) uses channel-specific feature sets to ensure only relevant features are used for each channel.

### B&M Channel (27 features)

All features listed in "Included Features" are used for B&M channel explainability:
- Categorical (5): is_outlet, is_comp_store, is_new_store, is_xmas_window, is_black_friday_window
- Store DNA (2): weeks_since_open, merchandising_sf
- Market Signals (2): brand_awareness_dma_roll_mean_4, brand_consideration_dma_roll_mean_4
- Cannibalization (4): cannibalization_pressure, min_dist_new_store_km, num_new_stores_within_10mi_last_52wk, num_new_stores_within_20mi_last_52wk
- Seasonality (4): dma_seasonal_weight, sin_woy, cos_woy, is_holiday
- Staffing (2): staffing_unique_associates_roll_mean_4, staffing_hours_roll_mean_4
- Operational Levers (4): pct_omni_channel_roll_mean_4, pct_value_product_roll_mean_4, pct_premium_product_roll_mean_4, pct_white_glove_roll_mean_4
- Financing Levers (3): pct_primary_financing_roll_mean_4, pct_secondary_financing_roll_mean_4, pct_tertiary_financing_roll_mean_4
- Other (1): horizon

### WEB Channel (20 features)

WEB channel uses all B&M features except the 7 B&M-only features:
- Categorical (5): is_outlet, is_comp_store, is_new_store, is_xmas_window, is_black_friday_window
- Store DNA (1): weeks_since_open
- Market Signals (2): brand_awareness_dma_roll_mean_4, brand_consideration_dma_roll_mean_4
- Seasonality (4): dma_seasonal_weight, sin_woy, cos_woy, is_holiday
- Operational Levers (4): pct_omni_channel_roll_mean_4, pct_value_product_roll_mean_4, pct_premium_product_roll_mean_4, pct_white_glove_roll_mean_4
- Financing Levers (3): pct_primary_financing_roll_mean_4, pct_secondary_financing_roll_mean_4, pct_tertiary_financing_roll_mean_4
- Other (1): horizon

The following 7 B&M-only features are **EXCLUDED** from WEB Model A:

| Feature | Reason for Exclusion |
|---------|---------------------|
| `merchandising_sf` | Physical store square footage - not applicable to digital channel |
| `cannibalization_pressure` | Physical store competition metric |
| `min_dist_new_store_km` | Physical store proximity - not applicable to WEB |
| `num_new_stores_within_10mi_last_52wk` | Physical store competition |
| `num_new_stores_within_20mi_last_52wk` | Physical store competition |
| `staffing_unique_associates_roll_mean_4` | Physical store staffing |
| `staffing_hours_roll_mean_4` | Physical store staffing |

These features measure physical store characteristics and competition, which are irrelevant for explaining WEB channel forecasts.

## Differential SHAP Strategy


1. **The Engine:** Model A includes Seasonality features (dma_seasonal_weight, sin_woy, cos_woy, is_holiday) to fit Model B accurately.
2. **The Report:** We show **Differential SHAP**:
   - Impact = SHAP(Scenario) - SHAP(Baseline)
3. **The Outcome:**
   - Since seasonality features are identical in both Scenario and Baseline, they cancel out.
   - The user sees only the impact of their decisions (e.g., "Increasing Staffing added +$10k").
