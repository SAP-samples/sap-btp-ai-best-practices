# Engineering Project Plan: Company X Sales Forecasting Model (Phase 1)


**Interpretability guardrail:** A single full-feature model lets autoregressive lags overshadow operational levers (e.g., a week-1 omnichannel push later shows up in `lag_1` and steals credit). We will maintain two variants: **Model B** (full features, production forecasts) and **Model A** (business-actionable levers + known-in-advance/static context only) used for SHAP/explainability.

## Epic 1: Data Ingestion & Cleaning (Foundation)
**Goal:** Establish a reliable data loading layer that handles all raw inputs and applies necessary cleaning rules.

*   **[DATA-01] Port and Verify Data Loaders**
    *   **Description:** Review and verify `prototype.io_utils` functions. Ensure `load_written_sales` correctly cleans "NULL" traffic strings and handles all column types. Ensure `load_store_master` correctly parses `proforma_annual_sales` and `is_outlet`.
    *   **Output:** Validated `io_utils.py` (already mostly done, just need verification).

*   **[DATA-02] Implement Traffic Missingness Flag**
    *   **Description:** Add logic to `load_written_sales` (or a downstream step) to create a `has_traffic_data` binary flag per store.
    *   **Logic:** If store has >X% missing traffic weeks or systematic nulls, flag as 0.
    *   **Usage:** Used to exclude stores from the Conversion Model training.

*   **[DATA-03] Ingest DMA Brand Awareness / Consideration Feeds**
    *   **Description:** Load awareness & consideration metrics from `Awareness_Consideration_2022-2025.xlsx`. Data structure: **7.4k rows = 39 aggregate market groupings × ~190 weeks** (not individual store-level). Use **YOUGOV_DMA_MAP** table (from Master Tables workbook, sheet 25) to cascade Market → Store DMA → individual stores. Provides 96.7% coverage for operational stores.
    *   **Requirements:** Data is already weekly; forward-fill any gaps. Expose both awareness (required) and consideration (optional Tier 3) signals. For 7 stores without direct mapping (CHAMPAIGN/SPRINGFIELD, SPARTANBURG/ASHEVILLE, etc.), apply fallback to "Single DMAs" groupings or sister-DMA.
    *   **Usage:** Awareness is now a top-15 WEB driver (Pooled ρ=+0.31) and must be present in the canonical training table.

*   **[DATA-04] Land CRM Demographic Mix Tables**
    *   **Description:** Ingest the 23 CRM percent-of-customer features (with lags 1/4 and rolling windows 4/8) that were refreshed in the driver screening. Persist clean weekly aggregates per `(store_id, dma_id)` so they can be treated as static/cross-sectional predictors.
    *   **Requirements:** Handle privacy constraints (PII stripped), normalize percentages to sum ≤ 1, and provide both latest snapshot and optional lagged views.
    *   **Usage:** Even though CRM mix has weak FE correlations, pooled ρ hits 0.15-0.21 for Sales/AOV, so the canonical table needs these columns for cross-sectional lift (Tier 3).

## Epic 2: Layer 0 Artifacts (Seasonality & Baselines)
**Goal:** Generate the "known-in-advance" baselines that serve as inputs to the ML models.

*   **[SEAS-01] Verify DMA Seasonal Weight Computation**
    *   **Description:** Verify `prototype.seasonality.compute_dma_seasonality`. Ensure it handles the detrending, normalization, and week 53 folding correctly.
    *   **Output:** A verified function producing `dma_seasonal_weight` per `dma_id` and `week_of_year`.

*   **[SEAS-02] Verify Sister-DMA Fallback**
    *   **Description:** Ensure `prototype.sister_dma` is correctly integrated into the seasonality computation. New stores/DMAs must inherit weights from their "Sister DMA".

*   **[SEAS-03] (Optional) DMA-Level Prophet Components**
    *   **Description:** Implement a lightweight per-DMA Prophet model to extract `trend`, `yearly_seasonality`, and `holiday_effect` components.
    *   **Priority:** Low (use as fallback if standard seasonal weights fail).

## Epic 3: Canonical Training Table Generation
**Goal:** Create the single source of truth for training, preventing leakage and ensuring consistency.

*   **[ETL-01] Define Canonical Schema**
    *   **Description:** Define the exact column list for the training table (Keys, Targets at $t_0+h$, Features at $t_0$, Features at $t_0+h$).
    *   **Output:** A schema definition object or document.

*   **[ETL-02] Implement "History Exploder"**
    *   **Description:** Create the core transformation logic. For every historical week $W$ (acting as $t_0$):
        1.  Generate 52 rows (horizons $h=1..52$).
        2.  Calculate `target_date = t0 + h`.
        3.  Join Targets (Sales, AOV, Conversion) from `target_date`.
    *   **Constraint:** This must be efficient (vectorized if possible).

*   **[ETL-03] Implement Feature Attachment (Join Logic)**
    *   **Description:** Join the "Observed at $t_0$" features and "Known at $t_0+h$" features to the exploded table.
    *   **Crucial:** Ensure "Observed" features are joined on $t_0$, NOT $t_0+h$.

## Epic 4: Feature Engineering
**Goal:** Implement the feature calculation logic (transformers).

*   **[FE-01] Time-Varying "Known in Advance" Features**
    *   **Description:** Implement transformer for: `woy_sin`, `woy_cos`, `is_holiday` (from `io_utils`), `weeks_to_holiday`, `dma_seasonal_weight` (lookup).
    *   **Context:** These depend on `target_date` ($t_0+h$).

*   **[FE-02] Sales & AOV Dynamics (B&M)**
    *   **Description:** Implement rolling means (4, 8, 13 weeks) and lags (1, 4, 13, 52 weeks) for Sales and AOV.
    *   **Specifics:** Use Winsorization (MAD-based) for rolling means to handle outliers.

*   **[FE-03] Web-Specific Dynamics**
    *   **Description:** Implement Web Sales rolling means (4 weeks) and `allocated_web_traffic` rolls/lags.
    *   **Source:** Use `Ecomm Traffic.csv` data, NOT `Written Sales Data`.

*   **[FE-04] Conversion & Omnichannel Features**
    *   **Description:** Implement `logit_conversion` rolls, `pct_omni_channel` rolls.
    *   **Note:** Exclude stores with `has_traffic_data=0`.

*   **[FE-05] Static Store DNA**
    *   **Description:** Implement `proforma_annual_sales`, `is_outlet`, `weeks_since_open` (at $t_0+h$).

*   **[FE-06] Cannibalization Pressure**
    *   **Description:** Implement the time-varying pressure formula:
        $$ Pressure_i = \sum_{j \neq i} \exp(-dist_{ij}/8) \times (1 + weeks\_open_j/13) $$
    *   **Input:** Store opening dates and pairwise distances (from `prototype.geo`).

*   **[FE-07] DMA Awareness & Consideration Features**
    *   **Description:** Join the `[DATA-03]` market-group awareness to the canonical table. First, map stores to awareness markets via **YOUGOV_DMA_MAP** (store's market_city → Market), then join at `(Market, target_week)` granularity. Provide `brand_awareness_dma_score` (required) and `brand_consideration_dma_score` (optional). Handle 7 unmapped stores via "Single DMAs" groupings or sister-DMA fallback.
    *   **Guidance:** Awareness should be treated as a static/slow-moving feature (Tier 1 for WEB, Tier 2 for B&M); FE correlations ≈ 0, so ensure it only leaks through cross-sectional columns. Data is already weekly; forward-fill gaps.
    *   **Validation:** Compare awareness correlations in the final feature set vs. driver screening benchmarks (Web Pooled ρ=+0.31, B&M Pooled ρ=+0.18) to confirm faithful ingestion. Verify 96.7%+ operational store coverage.

*   **[FE-08] CRM Demographic Mix Attachment**
    *   **Description:** Join the `[DATA-04]` CRM mix tables to the canonical dataset. Provide both latest-snapshot static features (e.g., `crm_owner_pct`, `crm_single_family_pct`) and the lagged/rolling variants so pooled effects are captured without leakage. Lag/rolling generation occurs in Epic 4 feature engineering steps.
    *   **Guidance:** Treat CRM percentages as Tier 3 cross-sectional predictors—include them in the feature schema but keep them segregated from time-varying operational levers. Document which specific segments showed the strongest pooled lift (homeowners, single-family, high income).
    *   **Checks:** Verify pooled correlations in the engineered dataset remain within ±0.02 of the driver screening values (ρ ≈ 0.15-0.21) so downstream models capture the intended baseline adjustments.

## Epic 5: Model Specification (CatBoost)
**Goal:** Configure and train the 5 global models with coherent multi-target behavior prioritized over per-target quantiles.

*   **[ML-00] Define Dual-Model Feature Partitions**
    *   **Description:** Freeze column lists for Model A (actionable levers + known-in-advance/static context, no autoregressive lags/rolls) vs. Model B (full feature set including lags/rolls). Ensure ETL outputs both views consistently for train/inference.

*   **[ML-01] Sales B&M P50 & P90 Models**
    *   **Description (Updated):** Use a joint multi-target CatBoost (e.g., MultiRMSE on log sales/AOV/orders) to preserve coherence across the P&L triangle. If P50/P90 are needed, derive them post-hoc from the joint model (e.g., bias/variance-based correction or residual quantiles) rather than separate quantile models that would break the coupling.
    *   **Features:** All B&M features + Horizon ($h$).

*   **[ML-02] Web Sales Models**
    *   **Description (Updated):** Mirror the coherent joint approach for WEB (log sales/AOV/orders) to keep Sales/AOV/Orders consistent. Add optional post-hoc quantile estimation if needed, but avoid separate quantile models that lose cross-target consistency.
    *   **Features:** Web-specific feature set.

*   **[ML-03] Conversion Model**
    *   **Description:** Configure CatBoost with `loss_function='RMSE'` (on logit target) and `sample_weight=traffic` on the Model B (full feature) view.

*   **[ML-04] Actionable-Only Explainability Models**
    *   **Description (Updated):** Replace standalone Model A training with a surrogate model that overfits Model B predictions using only the business-actionable lever features. Train the surrogate on Model B outputs (per target) so SHAP/driver analysis reflects how those specific levers influence Model B. Do not use surrogate predictions for production forecasts.

## Epic 6: Pipelines & Validation
**Goal:** Tie it all together into Training and Inference workflows.

*   **[PIPE-01] Cross-Validation Loop**
    *   **Description:** Implement Rolling Origin CV.
        *   Split 1: Train Jan '22 - Jun '24 -> Test Jul '24 - Sep '24
        *   Split 2: Train Jan '22 - Sep '24 -> Test Oct '24 - Dec '24
    *   **Artifacts:** Ensure Layer 0 artifacts are re-computed inside the training loop (no leakage).

*   **[PIPE-02] Inference Function**
    *   **Description:** Create `predict(history_df, store_ids, horizon=52)` function.
    *   **Steps:** Generate features -> Score models -> Inverse transform (Exp/Expit) -> Return Forecast DataFrame.

*   **[PIPE-03] Validation Reporting**
    *   **Description:** Implement wMAPE, Bias, and P90 Coverage calculations. Report Model B accuracy; attach Model A SHAP driver summaries for interpretability.
    *   **Slices:** Report metrics by Channel, Horizon Bucket (1-4, 5-13, 14-52), and Store Maturity.

*   **[PIPE-04] Explainability Pipeline (Model A)**
    *   **Description:** Generate SHAP values from Model A for each retrain, archive top driver tables (store/channel/horizon buckets), and compare attribution shifts vs. Model B to ensure lags are not absorbing operational effects.
