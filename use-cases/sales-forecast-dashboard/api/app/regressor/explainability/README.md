# Explainability Module

## Overview

This module provides SHAP-based model interpretation utilities for Company X Sales Forecasting System. The explainability layer is a critical component that enables business stakeholders to understand which operational levers and contextual factors drive forecast predictions across different stores, channels, and time horizons.

The module supports the dual-model architecture employed by the forecasting system:
- **Model A (Surrogate Model)**: Trained exclusively on business-actionable features for explainability
- **Model B (Production Model)**: Full-featured model with autoregressive lags for production forecasts

This separation prevents autoregressive lag features from overshadowing operational levers in SHAP attribution. For example, a week-1 omnichannel marketing push would later show up in `lag_1` features and steal credit from the actual marketing lever if both were included in the same explainability model.

## Architecture Context

### Dual-Model Strategy

The forecasting system maintains two distinct model variants:

1. **Model B (Full Features, Production Forecasts)**
   - Includes all features: lags, rolling averages, operational levers, static context
   - Optimized for prediction accuracy
   - Used for generating production forecasts
   - Predicts: Sales, AOV, Orders, Conversion, Traffic for B&M and WEB channels

2. **Model A (Business Levers, Explainability)**
   - Contains only business-actionable levers and known-in-advance/static context
   - Excludes autoregressive lag features and derived metrics that could obscure operational drivers
   - Used exclusively for SHAP analysis and driver attribution
   - Provides interpretable explanations that align with business decision-making

### Why This Matters

When a single full-feature model is used for both prediction and explanation, autoregressive lags dominate SHAP values because they directly encode recent outcomes. This makes it difficult to identify:
- Which marketing initiatives drove changes
- How seasonality and holidays affect different channels
- Which store characteristics amplify or dampen performance
- What controllable levers stakeholders should focus on

By training Model A on operational features only, SHAP values reveal actionable insights about business drivers rather than statistical patterns in recent history.

## Module Contents

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports and public API definition |
| `shap_analysis.py` | Core SHAP computation and visualization functions |
| `contributors.py` | Utilities for extracting and formatting top contributors per prediction |

### File Details

#### `shap_analysis.py` (9 functions)

Core SHAP computation and visualization capabilities:

- **`compute_shap_values(model, X)`**: Computes SHAP values using TreeExplainer for tree-based models
- **`get_top_features_by_shap(shap_values, feature_names, top_n)`**: Ranks features by mean absolute SHAP value
- **`plot_shap_summary(shap_values, X, output_path, ...)`**: Generates SHAP summary plot (beeswarm) showing feature importance and directionality
- **`plot_shap_importance(shap_values, X, output_path, ...)`**: Creates bar plot of mean absolute SHAP values per feature
- **`plot_shap_dependence(shap_values, X, feature, output_path, ...)`**: Generates dependence plot showing how feature values relate to SHAP contributions
- **`plot_horizon_sensitivity(shap_values, feature_names, horizons, ...)`**: Plots how feature importance varies by forecast horizon (1-52 weeks)
- **`plot_cohort_importance(shap_values, feature_names, cohort_values, ...)`**: Shows mean SHAP importance grouped by cohorts (e.g., store, DMA, channel)
- **`plot_fit_scatter(actual, predicted, output_path, ...)`**: Creates actual vs predicted scatter plot with R² metric

#### `contributors.py` (4 functions)

Per-row contributor extraction and aggregation:

- **`build_contributor_strings(shap_values, feature_names, feature_values, top_k)`**: Generates human-readable strings showing top-k contributors per prediction row (format: "feature=value:+/-shap; ...")
- **`get_top_contributors_dataframe(shap_values, feature_names, feature_values, top_k, ...)`**: Returns structured DataFrame with separate columns for each contributor rank (top_1_feature, top_1_value, top_1_shap, etc.)
- **`aggregate_feature_importance(shap_values, feature_names, group_by)`**: Aggregates mean absolute SHAP by feature, optionally grouped by a categorical variable
- **`filter_contributors_by_direction(shap_values, feature_names, direction, top_k)`**: Extracts top contributors filtered by positive or negative SHAP direction

#### `__init__.py`

Provides clean public API by exporting all functions from `shap_analysis` and `contributors` modules. Users can import directly from the explainability package:

```python
from forecasting.regressor.explainability import (
    compute_shap_values,
    build_contributor_strings,
    plot_shap_summary,
    # ... etc
)
```

## Key Classes and Methods

### SHAP Analysis Workflow

The typical SHAP analysis workflow follows these steps:

1. **Compute SHAP Values**: Use `compute_shap_values()` with Model A to generate attribution
2. **Identify Top Features**: Call `get_top_features_by_shap()` to rank features globally
3. **Generate Visualizations**: Create plots using `plot_shap_summary()`, `plot_shap_importance()`, etc.
4. **Extract Per-Row Contributors**: Use `build_contributor_strings()` or `get_top_contributors_dataframe()` for row-level attribution

### SHAP Value Interpretation

SHAP (SHapley Additive exPlanations) values represent the contribution of each feature to a prediction:
- **Positive SHAP**: Feature increases the prediction (e.g., higher brand awareness increases sales)
- **Negative SHAP**: Feature decreases the prediction (e.g., cannibalization pressure reduces sales)
- **Magnitude**: Larger absolute values indicate stronger influence

The sum of all SHAP values plus the base value (model's average prediction) equals the final prediction:
```
prediction = base_value + sum(shap_values)
```

### Horizon and Cohort Analysis

Two specialized functions enable deeper analysis:

- **Horizon Sensitivity** (`plot_horizon_sensitivity`): Shows how feature importance changes across forecast horizons. For example, `dma_seasonal_weight` may be more important for long-horizon forecasts (13-52 weeks) while recent rolling averages matter more for near-term horizons (1-4 weeks).

- **Cohort Importance** (`plot_cohort_importance`): Aggregates SHAP values by cohorts (stores, DMAs, channels) to identify which groups are most influenced by top features. Useful for understanding geographic or channel-specific drivers.

## Usage Examples

### Basic SHAP Analysis

```python
from forecasting.regressor.explainability import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_importance,
    get_top_features_by_shap,
)

# Assume model_a is trained Model A (business levers only)
# X_train contains the feature matrix used for training

# 1. Compute SHAP values
shap_values = compute_shap_values(model_a, X_train)

# 2. Get top 10 most important features globally
top_features = get_top_features_by_shap(
    shap_values,
    feature_names=X_train.columns.tolist(),
    top_n=10
)
print("Top 10 Features:", top_features)

# 3. Generate summary plot (beeswarm)
plot_shap_summary(
    shap_values,
    X_train,
    output_path="output/shap_summary.png",
    title="SHAP Summary: Model A (Business Levers)",
    max_display=20
)

# 4. Generate importance bar plot
plot_shap_importance(
    shap_values,
    X_train,
    output_path="output/shap_importance.png",
    title="Feature Importance by Mean |SHAP|",
    max_display=20
)
```

### Per-Row Contributor Extraction

```python
from forecasting.regressor.explainability import (
    build_contributor_strings,
    get_top_contributors_dataframe,
)

# Build human-readable contributor strings
contributor_strings = build_contributor_strings(
    shap_values,
    feature_names=X_train.columns.tolist(),
    feature_values=X_train,
    top_k=3
)

# Add to predictions DataFrame
predictions_df['top_contributors'] = contributor_strings
print(predictions_df[['store_id', 'target_date', 'prediction', 'top_contributors']].head())

# Example output:
# top_contributors: "horizon=4:+0.523; brand_awareness_dma_score=0.75:-0.312; is_outlet=1:+0.187"

# Or get structured DataFrame with separate columns
contributors_df = get_top_contributors_dataframe(
    shap_values,
    feature_names=X_train.columns.tolist(),
    feature_values=X_train,
    top_k=3,
    key_columns=predictions_df[['store_id', 'target_date', 'prediction']]
)
```

### Dependence and Interaction Analysis

```python
from forecasting.regressor.explainability import plot_shap_dependence

# Analyze how brand awareness affects predictions
plot_shap_dependence(
    shap_values,
    X_train,
    feature="brand_awareness_dma_score",
    output_path="output/dependence_awareness.png",
    interaction_feature="horizon",  # Color by horizon to see interaction
    title="SHAP Dependence: Brand Awareness (colored by Horizon)"
)

# Analyze cannibalization pressure
plot_shap_dependence(
    shap_values,
    X_train,
    feature="cannibalization_pressure",
    output_path="output/dependence_cannibalization.png",
    title="SHAP Dependence: Cannibalization Pressure"
)
```

### Horizon Sensitivity Analysis

```python
from forecasting.regressor.explainability import plot_horizon_sensitivity

# Analyze how feature importance varies by horizon
plot_horizon_sensitivity(
    shap_values,
    feature_names=X_train.columns.tolist(),
    horizons=X_train['horizon'],
    top_features=top_features[:10],  # Use top 10 from earlier
    output_path="output/horizon_sensitivity.png",
    title="Feature Importance by Forecast Horizon"
)
```

### Cohort Importance Analysis

```python
from forecasting.regressor.explainability import plot_cohort_importance

# Analyze SHAP importance by store
plot_cohort_importance(
    shap_values,
    feature_names=X_train.columns.tolist(),
    cohort_values=predictions_df['store_id'],
    top_features=top_features[:10],
    output_path="output/cohort_importance_store.png",
    cohort_name="Store",
    title="Top Stores by Feature Sensitivity",
    top_cohorts=15
)

# Analyze by DMA
plot_cohort_importance(
    shap_values,
    feature_names=X_train.columns.tolist(),
    cohort_values=predictions_df['dma_id'],
    top_features=top_features[:10],
    output_path="output/cohort_importance_dma.png",
    cohort_name="DMA",
    title="Top DMAs by Feature Sensitivity",
    top_cohorts=15
)
```

### Filtering by Direction

```python
from forecasting.regressor.explainability import filter_contributors_by_direction

# Get top positive contributors (features increasing prediction)
positive_drivers = filter_contributors_by_direction(
    shap_values,
    feature_names=X_train.columns.tolist(),
    direction="positive",
    top_k=3
)

# Get top negative contributors (features decreasing prediction)
negative_drivers = filter_contributors_by_direction(
    shap_values,
    feature_names=X_train.columns.tolist(),
    direction="negative",
    top_k=3
)

predictions_df['positive_drivers'] = positive_drivers
predictions_df['negative_drivers'] = negative_drivers
```

## Integration with Forecasting System

### Workflow Integration

The explainability module fits into the broader forecasting system as follows:

1. **Training Phase**:
   - Train Model B (full features) for production predictions
   - Train Model A (business levers only) for explainability
   - Both models share the same training data but use different feature subsets

2. **Prediction Phase**:
   - Use Model B to generate production forecasts
   - Use Model A to compute SHAP values for the same samples
   - Attach SHAP-based attribution to forecast outputs

3. **Reporting Phase**:
   - Generate SHAP visualizations for stakeholder presentations
   - Include top contributor strings in forecast delivery tables
   - Create horizon and cohort analyses for strategic planning

### Model A Feature Set

Model A should include only business-actionable and known-in-advance features:

**Time-Varying Known-in-Advance**:
- `horizon`: Forecast horizon (1-52 weeks)
- `woy_sin`, `woy_cos`: Week-of-year seasonality encoding
- `is_holiday`, `weeks_to_holiday`: Holiday indicators
- `dma_seasonal_weight`: DMA-specific seasonal patterns
- `target_week`, `target_month`, `target_quarter`: Temporal identifiers

**Static Store Context**:
- `proforma_annual_sales`: Store size proxy
- `is_outlet`: Outlet vs regular store indicator
- `weeks_since_open`: Store maturity
- `dma_id`, `market_city`: Geographic identifiers

**Cross-Sectional/Slow-Moving Signals**:
- `brand_awareness_dma_score`: DMA-level brand awareness (39 market groups)
- `brand_consideration_dma_score`: DMA-level consideration (optional)
- CRM demographic mix features (23 features): `crm_owner_pct`, `crm_single_family_pct`, etc.

**Time-Varying Operational Context**:
- `cannibalization_pressure`: Time-varying pressure from nearby store openings
- `weeks_since_open_j`: Used in cannibalization formula

**Features EXCLUDED from Model A** (used only in Model B):
- All lag features: `sales_lag_1`, `aov_lag_4`, etc.
- All rolling averages: `sales_rolling_mean_4w`, `traffic_rolling_mean_8w`, etc.
- Derived metrics that encode recent outcomes: `logit_conversion`, `pct_omni_channel`, etc.

### Output Artifacts

Typical output artifacts from the explainability module:

1. **SHAP Summary Plots**: High-level overview of feature importance and directionality
2. **SHAP Importance Bar Plots**: Ranked feature importance for presentations
3. **Dependence Plots**: Deep dives into specific feature relationships
4. **Horizon Sensitivity Plots**: Feature importance stratified by forecast horizon
5. **Cohort Importance Plots**: Store/DMA-level sensitivity rankings
6. **Fit Scatter Plots**: Model A performance validation (actual vs predicted)
7. **Contributor DataFrames**: Per-row attribution tables for forecast delivery
8. **Contributor Strings**: Human-readable explanations embedded in forecast files

### Best Practices

1. **Always use Model A for SHAP analysis**: Never compute SHAP values on Model B, as lag features will dominate attribution

2. **Compute SHAP on held-out validation data**: This ensures attribution reflects out-of-sample behavior

3. **Generate horizon-stratified plots**: Feature importance varies significantly by horizon (near-term vs long-term forecasts)

4. **Include contributor strings in forecast outputs**: Stakeholders need row-level explanations to trust predictions

5. **Validate Model A performance**: Model A should achieve reasonable accuracy (e.g., R² > 0.70) to ensure SHAP attributions are meaningful. If Model A performs poorly, SHAP values may reflect model errors rather than true feature importance.

6. **Aggregate SHAP by cohorts**: Store managers and regional leaders need cohort-level summaries to identify patterns across their portfolios

7. **Compare positive vs negative drivers**: Understanding what suppresses forecasts is as important as understanding what lifts them

8. **Document feature definitions**: Ensure SHAP visualizations are accompanied by clear feature definitions for non-technical stakeholders

## Dependencies

- **shap**: SHAP library for TreeExplainer and visualization
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **matplotlib**: Plotting backend
- **catboost** (or similar tree-based model): Model training

Install dependencies:
```bash
pip install shap numpy pandas matplotlib catboost
```

## Future Enhancements

Potential additions to the explainability module:

- **Interaction SHAP values**: Compute pairwise feature interactions to understand synergies
- **Temporal SHAP tracking**: Track how feature importance evolves over time as new data arrives
- **Automatic anomaly detection**: Flag predictions with unusual SHAP patterns for review
- **Counterfactual analysis**: Show how predictions would change under different lever settings
- **SHAP waterfall plots**: Per-row waterfall charts showing contribution breakdown
- **Feature group aggregation**: Aggregate SHAP by logical feature groups (seasonality, demographics, operational, etc.)

## References

- **SHAP Documentation**: https://shap.readthedocs.io/
- **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions" - Original SHAP paper
- **Project Plan**: `/forecasting/regressor/PROJECT_PLAN.md` - Overall forecasting system architecture
- **Engineering Instructions**: `/forecasting/regressor/ENGINEERING_INSTRUCTIONS.md` - Detailed technical specifications
