# Models Module

This module contains the core prediction models for the Company X sales forecasting system. It implements a dual-model, dual-channel architecture using CatBoost gradient boosting with multi-target regression.

## Overview

The models folder provides CatBoost-based predictors for forecasting key retail metrics across two sales channels (B&M and WEB). The system uses a sophisticated approach that balances accuracy and interpretability through two complementary modeling strategies:

- **Model B (Production)**: Full autoregressive features for maximum predictive accuracy
- **Model A (Explainability)**: Business-actionable levers only, used via surrogate modeling for SHAP-based interpretation

### Predicted Metrics

The models predict the following transformed metrics:
- **Sales** (log-transformed)
- **AOV** (Average Order Value, log-transformed)
- **Orders** (log-transformed, directly predicted)
- **Conversion Rate** (logit-transformed, B&M only - requires foot traffic data)
- **Traffic** (derived via Monte Carlo simulation, B&M only)

## File Structure

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Module entry point with exports and factory function | `get_predictor()` - Factory for channel-specific predictors |
| `base.py` | Abstract base class for all predictors | `BaseCatBoostPredictor`, `PredictionResult` |
| `bm_predictor.py` | B&M (Brick & Mortar) channel predictors | `BMPredictor`, `BMMultiObjectivePredictor`, `BMConversionPredictor` |
| `web_predictor.py` | WEB (E-commerce) channel predictors | `WEBPredictor`, `WEBMultiObjectivePredictor` |
| `traffic.py` | Monte Carlo traffic estimation with uncertainty quantification | `TrafficEstimator`, `estimate_traffic_quantiles()` |
| `surrogate.py` | Surrogate model for SHAP-based explainability | `SurrogateExplainer` |

## Key Components

### Base Classes (`base.py`)

**`BaseCatBoostPredictor`**
- Abstract base class providing common functionality for all predictors
- Handles feature preparation and channel-specific filtering
- Manages categorical feature encoding
- Provides model persistence (save/load)
- Automatically excludes irrelevant features based on channel:
  - B&M models exclude WEB-only features (web traffic lags, web sales lags, etc.)
  - WEB models exclude B&M-only features (conversion, cannibalization metrics, etc.)

**`PredictionResult`**
- Dataclass container for multi-objective prediction outputs
- Stores log_sales, log_aov, log_orders, and optionally logit_conversion
- Includes derived_log_orders (Sales - AOV) as an alternative to direct prediction

### B&M Predictors (`bm_predictor.py`)

**`BMMultiObjectivePredictor`**
- Predicts Sales, AOV, and Orders simultaneously using CatBoost's MultiRMSE loss
- Trains on samples with valid sales, AOV, and order_count > 0
- Automatically excludes WEB-only features
- Returns `PredictionResult` with all three predictions plus derived orders

**`BMConversionPredictor`**
- Predicts logit-transformed conversion rate for B&M stores
- Requires physical foot traffic data (has_traffic_data == 1)
- Single-target RMSE loss
- Conversion is B&M-specific since it requires store traffic

**`BMPredictor` (Facade)**
- Combines MultiObjectivePredictor and ConversionPredictor
- Provides complete B&M forecasts including traffic estimation
- Computes RMSE values on validation set for uncertainty quantification
- Supports model checkpoint save/load for both sub-models

### WEB Predictors (`web_predictor.py`)

**`WEBMultiObjectivePredictor`**
- Predicts Sales, AOV, and Orders for e-commerce channel
- Uses MultiRMSE loss for joint optimization
- Excludes B&M-only features (conversion, cannibalization, merchandising SF)
- No conversion prediction (web channel lacks physical traffic)

**`WEBPredictor` (Facade)**
- Wraps WEBMultiObjectivePredictor for consistent interface with BMPredictor
- Computes RMSE values for uncertainty quantification
- Simpler than BMPredictor since WEB doesn't require conversion or traffic models

### Traffic Estimation (`traffic.py`)

**`TrafficEstimator`**
- Derives traffic estimates using Monte Carlo simulation
- Implements the relationship: Traffic = Sales / (AOV * Conversion)
- Propagates prediction uncertainty through the formula
- Returns P10/P50/P90 percentiles for confidence intervals

**Process:**
1. Simulates draws from log-normal distributions (Sales, AOV) and logit-normal (Conversion)
2. Uses RMSE values from validation set as standard deviations
3. Computes traffic for each simulation draw
4. Aggregates percentiles across simulations

**`estimate_traffic_quantiles()`**
- Convenience function wrapping TrafficEstimator
- Default: 10,000 simulations with batching for memory efficiency

### Surrogate Explainability (`surrogate.py`)

**`SurrogateExplainer`**
- Trains a lightweight Model A to approximate Model B predictions
- Uses only business-actionable features (excludes lags, rolling means, contextual features)
- Enables SHAP-based interpretation of what drives Model B forecasts

**Key Features:**
- `fit()`: Train surrogate on Model A features to match Model B predictions
- `compute_shap_values()`: Generate SHAP values for input data
- `explain()`: Comprehensive SHAP analysis with plots (summary, dependence, cohort sensitivity)
- `build_contributor_strings()`: Per-row top contributor annotations

**Excluded from Model A:**
- Autoregressive lags (not actionable)
- Rolling averages (derived from lags)
- Temporal features (woy, month, quarter, fiscal_period)
- Metadata (profit_center_nbr, dma, dates)

**Actionable Features in Model A:**
- Pricing and promotions
- Product mix (premium %, value %, white glove %)
- Marketing spend
- Awareness metrics
- Store characteristics (comp store, outlet, new store)
- Seasonal flags (holiday windows, black friday)

## Model Architecture

### CatBoost with MultiRMSE Loss

Both B&M and WEB multi-objective predictors use CatBoost's **MultiRMSE** loss function for multi-target regression:

- **Advantages:**
  - Joint optimization of correlated targets (Sales, AOV, Orders)
  - Shared tree structure captures cross-metric dependencies
  - More efficient than training separate models
  - Consistent feature importance across metrics

- **Configuration:**
  - Default: 5000 iterations, learning rate 0.05, depth 6
  - Early stopping with validation set (if provided)
  - Categorical feature handling via CatBoost's native encoding
  - Random seed 42 for reproducibility

### Feature Engineering Integration

The models rely on features from the `forecasting.regressor.features` module:
- **Temporal features**: Weekly seasonality, fiscal periods, holiday windows
- **Autoregressive features**: Lags (1, 4, 13, 52 weeks) and rolling statistics
- **Market dynamics**: Cannibalization, awareness, competitive pressure
- **Store attributes**: Merchandising square footage, design, outlet status
- **Product mix**: Premium %, value %, white glove %, omni-channel %

## Usage Examples

### Training and Predicting with B&M Channel

```python
from forecasting.regressor.models import BMPredictor

# Initialize predictor
bm_predictor = BMPredictor(iterations=5000)

# Train on prepared data (with features and labels)
bm_predictor.fit(train_df, val_df)

# Generate predictions
predictions = bm_predictor.predict(test_df, estimate_traffic=True)

# Access results
log_sales = predictions.log_sales
log_aov = predictions.log_aov
log_orders = predictions.log_orders
logit_conversion = predictions.logit_conversion
traffic_p50 = predictions.traffic.p50  # Median traffic estimate

# Save trained models
bm_predictor.save_models("checkpoints/bm/")
```

### Training and Predicting with WEB Channel

```python
from forecasting.regressor.models import WEBPredictor

# Initialize predictor
web_predictor = WEBPredictor(iterations=5000)

# Train
web_predictor.fit(train_df, val_df)

# Predict
predictions = web_predictor.predict(test_df)

# Access results (no conversion or traffic for WEB)
log_sales = predictions.log_sales
log_aov = predictions.log_aov
log_orders = predictions.log_orders

# Save model
web_predictor.save_models("checkpoints/web/")
```

### Using Factory Function

```python
from forecasting.regressor.models import get_predictor

# Automatically get the right predictor for a channel
predictor = get_predictor("B&M")  # Returns BMPredictor
# predictor = get_predictor("WEB")  # Returns WEBPredictor

predictor.fit(train_df, val_df)
predictions = predictor.predict(test_df)
```

### Traffic Estimation

```python
from forecasting.regressor.models import estimate_traffic_quantiles

# Standalone traffic estimation (if you have predictions and RMSE values)
traffic_result = estimate_traffic_quantiles(
    log_sales_pred=log_sales,
    log_aov_pred=log_aov,
    logit_conv_pred=logit_conversion,
    sales_rmse=0.15,
    aov_rmse=0.08,
    conv_rmse=0.12,
    n_simulations=10000,
    random_seed=42
)

# Access percentiles
traffic_p10 = traffic_result.p10  # Lower bound
traffic_p50 = traffic_result.p50  # Median
traffic_p90 = traffic_result.p90  # Upper bound
```

### Surrogate Explainability

```python
from forecasting.regressor.models import SurrogateExplainer

# Initialize explainer
explainer = SurrogateExplainer()

# Train surrogate on Model A features to fit Model B predictions
# model_a_df has business lever features only
# model_b_predictions is the target to approximate
explainer.fit(model_a_df, model_b_predictions)

# Generate SHAP explanation with plots
contributor_df = explainer.explain(
    df=model_a_df,
    output_dir="output/shap/",
    name="bm_sales_h1",
    keys=["profit_center_nbr", "target_week_date"],
    target_label="pred_log_sales",
    top_k_contributors=3,
    generate_plots=True
)

# Compute SHAP values manually
shap_values = explainer.compute_shap_values(model_a_df)

# Get top features by SHAP importance
top_features = explainer.get_top_features(shap_values, top_n=10)

# Feature importance from surrogate model
importance_df = explainer.get_feature_importance()
```

## Integration with Broader Forecasting System

The models module is the core prediction engine within the larger forecasting pipeline:

```
forecasting/
├── regressor/
│   ├── features/          # Feature engineering (inputs to models)
│   │   ├── temporal_features.py
│   │   ├── autoregressive_features.py
│   │   ├── dynamics_features.py
│   │   └── ...
│   ├── models/            # THIS MODULE (prediction models)
│   │   ├── base.py
│   │   ├── bm_predictor.py
│   │   ├── web_predictor.py
│   │   ├── traffic.py
│   │   └── surrogate.py
│   ├── configs/           # Model configurations
│   │   └── model_config.py
│   └── pipeline/          # End-to-end orchestration
│       └── train_predict.py
```

### Typical Workflow

1. **Feature Engineering** (`features/`): Generate temporal, autoregressive, and market dynamics features
2. **Model Training** (`models/`): Train channel-specific predictors with MultiRMSE loss
3. **Prediction** (`models/`): Generate forecasts with uncertainty quantification
4. **Explainability** (`models/surrogate.py`): Train surrogate on Model A features for SHAP analysis
5. **Evaluation** (`pipeline/`): Assess accuracy, generate reports, visualize results

### Data Flow

```
Raw Data
    |
    v
Feature Engineering
    |
    v
Train/Val/Test Split
    |
    v
Model Training (BMPredictor / WEBPredictor)
    |
    +---> Multi-Objective Model (Sales, AOV, Orders)
    +---> Conversion Model (B&M only)
    |
    v
Predictions
    |
    +---> Traffic Estimation (Monte Carlo)
    +---> Surrogate Training (Model A)
    +---> SHAP Explainability
    |
    v
Inverse Transform (exp, sigmoid)
    |
    v
Final Forecasts (Sales, AOV, Orders, Conversion, Traffic)
```

## Model Variants: Model A vs Model B

### Model B (Production Model)
- **Purpose**: Maximum predictive accuracy for production forecasts
- **Features**: Full feature set including autoregressive lags, rolling statistics, temporal encodings
- **Use Case**: Generate final forecasts with highest accuracy
- **Training**: Standard fit() on BMPredictor / WEBPredictor

### Model A (Explainability Model)
- **Purpose**: Interpretability and business insights
- **Features**: Only actionable business levers (pricing, promotions, product mix, marketing)
- **Use Case**: Understand drivers via SHAP, scenario planning, what-if analysis
- **Training**: Via SurrogateExplainer, trained to approximate Model B predictions

### Why Surrogate Approach?
- Model B's autoregressive features (lags, rolling means) are **not actionable** for business planning
- Direct SHAP on Model B would highlight "last week's sales" as top driver (not insightful)
- Surrogate Model A uses only **controllable levers** that business can adjust
- SHAP on surrogate reveals: "What business actions drive Model B's predictions?"

## Configuration

Model hyperparameters are managed via `forecasting.regressor.configs.model_config`:

- `get_bm_multi_config()`: B&M multi-objective model config
- `get_bm_conversion_config()`: B&M conversion model config
- `get_web_multi_config()`: WEB multi-objective model config
- `get_surrogate_config()`: Surrogate explainability config

Default hyperparameters:
- **iterations**: 5000
- **learning_rate**: 0.05
- **depth**: 6
- **loss_function**: MultiRMSE (multi-objective), RMSE (conversion/surrogate)
- **early_stopping_rounds**: 100
- **random_seed**: 42

## Notes

- All predictions are in **transformed space** (log for Sales/AOV/Orders, logit for Conversion)
- Inverse transformations (exp, sigmoid) are applied downstream in the pipeline
- **Conversion** is B&M-only because it requires physical foot traffic data
- **Traffic** is derived (not directly predicted) via Monte Carlo simulation
- Channel-specific feature filtering is automatic based on predictor type
- RMSE values from validation set are used for traffic quantile estimation
- Surrogate R2 score indicates how well Model A approximates Model B (target: 0.90+)
