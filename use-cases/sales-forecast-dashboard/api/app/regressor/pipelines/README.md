# Forecasting Pipelines

## Overview

This folder contains the orchestration pipelines for the Company X sales forecasting system. These pipelines coordinate the end-to-end workflow from raw feature data to production predictions with explainability insights.

The forecasting system uses a dual-model architecture:
- **Model A**: Business lever features for explainability (e.g., awareness metrics, marketing spend)
- **Model B**: Full autoregressive features for high-accuracy predictions (e.g., lagged sales, seasonality)

The system predicts five key metrics across two channels (B&M and WEB):
- **Sales**: Total revenue
- **AOV**: Average Order Value
- **Orders**: Number of transactions
- **Conversion**: Order rate from traffic (B&M only)
- **Traffic**: Store footfall estimates (B&M only)

## Files

### Core Pipeline Modules

| File | Purpose | Key Components |
|------|---------|----------------|
| `__init__.py` | Public API and module exports | Exposes `train()`, `infer()`, `evaluate()` convenience functions and pipeline classes |
| `training.py` | End-to-end model training pipeline | `TrainingPipeline`, `TrainingResult`, `ChannelTrainingResult` |
| `inference.py` | Production scoring with saved models | `InferencePipeline`, `InferenceResult` |
| `evaluation.py` | Model evaluation and metrics computation | `EvaluationPipeline`, `EvaluationResult`, `MetricResult` |

### Detailed File Descriptions

#### `__init__.py`
Module initialization and public API definition. Exports all pipeline classes, result containers, and convenience functions for easy imports.

**Key Exports**:
- Training: `TrainingPipeline`, `TrainingResult`, `ChannelTrainingResult`, `train()`
- Inference: `InferencePipeline`, `InferenceResult`, `infer()`
- Evaluation: `EvaluationPipeline`, `EvaluationResult`, `MetricResult`, `evaluate()`, `compute_metrics()`

#### `training.py`
Orchestrates the complete model training workflow including data splitting, model training, surrogate explainability, and checkpoint management.

**Key Features**:
- Channel-specific training for B&M and WEB
- Multi-objective CatBoost models for Sales, AOV, and Orders
- Separate conversion model for B&M channel
- Traffic estimation via quantile models
- RMSE computation for quantile calibration
- Surrogate model training for SHAP-based explainability
- Bias correction options (lognormal mean adjustment)
- Time-based train/test splitting
- Checkpoint saving for production deployment

**Workflow Steps**:
1. Load and parse Model B (autoregressive) data
2. Split data by time into train/test sets
3. Train channel-specific predictors (BMPredictor, WEBPredictor)
4. Generate predictions on test set
5. Compute RMSE statistics for quantile estimation
6. Apply bias correction and compute quantiles (P50, P90, mean)
7. Train surrogate models using Model A features (optional)
8. Generate SHAP explanations with contributor analysis
9. Save models, residual stats, and predictions to checkpoints

**B&M Channel Specifics**:
- Multi-objective model: log(Sales), log(AOV), log(Orders)
- Conversion model: logit(Conversion) = Orders / Traffic
- Traffic estimator: P10, P50, P90 quantiles
- Surrogate cohorts: profit_center_nbr, dma, is_outlet, is_comp_store

**WEB Channel Specifics**:
- Multi-objective model: log(Sales), log(AOV), log(Orders)
- No conversion or traffic models
- Surrogate cohorts: profit_center_nbr, dma

#### `inference.py`
Loads trained models from checkpoints and generates predictions on new data with quantiles and optional explainability.

**Key Features**:
- Model checkpoint loading (CatBoost models and metadata)
- Residual statistics loading for quantile computation
- Channel-specific scoring
- Bias correction application
- Quantile prediction (P50, P90, mean)
- Traffic estimation for B&M
- Surrogate explainability (SHAP contributors) when Model A data provided

**Workflow Steps**:
1. Load model checkpoints from saved directory
2. Load residual statistics (RMSE values) for quantile calibration
3. Load surrogate models if explainability enabled
4. Filter input data by channel
5. Generate log-scale predictions using loaded models
6. Transform to business scale with exponential
7. Apply bias correction (optional)
8. Compute quantiles using saved RMSE values
9. Generate SHAP explanations via surrogate models (optional)
10. Save predictions to output directory

**Output Columns**:
- Log-scale: `pred_log_sales`, `pred_log_aov`, `pred_log_orders`, `pred_logit_conversion` (B&M)
- Business-scale quantiles: `pred_sales_mean`, `pred_sales_p50`, `pred_sales_p90`, etc.
- Traffic quantiles (B&M): `pred_traffic_p10`, `pred_traffic_p50`, `pred_traffic_p90`
- Contributors (optional): top feature importance via SHAP

#### `evaluation.py`
Evaluates model predictions against ground truth using multiple metrics across both log-scale and business-scale targets.

**Key Features**:
- Multiple evaluation metrics: MAE, WMAPE, Bias, R2
- Log-scale evaluation (direct model output quality)
- Business-scale evaluation (back-transformed predictions)
- Channel-specific metric computation
- Support for quantile predictions (mean, median, P50, P90)
- Conversion and traffic evaluation (B&M)
- Formatted summary tables and CSV export

**Metrics Computed**:
- **MAE**: Mean Absolute Error (same units as target)
- **WMAPE**: Weighted Mean Absolute Percentage Error (percentage)
- **Bias**: Mean Error (signed, indicates over/under prediction)
- **R2**: Coefficient of Determination (goodness of fit, 0-1 scale)

**Targets Evaluated**:
1. Log-scale: log(Sales), log(AOV), logit(Conversion)
2. Business-scale: Sales (mean/median), AOV (mean/median), Orders, Conversion, Traffic

**Workflow Steps**:
1. Load predictions DataFrame (contains both predictions and actuals)
2. Filter by channel if specified
3. Compute metrics for each target
4. Handle missing values and edge cases (zero division, constant targets)
5. Aggregate metrics across channels
6. Generate summary DataFrame and print formatted table
7. Save evaluation summary to CSV

## Pipeline Orchestration

The pipelines implement a modular, end-to-end forecasting workflow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Data Loading                                                     │
│    - Load Model B CSV (autoregressive features)                     │
│    - Load Model A CSV (business lever features, optional)           │
│    - Parse dates and validate schema                                │
│                                                                      │
│ 2. Data Splitting                                                   │
│    - Time-based split (train/test by date threshold)                │
│    - Separate processing for B&M and WEB channels                   │
│                                                                      │
│ 3. Model Training                                                   │
│    - Train CatBoost multi-objective regressors                      │
│    - Train B&M conversion model (logit)                             │
│    - Validate on test set                                           │
│                                                                      │
│ 4. Prediction & Calibration                                         │
│    - Generate predictions on test set                               │
│    - Compute RMSE for each target                                   │
│    - Estimate traffic quantiles (B&M)                               │
│    - Apply bias correction                                          │
│                                                                      │
│ 5. Surrogate Explainability                                         │
│    - Train surrogate models on Model A features                     │
│    - Generate SHAP values and contributors                          │
│    - Produce dependence plots and summary visualizations            │
│                                                                      │
│ 6. Checkpoint Management                                            │
│    - Save CatBoost models (.cbm files)                              │
│    - Save residual statistics (residual_stats.json)                 │
│    - Save surrogate models and metadata                             │
│    - Save predictions with contributors (predictions_*.csv)         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Model Loading                                                    │
│    - Load CatBoost checkpoints from training run                    │
│    - Load residual statistics for quantile computation              │
│    - Load surrogate models if explainability enabled                │
│                                                                      │
│ 2. Data Preparation                                                 │
│    - Load Model B CSV (new data to score)                           │
│    - Load Model A CSV (optional, for explainability)                │
│    - Parse dates and validate features                              │
│                                                                      │
│ 3. Prediction Generation                                            │
│    - Score data with loaded models by channel                       │
│    - Generate log-scale predictions                                 │
│    - Transform to business scale with exponential                   │
│    - Apply bias correction if configured                            │
│                                                                      │
│ 4. Quantile Computation                                             │
│    - Use saved RMSE values from training                            │
│    - Compute P50 (median), P90, and bias-corrected mean             │
│    - Estimate traffic quantiles (B&M)                               │
│                                                                      │
│ 5. Explainability (Optional)                                        │
│    - Apply surrogate models to new data                             │
│    - Generate SHAP contributors for each prediction                 │
│    - Merge contributors back to predictions                         │
│                                                                      │
│ 6. Output Generation                                                │
│    - Save predictions to CSV (predictions_bm.csv, predictions_web.csv) │
│    - Include log-scale, quantiles, and contributors                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       EVALUATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Data Loading                                                     │
│    - Load predictions CSV with actuals                              │
│    - Validate required columns exist                                │
│                                                                      │
│ 2. Metric Computation                                               │
│    - Compute MAE, WMAPE, Bias, R2 for each target                   │
│    - Handle log-scale and business-scale targets                    │
│    - Filter to valid (non-NaN) pairs                                │
│                                                                      │
│ 3. Multi-Target Evaluation                                          │
│    - Evaluate Sales (mean, median)                                  │
│    - Evaluate AOV (mean, median)                                    │
│    - Evaluate Orders, Conversion, Traffic                           │
│    - Evaluate log-space predictions                                 │
│                                                                      │
│ 4. Summary Generation                                               │
│    - Aggregate metrics by channel                                   │
│    - Create formatted summary table                                 │
│    - Save evaluation_summary.csv                                    │
│    - Print results to console                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Pipeline Stages

### 1. Data Loading and Preparation
- Loads feature data from CSV files (Model A and Model B)
- Parses temporal columns (origin_week_date)
- Validates required columns and data types
- Filters by channel (B&M vs WEB)

### 2. Feature Engineering
- Performed by external modules (not in pipelines folder)
- Model B: Autoregressive lags, seasonality, trends, dynamics features
- Model A: Business levers (awareness, marketing, pricing, inventory)
- Geography features: DMA, sister store/DMA relationships

### 3. Training Stage
- Time-based train/test split using config thresholds
- Multi-objective CatBoost training:
  - B&M: 3 targets (Sales, AOV, Orders) + 1 separate (Conversion)
  - WEB: 3 targets (Sales, AOV, Orders)
- Early stopping on validation set
- RMSE computation for quantile calibration
- Traffic estimation using historical quantile models (B&M)

### 4. Explainability Stage
- Trains surrogate models on Model A features
- Maps Model A features to Model B predictions
- Generates SHAP values and feature importance
- Identifies top contributors per prediction
- Produces visualizations:
  - Summary plots (global feature importance)
  - Dependence plots (feature interactions)
  - Cohort analysis (segment-specific patterns)

### 5. Inference Stage
- Loads trained models from checkpoints
- Scores new data without ground truth
- Applies saved RMSE for quantile computation
- Generates probabilistic forecasts (quantiles)
- Optional explainability via surrogate models

### 6. Evaluation Stage
- Computes performance metrics on test/validation data
- Multi-scale evaluation (log-space and business-space)
- Quantile evaluation (mean vs median vs P90)
- Channel-specific aggregation
- Summary reporting and visualization

## Model Architecture

### Dual-Model System

**Model B (Prediction Model)**:
- Purpose: High-accuracy forecasting with all available signals
- Features: Autoregressive lags, seasonality, trends, dynamics, external factors
- Training: CatBoost multi-objective regression
- Output: Log-scale predictions (log Sales, log AOV, log Orders, logit Conversion)
- Use case: Production forecasting, automated decision-making

**Model A (Explainability Model)**:
- Purpose: Interpretable predictions using business levers
- Features: Awareness, marketing spend, pricing, inventory, store attributes
- Training: Surrogate model trained to mimic Model B predictions
- Output: SHAP contributors showing feature impact
- Use case: Business insights, "what-if" scenario analysis, stakeholder communication

### Surrogate Explainability

The surrogate approach enables explainability without sacrificing prediction accuracy:

1. Train Model B with all features for best accuracy
2. Train Model A surrogate to approximate Model B's predictions using only business levers
3. Apply SHAP to Model A surrogate (fast and interpretable)
4. SHAP values explain how each business lever impacts the Model B prediction
5. High surrogate R2 (typically >0.9) validates that business levers capture most signal

**Benefits**:
- Get Model B's accuracy in production
- Get Model A's interpretability for stakeholders
- No trade-off between accuracy and explainability
- Fast SHAP computation (surrogate is simple)

## Configuration

Pipelines are configured via dataclasses in `forecasting.regressor.configs`:

### TrainingConfig
- `output_dir`: Where to save results
- `checkpoint_dir`: Where to save model files
- `channels`: List of channels to train (["B&M", "WEB"])
- `data_split`: Train/test date thresholds
- `bias_correction`: Lognormal bias correction flags
- `train_surrogate`: Whether to train explainability models
- `top_k_contributors`: Number of top features to track per prediction
- `key_columns`: Joining keys (profit_center_nbr, origin_week_date, etc.)

### InferenceConfig
- `checkpoint_dir`: Where to load model files from
- `output_dir`: Where to save predictions
- `channels`: List of channels to score
- `bias_correction`: Whether to apply bias correction
- `run_explainability`: Whether to generate SHAP contributors
- `key_columns`: Joining keys for Model A merge
- `top_k_contributors`: Number of contributors to include

### EvaluationConfig
- `output_dir`: Where to save evaluation results
- Metrics computed: MAE, WMAPE, Bias, R2

### BiasCorrection
- `correct_bm`: Apply lognormal correction to B&M (Sales, AOV)
- `correct_web`: Apply lognormal correction to WEB (Sales, AOV)
- `correct_web_sales`: Apply correction to WEB Sales only
- `correct_web_aov`: Apply correction to WEB AOV only

Bias correction adjusts for lognormal transformation bias: E[exp(X)] = exp(E[X] + 0.5*Var[X])

## Usage Examples

### Training Pipeline

```python
from forecasting.regressor.pipelines import train

# Basic training
result = train(
    model_b_path="data/model_b_features.csv",
    model_a_path="data/model_a_features.csv",
    output_dir="output",
)

# Training with bias correction
result = train(
    model_b_path="data/model_b_features.csv",
    model_a_path="data/model_a_features.csv",
    output_dir="output",
    correct_bm=True,
    correct_web_sales=True,
    train_surrogate=True,
)

# Access training metrics
print(f"B&M RMSE Sales: {result.bm_result.rmse_sales:.4f}")
print(f"B&M Surrogate R2: {result.bm_result.surrogate_r2:.4f}")
print(f"WEB RMSE Sales: {result.web_result.rmse_sales:.4f}")
```

### Inference Pipeline

```python
from forecasting.regressor.pipelines import infer

# Basic inference
result = infer(
    model_b_path="data/future_features.csv",
    checkpoint_dir="output/checkpoints",
    output_dir="output_infer",
)

# Inference with explainability
result = infer(
    model_b_path="data/future_features.csv",
    model_a_path="data/future_model_a.csv",
    checkpoint_dir="output/checkpoints",
    output_dir="output_infer",
    run_explainability=True,
    correct_bm=True,
)

# Access predictions
print(result.bm_predictions.head())
print(result.web_predictions.head())
```

### Evaluation Pipeline

```python
from forecasting.regressor.pipelines import evaluate

# Evaluate both channels
result = evaluate(
    predictions_bm_path="output/predictions_bm.csv",
    predictions_web_path="output/predictions_web.csv",
    output_dir="output",
)

# Print summary
result.print_summary()

# Save to CSV
result.save()

# Access specific metrics
df = result.to_dataframe()
bm_sales_mae = df[
    (df["channel"] == "B&M") & (df["target"].str.contains("Sales"))
]["mae"].values[0]
print(f"B&M Sales MAE: {bm_sales_mae:.2f}")
```

### Advanced Usage with Pipeline Classes

```python
from forecasting.regressor.pipelines import (
    TrainingPipeline,
    InferencePipeline,
    EvaluationPipeline,
)
from forecasting.regressor.configs import TrainingConfig, InferenceConfig
import pandas as pd
from pathlib import Path

# Load data
model_b_df = pd.read_csv("data/model_b.csv")
model_a_df = pd.read_csv("data/model_a.csv")

# Configure and train
config = TrainingConfig(
    output_dir=Path("output"),
    channels=["B&M", "WEB"],
    train_surrogate=True,
)

trainer = TrainingPipeline(config)
train_result = trainer.run(model_b_df, model_a_df)

# Configure and run inference
inference_config = InferenceConfig(
    checkpoint_dir=Path("output/checkpoints"),
    output_dir=Path("output_infer"),
    run_explainability=True,
)

inferrer = InferencePipeline(inference_config)
infer_result = inferrer.run(model_b_df, model_a_df)

# Evaluate
evaluator = EvaluationPipeline()
eval_result = evaluator.run(infer_result.bm_predictions)
eval_result.print_summary()
```

## Output Files

### Training Outputs

**Predictions**:
- `output/predictions_bm.csv`: B&M test set predictions with actuals and contributors
- `output/predictions_web.csv`: WEB test set predictions with actuals and contributors

**Checkpoints**:
- `output/checkpoints/bm_multi.cbm`: B&M multi-objective CatBoost model
- `output/checkpoints/bm_conversion.cbm`: B&M conversion CatBoost model
- `output/checkpoints/web_multi.cbm`: WEB multi-objective CatBoost model
- `output/checkpoints/surrogate_bm.cbm`: B&M surrogate model for explainability
- `output/checkpoints/surrogate_web.cbm`: WEB surrogate model for explainability
- `output/checkpoints/surrogate_bm.meta.json`: B&M surrogate metadata (features, R2)
- `output/checkpoints/surrogate_web.meta.json`: WEB surrogate metadata
- `output/checkpoints/residual_stats.json`: RMSE values for quantile computation

**Explainability**:
- `output/BM_Sales_summary.png`: SHAP summary plot for B&M
- `output/BM_Sales_dependence_*.png`: SHAP dependence plots for top features
- `output/WEB_Sales_summary.png`: SHAP summary plot for WEB
- `output/WEB_Sales_dependence_*.png`: SHAP dependence plots for top features

### Inference Outputs

**Predictions**:
- `output_infer/predictions_bm.csv`: B&M predictions with quantiles and contributors
- `output_infer/predictions_web.csv`: WEB predictions with quantiles and contributors

**Explainability** (if enabled):
- `output_infer/BM_Sales_summary.png`: SHAP summary plot
- `output_infer/WEB_Sales_summary.png`: SHAP summary plot
- Additional dependence plots

### Evaluation Outputs

**Metrics**:
- `output/evaluation_summary.csv`: Comprehensive metrics table (MAE, WMAPE, Bias, R2)

## Dependencies

The pipelines depend on other modules in the forecasting system:

- `forecasting.regressor.configs`: Configuration dataclasses
- `forecasting.regressor.models`: Predictor classes (BMPredictor, WEBPredictor, SurrogateExplainer)
- `forecasting.regressor.features`: Feature engineering modules (not directly imported)
- `forecasting.regressor.explainability`: SHAP analysis and visualization

External packages:
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `catboost`: Gradient boosting models
- `shap`: Model explainability
- `matplotlib`: Visualization (via explainability module)

## Design Principles

1. **Separation of Concerns**: Each pipeline has a single responsibility (train, infer, evaluate)
2. **Configuration-Driven**: Behavior controlled via config objects, not hardcoded
3. **Reproducibility**: Time-based splits, checkpoint management, saved statistics
4. **Production-Ready**: Checkpoints enable deployment without retraining
5. **Explainability First**: Surrogate models provide interpretability without accuracy loss
6. **Channel Isolation**: B&M and WEB trained separately to capture channel-specific patterns
7. **Uncertainty Quantification**: Quantile predictions enable risk-aware decision-making
8. **Modularity**: Convenience functions for simple cases, classes for customization

## Future Enhancements

Potential extensions planned or under consideration:

- Rolling origin cross-validation (per ENGINEERING_INSTRUCTIONS.md)
- Hyperparameter tuning pipelines
- Ensemble methods (combine Model A and Model B)
- Online learning / incremental updates
- Multi-horizon optimization
- Conformal prediction for calibrated intervals
- Automated retraining triggers (data drift detection)
- Model monitoring and performance tracking
- A/B testing framework for model versions
