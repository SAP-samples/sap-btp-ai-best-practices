# ML Model Documentation

## Overview

This document describes the machine learning model architecture, training pipeline, and inference process for the sales forecasting regressor. The system uses CatBoost gradient boosting with channel-specific models and multi-objective optimization.

### Key Characteristics

- **Framework**: CatBoost gradient boosting
- **Channels**: B&M (Brick & Mortar) and WEB
- **Targets**: Sales, AOV, Orders, Conversion (B&M only)
- **Multi-objective**: Sales, AOV, and Orders predicted simultaneously
- **Explainability**: Surrogate models with SHAP attribution

---

## Model Architecture

### Channel-Specific Models

The system trains separate models for each channel due to their different characteristics:

```
B&M Channel                              WEB Channel
    |                                        |
    +-> BMMultiObjectivePredictor            +-> WEBMultiObjectivePredictor
    |   (Sales, AOV, Orders)                 |   (Sales, AOV, Orders)
    |                                        |
    +-> BMConversionPredictor                +-> (No conversion model)
    |   (Conversion)                         |
    |                                        |
    +-> TrafficEstimator                     |
        (Derived from predictions)           |
    |                                        |
    v                                        v
BMPredictor (Facade)                   WEBPredictor (Facade)
```

### B&M Channel Models

**1. BMMultiObjectivePredictor** (`models/bm_predictor.py`)
- **Targets**: Sales, AOV, Orders (3 simultaneous outputs)
- **Loss Function**: MultiRMSE (multi-target RMSE)
- **Purpose**: Predicts the core business metrics

```python
# Target computation during training
target_cols = ["label_log_sales", "label_log_aov", "label_log_orders"]
y_train = train_subset[target_cols].values  # [n_samples, 3]
```

**2. BMConversionPredictor** (`models/bm_predictor.py`)
- **Target**: Conversion (logit-transformed)
- **Loss Function**: RMSE (single target)
- **Filter**: Only B&M rows with `has_traffic_data=1`
- **Purpose**: Predicts store conversion rate

```python
# Only train on stores with valid traffic data
mask = train_df["label_logit_conversion"].notna() & (train_df["has_traffic_data"] == 1)
train_subset = train_df[mask].copy()
```

**3. TrafficEstimator** (`models/traffic.py`)
- **Computation**: Traffic = Sales / (AOV * Conversion)
- **Purpose**: Derives traffic estimates from model predictions
- **Quantiles**: Provides p10, p50, p90 estimates

**4. BMPredictor** (`models/bm_predictor.py`)
- **Type**: Facade pattern
- **Purpose**: Combines multi-objective and conversion predictors
- **Output**: Complete B&M predictions with all metrics

### WEB Channel Models

**1. WEBMultiObjectivePredictor** (`models/web_predictor.py`)
- **Targets**: Sales, AOV, Orders (3 simultaneous outputs)
- **Loss Function**: MultiRMSE
- **Note**: No conversion model (no physical traffic)

**2. WEBPredictor** (`models/web_predictor.py`)
- **Type**: Facade pattern
- **Purpose**: Manages WEB predictions

### Shared Base Class

**BaseCatBoostPredictor** (`models/base.py`):
- Feature preparation and channel-specific filtering
- Categorical feature handling (CatBoost native support)
- Model persistence (save/load)
- Excludes channel-inappropriate features:
  - B&M models exclude WEB-only features
  - WEB models exclude B&M-only features (conversion, staffing, cannibalization)

---

## CatBoost Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `iterations` | 5000 | Number of boosting iterations |
| `learning_rate` | 0.05 | Step size for gradient descent |
| `depth` | 6 | Maximum tree depth |
| `loss_function` | MultiRMSE / RMSE | Multi-target or single-target |
| `random_seed` | 42 | Reproducibility |
| `early_stopping_rounds` | 100 | Stop if no improvement |

### Multi-Objective Output

The MultiRMSE loss function enables simultaneous prediction of multiple targets:

```python
self.model = CatBoostRegressor(
    iterations=self.iterations,
    learning_rate=self.learning_rate,
    depth=self.depth,
    loss_function="MultiRMSE",
    eval_metric="MultiRMSE",
    ...
)

# Output shape: [n_samples, 3] for (log_sales, log_aov, log_orders)
preds = self.model.predict(X)
log_sales = preds[:, 0]
log_aov = preds[:, 1]
log_orders = preds[:, 2]
```

---

## Training Pipeline

The training pipeline is orchestrated by `TrainingPipeline` in `pipelines/training.py`.

### Pipeline Flow

```
Input: Model B Data (full features)
       Model A Data (optional, for explainability)
       |
       v
1. Parse dates (origin_week_date)
       |
       v
2. For each channel (B&M, WEB):
   |
   +-> Filter to channel data
   |
   +-> Split train/test (time-based)
   |
   +-> Train channel predictor
   |
   +-> Compute RMSE on validation set
   |
   +-> Save predictions to CSV
   |
   +-> Save model checkpoints (.cbm)
   |
   +-> Train surrogate model (if Model A provided)
   |
   v
3. Save residual_stats.json
       |
       v
Output: TrainingResult with metrics
```

### Data Splitting

The pipeline uses time-based splitting configured via `TrainingConfig.data_split`:

```python
train_mask = self.config.data_split.get_train_mask(df_b_bm)
test_mask = self.config.data_split.get_test_mask(df_b_bm)
train_df = df_b_bm[train_mask]
test_df = df_b_bm[test_mask]
```

### RMSE Computation

RMSE values are computed on the validation set and used for:
1. Quantile estimation (p50, p90)
2. Bias correction
3. Traffic estimation

```python
def _compute_rmse(self, df: pd.DataFrame) -> None:
    # Sales RMSE
    valid_sales = df["label_log_sales"].notna()
    self.rmse_sales = np.sqrt(np.mean(
        (preds_multi.log_sales[valid_sales] - df.loc[valid_sales, "label_log_sales"])**2
    ))

    # AOV RMSE
    valid_aov = df["label_log_aov"].notna()
    self.rmse_aov = np.sqrt(np.mean(
        (preds_multi.log_aov[valid_aov] - df.loc[valid_aov, "label_log_aov"])**2
    ))

    # Conversion RMSE (B&M only)
    valid_conv = df["label_logit_conversion"].notna()
    self.rmse_conv = np.sqrt(np.mean(
        (preds_conv[valid_conv] - df.loc[valid_conv, "label_logit_conversion"])**2
    ))
```

### Checkpoint Saving

Models and metadata are saved to the checkpoint directory:

| File | Description |
|------|-------------|
| `bm_multi.cbm` | B&M multi-objective model |
| `bm_conversion.cbm` | B&M conversion model |
| `web_multi.cbm` | WEB multi-objective model |
| `surrogate_bm.cbm` | B&M surrogate for explainability |
| `surrogate_web.cbm` | WEB surrogate for explainability |
| `residual_stats.json` | RMSE values for quantile computation |

---

## Surrogate Models for Explainability

### Purpose

Surrogate models approximate Model B predictions using only Model A features (actionable business levers). This enables SHAP-based feature attribution without lag dominance.

### Architecture

```
Model B Predictions (from full feature model)
       |
       v
Surrogate Model (CatBoost)
   - Input: Model A features (29 actionable features)
   - Target: Model B log_sales predictions
       |
       v
SHAP Values
   - Feature importance
   - Per-prediction attributions
   - Dependence plots
```

### Training Process

```python
self.bm_surrogate = SurrogateExplainer(channel="B&M")
self.bm_surrogate.fit(
    model_a_df=explain_df,
    model_b_predictions=explain_df["pred_log_sales"].values
)
```

### SHAP Analysis Output

The surrogate generates:
1. **Feature importance**: Global ranking of features
2. **Top contributors**: Per-row attribution (top K features)
3. **Dependence plots**: Feature value vs SHAP value
4. **Cohort analysis**: Aggregated SHAP by store/DMA/type

---

## Inference Pipeline

The inference pipeline is managed by `InferencePipeline` in `pipelines/inference.py`.

### Pipeline Flow

```
Input: Model B Features (new data)
       Model A Features (optional)
       Checkpoint Directory
       |
       v
1. Load model checkpoints
       |
       v
2. Load residual statistics
       |
       v
3. For each channel:
   |
   +-> Generate predictions
   |
   +-> Apply bias correction (optional)
   |
   +-> Compute quantiles (p50, p90)
   |
   +-> Estimate traffic (B&M only)
   |
   +-> Run explainability (optional)
   |
   v
Output: InferenceResult with predictions
```

### Prediction Generation

```python
# Multi-objective predictions
preds_multi = self.multi_predictor.predict(df)
log_sales = preds_multi.log_sales
log_aov = preds_multi.log_aov
log_orders = preds_multi.log_orders

# Conversion prediction (B&M only)
preds_conv = self.conv_predictor.predict(df)
```

### Quantile Computation

Quantiles are computed using the log-normal assumption:

```python
z90 = 1.2815515655446004  # 90th percentile z-score

# Median (p50) - no bias correction
pred_sales_p50 = np.exp(log_sales)

# 90th percentile
pred_sales_p90 = np.exp(log_sales + z90 * rmse_sales)

# Mean (with bias correction)
sigma_sales = 0.5 * rmse_sales**2 if bias_correction else 0.0
pred_sales_mean = np.exp(log_sales + sigma_sales)
```

### Bias Correction

Log-normal bias correction adjusts for the mean-median discrepancy:

```python
# Without correction: E[exp(X)] is biased
# With correction: E[exp(X)] = exp(mu + sigma^2/2)

if bias.should_correct_bm():
    sigma_sales = 0.5 * preds.rmse_sales**2
    res_df["pred_sales_mean"] = np.exp(preds.log_sales + sigma_sales)
```

### Traffic Estimation (B&M Only)

Traffic is derived from the relationship: Orders = Traffic * Conversion

```python
# Traffic = Sales / (AOV * Conversion)
traffic = self.traffic_estimator.estimate(
    log_sales_pred=preds_multi.log_sales,
    log_aov_pred=preds_multi.log_aov,
    logit_conv_pred=preds_conv,
    sales_rmse=self.rmse_sales,
    aov_rmse=self.rmse_aov,
    conv_rmse=self.rmse_conv,
)
```

---

## Evaluation

The evaluation pipeline computes standard regression metrics in `pipelines/evaluation.py`.

### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| MAE | `mean(abs(y - y_pred))` | Mean Absolute Error |
| WMAPE | `sum(abs(y - y_pred)) / sum(abs(y))` | Weighted Mean Absolute Percentage Error |
| Bias | `mean(y_pred - y)` | Systematic over/under prediction |
| R2 | `1 - SS_res / SS_tot` | Coefficient of determination |

### Channel-Specific Summaries

Metrics are computed separately for:
- B&M Sales, AOV, Conversion
- WEB Sales, AOV

---

## Pipeline Entry Point

The unified entry point is `scripts/run_pipeline.py` with four subcommands:

### 1. Generate (`generate`)

Builds the canonical training table with features.

```bash
python -m app.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --model both \
    --output data/
```

**Options**:
- `--horizons START END`: Forecast horizon range (default: 1 52)
- `--model {A,B,both}`: Feature variant to generate
- `--include-crm`: Include CRM demographic features
- `--output`: Output directory or CSV path

### 2. Train (`train`)

Trains channel models and optional surrogates.

```bash
python -m app.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --model-a data/model_a.csv \
    --output output/
```

**Options**:
- `--model-b`: Path to Model B CSV (required)
- `--model-a`: Path to Model A CSV (optional, for explainability)
- `--channels`: Channels to train (default: bm web)
- `--correct-bm`, `--correct-web`: Bias correction flags
- `--no-surrogate`: Skip surrogate model training

### 3. Infer (`infer`)

Generates predictions using saved models.

```bash
python -m app.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/
```

**Options**:
- `--model-b`: Path to Model B feature CSV (required)
- `--model-a`: Path to Model A CSV (optional)
- `--checkpoints`: Directory with saved models
- `--channels`: Channels to process
- `--no-explainability`: Skip SHAP analysis

### 4. Evaluate (`evaluate`)

Computes metrics comparing predictions to ground truth.

```bash
python -m app.regressor.scripts.run_pipeline evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --predictions-web predictions/predictions_web.csv \
    --output output/
```

**Options**:
- `--predictions-bm`: Path to B&M predictions CSV
- `--predictions-web`: Path to WEB predictions CSV
- `--output`: Output directory for evaluation summary

---

## Prediction Output Format

### B&M Predictions

| Column | Description |
|--------|-------------|
| `pred_log_sales` | Log-scale sales prediction |
| `pred_log_aov` | Log-scale AOV prediction |
| `pred_log_orders` | Log-scale orders prediction |
| `pred_logit_conversion` | Logit-scale conversion prediction |
| `pred_sales_mean` | Sales mean (with bias correction) |
| `pred_sales_p50` | Sales median |
| `pred_sales_p90` | Sales 90th percentile |
| `pred_aov_mean` | AOV mean |
| `pred_aov_p50` | AOV median |
| `pred_aov_p90` | AOV 90th percentile |
| `pred_traffic_p10` | Traffic 10th percentile |
| `pred_traffic_p50` | Traffic median |
| `pred_traffic_p90` | Traffic 90th percentile |

### WEB Predictions

| Column | Description |
|--------|-------------|
| `pred_log_sales` | Log-scale sales prediction |
| `pred_log_aov` | Log-scale AOV prediction |
| `pred_log_orders` | Log-scale orders prediction |
| `pred_sales_mean` | Sales mean |
| `pred_sales_p50` | Sales median |
| `pred_sales_p90` | Sales 90th percentile |
| `pred_aov_mean` | AOV mean |
| `pred_aov_p50` | AOV median |
| `pred_aov_p90` | AOV 90th percentile |

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/run_pipeline.py` | Unified CLI entry point |
| `pipelines/training.py` | Training orchestration |
| `pipelines/inference.py` | Inference orchestration |
| `pipelines/evaluation.py` | Metrics computation |
| `models/base.py` | Base CatBoost predictor |
| `models/bm_predictor.py` | B&M channel models |
| `models/web_predictor.py` | WEB channel models |
| `models/traffic.py` | Traffic estimation |
| `models/surrogate.py` | Surrogate explainability model |
| `configs/model_config.py` | Model hyperparameters |
| `configs/training_config.py` | Training configuration |
| `configs/base.py` | Bias correction settings |
| `explainability/shap_analysis.py` | SHAP computation |
| `explainability/contributors.py` | Top contributor extraction |
