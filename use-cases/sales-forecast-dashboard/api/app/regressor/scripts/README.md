# Forecasting Regressor Scripts

This directory contains the complete command-line interface for the sales forecasting system. The scripts provide a unified pipeline for generating training data, training models, running inference, and evaluating predictions.

## Overview

The forecasting system predicts 5 key metrics (Sales, AOV, Orders, Conversion, Traffic) across 2 channels (B&M and WEB) using two model variants:

- **Model A**: Actionable/Explainability model with business levers only (41 features, no autoregressive features)
- **Model B**: Production model with full feature set including lags and rolling aggregations (57 features)

## Architecture

There are two ways to use these scripts:

1. **Unified Pipeline** (Recommended): Use `run_pipeline.py` with subcommands
2. **Individual Scripts**: Run each script separately for fine-grained control

All scripts are designed to be run as Python modules from the project root.

## Quick Start

### End-to-End Workflow

```bash
# 1. Generate training data
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --model both \
    --output data/training.csv

# 2. Train models
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/training_model_b.csv \
    --model-a data/training_model_a.csv \
    --output output/ \
    --channels bm web

# 3. Run inference
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# 4. Evaluate predictions
python -m forecasting.regressor.scripts.run_pipeline evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --predictions-web predictions/predictions_web.csv
```

## Script Reference

| Script | Purpose | Entry Point | Key Parameters |
|--------|---------|-------------|----------------|
| `run_pipeline.py` | Unified entry point with subcommands for all operations | `python -m forecasting.regressor.scripts.run_pipeline {generate\|train\|infer\|evaluate}` | Subcommand-specific |
| `generate_training_sample.py` | Generate canonical training tables with features | `python -m forecasting.regressor.scripts.generate_training_sample` | `--model-variant`, `--horizons`, `--output` |
| `train.py` | Train forecasting models on prepared data | `python -m forecasting.regressor.scripts.train` | `--model-b`, `--model-a`, `--output`, `--channels` |
| `infer.py` | Run inference with saved models | `python -m forecasting.regressor.scripts.infer` | `--model-b`, `--checkpoints`, `--output` |
| `evaluate.py` | Evaluate predictions against ground truth | `python -m forecasting.regressor.scripts.evaluate` | `--predictions-bm`, `--predictions-web` |

---

## Detailed Usage

### 1. run_pipeline.py - Unified Pipeline Entry Point

The recommended way to use the forecasting system. Provides a unified interface with subcommands.

#### Generate Subcommand

Generate canonical training tables with features.

```bash
# Generate both Model A and Model B
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --model both \
    --output data/training.csv

# Generate only Model B for specific horizons
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 13 \
    --model B \
    --output data/training_q1.csv

# Include CRM demographic features
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --model both \
    --include-crm \
    --output data/training_with_crm.csv
```

**Parameters:**
- `--horizons START END`: Horizon range (default: 1 52)
  - Examples: `1 4` (near-term), `1 13` (quarter), `1 52` (full year)
- `--model {A,B,both}`: Generate Model A, B, or both feature sets (default: both)
- `--output PATH`: Output CSV path (required)
- `--include-crm`: Include CRM demographic features

**Outputs:**
- If `--model both`: Creates `{output}_model_a.csv` and `{output}_model_b.csv`
- Otherwise: Creates single CSV file

#### Train Subcommand

Train forecasting models on prepared feature data.

```bash
# Train both channels with bias correction
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --model-a data/model_a.csv \
    --output output/ \
    --channels bm web \
    --correct-bm

# Train only B&M channel without surrogate model
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --output output/ \
    --channels bm \
    --no-surrogate

# Train with WEB sales bias correction only
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --output output/ \
    --channels web \
    --correct-web-sales
```

**Parameters:**
- `--model-b PATH`: Path to Model B CSV with full features (required)
- `--model-a PATH`: Path to Model A CSV for explainability (optional)
- `--output DIR`: Output directory (default: output)
- `--channels {bm,web}`: Channels to train (default: bm web)
- `--correct-bm`: Apply bias correction to B&M predictions
- `--correct-web`: Apply bias correction to all WEB predictions
- `--correct-web-sales`: Apply bias correction to WEB Sales only
- `--correct-web-aov`: Apply bias correction to WEB AOV only
- `--no-surrogate`: Skip surrogate model training for explainability

**Outputs:**
- `{output}/checkpoints/`: Saved model checkpoints
- `{output}/metrics/`: Training metrics and evaluation results
- `{output}/explainability/`: SHAP values and feature importance (if Model A provided)

#### Infer Subcommand

Run inference with saved models on new data.

```bash
# Run inference on both channels
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# Run inference with explainability
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features_b.csv \
    --model-a data/new_features_a.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# Run inference with bias correction
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/ \
    --correct-bm \
    --correct-web-sales
```

**Parameters:**
- `--model-b PATH`: Path to Model B feature CSV (required)
- `--model-a PATH`: Path to Model A feature CSV for explainability (optional)
- `--checkpoints DIR`: Directory with saved models (default: output/checkpoints)
- `--output DIR`: Output directory (default: output_infer)
- `--channels {bm,web}`: Channels to process (default: bm web)
- `--correct-bm`: Apply bias correction to B&M predictions
- `--correct-web`: Apply bias correction to all WEB predictions
- `--correct-web-sales`: Apply bias correction to WEB Sales only
- `--correct-web-aov`: Apply bias correction to WEB AOV only
- `--no-explainability`: Skip explainability analysis

**Outputs:**
- `{output}/predictions_bm.csv`: B&M channel predictions
- `{output}/predictions_web.csv`: WEB channel predictions
- `{output}/explainability/`: Feature attributions (if Model A provided and not skipped)

#### Evaluate Subcommand

Evaluate model predictions against ground truth.

```bash
# Evaluate both channels
python -m forecasting.regressor.scripts.run_pipeline evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --predictions-web predictions/predictions_web.csv \
    --output evaluation/

# Evaluate single channel
python -m forecasting.regressor.scripts.run_pipeline evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --output evaluation/
```

**Parameters:**
- `--predictions-bm PATH`: Path to B&M predictions CSV (at least one required)
- `--predictions-web PATH`: Path to WEB predictions CSV (at least one required)
- `--output DIR`: Output directory for evaluation summary (default: output)

**Metrics Computed:**
- MAE: Mean Absolute Error
- WMAPE: Weighted Mean Absolute Percentage Error
- Bias: Mean Error (signed)
- R2: Coefficient of Determination

**Outputs:**
- `{output}/evaluation_summary.txt`: Text summary of metrics
- `{output}/evaluation_metrics.csv`: Detailed metrics by channel and metric

---

### 2. generate_training_sample.py - Training Data Generation

Interactive/CLI tool for generating and inspecting canonical training tables. This script is useful for:
- Manual validation of feature engineering
- Creating samples for analysis
- Testing specific horizon ranges

#### Usage Modes

**Interactive Mode:**
```bash
python -m forecasting.regressor.scripts.generate_training_sample --interactive
```

This will prompt you for:
- Model variant (A or B)
- Horizon range (e.g., 1-4, 1-13, 1-52)
- Date filters (optional)
- Whether to include CRM features
- Output file path (optional)

**Command-Line Mode:**

```bash
# Quick test (Model B, horizons 1-2)
python -m forecasting.regressor.scripts.generate_training_sample -m B --horizons 1 2

# Full quarter forecast (horizons 1-13)
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B \
    --horizons 1 13 \
    --output data/training_model_b_q1.csv

# Filtered date range
python -m forecasting.regressor.scripts.generate_training_sample \
    -m A \
    --horizons 1 4 \
    --start-date 2024-07-01 \
    --end-date 2024-12-31 \
    --output data/training_model_a_h2.csv

# Full year with CRM features
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B \
    --horizons 1 52 \
    --include-crm \
    --output data/training_full_year.csv
```

#### Parameters

**Required:**
- `-m, --model-variant {A,B}`: Model variant
  - `A`: Actionable/Explainability (41 features, no autoregressive lags)
  - `B`: Production/Full (57 features, includes lags/rolls)
- `--horizons START END`: Horizon range (1-52)
  - Examples: `--horizons 1 4` (near-term), `--horizons 1 13` (quarter), `--horizons 1 52` (full year)

**Optional:**
- `--start-date YYYY-MM-DD`: Filter to origin dates >= this date
- `--end-date YYYY-MM-DD`: Filter to origin dates <= this date
- `--include-crm`: Include CRM demographic features
- `-o, --output PATH`: Save results to CSV file
- `--interactive`: Run in interactive mode with prompts

#### Output Summary

The script provides detailed diagnostics:

1. **Parameters Used**: Model variant, horizons, date filters
2. **Build Time**: Generation duration
3. **DataFrame Summary**:
   - Shape (rows × columns)
   - Date ranges (origin and target)
   - Horizon distribution
   - Channel distribution (B&M vs WEB)
   - Store count
4. **Feature Breakdown**:
   - Column categories (keys, labels, features)
   - Feature groups with NaN rates
   - Overall memory usage
5. **Sample Rows**: First 5 rows with key columns + sample features

#### Common Use Cases

**Validate Near-Term Forecasts (h=1-4):**
```bash
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B --horizons 1 4 --output validation_near_term.csv
```

**Compare Model A vs Model B:**
```bash
# Generate Model A
python -m forecasting.regressor.scripts.generate_training_sample \
    -m A --horizons 1 13 --output comparison_model_a.csv

# Generate Model B
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B --horizons 1 13 --output comparison_model_b.csv
```

**Test with Recent Data Only:**
```bash
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B \
    --horizons 1 4 \
    --start-date 2024-10-01 \
    --output recent_test.csv
```

**Generate Full Training Set:**
```bash
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B \
    --horizons 1 52 \
    --include-crm \
    --output data/canonical_training_full.csv
```

#### Performance Notes

- **Horizons 1-4**: ~1 minute, ~150K rows, ~90 MB
- **Horizons 1-13**: ~3 minutes, ~500K rows, ~300 MB
- **Horizons 1-52**: ~12 minutes, ~2M rows, ~1.2 GB

Date filtering reduces both generation time and output size.

#### Inspecting Results

After generation, inspect the output CSV:

```python
import pandas as pd

# Load generated table
df = pd.read_csv('training_model_b.csv')

# Check specific store forecast
store_334 = df[(df['profit_center_nbr'] == 334) &
               (df['channel'] == 'B&M') &
               (df['origin_week_date'] == '2024-10-01')]

print(store_334[['horizon', 'target_week_date', 'label_log_sales']].head(10))

# Inspect feature coverage
print(df.isna().sum() / len(df))

# Check feature correlations
print(df.corr()['label_log_sales'].sort_values(ascending=False).head(20))
```

#### Troubleshooting

**Issue: Script takes too long**
- Reduce horizon range (use 1-4 instead of 1-52)
- Add date filters to limit time range
- Start with smaller samples for validation

**Issue: High NaN rates in features**
- Web traffic features: Expected to be 100% NaN for B&M channel
- Conversion features: Expected to be NaN for stores without traffic data
- Lag features: Expected to have NaN for early time periods

**Issue: Missing expected features**
- Awareness features: Require proper YOUGOV_DMA_MAP configuration
- CRM features: Only included if `--include-crm` flag is used
- Check warnings in output for specific missing features

---

### 3. train.py - Model Training

Train forecasting models on prepared feature data. This is a standalone version of the train subcommand.

#### Usage

```bash
# Basic training
python -m forecasting.regressor.scripts.train \
    --model-b data/model_b.csv \
    --output output/

# With explainability model
python -m forecasting.regressor.scripts.train \
    --model-b data/model_b.csv \
    --model-a data/model_a.csv \
    --output output/

# With bias correction
python -m forecasting.regressor.scripts.train \
    --model-b data/model_b.csv \
    --output output/ \
    --correct-bm \
    --correct-web-sales
```

#### Parameters

- `--model-b PATH`: Path to Model B CSV (full features) - **required**
- `--model-a PATH`: Path to Model A CSV (business features) for explainability
- `--output DIR`: Output directory (default: output)
- `--channels {bm,web}`: Channels to train (default: bm web)
- `--correct-bm`: Apply bias correction to B&M predictions
- `--correct-web`: Apply bias correction to all WEB predictions
- `--correct-web-sales`: Apply bias correction to WEB Sales only
- `--correct-web-aov`: Apply bias correction to WEB AOV only
- `--no-surrogate`: Skip surrogate model training

#### Outputs

The script saves:
- Model checkpoints to `{output}/checkpoints/`
- Training metrics to console and `{output}/metrics/`
- Explainability artifacts if Model A provided

#### Training Summary

After training, the script prints:
- Train/test sample counts
- RMSE for Sales, AOV, Conversion (B&M), Traffic (WEB)
- Surrogate model R2 (if applicable)

---

### 4. infer.py - Inference

Run inference with saved models on new data. This is a standalone version of the infer subcommand.

#### Usage

```bash
# Basic inference
python -m forecasting.regressor.scripts.infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# With explainability
python -m forecasting.regressor.scripts.infer \
    --model-b data/new_features_b.csv \
    --model-a data/new_features_a.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# With bias correction, no explainability
python -m forecasting.regressor.scripts.infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/ \
    --correct-bm \
    --no-explainability
```

#### Parameters

- `--model-b PATH`: Path to Model B feature CSV - **required**
- `--model-a PATH`: Path to Model A feature CSV for explainability
- `--checkpoints DIR`: Directory with saved models (default: output/checkpoints)
- `--output DIR`: Output directory (default: output_infer)
- `--channels {bm,web}`: Channels to process (default: bm web)
- `--correct-bm`: Apply bias correction to B&M predictions
- `--correct-web`: Apply bias correction to all WEB predictions
- `--correct-web-sales`: Apply bias correction to WEB Sales only
- `--correct-web-aov`: Apply bias correction to WEB AOV only
- `--no-explainability`: Skip explainability analysis

#### Outputs

- `{output}/predictions_bm.csv`: B&M channel predictions with columns:
  - Keys: profit_center_nbr, dma, channel, origin_week_date, target_week_date, horizon
  - Labels: label_log_sales, label_AOV, label_ConversionRate (if available in input)
  - Predictions: pred_log_sales, pred_AOV, pred_ConversionRate
- `{output}/predictions_web.csv`: WEB channel predictions (similar structure)
- `{output}/explainability/`: Feature attributions (if requested)

---

### 5. evaluate.py - Evaluation

Evaluate model predictions against ground truth. This is a standalone version of the evaluate subcommand.

#### Usage

```bash
# Evaluate both channels
python -m forecasting.regressor.scripts.evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --predictions-web predictions/predictions_web.csv \
    --output evaluation/

# Evaluate single channel
python -m forecasting.regressor.scripts.evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --output evaluation/
```

#### Parameters

- `--predictions-bm PATH`: Path to B&M predictions CSV
- `--predictions-web PATH`: Path to WEB predictions CSV
- `--output DIR`: Output directory for evaluation summary (default: output)

**Note:** At least one of `--predictions-bm` or `--predictions-web` is required.

#### Metrics

The script computes and reports:
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actuals
- **WMAPE** (Weighted Mean Absolute Percentage Error): Percentage error weighted by actuals
- **Bias**: Mean signed error (positive = over-forecasting, negative = under-forecasting)
- **R2** (Coefficient of Determination): Proportion of variance explained

Metrics are computed for each target metric:
- Sales
- AOV (Average Order Value)
- Orders
- Conversion Rate (B&M only)
- Traffic (WEB only)

#### Outputs

- Console summary of all metrics
- `{output}/evaluation_summary.txt`: Text report
- `{output}/evaluation_metrics.csv`: Detailed metrics table

---

## Complete Workflow Examples

### Example 1: Quick Development Cycle

Test changes on a small dataset:

```bash
# 1. Generate small sample (h=1-4)
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 4 \
    --model both \
    --output dev/sample.csv

# 2. Train on sample
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b dev/sample_model_b.csv \
    --model-a dev/sample_model_a.csv \
    --output dev/models/

# 3. Run inference (same data for testing)
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b dev/sample_model_b.csv \
    --checkpoints dev/models/checkpoints \
    --output dev/predictions/

# 4. Evaluate
python -m forecasting.regressor.scripts.run_pipeline evaluate \
    --predictions-bm dev/predictions/predictions_bm.csv \
    --predictions-web dev/predictions/predictions_web.csv \
    --output dev/evaluation/
```

### Example 2: Production Training

Full year training with all features:

```bash
# 1. Generate full training set
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --model both \
    --include-crm \
    --output prod/training.csv

# 2. Train with bias correction
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b prod/training_model_b.csv \
    --model-a prod/training_model_a.csv \
    --output prod/models/ \
    --correct-bm \
    --correct-web-sales

# Models saved to prod/models/checkpoints/
```

### Example 3: Inference Only

Use pre-trained models for new forecasts:

```bash
# 1. Generate features for new data
python -m forecasting.regressor.scripts.generate_training_sample \
    -m B \
    --horizons 1 13 \
    --start-date 2025-01-01 \
    --output forecasts/q1_2025_features.csv

# 2. Run inference
python -m forecasting.regressor.scripts.infer \
    --model-b forecasts/q1_2025_features.csv \
    --checkpoints prod/models/checkpoints \
    --output forecasts/q1_2025_predictions/ \
    --correct-bm \
    --correct-web-sales
```

### Example 4: Comparing Bias Correction Settings

Test different bias correction configurations:

```bash
# Baseline (no correction)
python -m forecasting.regressor.scripts.train \
    --model-b data/model_b.csv \
    --output experiments/baseline/

# B&M correction only
python -m forecasting.regressor.scripts.train \
    --model-b data/model_b.csv \
    --output experiments/bm_corrected/ \
    --correct-bm

# Both channels corrected
python -m forecasting.regressor.scripts.train \
    --model-b data/model_b.csv \
    --output experiments/both_corrected/ \
    --correct-bm \
    --correct-web
```

---

## Model Variants

### Model A: Actionable/Explainability

**Purpose:** Business decision support and forecast explanation

**Features (41 total):**
- Time-varying: Seasonality, holidays, fiscal calendar
- Store DNA: Sales capacity, outlet flag, age, size
- Product Mix: Value/premium/white glove percentages
- Omnichannel: Cross-channel shopping rates
- Cannibalization: New store impacts
- Awareness: Brand metrics from surveys
- CRM: Customer demographics (optional)

**Excludes:** Autoregressive features (lags, rolling averages)

**Use Cases:**
- Explaining forecast drivers to stakeholders
- Understanding business lever impacts
- Scenario planning ("what-if" analysis)
- Model interpretation with SHAP values

### Model B: Production/Full

**Purpose:** Most accurate predictions for operational planning

**Features (57 total):**
- All Model A features
- Sales Lags: Recent historical sales patterns
- Rolling Averages: Smoothed trends
- AOV Patterns: Historical average order value
- Conversion Patterns: Historical conversion rates
- Traffic Patterns: Historical web traffic

**Use Cases:**
- Production forecasts for inventory planning
- Demand planning
- Financial projections
- Performance tracking

---

## Output Files Reference

### Training Outputs

```
output/
├── checkpoints/
│   ├── bm_sales_model.cbm          # B&M Sales CatBoost model
│   ├── bm_aov_model.cbm            # B&M AOV model
│   ├── bm_conv_model.cbm           # B&M Conversion model
│   ├── web_sales_model.cbm         # WEB Sales model
│   ├── web_aov_model.cbm           # WEB AOV model
│   ├── bm_surrogate_model.pkl      # B&M explainability surrogate
│   └── web_surrogate_model.pkl     # WEB explainability surrogate
├── metrics/
│   ├── training_summary.txt        # Overall training metrics
│   └── metrics_by_channel.csv      # Detailed metrics
└── explainability/
    ├── bm_feature_importance.csv   # B&M SHAP feature importance
    └── web_feature_importance.csv  # WEB SHAP feature importance
```

### Inference Outputs

```
predictions/
├── predictions_bm.csv              # B&M predictions with actuals
├── predictions_web.csv             # WEB predictions with actuals
└── explainability/
    ├── bm_attributions.csv         # Per-prediction feature attributions
    └── web_attributions.csv        # Per-prediction feature attributions
```

### Evaluation Outputs

```
evaluation/
├── evaluation_summary.txt          # Human-readable metrics summary
└── evaluation_metrics.csv          # Detailed metrics table
```

---

## Help Commands

Get detailed help for any script:

```bash
# Unified pipeline
python -m forecasting.regressor.scripts.run_pipeline --help
python -m forecasting.regressor.scripts.run_pipeline generate --help
python -m forecasting.regressor.scripts.run_pipeline train --help
python -m forecasting.regressor.scripts.run_pipeline infer --help
python -m forecasting.regressor.scripts.run_pipeline evaluate --help

# Individual scripts
python -m forecasting.regressor.scripts.generate_training_sample --help
python -m forecasting.regressor.scripts.train --help
python -m forecasting.regressor.scripts.infer --help
python -m forecasting.regressor.scripts.evaluate --help
```

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'forecasting'**
- Ensure you're running from the project root directory
- Use the module syntax: `python -m forecasting.regressor.scripts.{script_name}`

**FileNotFoundError: Checkpoint not found**
- Verify the checkpoint directory path
- Ensure training completed successfully
- Check that the correct model files exist (*.cbm files)

**High memory usage during generation**
- Reduce horizon range for initial testing
- Use date filters to limit data scope
- Process channels separately if needed

**Poor model performance**
- Check for sufficient training data (at least 1 year recommended)
- Verify feature quality and NaN rates
- Consider enabling bias correction flags
- Review feature importance to identify issues

**Bias in predictions**
- Enable appropriate bias correction flags:
  - `--correct-bm` for B&M channel
  - `--correct-web-sales` for WEB sales
  - `--correct-web-aov` for WEB AOV
- Review evaluation metrics to quantify bias

---

## Best Practices

1. **Start Small**: Test with horizons 1-4 before scaling to full year
2. **Use Pipeline**: Prefer `run_pipeline.py` for consistent workflows
3. **Version Control**: Save generated feature CSVs for reproducibility
4. **Monitor Metrics**: Always evaluate predictions before deploying
5. **Bias Correction**: Enable for production models after validating impact
6. **Explainability**: Use Model A for stakeholder communication
7. **Documentation**: Keep track of training configurations and results
8. **Validation**: Use recent held-out data to validate model performance

---

## Additional Resources

- **Pipeline Documentation**: `api/app/regressor/pipelines/`
- **Feature Engineering**: `api/app/regressor/features/`
- **ETL Process**: `api/app/regressor/etl/`
- **Project Plan**: `api/app/regressor/PROJECT_PLAN.md`
