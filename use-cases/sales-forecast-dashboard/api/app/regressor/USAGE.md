# Forecasting Regressor Usage Guide

## CLI Usage

The unified entry point supports four commands: `generate`, `train`, `infer`, and `evaluate`.

### Generate Training Data

```bash
# Generate both Model A and Model B features (default)
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --output data/training_data.csv

# Generate only Model B features
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 12 \
    --model B \
    --output data/model_b_features.csv

# Include CRM demographic features
python -m forecasting.regressor.scripts.run_pipeline generate \
    --horizons 1 52 \
    --include-crm \
    --output data/
```

### Train Models

```bash
# Train B&M and WEB models with explainability
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --model-a data/model_a.csv \
    --output output/

# Train only B&M channel
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --channels bm \
    --output output/

# Train with bias correction
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --model-a data/model_a.csv \
    --correct-bm \
    --correct-web-sales \
    --output output/

# Skip surrogate model training (faster, no SHAP analysis)
python -m forecasting.regressor.scripts.run_pipeline train \
    --model-b data/model_b.csv \
    --no-surrogate \
    --output output/
```

### Run Inference

```bash
# Run inference on new data
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# Include explainability analysis
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --model-a data/new_features_model_a.csv \
    --checkpoints output/checkpoints \
    --output predictions/

# Skip explainability (faster)
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --no-explainability \
    --output predictions/

# With bias correction
python -m forecasting.regressor.scripts.run_pipeline infer \
    --model-b data/new_features.csv \
    --checkpoints output/checkpoints \
    --correct-bm \
    --correct-web-sales \
    --correct-web-aov \
    --output predictions/
```

### Evaluate Predictions

```bash
# Evaluate B&M predictions
python -m forecasting.regressor.scripts.run_pipeline evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --output evaluation/

# Evaluate both channels
python -m forecasting.regressor.scripts.run_pipeline evaluate \
    --predictions-bm predictions/predictions_bm.csv \
    --predictions-web predictions/predictions_web.csv \
    --output evaluation/
```

---

## Python API Usage

### Basic Training and Inference

```python
from forecasting.regressor import train, infer, evaluate

# Train models
result = train(
    model_b_path="data/model_b.csv",
    model_a_path="data/model_a.csv",  # Optional, for explainability
    output_dir="output/",
    channels=["B&M", "WEB"],
)

print(f"B&M RMSE Sales: {result.bm_result.rmse_sales:.4f}")
print(f"WEB RMSE Sales: {result.web_result.rmse_sales:.4f}")

# Run inference
predictions = infer(
    model_b_path="data/new_features.csv",
    checkpoint_dir="output/checkpoints",
    output_dir="predictions/",
)

# Access predictions
bm_df = predictions.bm_predictions
web_df = predictions.web_predictions
print(f"B&M predictions: {len(bm_df)} rows")
```

### Using Pipeline Classes Directly

```python
from forecasting.regressor.pipelines import TrainingPipeline, InferencePipeline
from forecasting.regressor.configs import TrainingConfig, InferenceConfig, BiasCorrection
import pandas as pd

# Load data
model_b_data = pd.read_csv("data/model_b.csv")
model_a_data = pd.read_csv("data/model_a.csv")

# Configure training
config = TrainingConfig(
    output_dir="output/",
    channels=["B&M", "WEB"],
    bias_correction=BiasCorrection(
        correct_bm=True,
        correct_web_sales=True,
    ),
    train_surrogate=True,
)

# Train
pipeline = TrainingPipeline(config)
result = pipeline.run(model_b_data, model_a_data)

# Configure inference
infer_config = InferenceConfig(
    checkpoint_dir="output/checkpoints",
    output_dir="predictions/",
    run_explainability=True,
)

# Inference
infer_pipeline = InferencePipeline(infer_config)
predictions = infer_pipeline.run(model_b_data, model_a_data)
```

### Using Individual Predictors

```python
from forecasting.regressor.models import BMPredictor, WEBPredictor, get_predictor
import pandas as pd

# Load and filter data
data = pd.read_csv("data/model_b.csv")
bm_data = data[data["channel"] == "B&M"]

# Train B&M predictor
predictor = BMPredictor(iterations=5000)
predictor.fit(bm_data)

# Generate predictions with traffic estimates
predictions = predictor.predict(bm_data, estimate_traffic=True)

print(f"Sales (log): {predictions.log_sales[:5]}")
print(f"Traffic P50: {predictions.traffic.p50[:5]}")

# Save and load models
predictor.save_models("checkpoints/")

# Later: load and predict
loaded_predictor = BMPredictor()
loaded_predictor.load_models("checkpoints/")
new_predictions = loaded_predictor.predict(new_data)
```

### Using the Factory Function

```python
from forecasting.regressor.models import get_predictor

# Get predictor by channel name
bm_predictor = get_predictor("B&M")
web_predictor = get_predictor("WEB")
```

### Generating Training Data Programmatically

```python
from forecasting.regressor.etl import build_canonical_training_table

# Generate Model B features for horizons 1-12
df = build_canonical_training_table(
    horizons=range(1, 13),
    include_features=True,
    model_variant='B',
    include_crm=False,
)

print(f"Generated {len(df)} rows with {len(df.columns)} columns")
df.to_csv("data/model_b_h1-12.csv", index=False)
```

### Explainability Analysis

```python
from forecasting.regressor.models import SurrogateExplainer
import pandas as pd

# Load trained B&M surrogate
# Note: channel parameter ensures correct feature filtering
# - B&M: Uses all 23 Model A features
# - WEB: Excludes B&M-only features (merchandising_sf, cannibalization, etc.)
bm_surrogate = SurrogateExplainer(channel="B&M")
bm_surrogate.load_model(
    "output/checkpoints/surrogate_bm.cbm",
    meta_path="output/checkpoints/surrogate_bm.meta.json"
)

# Generate SHAP explanations for B&M
data_bm = pd.read_csv("data/model_a.csv")
data_bm = data_bm[data_bm["channel"] == "B&M"]
contributor_df = bm_surrogate.explain(
    df=data_bm,
    output_dir="shap_plots/",
    name="BM_Sales",
    keys=["profit_center_nbr", "origin_week_date", "horizon"],
    top_k_contributors=4,
)

# Load trained WEB surrogate (uses channel-specific features)
web_surrogate = SurrogateExplainer(channel="WEB")
web_surrogate.load_model(
    "output/checkpoints/surrogate_web.cbm",
    meta_path="output/checkpoints/surrogate_web.meta.json"
)

# Get feature importance
importance = bm_surrogate.get_feature_importance()
print(importance.head(10))
```

---

## Output Files

After training, the following files are created:

```
output/
├── predictions_bm.csv      # B&M predictions with quantiles
├── predictions_web.csv     # WEB predictions with quantiles
├── checkpoints/
│   ├── bm_multi.cbm        # B&M multi-objective model
│   ├── bm_conversion.cbm   # B&M conversion model
│   ├── web_multi.cbm       # WEB multi-objective model
│   ├── surrogate_bm.cbm    # B&M surrogate for SHAP
│   ├── surrogate_bm.meta.json
│   ├── surrogate_web.cbm   # WEB surrogate for SHAP
│   ├── surrogate_web.meta.json
│   └── residual_stats.json # RMSE values for quantiles
├── shap_summary_BM_Sales.png
├── shap_importance_BM_Sales.png
└── shap_dependence_*.png
```

## Prediction Columns

The prediction DataFrames include:

| Column | Description |
|--------|-------------|
| `pred_log_sales` | Log-scale sales prediction |
| `pred_sales_p50` | Median sales (exp of log prediction) |
| `pred_sales_p90` | 90th percentile sales |
| `pred_sales_mean` | Bias-corrected mean (if enabled) |
| `pred_log_aov` | Log-scale AOV prediction |
| `pred_aov_p50` | Median AOV |
| `pred_traffic_p10/p50/p90` | Traffic quantiles (B&M only) |
| `pred_logit_conversion` | Logit-scale conversion (B&M only) |
| `top_features_pred_log_sales` | Top SHAP contributors (if explainability enabled) |
