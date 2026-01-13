# Configuration Module

This folder contains configuration dataclasses and factory functions for the Company X Sales Forecasting regressor system. These configurations control model hyperparameters, feature selection, training pipeline settings, and inference behavior across both Model A (explainability) and Model B (production predictions).

## Overview

The configuration system provides a type-safe, composable way to configure:
- **CatBoost model hyperparameters** (iterations, learning rate, depth, loss functions)
- **Feature engineering rules** (categorical features, channel-specific exclusions)
- **Training pipeline settings** (data splits, bias correction, output paths)
- **Inference and evaluation configurations** (checkpoint loading, explainability settings)

The dual-model architecture requires careful feature partitioning:
- **Model A**: Business-actionable levers + known-in-advance/static features (for SHAP explainability)
- **Model B**: Full feature set including autoregressive lags/rolls (for production forecasts)

## File Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports, factory functions (`get_bm_multi_config`, `get_default_config`), and utility functions (`load_config`) |
| `base.py` | Base configuration dataclasses: `BaseConfig` (common fields) and `BiasCorrection` (log-normal correction settings) |
| `model_config.py` | Model specifications: `CatBoostHyperparams`, `FeatureConfig`, `ModelConfig`, and channel-specific factory functions |
| `training_config.py` | Pipeline configurations: `TrainingConfig`, `InferenceConfig`, `EvaluationConfig`, `DataSplitConfig` |

## Configuration Classes

### Base Configurations (`base.py`)

#### BaseConfig
Common fields shared across all configuration types:
- `name`: Configuration identifier (default: "default")
- `channel`: Channel type - "B&M", "WEB", or None for multi-channel
- `output_dir`: Output directory path (default: "output")
- `random_seed`: Random seed for reproducibility (default: 42)

#### BiasCorrection
Log-normal bias correction settings. When predicting in log-space and converting back via `exp()`, the mean is biased. Correction applies: `exp(mu + sigma^2/2)`
- `correct_bm`: Apply bias correction to B&M predictions (default: False)
- `correct_web`: Apply bias correction to all WEB predictions (default: False)
- `correct_web_sales`: Apply bias correction to WEB Sales only (default: False)
- `correct_web_aov`: Apply bias correction to WEB AOV only (default: False)

### Model Configurations (`model_config.py`)

#### CatBoostHyperparams
CatBoost model hyperparameters:
- `iterations`: Number of boosting iterations (default: 5000)
- `learning_rate`: Learning rate (default: 0.05)
- `depth`: Tree depth (default: 6)
- `loss_function`: Loss function - "MultiRMSE" for multi-target, "RMSE" for single-target (default: "MultiRMSE")
- `eval_metric`: Evaluation metric (defaults to `loss_function` if None)
- `early_stopping_rounds`: Early stopping patience (default: 50)
- `random_seed`: Random seed (default: 42)
- `verbose`: Logging verbosity (default: 100)

#### FeatureConfig
Feature engineering and selection rules:

**Categorical Features** (11 features):
- Store/market identifiers: `profit_center_nbr`, `dma`
- Store attributes: `is_outlet`, `is_comp_store`, `is_new_store`
- Holiday flags: `is_holiday`, `is_xmas_window`, `is_black_friday_window`, `is_pre_holiday_1wk`, `is_pre_holiday_2wk`, `is_pre_holiday_3wk`

**Universal Exclusions** (metadata, targets, store DNA):
- Metadata: `channel`, `origin_week_date`, `target_week_date`, `has_traffic_data`
- Targets and predictions: `label_log_sales`, `label_log_aov`, `label_logit_conversion`, `label_log_orders`, `predicted_*`
- Raw values: `total_sales`, `order_count`, `store_traffic`
- Store DNA: `store_design_sf` (not predictive for either channel)

**B&M-Only Features** (excluded from WEB models):
- Conversion features: `ConversionRate_lag_1`, `ConversionRate_lag_4`, `ConversionRate_roll_mean_*`
- Cannibalization features: `cannibalization_pressure`, `min_dist_new_store_km`, `num_new_stores_within_*`
- Physical space: `merchandising_sf`

**WEB-Only Features** (excluded from B&M models):
- Web traffic features: `allocated_web_traffic_lag_*`, `allocated_web_traffic_roll_mean_*`
- WEB sales autoregressive: `log_web_sales_lag_*`, `log_web_sales_roll_mean_*`, `vol_log_web_sales_13`
- WEB AOV features: `web_aov_roll_mean_*`
- DMA market features: `dma_web_penetration_pct`

#### ModelConfig
Complete model specification combining hyperparameters and feature configuration:
- `name`: Model identifier
- `channel`: Channel type ("B&M", "WEB", or None)
- `target`: Target variable - "sales", "aov", "orders", "conversion", or "multi"
- `hyperparams`: CatBoostHyperparams instance
- `features`: FeatureConfig instance

Methods:
- `get_exclude_features_for_channel(channel)`: Returns complete exclusion list for a given channel

### Training Configurations (`training_config.py`)

#### DataSplitConfig
Train/test data splitting configuration:
- `train_years`: Years for training data (default: [2022, 2023, 2024])
- `test_years`: Years for test data (default: [2025])
- `date_column`: Date column name (default: "origin_week_date")

Methods:
- `get_train_mask(df)`: Returns boolean mask for training data
- `get_test_mask(df)`: Returns boolean mask for test data

#### TrainingConfig
Complete training pipeline configuration (extends BaseConfig):
- `data_split`: DataSplitConfig instance
- `bias_correction`: BiasCorrection instance
- `channels`: List of channels to train (default: ["B&M", "WEB"])
- `train_surrogate`: Train surrogate models for explainability (default: True)
- `top_k_contributors`: Number of top SHAP contributors per row (default: 4)
- `checkpoint_subdir`: Checkpoint subdirectory name (default: "checkpoints")
- `key_columns`: Key columns for merging/alignment (default: ["profit_center_nbr", "origin_week_date", "horizon"])

Properties:
- `checkpoint_dir`: Full path to checkpoint directory (`output_dir / checkpoint_subdir`)

#### InferenceConfig
Inference pipeline configuration (extends BaseConfig):
- `checkpoint_dir`: Path to saved model checkpoints (default: "output/checkpoints")
- `bias_correction`: BiasCorrection instance
- `channels`: List of channels for inference (default: ["B&M", "WEB"])
- `run_explainability`: Run explainability with surrogate models (default: True, requires Model A data)
- `top_k_contributors`: Number of top SHAP contributors per row (default: 4)
- `key_columns`: Key columns for merging/alignment

#### EvaluationConfig
Evaluation pipeline configuration (extends BaseConfig):
- `channels`: Channels to evaluate (default: ["B&M", "WEB"])
- `metrics`: Metrics to compute (default: ["mae", "wmape", "bias", "r2"])
- `targets`: Targets to evaluate (default: ["log_sales", "log_aov", "logit_conversion", "sales", "aov", "orders", "conversion", "traffic"])

## Factory Functions

Pre-configured model builders for common use cases:

### Model Configurations

- `get_bm_multi_config()`: B&M multi-objective model (Sales, AOV, Orders)
  - Loss: MultiRMSE
  - Features: All B&M features + horizon
  - Use case: Production B&M forecasts

- `get_bm_conversion_config()`: B&M conversion model
  - Loss: RMSE
  - Target: logit(conversion)
  - Sample weights: store traffic

- `get_web_multi_config()`: WEB multi-objective model (Sales, AOV, Orders)
  - Loss: MultiRMSE
  - Features: All WEB features + horizon
  - Use case: Production WEB forecasts

- `get_surrogate_config()`: Surrogate explainability model
  - Loss: RMSE
  - Higher learning rate (0.1) and depth (8) to overfit Model B predictions
  - No early stopping
  - Use case: SHAP analysis on business levers (Model A)

### Default Configurations

- `get_default_config(channel, target="multi")`: Get default model config for channel/target
- `get_default_training_config(...)`: Get default training pipeline config
- `get_default_inference_config(...)`: Get default inference pipeline config

### Utility Functions

- `load_config(path)`: Load configuration from JSON or YAML file

## System Context

These configurations support the dual-model forecasting architecture:

### Model A (Explainability)
- **Purpose**: Explain how business levers (awareness, promotions, seasonality) drive outcomes
- **Features**: Actionable levers + known-in-advance/static context (NO autoregressive lags/rolls)
- **Usage**: SHAP analysis, driver attribution, business insights
- **Implementation**: Surrogate model trained to overfit Model B predictions using only lever features

### Model B (Production Forecasts)
- **Purpose**: Maximize forecast accuracy for planning and decision-making
- **Features**: Full feature set including autoregressive lags/rolls
- **Usage**: Production forecasts for Sales, AOV, Orders, Conversion, Traffic
- **Channels**: B&M (brick-and-mortar) and WEB (e-commerce)

### Targets
The system predicts 5 key metrics for both channels:
1. **Sales**: Total revenue (log-transformed)
2. **AOV**: Average Order Value (log-transformed)
3. **Orders**: Order count (log-transformed)
4. **Conversion**: Conversion rate (logit-transformed, B&M only with traffic data)
5. **Traffic**: Store/web traffic (derived from Sales = Traffic × Conversion × AOV)

### Multi-Target Coherence
Uses MultiRMSE loss to maintain coherence across the P&L triangle:
- Sales = Orders × AOV
- Orders = Traffic × Conversion
- This ensures consistent predictions that respect accounting identities

## Usage Examples

### Basic Model Configuration

```python
from forecasting.regressor.configs import get_bm_multi_config, get_web_multi_config

# Get pre-configured B&M model
bm_config = get_bm_multi_config()
print(bm_config.name)  # "bm_multi_objective"
print(bm_config.channel)  # "B&M"
print(bm_config.hyperparams.iterations)  # 5000

# Get pre-configured WEB model
web_config = get_web_multi_config()
```

### Custom Model Configuration

```python
from forecasting.regressor.configs import ModelConfig, CatBoostHyperparams

# Create custom configuration
custom_config = ModelConfig(
    name="custom_bm_model",
    channel="B&M",
    target="multi",
    hyperparams=CatBoostHyperparams(
        iterations=10000,
        learning_rate=0.03,
        depth=8,
        loss_function="MultiRMSE",
    )
)

# Get channel-specific feature exclusions
excluded = custom_config.get_exclude_features_for_channel("B&M")
# Returns: universal exclusions + WEB-only features
```

### Training Pipeline Configuration

```python
from forecasting.regressor.configs import TrainingConfig, BiasCorrection, DataSplitConfig

# Create training configuration
train_config = TrainingConfig(
    name="production_training",
    output_dir="output/prod",
    channels=["B&M", "WEB"],
    data_split=DataSplitConfig(
        train_years=[2022, 2023, 2024],
        test_years=[2025]
    ),
    bias_correction=BiasCorrection(
        correct_bm=False,
        correct_web=True
    ),
    train_surrogate=True,  # Generate Model A explainability
    top_k_contributors=4
)

# Access checkpoint directory
checkpoint_path = train_config.checkpoint_dir
# Returns: Path("output/prod/checkpoints")
```

### Inference Configuration

```python
from forecasting.regressor.configs import InferenceConfig, BiasCorrection

# Create inference configuration
infer_config = InferenceConfig(
    name="forecast_2025",
    checkpoint_dir="output/prod/checkpoints",
    output_dir="output/forecasts",
    channels=["B&M", "WEB"],
    bias_correction=BiasCorrection(correct_web=True),
    run_explainability=True  # Include SHAP analysis
)
```

### Using Factory Functions

```python
from forecasting.regressor.configs import (
    get_default_config,
    get_default_training_config,
    get_surrogate_config
)

# Get default config by channel
bm_config = get_default_config(channel="B&M", target="multi")
web_config = get_default_config(channel="WEB", target="multi")

# Get default training config with bias correction
train_config = get_default_training_config(
    output_dir="output/experiment",
    channels=["B&M", "WEB"],
    correct_bm=False,
    correct_web=True
)

# Get surrogate model config for explainability
surrogate_config = get_surrogate_config()
print(surrogate_config.hyperparams.learning_rate)  # 0.1 (higher to overfit)
print(surrogate_config.hyperparams.depth)  # 8 (deeper to capture complexity)
```

### Loading from File

```python
from forecasting.regressor.configs import load_config, ModelConfig

# Load from JSON file
config_dict = load_config("configs/my_model.json")

# Or from YAML file (requires PyYAML)
config_dict = load_config("configs/my_model.yaml")

# Create ModelConfig from dictionary
model_config = ModelConfig(**config_dict)
```

## Important Configuration Parameters

### Default Hyperparameters
- **Iterations**: 5000 (with early stopping at 50 rounds)
- **Learning Rate**: 0.05 (standard), 0.1 (surrogate models)
- **Tree Depth**: 6 (standard), 8 (surrogate models)
- **Loss Function**: MultiRMSE (multi-target), RMSE (single-target)

### Data Splits
- **Training Years**: 2022, 2023, 2024
- **Test Years**: 2025
- **Date Column**: "origin_week_date" (the $t_0$ observation point)

### Feature Categories
- **11 Categorical Features**: Store/market IDs, outlet flags, holiday indicators
- **Universal Exclusions**: Metadata, targets, predictions, store design SF
- **7 B&M-Only Features**: Conversion metrics, cannibalization, merchandising SF
- **13 WEB-Only Features**: Web traffic, web sales lags, web AOV, DMA penetration

### Bias Correction
- Applied when predictions are in log-space or logit-space
- Corrects for Jensen's inequality: E[exp(X)] != exp(E[X])
- Can be applied selectively by channel and target
- Formula: exp(mu + sigma^2/2) for log-space

## Relationship to Broader System

The configuration module is part of the `api/app/regressor/` package, which implements the complete forecasting pipeline:

1. **Data Loading** (`io_utils`): Load raw sales, traffic, and store master data
2. **Feature Engineering** (`features/`): Generate temporal, static, and dynamic features
3. **Configuration** (`configs/`): THIS MODULE - Define model and pipeline settings
4. **Training** (`train.py`): Train Model B and surrogate Model A
5. **Inference** (`predict.py`): Generate forecasts and explainability outputs
6. **Evaluation** (`evaluate.py`): Compute accuracy metrics and validation reports

The configuration system ensures consistency across these stages by:
- Centralizing feature exclusion rules (preventing WEB features in B&M models)
- Managing data splits (preventing temporal leakage)
- Controlling hyperparameters (ensuring reproducibility)
- Coordinating bias correction (maintaining consistency between training and inference)

## Notes

- All configurations use Python dataclasses for type safety and IDE autocomplete
- Configurations are immutable after `__post_init__` validation
- Factory functions provide sensible defaults for common scenarios
- Channel-specific feature exclusion is handled automatically via `get_exclude_features_for_channel()`
- The surrogate model (Model A) intentionally overfits Model B to preserve explainability fidelity
