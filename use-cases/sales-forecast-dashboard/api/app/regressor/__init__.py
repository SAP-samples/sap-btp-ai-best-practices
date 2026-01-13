"""
Company X Forecasting Regressor Package.

A comprehensive sales forecasting system using CatBoost models with:
- Dual-model architecture (Model B for accuracy, Model A for explainability)
- Channel-specific predictors (B&M and WEB)
- Multi-objective prediction (Sales, AOV, Orders, Conversion)
- Monte Carlo traffic estimation
- SHAP-based explainability

Modules:
    - configs: Configuration classes for models and pipelines
    - models: CatBoost predictors for each channel
    - pipelines: Training, inference, and evaluation orchestration
    - explainability: SHAP analysis utilities
    - etl: Data ingestion and canonical table generation
    - features: Feature engineering transformations
    - data_ingestion: Raw data loaders

Quick Start:
    # Train models
    from app.regressor.pipelines import train
    result = train(
        model_b_path="data/model_b.csv",
        model_a_path="data/model_a.csv",
        output_dir="output/",
    )

    # Run inference
    from app.regressor.pipelines import infer
    predictions = infer(
        model_b_path="data/new_features.csv",
        checkpoint_dir="output/checkpoints",
    )

    # Evaluate
    from app.regressor.pipelines import evaluate
    metrics = evaluate(
        predictions_bm_path="output/predictions_bm.csv",
        predictions_web_path="output/predictions_web.csv",
    )
    metrics.print_summary()

CLI Usage:
    # Unified entry point
    python -m forecasting.regressor.scripts.run_pipeline train|infer|evaluate

    # Individual scripts
    python -m forecasting.regressor.scripts.train --model-b data.csv
    python -m forecasting.regressor.scripts.infer --model-b data.csv --checkpoints models/
    python -m forecasting.regressor.scripts.evaluate --predictions-bm predictions.csv
"""

# Version
__version__ = "0.1.0"

# Convenience imports for common use cases
from app.regressor.pipelines import (
    TrainingPipeline,
    InferencePipeline,
    EvaluationPipeline,
    train,
    infer,
    evaluate,
)

from app.regressor.models import (
    BMPredictor,
    WEBPredictor,
    SurrogateExplainer,
    get_predictor,
)

from app.regressor.configs import (
    TrainingConfig,
    InferenceConfig,
    ModelConfig,
    get_default_config,
    get_default_training_config,
    get_default_inference_config,
)

__all__ = [
    # Pipelines
    "TrainingPipeline",
    "InferencePipeline",
    "EvaluationPipeline",
    "train",
    "infer",
    "evaluate",
    # Models
    "BMPredictor",
    "WEBPredictor",
    "SurrogateExplainer",
    "get_predictor",
    # Configs
    "TrainingConfig",
    "InferenceConfig",
    "ModelConfig",
    "get_default_config",
    "get_default_training_config",
    "get_default_inference_config",
]
