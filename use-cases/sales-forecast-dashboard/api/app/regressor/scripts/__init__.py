"""
CLI Scripts Module.

Provides command-line interfaces for the forecasting regressor:

- run_pipeline.py: Unified entry point with subcommands
    python -m forecasting.regressor.scripts.run_pipeline train|infer|evaluate

- train.py: Train models
    python -m forecasting.regressor.scripts.train --model-b data.csv

- infer.py: Run inference
    python -m forecasting.regressor.scripts.infer --model-b data.csv --checkpoints models/

- evaluate.py: Evaluate predictions
    python -m forecasting.regressor.scripts.evaluate --predictions-bm predictions.csv

- generate_training_sample.py: Generate canonical training data (existing)
    python -m forecasting.regressor.scripts.generate_training_sample
"""

__all__ = [
    "run_pipeline",
    "train",
    "infer",
    "evaluate",
    "generate_training_sample",
]
