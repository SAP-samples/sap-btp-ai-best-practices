"""
Training Configuration.

Defines configuration for the training pipeline including data splits,
validation settings, and output paths.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from .base import BaseConfig, BiasCorrection


@dataclass
class DataSplitConfig:
    """Configuration for train/test data splitting."""

    train_years: List[int] = field(default_factory=lambda: [2022, 2023, 2024])
    test_years: List[int] = field(default_factory=lambda: [2025])
    date_column: str = "origin_week_date"

    def get_train_mask(self, df):
        """Return boolean mask for training data."""
        import pandas as pd
        years = pd.to_datetime(df[self.date_column]).dt.year
        return years.isin(self.train_years)

    def get_test_mask(self, df):
        """Return boolean mask for test data."""
        import pandas as pd
        years = pd.to_datetime(df[self.date_column]).dt.year
        return years.isin(self.test_years)


@dataclass
class TrainingConfig(BaseConfig):
    """Complete training pipeline configuration."""

    # Data split configuration
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)

    # Bias correction settings
    bias_correction: BiasCorrection = field(default_factory=BiasCorrection)

    # Channels to train
    channels: List[str] = field(default_factory=lambda: ["B&M", "WEB"])

    # Whether to train surrogate models for explainability
    train_surrogate: bool = True

    # Number of top contributors to extract per row
    top_k_contributors: int = 4

    # Checkpoint directory (relative to output_dir)
    checkpoint_subdir: str = "checkpoints"

    # Key columns for merging/alignment
    key_columns: List[str] = field(default_factory=lambda: [
        "profit_center_nbr", "origin_week_date", "horizon"
    ])

    @property
    def checkpoint_dir(self) -> Path:
        """Get the full path to checkpoint directory."""
        return self.output_dir / self.checkpoint_subdir

    def validate(self) -> None:
        """Validate configuration."""
        for channel in self.channels:
            if channel not in ("B&M", "WEB"):
                raise ValueError(f"Invalid channel '{channel}'. Must be 'B&M' or 'WEB'.")


@dataclass
class InferenceConfig(BaseConfig):
    """Configuration for inference pipeline."""

    # Path to saved checkpoints
    checkpoint_dir: Path = field(default_factory=lambda: Path("output/checkpoints"))

    # Bias correction settings
    bias_correction: BiasCorrection = field(default_factory=BiasCorrection)

    # Channels to run inference on
    channels: List[str] = field(default_factory=lambda: ["B&M", "WEB"])

    # Whether to run explainability (requires Model A data and surrogate models)
    run_explainability: bool = True

    # Number of top contributors to extract per row
    top_k_contributors: int = 4

    # Key columns for merging/alignment
    key_columns: List[str] = field(default_factory=lambda: [
        "profit_center_nbr", "origin_week_date", "horizon"
    ])

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for evaluation pipeline."""

    # Channels to evaluate
    channels: List[str] = field(default_factory=lambda: ["B&M", "WEB"])

    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "mae", "wmape", "bias", "r2"
    ])

    # Targets to evaluate
    targets: List[str] = field(default_factory=lambda: [
        "log_sales", "log_aov", "logit_conversion",
        "sales", "aov", "orders", "conversion", "traffic"
    ])
