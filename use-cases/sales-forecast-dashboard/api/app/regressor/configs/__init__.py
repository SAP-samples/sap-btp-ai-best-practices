"""
Configuration Module.

Provides configuration classes and utilities for the forecasting regressor.

Usage:
    from app.regressor.configs import (
        TrainingConfig,
        InferenceConfig,
        ModelConfig,
        get_bm_multi_config,
        load_config,
    )
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional

from .base import BaseConfig, BiasCorrection
from .model_config import (
    ModelConfig,
    CatBoostHyperparams,
    FeatureConfig,
    get_bm_multi_config,
    get_bm_conversion_config,
    get_web_multi_config,
    get_surrogate_config,
)
from .training_config import (
    TrainingConfig,
    InferenceConfig,
    EvaluationConfig,
    DataSplitConfig,
)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the file format is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}. Use .json or .yaml")


def get_default_config(channel: str, target: str = "multi") -> ModelConfig:
    """
    Get a default model configuration for a given channel and target.

    Parameters
    ----------
    channel : str
        Channel type: 'B&M' or 'WEB'.
    target : str, optional
        Target type: 'multi', 'sales', 'aov', 'orders', 'conversion'.
        Defaults to 'multi'.

    Returns
    -------
    ModelConfig
        Pre-configured model configuration.

    Raises
    ------
    ValueError
        If the channel or target is not valid.
    """
    channel = channel.upper() if channel else channel

    if channel == "B&M" or channel == "BM":
        if target == "conversion":
            return get_bm_conversion_config()
        else:
            return get_bm_multi_config()
    elif channel == "WEB":
        return get_web_multi_config()
    else:
        raise ValueError(f"Invalid channel '{channel}'. Must be 'B&M' or 'WEB'.")


def get_default_training_config(
    output_dir: str = "output",
    channels: Optional[list] = None,
    correct_bm: bool = False,
    correct_web: bool = False,
) -> TrainingConfig:
    """
    Get a default training configuration.

    Parameters
    ----------
    output_dir : str, optional
        Output directory path.
    channels : list, optional
        List of channels to train. Defaults to ['B&M', 'WEB'].
    correct_bm : bool, optional
        Apply bias correction to B&M predictions.
    correct_web : bool, optional
        Apply bias correction to WEB predictions.

    Returns
    -------
    TrainingConfig
        Pre-configured training configuration.
    """
    return TrainingConfig(
        name="default_training",
        output_dir=Path(output_dir),
        channels=channels or ["B&M", "WEB"],
        bias_correction=BiasCorrection(
            correct_bm=correct_bm,
            correct_web=correct_web,
        ),
    )


def get_default_inference_config(
    checkpoint_dir: str = "output/checkpoints",
    output_dir: str = "output_infer",
    correct_bm: bool = False,
    correct_web: bool = False,
) -> InferenceConfig:
    """
    Get a default inference configuration.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Path to saved model checkpoints.
    output_dir : str, optional
        Output directory for predictions.
    correct_bm : bool, optional
        Apply bias correction to B&M predictions.
    correct_web : bool, optional
        Apply bias correction to WEB predictions.

    Returns
    -------
    InferenceConfig
        Pre-configured inference configuration.
    """
    return InferenceConfig(
        name="default_inference",
        checkpoint_dir=Path(checkpoint_dir),
        output_dir=Path(output_dir),
        bias_correction=BiasCorrection(
            correct_bm=correct_bm,
            correct_web=correct_web,
        ),
    )


__all__ = [
    # Base configs
    "BaseConfig",
    "BiasCorrection",
    # Model configs
    "ModelConfig",
    "CatBoostHyperparams",
    "FeatureConfig",
    # Training configs
    "TrainingConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "DataSplitConfig",
    # Factory functions
    "get_bm_multi_config",
    "get_bm_conversion_config",
    "get_web_multi_config",
    "get_surrogate_config",
    "get_default_config",
    "get_default_training_config",
    "get_default_inference_config",
    # Utilities
    "load_config",
]
