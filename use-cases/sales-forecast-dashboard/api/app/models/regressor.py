"""
Pydantic models for regressor API endpoints.

Defines request and response schemas for:
- Generate: Create canonical training data
- Train: Train forecasting models
- Infer: Run inference with trained models
- Evaluate: Evaluate model predictions
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Shared Models
# =============================================================================


class BiasCorrection(BaseModel):
    """Configuration for bias correction in predictions."""

    correct_bm: bool = Field(
        default=False,
        description="Apply bias correction to B&M predictions",
    )
    correct_web_sales: bool = Field(
        default=False,
        description="Apply bias correction to WEB Sales predictions",
    )
    correct_web_aov: bool = Field(
        default=False,
        description="Apply bias correction to WEB AOV predictions",
    )


class ChannelResult(BaseModel):
    """Training result for a single channel."""

    channel: str = Field(description="Channel name (B&M or WEB)")
    train_samples: int = Field(description="Number of training samples")
    test_samples: int = Field(description="Number of test samples")
    rmse_sales: float = Field(description="RMSE for log-scale sales predictions")
    rmse_aov: float = Field(description="RMSE for log-scale AOV predictions")
    rmse_conv: Optional[float] = Field(
        default=None,
        description="RMSE for logit-scale conversion predictions (B&M only)",
    )
    surrogate_r2: Optional[float] = Field(
        default=None,
        description="R2 score for surrogate model (if trained)",
    )


class MetricResult(BaseModel):
    """Evaluation metrics for a single target."""

    target: str = Field(description="Target variable name")
    channel: str = Field(description="Channel name (B&M or WEB)")
    n_samples: int = Field(description="Number of samples evaluated")
    mae: float = Field(description="Mean Absolute Error")
    wmape: float = Field(description="Weighted Mean Absolute Percentage Error")
    bias: float = Field(description="Mean signed error (bias)")
    r2: float = Field(description="Coefficient of determination (R2)")


# =============================================================================
# Generate Endpoint Models
# =============================================================================


class GenerateRequest(BaseModel):
    """Request schema for generating canonical training data."""

    horizons_start: int = Field(
        default=1,
        ge=1,
        le=52,
        description="Start of forecast horizon range (1-52 weeks)",
    )
    horizons_end: int = Field(
        default=52,
        ge=1,
        le=52,
        description="End of forecast horizon range (1-52 weeks)",
    )
    model_variant: Literal["A", "B", "both"] = Field(
        default="both",
        description="Model variant to generate: A (explainability), B (full features), or both",
    )
    include_crm: bool = Field(
        default=False,
        description="Include CRM demographic features",
    )
    output_path: str = Field(
        description="Output path for generated CSV file(s). If 'both' variant, will create model_a.csv and model_b.csv",
    )


class GenerateResponse(BaseModel):
    """Response schema for generate endpoint."""

    success: bool = Field(description="Whether generation completed successfully")
    rows_generated: int = Field(description="Total number of rows generated")
    columns: int = Field(description="Number of columns in output")
    output_files: List[str] = Field(description="List of output file paths")
    message: str = Field(description="Status message")


# =============================================================================
# Train Endpoint Models
# =============================================================================


class TrainRequest(BaseModel):
    """Request schema for training forecasting models."""

    model_b_path: str = Field(
        description="Path to Model B CSV file (full autoregressive features)",
    )
    model_a_path: Optional[str] = Field(
        default=None,
        description="Path to Model A CSV file (business levers for explainability)",
    )
    output_dir: str = Field(
        default="output",
        description="Output directory for checkpoints and predictions",
    )
    channels: List[str] = Field(
        default=["B&M", "WEB"],
        description="Channels to train. Valid values: B&M, WEB, bm, web",
    )
    bias_correction: BiasCorrection = Field(
        default_factory=BiasCorrection,
        description="Bias correction configuration",
    )
    train_surrogate: bool = Field(
        default=True,
        description="Whether to train surrogate models for SHAP explainability",
    )


class TrainResponse(BaseModel):
    """Response schema for train endpoint."""

    success: bool = Field(description="Whether training completed successfully")
    bm_result: Optional[ChannelResult] = Field(
        default=None,
        description="Training results for B&M channel",
    )
    web_result: Optional[ChannelResult] = Field(
        default=None,
        description="Training results for WEB channel",
    )
    checkpoint_dir: str = Field(description="Path to saved model checkpoints")
    output_files: List[str] = Field(description="List of output file paths")
    message: str = Field(description="Status message")


# =============================================================================
# Infer Endpoint Models
# =============================================================================


class InferRequest(BaseModel):
    """Request schema for running inference."""

    model_b_path: str = Field(
        description="Path to Model B CSV file with features for inference",
    )
    model_a_path: Optional[str] = Field(
        default=None,
        description="Path to Model A CSV file (required for explainability)",
    )
    checkpoint_dir: str = Field(
        default="output/checkpoints",
        description="Path to directory containing model checkpoints",
    )
    output_dir: str = Field(
        default="output_infer",
        description="Output directory for predictions",
    )
    channels: List[str] = Field(
        default=["B&M", "WEB"],
        description="Channels to run inference for. Valid values: B&M, WEB, bm, web",
    )
    bias_correction: BiasCorrection = Field(
        default_factory=BiasCorrection,
        description="Bias correction configuration",
    )
    run_explainability: bool = Field(
        default=True,
        description="Whether to run SHAP explainability analysis (requires model_a_path)",
    )


class InferResponse(BaseModel):
    """Response schema for inference endpoint."""

    success: bool = Field(description="Whether inference completed successfully")
    bm_predictions_count: Optional[int] = Field(
        default=None,
        description="Number of B&M predictions generated",
    )
    web_predictions_count: Optional[int] = Field(
        default=None,
        description="Number of WEB predictions generated",
    )
    output_files: List[str] = Field(description="List of output file paths")
    message: str = Field(description="Status message")


# =============================================================================
# Evaluate Endpoint Models
# =============================================================================


class EvaluateRequest(BaseModel):
    """Request schema for evaluating predictions."""

    predictions_bm_path: Optional[str] = Field(
        default=None,
        description="Path to B&M predictions CSV file",
    )
    predictions_web_path: Optional[str] = Field(
        default=None,
        description="Path to WEB predictions CSV file",
    )
    output_dir: str = Field(
        default="output",
        description="Output directory for evaluation results",
    )


class EvaluateResponse(BaseModel):
    """Response schema for evaluation endpoint."""

    success: bool = Field(description="Whether evaluation completed successfully")
    metrics: List[MetricResult] = Field(
        description="List of evaluation metrics by target and channel",
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Path to evaluation summary CSV file",
    )
    message: str = Field(description="Status message")
