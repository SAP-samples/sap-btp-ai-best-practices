"""
Service layer for regressor API endpoints.

Provides thin wrapper functions that call the existing pipeline
convenience functions from the regressor module.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..models.regressor import (
    BiasCorrection,
    ChannelResult,
    MetricResult,
)

logger = logging.getLogger(__name__)


def normalize_channels(channels: List[str]) -> List[str]:
    """Normalize channel names to uppercase format (B&M, WEB)."""
    normalized = []
    for ch in channels:
        ch_upper = ch.upper()
        if ch_upper in ("BM", "B&M"):
            normalized.append("B&M")
        elif ch_upper == "WEB":
            normalized.append("WEB")
        else:
            normalized.append(ch)
    return normalized


def generate_training_data(
    horizons_start: int,
    horizons_end: int,
    model_variant: str,
    include_crm: bool,
    output_path: str,
) -> Dict[str, Any]:
    """
    Generate canonical training table and save to disk.

    Parameters
    ----------
    horizons_start : int
        Start of forecast horizon range.
    horizons_end : int
        End of forecast horizon range.
    model_variant : str
        Model variant: "A", "B", or "both".
    include_crm : bool
        Whether to include CRM demographic features.
    output_path : str
        Output path for CSV file(s).

    Returns
    -------
    dict
        Generation result with row count, columns, and output files.
    """
    from ..regressor.etl import build_canonical_training_table

    horizons = range(horizons_start, horizons_end + 1)
    output_path = Path(output_path)
    output_files = []
    total_rows = 0
    total_cols = 0

    if model_variant == "both":
        # Generate both Model A and Model B
        output_dir = output_path if output_path.is_dir() or not output_path.suffix else output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Model B
        logger.info("Generating Model B features...")
        df_b = build_canonical_training_table(
            horizons=horizons,
            model_variant="B",
            include_crm=include_crm,
        )
        model_b_path = output_dir / "model_b.csv"
        df_b.to_csv(model_b_path, index=False)
        output_files.append(str(model_b_path))
        total_rows += len(df_b)
        total_cols = max(total_cols, len(df_b.columns))
        logger.info(f"Model B: {len(df_b)} rows, {len(df_b.columns)} columns")

        # Model A
        logger.info("Generating Model A features...")
        df_a = build_canonical_training_table(
            horizons=horizons,
            model_variant="A",
            include_crm=include_crm,
        )
        model_a_path = output_dir / "model_a.csv"
        df_a.to_csv(model_a_path, index=False)
        output_files.append(str(model_a_path))
        total_rows += len(df_a)
        total_cols = max(total_cols, len(df_a.columns))
        logger.info(f"Model A: {len(df_a)} rows, {len(df_a.columns)} columns")

    else:
        # Generate single variant
        logger.info(f"Generating Model {model_variant} features...")
        df = build_canonical_training_table(
            horizons=horizons,
            model_variant=model_variant,
            include_crm=include_crm,
        )

        # Ensure output path has .csv extension
        if not output_path.suffix:
            output_path = output_path / f"model_{model_variant.lower()}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        output_files.append(str(output_path))
        total_rows = len(df)
        total_cols = len(df.columns)
        logger.info(f"Model {model_variant}: {len(df)} rows, {len(df.columns)} columns")

    return {
        "rows": total_rows,
        "columns": total_cols,
        "output_files": output_files,
    }


def train_models(
    model_b_path: str,
    model_a_path: Optional[str],
    output_dir: str,
    channels: List[str],
    bias_correction: BiasCorrection,
    train_surrogate: bool,
) -> Dict[str, Any]:
    """
    Train forecasting models using the training pipeline.

    Parameters
    ----------
    model_b_path : str
        Path to Model B CSV file.
    model_a_path : str, optional
        Path to Model A CSV file for explainability.
    output_dir : str
        Output directory for checkpoints and predictions.
    channels : List[str]
        Channels to train (B&M, WEB).
    bias_correction : BiasCorrection
        Bias correction configuration.
    train_surrogate : bool
        Whether to train surrogate models.

    Returns
    -------
    dict
        Training results with metrics and file paths.
    """
    from ..regressor.pipelines import train

    channels = normalize_channels(channels)

    logger.info(f"Starting training for channels: {channels}")
    logger.info(f"Model B path: {model_b_path}")
    logger.info(f"Model A path: {model_a_path}")
    logger.info(f"Output dir: {output_dir}")

    result = train(
        model_b_path=model_b_path,
        model_a_path=model_a_path,
        output_dir=output_dir,
        channels=channels,
        correct_bm=bias_correction.correct_bm,
        correct_web_sales=bias_correction.correct_web_sales,
        correct_web_aov=bias_correction.correct_web_aov,
        train_surrogate=train_surrogate,
    )

    # Build response
    output_path = Path(output_dir)
    output_files = []

    # Collect output files
    checkpoint_dir = output_path / "checkpoints"
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.cbm"):
            output_files.append(str(f))
        if (checkpoint_dir / "residual_stats.json").exists():
            output_files.append(str(checkpoint_dir / "residual_stats.json"))

    # Prediction files
    if (output_path / "predictions_bm.csv").exists():
        output_files.append(str(output_path / "predictions_bm.csv"))
    if (output_path / "predictions_web.csv").exists():
        output_files.append(str(output_path / "predictions_web.csv"))

    # Convert channel results
    bm_result = None
    web_result = None

    if result.bm_result:
        bm_result = ChannelResult(
            channel=result.bm_result.channel,
            train_samples=result.bm_result.train_samples,
            test_samples=result.bm_result.test_samples,
            rmse_sales=result.bm_result.rmse_sales,
            rmse_aov=result.bm_result.rmse_aov,
            rmse_conv=result.bm_result.rmse_conv,
            surrogate_r2=result.bm_result.surrogate_r2,
        )

    if result.web_result:
        web_result = ChannelResult(
            channel=result.web_result.channel,
            train_samples=result.web_result.train_samples,
            test_samples=result.web_result.test_samples,
            rmse_sales=result.web_result.rmse_sales,
            rmse_aov=result.web_result.rmse_aov,
            rmse_conv=None,
            surrogate_r2=result.web_result.surrogate_r2,
        )

    return {
        "bm_result": bm_result,
        "web_result": web_result,
        "checkpoint_dir": str(result.checkpoint_dir),
        "output_files": output_files,
    }


def run_inference(
    model_b_path: str,
    model_a_path: Optional[str],
    checkpoint_dir: str,
    output_dir: str,
    channels: List[str],
    bias_correction: BiasCorrection,
    run_explainability: bool,
) -> Dict[str, Any]:
    """
    Run inference using the inference pipeline.

    Parameters
    ----------
    model_b_path : str
        Path to Model B CSV file with features.
    model_a_path : str, optional
        Path to Model A CSV file for explainability.
    checkpoint_dir : str
        Path to model checkpoints.
    output_dir : str
        Output directory for predictions.
    channels : List[str]
        Channels to process (B&M, WEB).
    bias_correction : BiasCorrection
        Bias correction configuration.
    run_explainability : bool
        Whether to run SHAP explainability.

    Returns
    -------
    dict
        Inference results with prediction counts and file paths.
    """
    from ..regressor.pipelines import infer

    channels = normalize_channels(channels)

    logger.info(f"Starting inference for channels: {channels}")
    logger.info(f"Model B path: {model_b_path}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Output dir: {output_dir}")

    result = infer(
        model_b_path=model_b_path,
        model_a_path=model_a_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        channels=channels,
        correct_bm=bias_correction.correct_bm,
        correct_web_sales=bias_correction.correct_web_sales,
        correct_web_aov=bias_correction.correct_web_aov,
        run_explainability=run_explainability,
    )

    # Save predictions
    result.save()

    # Build response
    output_path = Path(output_dir)
    output_files = []

    bm_count = None
    web_count = None

    if result.bm_predictions is not None:
        bm_count = len(result.bm_predictions)
        output_files.append(str(output_path / "predictions_bm.csv"))

    if result.web_predictions is not None:
        web_count = len(result.web_predictions)
        output_files.append(str(output_path / "predictions_web.csv"))

    return {
        "bm_predictions_count": bm_count,
        "web_predictions_count": web_count,
        "output_files": output_files,
    }


def evaluate_predictions(
    predictions_bm_path: Optional[str],
    predictions_web_path: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate model predictions using the evaluation pipeline.

    Parameters
    ----------
    predictions_bm_path : str, optional
        Path to B&M predictions CSV.
    predictions_web_path : str, optional
        Path to WEB predictions CSV.
    output_dir : str
        Output directory for evaluation results.

    Returns
    -------
    dict
        Evaluation results with metrics and output file path.
    """
    from ..regressor.pipelines import evaluate

    logger.info("Starting evaluation")
    if predictions_bm_path:
        logger.info(f"B&M predictions: {predictions_bm_path}")
    if predictions_web_path:
        logger.info(f"WEB predictions: {predictions_web_path}")

    result = evaluate(
        predictions_bm_path=predictions_bm_path,
        predictions_web_path=predictions_web_path,
        output_dir=output_dir,
    )

    # Save evaluation summary
    result.save()

    # Convert metrics to response format
    metrics = [
        MetricResult(
            target=m.target,
            channel=m.channel,
            n_samples=m.n_samples,
            mae=m.mae,
            wmape=m.wmape,
            bias=m.bias,
            r2=m.r2,
        )
        for m in result.metrics
    ]

    output_file = str(Path(output_dir) / "evaluation_summary.csv")

    return {
        "metrics": metrics,
        "output_file": output_file,
    }
