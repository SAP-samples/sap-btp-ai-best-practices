"""
Evaluation Pipeline.

Evaluates model predictions against ground truth:
- MAE: Mean Absolute Error
- WMAPE: Weighted Mean Absolute Percentage Error
- Bias: Mean Error (signed)
- R2: Coefficient of Determination

Supports evaluation of:
- Log-scale predictions (direct model output)
- Business-scale predictions (back-transformed)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.regressor.configs import EvaluationConfig


@dataclass
class MetricResult:
    """Result of computing metrics for a single target."""

    target: str
    channel: str
    n_samples: int
    mae: float
    wmape: float
    bias: float
    r2: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "channel": self.channel,
            "n_samples": self.n_samples,
            "mae": self.mae,
            "wmape": self.wmape,
            "bias": self.bias,
            "r2": self.r2,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    metrics: List[MetricResult]
    output_dir: Optional[Path] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        return pd.DataFrame([m.to_dict() for m in self.metrics])

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Save evaluation summary to CSV."""
        out = Path(output_dir) if output_dir else self.output_dir
        if out is None:
            raise ValueError("Output directory not specified.")

        out.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(out / "evaluation_summary.csv", index=False)
        print(f"Saved evaluation summary to {out / 'evaluation_summary.csv'}")

    def print_summary(self) -> None:
        """Print formatted summary table."""
        df = self.to_dataframe()
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(df.round(4).to_string(index=False))
        print("=" * 80)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid transformation."""
    return 1 / (1 + np.exp(-x))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    channel: str,
) -> Optional[MetricResult]:
    """
    Compute evaluation metrics for a single target.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    target : str
        Target name.
    channel : str
        Channel name.

    Returns
    -------
    MetricResult or None
        Metrics result, or None if no valid data.
    """
    # Filter to valid (non-NaN) pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    # Compute metrics
    error = y_pred - y_true
    abs_error = np.abs(error)

    mae = abs_error.mean()

    # WMAPE - handle zero sum
    if y_true.sum() == 0:
        wmape = np.nan
    else:
        wmape = 100 * abs_error.sum() / np.abs(y_true).sum()

    bias = error.mean()

    # R2 - handle constant target
    ss_res = (error ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    return MetricResult(
        target=target,
        channel=channel,
        n_samples=len(y_true),
        mae=mae,
        wmape=wmape,
        bias=bias,
        r2=r2,
    )


class EvaluationPipeline:
    """
    Model evaluation pipeline.

    Computes metrics for predictions against ground truth across
    multiple targets and channels.

    Parameters
    ----------
    config : EvaluationConfig
        Evaluation configuration.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def run(
        self,
        predictions: pd.DataFrame,
        actuals: Optional[pd.DataFrame] = None,
        channel: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Execute evaluation pipeline.

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame with predictions. Should contain both predictions
            and actuals if actuals not provided separately.
        actuals : pd.DataFrame, optional
            DataFrame with ground truth. If not provided, uses columns
            from predictions DataFrame.
        channel : str, optional
            Channel to evaluate. If not provided, infers from data.

        Returns
        -------
        EvaluationResult
            Evaluation metrics.
        """
        if actuals is not None:
            # Merge predictions with actuals
            # This assumes they share key columns
            df = predictions.merge(actuals, how="inner")
        else:
            df = predictions.copy()

        # Determine channel
        if channel is None:
            if "channel" in df.columns:
                channels = df["channel"].unique().tolist()
            else:
                channels = ["Unknown"]
        else:
            channels = [channel]
            if "channel" in df.columns:
                df = df[df["channel"] == channel]

        metrics = []

        for ch in channels:
            if "channel" in df.columns:
                ch_df = df[df["channel"] == ch]
            else:
                ch_df = df

            # Evaluate each target
            ch_metrics = self._evaluate_channel(ch_df, ch)
            metrics.extend(ch_metrics)

        result = EvaluationResult(
            metrics=metrics,
            output_dir=self.config.output_dir,
        )

        return result

    def _evaluate_channel(
        self,
        df: pd.DataFrame,
        channel: str,
    ) -> List[MetricResult]:
        """Evaluate all targets for a channel."""
        metrics = []

        # Log-scale targets (direct model output)
        log_targets = [
            ("pred_log_sales", "label_log_sales", "Log Sales"),
            ("pred_log_aov", "label_log_aov", "Log AOV"),
            ("pred_logit_conversion", "label_logit_conversion", "Logit Conversion"),
        ]

        for pred_col, true_col, name in log_targets:
            if pred_col in df.columns and true_col in df.columns:
                m = compute_metrics(
                    df[true_col].values,
                    df[pred_col].values,
                    f"{channel} {name}",
                    channel,
                )
                if m:
                    metrics.append(m)

        # Business-scale targets (back-transformed)

        # Sales - Mean Corrected
        if "pred_sales_mean" in df.columns and "total_sales" in df.columns:
            m = compute_metrics(
                df["total_sales"].values,
                df["pred_sales_mean"].values,
                f"{channel} Sales (Mean)",
                channel,
            )
            if m:
                metrics.append(m)

        # Sales - Median (P50)
        if "pred_sales_p50" in df.columns and "total_sales" in df.columns:
            m = compute_metrics(
                df["total_sales"].values,
                df["pred_sales_p50"].values,
                f"{channel} Sales (Median)",
                channel,
            )
            if m:
                metrics.append(m)
        elif "pred_log_sales" in df.columns and "total_sales" in df.columns:
            # Compute median from log
            pred_median = np.exp(df["pred_log_sales"].values)
            m = compute_metrics(
                df["total_sales"].values,
                pred_median,
                f"{channel} Sales (Median)",
                channel,
            )
            if m:
                metrics.append(m)

        # AOV
        if "pred_aov_mean" in df.columns and "order_count" in df.columns and "total_sales" in df.columns:
            # Compute actual AOV
            mask = df["order_count"] > 0
            actual_aov = np.where(mask, df["total_sales"] / df["order_count"], np.nan)
            m = compute_metrics(
                actual_aov,
                df["pred_aov_mean"].values,
                f"{channel} AOV (Mean)",
                channel,
            )
            if m:
                metrics.append(m)

        if "pred_aov_p50" in df.columns and "order_count" in df.columns and "total_sales" in df.columns:
            mask = df["order_count"] > 0
            actual_aov = np.where(mask, df["total_sales"] / df["order_count"], np.nan)
            m = compute_metrics(
                actual_aov,
                df["pred_aov_p50"].values,
                f"{channel} AOV (Median)",
                channel,
            )
            if m:
                metrics.append(m)

        # Orders
        if "pred_log_orders" in df.columns and "order_count" in df.columns:
            pred_orders = np.exp(df["pred_log_orders"].values)
            m = compute_metrics(
                df["order_count"].values,
                pred_orders,
                f"{channel} Orders",
                channel,
            )
            if m:
                metrics.append(m)

        # Conversion (B&M only)
        if "pred_logit_conversion" in df.columns and "store_traffic" in df.columns:
            # Compute actual and predicted conversion
            mask = df["store_traffic"] > 0
            actual_conv = np.where(mask, df["order_count"] / df["store_traffic"], np.nan)
            pred_conv = sigmoid(df["pred_logit_conversion"].values)

            m = compute_metrics(
                actual_conv,
                pred_conv,
                f"{channel} Conversion",
                channel,
            )
            if m:
                metrics.append(m)

        # Traffic (B&M only)
        if "pred_traffic_p50" in df.columns and "store_traffic" in df.columns:
            m = compute_metrics(
                df["store_traffic"].values,
                df["pred_traffic_p50"].values,
                f"{channel} Traffic (P50)",
                channel,
            )
            if m:
                metrics.append(m)

        return metrics


def evaluate_bm(predictions_path: Union[str, Path]) -> EvaluationResult:
    """
    Evaluate B&M predictions.

    Parameters
    ----------
    predictions_path : str or Path
        Path to B&M predictions CSV.

    Returns
    -------
    EvaluationResult
        Evaluation metrics.
    """
    df = pd.read_csv(predictions_path)
    pipeline = EvaluationPipeline()
    return pipeline.run(df, channel="B&M")


def evaluate_web(predictions_path: Union[str, Path]) -> EvaluationResult:
    """
    Evaluate WEB predictions.

    Parameters
    ----------
    predictions_path : str or Path
        Path to WEB predictions CSV.

    Returns
    -------
    EvaluationResult
        Evaluation metrics.
    """
    df = pd.read_csv(predictions_path)
    pipeline = EvaluationPipeline()
    return pipeline.run(df, channel="WEB")


def evaluate(
    predictions_bm_path: Optional[str] = None,
    predictions_web_path: Optional[str] = None,
    output_dir: str = "output",
) -> EvaluationResult:
    """
    Convenience function to evaluate both channels.

    Parameters
    ----------
    predictions_bm_path : str, optional
        Path to B&M predictions CSV.
    predictions_web_path : str, optional
        Path to WEB predictions CSV.
    output_dir : str, optional
        Output directory. Default "output".

    Returns
    -------
    EvaluationResult
        Combined evaluation metrics.
    """
    all_metrics = []

    if predictions_bm_path:
        bm_result = evaluate_bm(predictions_bm_path)
        all_metrics.extend(bm_result.metrics)

    if predictions_web_path:
        web_result = evaluate_web(predictions_web_path)
        all_metrics.extend(web_result.metrics)

    result = EvaluationResult(
        metrics=all_metrics,
        output_dir=Path(output_dir),
    )

    return result
