"""
Inference Pipeline.

Loads saved models and generates predictions on new data:
1. Load model checkpoints
2. Load residual statistics for quantile computation
3. Score data per channel
4. Apply bias correction and compute quantiles
5. Generate traffic estimates (B&M)
6. Apply surrogate explainability (optional)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from app.regressor.configs import InferenceConfig, BiasCorrection
from app.regressor.models import (
    BMPredictor,
    WEBPredictor,
    SurrogateExplainer,
    TrafficEstimator,
)


@dataclass
class InferenceResult:
    """Result of inference pipeline."""

    bm_predictions: Optional[pd.DataFrame] = None
    web_predictions: Optional[pd.DataFrame] = None
    output_dir: Optional[Path] = None

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Save predictions to CSV files."""
        out = Path(output_dir) if output_dir else self.output_dir
        if out is None:
            raise ValueError("Output directory not specified.")

        out.mkdir(parents=True, exist_ok=True)

        if self.bm_predictions is not None:
            self.bm_predictions.to_csv(out / "predictions_bm.csv", index=False)

        if self.web_predictions is not None:
            self.web_predictions.to_csv(out / "predictions_web.csv", index=False)


class InferencePipeline:
    """
    Production inference pipeline.

    Loads trained models from checkpoints and generates predictions
    on new data with quantiles and optional explainability.

    Parameters
    ----------
    config : InferenceConfig
        Inference configuration with checkpoint paths and options.
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.bm_predictor: Optional[BMPredictor] = None
        self.web_predictor: Optional[WEBPredictor] = None
        self.bm_surrogate: Optional[SurrogateExplainer] = None
        self.web_surrogate: Optional[SurrogateExplainer] = None
        self.residual_stats: dict = {}

    def _load_models(self) -> None:
        """Load model checkpoints."""
        checkpoint_dir = self.config.checkpoint_dir

        # Load B&M models
        if "B&M" in self.config.channels:
            bm_multi_path = checkpoint_dir / "bm_multi.cbm"
            bm_conv_path = checkpoint_dir / "bm_conversion.cbm"

            if bm_multi_path.exists() and bm_conv_path.exists():
                self.bm_predictor = BMPredictor()
                self.bm_predictor.load_models(checkpoint_dir)
                print(f"Loaded B&M models from {checkpoint_dir}")
            else:
                print(f"Warning: B&M model files not found in {checkpoint_dir}")

        # Load WEB models
        if "WEB" in self.config.channels:
            web_multi_path = checkpoint_dir / "web_multi.cbm"

            if web_multi_path.exists():
                self.web_predictor = WEBPredictor()
                self.web_predictor.load_models(checkpoint_dir)
                print(f"Loaded WEB models from {checkpoint_dir}")
            else:
                print(f"Warning: WEB model files not found in {checkpoint_dir}")

        # Load residual stats
        residual_path = checkpoint_dir / "residual_stats.json"
        if residual_path.exists():
            with open(residual_path, "r") as f:
                self.residual_stats = json.load(f)
            print(f"Loaded residual stats from {residual_path}")
        else:
            print(f"Warning: Residual stats not found at {residual_path}")
            self.residual_stats = {
                "bm": {"rmse_sales": 0.0, "rmse_aov": 0.0, "rmse_conv": 0.0},
                "web": {"rmse_sales": 0.0, "rmse_aov": 0.0},
            }

        # Set RMSE values on predictors
        if self.bm_predictor:
            bm_stats = self.residual_stats.get("bm", {})
            self.bm_predictor.set_rmse(
                sales=bm_stats.get("rmse_sales", 0.0),
                aov=bm_stats.get("rmse_aov", 0.0),
                conv=bm_stats.get("rmse_conv", 0.0),
            )

        if self.web_predictor:
            web_stats = self.residual_stats.get("web", {})
            self.web_predictor.set_rmse(
                sales=web_stats.get("rmse_sales", 0.0),
                aov=web_stats.get("rmse_aov", 0.0),
            )

        # Load surrogate models if explainability is enabled
        if self.config.run_explainability:
            self._load_surrogate_models()

    def _load_surrogate_models(self) -> None:
        """Load surrogate models for explainability."""
        checkpoint_dir = self.config.checkpoint_dir

        # B&M surrogate
        bm_surrogate_path = checkpoint_dir / "surrogate_bm.cbm"
        bm_meta_path = checkpoint_dir / "surrogate_bm.meta.json"
        if bm_surrogate_path.exists():
            self.bm_surrogate = SurrogateExplainer(channel="B&M")
            self.bm_surrogate.load_model(bm_surrogate_path, meta_path=bm_meta_path)
            print(f"Loaded B&M surrogate model from {bm_surrogate_path}")

        # WEB surrogate
        web_surrogate_path = checkpoint_dir / "surrogate_web.cbm"
        web_meta_path = checkpoint_dir / "surrogate_web.meta.json"
        if web_surrogate_path.exists():
            self.web_surrogate = SurrogateExplainer(channel="WEB")
            self.web_surrogate.load_model(web_surrogate_path, meta_path=web_meta_path)
            print(f"Loaded WEB surrogate model from {web_surrogate_path}")

    def run(
        self,
        model_b_data: pd.DataFrame,
        model_a_data: Optional[pd.DataFrame] = None,
        channels: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Execute the inference pipeline.

        Parameters
        ----------
        model_b_data : pd.DataFrame
            Model B features (full autoregressive).
        model_a_data : pd.DataFrame, optional
            Model A features (business levers) for explainability.
        channels : List[str], optional
            Channels to process. Defaults to config channels.

        Returns
        -------
        InferenceResult
            Predictions for each channel.
        """
        channels = channels or self.config.channels
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load models if not already loaded
        if self.bm_predictor is None and self.web_predictor is None:
            self._load_models()

        # Parse dates
        if "origin_week_date" in model_b_data.columns:
            model_b_data = model_b_data.copy()
            model_b_data["origin_week_date"] = pd.to_datetime(model_b_data["origin_week_date"])

        if model_a_data is not None and "origin_week_date" in model_a_data.columns:
            model_a_data = model_a_data.copy()
            model_a_data["origin_week_date"] = pd.to_datetime(model_a_data["origin_week_date"])

        result = InferenceResult(output_dir=output_dir)

        # Process B&M channel
        if "B&M" in channels and self.bm_predictor is not None:
            print("\n=== Inference: B&M Channel ===")
            result.bm_predictions = self._infer_bm_channel(
                model_b_data, model_a_data, output_dir
            )

        # Process WEB channel
        if "WEB" in channels and self.web_predictor is not None:
            print("\n=== Inference: WEB Channel ===")
            result.web_predictions = self._infer_web_channel(
                model_b_data, model_a_data, output_dir
            )

        # Save results
        result.save()

        print("\nInference complete.")
        return result

    def _infer_bm_channel(
        self,
        model_b_data: pd.DataFrame,
        model_a_data: Optional[pd.DataFrame],
        output_dir: Path,
    ) -> pd.DataFrame:
        """Generate B&M predictions."""
        # Filter to B&M data
        bm_mask = model_b_data["channel"] == "B&M"
        df_b_bm = model_b_data[bm_mask].copy()

        print(f"Scoring {len(df_b_bm)} B&M samples...")

        # Generate predictions
        preds = self.bm_predictor.predict(df_b_bm, estimate_traffic=True)

        # Build result DataFrame
        res_df = df_b_bm.copy()
        res_df = self._add_bm_predictions(res_df, preds)

        # Apply explainability if available
        if model_a_data is not None and self.bm_surrogate is not None:
            res_df = self._apply_bm_explainability(res_df, model_a_data, output_dir)

        return res_df

    def _add_bm_predictions(
        self,
        df: pd.DataFrame,
        preds,
    ) -> pd.DataFrame:
        """Add B&M predictions and quantiles to DataFrame."""
        # Log predictions
        df["pred_log_sales"] = preds.log_sales
        df["pred_log_aov"] = preds.log_aov
        df["pred_log_orders"] = preds.log_orders
        df["pred_logit_conversion"] = preds.logit_conversion

        # Get RMSE values
        bm_stats = self.residual_stats.get("bm", {})
        rmse_sales = bm_stats.get("rmse_sales", 0.0)
        rmse_aov = bm_stats.get("rmse_aov", 0.0)

        # Quantiles
        z90 = 1.2815515655446004
        bias = self.config.bias_correction

        # Sales
        sigma_sales = 0.5 * rmse_sales**2 if bias.should_correct_bm() else 0.0
        df["pred_sales_mean"] = np.exp(preds.log_sales + sigma_sales)
        df["pred_sales_p50"] = np.exp(preds.log_sales)
        df["pred_sales_p90"] = np.exp(preds.log_sales + z90 * rmse_sales)

        # AOV
        sigma_aov = 0.5 * rmse_aov**2 if bias.should_correct_bm() else 0.0
        df["pred_aov_mean"] = np.exp(preds.log_aov + sigma_aov)
        df["pred_aov_p50"] = np.exp(preds.log_aov)
        df["pred_aov_p90"] = np.exp(preds.log_aov + z90 * rmse_aov)

        # Traffic
        if preds.traffic is not None:
            df["pred_traffic_p10"] = preds.traffic.p10
            df["pred_traffic_p50"] = preds.traffic.p50
            df["pred_traffic_p90"] = preds.traffic.p90

        return df

    def _apply_bm_explainability(
        self,
        res_df: pd.DataFrame,
        model_a_data: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        """Apply B&M surrogate explainability."""
        keys = self.config.key_columns

        # Filter Model A to B&M
        df_a_bm = model_a_data[model_a_data["channel"] == "B&M"]

        # Merge
        explain_df = res_df[keys + ["pred_log_sales"]].merge(
            df_a_bm, on=keys, how="inner"
        )

        if len(explain_df) == 0:
            print("No matching Model A data for B&M explainability.")
            return res_df

        # Generate explanations
        contributor_df = self.bm_surrogate.explain(
            df=explain_df,
            output_dir=output_dir,
            name="BM_Sales",
            keys=keys,
            target_label="pred_log_sales",
            top_k_contributors=self.config.top_k_contributors,
            cohort_cols=["profit_center_nbr", "dma", "is_outlet", "is_comp_store"],
            dependence_top_n=8,
            color_feature="horizon",
        )

        if contributor_df is not None:
            res_df = res_df.merge(contributor_df, on=keys, how="left")

        return res_df

    def _infer_web_channel(
        self,
        model_b_data: pd.DataFrame,
        model_a_data: Optional[pd.DataFrame],
        output_dir: Path,
    ) -> pd.DataFrame:
        """Generate WEB predictions."""
        # Filter to WEB data
        web_mask = model_b_data["channel"] == "WEB"
        df_b_web = model_b_data[web_mask].copy()

        print(f"Scoring {len(df_b_web)} WEB samples...")

        # Generate predictions
        preds = self.web_predictor.predict(df_b_web)

        # Build result DataFrame
        res_df = df_b_web.copy()
        res_df = self._add_web_predictions(res_df, preds)

        # Apply explainability if available
        if model_a_data is not None and self.web_surrogate is not None:
            res_df = self._apply_web_explainability(res_df, model_a_data, output_dir)

        return res_df

    def _add_web_predictions(
        self,
        df: pd.DataFrame,
        preds,
    ) -> pd.DataFrame:
        """Add WEB predictions and quantiles to DataFrame."""
        # Log predictions
        df["pred_log_sales"] = preds.log_sales
        df["pred_log_aov"] = preds.log_aov
        df["pred_log_orders"] = preds.log_orders

        # Get RMSE values
        web_stats = self.residual_stats.get("web", {})
        rmse_sales = web_stats.get("rmse_sales", 0.0)
        rmse_aov = web_stats.get("rmse_aov", 0.0)

        # Quantiles
        z90 = 1.2815515655446004
        bias = self.config.bias_correction

        # Sales
        use_sales_corr = bias.should_correct_web_sales()
        sigma_sales = 0.5 * rmse_sales**2 if use_sales_corr else 0.0
        df["pred_sales_mean"] = np.exp(preds.log_sales + sigma_sales)
        df["pred_sales_p50"] = np.exp(preds.log_sales)
        df["pred_sales_p90"] = np.exp(preds.log_sales + z90 * rmse_sales)

        # AOV
        use_aov_corr = bias.should_correct_web_aov()
        sigma_aov = 0.5 * rmse_aov**2 if use_aov_corr else 0.0
        df["pred_aov_mean"] = np.exp(preds.log_aov + sigma_aov)
        df["pred_aov_p50"] = np.exp(preds.log_aov)
        df["pred_aov_p90"] = np.exp(preds.log_aov + z90 * rmse_aov)

        return df

    def _apply_web_explainability(
        self,
        res_df: pd.DataFrame,
        model_a_data: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        """Apply WEB surrogate explainability."""
        keys = self.config.key_columns

        # Filter Model A to WEB
        df_a_web = model_a_data[model_a_data["channel"] == "WEB"]

        # Merge
        explain_df = res_df[keys + ["pred_log_sales"]].merge(
            df_a_web, on=keys, how="inner"
        )

        if len(explain_df) == 0:
            print("No matching Model A data for WEB explainability.")
            return res_df

        # Generate explanations
        contributor_df = self.web_surrogate.explain(
            df=explain_df,
            output_dir=output_dir,
            name="WEB_Sales",
            keys=keys,
            target_label="pred_log_sales",
            top_k_contributors=self.config.top_k_contributors,
            cohort_cols=["profit_center_nbr", "dma"],
            dependence_top_n=8,
            color_feature="horizon",
        )

        if contributor_df is not None:
            res_df = res_df.merge(contributor_df, on=keys, how="left")

        return res_df


def infer(
    model_b_path: str,
    model_a_path: Optional[str] = None,
    checkpoint_dir: str = "output/checkpoints",
    output_dir: str = "output_infer",
    channels: Optional[List[str]] = None,
    correct_bm: bool = False,
    correct_web: bool = False,
    correct_web_sales: bool = False,
    correct_web_aov: bool = False,
    run_explainability: bool = True,
) -> InferenceResult:
    """
    Convenience function to run the inference pipeline.

    Parameters
    ----------
    model_b_path : str
        Path to Model B CSV file.
    model_a_path : str, optional
        Path to Model A CSV file for explainability.
    checkpoint_dir : str, optional
        Path to model checkpoints. Default "output/checkpoints".
    output_dir : str, optional
        Output directory. Default "output_infer".
    channels : List[str], optional
        Channels to process. Default ["B&M", "WEB"].
    correct_bm : bool, optional
        Apply bias correction to B&M predictions.
    correct_web : bool, optional
        Apply bias correction to all WEB predictions.
    correct_web_sales : bool, optional
        Apply bias correction to WEB Sales only.
    correct_web_aov : bool, optional
        Apply bias correction to WEB AOV only.
    run_explainability : bool, optional
        Whether to run explainability analysis. Default True.

    Returns
    -------
    InferenceResult
        Inference result with predictions.
    """
    # Load data
    model_b_data = pd.read_csv(model_b_path)
    model_a_data = pd.read_csv(model_a_path) if model_a_path else None

    # Create config
    config = InferenceConfig(
        checkpoint_dir=Path(checkpoint_dir),
        output_dir=Path(output_dir),
        channels=channels or ["B&M", "WEB"],
        bias_correction=BiasCorrection(
            correct_bm=correct_bm,
            correct_web=correct_web,
            correct_web_sales=correct_web_sales,
            correct_web_aov=correct_web_aov,
        ),
        run_explainability=run_explainability,
    )

    # Run pipeline
    pipeline = InferencePipeline(config)
    return pipeline.run(model_b_data, model_a_data)
