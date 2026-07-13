"""
Training Pipeline.

Orchestrates the complete training workflow:
1. Data loading and splitting
2. Channel-specific model training (B&M, WEB)
3. RMSE computation for quantile estimation
4. Surrogate model training for explainability
5. Checkpoint saving
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.regressor.configs import TrainingConfig, BiasCorrection
from app.regressor.models import (
    BMPredictor,
    WEBPredictor,
    SurrogateExplainer,
    BMPredictionResult,
    WEBPredictionResult,
)


@dataclass
class ChannelTrainingResult:
    """Result of training a single channel."""

    channel: str
    train_samples: int
    test_samples: int
    rmse_sales: float = 0.0
    rmse_aov: float = 0.0
    rmse_conv: float = 0.0  # B&M only
    surrogate_r2: float = 0.0


@dataclass
class TrainingResult:
    """Complete training pipeline result."""

    bm_result: Optional[ChannelTrainingResult] = None
    web_result: Optional[ChannelTrainingResult] = None
    output_dir: Path = field(default_factory=lambda: Path("output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("output/checkpoints"))

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
        }
        if self.bm_result:
            result["bm"] = {
                "train_samples": self.bm_result.train_samples,
                "test_samples": self.bm_result.test_samples,
                "rmse_sales": self.bm_result.rmse_sales,
                "rmse_aov": self.bm_result.rmse_aov,
                "rmse_conv": self.bm_result.rmse_conv,
                "surrogate_r2": self.bm_result.surrogate_r2,
            }
        if self.web_result:
            result["web"] = {
                "train_samples": self.web_result.train_samples,
                "test_samples": self.web_result.test_samples,
                "rmse_sales": self.web_result.rmse_sales,
                "rmse_aov": self.web_result.rmse_aov,
                "surrogate_r2": self.web_result.surrogate_r2,
            }
        return result


class TrainingPipeline:
    """
    Production training pipeline.

    Orchestrates training of:
    - B&M channel: Multi-objective (Sales, AOV, Orders) + Conversion
    - WEB channel: Multi-objective (Sales, AOV, Orders)
    - Surrogate models for SHAP-based explainability

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.bm_predictor: Optional[BMPredictor] = None
        self.web_predictor: Optional[WEBPredictor] = None
        self.bm_surrogate: Optional[SurrogateExplainer] = None
        self.web_surrogate: Optional[SurrogateExplainer] = None

    def run(
        self,
        model_b_data: pd.DataFrame,
        model_a_data: Optional[pd.DataFrame] = None,
        channels: Optional[List[str]] = None,
    ) -> TrainingResult:
        """
        Execute the full training pipeline.

        Parameters
        ----------
        model_b_data : pd.DataFrame
            Model B features (full autoregressive).
        model_a_data : pd.DataFrame, optional
            Model A features (business levers) for explainability.
        channels : List[str], optional
            Channels to train. Defaults to config channels.

        Returns
        -------
        TrainingResult
            Complete training result with metrics and paths.
        """
        channels = channels or self.config.channels
        output_dir = self.config.output_dir
        checkpoint_dir = self.config.checkpoint_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        result = TrainingResult(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

        residual_meta = {}

        # Parse dates
        if "origin_week_date" in model_b_data.columns:
            model_b_data = model_b_data.copy()
            model_b_data["origin_week_date"] = pd.to_datetime(model_b_data["origin_week_date"])

        if model_a_data is not None and "origin_week_date" in model_a_data.columns:
            model_a_data = model_a_data.copy()
            model_a_data["origin_week_date"] = pd.to_datetime(model_a_data["origin_week_date"])

        # Train B&M channel
        if "B&M" in channels:
            print("\n" + "=" * 60)
            print("Processing B&M Channel")
            print("=" * 60)

            bm_result, bm_preds_df = self._train_bm_channel(
                model_b_data, model_a_data, output_dir, checkpoint_dir
            )
            result.bm_result = bm_result
            residual_meta["bm"] = {
                "rmse_sales": bm_result.rmse_sales,
                "rmse_aov": bm_result.rmse_aov,
                "rmse_conv": bm_result.rmse_conv,
            }

        # Train WEB channel
        if "WEB" in channels:
            print("\n" + "=" * 60)
            print("Processing WEB Channel")
            print("=" * 60)

            web_result, web_preds_df = self._train_web_channel(
                model_b_data, model_a_data, output_dir, checkpoint_dir
            )
            result.web_result = web_result
            residual_meta["web"] = {
                "rmse_sales": web_result.rmse_sales,
                "rmse_aov": web_result.rmse_aov,
            }

        # Save residual stats for inference
        with open(checkpoint_dir / "residual_stats.json", "w") as f:
            json.dump(residual_meta, f, indent=2)

        print("\n" + "=" * 60)
        print("Training Pipeline Complete")
        print("=" * 60)

        return result

    def _train_bm_channel(
        self,
        model_b_data: pd.DataFrame,
        model_a_data: Optional[pd.DataFrame],
        output_dir: Path,
        checkpoint_dir: Path,
    ) -> Tuple[ChannelTrainingResult, pd.DataFrame]:
        """Train B&M channel models."""
        # Filter to B&M data
        bm_mask = model_b_data["channel"] == "B&M"
        df_b_bm = model_b_data[bm_mask].copy()

        # Split by time
        train_mask = self.config.data_split.get_train_mask(df_b_bm)
        test_mask = self.config.data_split.get_test_mask(df_b_bm)
        train_df = df_b_bm[train_mask]
        test_df = df_b_bm[test_mask]

        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # Train predictor
        self.bm_predictor = BMPredictor(iterations=5000)
        self.bm_predictor.fit(train_df, val_df=test_df)

        # Generate predictions
        preds = self.bm_predictor.predict(test_df, estimate_traffic=True)

        # Build result DataFrame
        res_df = self._build_bm_result_df(test_df, preds)

        # Save predictions
        res_df.to_csv(output_dir / "predictions_bm.csv", index=False)

        # Save models
        self.bm_predictor.save_models(checkpoint_dir)

        # Train surrogate for explainability
        surrogate_r2 = 0.0
        if model_a_data is not None and self.config.train_surrogate:
            surrogate_r2 = self._train_bm_surrogate(
                res_df, model_a_data, output_dir, checkpoint_dir
            )

        return ChannelTrainingResult(
            channel="B&M",
            train_samples=len(train_df),
            test_samples=len(test_df),
            rmse_sales=self.bm_predictor.rmse_sales,
            rmse_aov=self.bm_predictor.rmse_aov,
            rmse_conv=self.bm_predictor.rmse_conv,
            surrogate_r2=surrogate_r2,
        ), res_df

    def _build_bm_result_df(
        self,
        test_df: pd.DataFrame,
        preds: BMPredictionResult,
    ) -> pd.DataFrame:
        """Build B&M results DataFrame with predictions and quantiles."""
        res_df = test_df.copy()

        # Log predictions
        res_df["pred_log_sales"] = preds.log_sales
        res_df["pred_log_aov"] = preds.log_aov
        res_df["pred_log_orders"] = preds.log_orders
        res_df["pred_logit_conversion"] = preds.logit_conversion

        # Quantiles
        z90 = 1.2815515655446004
        bias = self.config.bias_correction

        # Sales quantiles
        sigma_sales = 0.5 * preds.rmse_sales**2 if bias.should_correct_bm() else 0.0
        res_df["pred_sales_mean"] = np.exp(preds.log_sales + sigma_sales)
        res_df["pred_sales_p50"] = np.exp(preds.log_sales)
        res_df["pred_sales_p90"] = np.exp(preds.log_sales + z90 * preds.rmse_sales)

        # AOV quantiles
        sigma_aov = 0.5 * preds.rmse_aov**2 if bias.should_correct_bm() else 0.0
        res_df["pred_aov_mean"] = np.exp(preds.log_aov + sigma_aov)
        res_df["pred_aov_p50"] = np.exp(preds.log_aov)
        res_df["pred_aov_p90"] = np.exp(preds.log_aov + z90 * preds.rmse_aov)

        # Traffic quantiles
        if preds.traffic is not None:
            res_df["pred_traffic_p10"] = preds.traffic.p10
            res_df["pred_traffic_p50"] = preds.traffic.p50
            res_df["pred_traffic_p90"] = preds.traffic.p90

        return res_df

    def _train_bm_surrogate(
        self,
        res_df: pd.DataFrame,
        model_a_data: pd.DataFrame,
        output_dir: Path,
        checkpoint_dir: Path,
    ) -> float:
        """Train B&M surrogate model for explainability."""
        keys = self.config.key_columns

        # Filter Model A to B&M channel
        df_a_bm = model_a_data[model_a_data["channel"] == "B&M"]

        # Merge to ensure alignment
        explain_df = res_df[keys + ["pred_log_sales"]].merge(
            df_a_bm, on=keys, how="inner"
        )

        if len(explain_df) == 0:
            print("Warning: No matching Model A data for B&M explainability.")
            return 0.0

        self.bm_surrogate = SurrogateExplainer(channel="B&M")
        self.bm_surrogate.fit(
            model_a_df=explain_df,
            model_b_predictions=explain_df["pred_log_sales"].values
        )

        # Save surrogate model
        self.bm_surrogate.save_model(
            model_path=checkpoint_dir / "surrogate_bm.cbm",
            meta_path=checkpoint_dir / "surrogate_bm.meta.json",
        )

        # Generate SHAP analysis
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

        # Merge contributors back to results if available
        if contributor_df is not None:
            # Re-save with contributors
            res_merged = res_df.merge(contributor_df, on=keys, how="left")
            res_merged.to_csv(output_dir / "predictions_bm.csv", index=False)

        return self.bm_surrogate.r2_score_value

    def _train_web_channel(
        self,
        model_b_data: pd.DataFrame,
        model_a_data: Optional[pd.DataFrame],
        output_dir: Path,
        checkpoint_dir: Path,
    ) -> Tuple[ChannelTrainingResult, pd.DataFrame]:
        """Train WEB channel models."""
        # Filter to WEB data
        web_mask = model_b_data["channel"] == "WEB"
        df_b_web = model_b_data[web_mask].copy()

        # Split by time
        train_mask = self.config.data_split.get_train_mask(df_b_web)
        test_mask = self.config.data_split.get_test_mask(df_b_web)
        train_df = df_b_web[train_mask]
        test_df = df_b_web[test_mask]

        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # Train predictor
        self.web_predictor = WEBPredictor(iterations=5000)
        self.web_predictor.fit(train_df, val_df=test_df)

        # Generate predictions
        preds = self.web_predictor.predict(test_df)

        # Build result DataFrame
        res_df = self._build_web_result_df(test_df, preds)

        # Save predictions
        res_df.to_csv(output_dir / "predictions_web.csv", index=False)

        # Save models
        self.web_predictor.save_models(checkpoint_dir)

        # Train surrogate for explainability
        surrogate_r2 = 0.0
        if model_a_data is not None and self.config.train_surrogate:
            surrogate_r2 = self._train_web_surrogate(
                res_df, model_a_data, output_dir, checkpoint_dir
            )

        return ChannelTrainingResult(
            channel="WEB",
            train_samples=len(train_df),
            test_samples=len(test_df),
            rmse_sales=self.web_predictor.rmse_sales,
            rmse_aov=self.web_predictor.rmse_aov,
            surrogate_r2=surrogate_r2,
        ), res_df

    def _build_web_result_df(
        self,
        test_df: pd.DataFrame,
        preds: WEBPredictionResult,
    ) -> pd.DataFrame:
        """Build WEB results DataFrame with predictions and quantiles."""
        res_df = test_df.copy()

        # Log predictions
        res_df["pred_log_sales"] = preds.log_sales
        res_df["pred_log_aov"] = preds.log_aov
        res_df["pred_log_orders"] = preds.log_orders

        # Quantiles
        z90 = 1.2815515655446004
        bias = self.config.bias_correction

        # Sales quantiles
        sigma_sales = 0.5 * preds.rmse_sales**2 if bias.should_correct_web_sales() else 0.0
        res_df["pred_sales_mean"] = np.exp(preds.log_sales + sigma_sales)
        res_df["pred_sales_p50"] = np.exp(preds.log_sales)
        res_df["pred_sales_p90"] = np.exp(preds.log_sales + z90 * preds.rmse_sales)

        # AOV quantiles
        sigma_aov = 0.5 * preds.rmse_aov**2 if bias.should_correct_web_aov() else 0.0
        res_df["pred_aov_mean"] = np.exp(preds.log_aov + sigma_aov)
        res_df["pred_aov_p50"] = np.exp(preds.log_aov)
        res_df["pred_aov_p90"] = np.exp(preds.log_aov + z90 * preds.rmse_aov)

        return res_df

    def _train_web_surrogate(
        self,
        res_df: pd.DataFrame,
        model_a_data: pd.DataFrame,
        output_dir: Path,
        checkpoint_dir: Path,
    ) -> float:
        """Train WEB surrogate model for explainability."""
        keys = self.config.key_columns

        # Filter Model A to WEB channel
        df_a_web = model_a_data[model_a_data["channel"] == "WEB"]

        # Merge to ensure alignment
        explain_df = res_df[keys + ["pred_log_sales"]].merge(
            df_a_web, on=keys, how="inner"
        )

        if len(explain_df) == 0:
            print("Warning: No matching Model A data for WEB explainability.")
            return 0.0

        self.web_surrogate = SurrogateExplainer(channel="WEB")
        self.web_surrogate.fit(
            model_a_df=explain_df,
            model_b_predictions=explain_df["pred_log_sales"].values
        )

        # Save surrogate model
        self.web_surrogate.save_model(
            model_path=checkpoint_dir / "surrogate_web.cbm",
            meta_path=checkpoint_dir / "surrogate_web.meta.json",
        )

        # Generate SHAP analysis
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

        # Merge contributors back to results if available
        if contributor_df is not None:
            res_merged = res_df.merge(contributor_df, on=keys, how="left")
            res_merged.to_csv(output_dir / "predictions_web.csv", index=False)

        return self.web_surrogate.r2_score_value


def train(
    model_b_path: str,
    model_a_path: Optional[str] = None,
    output_dir: str = "output",
    channels: Optional[List[str]] = None,
    correct_bm: bool = False,
    correct_web: bool = False,
    correct_web_sales: bool = False,
    correct_web_aov: bool = False,
    train_surrogate: bool = True,
) -> TrainingResult:
    """
    Convenience function to run the training pipeline.

    Parameters
    ----------
    model_b_path : str
        Path to Model B CSV file.
    model_a_path : str, optional
        Path to Model A CSV file for explainability.
    output_dir : str, optional
        Output directory. Default "output".
    channels : List[str], optional
        Channels to train. Default ["B&M", "WEB"].
    correct_bm : bool, optional
        Apply bias correction to B&M predictions.
    correct_web : bool, optional
        Apply bias correction to all WEB predictions.
    correct_web_sales : bool, optional
        Apply bias correction to WEB Sales only.
    correct_web_aov : bool, optional
        Apply bias correction to WEB AOV only.
    train_surrogate : bool, optional
        Whether to train surrogate models for explainability. Default True.

    Returns
    -------
    TrainingResult
        Training result with metrics and paths.
    """
    # Load data
    model_b_data = pd.read_csv(model_b_path)
    model_a_data = pd.read_csv(model_a_path) if model_a_path else None

    # Create config
    config = TrainingConfig(
        output_dir=Path(output_dir),
        channels=channels or ["B&M", "WEB"],
        bias_correction=BiasCorrection(
            correct_bm=correct_bm,
            correct_web=correct_web,
            correct_web_sales=correct_web_sales,
            correct_web_aov=correct_web_aov,
        ),
        train_surrogate=train_surrogate,
    )

    # Run pipeline
    pipeline = TrainingPipeline(config)
    return pipeline.run(model_b_data, model_a_data)
