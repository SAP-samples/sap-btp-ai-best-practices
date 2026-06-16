"""
Monte Carlo Traffic Estimation Module.

Computes Traffic = Sales / (AOV * Conversion) using probabilistic simulation
to provide P10/P50/P90 uncertainty bands.

The approach:
1. Model predictions are in log-space (log_sales, log_aov, logit_conversion)
2. We simulate draws from log-normal distributions (for Sales/AOV) and
   logit-normal distributions (for Conversion)
3. Traffic is computed for each simulation draw
4. Percentiles are computed across all simulations for uncertainty quantification

Memory Optimization Notes:
- Default n_simulations reduced from 10000 to 2000 (96% memory reduction per batch)
- Default batch_size reduced from 1000 to 200
- 2000 simulations provide <1% difference for P50 and 2-5% for P10/P90
- All parameters configurable via environment variables for deployment tuning
"""

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

# Memory-efficient defaults for Cloud Foundry deployment (1.5GB limit)
# Can be overridden via environment variables for accuracy vs memory tradeoff
DEFAULT_N_SIMULATIONS = int(os.getenv("TRAFFIC_N_SIMULATIONS", "2000"))
DEFAULT_BATCH_SIZE = int(os.getenv("TRAFFIC_BATCH_SIZE", "200"))
TRAFFIC_ESTIMATION_ENABLED = os.getenv("TRAFFIC_ESTIMATION_ENABLED", "true").lower() == "true"


@dataclass
class TrafficResult:
    """Container for traffic estimation results with uncertainty bands."""

    p10: np.ndarray  # 10th percentile (lower bound)
    p50: np.ndarray  # 50th percentile (median)
    p90: np.ndarray  # 90th percentile (upper bound)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "pred_traffic_p10": self.p10,
            "pred_traffic_p50": self.p50,
            "pred_traffic_p90": self.p90,
        }


class TrafficEstimator:
    """
    Monte Carlo estimator for traffic predictions.

    Traffic is derived from the relationship:
        Traffic = Sales / (AOV * Conversion)

    Since predictions are in log/logit space with known RMSE,
    we simulate draws from the error distributions to propagate
    uncertainty through the traffic formula.
    """

    def __init__(self, n_simulations: Optional[int] = None, random_seed: Optional[int] = None):
        """
        Initialize the traffic estimator.

        Parameters
        ----------
        n_simulations : int, optional
            Number of Monte Carlo simulations per sample.
            Default is controlled by TRAFFIC_N_SIMULATIONS env var (2000).
            Higher values improve percentile accuracy but increase memory usage.
        random_seed : int, optional
            Random seed for reproducibility.
        """
        self.n_simulations = n_simulations if n_simulations is not None else DEFAULT_N_SIMULATIONS
        self.random_seed = random_seed

    def estimate(
        self,
        log_sales_pred: np.ndarray,
        log_aov_pred: np.ndarray,
        logit_conv_pred: np.ndarray,
        sales_rmse: float,
        aov_rmse: float,
        conv_rmse: float,
        batch_size: Optional[int] = None,
    ) -> TrafficResult:
        """
        Estimate traffic distribution via Monte Carlo simulation.

        Uses batched processing to manage memory for large datasets.

        Parameters
        ----------
        log_sales_pred : np.ndarray
            Predicted mean log(Sales) for each sample.
        log_aov_pred : np.ndarray
            Predicted mean log(AOV) for each sample.
        logit_conv_pred : np.ndarray
            Predicted mean logit(Conversion) for each sample.
        sales_rmse : float
            RMSE (standard deviation) for Sales predictions (from validation).
        aov_rmse : float
            RMSE (standard deviation) for AOV predictions (from validation).
        conv_rmse : float
            RMSE (standard deviation) for Conversion predictions (from validation).
        batch_size : int, optional
            Number of samples to process at once.
            Default is controlled by TRAFFIC_BATCH_SIZE env var (200).

        Returns
        -------
        TrafficResult
            Traffic estimates with P10, P50, P90 uncertainty bands.
        """
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_samples = len(log_sales_pred)
        print(f"Running Monte Carlo Traffic Estimation for {n_samples} samples "
              f"(batch_size={batch_size}, n_simulations={self.n_simulations})...")

        p10_all = np.zeros(n_samples)
        p50_all = np.zeros(n_samples)
        p90_all = np.zeros(n_samples)

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_len = end - i

            # 1. Simulate Draws (Vectorized for batch)
            # Shape: [batch_len, n_simulations]
            # Use float32 to reduce memory by 50% (sufficient precision for percentile estimation)

            # Sales: Log-Normal distribution
            # exp(Normal(mu, sigma)) gives Log-Normal
            sales_draws = np.exp(
                np.random.normal(
                    loc=log_sales_pred[i:end, None],
                    scale=sales_rmse,
                    size=(batch_len, self.n_simulations)
                ).astype(np.float32)
            )

            # AOV: Log-Normal distribution
            aov_draws = np.exp(
                np.random.normal(
                    loc=log_aov_pred[i:end, None],
                    scale=aov_rmse,
                    size=(batch_len, self.n_simulations)
                ).astype(np.float32)
            )

            # Conversion: Sigmoid(Logit-Normal) -> Bounded [0, 1]
            # We add noise in logit space, then apply sigmoid
            logit_draws = np.random.normal(
                loc=logit_conv_pred[i:end, None],
                scale=conv_rmse,
                size=(batch_len, self.n_simulations)
            ).astype(np.float32)
            conv_draws = 1 / (1 + np.exp(-logit_draws))

            # Avoid division by zero (clip conversion > 0)
            conv_draws = np.maximum(conv_draws, np.float32(1e-6))

            # 2. Compute Traffic
            # T = S / (A * C)
            traffic_draws = sales_draws / (aov_draws * conv_draws)

            # 3. Compute Percentiles across simulations
            p10_all[i:end] = np.percentile(traffic_draws, 10, axis=1)
            p50_all[i:end] = np.percentile(traffic_draws, 50, axis=1)
            p90_all[i:end] = np.percentile(traffic_draws, 90, axis=1)

        return TrafficResult(p10=p10_all, p50=p50_all, p90=p90_all)


def estimate_traffic_quantiles(
    log_sales_pred: np.ndarray,
    log_aov_pred: np.ndarray,
    logit_conv_pred: np.ndarray,
    sales_rmse: float,
    aov_rmse: float,
    conv_rmse: float,
    n_simulations: Optional[int] = None,
    batch_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> TrafficResult:
    """
    Convenience function to estimate traffic quantiles.

    Parameters
    ----------
    log_sales_pred : np.ndarray
        Predicted mean log(Sales) for each sample.
    log_aov_pred : np.ndarray
        Predicted mean log(AOV) for each sample.
    logit_conv_pred : np.ndarray
        Predicted mean logit(Conversion) for each sample.
    sales_rmse : float
        RMSE for Sales predictions.
    aov_rmse : float
        RMSE for AOV predictions.
    conv_rmse : float
        RMSE for Conversion predictions.
    n_simulations : int, optional
        Number of Monte Carlo simulations.
        Default is controlled by TRAFFIC_N_SIMULATIONS env var (2000).
    batch_size : int, optional
        Batch size for processing.
        Default is controlled by TRAFFIC_BATCH_SIZE env var (200).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    TrafficResult
        Traffic estimates with P10, P50, P90 uncertainty bands.
    """
    estimator = TrafficEstimator(n_simulations=n_simulations, random_seed=random_seed)
    return estimator.estimate(
        log_sales_pred=log_sales_pred,
        log_aov_pred=log_aov_pred,
        logit_conv_pred=logit_conv_pred,
        sales_rmse=sales_rmse,
        aov_rmse=aov_rmse,
        conv_rmse=conv_rmse,
        batch_size=batch_size,
    )
