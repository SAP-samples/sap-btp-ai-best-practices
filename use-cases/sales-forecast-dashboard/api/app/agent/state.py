"""
Agent state management for the forecasting agent.

Defines the state structure following Agent_plan.md Section 2.1:
- origin_date: The "Today" of the simulation (t0)
- horizon_weeks: How far out we are predicting (h)
- dma_filter / store_filter: Scope of analysis
- scenarios: Dict of scenario_name -> ScenarioData (DataFrame-based)
- active_scenario: Currently edited/viewed scenario

Scenarios store DataFrames with one row per (store, horizon) combination,
containing all Model A and Model B features for direct use with InferencePipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import pandas as pd
from langgraph.graph.message import add_messages

from app.agent.paths import CHECKPOINT_DIR as AGENT_CHECKPOINT_DIR, get_checkpoint_dir


@dataclass
class ScenarioData:
    """
    A single scenario's feature data as a DataFrame.

    The DataFrame has one row per (store, horizon) combination.
    Columns include all Model A and Model B features plus metadata:
    - Keys: profit_center_nbr, dma, channel, origin_week_date, target_week_date, horizon
    - Features: All MODEL_A_FEATURES + MODEL_B_ADDITIONAL_FEATURES
    - Labels (for historical data): label_log_sales, label_log_aov, etc.

    Attributes
    ----------
    name : str
        Unique scenario name (e.g., "baseline", "aggressive_marketing")
    df : pd.DataFrame
        Feature DataFrame with all store/horizon combinations
    parent_scenario : Optional[str]
        Name of the scenario this was forked from (for audit trail)
    created_at : str
        ISO timestamp when scenario was created
    modifications : List[Dict[str, Any]]
        Audit trail of all modifications applied to this scenario
    """

    name: str
    df: pd.DataFrame
    parent_scenario: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    channel: Optional[str] = None  # Track which channel created this scenario (B&M or WEB)

    def copy(self, new_name: str) -> "ScenarioData":
        """
        Create a deep copy of this scenario with a new name.

        Parameters
        ----------
        new_name : str
            Name for the new scenario

        Returns
        -------
        ScenarioData
            New scenario with copied DataFrame and this scenario as parent
        """
        return ScenarioData(
            name=new_name,
            df=self.df.copy(),
            parent_scenario=self.name,
            created_at=datetime.now().isoformat(),
            modifications=[],
            channel=self.channel,  # Preserve channel from parent scenario
        )

    def add_modification(
        self,
        feature_name: str,
        modification_type: str,
        old_value: Any,
        new_value: Any,
        scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a modification in the audit trail.

        Parameters
        ----------
        feature_name : str
            Name of the modified feature
        modification_type : str
            Type of modification (set, increase, decrease, multiply)
        old_value : Any
            Previous value(s) - can be scalar or summary
        new_value : Any
            New value(s) - can be scalar or summary
        scope : Optional[Dict[str, Any]]
            Scope filters applied (stores, DMAs, horizons)
        """
        self.modifications.append(
            {
                "timestamp": datetime.now().isoformat(),
                "feature": feature_name,
                "type": modification_type,
                "old_value": old_value,
                "new_value": new_value,
                "scope": scope or {},
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this scenario.

        Returns
        -------
        Dict[str, Any]
            Summary including name, parent, row count, modification count
        """
        return {
            "name": self.name,
            "parent": self.parent_scenario,
            "channel": self.channel,
            "created_at": self.created_at,
            "num_rows": len(self.df) if self.df is not None else 0,
            "num_modifications": len(self.modifications),
            "stores": (
                self.df["profit_center_nbr"].unique().tolist()
                if self.df is not None and "profit_center_nbr" in self.df.columns
                else []
            ),
            "horizons": (
                sorted(self.df["horizon"].unique().tolist())
                if self.df is not None and "horizon" in self.df.columns
                else []
            ),
        }


@dataclass
class PredictionResult:
    """
    Cached prediction results for a scenario.

    Contains the full predictions DataFrame with quantiles and optional SHAP values.

    Attributes
    ----------
    scenario_name : str
        Name of the scenario these predictions are for
    predictions_df : pd.DataFrame
        DataFrame with predictions including:
        - Keys: profit_center_nbr, dma, channel, horizon
        - Predictions: pred_log_sales, pred_log_aov, pred_log_orders
        - Quantiles: pred_sales_p50, pred_sales_p90, pred_aov_p50, etc.
        - Traffic (B&M): pred_traffic_p10, pred_traffic_p50, pred_traffic_p90
    shap_df : Optional[pd.DataFrame]
        SHAP values per row (if explainability was run)
    generated_at : str
        ISO timestamp when predictions were generated
    metadata : Dict[str, Any]
        Additional metadata (rmse, r2, model_version)
    """

    scenario_name: str
    predictions_df: pd.DataFrame
    shap_df: Optional[pd.DataFrame] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics across all stores/horizons.

        Returns
        -------
        Dict[str, float]
            Total and mean metrics for sales, AOV, etc.
        """
        df = self.predictions_df
        result = {}

        if "pred_sales_p50" in df.columns:
            result["total_sales_p50"] = df["pred_sales_p50"].sum()
            result["mean_sales_p50"] = df["pred_sales_p50"].mean()

        if "pred_aov_p50" in df.columns:
            result["mean_aov_p50"] = df["pred_aov_p50"].mean()

        if "pred_traffic_p50" in df.columns:
            result["total_traffic_p50"] = df["pred_traffic_p50"].sum()

        return result


class AgentState(TypedDict):
    """
    Main state structure for the forecasting agent.

    Aligns with Agent_plan.md Section 2.1:
    - origin_date: The "Today" of the simulation (t0)
    - horizon_weeks: How far out we are predicting (h)
    - dma_filter / store_filter: Scope of analysis
    - scenarios: Dict of scenario_name -> ScenarioData
    - active_scenario: Currently edited/viewed scenario

    Attributes
    ----------
    messages : list
        Conversation history with add_messages annotation for LangGraph
    origin_date : Optional[str]
        The simulation "today" date in YYYY-MM-DD format (t0)
    horizon_weeks : int
        Number of weeks to forecast (default 13, max 52)
    dma_filter : List[str]
        List of DMAs to include in analysis (empty = all)
    store_filter : List[int]
        List of profit_center_nbr to include (empty = all)
    channel : str
        Active channel: "B&M" or "WEB"
    scenarios : Dict[str, ScenarioData]
        All scenarios in this session, keyed by scenario name
    active_scenario : str
        Name of the currently active scenario
    predictions : Dict[str, PredictionResult]
        Cached predictions, keyed by scenario name
    checkpoint_dir : str
        Path to model checkpoints directory
    """

    messages: Annotated[list, add_messages]
    origin_date: Optional[str]
    horizon_weeks: int
    dma_filter: List[str]
    store_filter: List[int]
    channel: str
    scenarios: Dict[str, ScenarioData]
    active_scenario: str
    predictions: Dict[str, PredictionResult]
    checkpoint_dir: str


def create_initial_state(
    checkpoint_dir: Optional[str] = None,
) -> AgentState:
    """
    Create an empty initial state for a new agent session.

    Parameters
    ----------
    checkpoint_dir : Optional[str]
        Path to model checkpoints. Default: uses agent input checkpoints directory.

    Returns
    -------
    AgentState
        Empty state with all fields initialized to defaults
    """
    if checkpoint_dir is None:
        checkpoint_dir = str(get_checkpoint_dir())
    return {
        "messages": [],
        "origin_date": None,
        "horizon_weeks": 13,
        "dma_filter": [],
        "store_filter": [],
        "channel": "B&M",
        "scenarios": {},
        "active_scenario": "baseline",
        "predictions": {},
        "checkpoint_dir": checkpoint_dir,
    }


def get_default_checkpoint_dir() -> Path:
    """
    Get the default checkpoint directory path.

    Prefers the agent input checkpoints directory, with fallback to legacy locations.

    Returns
    -------
    Path
        Absolute path to checkpoint directory
    """
    return get_checkpoint_dir()


__all__ = [
    "ScenarioData",
    "PredictionResult",
    "AgentState",
    "create_initial_state",
    "get_default_checkpoint_dir",
]
