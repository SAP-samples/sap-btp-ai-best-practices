"""
Session management for the forecasting agent.

Provides a SessionManager that maintains state across tool calls.
This enables stateful multi-turn conversations where scenarios and modifications
persist within a session.

The SessionManager also handles loading of inference models and data loaders,
caching them for reuse across tool invocations.

NOTE: This is no longer a singleton. Each user session gets its own instance
managed by the SessionStore in session_store.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.agent.state import (
    AgentState,
    PredictionResult,
    ScenarioData,
    create_initial_state,
    get_default_checkpoint_dir,
)


class SessionManager:
    """
    Manager for agent session state.

    Provides centralized access to the AgentState across all tools,
    ensuring consistent state management during multi-turn conversations.

    Also caches expensive resources like loaded models and data loaders.

    NOTE: This is no longer a singleton. Each user gets their own instance.

    Usage
    -----
    # Create a session manager (typically done by SessionStore)
    manager = SessionManager(session_id="user123")

    # Access state
    state = manager.get_state()

    # Update state fields
    manager.set_origin_date("2024-01-15")
    manager.add_scenario(scenario_data)

    # Reset for new session
    manager.reset()
    """

    def __init__(self, session_id: str = "default", checkpoint_dir: Optional[str] = None):
        """
        Initialize a new SessionManager.

        Parameters
        ----------
        session_id : str
            Unique identifier for this session
        checkpoint_dir : Optional[str]
            Path to model checkpoints. If None, uses default.
        """
        self.session_id = session_id
        self._checkpoint_dir = checkpoint_dir or str(get_default_checkpoint_dir())
        self._state: Optional[AgentState] = None

        # Cached resources (per-session)
        self._inference_pipeline = None
        self._store_master: Optional[pd.DataFrame] = None
        self._models_loaded: bool = False

        # Tool result storage for report generation
        self._last_forecast_results: Optional[str] = None
        self._last_scenario_comparison: Optional[str] = None
        self._last_explanation: Optional[str] = None
        self._last_sensitivity: Optional[str] = None
        self._last_yoy_actuals: Optional[str] = None  # Previous year actual sales

        # Dual-level file tracking:
        # - Per-request: cleared each request, used for chat attachments
        # - Session-level: persists for session, used for report generation
        self._current_request_plot_files: List[str] = []
        self._current_request_export_files: List[str] = []
        self._session_plot_files: List[str] = []
        self._session_export_files: List[str] = []

    def reset(self, checkpoint_dir: Optional[str] = None) -> None:
        """
        Reset the session to initial state.

        Clears all scenarios, predictions, and cached resources.

        Parameters
        ----------
        checkpoint_dir : Optional[str]
            New checkpoint directory. If None, uses current.
        """
        if checkpoint_dir:
            self._checkpoint_dir = checkpoint_dir
        self._state = create_initial_state(checkpoint_dir=self._checkpoint_dir)
        self._inference_pipeline = None
        self._store_master = None
        self._models_loaded = False
        # Clear stored tool results
        self._last_forecast_results = None
        self._last_scenario_comparison = None
        self._last_explanation = None
        self._last_sensitivity = None
        self._last_yoy_actuals = None
        # Clear all file tracking (both per-request and session-level)
        self._current_request_plot_files = []
        self._current_request_export_files = []
        self._session_plot_files = []
        self._session_export_files = []

    def get_state(self) -> AgentState:
        """
        Get the current session state.

        Creates initial state if not yet initialized.

        Returns
        -------
        AgentState
            Current session state
        """
        if self._state is None:
            self._state = create_initial_state(checkpoint_dir=self._checkpoint_dir)
        return self._state

    def set_state(self, state: AgentState) -> None:
        """
        Replace the entire state (use with caution).

        Parameters
        ----------
        state : AgentState
            New state to set
        """
        self._state = state

    # -------------------------------------------------------------------------
    # State field accessors
    # -------------------------------------------------------------------------

    def get_origin_date(self) -> Optional[str]:
        """Get the current origin date (t0)."""
        return self.get_state().get("origin_date")

    def set_origin_date(self, origin_date: str) -> None:
        """Set the origin date (t0)."""
        self.get_state()["origin_date"] = origin_date

    def get_horizon_weeks(self) -> int:
        """Get the forecast horizon in weeks."""
        return self.get_state().get("horizon_weeks", 13)

    def set_horizon_weeks(self, horizon_weeks: int) -> None:
        """Set the forecast horizon in weeks."""
        if not 1 <= horizon_weeks <= 52:
            raise ValueError(f"horizon_weeks must be 1-52, got {horizon_weeks}")
        self.get_state()["horizon_weeks"] = horizon_weeks

    def get_channel(self) -> str:
        """Get the active channel."""
        return self.get_state().get("channel", "B&M")

    def set_channel(self, channel: str) -> None:
        """Set the active channel."""
        if channel not in ("B&M", "WEB"):
            raise ValueError(f"channel must be 'B&M' or 'WEB', got {channel}")
        self.get_state()["channel"] = channel
        # Invalidate cached inference pipeline when channel changes
        # since B&M and WEB use different models
        self._inference_pipeline = None

    def get_dma_filter(self) -> list:
        """Get the DMA filter list."""
        return self.get_state().get("dma_filter", [])

    def set_dma_filter(self, dmas: list) -> None:
        """Set the DMA filter list."""
        self.get_state()["dma_filter"] = dmas

    def get_store_filter(self) -> list:
        """Get the store filter list."""
        return self.get_state().get("store_filter", [])

    def set_store_filter(self, stores: list) -> None:
        """Set the store filter list."""
        self.get_state()["store_filter"] = stores

    def get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory path."""
        return Path(self.get_state().get("checkpoint_dir", self._checkpoint_dir))

    def set_checkpoint_dir(self, path: str) -> None:
        """Set the checkpoint directory path."""
        self._checkpoint_dir = path
        self.get_state()["checkpoint_dir"] = path
        # Clear cached models when checkpoint changes
        self._inference_pipeline = None
        self._models_loaded = False

    # -------------------------------------------------------------------------
    # Scenario management
    # -------------------------------------------------------------------------

    def get_active_scenario_name(self) -> str:
        """Get the name of the active scenario."""
        return self.get_state().get("active_scenario", "baseline")

    def set_active_scenario(self, scenario_name: str) -> None:
        """
        Set the active scenario.

        Parameters
        ----------
        scenario_name : str
            Name of scenario to make active

        Raises
        ------
        KeyError
            If scenario doesn't exist
        """
        if scenario_name not in self.get_state()["scenarios"]:
            raise KeyError(
                f"Scenario '{scenario_name}' not found. "
                f"Available: {list(self.get_state()['scenarios'].keys())}"
            )
        self.get_state()["active_scenario"] = scenario_name

    def get_scenario(self, scenario_name: Optional[str] = None) -> Optional[ScenarioData]:
        """
        Get a scenario by name.

        Parameters
        ----------
        scenario_name : Optional[str]
            Name of scenario. If None, returns active scenario.

        Returns
        -------
        Optional[ScenarioData]
            The scenario, or None if not found
        """
        name = scenario_name or self.get_active_scenario_name()
        return self.get_state()["scenarios"].get(name)

    def get_active_scenario(self) -> Optional[ScenarioData]:
        """Get the active scenario."""
        return self.get_scenario(self.get_active_scenario_name())

    def add_scenario(self, scenario: ScenarioData) -> None:
        """
        Add a scenario to the session.

        Parameters
        ----------
        scenario : ScenarioData
            Scenario to add
        """
        self.get_state()["scenarios"][scenario.name] = scenario

    def list_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        List all scenarios with summaries.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping of scenario name to summary dict
        """
        return {
            name: scenario.get_summary()
            for name, scenario in self.get_state()["scenarios"].items()
        }

    def delete_scenario(self, scenario_name: str) -> bool:
        """
        Delete a scenario.

        Parameters
        ----------
        scenario_name : str
            Name of scenario to delete

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        if scenario_name == "baseline":
            raise ValueError("Cannot delete baseline scenario")
        if scenario_name in self.get_state()["scenarios"]:
            del self.get_state()["scenarios"][scenario_name]
            # Also delete any cached predictions
            if scenario_name in self.get_state()["predictions"]:
                del self.get_state()["predictions"][scenario_name]
            return True
        return False

    # -------------------------------------------------------------------------
    # Prediction management
    # -------------------------------------------------------------------------

    def get_prediction(self, scenario_name: str) -> Optional[PredictionResult]:
        """
        Get cached prediction for a scenario.

        Parameters
        ----------
        scenario_name : str
            Name of scenario

        Returns
        -------
        Optional[PredictionResult]
            Cached prediction, or None if not found
        """
        return self.get_state()["predictions"].get(scenario_name)

    def add_prediction(self, prediction: PredictionResult) -> None:
        """
        Cache a prediction result.

        Parameters
        ----------
        prediction : PredictionResult
            Prediction to cache
        """
        self.get_state()["predictions"][prediction.scenario_name] = prediction

    def has_prediction(self, scenario_name: str) -> bool:
        """Check if predictions exist for a scenario."""
        return scenario_name in self.get_state()["predictions"]

    def invalidate_prediction(self, scenario_name: str) -> None:
        """
        Invalidate cached predictions for a scenario.

        Call this when a scenario is modified to force re-prediction.

        Parameters
        ----------
        scenario_name : str
            Name of scenario to invalidate
        """
        if scenario_name in self.get_state()["predictions"]:
            del self.get_state()["predictions"][scenario_name]

    # -------------------------------------------------------------------------
    # Resource caching (models, data)
    # -------------------------------------------------------------------------

    def get_inference_pipeline(self):
        """
        Get or create the InferencePipeline.

        Lazily loads models from checkpoints on first access.

        Returns
        -------
        InferencePipeline
            Configured inference pipeline
        """
        if self._inference_pipeline is None:
            from app.regressor.pipelines import InferencePipeline
            from app.regressor.configs import InferenceConfig

            checkpoint_dir = self.get_checkpoint_dir()
            if not checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {checkpoint_dir}. "
                    "Please ensure model checkpoints are available."
                )

            config = InferenceConfig(
                checkpoint_dir=checkpoint_dir,
                output_dir=checkpoint_dir.parent / "infer",
                channels=[self.get_channel()],
                run_explainability=False,  # We handle explainability separately
            )
            self._inference_pipeline = InferencePipeline(config)
            self._models_loaded = True

        return self._inference_pipeline

    def get_store_master(self) -> pd.DataFrame:
        """
        Get the store master DataFrame.

        Lazily loads on first access.

        Returns
        -------
        pd.DataFrame
            Store master with profit_center_nbr, dma, lat/lon, etc.
        """
        if self._store_master is None:
            from app.agent.hana_loader import load_store_master

            self._store_master = load_store_master()

        return self._store_master

    def are_models_loaded(self) -> bool:
        """Check if models have been loaded."""
        return self._models_loaded

    # -------------------------------------------------------------------------
    # Message storage (for report generation)
    # -------------------------------------------------------------------------

    def get_messages(self) -> list:
        """
        Get stored conversation messages.

        Returns
        -------
        list
            List of LangChain message objects from the conversation
        """
        return self.get_state().get("messages", [])

    def set_messages(self, messages: list) -> None:
        """
        Store conversation messages from the agent.

        Called by the agent runner after each invocation to enable
        report generation from conversation history.

        Parameters
        ----------
        messages : list
            List of LangChain message objects
        """
        self.get_state()["messages"] = messages

    def append_messages(self, messages: list) -> None:
        """
        Append new messages to the existing conversation.

        Parameters
        ----------
        messages : list
            New messages to append
        """
        current = self.get_state().get("messages", [])
        current.extend(messages)
        self.get_state()["messages"] = current

    # -------------------------------------------------------------------------
    # Session summary
    # -------------------------------------------------------------------------

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session state.

        Returns
        -------
        Dict[str, Any]
            Summary including origin date, scenarios, predictions, etc.
        """
        state = self.get_state()
        return {
            "session_id": self.session_id,
            "origin_date": state.get("origin_date"),
            "horizon_weeks": state.get("horizon_weeks", 13),
            "channel": state.get("channel", "B&M"),
            "dma_filter": state.get("dma_filter", []),
            "store_filter": state.get("store_filter", []),
            "active_scenario": state.get("active_scenario", "baseline"),
            "num_scenarios": len(state.get("scenarios", {})),
            "scenario_names": list(state.get("scenarios", {}).keys()),
            "num_predictions": len(state.get("predictions", {})),
            "predictions_for": list(state.get("predictions", {}).keys()),
            "models_loaded": self._models_loaded,
            "checkpoint_dir": str(self.get_checkpoint_dir()),
        }

    # -------------------------------------------------------------------------
    # Tool result storage (for report generation)
    # -------------------------------------------------------------------------

    def store_forecast_results(self, results: str) -> None:
        """
        Store forecast results for report generation.

        Parameters
        ----------
        results : str
            JSON string of forecast results
        """
        self._last_forecast_results = results

    def store_scenario_comparison(self, results: str) -> None:
        """
        Store scenario comparison results for report generation.

        Parameters
        ----------
        results : str
            JSON string of comparison results
        """
        self._last_scenario_comparison = results

    def store_explanation(self, results: str) -> None:
        """
        Store explanation results for report generation.

        Parameters
        ----------
        results : str
            JSON string of explanation results
        """
        self._last_explanation = results

    def store_sensitivity(self, results: str) -> None:
        """
        Store sensitivity analysis results for report generation.

        Parameters
        ----------
        results : str
            JSON string of sensitivity results
        """
        self._last_sensitivity = results

    def store_yoy_actuals(self, results: str) -> None:
        """
        Store previous year actuals for report generation.

        Parameters
        ----------
        results : str
            JSON string of previous year actual sales data
        """
        self._last_yoy_actuals = results

    def add_plot_file(self, path: str) -> None:
        """
        Add a plot file path for both current request and session tracking.

        Current request files are used for chat attachments.
        Session files are used for report generation.

        Parameters
        ----------
        path : str
            Absolute path to the plot file
        """
        if path:
            # Add to current request tracking (for chat attachments)
            if path not in self._current_request_plot_files:
                self._current_request_plot_files.append(path)
            # Add to session tracking (for report generation)
            if path not in self._session_plot_files:
                self._session_plot_files.append(path)

    def add_export_file(self, path: str) -> None:
        """
        Add an export file path (CSV, PDF) for both current request and session tracking.

        Parameters
        ----------
        path : str
            Absolute path to the export file
        """
        if path:
            # Add to current request tracking (for chat attachments)
            if path not in self._current_request_export_files:
                self._current_request_export_files.append(path)
            # Add to session tracking (for report generation)
            if path not in self._session_export_files:
                self._session_export_files.append(path)

    def get_generated_files(self) -> List[str]:
        """
        Get files generated in the current request (for chat attachments).

        Returns
        -------
        List[str]
            List of absolute file paths for current request files
        """
        return list(self._current_request_plot_files) + list(self._current_request_export_files)

    def clear_generated_files(self) -> None:
        """
        Clear only per-request generated files.

        Call this before each query to track only files from the current request.
        Session-level files are preserved for report generation.
        """
        self._current_request_plot_files = []
        self._current_request_export_files = []

    def get_stored_results(self) -> Dict[str, Any]:
        """
        Get all stored tool results for report generation.

        Returns session-level plot/export files (not per-request files).

        Returns
        -------
        Dict[str, Any]
            Dictionary with forecast_results, scenario_comparison,
            explanation, sensitivity, yoy_actuals, plot_files, and export_files
        """
        return {
            "forecast_results": self._last_forecast_results,
            "scenario_comparison": self._last_scenario_comparison,
            "explanation": self._last_explanation,
            "sensitivity": self._last_sensitivity,
            "yoy_actuals": self._last_yoy_actuals,
            "plot_files": list(self._session_plot_files),
            "export_files": list(self._session_export_files),
        }

    def clear_stored_results(self) -> None:
        """Clear all stored tool results and session-level files."""
        self._last_forecast_results = None
        self._last_scenario_comparison = None
        self._last_explanation = None
        self._last_sensitivity = None
        self._last_yoy_actuals = None
        self._current_request_plot_files = []
        self._current_request_export_files = []
        self._session_plot_files = []
        self._session_export_files = []


# The get_session() function is now provided by session_store.py
# which uses a context variable to get the current user's session.
# This import is for backwards compatibility with existing tools.
def get_session() -> SessionManager:
    """
    Get the current session from context.

    This function is provided for backwards compatibility.
    It retrieves the session from the context variable set by session_store.

    Returns
    -------
    SessionManager
        The current user's session manager

    Raises
    ------
    RuntimeError
        If called outside of a session context
    """
    from app.agent.session_store import get_current_session
    return get_current_session()


__all__ = [
    "SessionManager",
    "get_session",
]
