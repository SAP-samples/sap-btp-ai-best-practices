"""
Centralized path configuration for the forecasting agent.

All agent tools and modules should import paths from this module
rather than defining their own path constants. This ensures consistency
and makes future path changes easier to manage.

The input/ directory contains all data files and model checkpoints
required for the agent to function:
- model_b.csv: Historical features for forecasting
- BDF Data Model Master Tables.xlsx: Store master, DMA mappings
- budget_marketing.xlsx: Marketing budget by DMA
- Awareness_Consideration_2022-2025.xlsx: Brand awareness metrics
- checkpoints/: Model checkpoint files (.cbm)

Environment Variables:
- AGENT_INPUT_DIR: Override the base input directory
- CHECKPOINT_DIR: Override the checkpoints directory
"""
import os
from pathlib import Path

# Base directory for agent inputs - can be overridden via environment
_default_input_dir = Path(__file__).parent / "input"
AGENT_INPUT_DIR = Path(os.getenv("AGENT_INPUT_DIR", str(_default_input_dir)))

# Data files
MODEL_B_PATH = AGENT_INPUT_DIR / "model_b.csv"
MASTER_TABLES_PATH = AGENT_INPUT_DIR / "BDF Data Model Master Tables.xlsx"
BUDGET_PATH = AGENT_INPUT_DIR / "budget_marketing.xlsx"
AWARENESS_PATH = AGENT_INPUT_DIR / "Awareness_Consideration_2022-2025.xlsx"
PREDICTIONS_BM_PATH = AGENT_INPUT_DIR / "predictions_bm.csv"

# Model checkpoints - can be overridden via environment
_default_checkpoint_dir = AGENT_INPUT_DIR / "checkpoints"
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(_default_checkpoint_dir)))

# Output directory for agent-generated files (plots, reports, etc.)
_default_output_dir = Path(__file__).parent / "output"
AGENT_OUTPUT_DIR = Path(os.getenv("AGENT_OUTPUT_DIR", str(_default_output_dir)))


def get_model_b_path() -> Path:
    """Get the path to model_b.csv."""
    return MODEL_B_PATH


def get_checkpoint_dir() -> Path:
    """Get the path to checkpoints directory."""
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_DIR}. "
            "Ensure model checkpoints are deployed or set CHECKPOINT_DIR environment variable."
        )
    return CHECKPOINT_DIR


def get_master_tables_path() -> Path:
    """Get the path to BDF Data Model Master Tables.xlsx."""
    return MASTER_TABLES_PATH


def get_budget_path() -> Path:
    """Get the path to budget_marketing.xlsx."""
    return BUDGET_PATH


def get_awareness_path() -> Path:
    """Get the path to Awareness_Consideration_2022-2025.xlsx."""
    return AWARENESS_PATH


def get_predictions_bm_path() -> Path:
    """Get the path to predictions_bm.csv."""
    return PREDICTIONS_BM_PATH


def get_output_dir() -> Path:
    """Get the output directory for agent-generated files."""
    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return AGENT_OUTPUT_DIR


__all__ = [
    "AGENT_INPUT_DIR",
    "MODEL_B_PATH",
    "MASTER_TABLES_PATH",
    "BUDGET_PATH",
    "AWARENESS_PATH",
    "PREDICTIONS_BM_PATH",
    "CHECKPOINT_DIR",
    "AGENT_OUTPUT_DIR",
    "get_model_b_path",
    "get_checkpoint_dir",
    "get_master_tables_path",
    "get_budget_path",
    "get_awareness_path",
    "get_predictions_bm_path",
    "get_output_dir",
]
