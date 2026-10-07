"""Deduction agent tests. Make ``app`` importable without PYTHONPATH (runs before every module here)."""
import sys
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[3] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))
