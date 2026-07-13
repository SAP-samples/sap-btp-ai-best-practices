from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_ROOT = ROOT / "api"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("METAL_COMPOSITION_PREWARM_ON_STARTUP", "false")
os.environ.setdefault("METAL_COMPOSITION_DATA_SOURCE", "workbook")
