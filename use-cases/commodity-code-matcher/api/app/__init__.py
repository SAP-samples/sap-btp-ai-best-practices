"""FastAPI application package for the doc_extraction service."""

from pathlib import Path
import sys

# Ensure the project root (parent of api/) is on sys.path when running from the api directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
