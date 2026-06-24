import json
from pathlib import Path
from typing import Any, Dict


_SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "s4_seed.json"


def load_seed() -> Dict[str, Any]:
    """Load mock S/4 seed data from api/app/mock/data/s4_seed.json."""
    try:
        return json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"businessPartners": [], "purchaseOrders": [], "invoices": []}
