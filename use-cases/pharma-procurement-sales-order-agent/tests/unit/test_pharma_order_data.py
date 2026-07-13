from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.tools.pharma_order.data_store import DATASETS, DATA_DIR, search_records


class PharmaOrderDataTests(unittest.TestCase):
    def test_all_pharma_order_json_datasets_are_valid(self):
        for file_name in DATASETS.values():
            path = DATA_DIR / file_name
            with self.subTest(file_name=file_name):
                self.assertTrue(path.exists())
                with path.open("r", encoding="utf-8") as handle:
                    self.assertIsNotNone(json.load(handle))

    def test_anchor_pricing_question_has_matching_data(self):
        result = search_records("pricing", "Northstar", "Glycemor", "10 mg", limit=3)
        self.assertGreater(result["match_count"], 0)


if __name__ == "__main__":
    unittest.main()
