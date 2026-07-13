from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.routers import pharma_order


class PharmaOrderCapabilitiesTests(unittest.TestCase):
    def test_capabilities_endpoint_describes_test_ui_and_complex_chain(self):
        payload = asyncio.run(pharma_order.pharma_order_capabilities())
        self.assertEqual(payload["agent"], "pharma-order")
        self.assertEqual(payload["mode"], "test-ui")
        capability_ids = {item["id"] for item in payload["capabilities"]}
        self.assertIn("pricing", capability_ids)
        self.assertIn("complex_resolution_chain", capability_ids)
        complex_capability = next(item for item in payload["capabilities"] if item["id"] == "complex_resolution_chain")
        self.assertIn("lookup_material_by_ndc", complex_capability["expected_tools"])
        self.assertIn("get_pricing_for_customer_material", complex_capability["expected_tools"])
        self.assertGreaterEqual(len(complex_capability["source_structures"]), 5)


if __name__ == "__main__":
    unittest.main()
