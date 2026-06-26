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


class PharmaOrderRouterTests(unittest.TestCase):
    def test_pharma_order_router_response_shape(self):
        original = pharma_order.run_pharma_order_agent

        async def fake_agent(**kwargs):
            return {
                "answer": "Northstar has a synthetic demo price for Glycemor 10 mg.",
                "markdown": "Northstar has a synthetic demo price for Glycemor 10 mg.",
                "model": "fake-model",
                "provider": "fake-provider",
                "tool_call_count": 1,
                "usage": {"input_tokens": 10, "output_tokens": 12},
                "tool_calls": [{"name": "get_pricing_for_customer_material"}],
                "tool_results": [],
            }

        try:
            pharma_order.run_pharma_order_agent = fake_agent
            payload = pharma_order.PharmaOrderRequest(
                question="What is the price for Northstar for Glycemor 10 mg?",
                include_trace=True,
            )
            response = asyncio.run(pharma_order.ask_pharma_order(payload, api_key="test-key"))
        finally:
            pharma_order.run_pharma_order_agent = original

        self.assertTrue(response.success)
        self.assertEqual(response.tool_call_count, 1)
        self.assertIn("Northstar", response.answer)


if __name__ == "__main__":
    unittest.main()
