"""Pipeline decisions after an agent run (agent, gate and database patched)."""

import asyncio
import unittest
from unittest.mock import patch

from app.order_agent.runner import RunOutcome
from app.orders import pipeline
from app.orders.posting import PostingError


class PipelineTests(unittest.TestCase):
    """Retry, failure and gate hand-off of OrderPipeline.process."""

    def run_process(self, outcome, attempts=1, gate_status="NEEDS_REVIEW"):
        """Run process() with a fixed agent outcome; return (status, set_status calls, gate calls)."""
        statuses, gate_calls = [], []

        async def fake_run(order_id, note=None):
            return outcome

        def fake_gate(order_id, summary, post):
            gate_calls.append((order_id, summary, post))
            return gate_status

        with patch.object(pipeline, "run_order_agent", fake_run), \
             patch.object(pipeline, "finalize_after_agent", fake_gate), \
             patch.object(pipeline.repository, "get_header", lambda order_id: {"ATTEMPTS": attempts}), \
             patch.object(pipeline.repository, "set_status", lambda order_id, status, msg=None, **kw: statuses.append(status)):
            result = asyncio.run(pipeline.OrderPipeline().process("order-1"))
        return result, statuses, gate_calls

    def test_completed_run_goes_through_the_gate_with_posting_callback(self):
        status, statuses, gate_calls = self.run_process(RunOutcome("COMPLETED", summary="done"), gate_status="POSTED")
        self.assertEqual(status, "POSTED")
        self.assertEqual(gate_calls, [("order-1", "done", pipeline.post_for_gate)])
        self.assertEqual(statuses, [])

    def test_error_is_retried_then_failed(self):
        settings_attempts = pipeline.get_settings().max_attempts
        status, statuses, _ = self.run_process(RunOutcome("ERROR", error="boom"), attempts=settings_attempts - 1)
        self.assertEqual((status, statuses), ("QUEUED", ["QUEUED"]))
        status, statuses, _ = self.run_process(RunOutcome("TIMEOUT", error="slow"), attempts=settings_attempts)
        self.assertEqual((status, statuses), ("FAILED", ["FAILED"]))

    def test_ecc_rejection_marks_order_failed(self):
        recorded = []
        with patch.object(pipeline, "post_order", side_effect=PostingError("ECC rejected the order: no pricing")), \
             patch.object(pipeline.repository, "set_status", lambda order_id, status, msg=None, **kw: recorded.append((status, msg))):
            self.assertEqual(pipeline.post_for_gate("order-1"), "")
        self.assertEqual(recorded, [("FAILED", "ECC rejected the order: no pricing")])


if __name__ == "__main__":
    unittest.main()
