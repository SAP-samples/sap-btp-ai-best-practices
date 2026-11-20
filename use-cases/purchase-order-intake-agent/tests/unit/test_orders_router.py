"""HTTP contracts of the review endpoints (service layer patched, no HANA)."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.models.orders import OrderUpdate, ReprocessRequest
from app.orders.posting import PostingError
from app.orders.service import ReviewError
from app.routers import orders as router


def fake_request():
    """Request stub with an app.state event to observe pipeline wake-ups."""
    event = asyncio.Event()
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(order_queue_event=event))), event


class OrdersRouterTests(unittest.TestCase):
    """Status codes and payloads the UI relies on."""

    def test_confirm_with_missing_data_returns_422_with_issues(self):
        issues = [{"severity": "ERROR", "item_no": 10, "field": "MATERIAL", "code": "MISSING", "message": "Material missing"}]
        with patch.object(router, "confirm_order", side_effect=ReviewError("Required data is missing or invalid", 422, issues)):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(router.confirm("order-1", None))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail["issues"], issues)

    def test_second_confirm_returns_409(self):
        with patch.object(router, "confirm_order", side_effect=PostingError("Order is already posted as 9100000001", 409)):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(router.confirm("order-1", None))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_confirm_saves_edits_then_posts(self):
        calls = []
        with patch.object(router, "save_user_edits", lambda *a: calls.append(("save", a))), \
             patch.object(router, "confirm_order", lambda order_id, post: calls.append(("confirm", order_id)) or "9100000007"):
            response = asyncio.run(router.confirm("order-1", OrderUpdate(header={"ship_to": "762431"})))
        self.assertEqual(response.sales_doc, "9100000007")
        self.assertEqual([c[0] for c in calls], ["save", "confirm"])
        self.assertEqual(calls[0][1], ("order-1", {"ship_to": "762431"}, None, None))

    def test_item_update_keeps_unset_fields_out(self):
        captured = {}
        with patch.object(router, "save_user_edits", lambda order_id, header, items, cls: captured.update(items=items)), \
             patch.object(router, "build_detail", lambda order_id: "detail"):
            asyncio.run(router.save_order("order-1", OrderUpdate(items=[{"item_no": 10, "quantity": 5}])))
        self.assertEqual(captured["items"], [{"item_no": 10, "quantity": 5.0}])

    def test_delete_returns_deleted(self):
        with patch.object(router, "delete_order", lambda order_id: None):
            response = asyncio.run(router.delete("order-1"))
        self.assertTrue(response.deleted)

    def test_delete_while_processing_returns_409(self):
        with patch.object(router, "delete_order", side_effect=ReviewError("processing", 409)):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(router.delete("order-1"))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_reprocess_queues_and_wakes_workers(self):
        request, event = fake_request()
        with patch.object(router, "request_reprocess", lambda order_id, note: None):
            response = asyncio.run(router.reprocess("order-1", request, ReprocessRequest(note="use ship-to 762431")))
        self.assertEqual(response.status, "QUEUED")
        self.assertTrue(event.is_set())


if __name__ == "__main__":
    unittest.main()
