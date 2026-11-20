"""Purchase-order endpoints: upload, list, detail and attachment download.

All endpoints require the ``X-API-Key`` header. Database work runs in worker threads
(``asyncio.to_thread``) so concurrent requests and background agent runs never block the
event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile

from ..ecc.sales_documents import list_sales_orders
from ..models.orders import (
    AttachmentInfo,
    ConfirmResponse,
    DeleteResponse,
    FieldState,
    Issue,
    MessageInfo,
    OrderDetail,
    OrderHeader,
    OrderItem,
    OrderListResponse,
    OrderSummary,
    OrderUpdate,
    ReprocessRequest,
    RunSummary,
    SalesOrderListResponse,
    SalesOrderSummary,
    StatusResponse,
    UploadResponse,
)
from ..intake.service import IntakeError, UploadedFile, intake_upload
from ..orders import repository
from ..orders.matching import alpha_out
from ..orders.posting import PostingError, post_order
from ..orders.service import ReviewError, confirm_order, delete_order, request_reprocess, save_user_edits
from ..security import get_api_key

router = APIRouter(dependencies=[Depends(get_api_key)])

# Columns shown without leading zeros in the API.
ALPHA_COLUMNS = {"SOLD_TO", "SHIP_TO", "MATERIAL", "SALES_DOC"}


def to_api(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row to API keys (lower case) and display number formats.

    Args:
        row: Row with upper-case keys.

    Returns:
        Dict with snake_case keys.
    """
    return {key.lower(): alpha_out(value) if key in ALPHA_COLUMNS else value for key, value in row.items()}


def viewer_for(row: dict[str, Any]) -> str:
    """Pick how the UI displays a stored file."""
    mime = (row.get("MIME_TYPE") or "").lower()
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    if row.get("EXTRACTED_TEXT"):
        return "text"
    return "download"


def notify_pipeline(request: Request) -> None:
    """Wake idle order workers after new orders were queued (no-op before the pipeline starts)."""
    event = getattr(request.app.state, "order_queue_event", None)
    if event is not None:
        event.set()


def build_detail(order_id: str) -> OrderDetail | None:
    """Assemble the order detail from header, items, field states, message and files (blocking).

    Args:
        order_id: Order id.

    Returns:
        OrderDetail or None if the order does not exist.
    """
    header = repository.get_header(order_id)
    if header is None:
        return None
    fields: dict[str, dict[str, FieldState]] = {}
    issues: list[Issue] = []
    for state in repository.get_field_states(order_id):
        item_no, name = state["ITEM_NO"], state["FIELD_NAME"]
        fields.setdefault(str(item_no), {})[name] = FieldState(**to_api(state))
        if state.get("ISSUE_SEVERITY"):
            issues.append(Issue(severity=state["ISSUE_SEVERITY"], item_no=item_no, field=name,
                                code=state.get("ISSUE_CODE"), message=state.get("ISSUE_TEXT") or ""))
    order = {"ERROR": 0, "REVIEW": 1, "WARNING": 2, "INFO": 3}
    issues.sort(key=lambda issue: (order[issue.severity], issue.item_no))
    message = repository.get_message(header["MESSAGE_ID"]) or {"SOURCE_TYPE": "UNKNOWN"}
    attachments = [AttachmentInfo(**to_api(row), viewer=viewer_for(row))
                   for row in repository.get_attachments(header["MESSAGE_ID"])]
    run = repository.get_latest_run(order_id)
    return OrderDetail(
        header=OrderHeader(**to_api(header)),
        items=[OrderItem(**to_api(item)) for item in repository.get_items(order_id)],
        fields=fields,
        issues=issues,
        message=MessageInfo(**to_api(message)),
        attachments=attachments,
        run=RunSummary(**to_api(run)) if run else None,
    )


@router.post("/orders/upload", response_model=UploadResponse, status_code=202)
async def upload_orders(
    request: Request,
    files: list[UploadFile] = File(default=[]),
    email_text: Optional[str] = Form(default=None),
    sales_org: Optional[str] = Form(default=None),
) -> UploadResponse:
    """Receive e-mails / documents and queue one order per message.

    Args:
        request: Incoming request (used to wake the pipeline).
        files: .eml, PDF, image, .docx, .xlsx, .txt or .csv files.
        email_text: Optional pasted e-mail content.
        sales_org: Optional sales organisation (simulates the inbox mapping).

    Returns:
        202 with the created orders, all in status QUEUED.
    """
    uploaded = [UploadedFile(f.filename or "upload", f.content_type, await f.read()) for f in files]
    try:
        created = await asyncio.to_thread(intake_upload, uploaded, email_text, sales_org)
    except IntakeError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    notify_pipeline(request)
    return UploadResponse(orders=created)


@router.get("/orders", response_model=OrderListResponse)
async def list_orders(
    status: Optional[str] = Query(default=None, description="Comma-separated statuses"),
    q: Optional[str] = Query(default=None, description="Search in PO number, customer, sender, subject"),
    limit: int = Query(default=200, ge=1, le=1000),
) -> OrderListResponse:
    """List orders, newest first."""
    statuses = [s.strip().upper() for s in status.split(",") if s.strip()] if status else None
    rows = await asyncio.to_thread(repository.list_orders, statuses, q, limit)
    return OrderListResponse(items=[OrderSummary(**to_api(row)) for row in rows])


@router.get("/orders/{order_id}", response_model=OrderDetail)
async def get_order(order_id: str) -> OrderDetail:
    """Return everything the review page needs for one order."""
    detail = await asyncio.to_thread(build_detail, order_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return detail


@router.get("/orders/{order_id}/attachments/{attachment_id}")
async def get_attachment(order_id: str, attachment_id: str) -> Response:
    """Stream one stored file for the document viewer (inline, never sniffed)."""
    row = await asyncio.to_thread(repository.get_attachment_content, order_id, attachment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return Response(
        content=row["CONTENT"] or b"",
        media_type=row["MIME_TYPE"] or "application/octet-stream",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(row['FILE_NAME'])}",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=300",
        },
    )


def _save(order_id: str, update: OrderUpdate) -> None:
    """Apply an OrderUpdate (blocking)."""
    items = None if update.items is None else [item.model_dump(exclude_unset=True) for item in update.items]
    save_user_edits(order_id, update.header, items, update.classification)


def _review_error(exc: ReviewError) -> HTTPException:
    """Map a ReviewError to an HTTP error; 422 carries the blocking issues."""
    detail = {"message": exc.message, "issues": exc.issues} if exc.issues else exc.message
    return HTTPException(status_code=exc.status_code, detail=detail)


@router.put("/orders/{order_id}", response_model=OrderDetail)
async def save_order(order_id: str, update: OrderUpdate) -> OrderDetail:
    """Save reviewer edits: changed fields become user-confirmed, the order is re-validated."""
    try:
        await asyncio.to_thread(_save, order_id, update)
    except ReviewError as exc:
        raise _review_error(exc) from exc
    return await asyncio.to_thread(build_detail, order_id)


@router.post("/orders/{order_id}/confirm", response_model=ConfirmResponse)
async def confirm(order_id: str, update: Optional[OrderUpdate] = None) -> ConfirmResponse:
    """Optionally save edits, re-validate and post; 422 lists the ERROR issues that block posting."""
    try:
        if update is not None and (update.header or update.items is not None or update.classification):
            await asyncio.to_thread(_save, order_id, update)
        sales_doc = await asyncio.to_thread(confirm_order, order_id, post_order)
    except ReviewError as exc:
        raise _review_error(exc) from exc
    except PostingError as exc:
        detail = {"message": exc.message, "issues": exc.messages} if exc.messages else exc.message
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    return ConfirmResponse(order_id=order_id, status="POSTED", sales_doc=alpha_out(sales_doc))


@router.post("/orders/{order_id}/reprocess", response_model=StatusResponse, status_code=202)
async def reprocess(order_id: str, request: Request, body: Optional[ReprocessRequest] = None) -> StatusResponse:
    """Queue the order for a new agent run, optionally with an instruction for the agent."""
    try:
        await asyncio.to_thread(request_reprocess, order_id, body.note if body else None)
    except ReviewError as exc:
        raise _review_error(exc) from exc
    notify_pipeline(request)
    return StatusResponse(order_id=order_id, status="QUEUED")


@router.delete("/orders/{order_id}", response_model=DeleteResponse)
async def delete(order_id: str) -> DeleteResponse:
    """PoC only: delete an order (and its mock sales document) so the document can be uploaded again."""
    try:
        await asyncio.to_thread(delete_order, order_id)
    except ReviewError as exc:
        raise _review_error(exc) from exc
    return DeleteResponse(order_id=order_id)


@router.get("/sales-orders", response_model=SalesOrderListResponse)
async def sales_orders(limit: int = Query(default=100, ge=1, le=1000)) -> SalesOrderListResponse:
    """Order history: sales documents posted to the (mock) ECC system."""
    rows = await asyncio.to_thread(list_sales_orders, limit)
    return SalesOrderListResponse(items=[SalesOrderSummary(
        sales_doc=alpha_out(r["VBELN"]), order_type=r.get("AUART"), sales_org=r.get("VKORG"),
        sold_to=alpha_out(r.get("KUNNR")), sold_to_name=r.get("SOLD_TO_NAME"), ship_to=alpha_out(r.get("SHIP_TO")),
        ship_to_name=r.get("SHIP_TO_NAME"), po_number=r.get("BSTKD"), po_date=r.get("BSTDK"), currency=r.get("WAERK"),
        net_value=r.get("NETWR"), item_count=r.get("ITEM_COUNT") or 0, created_on=r.get("ERDAT"),
        order_id=r.get("ORDER_ID"),
    ) for r in rows])
