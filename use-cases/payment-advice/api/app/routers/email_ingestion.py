"""API-key protected email inbox and scoped payment-advice review endpoints."""
import asyncio
from email.utils import parseaddr
from urllib.parse import quote
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from ..security import get_api_key
from ..email_ingestion import corrections
from ..email_ingestion.store import Conflict
from ..email_ingestion.intake import intake, refresh_email
from ..email_ingestion.gmail import binding, GmailError
from ..email_ingestion.domain import MAX_FILE, MAX_TOTAL, MAX_ATTACHMENTS

router = APIRouter(dependencies=[Depends(get_api_key)])
advices = APIRouter(dependencies=[Depends(get_api_key)])


class IntakeSizeLimit:
    """Bound multipart intake before parser spooling, including chunked requests."""
    def __init__(self, app):
        """Wrap the ASGI application without buffering the uploaded body."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Reject an oversized manual intake body; pass other endpoints unchanged."""
        if scope['type'] != 'http' or scope.get('path') != '/api/email-ingestion/manual':
            return await self.app(scope, receive, send)
        limit = MAX_TOTAL + 2 * 1024 * 1024  # one MiB body plus multipart overhead
        headers = dict(scope.get('headers', []))
        length = headers.get(b'content-length', b'0')
        if length.isdigit() and int(length) > limit:
            return await Response('Email exceeds the intake size limit', status_code=413)(scope, receive, send)
        total = 0

        async def bounded_receive():
            """Count streamed bytes and stop before exceeding the intake ceiling."""
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > limit:
                raise HTTPException(413, 'Email exceeds the intake size limit')
            return message

        return await self.app(scope, bounded_receive, send)


def storage(request: Request):
    """Require healthy HANA workspace storage; never silently use local persistence."""
    store = getattr(request.app.state, 'workspace', None)
    if store is None:
        raise HTTPException(503, 'HANA workspace is unavailable')
    return store


def checked(function, *args, **kwargs):
    """Map expected domain failures to safe HTTP errors without database/credential leakage."""
    try:
        return function(*args, **kwargs)
    except KeyError:
        raise HTTPException(404, 'Item not found') from None
    except (Conflict, IntegrityError):
        raise HTTPException(409, 'Item changed. Reload before retrying.') from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get('/status')
def status(request: Request):
    """Report configuration and last durable fetch without revealing authentication material."""
    store = storage(request)
    try:
        binding()
        configured, message = True, 'Destination service bound; fetch to verify Gmail authorization'
    except GmailError as exc:
        configured, message = False, str(exc)
    try:
        checkpoint = store.get('gmail-checkpoint')
    except KeyError:
        checkpoint = {}
    jobs = store.list('job', limit=100)
    fetch = next((j for j in jobs if j.get('operation') == 'fetch'), None)
    return {'configured': configured, 'message': message, 'last_successful_fetch': checkpoint.get('last_successful_fetch'),
            'mailbox': checkpoint.get('mailbox'), 'last_fetch': fetch}


@router.post('/fetch', status_code=202)
def fetch(request: Request):
    """Queue a Gmail fetch immediately; worker persists its progress in HANA."""
    store = storage(request)
    try:
        binding()
    except GmailError as exc:
        raise HTTPException(503, str(exc)) from None
    for state in ('queued', 'processing'):
        existing = next((j for j in store.list('job', status=state) if j.get('operation') == 'fetch'), None)
        if existing:
            return {'run_id': existing['id']}
    return {'run_id': store.insert('job', {'operation': 'fetch', 'processed': 0}, status='queued')}


@router.get('/runs/{run_id}')
def run(run_id: str, request: Request):
    """Read one processing run's durable state and counters."""
    record = checked(storage(request).get, run_id)
    if record['kind'] != 'job':
        raise HTTPException(404, 'Run not found')
    return record


@router.post('/manual', status_code=202)
async def manual(request: Request, sender: str = Form(..., max_length=320), subject: str = Form('', max_length=1000),
                 body: str = Form('', max_length=1000000), customer: str = Form(''), files: list[UploadFile] = File(default=[])):
    """Normalize manual email inputs and enqueue exactly the same pipeline as Gmail."""
    name, address = parseaddr(sender)
    if '@' not in address or '\n' in sender or '\r' in sender:
        raise HTTPException(400, 'A valid sender email address is required')
    attachments, warnings, total = [], [], 0
    try:
        for index, file in enumerate(files):
            if index >= MAX_ATTACHMENTS:
                warnings.append({'filename': (file.filename or '')[:200], 'error': 'Maximum ten attachments'})
                continue
            data = await file.read(MAX_FILE + 1)
            total += len(data)
            if len(data) > MAX_FILE or total > MAX_TOTAL:
                warnings.append({'filename': (file.filename or '')[:200], 'error': 'Attachment size limit exceeded'})
                continue
            attachments.append((file.filename or 'attachment', data))
        email_id = await asyncio.to_thread(checked, intake, storage(request), {'sender': address, 'sender_name': name,
            'subject': subject, 'body': body, 'customer': customer or None, 'attachments': attachments, 'warnings': warnings})
        return {'email_id': email_id}
    finally:
        for file in files:
            await file.close()


@router.get('/inbox')
def inbox(request: Request, status: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    """Return an email-level page and unfiltered statistics from HANA."""
    store = storage(request)
    rows = store.list('email', status=status, offset=offset, limit=limit)
    return {'emails': [{k: v for k, v in r.items() if k != 'body'} for r in rows], 'counts': store.counts()}


@router.get('/inbox/{email_id}')
def detail(email_id: str, request: Request):
    """Return the original email and its advice/attachment metadata without file bytes."""
    store = storage(request)
    email = checked(store.get, email_id)
    if email['kind'] != 'email':
        raise HTTPException(404, 'Email not found')
    return {'email': email, 'advices': store.list('advice', parent=email_id), 'attachments': store.list('attachment', parent=email_id)}


@router.get('/attachments/{attachment_id}')
def attachment(attachment_id: str, request: Request, preview: bool = False):
    """Serve original bytes only with API authorization, or a bounded inert preview."""
    from ..email_ingestion.preview import preview as text_preview
    store = storage(request)
    item = checked(store.get, attachment_id)
    if item['kind'] != 'attachment':
        raise HTTPException(404, 'Attachment not found')
    content = store.content(attachment_id)
    if preview:
        if item.get('error'):
            return {'text': item['error'], 'mime_type': item['mime_type']}
        return {'text': checked(text_preview, item['filename'], content), 'mime_type': item['mime_type']}
    return Response(content, media_type=item['mime_type'], headers={
        'Content-Disposition': "attachment; filename*=UTF-8''" + quote(item['filename'], safe=''),
        'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store', 'Content-Security-Policy': "sandbox; default-src 'none'"})


class Revision(BaseModel):
    """Required optimistic concurrency token for an advice mutation."""
    revision: int = Field(ge=0)


@router.delete('/inbox/{email_id}')
def delete_email(email_id: str, body: Revision, request: Request):
    """Delete one complete idle inbox entry under optimistic concurrency control."""
    return {'deleted': checked(storage(request).delete_email, email_id, body.revision)}


class Edit(Revision):
    """An explicit, typed correction targeting a unique row or header."""
    target: str = Field(max_length=200)
    field: str = Field(max_length=100)
    value: object
    reason: str = Field(min_length=1, max_length=10000)


class FieldEdit(BaseModel):
    """One typed cell correction inside a batch; same bounds as a single Edit."""
    target: str = Field(max_length=200)
    field: str = Field(max_length=100)
    value: object


class EditMany(Revision):
    """Several cell corrections accepted together from the review table as one revision."""
    edits: list[FieldEdit] = Field(min_length=1, max_length=500)
    reason: str = Field(default='Edited in review table', min_length=1, max_length=10000)


class CustomerChange(Revision):
    """Explicit customer selection, queued for safe reprocessing."""
    client_key: str = Field(max_length=60)


class Chat(Revision):
    """Only ephemeral browser conversation is accepted; result scope stays server-owned."""
    message: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=40)


@advices.get('/{advice_id}')
def result(advice_id: str, request: Request):
    """Read the current result, original evidence and revision of one advice."""
    return checked(corrections.advice_record, storage(request), advice_id)


@advices.post('/{advice_id}/edit')
def edit(advice_id: str, body: Edit, request: Request):
    """Persist a field correction and refresh the email's review status."""
    return checked(corrections.edit, storage(request), advice_id, **body.model_dump())


@advices.post('/{advice_id}/edits')
def edit_many(advice_id: str, body: EditMany, request: Request):
    """Persist all accepted table edits atomically, or none if any value is invalid."""
    edits = [item.model_dump() for item in body.edits]
    return checked(corrections.edit_many, storage(request), advice_id, body.revision, edits, body.reason)


@advices.post('/{advice_id}/revert')
def revert(advice_id: str, body: Revision, request: Request):
    """Remove all corrections and restore the values produced by extraction and interpretation."""
    return checked(corrections.revert_all, storage(request), advice_id, body.revision)


@advices.post('/{advice_id}/undo')
def undo(advice_id: str, body: Revision, request: Request):
    """Undo the last correction under the supplied advice revision."""
    return checked(corrections.undo, storage(request), advice_id, body.revision)


@advices.post('/{advice_id}/review/{action}')
def review(advice_id: str, action: str, body: Revision, request: Request):
    """Mark reviewed or reopen locally; posting to S/4HANA is the separate /s4/post step."""
    store = storage(request)
    advice = checked(corrections.advice_record, store, advice_id, body.revision, mutable=True)
    if action not in {'reviewed', 'reopen'} or advice['status'] in {'queued', 'processing', 'failed'}:
        raise HTTPException(400, 'This advice cannot be reviewed yet')
    advice['status'] = 'reviewed' if action == 'reviewed' else 'needs_review'
    saved = checked(store.update, advice)
    checked(refresh_email, store, advice['parent_id'])
    return saved


def queue_retry(store, advice_id, revision, client_key=None):
    """Atomically enqueue a single retry, preserving corrections and original results."""
    advice = corrections.advice_record(store, advice_id, revision, mutable=True)
    if advice['status'] in {'queued', 'processing'}:
        raise Conflict('Already processing')
    if client_key is not None:
        from ..payment_advice.customers import get_customer
        customer = get_customer(store.engine, client_key)
        if not customer or customer.status != 'active':
            raise ValueError('Select an active registered customer')
        advice.update(client_key=client_key, matching={'method': 'manual', 'client_key': client_key})
    advice.update(status='queued', stage='queued', error=None)
    with store.engine.begin() as conn:
        saved = store.update(advice, conn=conn)
        run_id = store.insert('job', {'advice_id': advice_id}, parent=advice_id, status='queued', conn=conn)
    refresh_email(store, advice['parent_id'])
    return {**saved, 'run_id': run_id}


@advices.post('/{advice_id}/retry')
def retry(advice_id: str, body: Revision, request: Request):
    """Retry extraction without silently dropping analyst corrections."""
    return checked(queue_retry, storage(request), advice_id, body.revision)


@advices.post('/{advice_id}/customer')
def customer(advice_id: str, body: CustomerChange, request: Request):
    """Correct the customer and reprocess through its schema/playbook selection."""
    return checked(queue_retry, storage(request), advice_id, body.revision, body.client_key)


@advices.post('/{advice_id}/chat')
async def chat(advice_id: str, body: Chat, request: Request):
    """Invoke an ephemeral, server-bound advice assistant, returning changed row IDs."""
    from ..email_ingestion.assistant import chat as advice_chat
    import json
    if len(json.dumps(body.history)) > 100000:
        raise HTTPException(400, 'Conversation context is too large')
    runtime = getattr(request.app.state, 'runtime', None)
    if runtime is None:
        raise HTTPException(503, 'Advice assistant is unavailable')
    store = storage(request)
    checked(corrections.advice_record, store, advice_id, body.revision)
    try:
        return await advice_chat(runtime, store, advice_id, body.revision, body.message, body.history,
                                 dox=getattr(request.app.state, 'dox', None),
                                 settings=getattr(request.app.state, 'settings', None))
    except Conflict:
        raise HTTPException(409, 'Advice changed; reload before retrying') from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except Exception:
        raise HTTPException(503, 'Advice assistant failed. Reload to check whether any correction was applied.') from None


@advices.get('/{advice_id}/proposals')
def proposals(advice_id: str, request: Request):
    """List pending and confirmed proposals scoped to this advice."""
    store = storage(request)
    checked(corrections.advice_record, store, advice_id)
    return store.list('proposal', parent=advice_id)


@advices.post('/{advice_id}/proposals/{proposal_id}/confirm')
def confirm(advice_id: str, proposal_id: str, request: Request):
    """Explicitly confirm one immutable proposal ID; the agent cannot call this route."""
    return checked(corrections.confirm, storage(request), advice_id, proposal_id)
