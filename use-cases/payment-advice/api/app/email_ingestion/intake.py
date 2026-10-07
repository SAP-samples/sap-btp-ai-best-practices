"""Transactional intake shared by manual messages and Gmail synchronization."""
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4, uuid5, NAMESPACE_URL
from pathlib import Path
from .domain import MAX_ATTACHMENTS, MAX_TOTAL, MAX_FILE, prepare_attachment, match_customer, email_status
from .store import Conflict
from ..payment_advice.customers import list_customers, find_customers


def intake(store, message, source='manual', mailbox='', owner=None):
    """Save an email, original attachments and one durable job per supported advice."""
    email_id = str(uuid5(NAMESPACE_URL, mailbox + ':' + message['gmail_id'])) if source == 'gmail' else str(uuid4())
    try:
        store.get(email_id)
        return email_id
    except KeyError:
        pass
    matching = match_customer(message['sender'], message.get('sender_name', ''),
        [asdict(c) for c in list_customers(store.engine)],
        lambda q: find_customers(store.engine, q), message.get('customer'),
        subject=message.get('subject', ''), body=message.get('body', ''))
    warnings, prepared, originals, total = list(message.get('warnings', [])), [], [], 0
    for index, (filename, data) in enumerate(message.get('attachments', [])):
        try:
            total += len(data)
            if index >= MAX_ATTACHMENTS or total > MAX_TOTAL:
                raise ValueError('Email attachment limit exceeded')
            prepared.append(prepare_attachment(filename, data))
        except ValueError as exc:
            warnings.append({'filename': filename[:200], 'error': str(exc)})
            if index < MAX_ATTACHMENTS and total <= MAX_TOTAL and len(data) <= MAX_FILE:
                originals.append({'filename': Path(filename.replace('\\', '/')).name.replace('\x00', '')[:200],
                    'size': len(data), 'mime_type': 'application/octet-stream', 'error': str(exc), 'content': data})
    body = message.get('body', '')
    body_only = not prepared and bool(body.strip())
    if body_only:
        prepared.append(prepare_attachment('email-body.txt', body.encode('utf-8')))
    if not prepared:
        warnings.append({'error': 'No supported document or nonempty email body to process'})
    email = {k: message.get(k, '') for k in ('sender', 'sender_name', 'subject', 'body', 'gmail_id')}
    email.update(source=source, matching=matching, client_key=matching['client_key'], warnings=warnings,
        received_at=message.get('received_at') or datetime.now(timezone.utc).isoformat(),
        attachment_count=len(message.get('attachments', [])) + len(message.get('warnings', [])), payment_references=[])
    with store.engine.begin() as conn:
        if owner:
            store.fence(conn, owner)
        store.insert('email', email, status='processing' if prepared else 'needs_review', record_id=email_id, conn=conn)
        for original in originals:
            content = original.pop('content')
            store.insert('attachment', original, parent=email_id, content=content, conn=conn)
        for index, attachment in enumerate(prepared):
            content = attachment.pop('content')
            attachment['body_only'] = body_only
            attachment_id = store.insert('attachment', attachment, parent=email_id, content=content, conn=conn)
            advice_id = str(uuid5(NAMESPACE_URL, email_id + ':' + str(index)))
            store.insert('advice', {'attachment_id': attachment_id, 'filename': attachment['filename'],
                'client_key': matching['client_key'], 'matching': matching, 'corrections': []},
                parent=email_id, record_id=advice_id, status='queued', conn=conn)
            store.insert('job', {'advice_id': advice_id}, parent=advice_id, status='queued', conn=conn)
    return email_id


def refresh_email(store, email_id, conn=None):
    """Recompute the complete rollup after a CAS conflict, including fresh child states."""
    if conn is not None:
        # Terminal worker results/job state/rollup commit together. The email lock
        # serializes siblings without holding it through document processing.
        store.lock_record(email_id, conn)
    while True:
        email = store.get(email_id, conn)
        advices = store.list('advice', parent=email_id, conn=conn)
        email['status'] = email_status([a['status'] for a in advices], email.get('warnings'))
        email['payment_references'] = list(dict.fromkeys(str(a.get('result', {}).get('header', {}).get('payment_reference'))
            for a in advices if a.get('result', {}).get('header', {}).get('payment_reference')))
        customers = {a.get('client_key') for a in advices}
        email['client_key'] = next(iter(customers)) if len(customers) == 1 else None
        try:
            return store.update(email, conn=conn)
        except Conflict:
            if conn is not None:
                raise
            continue
