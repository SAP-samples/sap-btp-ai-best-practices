"""Pure email normalization, customer matching and validated advice edits."""
from __future__ import annotations

import copy
import math
import mimetypes
import re
from datetime import date
from pathlib import Path
from typing import Any

MAX_FILE = 25 * 1024 * 1024
MAX_TOTAL = 50 * 1024 * 1024
MAX_ATTACHMENTS = 10
EXTENSIONS = {'.pdf', '.xlsx', '.csv', '.tsv', '.txt', '.docx', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
GENERIC = {'noreply', 'no', 'reply', 'mail', 'email', 'com', 'net', 'org', 'co', 'uk', 'ca', 'gmail', 'outlook', 'hotmail', 'www'}
# Legal-form suffixes ignored when looking for a customer name inside free text,
# so "Northwind Inc." is found in a body that only says "Northwind".
LEGAL = {'inc', 'ltd', 'llc', 'gmbh', 'corp', 'corporation', 'co', 'company', 'sa', 'sarl', 'bv', 'nv', 'ag', 'plc', 'limited', 'the'}
# Subject words that describe the email itself, never the customer; kept out of fuzzy search.
SUBJECT_STOP = {'re', 'fw', 'fwd', 'payment', 'payments', 'remittance', 'advice', 'invoice', 'invoices',
                'from', 'for', 'and', 'ref', 'reference', 'notification', 'statement'}
# ponytail: only the head of the body is scanned; the customer is named near the top while quoted
# history and disclaimers below are noise. Common-word customer names (e.g. "Target") can still
# false-hit here; mitigated by lowest precedence, uniqueness and analyst override.
BODY_SCAN_CHARS = 20000


def _text_hits(text: str, active: dict) -> list[str]:
    """Find customers whose name appears as whole words inside free text.

    Args:
        text: Subject or body text to scan.
        active: Active customers keyed by client_key (each with a 'display_name').

    Returns:
        Sorted client keys whose client_key or display_name (legal suffixes removed,
        at least 3 characters) occurs as a contiguous whole-word sequence in text.
    """
    # Space-padded token string: substring checks on ' needle ' respect word boundaries,
    # so customer "Other" does not hit the word "otherwise".
    hay = ' ' + ' '.join(re.findall(r'[a-z0-9]+', text.lower())) + ' '
    hits = []
    for key, c in active.items():
        for name in (key, c['display_name']):
            needle = ' '.join(t for t in re.findall(r'[a-z0-9]+', name.lower()) if t not in LEGAL)
            if len(needle) >= 3 and f' {needle} ' in hay:
                hits.append(key)
                break
    return sorted(hits)


def prepare_attachment(filename: str, content: bytes) -> dict:
    """Validate an uploaded file and return safe metadata plus original bytes."""
    filename = Path(filename.replace('\\', '/')).name.replace('\x00', '')[:200]
    ext = Path(filename).suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError(f'Unsupported attachment: {filename}')
    if len(content) > MAX_FILE:
        raise ValueError(f'Attachment exceeds 25 MB: {filename}')
    if not content:
        raise ValueError(f'Empty attachment: {filename}')
    if ext == '.pdf' and not content.startswith(b'%PDF-'):
        raise ValueError(f'Invalid PDF: {filename}')
    if ext in {'.xlsx', '.docx'} and not content.startswith(b'PK'):
        raise ValueError(f'Invalid Office file: {filename}')
    if ext in {'.xlsx', '.docx'}:
        from io import BytesIO
        from zipfile import ZipFile, BadZipFile
        try:
            with ZipFile(BytesIO(content)) as archive:
                if len(archive.infolist()) > 10000 or sum(i.file_size for i in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError('Office archive exceeds the expanded size limit')
                expected = 'xl/workbook.xml' if ext == '.xlsx' else 'word/document.xml'
                if expected not in archive.namelist():
                    raise ValueError('Office archive does not contain the expected document')
        except BadZipFile:
            raise ValueError('Invalid Office archive') from None
    if ext == '.png' and not content.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError(f'Invalid PNG: {filename}')
    if ext in {'.jpg', '.jpeg'} and not content.startswith(b'\xff\xd8'):
        raise ValueError(f'Invalid JPEG: {filename}')
    if ext in {'.tif', '.tiff'} and not content.startswith((b'II*\x00', b'MM\x00*')):
        raise ValueError(f'Invalid TIFF: {filename}')
    return {'filename': filename, 'size': len(content),
            'mime_type': mimetypes.guess_type(filename)[0] or 'application/octet-stream', 'content': content}


def match_customer(sender: str, sender_name: str, customers: list[dict], search: Any,
                   override: str | None = None, subject: str = '', body: str = '') -> dict:
    """Resolve strong customer matches; retain ranked evidence for canonical fallback.

    Exact evidence is tiered: sender address/name, then subject, then the head of
    the body. The first source with any exact hit decides; a unique hit is accepted,
    several hits leave the email unconfirmed. Otherwise sender and subject terms go
    through fuzzy search, accepted only at >= 0.85 with a 0.15 lead.

    Args:
        sender: Sender email address.
        sender_name: Sender display name (may be empty).
        customers: Registered customers as dicts with client_key, display_name, status.
        search: Fuzzy search callable returning (client_key, display_name, score) tuples.
        override: Analyst-chosen client_key, bypassing matching.
        subject: Email subject.
        body: Plain-text email body.

    Returns:
        Dict with client_key (None when unconfirmed), method (manual, exact_sender,
        exact_subject, exact_body, fuzzy or unconfirmed), ranked candidates and, when
        matched automatically, per-source exact hits and the fuzzy terms used.
    """
    active = {c['client_key']: c for c in customers if c.get('status', 'active') == 'active'}
    if override:
        if override not in active:
            raise ValueError('Customer override must identify an active registered customer')
        return {'client_key': override, 'method': 'manual', 'candidates': []}
    domain = sender.rsplit('@', 1)[-1].lower()
    terms = set(re.findall(r'[a-z0-9]+', domain)) - GENERIC
    if sender_name.strip():
        terms.add(sender_name.strip().lower())
    exact = []
    for key, c in active.items():
        names = {re.sub(r'[^a-z0-9]', '', s.lower()) for s in (key, c['display_name'])}
        if any(re.sub(r'[^a-z0-9]', '', term) in names for term in terms):
            exact.append(key)
    # Tiered exact evidence: a lower-precedence source is only read when every
    # higher one produced no hit at all.
    hits = {'sender': sorted(exact), 'subject': [], 'body': []}
    decided = hits['sender']
    if not decided:
        decided = hits['subject'] = _text_hits(subject, active)
    if not decided:
        decided = hits['body'] = _text_hits(body[:BODY_SCAN_CHARS], active)
    if len(decided) == 1:
        source = next(s for s in ('sender', 'subject', 'body') if hits[s])
        return {'client_key': decided[0], 'method': 'exact_' + source,
                'candidates': [{'client_key': decided[0], 'score': 1.0}], 'hits': hits}
    # Subject words join the fuzzy terms; descriptive words, numbers and short tokens
    # are dropped and the count is capped to bound the number of HANA queries.
    subject_terms = [t for t in re.findall(r'[a-z0-9]+', subject.lower())
                     if len(t) >= 3 and not t.isdigit() and t not in GENERIC | SUBJECT_STOP | LEGAL]
    terms |= set(list(dict.fromkeys(subject_terms))[:10])
    scores = {}
    for term in sorted(terms):
        for key, name, score in search(term):
            if key in active:
                scores[key] = max(scores.get(key, 0), float(score))
    ranked = [{'client_key': k, 'score': v} for k, v in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    first = ranked[0]['score'] if ranked else 0
    second = ranked[1]['score'] if len(ranked) > 1 else 0
    # Ambiguous exact evidence (2+ hits in the deciding source) blocks fuzzy acceptance.
    key = ranked[0]['client_key'] if ranked and first >= .85 and first - second >= .15 and len(decided) < 2 else None
    return {'client_key': key, 'method': 'fuzzy' if key else 'unconfirmed', 'candidates': ranked,
            'terms': sorted(terms), 'hits': hits}


def email_status(statuses: list[str], warnings: list | None = None) -> str:
    """Roll document states into one email state without masking partial failures."""
    if any(s in {'queued', 'processing'} for s in statuses):
        return 'processing'
    if statuses and all(s == 'posted' for s in statuses):
        return 'posted'
    if statuses and all(s in {'reviewed', 'posted'} for s in statuses):
        return 'reviewed'
    if statuses and all(s == 'failed' for s in statuses):
        return 'failed'
    if not statuses or warnings or any(s in {'failed', 'needs_review'} for s in statuses):
        return 'needs_review'
    return 'ready'


def apply_edit(payload: dict, target: str, field: str, value: Any) -> tuple[dict, list[str]]:
    """Return a copied advice with one typed field changed; reject ambiguous targets."""
    from app.payment_advice.canonical import (DERIVED_HEADER_FIELD_NAMES, DERIVED_LINE_FIELD_NAMES,
                                              HEADER_FIELD_NAMES, LINE_FIELD_NAMES)
    result = copy.deepcopy(payload)
    # Rule-derived fields (customer account, payer, company code) are corrected like extracted ones.
    allowed = set(HEADER_FIELD_NAMES) | set(DERIVED_HEADER_FIELD_NAMES) if target == 'header' else \
        set(LINE_FIELD_NAMES) | set(DERIVED_LINE_FIELD_NAMES) | {'reason_code', 'document_nature', 'residual_items', 'rationale', 'flags'}
    if field not in allowed:
        raise ValueError('Unsupported editable field')
    if target == 'header':
        row, ids = result['header'], []
    else:
        rows = [r for r in result['line_items'] if r.get('row_id') == target]
        if not rows:
            rows = [r for r in result['line_items'] if target in (r.get('invoice_reference'), r.get('customer_document_reference'))]
        if len(rows) != 1:
            raise ValueError('Reference is ambiguous or missing; specify a unique row_id')
        row, ids = rows[0], [rows[0]['row_id']]
    if value is not None:
        if field.endswith('_amount'):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Amounts must be finite numbers')
        elif field.endswith('_date'):
            if not isinstance(value, str):
                raise ValueError('Dates must be ISO YYYY-MM-DD strings')
            date.fromisoformat(value)
        elif field == 'residual_items':
            refs = {r.get('invoice_reference') for r in result['line_items']}
            if not isinstance(value, list) or any(not isinstance(v, str) or v not in refs for v in value):
                raise ValueError('Linked invoices must exist in this advice')
        elif field == 'flags':
            if not isinstance(value, list) or len(value) > 50 or any(not isinstance(v, str) or len(v) > 200 for v in value):
                raise ValueError('Flags must be a list of at most 50 texts of at most 200 characters')
        elif not isinstance(value, str) or len(value) > 10000:
            raise ValueError('Value must be text of at most 10000 characters')
        if field == 'document_nature' and value not in {'invoice', 'credit_memo', 'chargeback', 'allowance_reversal', 'unknown'}:
            raise ValueError('Invalid document nature')
    row[field] = value
    return result, ids


def stable_rows(rows: list[dict], previous: list[dict], advice_id: str):
    """Retain unique row identities across reordering; new or ambiguous rows get fresh IDs."""
    from uuid import uuid4
    from app.payment_advice.canonical import LINE_FIELD_NAMES
    for row in rows:
        key = (row.get('invoice_reference'), row.get('customer_document_reference'))
        candidates = [r for r in previous if (r.get('invoice_reference'), r.get('customer_document_reference')) == key]
        peers = [r for r in rows if (r.get('invoice_reference'), r.get('customer_document_reference')) == key]
        if not any(key) or len(candidates) != 1 or len(peers) != 1:
            candidates = [r for r in candidates if all(r.get(k) == row.get(k) for k in LINE_FIELD_NAMES)]
            peers = [r for r in peers if all(r.get(k) == row.get(k) for k in LINE_FIELD_NAMES)]
        if len(candidates) == 1 and len(peers) == 1:
            row['row_id'] = candidates[0]['row_id']
        else:
            row['row_id'] = advice_id + ':' + uuid4().hex[:16]
