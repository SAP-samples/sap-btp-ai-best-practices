"""Transactional in-memory repository double, exclusively for offline tests."""
import copy
import threading
from contextlib import contextmanager
from uuid import uuid4
from app.email_ingestion.store import Conflict


class MemoryStore:
    """Emulate repository revisions/rollback, never used by application code."""
    def __init__(self):
        """Start with empty records and binary content."""
        self.records, self.bytes = {}, {}
        self.engine = self
        self._transaction_lock = threading.RLock()

    @contextmanager
    def begin(self):
        """Restore the snapshot when a transaction raises."""
        with self._transaction_lock:
            before = copy.deepcopy((self.records, self.bytes))
            try:
                yield self
            except Exception:
                self.records, self.bytes = before
                raise

    def insert(self, kind, payload, parent='', status='ready', record_id=None, content=None, conn=None):
        """Insert a fresh record; reject duplicate primary keys."""
        record_id = record_id or str(uuid4())
        if record_id in self.records:
            raise Conflict('duplicate')
        self.records[record_id] = copy.deepcopy({**payload, 'id': record_id, 'kind': kind, 'parent_id': parent, 'status': status, 'revision': 0})
        self.bytes[record_id] = content
        return record_id

    def get(self, record_id, conn=None):
        """Read a detached record like decoded JSON from HANA."""
        return copy.deepcopy(self.records[record_id])

    def update(self, record, expected=None, conn=None):
        """Reject stale revisions and increment accepted updates."""
        expected = record['revision'] if expected is None else expected
        if self.records[record['id']]['revision'] != expected:
            raise Conflict('stale')
        self.records[record['id']] = copy.deepcopy({**record, 'revision': expected + 1})
        return self.get(record['id'])

    def delete_email(self, email_id, revision):
        """Delete an idle email tree with the same revision contract as HANA storage."""
        with self.begin():
            email = self.get(email_id)
            if email['kind'] != 'email':
                raise KeyError(email_id)
            if email['revision'] != revision:
                raise Conflict('stale')
            if any(record['status'] in {'queued', 'processing'} for record in self.list('advice', parent=email_id)):
                raise ValueError('Wait for processing to finish before deleting this entry')
            targets = {email_id}
            while descendants := {record_id for record_id, record in self.records.items() if record['parent_id'] in targets} - targets:
                targets.update(descendants)
            for record_id in targets:
                self.records.pop(record_id, None)
                self.bytes.pop(record_id, None)
        return email_id

    def list(self, kind, parent=None, limit=100, offset=0, status=None, conn=None):
        """List records satisfying the repository filters."""
        return [copy.deepcopy(r) for r in reversed(list(self.records.values())) if r['kind'] == kind
                and (parent is None or r['parent_id'] == parent) and (not status or r['status'] == status)][offset:offset+limit]

    def lock_record(self, record_id, conn):
        """Validate the row; begin() already holds this double's transaction lock."""
        self.get(record_id)

    @contextmanager
    def canonical_schema_guard(self):
        """Provide the test equivalent of a scoped schema-creation transaction."""
        with self._transaction_lock:
            yield

    def fence(self, conn, owner, *, require_running=True):
        """Fake one valid worker owner; lost ownership raises for fencing tests."""
        if owner == 'lost':
            raise Conflict('lease lost')

    def content(self, record_id):
        """Read retained original bytes."""
        return self.bytes[record_id]
