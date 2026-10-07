"""HANA record repository: atomic revisions, durable jobs, attachments and audit.

Only this module knows SQL. Records use application-generated UUIDs and a small
typed envelope; business payloads are JSON NCLOBs, original files are BLOBs.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from uuid import uuid4
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from app.payment_advice.hana_schema import ColumnContract, assert_table_contract

TABLE = 'PAYMENT_ADVICE_EXTRACTOR_WORKSPACE'
ADVICE_SLOTS = tuple(f'workspace-advice-{index}' for index in range(4))
FETCH_SLOT = 'workspace-fetch'
LEASE_SECONDS = 120


class Conflict(ValueError):
    """A record changed since the caller read it; reload before retrying."""


class Store:
    """Persist workspace records with explicit HANA transactions."""

    def __init__(self, engine):
        """Bind the existing application's SQLAlchemy HANA engine."""
        self.engine = engine
        self._locks_initialized = set()

    def ensure(self):
        """Create missing storage and validate existing catalog contracts."""
        contract = {
            'ID': ColumnContract('NVARCHAR', False, length=64, primary_key=True),
            'KIND': ColumnContract('NVARCHAR', False, length=24),
            'PARENT_ID': ColumnContract('NVARCHAR', False, length=64),
            'STATUS': ColumnContract('NVARCHAR', False, length=24),
            'REVISION': ColumnContract('INTEGER', False),
            'UPDATED_AT': ColumnContract('TIMESTAMP', False),
            'LEASE_UNTIL': ColumnContract('TIMESTAMP', True),
            'PAYLOAD': ColumnContract('NCLOB', False),
            'CONTENT': ColumnContract('BLOB', True),
        }
        with self.engine.begin() as conn:
            schema = conn.execute(text('SELECT CURRENT_SCHEMA FROM DUMMY')).scalar()
            exists = conn.execute(text('SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME=:s AND TABLE_NAME=:t'), {'s': schema, 't': TABLE}).scalar()
            if not exists:
                conn.execute(text(f'''CREATE COLUMN TABLE "{TABLE}" (
                    "ID" NVARCHAR(64) PRIMARY KEY, "KIND" NVARCHAR(24) NOT NULL,
                    "PARENT_ID" NVARCHAR(64) NOT NULL, "STATUS" NVARCHAR(24) NOT NULL,
                    "REVISION" INTEGER NOT NULL, "UPDATED_AT" TIMESTAMP NOT NULL,
                    "LEASE_UNTIL" TIMESTAMP, "PAYLOAD" NCLOB NOT NULL, "CONTENT" BLOB)'''))
            assert_table_contract(conn, schema, TABLE, contract)
        return schema

    @staticmethod
    def decode(row):
        """Normalize HANA mapping keys and decode one JSON payload."""
        r = {k.lower(): v for k, v in row.items()}
        payload = r['payload'].read() if hasattr(r['payload'], 'read') else r['payload']
        return {**json.loads(payload), **{k: r[k] for k in ('id', 'kind', 'parent_id', 'status', 'revision')}, 'updated_at': str(r['updated_at'])}

    def get(self, record_id, conn=None):
        """Read metadata without loading file bytes; raise for an absent record."""
        if conn is None:
            with self.engine.connect() as c:
                return self.get(record_id, c)
        row = conn.execute(text(f'SELECT "ID","KIND","PARENT_ID","STATUS","REVISION","UPDATED_AT","PAYLOAD" FROM "{TABLE}" WHERE "ID"=:id'), {'id': record_id}).mappings().first()
        if row is None:
            raise KeyError(record_id)
        return self.decode(row)

    def list(self, kind, parent=None, limit=100, offset=0, status=None, conn=None):
        """Return a bounded newest-first page of a record kind."""
        if conn is None:
            with self.engine.connect() as connection:
                return self.list(kind, parent, limit, offset, status, connection)
        where, params = '"KIND"=:kind', {'kind': kind, 'limit': limit, 'offset': offset}
        if parent is not None:
            where += ' AND "PARENT_ID"=:parent'
            params['parent'] = parent
        if status:
            where += ' AND "STATUS"=:status'
            params['status'] = status
        rows = conn.execute(text(f'SELECT "ID","KIND","PARENT_ID","STATUS","REVISION","UPDATED_AT","PAYLOAD" FROM "{TABLE}" WHERE {where} ORDER BY "UPDATED_AT" DESC, "ID" LIMIT :limit OFFSET :offset'), params).mappings().all()
        return [self.decode(r) for r in rows]

    def lock_record(self, record_id, conn):
        """Lock an existing record until the caller's HANA transaction finishes."""
        row = conn.execute(text(f'SELECT "ID" FROM "{TABLE}" WHERE "ID"=:id FOR UPDATE'), {'id': record_id}).first()
        if row is None:
            raise KeyError(record_id)

    def _ensure_lock(self, record_id):
        """Create a known lock row once per repository, safely across processes."""
        if record_id not in self._locks_initialized:
            try:
                self.insert('lock', {}, record_id=record_id)
            except IntegrityError:
                pass
            self._locks_initialized.add(record_id)

    @contextmanager
    def canonical_schema_guard(self):
        """Serialize only shared canonical schema lookup/create/activation, not extraction."""
        record_id = 'workspace-canonical-schema'
        self._ensure_lock(record_id)
        with self.engine.begin() as conn:
            self.lock_record(record_id, conn)
            yield

    def counts(self, kind='email'):
        """Return server-side status counts for the entire inbox."""
        with self.engine.connect() as conn:
            return dict(conn.execute(text(f'SELECT "STATUS",COUNT(*) FROM "{TABLE}" WHERE "KIND"=:kind GROUP BY "STATUS"'), {'kind': kind}).all())

    def insert(self, kind, payload, parent='', status='ready', record_id=None, content=None, conn=None):
        """Insert a record; caller may group related records in one transaction."""
        record_id = record_id or str(uuid4())
        if conn is None:
            with self.engine.begin() as c:
                return self.insert(kind, payload, parent, status, record_id, content, c)
        conn.execute(text(f'''INSERT INTO "{TABLE}" ("ID","KIND","PARENT_ID","STATUS","REVISION","UPDATED_AT","PAYLOAD","CONTENT")
            VALUES (:id,:kind,:parent,:status,0,CURRENT_UTCTIMESTAMP,:payload,:content)'''),
            {'id': record_id, 'kind': kind, 'parent': parent, 'status': status,
             'payload': json.dumps(payload, default=str, allow_nan=False), 'content': content})
        return record_id

    def update(self, record, expected=None, conn=None):
        """Compare-and-swap a record; fail rather than overwrite a concurrent edit."""
        if conn is None:
            with self.engine.begin() as c:
                return self.update(record, expected, c)
        revision = record['revision'] if expected is None else expected
        payload = {k: v for k, v in record.items() if k not in {'id', 'kind', 'parent_id', 'status', 'revision', 'updated_at'}}
        changed = conn.execute(text(f'''UPDATE "{TABLE}" SET "PAYLOAD"=:payload,"STATUS"=:status,
            "REVISION"="REVISION"+1,"UPDATED_AT"=CURRENT_UTCTIMESTAMP WHERE "ID"=:id AND "REVISION"=:revision'''),
            {'id': record['id'], 'revision': revision, 'status': record['status'], 'payload': json.dumps(payload, default=str, allow_nan=False)})
        if changed.rowcount != 1:
            raise Conflict('This item changed. Reload and try again.')
        return self.get(record['id'], conn)

    def delete_email(self, email_id, revision):
        """Atomically delete one idle email and all workspace records owned by it."""
        with self.engine.begin() as conn:
            self.lock_record(email_id, conn)
            email = self.get(email_id, conn)
            if email['kind'] != 'email':
                raise KeyError(email_id)
            if email['revision'] != revision:
                raise Conflict('This item changed. Reload and try again.')
            advices = self.list('advice', parent=email_id, conn=conn)
            locked = []
            for advice in advices:
                self.lock_record(advice['id'], conn)
                current = self.get(advice['id'], conn)
                if current['status'] in {'queued', 'processing'}:
                    raise ValueError('Wait for processing to finish before deleting this entry')
                locked.append(current)
            for advice in locked:
                conn.execute(text(f'DELETE FROM "{TABLE}" WHERE "PARENT_ID"=:parent'), {'parent': advice['id']})
            conn.execute(text(f'DELETE FROM "{TABLE}" WHERE "PARENT_ID"=:email OR "ID"=:email'), {'email': email_id})
        return email_id

    def content(self, record_id):
        """Load original attachment bytes only on demand."""
        with self.engine.connect() as conn:
            data = conn.execute(text(f'SELECT "CONTENT" FROM "{TABLE}" WHERE "ID"=:id'), {'id': record_id}).scalar()
            return data.read() if hasattr(data, 'read') else bytes(data or b'')

    def claim_next(self, slot, owner, *, fetch=False):
        """Atomically reserve a global slot and one queued/expired job; return it or None.

        Existing columns suffice: lock PARENT_ID holds the unique execution owner;
        job LEASE_UNTIL controls crash recovery. Job parent remains the advice ID.
        Every execution gets a fresh owner token, so a late result cannot commit
        after lease takeover, even when the same process recovers its own job.
        """
        if slot not in ((FETCH_SLOT,) if fetch else ADVICE_SLOTS):
            raise ValueError('Invalid processing slot')
        self._ensure_lock(slot)
        with self.engine.begin() as conn:
            # Do not race the old deployed worker during the manual version handover.
            legacy = conn.execute(text(f'''SELECT "ID" FROM "{TABLE}" WHERE "ID"='workspace-worker'
                AND "LEASE_UNTIL">CURRENT_UTCTIMESTAMP''')).first()
            if legacy:
                return None
            reserved = conn.execute(text(f'''UPDATE "{TABLE}" SET "PARENT_ID"=:owner,
                "LEASE_UNTIL"=ADD_SECONDS(CURRENT_UTCTIMESTAMP,:seconds)
                WHERE "ID"=:slot AND ("LEASE_UNTIL" IS NULL OR "LEASE_UNTIL"<=CURRENT_UTCTIMESTAMP)'''),
                {'owner': owner, 'slot': slot, 'seconds': LEASE_SECONDS})
            if reserved.rowcount != 1:
                return None
            parent_filter = "=''" if fetch else "<>''"
            row = conn.execute(text(f'''SELECT "ID","KIND","PARENT_ID","STATUS","REVISION","UPDATED_AT","PAYLOAD"
                FROM "{TABLE}" WHERE "KIND"='job' AND "STATUS" IN ('queued','processing')
                AND "PARENT_ID"{parent_filter} AND ("LEASE_UNTIL" IS NULL OR "LEASE_UNTIL"<=CURRENT_UTCTIMESTAMP)
                ORDER BY "UPDATED_AT","ID" LIMIT 1''')).mappings().first()
            if row is None:
                conn.execute(text(f'UPDATE "{TABLE}" SET "LEASE_UNTIL"=NULL WHERE "ID"=:slot'), {'slot': slot})
                return None
            job = self.decode(row)
            claimed = conn.execute(text(f'''UPDATE "{TABLE}" SET "LEASE_UNTIL"=ADD_SECONDS(CURRENT_UTCTIMESTAMP,:seconds)
                WHERE "ID"=:id AND "REVISION"=:revision
                AND ("LEASE_UNTIL" IS NULL OR "LEASE_UNTIL"<=CURRENT_UTCTIMESTAMP)'''),
                {'id': job['id'], 'revision': job['revision'], 'seconds': LEASE_SECONDS})
            if claimed.rowcount != 1:
                # Roll back slot reservation if another slot claimed the same candidate.
                raise Conflict('Job claimed by another worker')
            job.update(status='processing', worker_owner=owner, worker_slot=slot)
            saved = self.update(job, conn=conn)
            conn.execute(text(f'UPDATE "{TABLE}" SET "PAYLOAD"=:payload WHERE "ID"=:slot'),
                         {'slot': slot, 'payload': json.dumps({'job_id': job['id']})})
            return saved

    def renew(self, owner, job_id):
        """Renew a still-valid slot and its job together; expired owners cannot revive."""
        with self.engine.begin() as conn:
            self.fence(conn, owner)
            job = self.get(job_id, conn)
            if job.get('worker_owner') != owner or job['status'] != 'processing':
                raise Conflict('Job ownership lost')
            conn.execute(text(f'''UPDATE "{TABLE}" SET "LEASE_UNTIL"=ADD_SECONDS(CURRENT_UTCTIMESTAMP,:seconds)
                WHERE "ID"=:id OR ("KIND"='lock' AND "PARENT_ID"=:owner)'''),
                {'seconds': LEASE_SECONDS, 'id': job_id, 'owner': owner})
            return True

    def release(self, owner, job_id):
        """Release only the current execution; shutdown leaves its job recoverable."""
        with self.engine.begin() as conn:
            self.fence(conn, owner, require_running=False)
            job = self.get(job_id, conn)
            if job.get('worker_owner') != owner:
                raise Conflict('Job ownership lost')
            conn.execute(text(f'''UPDATE "{TABLE}" SET "LEASE_UNTIL"=NULL
                WHERE "ID"=:id OR ("KIND"='lock' AND "PARENT_ID"=:owner)'''), {'id': job_id, 'owner': owner})

    def fence(self, conn, owner, *, require_running=True):
        """Lock the slot and verify its job before committing; terminal jobs reject late writes."""
        held = conn.execute(text(f'''SELECT "ID" FROM "{TABLE}" WHERE "KIND"='lock'
            AND "PARENT_ID"=:owner AND "LEASE_UNTIL">CURRENT_UTCTIMESTAMP FOR UPDATE'''), {'owner': owner}).first()
        if not held:
            raise Conflict('Worker lease lost; another worker will recover this job.')
        lock = self.get(held[0], conn)
        # A commit that crosses the lease deadline must serialize with takeover
        # in a different slot, not merely with renewal of this slot.
        self.lock_record(lock['job_id'], conn)
        job = self.get(lock['job_id'], conn)
        if job.get('worker_owner') != owner or (require_running and job['status'] != 'processing'):
            raise Conflict('Job ownership lost or processing already finished')
