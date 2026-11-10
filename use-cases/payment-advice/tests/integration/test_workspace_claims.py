"""Opt-in real-HANA queue checks using a unique, disposable test table only.

Run from repository root:
    RUN_HANA_QUEUE_TESTS=1 PYTHONPATH=api .venv/bin/python -m unittest discover -s tests/integration -v

Reads credentials through the existing helper. Creates and drops only this run's
PAYMENT_ADVICE_QUEUE_TEST_<uuid> table; never claims or modifies application jobs.
"""
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from unittest.mock import patch
from sqlalchemy import text
from app.email_ingestion import store as repository
from app.email_ingestion.store import Store, Conflict, ADVICE_SLOTS, FETCH_SLOT


@unittest.skipUnless(os.getenv('RUN_HANA_QUEUE_TESTS') == '1', 'Opt-in real HANA test')
class HanaClaimTests(unittest.TestCase):
    """Exercise production SQL and cross-connection fencing on SAP HANA itself."""

    @classmethod
    def setUpClass(cls):
        """Create a unique table with the production bootstrap and column contract."""
        from dotenv import load_dotenv
        from app.payment_advice.db import get_engine
        load_dotenv('api/.env')
        cls.engine = get_engine()
        cls.table = 'PAYMENT_ADVICE_QUEUE_TEST_' + uuid4().hex.upper()
        cls.patch = patch.object(repository, 'TABLE', cls.table)
        cls.patch.start()
        try:
            Store(cls.engine).ensure()
        except Exception:
            cls.patch.stop()
            cls.engine.dispose()
            raise

    @classmethod
    def tearDownClass(cls):
        """Drop only the exact table this test run created and restore module constants."""
        try:
            with cls.engine.begin() as conn:
                conn.execute(text(f'DROP TABLE "{cls.table}"'))
        finally:
            cls.patch.stop()
            cls.engine.dispose()

    def setUp(self):
        """Clear synthetic rows between cases, retaining the validated table."""
        with self.engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{self.table}"'))
        self.store = Store(self.engine)

    def job(self, fetch=False):
        """Insert a synthetic queued job in the desired lane."""
        return self.store.insert('job', {'operation': 'fetch'} if fetch else {'advice_id': 'test-advice'},
                                 parent='' if fetch else str(uuid4()), status='queued')

    def test_competing_instances_claim_one_job_once(self):
        """Concurrent slots/Store instances must not obtain the same queued job."""
        job_id = self.job()
        def claim(slot):
            """Race the production claim transaction from separate pooled connections."""
            try:
                return Store(self.engine).claim_next(slot, str(uuid4()))
            except Conflict:
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            claimed = [r for r in pool.map(claim, ADVICE_SLOTS) if r]
        self.assertEqual([j['id'] for j in claimed], [job_id])

    def test_four_slots_and_independent_fetch(self):
        """Shared global slots bound advice concurrency without starving Gmail."""
        for _ in range(5):
            self.job()
        fetch_id = self.job(fetch=True)
        claims = [self.store.claim_next(slot, str(uuid4())) for slot in ADVICE_SLOTS]
        self.assertEqual(len({j['id'] for j in claims}), 4)
        other = Store(self.engine)
        for slot in ADVICE_SLOTS:
            self.assertIsNone(other.claim_next(slot, str(uuid4())))
        self.assertEqual(other.claim_next(FETCH_SLOT, 'fetch-owner', fetch=True)['id'], fetch_id)
        self.assertEqual(len(self.store.list('job', status='queued')), 1)

    def test_expired_job_recovered_old_owner_cannot_commit(self):
        """An expired execution cannot renew, release, or commit after takeover."""
        job_id = self.job()
        self.store.claim_next(ADVICE_SLOTS[0], 'old')
        self.assertTrue(self.store.renew('old', job_id))
        with self.engine.begin() as conn:
            conn.execute(text(f'''UPDATE "{self.table}" SET "LEASE_UNTIL"=ADD_SECONDS(CURRENT_UTCTIMESTAMP,-1)
                WHERE "ID"=:job OR "ID"=:slot'''), {'job': job_id, 'slot': ADVICE_SLOTS[0]})
        recovered = Store(self.engine).claim_next(ADVICE_SLOTS[1], 'new')
        self.assertEqual(recovered['id'], job_id)
        for action in (lambda: self.store.renew('old', job_id), lambda: self.store.release('old', job_id)):
            with self.assertRaises(Conflict):
                action()
        with self.engine.begin() as conn, self.assertRaises(Conflict):
            self.store.fence(conn, 'old')
        self.assertTrue(self.store.renew('new', job_id))
        self.assertEqual(len(self.store.list('job')), 1)

    def test_terminal_job_rejects_late_writes_and_releases_slot(self):
        """A timed-out provider thread cannot advance a checkpoint after job failure."""
        job_id = self.job(fetch=True)
        job = self.store.claim_next(FETCH_SLOT, 'owner', fetch=True)
        job['status'] = 'failed'
        self.store.update(job)
        with self.engine.begin() as conn, self.assertRaises(Conflict):
            self.store.fence(conn, 'owner')
        self.store.release('owner', job_id)
        new_id = self.job(fetch=True)
        self.assertEqual(self.store.claim_next(FETCH_SLOT, 'next', fetch=True)['id'], new_id)

    def test_shutdown_release_recovers_unfinished_job(self):
        """Graceful shutdown relinquishes its exact job without duplicating records."""
        job_id = self.job()
        self.store.claim_next(ADVICE_SLOTS[0], 'old')
        self.store.release('old', job_id)
        self.assertEqual(self.store.claim_next(ADVICE_SLOTS[0], 'new')['id'], job_id)
        self.assertEqual(len(self.store.list('job')), 1)

    def test_legacy_live_worker_blocks_upgrade_claims(self):
        """Do not concurrently process jobs owned by the pre-upgrade serial worker."""
        self.job()
        self.store.insert('lock', {}, record_id='workspace-worker')
        with self.engine.begin() as conn:
            conn.execute(text(f'''UPDATE "{self.table}" SET "LEASE_UNTIL"=ADD_SECONDS(CURRENT_UTCTIMESTAMP,120)
                WHERE "ID"='workspace-worker' '''))
        self.assertIsNone(self.store.claim_next(ADVICE_SLOTS[0], 'new'))

    def test_parallel_siblings_finish_with_atomic_job_and_email_states(self):
        """Real HANA completion transactions must preserve both results and the final rollup."""
        import asyncio
        from types import SimpleNamespace
        from app.email_ingestion.worker import process_advice
        email_id = self.store.insert('email', {'warnings': []}, status='processing')
        advice_ids = [self.store.insert('advice', {'corrections': [], 'client_key': None},
                                       parent=email_id, status='queued') for _ in range(2)]
        for advice_id in advice_ids:
            self.store.insert('job', {'advice_id': advice_id}, parent=advice_id, status='queued')
        jobs = [self.store.claim_next(ADVICE_SLOTS[i], f'owner-{i}') for i in range(2)]
        state = SimpleNamespace(dox=object(), settings=object(), runtime=None)

        async def run():
            """Run production orchestration/transactions, replacing only remote extraction."""
            await asyncio.gather(*(process_advice(state, self.store, job, f'owner-{i}') for i, job in enumerate(jobs)))

        with patch('app.email_ingestion.worker.extract_document', return_value=(
                {'canonical': {'header': {}, 'line_items': []}}, [])):
            asyncio.run(run())
        self.assertTrue(all(self.store.get(j['id'])['status'] == 'completed' for j in jobs))
        self.assertTrue(all(self.store.get(a)['status'] == 'needs_review' for a in advice_ids))
        self.assertEqual(self.store.get(email_id)['status'], 'needs_review')
        self.assertEqual(len(self.store.list('processing_result')), 2)

    def test_canonical_guard_serializes_only_its_scope(self):
        """Shared schema setup locks exclude another instance until setup exits."""
        import threading
        entered = threading.Event()
        other = Store(self.engine)
        # Initialize outside the held guard so the test observes row locking,
        # rather than a duplicate-key insertion waiting for the first transaction.
        with other.canonical_schema_guard():
            pass

        def setup():
            """Enter the production row guard from another connection."""
            with other.canonical_schema_guard():
                entered.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.store.canonical_schema_guard():
                task = pool.submit(setup)
                self.assertFalse(entered.wait(.2), 'Concurrent schema setup entered the held guard')
            task.result(timeout=10)
        self.assertTrue(entered.is_set())
