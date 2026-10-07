"""Offline scheduling tests with real intake/extraction orchestration.

Run: PYTHONPATH=api .venv/bin/python -m unittest tests.unit.test_parallel_worker -v
Only remote extraction and persistence transport are replaced; real HANA claims
are additionally checked by tests/integration/test_workspace_claims.py.
"""
import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.email_ingestion import worker
from app.email_ingestion.intake import intake
from app.email_ingestion.store import Conflict
from tests.unit.workspace_support import MemoryStore


class SchedulingStore(MemoryStore):
    """Thread-safe claim double for observing the production async scheduler."""

    def __init__(self):
        """Track active job and slot ownership independently of task execution."""
        super().__init__()
        self.owners, self.slots = {}, {}
        self.lock = threading.RLock()

    def claim_next(self, slot, owner, *, fetch=False):
        """Claim an unowned queued job in the requested lane atomically."""
        with self.lock:
            if slot in self.slots:
                return None
            jobs = list(reversed(self.list('job', status='queued')))
            for job in jobs:
                if (job.get('operation') == 'fetch') != fetch:
                    continue
                job['status'] = 'processing'
                job = self.update(job)
                self.owners[owner] = job['id']
                self.slots[slot] = owner
                return job
            return None

    def renew(self, owner, job_id):
        """Report whether the job still belongs to this execution."""
        return self.owners.get(owner) == job_id

    def release(self, owner, job_id):
        """Drop only this execution's slot so another job can start."""
        with self.lock:
            self.owners.pop(owner, None)
            self.slots = {k: v for k, v in self.slots.items() if v != owner}


async def eventually(predicate, seconds=1):
    """Wait briefly for an observed state; return False on test deadline."""
    async def wait():
        """Poll the observable result without blocking the event loop."""
        while not predicate():
            await asyncio.sleep(.005)
    try:
        await asyncio.wait_for(wait(), seconds)
        return True
    except TimeoutError:
        return False


class ParallelWorkerTests(unittest.IsolatedAsyncioTestCase):
    """Catch serial awaiting, slot starvation and missing partial-result persistence."""

    def setUp(self):
        """Use deterministic unknown customers and a short scheduler polling period."""
        self.store = SchedulingStore()
        self.state = SimpleNamespace(workspace=self.store, dox=object(), settings=SimpleNamespace(service_key_data={}), runtime=None)
        for target, value in [('app.email_ingestion.intake.list_customers', []),
                              ('app.email_ingestion.intake.find_customers', []),
                              ('app.email_ingestion.worker.ServiceKey.from_json', object()),
                              ('app.email_ingestion.worker.SapDoxClient', object())]:
            p = patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(worker, 'POLL_SECONDS', .01, create=True)
        p.start()
        self.addCleanup(p.stop)

    def add_email(self, text):
        """Save a real manual intake record and return its advice ID."""
        email_id = intake(self.store, {'sender': 'test@example.invalid', 'body': text})
        return self.store.list('advice', parent=email_id)[0]['id']

    async def test_four_advices_overlap_and_fetch_is_independent(self):
        """Four blocked extractions must not prevent fetch or accept a fifth extraction."""
        for value in range(4):
            self.add_email(str(value))
        fetch_id = self.store.insert('job', {'operation': 'fetch'}, status='queued')
        started, gate = set(), threading.Event()

        def extract(path, *args, **kwargs):
            """Simulate slow Document AI while retaining real worker orchestration."""
            started.add(path.read_text())
            gate.wait(5)
            return {'canonical': {'header': {}, 'line_items': []}}

        def fetch(store, job, owner):
            """Complete a remote fetch without Gmail credentials."""
            return job

        with patch.object(worker, 'extract_to_canonical', side_effect=extract), \
             patch.object(worker, 'synchronize', side_effect=fetch), \
             patch.object(worker, 'binding', return_value={}, create=True):
            task = asyncio.create_task(worker.run_worker(self.state))
            try:
                self.assertTrue(await eventually(lambda: len(started) == 4), 'Advice jobs are still serialized')
                self.assertTrue(await eventually(lambda: self.store.get(fetch_id)['status'] == 'completed'), 'Fetch is blocked by extraction')
                last = self.add_email('later')
                await asyncio.sleep(.05)
                self.assertEqual(len(started), 4, 'Global advice concurrency limit exceeded')
                self.assertEqual(self.store.get(last)['status'], 'queued')
                gate.set()
                self.assertTrue(await eventually(lambda: self.store.get(last)['status'] == 'needs_review'))
                self.assertEqual(len(self.store.list('advice')), 5)
            finally:
                gate.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_failure_does_not_stop_later_jobs(self):
        """A failed extraction must leave another advice able to finish."""
        bad, good = self.add_email('bad'), self.add_email('good')

        def extract(path, *args, **kwargs):
            """Fail only one remote document."""
            if path.read_text() == 'bad':
                raise RuntimeError('private provider detail')
            return {'canonical': {'header': {}, 'line_items': []}}

        with patch.object(worker, 'extract_to_canonical', side_effect=extract), \
             patch.object(worker, 'binding', side_effect=worker.GmailError('not configured'), create=True):
            task = asyncio.create_task(worker.run_worker(self.state))
            try:
                self.assertTrue(await eventually(lambda: self.store.get(good)['status'] == 'needs_review'))
                self.assertEqual(self.store.get(bad)['status'], 'failed')
                self.assertNotIn('private provider detail', self.store.get(bad)['error'])
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_extraction_visible_while_interpretation_waits(self):
        """Persist the canonical result before the downstream agent returns."""
        advice_id = self.add_email('document')
        advice = self.store.get(advice_id)
        advice['client_key'] = 'customer'
        self.store.update(advice)
        job = self.store.list('job', parent=advice_id)[0]
        entered, release = asyncio.Event(), asyncio.Event()
        canonical = {'header': {'payment_reference': 'P1'}, 'line_items': [{'invoice_reference': 'I1', 'net_amount': -10}]}

        async def interpret(*args, **kwargs):
            """Pause after extraction has finished but before interpretation is saved."""
            import json
            entered.set()
            await release.wait()
            yield json.dumps({'type': 'result', 'enriched': canonical})

        with patch.object(worker, 'extract_to_canonical', return_value={'canonical': canonical}), \
             patch.object(worker, 'get_playbook', return_value=None), \
             patch.object(worker, 'interpret_events', side_effect=interpret):
            task = asyncio.create_task(worker.process_advice(self.state, self.store, job, 'worker'))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                saved = self.store.get(advice_id)
                self.assertEqual(saved.get('stage'), 'interpreting')
                self.assertEqual(saved.get('original_extraction'), canonical)
                self.assertEqual(saved['status'], 'processing')
            finally:
                release.set()
                await task

    async def test_restart_reuses_saved_extraction(self):
        """Restarting the same interrupted job must not upload the document again."""
        advice_id = self.add_email('document')
        advice = self.store.get(advice_id)
        advice['client_key'] = 'customer'
        self.store.update(advice)
        job = self.store.list('job', parent=advice_id)[0]
        entered = asyncio.Event()
        canonical = {'header': {}, 'line_items': []}
        raw = {'schema_fields': [], 'header': {'document_no': 'P1'}, 'line_items': []}

        async def slow(*args, **kwargs):
            """Interrupt interpretation only after its extraction checkpoint exists."""
            entered.set()
            await asyncio.Event().wait()
            yield ''

        with patch.object(worker, 'extract_to_canonical', return_value={'canonical': canonical, 'raw_extraction': raw}) as extract, \
             patch.object(worker, 'get_playbook', return_value=None):
            with patch.object(worker, 'interpret_events', side_effect=slow):
                task = asyncio.create_task(worker.process_advice(self.state, self.store, job, 'first'))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await worker.process_advice(self.state, self.store, job, 'second')
            self.assertEqual(extract.call_count, 1, 'Recovery uploaded the already extracted document again')
        self.assertEqual(self.store.get(advice_id)['raw_extraction'], raw, 'Recovery lost the raw extraction')
        self.assertEqual(self.store.list('processing_result', parent=advice_id)[0]['raw_extraction'], raw)
        self.assertEqual(self.store.get(advice_id)['status'], 'needs_review')
        self.assertEqual(self.store.get(job['id'])['status'], 'completed', 'Advice became reviewable before its job finished')
        self.assertEqual(len(self.store.list('advice')), 1)

    async def test_job_deadline_does_not_hold_its_slot_forever(self):
        """A hung provider is failed safely and its slot is released without stalling the loop."""
        advice_id = self.add_email('hung')
        job = self.store.claim_next('workspace-advice-0', 'owner')
        cancelled = asyncio.Event()

        async def hang(*args):
            """Represent an async provider that never responds before cancellation."""
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(worker, 'process_advice', side_effect=hang), \
             patch.object(worker, 'JOB_TIMEOUT_SECONDS', .02):
            await worker.execute_job(self.state, job, 'owner')
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.store.get(advice_id)['status'], 'failed')
        self.assertIn('timed out', self.store.get(advice_id)['error'])
        self.assertFalse(self.store.slots)

    async def test_lease_loss_cancels_only_its_execution(self):
        """A lost lease cancels work, leaving recovery rather than marking a false failure."""
        self.add_email('lost')
        job = self.store.claim_next('workspace-advice-0', 'owner')
        cancelled = asyncio.Event()

        async def hang(*args):
            """Observe cancellation without producing a successful result."""
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(worker, 'process_advice', side_effect=hang), \
             patch.object(worker, 'HEARTBEAT_SECONDS', .01), \
             patch.object(self.store, 'renew', side_effect=Conflict('lost')):
            await worker.execute_job(self.state, job, 'owner')
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.store.get(job['id'])['status'], 'processing')

    async def test_slow_gmail_fetch_does_not_block_advice(self):
        """The fetch lane may be waiting on Gmail while a later manual email finishes."""
        self.store.insert('job', {'operation': 'fetch'}, status='queued')
        gate, entered = threading.Event(), threading.Event()

        def slow_fetch(store, job, owner):
            """Hold only the remote Gmail transport."""
            entered.set()
            gate.wait(5)
            return job

        with patch.object(worker, 'binding', return_value={}), \
             patch.object(worker, 'synchronize', side_effect=slow_fetch), \
             patch.object(worker, 'extract_to_canonical', return_value={'canonical': {'header': {}, 'line_items': []}}):
            task = asyncio.create_task(worker.run_worker(self.state))
            try:
                self.assertTrue(await eventually(entered.is_set))
                advice_id = self.add_email('later')
                self.assertTrue(await eventually(lambda: self.store.get(advice_id)['status'] == 'needs_review'))
                self.assertFalse(gate.is_set())
            finally:
                gate.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_cancelling_extraction_keeps_files_until_thread_finishes(self):
        """Shutdown must not delete a temporary document while its provider thread reads it."""
        advice_id = self.add_email('still reading')
        job = self.store.list('job', parent=advice_id)[0]
        entered, release, paths = threading.Event(), threading.Event(), []

        def extract(path, *args, **kwargs):
            """Hold a real temporary file open across cancellation of the async caller."""
            paths.append(path)
            entered.set()
            release.wait(5)
            self.assertEqual(path.read_text(), 'still reading')
            return {'canonical': {'header': {}, 'line_items': []}}

        with patch.object(worker, 'extract_to_canonical', side_effect=extract):
            task = asyncio.create_task(worker.process_advice(self.state, self.store, job, 'owner'))
            try:
                self.assertTrue(await eventually(entered.is_set))
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.assertTrue(paths[0].exists())
                self.assertNotIn('original_extraction', self.store.get(advice_id))
            finally:
                release.set()
                self.assertTrue(await eventually(lambda: paths and not paths[0].exists()))

    async def test_parallel_interpretations_keep_document_context_separate(self):
        """The real interpretation service must not swap bound documents across awaits."""
        from app.deduction_agent.document_context import get_current_document
        # Import outside the timed overlap assertion (SDK imports are expensive).
        from app.email_ingestion import assistant
        self.state.runtime = object()
        ids = [self.add_email('FIRST'), self.add_email('SECOND')]
        for advice_id in ids:
            advice = self.store.get(advice_id)
            advice['client_key'] = 'customer'
            self.store.update(advice)
        entered, both, observed = [], asyncio.Event(), []

        def extract(path, *args, **kwargs):
            """Return distinguishable canonical rows for the two remote documents."""
            return {'canonical': {'header': {}, 'line_items': [
                {'invoice_reference': path.read_text(), 'gross_amount': -10}]}}

        def runtime(*args):
            """Pause the model after it reads its server-bound source document."""
            async def invoke(**kwargs):
                """Verify the same ContextVar value remains after another invocation runs."""
                before = get_current_document()['line_items'][0]['invoice_reference']
                entered.append(before)
                if len(entered) == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), 1)
                after = get_current_document()['line_items'][0]['invoice_reference']
                observed.append((before, after))
                return SimpleNamespace(output_parsed={'interpretations': []}, messages=[])
            return SimpleNamespace(ainvoke=invoke)

        with patch.object(worker, 'extract_to_canonical', side_effect=extract), \
             patch.object(worker, 'get_playbook', return_value=None), \
             patch.object(assistant, 'scoped_runtime', side_effect=runtime):
            await asyncio.gather(*(worker.process_advice(self.state, self.store,
                self.store.list('job', parent=advice_id)[0], str(index)) for index, advice_id in enumerate(ids)))
        self.assertEqual(set(observed), {('FIRST', 'FIRST'), ('SECOND', 'SECOND')})
        self.assertEqual([self.store.get(i)['result']['line_items'][0]['invoice_reference'] for i in ids], ['FIRST', 'SECOND'])
