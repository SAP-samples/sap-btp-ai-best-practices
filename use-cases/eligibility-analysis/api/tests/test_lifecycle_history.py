"""Verify versioned lifecycle persistence and strict event definitions."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from app.services.database.backend import BackendType, DatabaseBackend
from app.services.lifecycle.store import LifecycleStore
from app.services.lifecycle.importer import normalize_history


class LifecycleHistoryTests(unittest.TestCase):
    """Use explicit isolated SQL to test import, activation, and reopening."""

    def test_target_never_substitutes_paid_on(self):
        """A missing reconciliation-file date cannot quietly change target meaning."""
        frame = pd.DataFrame({'Company Code':['F1'],'Customer':['C1'],'Invoice Reference':['ref'],
                              'Summary File Date (UTC)':['2026-01-01'],
                              'Reconciliation File Date (UTC)':[None],
                              'Paid On (Europe, Madrid)':['2026-01-10']})
        rows, metadata = normalize_history(frame)
        self.assertEqual(rows,[])
        self.assertEqual(metadata['invalid_outcome_rows'],1)

    def test_version_activation_and_reopening(self):
        """A saved dataset survives new store instances and cannot be overwritten."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'history.db'
            backend = DatabaseBackend(BackendType.SQLITE)
            store = LifecycleStore(backend,path)
            rows = [{'invoice_key':'one','credit_start':'2026-01-01','credit_release':'2026-01-31',
                     'outcome_known_at':'2026-01-31','credit_duration_days':30}]
            store.import_dataset('v1',rows,{'source_hash':'hash'})
            store.activate('v1')
            reopened = LifecycleStore(backend,path)
            self.assertEqual(reopened.active_dataset()['dataset_id'],'v1')
            self.assertEqual(reopened.records('v1')[0]['credit_duration_days'],30)
            with self.assertRaises(ValueError): store.import_dataset('v1',rows,{'source_hash':'changed'})
            with self.assertRaises(ValueError): store.import_dataset('empty',[],{})

    def test_fixed_reference_ignores_scenario_date_but_excludes_query_identity(self):
        """Only explicit reference inference admits later outcomes; evaluation retains its cutoff."""
        with tempfile.TemporaryDirectory() as directory:
            store=LifecycleStore(DatabaseBackend(BackendType.SQLITE),Path(directory)/'reference.db')
            rows=[{'invoice_key':key,'outcome_known_at':'2026-02-01','credit_duration_days':30}
                  for key in ('example','current-invoice')]
            store.import_dataset('reference',rows,{'source_hash':'source'})
            store.activate('reference')
            chronological,_=store.context('2025-01-28',{'current-invoice'})
            self.assertTrue(chronological.empty)
            reference,metadata=store.context('2025-01-28',{'current-invoice'},context_policy='fixed_reference')
            self.assertEqual(reference.invoice_key.tolist(),['example'])
            self.assertEqual(metadata['context_policy'],'fixed_reference')
            self.assertIsNone(metadata['effective_as_of'])
            self.assertEqual(metadata['scenario_as_of'],'2025-01-28')
            self.assertEqual(metadata['excluded_rows'],1)
            with self.assertRaises(ValueError): store.context('2025-01-28',set(),context_policy='unknown')
