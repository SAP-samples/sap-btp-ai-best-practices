"""Temporal fold and source-aware metrics contracts."""
import unittest
import pandas as pd
from app.services.lifecycle.evaluation import chronological_folds, lifetime_metrics


class ChronologicalTests(unittest.TestCase):
    """Prevent future outcomes and fallback rows from overstating model validation."""

    def test_context_precedes_every_query(self):
        """Freeze context before each forward-only window and exclude its identities."""
        rows=pd.DataFrame([dict(invoice_key=str(i),credit_start=f'2026-01-{i+1:02d}',
             outcome_known_at=f'2026-01-{i+2:02d}',credit_duration_days=1) for i in range(20)])
        folds=chronological_folds(rows,2,3,5)
        self.assertEqual(len(folds),2)
        for fold in folds:
            self.assertFalse(set(fold['query_keys'])&set(fold['context_keys']))
            context=rows[rows.invoice_key.isin(fold['context_keys'])]
            self.assertTrue((pd.to_datetime(context.outcome_known_at,utc=True)<pd.Timestamp(fold['as_of'])).all())

    def test_observable_days_excludes_censored_recent_fundings(self):
        """Fundings too close to the last outcome cannot show long lifetimes and are skipped."""
        rows=pd.DataFrame([dict(invoice_key=str(i),credit_start=f'2026-01-{i+1:02d}',
             outcome_known_at=f'2026-01-{i+2:02d}',credit_duration_days=1) for i in range(20)])
        last_outcome=pd.Timestamp('2026-01-21',tz='UTC')
        folds=chronological_folds(rows,2,3,5,observable_days=5)
        self.assertEqual(len(folds),2)
        queries=rows[rows.invoice_key.isin([key for fold in folds for key in fold['query_keys']])]
        self.assertTrue((pd.to_datetime(queries.credit_start,utc=True)<=last_outcome-pd.Timedelta(days=5)).all())

    def test_coverage_excludes_fallback(self):
        """Operational fallback predictions are not model predictions."""
        metrics=lifetime_metrics([30,60],[25,28],['rpt1','fallback_default_weeks'])
        self.assertEqual(metrics['model_coverage'],.5)
        self.assertEqual(metrics['fallback_count'],1)
        self.assertEqual(metrics['model_only']['mae_days'],5)
