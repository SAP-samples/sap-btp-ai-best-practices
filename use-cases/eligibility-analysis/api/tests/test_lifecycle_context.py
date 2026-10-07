"""Temporal lifecycle context must exclude outcomes unavailable at prediction time."""
import unittest
import pandas as pd
from app.services.lifecycle.context import admissible_history, invoice_key


class LifecycleContextTests(unittest.TestCase):
    """Catch target leakage and ambiguous identity matching."""

    def test_late_outcome_same_invoice_and_invalid_duration_are_excluded(self):
        """Earlier funding alone does not make an outcome available for prediction."""
        rows = pd.DataFrame({'invoice_key':['past','late','query','bad','future'],
                             'outcome_known_at':['2026-01-10','2026-02-10','2026-01-12','2026-01-01','2026-01-01'],
                             'credit_duration_days':[30,45,20,float('inf'),-2]})
        result = admissible_history(rows,pd.Timestamp('2026-02-01',tz='UTC'),{'query'})
        self.assertEqual(result['invoice_key'].tolist(),['past'])

    def test_identity_uses_business_keys_not_reference_alone(self):
        """The same reference at another customer is a distinct invoice identity."""
        row = {'Company Code':'F1','Customer':'C1','Invoice Reference':'ref'}
        self.assertNotEqual(invoice_key(row),invoice_key({**row,'Customer':'C2'}))
        self.assertIsNone(invoice_key({'Invoice Reference':'ref'}))
