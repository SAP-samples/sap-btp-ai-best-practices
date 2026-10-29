"""Ensure historical diagnostics never change candidate scope or compare against retries."""
from datetime import datetime, timezone
import io
import unittest
from app.services.workspace.insights import comparison_rows, metrics, build_insights
from app.services.workspace.analysis_exports import insights_workbook, insights_pdf
from openpyxl import load_workbook


class WorkspaceInsightsTests(unittest.TestCase):
    """Exercise temporal/source exclusions and distinct current/history denominators."""

    def test_current_future_and_repeated_sources_are_excluded(self):
        """Only trustworthy earlier unique events enter the history denominator."""
        at = datetime(2026, 2, 1, tzinfo=timezone.utc)
        rows = [dict(analysis_id="current", source_event_id="x", evaluated_at=at),
                dict(analysis_id="past", source_event_id="a", evaluated_at=datetime(2026,1,10,tzinfo=timezone.utc)),
                dict(analysis_id="repeat", source_event_id="a", evaluated_at=datetime(2026,1,20,tzinfo=timezone.utc)),
                dict(analysis_id="future", source_event_id="b", evaluated_at=datetime(2026,2,2,tzinfo=timezone.utc)),
                dict(analysis_id="legacy", evaluated_at=datetime(2026,1,10,tzinfo=timezone.utc))]
        actual = comparison_rows(rows, "current", at, 90)
        self.assertEqual([row['analysis_id'] for row in actual], ['repeat'])

    def test_current_source_reanalysis_never_becomes_baseline(self):
        """Current source rows are excluded even under a different analysis ID."""
        at = datetime(2026,2,1,tzinfo=timezone.utc)
        rows = [dict(analysis_id='past', source_event_id='same', evaluated_at=datetime(2026,1,1,tzinfo=timezone.utc))]
        self.assertEqual(comparison_rows(rows,'current',at,90,{'same'}), [])

    def test_ineligible_current_selection_keeps_all_outcomes_in_history(self):
        """A 100-percent rejected selection must not force a rejected-only baseline."""
        good = {'eligible':True,'invoice':{'amount_original':'100','original_currency':'EUR'},'diagnostics':{'failed_rules':[]}}
        bad = {**good,'eligible':False,'diagnostics':{'failed_rules':[{'rule_code':'R1','description':'Due date too close'}]}}
        result = build_insights([bad], [good,good,good,bad])
        self.assertEqual(result['current_metrics']['not_eligible_rate'], 100)
        self.assertEqual(result['historical_metrics']['not_eligible_rate'], 25)
        self.assertEqual(result['historical_metrics']['total'], 4)
        self.assertEqual(result['evidence'][0]['rule_code'], 'R1')

    def test_empty_baseline_has_no_invented_trend(self):
        """Insufficient history must not produce mock alerts or zero-percent certainty."""
        result = build_insights([], [])
        self.assertEqual(result['alerts'], [])
        self.assertIsNone(result['historical_metrics']['not_eligible_rate'])

    def test_exports_share_the_displayed_denominators(self):
        """Export functions consume the exact diagnostic snapshot used by the UI."""
        snapshot = build_insights([], [])
        snapshot.update(scope={'row_ids':[]}, comparison_as_of='2026-02-01', comparison_exclusions={})
        book = load_workbook(io.BytesIO(insights_workbook(snapshot)), data_only=True)
        self.assertEqual(book['Metrics']['B2'].value, 0)
        self.assertEqual(book['Metrics']['C2'].value, 0)
        self.assertTrue(insights_pdf(snapshot).startswith(b'%PDF'))
