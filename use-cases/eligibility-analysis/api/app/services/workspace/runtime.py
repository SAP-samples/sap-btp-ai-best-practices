"""Adapt saved workspace snapshots to the existing RPT-1 and CP-SAT engines."""
import json
import math
import pandas as pd
from .candidates import canonical_candidates
from .settings import validate_settings, cents
from ...optimizer.model.limits import ResolvedLimits
from ...optimizer.opt.optimizer_multi_week import optimize_multi_week, MultiWeekOptimizerSettings
from ...optimizer.model.reconciliation_calendar import planned_funding_dates
from ..lifecycle.prediction import estimate_from_history


def records(frame):
    """Serialize a dataframe into strict JSON records with ISO dates and null missing values."""
    return json.loads(frame.to_json(orient='records', date_format='iso'))


class WorkspaceRuntime:
    """Keep historical context separate from exact saved candidate and settings inputs."""

    def __init__(self, service, history):
        """Inject the authoritative workspace and lifecycle stores."""
        self.service, self.history = service, history

    def inputs(self, run):
        """Validate and reconstruct exact source candidates without reading local workbooks."""
        wanted = set(run['row_ids'])
        rows = [row for row in self.service.analyses.all_rows(run['analysis_id']) if row['row_id'] in wanted]
        if len(rows) != len(wanted) or any(not row['eligible'] for row in rows):
            raise ValueError('Saved candidate scope is incomplete or no longer eligible')
        preview = validate_settings(run['settings'], rows)
        if preview['readiness_issues']:
            raise ValueError('Saved credit settings are not ready')
        analysis = self.service.analyses.get(run['analysis_id'])
        return canonical_candidates(analysis, rows, run['settings']), preview, analysis

    def estimate(self, run):
        """Estimate current lifetimes using the active fixed reference, independent of scenario dates.

        Lifetimes are measured from each invoice's earliest planned funding Wednesday: the
        first planning week on or after its offer date (never before planning_start). If the
        optimizer schedules the invoice later, the duration still starts at that week, which
        overstates exposure rather than understating it.
        """
        candidates, _, analysis = self.inputs(run)
        candidates['Planned Funding Date'] = planned_funding_dates(
            candidates['Offer File Date (UTC)'], run['settings']['planning_start'])
        try:
            output, metadata = estimate_from_history(self.history, candidates, analysis['analysis_date'], context_policy='fixed_reference')
        except Exception as error:
            output, metadata = candidates, {'status': 'estimation_failed', 'error_type': type(error).__name__}
        predictions = []
        for row in records(output):
            source = 'rpt1' if row.get('expected_lifetime_source') == 'RPT-1' else 'fallback_default_weeks'
            predictions.append(dict(row_id=row['row_id'], source=source,
                expected_lifetime_days=row.get('expected_lifetime_days'),
                confidence=row.get('expected_lifetime_confidence'),
                reason=None if source == 'rpt1' else metadata.get('status', 'Model prediction unavailable'),
                original_amount=row['Original Amount'], original_currency=row['Original Currency'],
                normalized_amount=row['Normalized Amount'], invoice_reference=row['Invoice Reference']))
        return predictions, metadata

    def solve(self, run):
        """Solve once using saved predictions, explicit limits and projected opening balances."""
        candidates, preview, _ = self.inputs(run)
        predictions = {row['row_id']: row for row in run['preparation']['predictions']}
        candidates['expected_lifetime_days'] = candidates.row_id.map(lambda key: predictions[key]['expected_lifetime_days'])
        candidates['expected_lifetime_weeks'] = candidates.expected_lifetime_days.map(lambda days: max(1, math.ceil(days / 7)))
        candidates['expected_lifetime_source'] = candidates.row_id.map(lambda key: predictions[key]['source'])
        # Credit notes cannot manufacture capacity. Record them separately from solver nonselection.
        excluded = candidates[candidates['Purchase Price'] <= 0].copy()
        excluded['exclusion_reason'] = 'Nonpositive purchase price'
        eligible = candidates[candidates['Purchase Price'] > 0].copy()
        settings = run['settings']
        limits = ResolvedLimits(
            {key: cents(value) for key, value in settings['facility_limits_by_company_code'].items()},
            {key: cents(value) for key, value in settings['customer_limits'].items()},
            {key: cents(value) for key, value in settings['group_limits'].items()}, settings['customer_to_group'])
        weeks = [pd.Timestamp(row['week_start']) for row in preview['weekly_opening_preview']]
        opening = {pd.Timestamp(row['week_start']): {kind: {key: float(value) for key, value in values.items()}
                   for kind, values in row['opening'].items()} for row in preview['weekly_opening_preview']}
        result = optimize_multi_week(eligible, limits, weeks, opening, MultiWeekOptimizerSettings(
            horizon_weeks=len(weeks), attempt_cap=len(weeks), calendar_version='monday-v1'))
        return dict(solver_status=result.status, objective_amount=result.objective_amount, currency='EUR',
            candidates=records(candidates), selected=records(result.selected_df),
            not_selected=records(result.not_selected_df), pre_excluded=records(excluded),
            weekly_plan=records(result.weekly_plan_df), week_starts=[week.date().isoformat() for week in weeks],
            exposure={'facility': result.facility_weekly_usage, 'customer': result.customer_weekly_usage,
                      'group': result.group_weekly_usage})
