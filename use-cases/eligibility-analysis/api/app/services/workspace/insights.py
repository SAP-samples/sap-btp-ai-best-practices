"""Source-grounded eligibility diagnostics with distinct current and historical scopes."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from ...models.workspace import WorkspaceValidationError
from .analysis_store import filter_rows
from .schema import decode_json, transaction


def comparison_rows(history, analysis_id, as_of, lookback_days, current_source_ids=None):
    """Return the latest earlier observation per source, excluding current-source revisions."""
    start = as_of - timedelta(days=lookback_days)
    seen = {}
    for row in history:
        source_id, evaluated = row.get('source_event_id'), row.get('evaluated_at')
        if not source_id or evaluated is None or row.get('analysis_id') == analysis_id:
            continue
        if source_id in (current_source_ids or set()) or not start <= evaluated < as_of:
            continue
        if source_id not in seen or seen[source_id]['evaluated_at'] < evaluated:
            seen[source_id] = row
    return list(seen.values())


def metrics(rows):
    """Count outcomes and sum original money separately by currency, without inventing FX."""
    rejected = sum(not row['eligible'] for row in rows)
    totals, rejected_totals = defaultdict(Decimal), defaultdict(Decimal)
    missing_amount = 0
    for row in rows:
        invoice = row['invoice']
        raw = invoice.get('amount_original')
        if raw is None:
            missing_amount += 1
            continue
        amount = Decimal(str(raw))
        if not amount.is_finite():
            missing_amount += 1
            continue
        currency = invoice['original_currency']
        totals[currency] += amount
        if not row['eligible']:
            rejected_totals[currency] += amount
    return dict(total=len(rows), eligible=len(rows)-rejected, not_eligible=rejected,
                not_eligible_rate=round(100*rejected/len(rows), 2) if rows else None,
                amounts={key:str(value) for key,value in totals.items()},
                not_eligible_amounts={key:str(value) for key,value in rejected_totals.items()},
                missing_amount_count=missing_amount)


def build_insights(current, history):
    """Build deterministic rule evidence; unsupported payment-block/credit-note claims are absent."""
    current_metrics, historical_metrics = metrics(current), metrics(history)
    current_rules, past_rules = Counter(), Counter()
    for population, counts in ((current,current_rules),(history,past_rules)):
        for row in population:
            counts.update({failure['rule_code'] for failure in row['diagnostics']['failed_rules']})
    evidence = [dict(rule_code=code, current_count=current_rules[code], historical_count=past_rules[code])
                for code in sorted(current_rules.keys() | past_rules.keys())]
    alerts = []
    if len(history) >= 3 and current:
        for item in evidence:
            current_rate = 100 * item['current_count'] / len(current)
            historical_rate = 100 * item['historical_count'] / len(history)
            if current_rate > historical_rate and item['current_count']:
                alerts.append(dict(severity='warning', rule_code=item['rule_code'],
                                   title=f"{item['rule_code']}: higher rejection share in current scope",
                                   description=f"{item['current_count']} of {len(current)} current invoices ({current_rate:.1f}%) versus "
                                   f"{item['historical_count']} of {len(history)} earlier invoices ({historical_rate:.1f}%).",
                                   current_rate=round(current_rate,2), historical_rate=round(historical_rate,2)))
    dates = defaultdict(list)
    for row in history:
        if row.get('analysis_date'):
            dates[row['analysis_date']].append(row)
    trend = [{'date':key, **metrics(value)} for key,value in sorted(dates.items())]
    return dict(current_metrics=current_metrics, historical_metrics=historical_metrics,
                alerts=alerts, evidence=evidence, trend=trend,
                history_status='available' if len(history) >= 3 else 'insufficient_history')


class WorkspaceInsights:
    """Read the saved current population and admissible prior HANA observations."""

    def __init__(self, analyses):
        """Use the canonical analysis store; do not duplicate customer-log writes."""
        self.analyses = analyses

    def analyze(self, analysis_id, row_ids=None, filters=None, lookback_days=90):
        """Freeze a current scope and compare it against earlier unique source outcomes."""
        if not 1 <= lookback_days <= 3650:
            raise WorkspaceValidationError('Lookback must be between 1 and 3650 days','lookback_days')
        filters = filters or {}
        analysis = self.analyses.get(analysis_id)
        if analysis['settings'].get('source_kind') == 'selection':
            raise WorkspaceValidationError('Eligibility was approved upstream; no local rule diagnostics exist')
        all_current = self.analyses.all_rows(analysis_id)
        current = filter_rows(all_current, filters)
        if row_ids is not None:
            valid = {row['row_id'] for row in all_current}
            if missing := set(row_ids) - valid:
                raise WorkspaceValidationError('Unknown source rows','row_ids',sorted(missing))
            # Explicit selection owns scope, including rows retained across table filters.
            current = filter_rows([row for row in all_current if row['row_id'] in set(row_ids)],
                                  {key:value for key,value in filters.items() if key not in ('status','search')})
        as_of = datetime.fromisoformat(analysis['created_at'])
        start = (as_of-timedelta(days=lookback_days)).isoformat()
        history = []
        with transaction(self.analyses.backend, self.analyses.db_path) as cursor:
            cursor.execute('SELECT metadata FROM RECEIVABLES_ANALYSES '
                           'WHERE created_at >= ? AND created_at < ? AND analysis_id <> ?',
                           (start,as_of.isoformat(),analysis_id))
            previous = [decode_json(record[0]) for record in cursor.fetchall()]
        # Read each immutable analysis once instead of repeating its large metadata
        # beside every invoice in a joined LOB result set.
        for metadata in previous:
            if metadata['settings'].get('source_kind') == 'selection' or metadata['analysis_date'] > analysis['analysis_date']:
                continue
            for row in self.analyses.all_rows(metadata['analysis_id']):
                row.update(analysis_id=metadata['analysis_id'], evaluated_at=datetime.fromisoformat(metadata['created_at']),
                           analysis_date=metadata['analysis_date'])
                history.append(row)
        unique = comparison_rows(history, analysis_id, as_of, lookback_days,
                                 {row['source_event_id'] for row in all_current})
        comparable = filter_rows(unique, {key:value for key,value in filters.items() if key not in ('status','search')})
        snapshot = build_insights(current, comparable)
        snapshot.update(scope=dict(analysis_id=analysis_id,row_ids=[row['row_id'] for row in current],filters=filters,lookback_days=lookback_days),
                        comparison_as_of=as_of.isoformat(), comparison_exclusions={
                            'repeated_or_current_source_rows':len(history)-len(unique),
                            'legacy_provenance':'Legacy logs without trustworthy source/batch identity are excluded'},
                        comparison_period={'from':start,'until':as_of.isoformat()},
                        current_rows=current)
        return snapshot
