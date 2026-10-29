"""Generate run-bound workbooks and reports independently of model and solver stages."""
import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from openpyxl import Workbook
from .analysis_exports import safe_cell, workbook_bytes, eligibility_workbook
from ...optimizer.report.markdown_to_pdf import markdown_to_pdf

XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
FILES = {
    'eligibility': ('Eligibility results', 'eligibility.xlsx', XLSX),
    'selected': ('Recommended invoices', 'selected.xlsx', XLSX),
    'excluded': ('Not recommended and pre-excluded invoices', 'excluded.xlsx', XLSX),
    'weekly-plan': ('Weekly recommendation plan', 'weekly-plan.xlsx', XLSX),
    'exposure': ('Weekly exposure', 'exposure.xlsx', XLSX),
    'report-pdf': ('Recommendation report', 'report.pdf', 'application/pdf'),
    'snapshot': ('Run snapshot and assumptions', 'snapshot.json', 'application/json'),
    'report-markdown': ('Report source', 'report.md', 'text/markdown'),
    'all-files': ('All files', 'all-files.zip', 'application/zip'),
}


def funding_outcome(row_id, candidate_ids, selected_ids):
    """Classify a source row independently of its eligibility outcome."""
    if row_id not in candidate_ids: return 'not_in_run'
    return 'selected' if row_id in selected_ids else 'not_selected'


def rows_workbook(sheets):
    """Export safe scalar values with stable row IDs and explicit amount columns."""
    book = Workbook(); book.remove(book.active)
    for title, rows in sheets.items():
        sheet = book.create_sheet(title)
        keys = list(dict.fromkeys(key for row in rows for key in row))
        sheet.append(keys or ['No rows'])
        for row in rows:
            sheet.append([safe_cell(json.dumps(row.get(key)) if isinstance(row.get(key), (dict,list)) else row.get(key)) for key in keys])
        sheet.freeze_panes = 'A2'; sheet.auto_filter.ref = sheet.dimensions
    return workbook_bytes(book)


def report_markdown(run):
    """Summarize saved outcomes, capacity and assumptions in readable business language."""
    result=run['result'];preparation=run.get('preparation') or {};settings=run['settings']
    history=preparation.get('history') or {};ack=preparation.get('acknowledgement')
    fallback=preparation.get('fallback_count',0);total=len(run['row_ids']);selected=len(result['selected'])
    rate=selected/total if total else 0
    lines=['# Invoice Recommendation Report','',f"Run {run['run_id']}",'','## Executive summary','',
        f"The optimizer recommends **{selected} of {total} invoices** ({rate:.1%}) for a **recommended amount of EUR {result['objective_amount']:,.2f}**.",'',
        f"Solver status: **{result['solver_status']}**. "+('The primary recommendation objective is proven optimal for the saved scope and assumptions.' if result['solver_status']=='OPTIMAL' else 'This result does not establish a globally optimal recommended amount.'),'',
        f"{len(result.get('pre_excluded',[]))} invoices were screened for nonpositive purchase prices before solving; {len(result.get('not_selected',[]))} candidates were not recommended.",'',
        '## Capacity review','', '| Level | Entity | Peak used EUR | Limit EUR | Peak utilization |','|---|---|---:|---:|---:|']
    peaks=[]
    for kind,by_week in result.get('exposure',{}).items():
        entities={entity for values in by_week.values() for entity in values}
        for entity in entities:
            if entity not in settings.get({'facility':'facility_limits_by_company_code','customer':'customer_limits','group':'group_limits'}[kind],{}):continue
            peak=max((values[entity] for values in by_week.values() if entity in values),key=lambda item:item['utilization_pct'])
            peaks.append((kind,str(entity),peak))
    for kind,entity,peak in sorted(peaks,key=lambda item:item[2]['utilization_pct'],reverse=True)[:15]:
        lines.append(f"| {kind.title()} | {entity.replace('|',' / ')} | {peak['used_total']:,.2f} | {peak['limit']:,.2f} | {peak['utilization_pct']:.1f}% |")
    if peaks:
        highest=max(item[2]['utilization_pct'] for item in peaks)
        lines.extend(['',f"The highest modeled utilization is {highest:.1f}%. "+('Review entities close to their caps before changing recommendation assumptions.' if highest>=90 else 'No modeled entity reaches 90% of its saved limit.'),
            '', 'This table shows the fifteen highest-utilization constraints. The exposure workbook contains every entity and week.'])
    lines.extend(['','## Planning and opening exposure','',
        f"Planning begins **{settings.get('planning_start','Unavailable')}** across **{settings.get('horizon_weeks','Unavailable')} weeks**, with Monday recommendation dates.",'',
        'Opening exposure remains consumed unless an expected repayment is supplied. A repayment releases capacity on the first Monday on or after its date; any unscheduled residual stays consumed.',''])
    repayments=settings.get('expected_repayments',[])
    if not repayments:lines.append('**No expected repayments were supplied.** Opening balances remain constant throughout the horizon.')
    else:
        lines.extend(['| Customer | Facility | Expected repayment date | Amount | Currency |','|---|---|---|---:|---|'])
        for row in repayments:lines.append(f"| {row['customer_id']} | {row['facility_id']} | {row['release_date']} | {row['amount']} | {row['currency']} |")
    rates=settings.get('currency_rates',{})
    if rates:
        lines.extend(['','Currency conversions use the saved EUR-per-unit rates:'])
        for currency,rate in rates.items():lines.append(f"- {currency}: {rate['eur_per_unit']} EUR per unit, as of {rate['as_of']}.")
    lines.extend(['','## Lifetime estimates and acknowledgement','',
        f"RPT-1 supplied valid estimates for **{total-fallback} of {total} invoices**. **{fallback} invoices use the four-week (28-day) fallback**.",''])
    if fallback:
        lines.append(f"The fallback was explicitly acknowledged at {ack['accepted_at']}." if ack else 'No fallback acknowledgement is recorded; review this inconsistency before relying on the plan.')
        for currency,amount in preparation.get('fallback_amounts_by_currency',{}).items():lines.append(f"- Affected original amount: {currency} {amount}.")
    else:lines.append('Fallback acknowledgement was not required for this run.')
    lines.extend(['',f"Historical dataset: {history.get('dataset_id') or 'Unavailable'}. Context policy: {history.get('context_policy','chronological')}. Context cutoff: {history.get('effective_as_of') or ('Not applied for fixed reference' if history.get('context_policy') == 'fixed_reference' else 'Unavailable')}. Available reference outcomes: {history.get('admitted_rows','Unavailable')}.",'',
        history.get('availability') or 'Historical outcome-availability details are included in the saved snapshot.', '',
        'Forecast feasibility does not guarantee realized exposure compliance. Longer actual lifetimes can delay expected capacity release.','',
        '## Source and outcome boundaries','',
        'Eligibility is assessed separately or approved upstream for direct recommendation uploads. Invoices outside the saved candidate scope are not optimizer rejections. Historical context informs diagnostics and lifetime estimation; it is never added to this offer as a recommendation candidate.','',
        'The accompanying workbooks and snapshot preserve every source ID, original currency, normalized amount, limit, opening balance, exchange rate, repayment, prediction and recommendation date.'])
    return '\n'.join(lines)


def render_pdf(markdown):
    """Use the existing PDF renderer; temporary render files are not application storage."""
    with TemporaryDirectory(prefix='recommendation-report-') as directory:
        target = Path(directory) / 'report.pdf'
        markdown_to_pdf(markdown, target)
        return target.read_bytes()


class WorkspaceArtifacts:
    """Persist ready/failed manifests separately from immutable completed optimization."""

    def __init__(self, runs, artifact_store, analyses=None):
        """Inject saved runs, the shared artifact store and optional source analysis store."""
        self.runs, self.store, self.analyses = runs, artifact_store, analyses

    def manifest(self, run_id):
        """Return ready/failed download entries scoped to an existing authorized run."""
        self.runs.get(run_id)
        saved = self.store.get_text_artifact(run_id, 'workspace-manifest')
        return json.loads(saved) if saved else [self.entry(run_id, key, 'pending') for key in FILES]

    def entry(self, run_id, key, status, error=None):
        """Create one allowlisted public artifact descriptor."""
        label, filename, media = FILES[key]
        if key == 'eligibility' and self.analyses and self.analyses.get(self.runs.get(run_id)['analysis_id'])['settings'].get('source_kind') == 'selection':
            label = 'Upstream eligibility status'
        return dict(artifact_id=key, label=label, filename=filename, media_type=media, status=status,
                    error=error, download_url=f'/api/workspace/runs/{run_id}/artifacts/{key}' if status=='ready' else None)

    def generate_report(self, run_id):
        """Generate missing files or retry failures exclusively from the immutable run snapshot."""
        run = self.runs.get(run_id)
        if run['status'] != 'completed': raise ValueError('Complete the recommendation before generating its files')
        result = run['result']; markdown = report_markdown(run)
        exposure = [dict(week_start=week, level=kind, entity_id=entity, **values)
                    for kind, weeks in result.get('exposure', {}).items() for week, entities in weeks.items()
                    for entity, values in entities.items()]
        generators = {
            'selected': lambda: rows_workbook({'Recommended': result['selected']}),
            'excluded': lambda: rows_workbook({'Not recommended': result.get('not_selected',[]), 'Pre-excluded':result.get('pre_excluded',[])}),
            'weekly-plan': lambda: rows_workbook({'Recommendation plan':result.get('weekly_plan',[])}),
            'exposure': lambda: rows_workbook({'Exposure':exposure}),
            'snapshot': lambda: json.dumps(run, indent=2, ensure_ascii=False).encode(),
            'report-markdown': lambda: markdown.encode(),
            'report-pdf': lambda: render_pdf(markdown),
        }
        if self.analyses:
            generators['eligibility'] = lambda: eligibility_workbook(self.analyses.all_rows(run['analysis_id']))
        manifest = []
        for key in FILES:
            if key == 'all-files': continue
            try:
                if self.store.get_binary(run_id, key) is None:
                    if key not in generators: raise ValueError('Original eligibility source is unavailable')
                    self.store.put_binary(run_id, key, generators[key]())
                manifest.append(self.entry(run_id, key, 'ready'))
            except Exception as error:
                manifest.append(self.entry(run_id, key, 'failed', str(error)))
        try:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                for item in manifest:
                    if item['status']=='ready': archive.writestr(item['filename'], self.store.get_binary(run_id,item['artifact_id']))
                archive.writestr('manifest.json', json.dumps(manifest,indent=2))
            self.store.put_binary(run_id, 'all-files', buffer.getvalue())
            archive_entry = self.entry(run_id, 'all-files', 'ready')
            if any(item['status'] != 'ready' for item in manifest):
                archive_entry.update(label='Available files (incomplete)', error='Some files failed; retry missing reports')
            manifest.append(archive_entry)
        except Exception as error:
            manifest.append(self.entry(run_id,'all-files','failed',str(error)))
        self.store.upsert_text_artifact(run_id,'workspace-manifest',json.dumps(manifest))
        return manifest

    def download(self, run_id, artifact_id):
        """Read only allowlisted, persisted files belonging to an existing run."""
        self.runs.get(run_id)
        if artifact_id not in FILES: raise LookupError('Unknown artifact')
        content = self.store.get_binary(run_id, artifact_id)
        if content is None: raise LookupError('Artifact not ready')
        return content
