/** Verify saved-analysis, download, and invoice display semantics without browser state. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  analysisPage,
  savedAnalysisDeletion,
  savedAnalysisRow,
  updateSavedAnalysisSelection,
} from '../src/pages/workspace/components/saved-analysis-model.js';
import {buildDownloadGroups} from '../src/pages/workspace/components/download-model.js';
import {
  formatInvoiceDate,
  invoiceAmount,
  recommendationOutcome,
} from '../src/pages/workspace/components/invoice-display.js';

test('saved analyses paginate by tens and repeated filenames remain distinguishable', () => {
  assert.deepEqual(analysisPage(2),{limit:10,offset:10,page:2});
  const first=savedAnalysisRow({analysis_id:'aaaaaaaa-1111',filename:'offer.xlsx',analysis_date:'2026-02-01',created_at:'2026-09-08T08:00:00Z',total_invoices:28,settings:{source_kind:'selection'}});
  const second=savedAnalysisRow({analysis_id:'bbbbbbbb-2222',filename:'offer.xlsx',analysis_date:'2026-02-01',created_at:'2026-09-08T09:00:00Z',total_invoices:28,settings:{source_kind:'selection'}});
  assert.equal(first.filename,second.filename);
  assert.notEqual(first.shortId,second.shortId);
  assert.notEqual(first.uploadedAt,second.uploadedAt);
  assert.equal(first.workflow,'Recommendation');
});

test('saved upload deletion is a no-op until at least one upload is selected', () => {
  assert.equal(savedAnalysisDeletion(new Set()),null);
  const selected=updateSavedAnalysisSelection(new Set(['analysis-a']),'analysis-b',true);
  assert.deepEqual([...selected],['analysis-a','analysis-b']);
  assert.deepEqual(savedAnalysisDeletion(selected),{analysis_ids:['analysis-a','analysis-b']});
  assert.deepEqual([...updateSavedAnalysisSelection(selected,'analysis-a',false)],['analysis-b']);
});

test('download groups prioritize ZIP, describe files, and expose retry only for failures', () => {
  const model=buildDownloadGroups([
    {artifact_id:'selected',filename:'selected.xlsx',label:'Recommended invoices',status:'ready'},
    {artifact_id:'weekly-plan',filename:'weekly-plan.xlsx',label:'Weekly funding plan',status:'ready'},
    {artifact_id:'report-pdf',filename:'report.pdf',label:'Optimizer run report',status:'failed',error:'renderer'},
    {artifact_id:'snapshot',filename:'snapshot.json',label:'Run snapshot and assumptions',status:'pending'},
    {artifact_id:'all-files',filename:'all-files.zip',label:'All files',status:'ready'},
  ]);
  assert.equal(model.primary.artifactId,'all-files');
  assert.equal(model.primary.fileType,'ZIP');
  assert.equal(model.groups[0].items[0].fileType,'XLSX');
  assert.equal(model.groups[0].items.find(item=>item.artifactId==='weekly-plan').label,'Weekly recommendation plan');
  assert.equal(model.retryVisible,true);
  const failedReport=model.groups.flatMap(group=>group.items).find(item=>item.artifactId==='report-pdf');
  assert.equal(failedReport.label,'Recommendation report');
  assert.equal(failedReport.canPreview,false);
  assert.equal(buildDownloadGroups([{artifact_id:'report-pdf',filename:'report.pdf',status:'ready'}]).groups[0].items[0].canPreview,true);
  assert.equal(buildDownloadGroups([{artifact_id:'all-files',filename:'all-files.zip',status:'ready'}]).retryVisible,false);
});

test('invoice formatting and recommendation outcomes are semantic and saved-run bound', () => {
  assert.equal(invoiceAmount({amount_original:1234.5,original_currency:'EUR'}),'1,234.50 EUR');
  assert.equal(formatInvoiceDate('2026-02-01T12:00:00Z'),'Feb 1, 2026');
  const run={status:'completed',row_ids:['selected','not-selected','screened'],result:{selected:[{row_id:'selected'}],pre_excluded:[{row_id:'screened'}]}};
  assert.deepEqual(recommendationOutcome('selected',run),{label:'Recommended',design:'Positive'});
  assert.deepEqual(recommendationOutcome('screened',run),{label:'Screened out',design:'Critical'});
  assert.deepEqual(recommendationOutcome('not-selected',run),{label:'Not recommended',design:'Neutral'});
  assert.deepEqual(recommendationOutcome('outside',run),{label:'Outside recommendation',design:'Neutral'});
  assert.deepEqual(recommendationOutcome('selected',null),{label:'No recommendation',design:'Neutral'});
});
