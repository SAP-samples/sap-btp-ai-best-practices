/** Verify population boundaries independently of DOM rendering and backend state. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  candidateScope,
  createWorkspaceState,
  requestedWorkspaceView,
  resultsAreReady,
  savedRunLabel,
  switchAnalysis,
  updatePageSelection,
} from '../src/pages/workspace/state.js';
import {analysisForm} from '../src/services/workspace-form.js';

test('all eligible never serializes display filters or selected rows', () => {
  assert.deepEqual(candidateScope('all_eligible', new Set(['a']), new Set(['a','c'])),
    {mode:'all_eligible', row_ids:[]});
});
test('explicit selected scope excludes ineligible rows', () => {
  assert.deepEqual(candidateScope('selected', new Set(['a','b']), new Set(['a'])),
    {mode:'selected', row_ids:['a']});
  assert.throws(() => candidateScope('selected', new Set(['b']), new Set(['a'])));
});
test('selection across pages preserves off-page selections and removes deselected visible rows', () => {
  const state = createWorkspaceState();
  state.selectedRowIds = new Set(['a','c']);
  updatePageSelection(state, ['a','b'], new Set(['b']));
  assert.deepEqual([...state.selectedRowIds].sort(), ['b','c']);
  state.filters = {status:'not_eligible'};
  assert.deepEqual([...state.selectedRowIds].sort(), ['b','c']);
});
test('switching analysis clears selection and run identity', () => {
  const state = createWorkspaceState();
  state.selectedRowIds.add('old');
  state.runId = 'old-run';
  state.revision = 5;
  switchAnalysis(state, 'new');
  assert.equal(state.analysisId, 'new');
  assert.equal(state.runId, null);
  assert.equal(state.revision, null);
  assert.equal(state.selectedRowIds.size, 0);
});
test('upload form preserves selected date and request identity on retry', () => {
  const file = new Blob(['uploaded bytes']);
  const form = analysisForm(file, {analysisDate:'2026-02-02', settings:{nddt:6}, requestKey:'retry-key'});
  assert.equal(form.get('analysis_date'), '2026-02-02');
  assert.equal(form.get('request_key'), 'retry-key');
  assert.deepEqual(JSON.parse(form.get('settings')), {nddt:6});
});

test('saved runs lead with a readable status and scope', () => {
  const label=savedRunLabel({status:'awaiting_lifetime_acknowledgement',row_ids:['a','b'],run_id:'abcdefgh-1234'});
  assert.equal(label,'Review lifetime assumptions · 2 invoices · abcdefgh');
  assert.ok(!label.includes('awaiting_lifetime_acknowledgement'));
});

test('completed results are announced without switching views automatically', () => {
  const run={status:'completed',result:{selected:[]}};
  assert.equal(resultsAreReady(run),true);
  assert.equal(requestedWorkspaceView('',run),'invoices');
  assert.equal(requestedWorkspaceView('?run=run-1&view=results',run),'results');
  assert.equal(requestedWorkspaceView('?run=run-1&view=results',{status:'optimizing'}),'invoices');
});
