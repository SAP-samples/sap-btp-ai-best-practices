/**
 * Tests of the "S/4 posting" tab: stale checks, unconfirmed posts, choosing the S/4 document of a line,
 * the "How it was matched" column and the match overview.
 *
 * Run with: node --test ui/tests/s4-panel.test.mjs   (no browser dependencies)
 */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

/** Minimal DOM double: just enough of Element for the panel's inert rendering. */
class Element {
  constructor(tagName = '') { this.tagName = tagName.toUpperCase(); this.children = []; this.listeners = {}; this.attributes = {}; }
  append(...children) { this.children.push(...children); }
  setAttribute(name, text) { this.attributes[name] = text; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
}

/** Read a page module as plain script: imports dropped, exports made local. */
function moduleSource(name) {
  return readFileSync(new URL('../src/pages/home/' + name, import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '').replace(/^export /gm, '') + '\n';
}

/**
 * Render the S/4 tab for one advice with a recording `post` and a confirm that always says yes.
 * @returns {{content: Element, posts: object[], confirms: string[]}}
 */
function render(advice) {
  const posts = [], confirms = [];
  const renderS4Panel = vm.runInNewContext(moduleSource('dom.js') + moduleSource('s4-choice.js') + moduleSource('s4-panel.js') + 'renderS4Panel',
    {document: {createElement: tag => new Element(tag)}, window: {confirm: text => { confirms.push(text); return true; }}, JSON});
  const content = new Element('div');
  renderS4Panel(content, advice, {post: async (path, body) => posts.push(JSON.parse(JSON.stringify({path, body}))),
    onError: error => { throw error; }});
  return {content, posts, confirms};
}

/** All rendered elements matching a predicate. */
function findAll(element, predicate) {
  return [element, ...element.children.flatMap(child => findAll(child, predicate))].filter(predicate);
}
/** Concatenated text of an element and its children. */
function text(element) { return [element.textContent || '', ...element.children.map(text)].join(' ').replace(/\s+/g, ' ').trim(); }
/** The button with exactly this label. */
function buttonNamed(content, label) { return findAll(content, e => e.tagName === 'BUTTON' && e.textContent === label); }
/** Status line text. */
function status(content) { return content.children[0].textContent; }
/** Header cells and body rows of the lines table (the one with the "#" column). */
function linesTable(content) {
  const table = findAll(content, e => e.tagName === 'TABLE' && findAll(e, th => th.tagName === 'TH' && th.textContent === '#').length)[0];
  const headings = findAll(table, e => e.tagName === 'TH').map(e => e.textContent);
  const rows = findAll(table, e => e.tagName === 'TBODY')[0].children.map(tr => tr.children.map(td => td.textContent));
  return {headings, rows, cell: (row, heading) => rows[row][headings.indexOf(heading)]};
}

const derived = {company_code: '1000', customer: '0010000001', currency: 'EUR', customers: ['0010000001']};
const item = {company_code: '1000', accounting_document: '1800000101', fiscal_year: '2026', accounting_document_item: '001',
  customer: '0010000001', amount: '1250.00', currency: 'EUR'};
/** A matched check line; `extra` overrides any field. */
function matchedLine(extra = {}) {
  return {index: 0, row_id: 'r0', reference: 'INV-1', kind: 'matched', status: 'open', matched_by: 'DocumentReferenceID',
    net: '1250.00', s4_item: item, ...extra};
}
/** An advice with one stored check; `s4` overrides check fields. */
function advice(s4 = {}, extra = {}) {
  return {status: 'reviewed', revision: 6, result: {header: {}, line_items: [{row_id: 'r0'}]},
    s4: {ready: true, issues: [], lines: [], derived, payload: {}, ...s4}, ...extra};
}
/** Two documents S/4 found for one reference. */
const candidates = [
  {pick: '1000/1800000101/2026/001', company_code: '1000', accounting_document: '1800000101', fiscal_year: '2026',
   accounting_document_item: '001', customer: '0010000001', amount: '1250.00', open_amount: '1250.00', currency: 'EUR',
   state: 'open', matched_by: 'DocumentReferenceID', variant: null, special_gl: ''},
  {pick: '1000/1800000102/2025/001', company_code: '1000', accounting_document: '1800000102', fiscal_year: '2025',
   accounting_document_item: '001', customer: '0010000001', amount: '1250.00', open_amount: '0.00', currency: 'EUR',
   state: 'cleared', matched_by: 'AccountingDocument', variant: 'INV1', special_gl: ''},
];

test('a stale check disables Post to S/4 and asks for a new check', () => {
  const {content} = render(advice({stale: true}));
  assert.equal(status(content), 'The advice changed after this check. Check against S/4 again.');
  assert.match(content.children[0].className, /s4-blocking/);
  assert.equal(buttonNamed(content, 'Post to S/4')[0].disabled, true);
  assert.equal(buttonNamed(content, 'Check against S/4')[0].disabled, false);
});

test('an unconfirmed post explains the situation and Release for changes posts /s4/release', async () => {
  const {content, posts, confirms} = render(advice({post_outcome: 'unknown', post_error: 'timeout after 60 s',
    posting_started_at: '2026-03-01T10:00:00+00:00'}));
  assert.equal(status(content), 'S/4 did not confirm the last post (timeout after 60 s). Post again to check S/4 and finish, ' +
    'or release the advice to change it.');
  const bar = content.children.find(child => child.className === 'advice-actions');
  assert.deepEqual(bar.children.map(b => b.textContent), ['Check against S/4', 'Post to S/4', 'Release for changes']);
  await buttonNamed(content, 'Release for changes')[0].listeners.click();
  assert.match(confirms[0], /only works when S\/4 holds no payment advice for this payment reference/);
  assert.deepEqual(posts, [{path: '/s4/release', body: {revision: 6}}]);
  // A plain rejection keeps the existing message and offers no release.
  const rejected = render(advice({post_error: 'Company code 1000 is locked'})).content;
  assert.equal(status(rejected), 'S/4 rejected the last post: Company code 1000 is locked');
  assert.equal(buttonNamed(rejected, 'Release for changes').length, 0);
});

test('an unfinished post can be posted again even when the new check blocks', async () => {
  // S/4 may hold the advice and may have cleared the invoice with it, so the new check is not ready.
  const {content, posts, confirms} = render(advice({posting_started_at: '2026-03-01T10:00:00+00:00', ready: false, payload: null}));
  const send = buttonNamed(content, 'Post to S/4')[0];
  assert.equal(send.disabled, false);
  await send.listeners.click();
  assert.match(confirms[0], /unfinished post/);
  assert.deepEqual(posts, [{path: '/s4/post', body: {revision: 6}}]);
});

test('a post that was started and never finished can be released too', () => {
  const {content} = render(advice({posting_started_at: '2026-03-01T10:00:00+00:00'}));
  assert.equal(status(content), 'A post of this advice was started and did not finish. Post again to check S/4 and finish, ' +
    'or release the advice to change it.');
  assert.equal(buttonNamed(content, 'Release for changes').length, 1);
});

test('a line with two S/4 documents offers a choice that saves s4_document as one batch', async () => {
  const line = matchedLine({kind: 'unresolved', status: 'ambiguous', s4_item: null, matched_by: null, candidates});
  const {content, posts, confirms} = render(advice({ready: false, lines: [line],
    issues: [{severity: 'blocking', code: 'ambiguous_reference', line: 0, message: 'Line 1 (INV-1) matches 2 S/4 documents.'}]}));
  const block = findAll(content, e => e.className === 's4-choice')[0];
  assert.equal(block.children[0].textContent, 'Line 1 (INV-1): choose the S/4 document');
  const rows = findAll(block, e => e.tagName === 'TBODY')[0].children.map(tr => tr.children.slice(0, 6).map(td => td.textContent));
  assert.deepEqual(rows, [
    ['1000/1800000101/2026', '0010000001', '1250.00 EUR', '1250.00 EUR', 'Open', 'Reference'],
    ['1000/1800000102/2025', '0010000001', '1250.00 EUR', '0.00 EUR', 'Already cleared', 'Document number as INV1']]);
  const use = buttonNamed(block, 'Use this document');
  assert.equal(use.length, 2);
  await use[0].listeners.click();
  assert.match(confirms[0], /1000\/1800000101\/2026/);
  assert.deepEqual(posts, [{path: '/edits', body: {revision: 6, reason: 'S/4 document chosen by the reviewer',
    edits: [{target: 'r0', field: 's4_document', value: '1000/1800000101/2026/001'}]}}]);
});

test('choosing a document for a residual row also clears s4_treatment in the same batch', async () => {
  const line = matchedLine({kind: 'residual', status: 'residual', candidates});
  const {content, posts} = render(advice({lines: [line]}, {result: {header: {}, line_items: [{row_id: 'r0', s4_treatment: 'residual'}]}}));
  await buttonNamed(content, 'Use this document')[1].listeners.click();
  assert.deepEqual(posts[0].body.edits, [{target: 'r0', field: 's4_document', value: '1000/1800000102/2025/001'},
    {target: 'r0', field: 's4_treatment', value: null}]);
});

test('items of one document are told apart by item number, and no choice is offered once posted', () => {
  const sameDocument = candidates.map((c, i) => ({...c, accounting_document: '1800000101', fiscal_year: '2026',
    accounting_document_item: '00' + (i + 1), state: 'open'}));
  const {content} = render(advice({lines: [matchedLine({candidates: sameDocument})]}));
  const block = findAll(content, e => e.className === 's4-choice')[0];
  assert.deepEqual(findAll(block, e => e.tagName === 'TBODY')[0].children.map(tr => tr.children[0].textContent),
    ['1000/1800000101/2026 item 001', '1000/1800000101/2026 item 002']);
  const posted = render(advice({lines: [matchedLine({candidates})], posted: {key: {PaymentAdvice: '0425'}, read_back: {}}},
    {status: 'posted'})).content;
  assert.equal(findAll(posted, e => e.className === 's4-choice').length, 0);
});

test('Undo choice posts an /edit that clears s4_document', async () => {
  const line = matchedLine({resolution: {method: 'reviewer', steps: [{rule: 'reviewer_pick', kept: 1, dropped: 1}], warning: null}});
  const {content, posts} = render(advice({lines: [line]}));
  const note = findAll(content, e => e.tagName === 'LI' && e.className === 's4-info')[0];
  assert.equal(note.textContent, 'Line 1 (INV-1) uses S/4 document 1000/1800000101/2026, chosen by the reviewer.');
  await buttonNamed(note, 'Undo choice')[0].listeners.click();
  assert.deepEqual(posts, [{path: '/edit', body: {revision: 6, target: 'r0', field: 's4_document', value: null,
    reason: 'Undo S/4 document choice'}}]);
});

test('the lines table shows the open amount and how each line was matched', () => {
  const lines = [
    matchedLine({open_amount: '250.00', resolution: {method: 'unique', steps: [], warning: null}}),
    matchedLine({index: 1, row_id: 'r1', reference: 'CD-7', reference_source: 'customer_document_reference',
      resolution: {method: 'auto', warning: 'Two documents had the same amount.', steps: [
        {rule: 'company_code', kept: 2, dropped: 1}, {rule: 'open', kept: 1, dropped: 1}]}}),
    matchedLine({index: 2, row_id: 'r2', resolution: {method: 'reviewer', steps: [], warning: null},
      s4_items: [item, {...item, accounting_document_item: '002'}]}),
    {index: 3, row_id: 'r3', reference: 'D-1', kind: 'deduction', status: 'not_found', net: '-50.00',
     related_document: '1000/1800000101/2026'},
    matchedLine({index: 4, row_id: 'r4', kind: 'unresolved', status: 'sign_mismatch'}),
    matchedLine({index: 5, row_id: 'r5', kind: 'unresolved', status: 'special_gl'}),
  ];
  const table = linesTable(render(advice({lines})).content);
  assert.equal(table.headings.indexOf('Open in S/4'), table.headings.indexOf('S/4 amount') + 1);
  assert.equal(table.headings.at(-1), 'How it was matched');
  assert.equal(table.cell(0, 'Open in S/4'), '250.00 EUR');
  assert.equal(table.cell(1, 'Open in S/4'), '');
  assert.equal(table.cell(0, 'How it was matched'), 'Only document with this reference');
  assert.equal(table.cell(1, 'How it was matched'), 'Chosen among 3 by company code, still open - please confirm');
  assert.equal(table.cell(2, 'How it was matched'), 'Chosen by the reviewer');
  assert.equal(table.cell(1, 'Reference'), 'CD-7 (customer document no.)');
  assert.equal(table.cell(2, 'S/4 document'), '1000/1800000101/2026 item 001, 1000/1800000101/2026 item 002 (DocumentReferenceID)');
  assert.equal(table.cell(3, 'S/4 document'), 'deducts from 1000/1800000101/2026');
  assert.equal(table.cell(4, 'S/4 status'), 'Invoice/credit mismatch');
  assert.equal(table.cell(5, 'S/4 status'), 'Special G/L item');
});

test('the match overview lists how the lines were matched', () => {
  const summary = {lines: 10, matched: 9, matched_by: {DocumentReferenceID: 7, AccountingDocument: 2, BillingDocument: 0},
    resolution: {unique: 6, auto: 2, reviewer: 1}, deductions: 1, residuals: 0, unresolved: 1, blocking: 1, warnings: 0};
  const {content} = render(advice({summary}));
  const entries = findAll(content, e => e.className === 'structured-list')[1].children.map(text);
  assert.deepEqual(entries, ['lines matched to S/4 9 of 10', 'matched by reference 7, document number 2',
    'matched without a choice 6', 'chosen by rules 2', 'chosen by the reviewer 1', 'deductions 1',
    'posted as open item 0', 'not resolved 1']);
});

test('an advice that pays several company codes shows them per line and in the overview', () => {
  const elsewhere = {...item, company_code: '2000', accounting_document: '1800000202'};
  const summary = {lines: 2, matched: 2, matched_by: {DocumentReferenceID: 2}, resolution: {unique: 2, auto: 0, reviewer: 0},
    company_codes: {1000: 1, 2000: 1}, deductions: 0, residuals: 0, unresolved: 0, blocking: 0, warnings: 1};
  const {content} = render(advice({summary, lines: [matchedLine(), matchedLine({index: 1, row_id: 'r1', s4_item: elsewhere})]}));
  const table = linesTable(content);
  assert.equal(table.cell(0, 'S/4 document'), '1000/1800000101/2026 (DocumentReferenceID)');
  assert.equal(table.cell(1, 'S/4 document'), '2000/1800000202/2026 (DocumentReferenceID)');
  const entries = findAll(content, e => e.className === 'structured-list')[1].children.map(text);
  assert.ok(entries.includes('matched lines per company code 1000: 1, 2000: 1'));
});

test('a matched line outside the advice company code asks the reviewer to confirm its document', () => {
  const elsewhere = {...candidates[0], pick: '2000/1800000202/2026/001', company_code: '2000', accounting_document: '1800000202'};
  const line = matchedLine({s4_item: {...item, company_code: '2000', accounting_document: '1800000202'}, candidates: [elsewhere]});
  const {content} = render(advice({ready: false, lines: [line],
    issues: [{severity: 'blocking', code: 'cross_company_unconfirmed', line: 0, message: 'Line 1 (INV-1) was matched outside company code 1000.'}]}));
  const block = findAll(content, e => e.className === 's4-choice')[0];
  assert.equal(block.children[0].textContent, 'Line 1 (INV-1): confirm the S/4 document');
  assert.equal(buttonNamed(block, 'Use this document').length, 1);
});

test('a check stored before these fields existed renders as before', () => {
  const old = {ready: true, issues: [], derived, payload: {},
    lines: [{index: 0, row_id: 'r0', reference: 'INV-1', kind: 'matched', status: 'open', matched_by: 'DocumentReferenceID',
             net: '1250.00', s4_item: item}]};
  const {content} = render({status: 'reviewed', revision: 2, result: {header: {}, line_items: []}, s4: old});
  assert.equal(status(content), 'Ready to post. Posting re-checks against live S/4 data first.');
  assert.equal(buttonNamed(content, 'Post to S/4')[0].disabled, false);
  assert.equal(buttonNamed(content, 'Release for changes').length, 0);
  assert.equal(findAll(content, e => e.className === 'structured-list').length, 1);
  assert.equal(findAll(content, e => e.className === 's4-choice').length, 0);
  const table = linesTable(content);
  assert.equal(table.cell(0, 'S/4 document'), '1000/1800000101/2026 (DocumentReferenceID)');
  assert.equal(table.cell(0, 'Open in S/4'), '');
  assert.equal(table.cell(0, 'How it was matched'), '');
});
