/** Run with: node --test ui/tests/inbox-concurrency.test.mjs (no browser dependencies). */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

/** Minimal DOM double to exercise the actual inbox controller's async boundaries. */
class Element {
  constructor(tagName = '') { this.tagName = tagName.toUpperCase(); this.children = []; this.listeners = {}; this.dataset = {}; this.isConnected = true; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute() {}
  addEventListener(name, callback) { this.listeners[name] = callback; }
}
/** Create externally resolvable requests to force deterministic response ordering. */
function deferred() { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise,resolve}; }
/** Load production functions, stopping before event registration and background timers. */
function controller(request) {
  const elements = new Map();
  const root = new Element();
  root.querySelector = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const source = moduleSource('dom.js') + moduleSource('advice-table.js') + readFileSync(new URL('../src/pages/home/home.js', import.meta.url),'utf8')
    .replace(/^import .*;$/gm,'').replace('export default function','function')
    .replace('  get("manual-open").onclick', `  return {refresh,renderRows,renderChat,workspace,root,conversations,
      setPage: emails => {page=emails},
      setDetails: data => {details=data},
      state: () => ({details,selectedEmail}),
      select: id => {selectedEmail=id;details=null}};\n  get("manual-open").onclick`);
  return vm.runInNewContext(source + '\ninitHomePage()', {
    request,
    renderSafeMarkdown:(host, source) => {
      const strong = new Element('strong');
      strong.textContent = source.replaceAll('**','');
      host.replaceChildren(strong);
    },
    document:{getElementById:()=>root,createElement:tag=>new Element(tag)},
    URL:{revokeObjectURL:()=>{}}, window:{confirm:()=>true}, console, renderS4Panel:()=>{},
  });
}
/** Read a sibling page module as plain script: imports dropped, exports made local. */
function moduleSource(name) {
  return readFileSync(new URL('../src/pages/home/' + name, import.meta.url),'utf8')
    .replace(/^import .*;$/gm,'').replace(/^export /gm,'') + '\n';
}
/** Render the S/4 tab for one advice and return the button labels with their disabled state. */
function s4Buttons(advice) {
  const render = vm.runInNewContext(moduleSource('dom.js') + moduleSource('s4-panel.js') + 'renderS4Panel',
    {document:{createElement:tag=>new Element(tag)}, window:{confirm:()=>true}, JSON});
  const content = new Element('div');
  render(content, advice, {post: async () => {}, onError: () => {}});
  const bar = content.children.find(child => child.className === 'advice-actions');
  return bar.children.map(b => [b.textContent, b.disabled]);
}
/** Locate the customer toggle in a rendered inbox row. */
function toggle(view, index) { return view.root.querySelector('#inbox-rows').children[index].children[0].children[0]; }

test('late detail A cannot overwrite email B selection', async () => {
  const a = deferred(), b = deferred();
  const view = controller(path => path.endsWith('/a') ? a.promise : b.promise);
  view.setPage([{id:'a',received_at:0},{id:'b',received_at:0}]);
  await view.renderRows();
  const first = toggle(view,0).listeners.click();
  const second = toggle(view,1).listeners.click();
  b.resolve({email:{id:'b'},advices:[]}); await second;
  a.resolve({email:{id:'a'},advices:[]}); await first;
  assert.equal(view.state().selectedEmail,'b');
  assert.equal(view.state().details.email.id,'b');
});

test('forced refresh during polling is queued rather than discarded', async () => {
  const pending = deferred(); let lists = 0;
  const view = controller(path => {
    if (path.endsWith('/status')) return Promise.resolve({message:'Unconfigured',counts:{}});
    lists += 1;
    return lists === 1 ? pending.promise : Promise.resolve({emails:[],counts:{}});
  });
  const poll = view.refresh();
  const mutation = view.refresh(true);
  pending.resolve({emails:[],counts:{}});
  await Promise.all([poll,mutation]);
  assert.equal(lists,2);
});

test('background row refresh preserves an unsent advice chat draft', () => {
  const view = controller(() => Promise.resolve([]));
  const first = new Element(), second = new Element();
  const advice = {id:'draft-advice',filename:'test.txt',status:'ready'};
  view.renderChat(first, advice);
  const input = first.children[2];
  input.value = 'Change invoice I1';
  input.oninput?.();
  view.renderChat(second, advice);
  assert.equal(second.children[2].value, 'Change invoice I1');
});

/** Collect inert rendered text from the DOM double. */
function renderedText(element) {
  return [element.textContent || '', ...element.children.map(renderedText)].join(' ');
}

/** Find all rendered elements matching one structural predicate. */
function findAll(element, predicate) {
  return [element, ...element.children.flatMap(child => findAll(child, predicate))].filter(predicate);
}

test('interpretation renders as a key-value list instead of JSON text', async () => {
  const view = controller(() => Promise.resolve([]));
  view.setDetails({email:{}, attachments:[], advices:[{
    id:'a', filename:'test.txt', status:'ready',
    result:{header:{interpretation:{client_key:'customer',unresolved_count:2}},line_items:[]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const listItems = findAll(host, element => element.tagName === 'LI').map(element => renderedText(element).trim());
  assert.deepEqual(listItems, ['client key customer', 'unresolved count 2']);
  assert.doesNotMatch(renderedText(host), /\{"client_key"/);
});

test('flags use bullets only when populated and internal line-item fields are not displayed', async () => {
  const view = controller(() => Promise.resolve([]));
  view.setDetails({email:{}, attachments:[], advices:[{
    id:'a', filename:'test.txt', status:'ready',
    result:{header:{},line_items:[
      {invoice_reference:'EMPTY',flags:[],residual_items:['SHOULD-NOT-RENDER'],row_id:'ROW-ID-ONE'},
      {invoice_reference:'FLAGGED',flags:['customer-unconfirmed','manual-review'],residual_items:['HIDDEN'],row_id:'ROW-ID-TWO'},
    ]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const table = findAll(host, element => element.className?.startsWith('advice-lines'))[0];
  const headings = findAll(table, element => element.tagName === 'TH').map(renderedText);
  const rows = findAll(table, element => element.tagName === 'TR').slice(1);
  const flagsIndex = headings.indexOf('Flags');
  assert.equal(headings.includes('Linked invoices'), false);
  assert.equal(headings.includes('Row ID'), false);
  assert.equal(renderedText(rows[0].children[flagsIndex]).trim(), '');
  assert.deepEqual(findAll(rows[1].children[flagsIndex], element => element.tagName === 'LI').map(renderedText), ['customer-unconfirmed','manual-review']);
  assert.doesNotMatch(renderedText(table), /SHOULD-NOT-RENDER|HIDDEN|ROW-ID-ONE|ROW-ID-TWO/);
});

test('delete action removes the complete selected email through the inbox API', async () => {
  const calls = [];
  const view = controller(async (path, method, body) => {
    calls.push({path,method,body});
    if (method === 'DELETE') return {deleted:'email-1'};
    if (path.includes('/inbox?')) return {emails:[],counts:{}};
    if (path.endsWith('/status')) return {message:'Ready',counts:{}};
    return [];
  });
  view.select('email-1');
  view.setDetails({email:{id:'email-1',revision:4},attachments:[],advices:[{
    id:'advice-1',filename:'test.txt',status:'needs_review',revision:2,result:{header:{},line_items:[]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const deleteButton = findAll(host, element => element.textContent === 'Delete')[0];

  assert.equal(deleteButton.className, 'danger');
  await deleteButton.listeners.click();
  const deletion = calls.find(call => call.method === 'DELETE');
  assert.equal(deletion.path, '/api/email-ingestion/inbox/email-1');
  assert.equal(deletion.body.revision, 4);
  assert.equal(view.state().selectedEmail, null);
  assert.equal(view.state().details, null);
});

test('advice chat renders assistant Markdown while keeping user Markdown literal', () => {
  const view = controller(() => Promise.resolve([]));
  view.conversations.set('chat-advice',[
    {role:'user',content:'**literal user text**'},
    {role:'assistant',content:'**formatted reply**'},
  ]);
  const host = new Element();
  view.renderChat(host,{id:'chat-advice',filename:'test.txt',status:'ready'});
  const messages = findAll(host, element => element.className?.startsWith('advice-chat-message'));
  assert.equal(messages[0].children.length,0);
  assert.equal(messages[0].textContent,'**literal user text**');
  assert.equal(messages[1].children[0].tagName,'STRONG');
  assert.equal(renderedText(messages[1]).trim(),'formatted reply');
});

test('extracted rows are visible before interpretation finishes without review controls', async () => {
  const view = controller(() => Promise.resolve([]));
  view.setDetails({email:{subject:'Test'}, attachments:[], advices:[{
    id:'a', filename:'test.txt', status:'processing', stage:'interpreting',
    original_extraction:{header:{payment_reference:'PAY-1'},line_items:[{invoice_reference:'INV-1',net_amount:100}]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const text = renderedText(host);
  assert.match(text, /INV-1/);
  assert.match(text, /PAY-1/);
  assert.match(text, /Interpreting deductions/);
  assert.doesNotMatch(text, /Mark reviewed/);
});

test('reprocessing keeps corrected values visible instead of replacing them with raw extraction', async () => {
  const view = controller(() => Promise.resolve([]));
  view.setDetails({email:{}, attachments:[], advices:[{
    id:'a', filename:'test.txt', status:'processing', stage:'interpreting',
    result:{header:{},line_items:[{invoice_reference:'CORRECTED'}]},
    original_extraction:{header:{},line_items:[{invoice_reference:'RAW'}]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const text = renderedText(host);
  assert.match(text, /CORRECTED/);
  assert.doesNotMatch(text, /RAW/);
  assert.match(text, /Previous saved result/);
});

test('S/4 posting is only offered for reviewed advices with a ready check', () => {
  const ready = {ready:true, issues:[], lines:[], derived:{company_code:'CA01', customer:'0010053628', currency:'CAD', customers:['0010053628']}, payload:{}};
  const base = {result:{header:{}, line_items:[]}, revision:3};
  assert.deepEqual(s4Buttons({...base, status:'ready'}), [['Check against S/4', false], ['Post to S/4', true]]);
  assert.deepEqual(s4Buttons({...base, status:'ready', s4:ready}), [['Check against S/4', false], ['Post to S/4', true]]);
  assert.deepEqual(s4Buttons({...base, status:'reviewed', s4:ready}), [['Check against S/4', false], ['Post to S/4', false]]);
  assert.deepEqual(s4Buttons({...base, status:'reviewed', s4:{...ready, ready:false}}), [['Check against S/4', false], ['Post to S/4', true]]);
  assert.deepEqual(s4Buttons({...base, status:'posted', s4:{...ready, posted:{key:{PaymentAdvice:'0425'}, read_back:{}}}}), []);
});

test('table cells are editable and Accept Changes sends one typed batch for the displayed revision', async () => {
  const calls = [];
  const view = controller(async (path, method, body) => {
    calls.push({path,method,body});
    if (path.includes('/inbox?')) return {emails:[],counts:{}};
    if (path.endsWith('/status')) return {message:'Ready',counts:{}};
    return {};
  });
  view.setDetails({email:{}, attachments:[], advices:[{
    id:'adv', filename:'test.txt', status:'reviewed', revision:7, corrections:[],
    result:{header:{},line_items:[{row_id:'r1',invoice_reference:'INV-1',net_amount:-12500,reason_code:'323',flags:['a']}]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const accept = findAll(host, element => element.textContent === 'Accept Changes')[0];
  const undo = findAll(host, element => element.textContent === 'Undo Changes')[0];
  assert.equal(accept.disabled, true);
  assert.equal(undo.disabled, true);
  const row = findAll(host, element => element.dataset?.rowId === 'r1')[0];
  const [net, reason, flags] = [6, 8, 9].map(index => row.children[index]);
  assert.equal(net.contentEditable, 'plaintext-only');
  net.innerText = ' -12000.5 '; net.oninput();
  reason.innerText = '321'; reason.oninput();
  flags.innerText = ''; flags.oninput();
  assert.equal(accept.disabled, false);
  assert.match(net.className, /changed/);
  await accept.listeners.click();
  const batch = calls.find(call => call.path.endsWith('/edits'));
  assert.equal(batch.path, '/api/payment-advice/advices/adv/edits');
  assert.deepEqual(JSON.parse(JSON.stringify(batch.body)), {revision:7, edits:[
    {target:'r1',field:'net_amount',value:-12000.5},
    {target:'r1',field:'reason_code',value:'321'},
    {target:'r1',field:'flags',value:[]}]});
});

test('a non-numeric amount is rejected locally and Undo Changes reverts accepted corrections', async () => {
  const calls = [];
  const view = controller(async (path, method, body) => {
    calls.push({path,method,body});
    if (path.includes('/inbox?')) return {emails:[],counts:{}};
    if (path.endsWith('/status')) return {message:'Ready',counts:{}};
    return {};
  });
  view.setDetails({email:{}, attachments:[], advices:[{
    id:'adv2', filename:'test.txt', status:'needs_review', revision:3, corrections:[{target:'r1'}],
    result:{header:{},line_items:[{row_id:'r1',invoice_reference:'INV-1',gross_amount:10}]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const row = findAll(host, element => element.dataset?.rowId === 'r1')[0];
  row.children[4].innerText = 'abc'; row.children[4].oninput();
  await findAll(host, element => element.textContent === 'Accept Changes')[0].listeners.click();
  assert.equal(calls.some(call => call.path.endsWith('/edits')), false);
  await findAll(host, element => element.textContent === 'Undo Changes')[0].listeners.click();
  const revert = calls.find(call => call.path.endsWith('/revert'));
  assert.equal(revert.path, '/api/payment-advice/advices/adv2/revert');
  assert.equal(revert.body.revision, 3);
});

test('posted advices render a read-only table without edit buttons', async () => {
  const view = controller(() => Promise.resolve([]));
  view.setDetails({email:{}, attachments:[], advices:[{
    id:'adv3', filename:'test.txt', status:'posted', revision:1,
    result:{header:{},line_items:[{row_id:'r1',invoice_reference:'INV-1'}]}
  }]});
  const host = new Element();
  await view.workspace(host);
  const row = findAll(host, element => element.dataset?.rowId === 'r1')[0];
  assert.equal(row.children[0].contentEditable, undefined);
  assert.equal(findAll(host, element => element.textContent === 'Accept Changes').length, 0);
});
