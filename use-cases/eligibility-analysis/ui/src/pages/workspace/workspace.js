/** Compose the saved invoice workspace using UI5 controls and the actual API. */
import '@ui5/webcomponents/dist/Select.js';
import '@ui5/webcomponents/dist/Option.js';
import '@ui5/webcomponents/dist/Input.js';
import '@ui5/webcomponents/dist/Dialog.js';
import '@ui5/webcomponents/dist/MessageStrip.js';
import {
  createWorkspaceState,
  requestedWorkspaceView,
  resultsAreReady,
  switchAnalysis,
  updatePageSelection,
} from './state.js';
import {mountInvoiceTable, invoiceAmount} from './components/invoice-table.js';
import * as api from '../../services/workspace-api.js';
import {mountAnalysisPanel} from './components/analysis-panel.js';
import {setAssistantWorkspaceContext} from '../../modules/chatbot.js';
import {mountFunding} from './funding-controller.js';
import {mountInsightsDialog} from './components/insights-dialog.js';
import {mountSavedAnalyses} from './components/saved-analyses.js';

/** Mount a route-scoped workspace and release all requests/listeners on destroy. */
export function mountWorkspace(root, routeContext={}) {
  const state = createWorkspaceState();
  const signal = state.abortController.signal;
  const find = selector => root.querySelector(selector);
  let analysis = null;
  let pageRequest = null;
  let analysisRequest = 0;
  const funding = mountFunding(root, {state,showMessage,onRun:publishRun});
  const upload = mountAnalysisPanel(find('#ws-panels'), {onAnalyzed:saved => openAnalysis(saved.analysis_id)});
  const insights = mountInsightsDialog(find('#ws-panels'));
  const savedAnalyses = mountSavedAnalyses(find('#ws-saved-dialog'),{
    onOpen:openAnalysis,
    onDeleted:ids=>{if(ids.includes(state.analysisId))location.assign('/workspace');},
  });
  const table = mountInvoiceTable(find('#ws-table'), {
    onSelection: (pageIds, selected) => { updatePageSelection(state, pageIds, selected); renderSelection(); },
    onInspect: inspectInvoice,
    onPage: direction => { state.offset = Math.max(0, state.offset + direction * 50); refreshPage(); },
  });

  /** Show a safe, actionable status without exposing backend objects as HTML. */
  function showMessage(message, design='Information') {
    const strip = find('#ws-message');
    strip.textContent = message;
    strip.design = design;
    strip.hidden = !message;
  }

  /** Restore a run in the URL and keep the table, workflow and assistant aligned. */
  function publishRun(run,{preserveLocation=false}={}) {
    table.setRun(run);
    state.runId=run?.run_id??null;
    state.revision=run?.revision??null;
    const url=new URL(location.href);
    if(run)url.searchParams.set('run',run.run_id);
    else if(!preserveLocation){url.searchParams.delete('run');url.searchParams.delete('view');}
    history.replaceState(history.state,'',url);
    find('#ws-results-ready').hidden=!resultsAreReady(run);
    find('#ws-view-results').disabled=!resultsAreReady(run);
    activateView(run?requestedWorkspaceView(location.search,run):'invoices',{replaceUrl:false});
    publishAssistantScope();
  }

  /** Switch primary workspace content while preserving only an explicit results choice. */
  function activateView(requested,{replaceUrl=true}={}) {
    const view=requested==='results'&&resultsAreReady(funding.getRun())?'results':'invoices';
    state.view=view;
    find('#ws-invoices-view').hidden=view!=='invoices';
    find('#ws-results-view').hidden=view!=='results';
    find('#ws-view-invoices').design=view==='invoices'?'Emphasized':'Transparent';
    find('#ws-view-results').design=view==='results'?'Emphasized':'Transparent';
    find('#ws-view-invoices').setAttribute('aria-pressed',String(view==='invoices'));
    find('#ws-view-results').setAttribute('aria-pressed',String(view==='results'));
    if(replaceUrl){const url=new URL(location.href);if(view==='results')url.searchParams.set('view','results');else url.searchParams.delete('view');history.replaceState(history.state,'',url);}
    if(view==='results')requestAnimationFrame(()=>funding.resizeResults());
  }

  /** Send source references separately from user prompt text. */
  function publishAssistantScope() {setAssistantWorkspaceContext({analysis_id:state.analysisId,run_id:state.runId,revision:state.revision,row_ids:[...state.selectedRowIds],filters:state.filters});}

  /** Render selected and excluded counts, independent of the visible page. */
  function renderSelection() {
    publishAssistantScope();
    const total = state.selectedRowIds.size;
    const eligible = [...state.selectedRowIds].filter(id => state.eligibleRowIds.has(id)).length;
    find('#ws-selection').textContent = total ? `${total} selected · ${eligible} eligible · ${total - eligible} ineligible excluded from the recommendation` :
      'All eligible invoices are the default recommendation scope';
  }

  /** Fetch the latest visible page; abort superseded requests to avoid stale rendering. */
  async function refreshPage() {
    if (!state.analysisId) return;
    pageRequest?.abort();
    const request = new AbortController();
    pageRequest = request;
    try {
      const page = await api.listInvoices(state.analysisId, {...state.filters, limit:50, offset:state.offset}, request.signal);
      if (!signal.aborted && pageRequest === request) table.render(page, state.selectedRowIds, state.offset);
    } catch (error) { if (error.name !== 'AbortError' && !signal.aborted) showMessage(error.message, 'Negative'); }
  }

  /** Open immutable metadata and reconstruct a clean selection context. */
  async function openAnalysis(id) {
    try {
      const request = ++analysisRequest;
      const saved = await api.getAnalysis(id, signal);
      if (signal.aborted || request !== analysisRequest) return;
      analysis = saved;
      switchAnalysis(state, id);
      // Keep refresh and copied URLs bound to the offer actually on screen.
      const target=`/workspace/${encodeURIComponent(id)}`;
      history.replaceState(history.state, '', target+(location.pathname===target?location.search:''));
      table.setRun(null);
      activateView('invoices',{replaceUrl:false});
      state.eligibleRowIds = new Set(saved.eligible_row_ids || []);
      const sellers = find('#ws-seller');
      sellers.querySelectorAll('ui5-option:not(:first-child)').forEach(option => option.remove());
      for (const value of saved.filter_options?.seller_id || []) {
        const option = document.createElement('ui5-option'); option.value = value; option.textContent = value; sellers.append(option);
      }
      sellers.querySelector('ui5-option').selected = true;
      find('#ws-status').querySelector('ui5-option').selected = true;
      find('#ws-search').value = '';
      find('#ws-empty').hidden = true;
      find('#ws-analysis').hidden = false;
      find('#ws-filename').textContent = saved.filename;
      const selection = saved.settings.source_kind === 'selection';
      find('#ws-analysis-date').textContent = selection ? `Invoice recommendation · As of ${saved.analysis_date} · Eligibility approved upstream · ${saved.settings.historical_rows_excluded} historical rows excluded` : `Eligibility analysis · ${saved.analysis_date}`;
      for (const id of ['#ws-rules','#ws-insights']) find(id).hidden = selection;
      find('#ws-rejected').parentElement.hidden = selection;
      find('#ws-status').parentElement.hidden = selection;
      find('#ws-total').nextElementSibling.textContent = selection ? 'Imported candidates; historical rows excluded' : 'Original uploaded population';
      find('#ws-eligible').nextElementSibling.textContent = selection ? 'Approved upstream' : 'Ready for a recommendation';
      find('#ws-total').textContent = saved.total_invoices.toLocaleString();
      find('#ws-eligible').textContent = saved.eligible_count.toLocaleString();
      find('#ws-rejected').textContent = saved.not_eligible_count.toLocaleString();
      renderSelection();
      showMessage('');
      await refreshPage();
      if (request === analysisRequest && !signal.aborted) await funding.open(saved);
    } catch (error) { if (error.name !== 'AbortError') showMessage(error.message, 'Negative'); }
  }

  /** Present original fields and actual rule evidence using safe DOM text. */
  function inspectInvoice(row) {
    const dialog = find('#ws-details');
    const body = dialog.querySelector('.ws-detail-body');
    body.replaceChildren();
    const details = document.createElement('dl');
    for (const [key, value] of Object.entries({Reference:row.invoice.invoice_ref,
      Customer:row.invoice.debtor_name, Seller:row.invoice.seller_name, Amount:invoiceAmount(row.invoice),
      'Issue date':row.invoice.issuance_date, 'Due date':row.invoice.due_date,
      Eligibility:row.eligibility_source === 'upstream' ? 'Approved upstream' : row.eligible ? 'Eligible' : 'Not eligible', 'Source row':row.source_row_number})) {
      const term = document.createElement('dt'); term.textContent = key;
      const definition = document.createElement('dd'); definition.textContent = value ?? 'Unavailable';
      details.append(term, definition);
    }
    body.append(details);
    for (const failure of row.diagnostics.failed_rules) {
      const card = document.createElement('article');
      const title = document.createElement('strong'); title.textContent = `${failure.rule_code} · ${failure.description}`;
      card.append(title);
      for (const text of failure.diagnostic?.bullets || [failure.details]) {
        if (!text) continue;
        const paragraph = document.createElement('p'); paragraph.textContent = text; card.append(paragraph);
      }
      body.append(card);
    }
    dialog.open = true;
  }

  /** Open the paginated durable-analysis chooser. */
  function openSaved() {savedAnalyses.open();}

  find('#ws-saved').addEventListener('click', openSaved, {signal});
  find('#ws-view-invoices').addEventListener('click',()=>activateView('invoices'),{signal});
  find('#ws-view-results').addEventListener('click',()=>activateView('results'),{signal});
  for (const id of ['#ws-upload-selection','#ws-empty-selection']) find(id).addEventListener('click', () => upload.open(null, 'selection'), {signal});
  find('#ws-upload').addEventListener('click', () => upload.open(), {signal});
  find('#ws-empty-upload').addEventListener('click', () => upload.open(), {signal});
  find('#ws-insights').addEventListener('click', () => insights.open({analysisId:state.analysisId,
    rowIds:state.selectedRowIds.size ? [...state.selectedRowIds] : null,filters:state.filters}), {signal});
  find('#ws-go').addEventListener('click', () => {
    state.filters = {status:find('#ws-status').selectedOption.value, search:find('#ws-search').value,
      seller_id:find('#ws-seller').selectedOption.value};
    state.offset = 0;
    publishAssistantScope();
    refreshPage();
  }, {signal});
  find('#ws-rules').addEventListener('click', () => {
    upload.open(analysis.settings);
  }, {signal});
  root.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => { button.closest('ui5-dialog').open = false; }, {signal}));
  if (routeContext?.params?.analysisId) openAnalysis(routeContext.params.analysisId);

  /** Abort all active requests and release the table when navigating elsewhere. */
  function destroy() { setAssistantWorkspaceContext({}); state.abortController.abort(); pageRequest?.abort(); table.destroy(); upload.destroy(); insights.destroy(); savedAnalyses.destroy(); funding.destroy(); }
  return {destroy, state, openAnalysis, showMessage, refreshPage};
}

/** Router lifecycle entry point for the workspace page. */
export default function init(routeContext) {
  return mountWorkspace(document.querySelector('#receivables-workspace'), routeContext);
}
