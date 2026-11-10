/** Contextual UI5 eligibility diagnostics with scope-bound export actions. */
import Chart from 'chart.js/auto';
import {getInsights, analysisExport, saveBlob} from '../../../services/workspace-api.js';

/** Mount scoped insights with cancellation and chart cleanup on every refresh. */
export function mountInsightsDialog(host) {
  const dialog = document.createElement('ui5-dialog');
  dialog.headerText = 'Eligibility Insights';
  dialog.className = 'ws-insights-dialog';
  dialog.innerHTML = `<div class="ws-insights-body"><p data-scope></p><div class="ws-filters">
    <label>Seller ID<ui5-input data-filter="seller_id" placeholder="All sellers"></ui5-input></label>
    <label>Debtor ID<ui5-input data-filter="debtor_id" placeholder="All debtors"></ui5-input></label>
    <label>Program<ui5-input data-filter="programa" placeholder="All programs"></ui5-input></label>
    <label>Insurer ID<ui5-input data-filter="insurer_id" placeholder="All insurers"></ui5-input></label>
    <label>History<ui5-select data-lookback><ui5-option value="90">Last 90 days</ui5-option><ui5-option value="180">Last 180 days</ui5-option><ui5-option value="365">Last year</ui5-option></ui5-select></label>
    <ui5-button data-refresh>Analyze</ui5-button></div><ui5-message-strip data-notice hide-close-button></ui5-message-strip>
    <div class="ws-metrics ws-card" data-metrics></div><div class="ws-trend"><canvas aria-label="Historical rejection rates" role="img"></canvas></div>
    <div data-alerts></div><details><summary>Rule evidence and comparison dates</summary><pre data-evidence></pre></details></div>
    <div slot="footer" class="ws-actions"><ui5-button data-export="insights-pdf" icon="pdf-attachment">Export PDF</ui5-button>
    <ui5-button data-export="insights-excel" icon="excel-attachment">Export Excel</ui5-button><ui5-button data-close>Close</ui5-button></div>`;
  host.append(dialog);
  const controller = new AbortController();
  const signal = controller.signal;
  let context = null;
  let pending = null;
  let snapshot = null;
  let chart = null;
  const notice = dialog.querySelector('[data-notice]');
  /** Load only the latest requested scope, keeping exports disabled while stale. */
  async function refresh() {
    pending?.abort();
    const request = new AbortController(); pending = request;
    snapshot = null;
    dialog.querySelectorAll('[data-export]').forEach(button => { button.disabled = true; });
    notice.textContent = 'Comparing current invoices with earlier eligibility outcomes…'; notice.design = 'Information';
    const filters = {...context.filters};
    dialog.querySelectorAll('[data-filter]').forEach(input => { filters[input.dataset.filter] = input.value.trim(); });
    try {
      const result = await getInsights(context.analysisId,{row_ids:context.rowIds,filters,
        lookback_days:Number(dialog.querySelector('[data-lookback]').selectedOption.value)},request.signal);
      if (signal.aborted || pending !== request) return;
      snapshot = result;
      const current = result.current_metrics, historical = result.historical_metrics;
      notice.textContent = result.history_status === 'insufficient_history' ?
        'Insufficient comparable history. The current eligibility results remain available.' :
        `Compared with ${historical.total} earlier invoices. Current source rows and repeated analyses are excluded.`;
      notice.design = result.history_status === 'insufficient_history' ? 'Information' : 'Positive';
      const metrics = dialog.querySelector('[data-metrics]'); metrics.replaceChildren();
      for (const [label,value] of [['Current invoices',current.total],['Current not eligible',current.not_eligible_rate == null ? '—' : `${current.not_eligible_rate}%`],
        ['Historical invoices',historical.total],['Historical not eligible',historical.not_eligible_rate == null ? '—' : `${historical.not_eligible_rate}%`]]) {
        const card = document.createElement('div'); const title = document.createElement('span'); const number = document.createElement('strong');
        title.textContent = label; number.textContent = value; card.append(title,number); metrics.append(card);
      }
      const alerts = dialog.querySelector('[data-alerts]'); alerts.replaceChildren();
      for (const alert of result.alerts) {
        const card = document.createElement('article'); card.className = 'ws-insight-alert';
        const title = document.createElement('h3'); title.textContent = alert.title;
        const description = document.createElement('p'); description.textContent = alert.description;
        card.append(title,description); alerts.append(card);
      }
      if (!result.alerts.length) alerts.textContent = 'No supported alerts for this scope.';
      dialog.querySelector('[data-evidence]').textContent = JSON.stringify({rules:result.evidence,period:result.comparison_period,exclusions:result.comparison_exclusions},null,2);
      chart?.destroy(); chart = null;
      const chartHost = dialog.querySelector('.ws-trend'); chartHost.hidden = !result.trend.length;
      if (result.trend.length) chart = new Chart(dialog.querySelector('canvas'), {type:'bar',
        data:{labels:result.trend.map(point => point.date),datasets:[{label:'Not eligible (%)',data:result.trend.map(point => point.not_eligible_rate),backgroundColor:'#7497b5'}]},
        options:{responsive:true,maintainAspectRatio:false,scales:{y:{min:0,max:100}}}});
      dialog.querySelectorAll('[data-export]').forEach(button => { button.disabled = false; });
    } catch (error) { if (error.name !== 'AbortError') { notice.textContent = error.message; notice.design = 'Negative'; } }
  }
  dialog.querySelector('[data-refresh]').addEventListener('click',refresh,{signal});
  dialog.querySelector('[data-close]').addEventListener('click',() => { dialog.open = false; },{signal});
  dialog.addEventListener('close',() => { pending?.abort(); chart?.destroy(); chart = null; },{signal});
  dialog.querySelectorAll('[data-export]').forEach(button => button.addEventListener('click',async () => {
    if (!snapshot) return;
    try { saveBlob(await analysisExport(context.analysisId,button.dataset.export,snapshot.scope_token),`${button.dataset.export}.${button.dataset.export.endsWith('pdf') ? 'pdf' : 'xlsx'}`); }
    catch (error) { notice.textContent = error.message; notice.design = 'Negative'; }
  },{signal}));
  /** Freeze the initiating table scope before opening the modal. */
  function open(value) {
    context = structuredClone(value);
    dialog.querySelector('[data-scope]').textContent = context.rowIds ? `${context.rowIds.length} selected source rows` : 'Current filtered invoice population';
    dialog.querySelectorAll('[data-filter]').forEach(input => { input.value = value.filters?.[input.dataset.filter] || ''; });
    dialog.open = true; refresh();
  }
  /** Release chart, network request and all event subscriptions. */
  function destroy() { controller.abort(); pending?.abort(); chart?.destroy(); dialog.remove(); }
  return {open,destroy};
}
