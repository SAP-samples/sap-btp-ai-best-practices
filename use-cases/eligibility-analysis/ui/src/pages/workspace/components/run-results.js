/** Executive recommendation dashboard and paginated evidence drill-downs. */
import Chart from 'chart.js/auto';
import '@ui5/webcomponents/dist/Button.js';
import '@ui5/webcomponents/dist/Icon.js';
import '@ui5/webcomponents/dist/Tag.js';
import {
  buildResultsSummary,
  formatMoney,
  paginateRows,
} from './run-results-model.js';

/**
 * Create a text-only HTML element.
 * @param {string} tag HTML tag name.
 * @param {string} className Optional CSS class.
 * @param {unknown} text Optional text content.
 * @returns {HTMLElement} Detached element.
 */
function textElement(tag,className='',text='') {
  const node=document.createElement(tag);
  if(className)node.className=className;
  node.textContent=String(text??'');
  return node;
}

/**
 * Find a compatible saved invoice field without changing the REST contract.
 * @param {object} row Saved result row.
 * @param {string[]} keys Ordered candidate keys.
 * @param {unknown} fallback Value used when no field is present.
 * @returns {unknown} First available field value.
 */
function field(row,keys,fallback='—') {
  for(const key of keys)if(row?.[key]!=null&&row[key]!=='')return row[key];
  return fallback;
}

/**
 * Render a compact, horizontally scrollable evidence table.
 * @param {HTMLElement} host Table host.
 * @param {Array<{label:string,value:(row:object)=>unknown}>} columns Column definitions.
 * @param {Array<object>} rows Current page rows.
 * @param {string} emptyMessage Message for an empty population.
 */
function renderTable(host,columns,rows,emptyMessage) {
  host.replaceChildren();
  if(!rows.length){host.append(textElement('p','ws-result-empty',emptyMessage));return;}
  const scroll=textElement('div','ws-result-table-scroll');
  const table=document.createElement('table');table.className='ws-result-table';
  const header=table.createTHead().insertRow();
  for(const column of columns)header.append(textElement('th','',column.label));
  const body=table.createTBody();
  for(const item of rows) {
    const row=body.insertRow();
    for(const column of columns)row.insertCell().textContent=String(column.value(item)??'—');
  }
  scroll.append(table);host.append(scroll);
}

/**
 * Render and operate a one-based 50-row paginator around a table.
 * @param {HTMLElement} host Panel content host.
 * @param {Array<object>} rows Complete detail population.
 * @param {number} requestedPage Requested one-based page.
 * @param {Array<object>} columns Table column definitions.
 * @param {string} emptyMessage Message for an empty population.
 * @param {(page:number)=>void} onPage Page-change callback.
 */
function renderPagedTable(host,rows,requestedPage,columns,emptyMessage,onPage) {
  const page=paginateRows(rows,requestedPage);
  const tableHost=textElement('div','ws-result-table-host');host.append(tableHost);
  renderTable(tableHost,columns,page.items,emptyMessage);
  const footer=textElement('div','ws-result-pagination');
  const start=page.total?(page.page-1)*page.pageSize+1:0;
  const end=Math.min(page.page*page.pageSize,page.total);
  footer.append(textElement('span','',page.total?`${start}–${end} of ${page.total}`:'0 rows'));
  const actions=textElement('div','ws-actions');
  const previous=document.createElement('ui5-button');previous.design='Transparent';previous.icon='navigation-left-arrow';
  previous.accessibleName='Previous result page';previous.disabled=page.page===1;
  const next=document.createElement('ui5-button');next.design='Transparent';next.icon='navigation-right-arrow';
  next.accessibleName='Next result page';next.disabled=page.page===page.totalPages;
  previous.addEventListener('click',()=>onPage(page.page-1));
  next.addEventListener('click',()=>onPage(page.page+1));
  actions.append(previous,next);footer.append(actions);host.append(footer);
}

/**
 * Mount a result dashboard that presents recommendations separately from eligibility.
 * @param {HTMLElement} host Results-view host.
 * @param {{onDownload?:()=>void,onOpenReport?:()=>void}} actions External actions owned by the workspace controller.
 * @returns {{render:(run:object|null)=>void,resize:()=>void,destroy:()=>void}} Component lifecycle.
 */
export function mountRunResults(host,{onDownload=()=>{},onOpenReport=()=>{}}={}) {
  let chart=null;
  let summary=null;
  const pages={selected:1,notSelected:1,capacity:1};
  let activeTab='overview';

  /** Destroy the active chart before replacing or removing its canvas. */
  function clearChart() {chart?.destroy();chart=null;}

  /** Render the weekly chart and its exact accessible data table. */
  function renderOverview(panel) {
    const content=textElement('div','ws-overview-grid');
    const chartCard=textElement('article','ws-result-panel ws-chart-panel');
    chartCard.append(textElement('h2','','Weekly exposure and recommendation'));
    chartCard.append(textElement('p','ws-muted','Opening exposure plus the incremental amount outstanding from this saved recommendation.'));
    const chartHost=textElement('div','ws-result-chart');
    const canvas=document.createElement('canvas');canvas.setAttribute('role','img');
    canvas.setAttribute('aria-label','Stacked weekly facility exposure. Exact weekly values are available directly below the chart.');
    chartHost.append(canvas);chartCard.append(chartHost);
    const values=document.createElement('details');
    values.append(textElement('summary','','Weekly values'));
    const valuesHost=textElement('div');values.append(valuesHost);
    renderTable(valuesHost,[
      {label:'Week',value:row=>row.week},
      {label:'Opening exposure',value:row=>formatMoney(row.opening,summary.currency)},
      {label:'Recommended outstanding',value:row=>formatMoney(row.recommended,summary.currency)},
      {label:'Total exposure',value:row=>formatMoney(row.total,summary.currency)},
    ],summary.weekly,'No weekly exposure was saved for this recommendation.');
    chartCard.append(values);content.append(chartCard);

    const constraintCard=textElement('article','ws-result-panel ws-constraint-panel');
    constraintCard.append(textElement('h2','','Constraints closest to limit'));
    constraintCard.append(textElement('p','ws-muted','Peak utilization for each facility, customer, or group, ranked by proximity to its saved limit.'));
    const list=textElement('ol','ws-constraint-list');
    for(const item of summary.constraints.slice(0,5)) {
      const row=textElement('li');
      const heading=textElement('div','ws-constraint-heading');
      heading.append(textElement('strong','',item.entity),textElement('span','',`${item.utilizationPct.toFixed(1)}%`));
      const track=textElement('div','ws-capacity-track');
      const fill=textElement('span','ws-capacity-fill');fill.style.width=`${Math.min(100,item.utilizationPct)}%`;track.append(fill);
      row.append(heading,textElement('small','',`${item.level} · ${item.week} · ${formatMoney(item.total,summary.currency)} of ${formatMoney(item.limit,summary.currency)}`),track);
      list.append(row);
    }
    if(!list.children.length)constraintCard.append(textElement('p','ws-result-empty','No capacity utilization rows were saved.'));
    else constraintCard.append(list);
    content.append(constraintCard);panel.append(content);

    const takeaways=textElement('article','ws-takeaways');
    const takeawayIcon=document.createElement('ui5-icon');takeawayIcon.name='lightbulb';
    const takeawayCopy=textElement('div');takeawayCopy.append(textElement('h2','','Key takeaways'));
    const takeawayList=textElement('ul');
    takeawayList.append(
      textElement('li','',`${summary.metrics.selectedCount.toLocaleString()} invoices (${summary.metrics.selectionRatePct.toFixed(1)}%) are recommended for ${formatMoney(summary.metrics.recommendedAmount,summary.currency)}.`),
      textElement('li','',summary.constraints.length
        ? `${summary.constraints[0].entity} is closest to its saved limit at ${summary.metrics.peakUtilizationPct.toFixed(1)}% utilization.`
        : 'No capacity utilization rows were saved for this recommendation.'),
      textElement('li','',summary.solverStatus==='OPTIMAL'
        ? 'The primary objective is proven optimal for this saved scope and its assumptions.'
        : 'The saved recommendation is feasible; global optimality has not been proven.'),
    );
    takeawayCopy.append(takeawayList);takeaways.append(takeawayIcon,takeawayCopy);panel.append(takeaways);

    clearChart();
    chart=new Chart(canvas,{
      type:'bar',
      data:{labels:summary.weekly.map(row=>row.week),datasets:[
        {label:`Opening exposure (${summary.currency})`,data:summary.weekly.map(row=>row.opening),backgroundColor:'#9fb2c5',borderRadius:3},
        {label:`Recommended outstanding (${summary.currency})`,data:summary.weekly.map(row=>row.recommended),backgroundColor:'#0a6ed1',borderRadius:3},
      ]},
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{legend:{position:'bottom',labels:{usePointStyle:true,boxWidth:8}}},
        scales:{
          x:{stacked:true,grid:{display:false}},
          y:{stacked:true,beginAtZero:true,ticks:{callback:value=>new Intl.NumberFormat('en-US',{notation:'compact'}).format(value)}},
        },
      },
    });
  }

  /** Render the active evidence tab and its independent page position. */
  function renderActivePanel() {
    const panel=host.querySelector('[data-results-panel]');
    if(!panel||!summary)return;
    panel.replaceChildren();clearChart();
    if(activeTab==='overview'){renderOverview(panel);return;}
    if(activeTab==='selected') {
      panel.append(textElement('h2','','Selected invoices'));
      panel.append(textElement('p','ws-muted','Invoices included in the saved recommendation and their planned recommendation week.'));
      renderPagedTable(panel,summary.selectedRows,pages.selected,[
        {label:'Invoice',value:row=>field(row,['Invoice Reference','invoice_reference','row_id'])},
        {label:'Customer',value:row=>field(row,['Customer Name','Customer','debtor_name'])},
        {label:'Original amount',value:row=>formatMoney(field(row,['Original Amount','original_amount'],0),field(row,['Original Currency','original_currency'],summary.currency))},
        {label:`Recommended (${summary.currency})`,value:row=>formatMoney(field(row,['Normalized Amount','normalized_amount'],0),summary.currency)},
        {label:'Recommendation week',value:row=>field(row,['planned_week_start_iso','planned_week_start'])},
      ],'No invoices were selected for this recommendation.',page=>{pages.selected=page;renderActivePanel();});
      return;
    }
    if(activeTab==='notSelected') {
      panel.append(textElement('h2','','Non-selected invoices'));
      panel.append(textElement('p','ws-muted','Invoices remain eligible unless shown as screened before optimization. Non-selection is a recommendation outcome, not an eligibility decision.'));
      renderPagedTable(panel,summary.notSelectedRows,pages.notSelected,[
        {label:'Invoice',value:row=>field(row,['Invoice Reference','invoice_reference','row_id'])},
        {label:'Customer',value:row=>field(row,['Customer Name','Customer','debtor_name'])},
        {label:'Outcome',value:row=>row.outcome},
        {label:'Reason',value:row=>field(row,['excluded_reason','exclusion_reason'],'Capacity or scheduling constraints')},
      ],'Every scoped invoice was selected for this recommendation.',page=>{pages.notSelected=page;renderActivePanel();});
      return;
    }
    if(activeTab==='capacity') {
      panel.append(textElement('h2','','Capacity by week'));
      panel.append(textElement('p','ws-muted','Opening and incremental recommended exposure compared with the saved facility, customer, and group limits.'));
      renderPagedTable(panel,summary.capacityRows,pages.capacity,[
        {label:'Week',value:row=>row.week},
        {label:'Level',value:row=>row.level},
        {label:'Entity',value:row=>row.entity},
        {label:'Opening',value:row=>formatMoney(row.opening,summary.currency)},
        {label:'Recommended',value:row=>formatMoney(row.recommended,summary.currency)},
        {label:'Total',value:row=>formatMoney(row.total,summary.currency)},
        {label:'Limit',value:row=>formatMoney(row.limit,summary.currency)},
        {label:'Utilization',value:row=>`${row.utilizationPct.toFixed(1)}%`},
      ],'No capacity rows were saved for this recommendation.',page=>{pages.capacity=page;renderActivePanel();});
      return;
    }
    panel.append(textElement('h2','','Assumptions and provenance'));
    panel.append(textElement('p','ws-muted','The saved inputs used for this computation. These fields do not represent financing, payment, or disbursement.'));
    const grid=textElement('dl','ws-assumption-grid');
    const history=summary.assumptions.history;
    const values=[
      ['Planning start',summary.assumptions.planningStart||'Unavailable'],
      ['Planning horizon',`${summary.assumptions.horizonWeeks} weeks`],
      ['RPT-1 estimates',summary.assumptions.rpt1Count.toLocaleString()],
      ['28-day fallbacks',summary.assumptions.fallbackCount.toLocaleString()],
      ['Acknowledgement',summary.assumptions.acknowledgementStatus.replaceAll('_',' ')],
      ['Historical source',history.dataset_id||history.table||history.status||'Unavailable'],
      ['Run ID',summary.runId],
    ];
    for(const [term,value] of values)grid.append(textElement('dt','',term),textElement('dd','',value));
    panel.append(grid);
  }

  /** Select a results tab and keep keyboard and ARIA state aligned. */
  function selectTab(name,focus=false) {
    activeTab=name;
    for(const tab of host.querySelectorAll('[data-results-tab]')) {
      const selected=tab.dataset.resultsTab===name;
      tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;
      if(selected&&focus)tab.focus();
    }
    renderActivePanel();
  }

  /** Build the stable result shell and render a completed saved run. */
  function render(run) {
    clearChart();host.replaceChildren();
    if(run?.status!=='completed'||!run.result){summary=null;return;}
    summary=buildResultsSummary(run);activeTab='overview';Object.assign(pages,{selected:1,notSelected:1,capacity:1});
    const header=textElement('header','ws-result-header');
    const heading=textElement('div');
    const status=document.createElement('ui5-tag');status.design=summary.solverStatus==='OPTIMAL'?'Positive':'Information';status.textContent=summary.solverStatus;
    heading.append(status,textElement('h1','','Recommendation results'),textElement('p','ws-muted',`Saved computation ${summary.runId.slice(0,8)} · Eligibility remains unchanged by this recommendation.`));
    const actions=textElement('div','ws-actions');
    const report=document.createElement('ui5-button');report.icon='document-text';report.textContent='Open report';report.addEventListener('click',onOpenReport);
    const download=document.createElement('ui5-button');download.design='Emphasized';download.icon='download';download.textContent='Download files';download.addEventListener('click',onDownload);
    actions.append(report,download);header.append(heading,actions);host.append(header);

    const metrics=textElement('section','ws-result-metrics');metrics.setAttribute('aria-label','Recommendation summary');
    for(const [label,value,detail,iconName] of [
      ['Selected invoices',summary.metrics.selectedCount.toLocaleString(),`of ${run.row_ids.length.toLocaleString()} scoped`,'document-text'],
      ['Recommended amount',formatMoney(summary.metrics.recommendedAmount,summary.currency),'Saved optimizer objective','money-bills'],
      ['Recommendation rate',`${summary.metrics.selectionRatePct.toFixed(1)}%`,'Of the saved candidate scope','pie-chart'],
      ['Peak utilization',`${summary.metrics.peakUtilizationPct.toFixed(1)}%`,'Closest saved capacity limit','bar-chart'],
    ]) {
      const card=textElement('article');
      const icon=document.createElement('ui5-icon');icon.name=iconName;icon.className='ws-result-metric-icon';
      const copy=textElement('div');copy.append(textElement('span','',label),textElement('strong','',value),textElement('small','',detail));
      card.append(icon,copy);metrics.append(card);
    }
    host.append(metrics);
    const guidance=textElement('section',`ws-guidance ${summary.solverStatus==='OPTIMAL'?'is-optimal':'is-feasible'}`);
    guidance.append(textElement('strong','',summary.solverStatus==='OPTIMAL'?'Optimal recommendation':'Feasible recommendation'),textElement('p','',summary.guidance));host.append(guidance);

    const tabs=textElement('div','ws-result-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Recommendation details');
    const definitions=[['overview','Overview'],['selected',`Selected (${summary.selectedRows.length})`],['notSelected',`Non-selected (${summary.notSelectedRows.length})`],['capacity',`Capacity (${summary.capacityRows.length})`],['assumptions','Assumptions']];
    for(const [name,label] of definitions) {
      const tab=textElement('button','ws-result-tab',label);tab.type='button';tab.dataset.resultsTab=name;tab.setAttribute('role','tab');
      tab.addEventListener('click',()=>selectTab(name));
      tab.addEventListener('keydown',event=>{
        const all=[...tabs.querySelectorAll('[data-results-tab]')];const index=all.indexOf(event.currentTarget);
        const next=event.key==='ArrowRight'?all[(index+1)%all.length]:event.key==='ArrowLeft'?all[(index-1+all.length)%all.length]:event.key==='Home'?all[0]:event.key==='End'?all.at(-1):null;
        if(next){event.preventDefault();selectTab(next.dataset.resultsTab,true);}
      });
      tabs.append(tab);
    }
    host.append(tabs,textElement('section','ws-result-detail'));
    host.lastElementChild.dataset.resultsPanel='';selectTab('overview');
  }

  return {render,resize(){chart?.resize();},destroy(){clearChart();host.replaceChildren();summary=null;}};
}
