/** Connect one offer's saved runs, credit settings, scope and recommendation panels. */
import * as api from '../../services/workspace-api.js';
import {candidateScope,runStatusLabel,savedRunLabel} from './state.js';
import {mountCreditPanel} from './components/credit-panel.js';
import {mountRunProgress} from './components/run-progress.js';
import {mountRunResults} from './components/run-results.js';
import {mountDownloadMenu} from './components/download-menu.js';

/** Mount recommendation orchestration independently of upload and eligibility controllers. */
export function mountFunding(root,{state,showMessage,onRun=()=>{}}) {
  const find=s=>root.querySelector(s);const host=find('#ws-panels');let analysis=null,current=null,busy=false,generation=0;
  const downloads=mountDownloadMenu(host);
  const results=mountRunResults(find('#ws-results'),{
    onDownload:()=>current&&downloads.open(current.run_id),
    onOpenReport:()=>current&&downloads.openReport(current.run_id).catch(error=>showMessage(error.message,'Negative')),
  });
  const credit=mountCreditPanel(host,{onSaved:async run=>{await receive(run);await savedRuns();}});
  const progress=mountRunProgress(host,{onState:receive,onError:message=>showMessage(message,'Negative')});
  const selector=find('#ws-run-select');
  /** Publish current durable state and enable only meaningful actions. */
  async function receive(run) {
    if(run.analysis_id!==analysis?.analysis_id)return;
    const changed=current?.run_id!==run.run_id||current?.revision!==run.revision;current=run;
    find('#ws-plan-state').textContent=runStatusLabel(run);
    find('#ws-download').disabled=run.status!=='completed';
    find('#ws-review-run').hidden=run.status!=='awaiting_lifetime_acknowledgement';
    find('#ws-cancel-run').hidden=!['estimating_lifetimes','awaiting_lifetime_acknowledgement'].includes(run.status);
    find('#ws-retry-solve').hidden=run.status!=='failed'||!run.preparation?.predictions?.length;
    find('#ws-optimize').disabled=['estimating_lifetimes','optimizing','awaiting_lifetime_acknowledgement'].includes(run.status);
    if(run.status==='failed')showMessage(run.result?.error||'The run failed. Review its settings and retry explicitly.','Negative');
    else if(run.status==='draft'&&run.readiness_issues.length)showMessage(`${run.readiness_issues.length} credit-setting issues remain. Open Credit settings to review them.`);
    else showMessage('');
    if(changed){results.render(run);onRun(run);}
    const option=[...selector.querySelectorAll('ui5-option')].find(item=>item.value===run.run_id);
    if(option){for(const item of selector.querySelectorAll('ui5-option'))item.selected=item===option;option.textContent=savedRunLabel(run);}
  }
  /** Read saved runs, preserving exact IDs for reopening completed plans and pending reviews. */
  async function savedRuns() {
    if(!analysis)return;const id=analysis.analysis_id;const response=await api.listRuns(id);if(analysis?.analysis_id!==id)return;
    find('#ws-run-controls').hidden=!response.items.length;
    selector.replaceChildren();const empty=document.createElement('ui5-option');empty.value='';empty.textContent='Choose saved run';selector.append(empty);
    for(const run of response.items){const option=document.createElement('ui5-option');option.value=run.run_id;option.textContent=savedRunLabel(run);option.selected=run.run_id===current?.run_id;selector.append(option);}
    return response.items;
  }
  /** Create a new draft when scope or immutable state differs; copy and revalidate saved settings. */
  async function draftFor(scope) {
    const token=generation;const owner=analysis;
    const desired=scope.mode==='all_eligible'?[...state.eligibleRowIds]:scope.row_ids;
    if(!desired.length)throw new Error('Select at least one eligible invoice for a recommendation.');
    if(current?.status==='draft'&&current.row_ids.length===desired.length&&desired.every(id=>current.row_ids.includes(id)))return current;
    const savedSettings=current?.settings;const draft=await api.createRun(owner.analysis_id,scope);
    ensureActive(token);
    const configured=savedSettings&&Object.keys(savedSettings).length?await api.saveSettings(draft.run_id,draft.revision,savedSettings):draft;
    ensureActive(token);return configured;
  }
  /** Reject late responses when navigation has changed the active source context. */
  function ensureActive(token) {if(token!==generation)throw new DOMException('Offer changed','AbortError');}
  /** Serialize UI intent and pin every continuation to the offer present at its start. */
  async function action(work) {
    if(busy)return;busy=true;const token=generation;const owner=analysis;
    try{await work(()=>ensureActive(token),owner);}catch(error){if(error.name!=='AbortError'&&token===generation)showMessage(error.message,'Negative');}finally{if(token===generation)busy=false;}
  }
  find('#ws-credit').addEventListener('click',()=>action(async(check,owner)=>{
    const run=current?.status==='draft'?current:await draftFor({mode:'all_eligible',row_ids:[]});check();
    await receive(run);check();credit.open(run,owner);await savedRuns();
  }));
  find('#ws-optimize').addEventListener('click',()=>action(async(check,owner)=>{
    const scope=candidateScope(find('#ws-optimize-scope').selectedOption.value,state.selectedRowIds,state.eligibleRowIds);
    const run=await draftFor(scope);check();await receive(run);check();
    if(run.readiness_issues.length){credit.open(run,owner);await savedRuns();return;}
    const prepared=await api.prepareRun(run);check();progress.start(prepared);await savedRuns();
  }));
  find('#ws-review-run').addEventListener('click',()=>progress.review());
  find('#ws-cancel-run').addEventListener('click',()=>action(async(check)=>{const run=await api.cancelRun(current);check();progress.start(run);}));
  find('#ws-retry-solve').addEventListener('click',()=>action(async(check)=>{const run=await api.retrySolve(current);check();progress.start(run);}));
  find('#ws-download').addEventListener('click',()=>downloads.open(current.run_id));
  selector.addEventListener('change',()=>action(async(check)=>{if(selector.selectedOption.value){const run=await api.getRun(selector.selectedOption.value);check();progress.start(run);}}));
  return {
    async open(saved){const requested=new URL(location.href).searchParams.get('run');generation++;busy=false;const token=generation;analysis=saved;current=null;progress.stop();credit.close();downloads.close();results.render(null);onRun(null,{preserveLocation:true});find('#ws-download').disabled=true;find('#ws-optimize').disabled=false;
      for(const id of ['#ws-review-run','#ws-cancel-run','#ws-retry-solve'])find(id).hidden=true;
      find('#ws-run-controls').hidden=true;find('#ws-plan-state').textContent='Not started';const runs=await savedRuns();if(token!==generation)return;
      const pending=runs?.find(run=>run.run_id===requested)||runs?.find(run=>['estimating_lifetimes','optimizing','awaiting_lifetime_acknowledgement'].includes(run.status));if(pending)progress.start(pending);else onRun(null);
    },
    getRun:()=>current,
    resizeResults:()=>results.resize(),
    destroy(){generation++;progress.destroy();credit.destroy();results.destroy();downloads.destroy();}
  };
}
