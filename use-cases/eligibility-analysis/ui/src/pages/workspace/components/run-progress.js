/** Real preparation progress and an explicit saved-assumption acknowledgement gate. */
import * as api from '../../../services/workspace-api.js';
import {watchRun} from './run-polling.js';

/** Mount a durable-state observer; closing a dialog never acknowledges its assumptions. */
export function mountRunProgress(host,{onState,onError}) {
  const dialog=document.createElement('ui5-dialog');dialog.headerText='Review lifetime assumptions';
  dialog.innerHTML='<div class="ws-form"><p data-summary></p><div data-totals></div><details><summary>Affected invoices and reasons</summary><div data-rows></div></details><ui5-checkbox data-accept text="I acknowledge the four-week lifetime assumption for these invoices"></ui5-checkbox><ui5-message-strip data-error hidden hide-close-button></ui5-message-strip></div><div slot="footer" class="ws-actions"><ui5-button data-later>Review later</ui5-button><ui5-button data-cancel>Cancel run</ui5-button><ui5-button data-continue design="Emphasized" disabled>Continue recommendation</ui5-button></div>';
  host.append(dialog);let current=null,watch=null,shownPreparation=null,epoch=0;
  /** Display only the server-issued snapshot and reset consent when it changes. */
  function review() {
    if(!current||current.status!=='awaiting_lifetime_acknowledgement')return;
    const preparation=current.preparation;dialog.querySelector('[data-summary]').textContent=`RPT-1 could not supply a valid lifetime for ${preparation.fallback_count} invoices. Continue using 28 days (four weeks) for those invoices?`;
    dialog.querySelector('[data-totals]').textContent=Object.entries(preparation.fallback_amounts_by_currency||{}).map(([currency,amount])=>`${currency} ${amount}`).join(' · ');
    const rows=dialog.querySelector('[data-rows]');rows.replaceChildren();
    for(const row of preparation.predictions.filter(p=>p.source!=='rpt1')) {const p=document.createElement('p');p.textContent=`${row.invoice_reference||row.row_id} · ${row.reason}`;rows.append(p);}
    dialog.querySelector('[data-accept]').checked=false;dialog.querySelector('[data-continue]').disabled=true;dialog.querySelector('[data-error]').hidden=true;dialog.open=true;
  }
  /** Observe state changes and open a newly pending preparation once without mutating it. */
  async function receive(run) {
    current=run;await onState(run);
    if(run.status==='awaiting_lifetime_acknowledgement'&&shownPreparation!==run.preparation_id){shownPreparation=run.preparation_id;review();}
    if(run.status!=='awaiting_lifetime_acknowledgement')dialog.open=false;
  }
  /** Start or replace a route-scoped poller for an explicit saved run. */
  function start(run) {
    epoch++;watch?.abort();watch=new AbortController();receive(run);
    if(!['completed','failed','cancelled'].includes(run.status)) watchRun(run.run_id,{api,onState:receive,signal:watch.signal}).catch(error=>onError(error.message));
  }
  dialog.querySelector('[data-accept]').addEventListener('change',event=>dialog.querySelector('[data-continue]').disabled=!event.target.checked);
  dialog.querySelector('[data-later]').addEventListener('click',()=>dialog.open=false);
  /** Submit one deliberate action, refreshing conflicts without transferring consent. */
  async function action(operation) {
    const button=dialog.querySelector('[data-continue]');button.disabled=true;const token=epoch,target=current;
    try {const saved=await operation(target);if(token!==epoch)return;dialog.open=false;start(saved);}
    catch(error){if(token!==epoch)return;if(error.status===409){shownPreparation=null;const refreshed=await api.getRun(target.run_id);if(token===epoch)start(refreshed);}else{const strip=dialog.querySelector('[data-error]');strip.textContent=error.message;strip.hidden=false;}}
    finally{button.disabled=!dialog.querySelector('[data-accept]').checked;}
  }
  dialog.querySelector('[data-continue]').addEventListener('click',()=>{if(dialog.querySelector('[data-accept]').checked)action(api.acknowledgeRun);});
  dialog.querySelector('[data-cancel]').addEventListener('click',()=>action(api.cancelRun));
  return {start,review,stop(){epoch++;watch?.abort();dialog.open=false;current=null;},destroy(){epoch++;watch?.abort();dialog.remove();}};
}
