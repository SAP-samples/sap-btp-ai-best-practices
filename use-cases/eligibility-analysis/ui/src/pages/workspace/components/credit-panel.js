/** Contextual credit settings with a reviewed draft and authoritative server preview. */
import * as api from '../../../services/workspace-api.js';
import {mappingTable,associationTable,inputField} from './settings-fields.js';
import {mountRepaymentEditor} from './repayment-editor.js';

/** Mount a UI5 settings dialog; save is the only operation that persists panel edits. */
export function mountCreditPanel(host,{onSaved}) {
  const dialog=document.createElement('ui5-dialog');dialog.headerText='Credit limits & exposure';dialog.className='ws-credit-dialog';
  dialog.innerHTML=`<div class="ws-form ws-credit-body"><span class="ws-eyebrow">RUN SETTINGS</span>
    <p>Set capacity for this offer. These settings are saved with the run.</p>
    <div class="ws-form-grid"><label>Planning start<ui5-date-picker data-start format-pattern="yyyy-MM-dd" accessible-name="Planning start"></ui5-date-picker></label>
      <label>Planning horizon<ui5-select data-horizon accessible-name="Planning horizon"></ui5-select></label></div>
    <ui5-file-uploader data-import accept=".yaml,.yml,.xlsx,.json" accessible-name="Import credit settings"><ui5-button icon="upload">Import YAML / Excel</ui5-button></ui5-file-uploader>
    <p>EUR · Opening exposure is credit already consumed before this run.</p><div data-customers></div>
    <details><summary>Facility & group limits and associations</summary><div data-associations class="ws-form-grid"></div>
      <ui5-button data-mappings>Apply associations to limit tables</ui5-button><div data-facilities></div><div data-groups></div></details>
    <ui5-checkbox data-zero text="I confirm omitted opening balances are zero"></ui5-checkbox>
    <details><summary>Currency conversion to EUR</summary><div data-fx class="ws-form-grid"></div></details>
    <hr><h3>Expected repayments <small>Optional</small></h3><p>Release part of opening exposure on specified dates. Any remaining balance stays consumed.</p>
    <div data-repayments></div><ui5-message-strip data-no-repayments hide-close-button>No repayments supplied. Opening exposure remains constant across all weeks.</ui5-message-strip>
    <ui5-button data-preview>Preview opening exposure</ui5-button><div data-preview-output></div>
    <ui5-message-strip data-message hidden hide-close-button></ui5-message-strip></div>
    <div slot="footer" class="ws-actions"><ui5-button data-cancel>Cancel</ui5-button><ui5-button data-save design="Emphasized">Save credit settings</ui5-button></div>`;
  host.append(dialog);const find=s=>dialog.querySelector(s);let run=null,draft=null,busy=false,generation=0;
  const repayments=mountRepaymentEditor(find('[data-repayments]'),count=>{find('[data-no-repayments]').hidden=Boolean(count);});
  /** Show readable server validation issues without losing the user's draft. */
  function message(text,design='Information') {const strip=find('[data-message]');strip.textContent=text;strip.design=design;strip.hidden=!text;}
  /** Seed empty required fields from the saved offer, never invented credit amounts. */
  function initialize(settings,analysis) {
    const next=structuredClone(settings||{});const start=new Date(`${analysis.analysis_date}T12:00:00`);
    start.setDate(start.getDate()+((8-start.getDay())%7));
    next.planning_start ||= `${start.getFullYear()}-${String(start.getMonth()+1).padStart(2,'0')}-${String(start.getDate()).padStart(2,'0')}`;
    next.horizon_weeks ||= 12;
    for(const key of ['customer_limits','facility_limits_by_company_code','group_limits','customer_to_group','seller_to_facility','customer_to_facility','currency_rates']) next[key] ||= {};
    next.base_exposure ||= {};for(const kind of ['customer','facility','group']) next.base_exposure[kind] ||= {};
    for(const id of analysis.filter_options.debtor_id) {next.customer_limits[id] ??= '';next.customer_to_facility[id] ??= '';next.customer_to_group[id] ??= '';}
    for(const id of analysis.filter_options.seller_id) next.seller_to_facility[id] ??= '';
    for(const currency of analysis.filter_options.original_currency.filter(v=>v!=='EUR')) next.currency_rates[currency] ||= {eur_per_unit:'',as_of:next.planning_start};
    next.expected_repayments ||= [];return next;
  }
  /** Rebuild editable tables from the current draft after file import or association edits. */
  function render() {
    find('[data-start]').value=draft.planning_start;
    const horizon=find('[data-horizon]');horizon.replaceChildren();
    for(const weeks of [...new Set([4,8,12,15,26,52,draft.horizon_weeks])].sort((a,b)=>a-b)) {
      const option=document.createElement('ui5-option');option.value=String(weeks);option.textContent=`${weeks} weeks`;option.selected=weeks===draft.horizon_weeks;horizon.append(option);
    }
    mappingTable(find('[data-customers]'),'Customer limits',draft.customer_limits,{opening:draft.base_exposure.customer});
    mappingTable(find('[data-facilities]'),'Facility limits',draft.facility_limits_by_company_code,{opening:draft.base_exposure.facility});
    mappingTable(find('[data-groups]'),'Group limits',draft.group_limits,{opening:draft.base_exposure.group});
    const associations=find('[data-associations]');associations.replaceChildren();
    for(const [title,key] of [['Seller → facility','seller_to_facility'],['Customer → facility','customer_to_facility'],['Customer → group (optional)','customer_to_group']]) {
      const section=document.createElement('div');associationTable(section,title,draft[key]);associations.append(section);
    }
    find('[data-zero]').checked=Boolean(draft.opening_confirmed_zero);
    const fx=find('[data-fx]');fx.replaceChildren();
    for(const [currency,rate] of Object.entries(draft.currency_rates)) for(const [key,label] of [['eur_per_unit','EUR per unit'],['as_of','As of (YYYY-MM-DD)']]) {
      const field=document.createElement('label');field.textContent=`${currency} · ${label}`;
      field.append(inputField(rate[key],`${currency} ${label}`,v=>rate[key]=v));fx.append(field);
    }
    repayments.render(draft);find('[data-preview-output]').replaceChildren();
    find('[data-no-repayments]').hidden=Boolean(draft.expected_repayments.length);
  }
  /** Read scalar controls and omit blank optional associations before backend validation. */
  function payload() {
    const value=structuredClone(draft);value.planning_start=find('[data-start]').value;
    value.horizon_weeks=Number(find('[data-horizon]').selectedOption.value);value.opening_confirmed_zero=find('[data-zero]').checked;
    for(const key of ['customer_to_group','customer_to_facility','seller_to_facility']) value[key]=Object.fromEntries(Object.entries(value[key]).filter(([,v])=>v.trim()));
    value.currency_rates=Object.fromEntries(Object.entries(value.currency_rates).filter(([,rate])=>String(rate.eur_per_unit??"" ).trim()));
    return value;
  }
  /** Render the server's opening balances with a table that also works without charts. */
  function preview(result) {
    const output=find('[data-preview-output]');output.replaceChildren();const table=document.createElement('table');table.className='ws-edit-table';
    const head=table.createTHead().insertRow();for(const name of ['Week','Customer opening (EUR)']) {const cell=document.createElement('th');cell.textContent=name;head.append(cell);}
    for(const row of result.weekly_opening_preview) {const tr=table.insertRow();tr.insertCell().textContent=row.week_start;tr.insertCell().textContent=Object.values(row.opening.customer).reduce((sum,value)=>sum+Number(value),0).toLocaleString(undefined,{minimumFractionDigits:2});}
    output.append(table);
    message([...result.readiness_issues.map(issue=>`${issue.path}: ${issue.message}`),...result.warnings].join('\n')||'Credit settings are ready.',result.readiness_issues.length?'Critical':'Positive');
  }
  /** Guard long server actions and preserve edits after validation or stale-revision errors. */
  async function action(work) {
    if(busy)return;busy=true;const token=generation;const owner=run;find('[data-save]').disabled=true;
    const check=()=>{if(token!==generation)throw new DOMException('Settings panel changed','AbortError');};
    try {await work(check,owner);} catch(error){if(error.name!=='AbortError')message(error.message,'Negative');} finally{if(token===generation){busy=false;find('[data-save]').disabled=false;}}
  }
  find('[data-preview]').addEventListener('click',()=>action(async(check,owner)=>{const result=await api.previewSettings(owner.run_id,owner.revision,payload());check();preview(result);}));
  find('[data-save]').addEventListener('click',()=>action(async(check,owner)=>{
    const saved=await api.saveSettings(owner.run_id,owner.revision,payload());check();run=saved;await onSaved(saved);check();dialog.open=false;
  }));
  find('[data-import]').addEventListener('change',()=>action(async(check,owner)=>{
    const file=find('[data-import]').files?.[0];if(!file)return;
    const result=await api.previewSettingsImport(owner.run_id,file,owner.revision);check();draft=result.normalized_settings;render();preview(result);
  }));
  find('[data-mappings]').addEventListener('click',()=>{
    draft=payload();
    for(const facility of Object.values(draft.seller_to_facility).filter(Boolean)) draft.facility_limits_by_company_code[facility] ??= '';
    for(const group of Object.values(draft.customer_to_group).filter(Boolean)) draft.group_limits[group] ??= '';render();
  });
  function close(){generation++;busy=false;dialog.open=false;}
  find('[data-cancel]').addEventListener('click',close);
  return {open(saved,analysis){generation++;busy=false;find('[data-save]').disabled=false;run=saved;draft=initialize(saved.settings,analysis);render();message('');dialog.open=true;},
    close,destroy(){close();repayments.destroy();dialog.remove();}};
}
