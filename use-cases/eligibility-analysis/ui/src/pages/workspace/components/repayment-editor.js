/** Optional repayment rows with server-derived opening exposure preview. */
import {inputField} from './settings-fields.js';

/** Mount a mutable panel draft; no client-side exposure simulation is performed. */
export function mountRepaymentEditor(host,onChange=()=>{}) {
  let draft=null;
  /** Render supplied rows and keep edits in the unsaved settings object. */
  function render(settings) {
    draft=settings;host.replaceChildren();
    const toggle=document.createElement('ui5-checkbox');toggle.text='Add an expected repayment schedule';
    toggle.checked=Boolean(draft.expected_repayments.length);host.append(toggle);
    const rows=document.createElement('div');host.append(rows);
    const add=document.createElement('ui5-button');add.textContent='Add repayment';host.append(add);
    /** Rebuild row controls following add/remove operations. */
    function draw() {
      onChange(draft.expected_repayments.length);rows.replaceChildren();add.hidden=!toggle.checked;
      if(!toggle.checked) return;
      draft.expected_repayments.forEach((record,index)=>{
        const card=document.createElement('div');card.className='ws-repayment-row';
        for(const [key,label,type] of [['customer_id','Customer','Text'],['facility_id','Facility','Text'],['release_date','Repayment date (YYYY-MM-DD)','Text'],['amount','Amount','Number'],['currency','Currency','Text']]) {
          const field=document.createElement('label');field.textContent=label;
          field.append(inputField(record[key],`Repayment ${index+1} ${label}`,v=>record[key]=v,type));card.append(field);
        }
        const group=document.createElement('small');group.textContent=`Group: ${draft.customer_to_group[record.customer_id]||'None'}`;card.append(group);
        const remove=document.createElement('ui5-button');remove.textContent='Remove';remove.design='Transparent';
        remove.addEventListener('click',()=>{draft.expected_repayments.splice(index,1);draw();});card.append(remove);rows.append(card);
      });
    }
    toggle.addEventListener('change',()=>{if(!toggle.checked) draft.expected_repayments=[];draw();});
    add.addEventListener('click',()=>{draft.expected_repayments.push({customer_id:'',facility_id:'',release_date:'',amount:'',currency:'EUR'});draw();});draw();
  }
  return {render,destroy:()=>host.replaceChildren()};
}
