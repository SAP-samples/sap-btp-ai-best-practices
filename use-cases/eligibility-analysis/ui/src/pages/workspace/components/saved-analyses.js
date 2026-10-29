/** Paginated, metadata-rich saved-analysis chooser backed by the existing HANA endpoint. */
import '@ui5/webcomponents/dist/Button.js';
import '@ui5/webcomponents/dist/CheckBox.js';
import '@ui5/webcomponents/dist/Tag.js';
import * as api from '../../../services/workspace-api.js';
import {analysisPage,savedAnalysisDeletion,savedAnalysisRow,updateSavedAnalysisSelection} from './saved-analysis-model.js';

/**
 * Create a text-only element for structured saved-analysis content.
 * @param {string} tag HTML tag name.
 * @param {string} className CSS class.
 * @param {unknown} content Text content.
 * @returns {HTMLElement} Detached element.
 */
function textElement(tag,className='',content='') {
  const node=document.createElement(tag);if(className)node.className=className;node.textContent=String(content??'');return node;
}

/**
 * Mount the saved-analysis list into its existing dialog.
 * @param {HTMLElement} dialog UI5 dialog already owned by the workspace page.
 * @param {{onOpen:(id:string)=>void,onDeleted?:(ids:string[])=>void}} actions Navigation and deletion callbacks.
 * @returns {{open:()=>Promise<void>,destroy:()=>void}} Component lifecycle.
 */
export function mountSavedAnalyses(dialog,{onOpen,onDeleted=()=>{}}) {
  const host=dialog.querySelector('.ws-saved-list');
  const deleteButton=dialog.querySelector('[data-delete-selected]');
  const controller=new AbortController();
  let page=1;
  let generation=0;
  let selected=new Set();
  let deleting=false;

  /** Fetch and render one server-side page without merging repeated filenames. */
  async function load(requestedPage) {
    page=analysisPage(requestedPage).page;
    const token=++generation;host.replaceChildren(textElement('p','ws-saved-loading','Loading saved analyses…'));
    try {
      const query=analysisPage(page);
      const response=await api.listAnalyses({limit:query.limit,offset:query.offset},controller.signal);
      if(token!==generation)return;
      const totalPages=Math.max(1,Math.ceil(response.total/query.limit));
      if(page>totalPages)return load(totalPages);
      host.replaceChildren();
      const header=textElement('div','ws-saved-header');
      for(const label of ['Select','Workflow and file','Analysis date','Uploaded','Invoices'])header.append(textElement('span','',label));
      host.append(header);
      const rows=response.items.map(savedAnalysisRow);
      for(const saved of rows) {
        const row=textElement('div','ws-saved-row');
        const checkbox=document.createElement('ui5-checkbox');checkbox.checked=selected.has(saved.id);
        checkbox.accessibleName=`Select ${saved.workflow} ${saved.filename}, ID ${saved.shortId}`;
        checkbox.addEventListener('change',()=>{selected=updateSavedAnalysisSelection(selected,saved.id,checkbox.checked);},{signal:controller.signal});
        const button=textElement('button','ws-saved-open');button.type='button';
        button.setAttribute('aria-label',`Open ${saved.workflow} ${saved.filename}, analysis ${saved.analysisDate}, ID ${saved.shortId}`);
        const identity=textElement('span','ws-saved-identity');
        const type=document.createElement('ui5-tag');type.design=saved.workflow==='Recommendation'?'Information':'Neutral';type.textContent=saved.workflow;
        const file=textElement('span');file.append(textElement('strong','',saved.filename),textElement('small','',`ID ${saved.shortId}`));identity.append(type,file);
        button.append(identity,textElement('span','',saved.analysisDate),textElement('span','',saved.uploadedAt),textElement('span','ws-saved-count',saved.invoiceCount.toLocaleString()));
        button.addEventListener('click',()=>{dialog.open=false;onOpen(saved.id);},{signal:controller.signal});
        row.append(checkbox,button);host.append(row);
      }
      if(!rows.length)host.append(textElement('p','ws-result-empty','No saved analyses found.'));
      const footer=textElement('div','ws-saved-pagination');
      footer.append(textElement('span','',response.total?`Page ${page} of ${totalPages} · ${response.total} analyses`:'0 analyses'));
      const actions=textElement('span','ws-actions');
      const previous=document.createElement('ui5-button');previous.design='Transparent';previous.icon='navigation-left-arrow';previous.accessibleName='Previous saved analyses page';previous.disabled=page===1;
      const next=document.createElement('ui5-button');next.design='Transparent';next.icon='navigation-right-arrow';next.accessibleName='Next saved analyses page';next.disabled=page>=totalPages;
      previous.addEventListener('click',()=>load(page-1),{signal:controller.signal});next.addEventListener('click',()=>load(page+1),{signal:controller.signal});actions.append(previous,next);footer.append(actions);host.append(footer);
    } catch(error) {
      if(error.name!=='AbortError'&&token===generation)host.replaceChildren(textElement('p','ws-result-empty',error.message));
    }
  }

  /** Delete the exact cross-page selection and reload the remaining server page. */
  async function deleteSelected() {
    const request=savedAnalysisDeletion(selected);
    if(!request||deleting)return;
    deleting=true;deleteButton.disabled=true;
    try {
      const result=await api.deleteAnalyses(request.analysis_ids,controller.signal);
      selected=new Set();onDeleted(result.deleted_analysis_ids);await load(page);
    } catch(error) {
      if(error.name!=='AbortError'){
        const message=textElement('p','ws-saved-error',error.message);message.setAttribute('role','alert');host.prepend(message);
      }
    } finally {deleting=false;deleteButton.disabled=false;}
  }

  deleteButton.addEventListener('click',deleteSelected,{signal:controller.signal});

  return {
    async open(){dialog.open=true;page=1;selected=new Set();await load(page);},
    destroy(){generation++;controller.abort();host.replaceChildren();},
  };
}
