/** Grouped recommendation downloads, safe report preview, and failure-only retry. */
import '@ui5/webcomponents/dist/Button.js';
import '@ui5/webcomponents/dist/Dialog.js';
import '@ui5/webcomponents/dist/Icon.js';
import '@ui5/webcomponents/dist/Tag.js';
import * as api from '../../../services/workspace-api.js';
import {formatAssistantMarkdown} from '../../../modules/chat-markdown.js';
import {buildDownloadGroups} from './download-model.js';

/**
 * Create a text-only DOM element.
 * @param {string} tag HTML tag name.
 * @param {string} className CSS class.
 * @param {unknown} text Text content.
 * @returns {HTMLElement} Detached element.
 */
function textElement(tag,className='',text='') {
  const node=document.createElement(tag);if(className)node.className=className;node.textContent=String(text??'');return node;
}

/**
 * Translate an artifact state into a semantic UI5 tag.
 * @param {string} status Artifact readiness state.
 * @returns {HTMLElement} Status tag.
 */
function statusTag(status) {
  const tag=document.createElement('ui5-tag');
  tag.design=status==='ready'?'Positive':status==='failed'?'Negative':'Information';
  tag.textContent=status==='ready'?'Ready':status==='failed'?'Failed':'Pending';
  return tag;
}

/**
 * Mount the recommendation-file dialog beside the optimizer action.
 * @param {HTMLElement} host Shared workspace panel host.
 * @returns {{open:(id:string)=>Promise<void>,openReport:(id:string)=>Promise<void>,close:()=>void,destroy:()=>void}} Component lifecycle.
 */
export function mountDownloadMenu(host) {
  const dialog=document.createElement('ui5-dialog');dialog.headerText='Recommendation files';
  const previewDialog=document.createElement('ui5-dialog');previewDialog.headerText='Recommendation report';
  const previewContent=textElement('article','ws-report-preview markdown-content');previewContent.setAttribute('aria-label','Recommendation report preview');
  const previewFooter=textElement('div','ws-actions');previewFooter.slot='footer';
  const closePreviewButton=document.createElement('ui5-button');closePreviewButton.textContent='Close report';previewFooter.append(closePreviewButton);previewDialog.append(previewContent,previewFooter);
  const list=textElement('div','ws-download-list');
  const footer=textElement('div','ws-actions');footer.slot='footer';
  const retry=document.createElement('ui5-button');retry.icon='refresh';retry.textContent='Retry failed files';retry.hidden=true;
  const closeButton=document.createElement('ui5-button');closeButton.textContent='Close';footer.append(retry,closeButton);dialog.append(list,footer);host.append(dialog,previewDialog);
  let runId=null;
  let generation=0;

  /** Download one ready file using its unchanged artifact ID and filename. */
  async function download(item,button) {
    button.disabled=true;
    try{api.saveBlob(await api.runArtifact(runId,item.artifactId),item.filename);}
    finally{button.disabled=false;}
  }

  /** Preview the existing report source with the same safe Markdown policy as chat. */
  async function openReport(id=runId) {
    previewContent.replaceChildren(textElement('p','ws-saved-loading','Loading recommendation report…'));
    previewDialog.open=true;
    try {
      const markdown=await (await api.runArtifact(id,'report-markdown')).text();
      previewContent.innerHTML=formatAssistantMarkdown(markdown);
    } catch(error) {
      previewContent.replaceChildren(textElement('p','ws-download-error',error.message));
      throw error;
    }
  }

  /** Close the authenticated report preview and clear its rendered contents. */
  function closePreview() {
    previewDialog.open=false;previewContent.replaceChildren();
  }

  /** Build one file row with description, status, and only meaningful actions. */
  function artifactElement(item) {
    const row=textElement('article','ws-download-row');
    const type=textElement('div','ws-download-type');
    const icon=document.createElement('ui5-icon');icon.name=item.fileType==='XLSX'?'excel-attachment':item.fileType==='PDF'?'pdf-attachment':'document-text';type.append(icon,textElement('strong','',item.fileType));
    const description=textElement('div','ws-download-description');description.append(textElement('strong','',item.label),textElement('small','',item.description));
    const actions=textElement('div','ws-download-actions');actions.append(statusTag(item.status));
    if(item.canPreview){const preview=document.createElement('ui5-button');preview.design='Transparent';preview.textContent='Preview';preview.addEventListener('click',()=>openReport().catch(error=>{list.prepend(textElement('p','ws-download-error',error.message));}));actions.append(preview);}
    const save=document.createElement('ui5-button');save.design='Transparent';save.icon='download';save.textContent='Download';save.disabled=!item.canDownload;
    save.addEventListener('click',()=>download(item,save).catch(error=>{list.prepend(textElement('p','ws-download-error',error.message));}));actions.append(save);
    row.append(type,description,actions);
    if(item.error)row.append(textElement('small','ws-download-item-error',item.error));
    return row;
  }

  /** Refresh the manifest without rerunning any model or optimizer stage. */
  async function refresh() {
    const id=runId,token=generation;list.replaceChildren(textElement('p','ws-saved-loading','Loading recommendation files…'));
    const {items}=await api.listArtifacts(id);if(token!==generation)return;
    const model=buildDownloadGroups(items);list.replaceChildren();retry.hidden=!model.retryVisible;
    if(model.primary) {
      const primary=textElement('section','ws-download-primary');
      const copy=textElement('div');copy.append(textElement('span','ws-eyebrow','COMPLETE PACKAGE'),textElement('h3','',model.primary.label),textElement('p','',model.primary.description));
      const action=document.createElement('ui5-button');action.design='Emphasized';action.icon='download';action.textContent='Download ZIP';action.disabled=!model.primary.canDownload;
      action.addEventListener('click',()=>download(model.primary,action).catch(error=>{list.prepend(textElement('p','ws-download-error',error.message));}));primary.append(copy,statusTag(model.primary.status),action);list.append(primary);
    }
    for(const group of model.groups) {
      const section=textElement('section','ws-download-group');section.append(textElement('h3','',group.label));
      for(const item of group.items)section.append(artifactElement(item));list.append(section);
    }
  }

  /** Invalidate pending manifest responses when leaving this run. */
  function close(){generation++;dialog.open=false;}

  closeButton.addEventListener('click',close);
  closePreviewButton.addEventListener('click',closePreview);
  retry.addEventListener('click',async()=>{retry.disabled=true;try{await api.retryReport(runId);await refresh();}catch(error){list.prepend(textElement('p','ws-download-error',error.message));}finally{retry.disabled=false;}});
  return {
    async open(id){generation++;runId=id;dialog.open=true;try{await refresh();}catch(error){list.replaceChildren(textElement('p','ws-download-error',error.message));}},
    openReport,
    close,
    destroy(){close();closePreview();dialog.remove();previewDialog.remove();},
  };
}
