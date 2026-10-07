/** One-offer upload and explicit eligibility-rule settings in a UI5 dialog. */
import '@ui5/webcomponents/dist/FileUploader.js';
import '@ui5/webcomponents/dist/DatePicker.js';
import '@ui5/webcomponents/dist/CheckBox.js';
import {analyzeFile, importSelection} from '../../../services/workspace-api.js';

/** Mount an upload dialog; changed inputs get a new retry key, uncertain retries reuse it. */
export function mountAnalysisPanel(host, {onAnalyzed}) {
  const dialog = document.createElement('ui5-dialog');
  dialog.headerText = 'Analyze offer';
  dialog.className = 'ws-upload-dialog';
  dialog.innerHTML = `<div class="ws-form"><p>Upload one offer file and choose the date used for eligibility checks.</p>
    <label>Offer file<ui5-file-uploader accept=".xlsx" placeholder="Choose one Excel offer" accessible-name="Offer file"><ui5-button icon="upload">Choose file</ui5-button></ui5-file-uploader></label>
    <label>Analysis date<ui5-date-picker format-pattern="yyyy-MM-dd" accessible-name="Analysis date"></ui5-date-picker></label>
    <details><summary>Eligibility rules</summary><div class="ws-form-grid">
      <label>Minimum days to due date<ui5-input data-field="nddt" type="Number" value="6"></ui5-input></label>
      <label>Maximum tenor (days)<ui5-input data-field="teih" type="Number" value="15"></ui5-input></label>
      <label>Minimum days since issue<ui5-input data-field="isspur" type="Number" value="0"></ui5-input></label>
      <label>Allowed currencies<ui5-input data-field="eligible_currencies" value="EUR, USD"></ui5-input></label></div></details>
    <ui5-message-strip hidden hide-close-button></ui5-message-strip></div>
    <div slot="footer" class="ws-actions"><ui5-button data-cancel design="Transparent">Cancel</ui5-button>
    <ui5-button data-submit design="Emphasized">Analyze eligibility</ui5-button></div>`;
  host.append(dialog);
  const controller = new AbortController();
  const signal = controller.signal;
  const date = dialog.querySelector('ui5-date-picker');
  const today = new Date();
  date.value = `${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(today.getDate()).padStart(2,'0')}`;
  let requestKey = crypto.randomUUID();
  let lastFingerprint = null;
  let busy = false;
  let mode = 'eligibility';
  const submit = dialog.querySelector('[data-submit]');
  const cancel = dialog.querySelector('[data-cancel]');
  const message = dialog.querySelector('ui5-message-strip');
  /** Submit exactly one current workbook, retaining a key only for unchanged inputs. */
  async function analyze() {
    if (busy) return;
    const file = dialog.querySelector('ui5-file-uploader').files?.[0];
    if (!file) { message.textContent = 'Choose an Excel offer file.'; message.design = 'Critical'; message.hidden = false; return; }
    const settings = {};
    dialog.querySelectorAll('[data-field]').forEach(input => {
      settings[input.dataset.field] = input.dataset.field === 'eligible_currencies' ? input.value.split(',').map(value => value.trim().toUpperCase()).filter(Boolean) : Number(input.value);
    });
    const fingerprint = JSON.stringify([file.name,file.size,file.lastModified,date.value,settings,mode]);
    if (lastFingerprint !== fingerprint) requestKey = crypto.randomUUID();
    lastFingerprint = fingerprint;
    busy = true; submit.disabled = true; cancel.disabled = true;
    message.textContent = mode === 'selection' ? 'Importing eligible candidates…' : 'Checking invoice eligibility…'; message.design = 'Information'; message.hidden = false;
    try {
      const result = await (mode === 'selection' ? importSelection : analyzeFile)(file,{analysisDate:date.value,settings,requestKey},signal);
      if (signal.aborted) return;
      dialog.open = false;
      requestKey = crypto.randomUUID(); lastFingerprint = null;
      await onAnalyzed(result);
    } catch (error) {
      if (error.name !== 'AbortError') { message.textContent = error.message; message.design = 'Negative'; }
    } finally { busy = false; submit.disabled = false; cancel.disabled = false; }
  }
  submit.addEventListener('click',analyze,{signal});
  cancel.addEventListener('click',() => { dialog.open = false; },{signal});
  /** Open current draft rule settings without changing saved prior analyses. */
  function open(settings, sourceKind='eligibility') {
    mode = sourceKind;
    dialog.headerText = mode === 'selection' ? 'Upload for invoice recommendation' : 'Upload for invoice eligibility';
    dialog.querySelector('.ws-form > p').textContent = mode === 'selection' ? 'Upload one extraction of invoices already approved for eligibility. Original arrival dates are preserved; explicitly marked historical rows are excluded.' : 'Upload one offer file and choose the date used for eligibility checks.';
    dialog.querySelector('details').hidden = mode === 'selection';
    submit.textContent = mode === 'selection' ? 'Import invoices' : 'Analyze eligibility';
    date.accessibleName = mode === 'selection' ? 'As-of date' : 'Analysis date';
    date.parentElement.firstChild.textContent = mode === 'selection' ? 'As-of date' : 'Analysis date';
    const uploader = dialog.querySelector('ui5-file-uploader');
    uploader.accessibleName = mode === 'selection' ? 'Extraction file' : 'Offer file';
    uploader.parentElement.firstChild.textContent = mode === 'selection' ? 'Extraction file' : 'Offer file';
    message.hidden = true;
    if (settings) for (const [key,value] of Object.entries(settings)) {
      const input = dialog.querySelector(`[data-field="${key}"]`);
      if (input) input.value = Array.isArray(value) ? value.join(', ') : String(value);
    }
    dialog.open = true;
  }
  /** Stop requests/listeners and remove the dialog when its workspace is destroyed. */
  function destroy() { controller.abort(); dialog.remove(); }
  return {open,destroy};
}
