/** Pure display and paging helpers for durable saved analyses. */

export const SAVED_ANALYSIS_PAGE_SIZE=10;

/**
 * Convert a one-based page into the existing analyses endpoint query.
 * @param {number} requestedPage One-based requested page.
 * @param {number} pageSize Server page size.
 * @returns {{limit:number,offset:number,page:number}} Stable pagination query.
 */
export function analysisPage(requestedPage=1,pageSize=SAVED_ANALYSIS_PAGE_SIZE) {
  const page=Math.max(1,Number(requestedPage)||1);
  return {limit:pageSize,offset:(page-1)*pageSize,page};
}

/**
 * Return a new saved-upload selection after one checkbox change.
 * @param {Set<string>} selected Current analysis IDs selected across pages.
 * @param {string} analysisId Changed analysis ID.
 * @param {boolean} checked Whether the analysis is selected.
 * @returns {Set<string>} Updated independent selection.
 */
export function updateSavedAnalysisSelection(selected,analysisId,checked) {
  const updated=new Set(selected);
  if(checked)updated.add(analysisId);else updated.delete(analysisId);
  return updated;
}

/**
 * Build the bulk deletion body, or no operation for an empty selection.
 * @param {Set<string>} selected Selected analysis IDs.
 * @returns {{analysis_ids:string[]}|null} Request body or null when nothing is selected.
 */
export function savedAnalysisDeletion(selected) {
  return selected.size?{analysis_ids:[...selected]}:null;
}

/**
 * Format an ISO upload timestamp in a compact, timezone-explicit way.
 * @param {string|null} value ISO timestamp.
 * @returns {string} Human-readable UTC timestamp.
 */
function uploadTimestamp(value) {
  const date=new Date(value);
  if(!value||Number.isNaN(date.getTime()))return 'Unavailable';
  return `${new Intl.DateTimeFormat('en-US',{
    month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit',timeZone:'UTC',
  }).format(date)} UTC`;
}

/**
 * Project one analysis into the structured saved-upload row shown in the dialog.
 * @param {object} saved Analysis summary returned by the existing REST endpoint.
 * @returns {object} Display-safe metadata retaining the full analysis ID for navigation.
 */
export function savedAnalysisRow(saved) {
  return {
    id:saved.analysis_id,
    shortId:String(saved.analysis_id||'').slice(0,8),
    workflow:saved.settings?.source_kind==='selection'?'Recommendation':'Eligibility',
    filename:saved.filename||'Unnamed upload',
    analysisDate:saved.analysis_date||'Unavailable',
    uploadedAt:uploadTimestamp(saved.created_at),
    invoiceCount:Number(saved.total_invoices)||0,
  };
}
