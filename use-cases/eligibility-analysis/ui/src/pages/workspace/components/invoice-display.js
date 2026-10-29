/** Pure text and semantic-tag helpers for workspace invoice rows. */

/**
 * Format one original-currency amount; absent amounts remain visibly unavailable.
 * @param {object} invoice Canonical invoice payload.
 * @returns {string} Grouped monetary value and original currency.
 */
export function invoiceAmount(invoice) {
  const value=invoice.amount_original??invoice.total_invoice_amount_original;
  const number=Number(value);
  if(value==null||!Number.isFinite(number))return 'Unavailable';
  return `${new Intl.NumberFormat('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}).format(number)} ${invoice.original_currency||''}`.trim();
}

/**
 * Format an ISO invoice date without shifting its calendar day.
 * @param {string|null} value ISO date or datetime.
 * @returns {string} Compact localized date or Unavailable.
 */
export function formatInvoiceDate(value) {
  if(!value)return 'Unavailable';
  const date=new Date(String(value).length===10?`${value}T00:00:00Z`:value);
  if(Number.isNaN(date.getTime()))return 'Unavailable';
  return new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(date);
}

/**
 * Derive a recommendation tag from saved run membership without changing eligibility.
 * @param {string} rowId Stable source row ID.
 * @param {object|null} run Current saved run.
 * @returns {{label:string,design:string}} UI5 tag text and semantic design.
 */
export function recommendationOutcome(rowId,run) {
  if(!run)return {label:'No recommendation',design:'Neutral'};
  if(!run.row_ids?.includes(rowId))return {label:'Outside recommendation',design:'Neutral'};
  if(run.status!=='completed')return {label:'Pending recommendation',design:'Information'};
  if(run.result?.selected?.some(row=>row.row_id===rowId))return {label:'Recommended',design:'Positive'};
  if(run.result?.pre_excluded?.some(row=>row.row_id===rowId))return {label:'Screened out',design:'Critical'};
  return {label:'Not recommended',design:'Neutral'};
}
