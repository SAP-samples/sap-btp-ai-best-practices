/** UI5 table binding with key-based selection and safe text-only invoice rendering. */
import '@ui5/webcomponents/dist/Table.js';
import '@ui5/webcomponents/dist/TableHeaderRow.js';
import '@ui5/webcomponents/dist/TableHeaderCell.js';
import '@ui5/webcomponents/dist/TableRow.js';
import '@ui5/webcomponents/dist/TableCell.js';
import '@ui5/webcomponents/dist/TableSelectionMulti.js';
import '@ui5/webcomponents/dist/Tag.js';
import '@ui5/webcomponents/dist/Button.js';
import {formatInvoiceDate,invoiceAmount,recommendationOutcome} from './invoice-display.js';

export {invoiceAmount} from './invoice-display.js';

/** Mount a real UI5 table, retaining listeners until explicit route destruction. */
export function mountInvoiceTable(host, {onSelection, onInspect, onPage}) {
  host.innerHTML = `<div class="ws-table-scroll"><ui5-table accessible-name="Offer invoices" overflow-mode="Scroll">
    <ui5-table-header-row slot="headerRow" sticky></ui5-table-header-row>
    <ui5-table-selection-multi slot="features"></ui5-table-selection-multi>
  </ui5-table></div><div class="ws-table-footer"><span data-page-label></span><div>
  <ui5-button data-prev design="Transparent" icon="navigation-left-arrow" accessible-name="Previous invoice page"></ui5-button>
  <ui5-button data-next design="Transparent" icon="navigation-right-arrow" accessible-name="Next invoice page"></ui5-button></div></div>`;
  const table = host.querySelector('ui5-table');
  const header = host.querySelector('ui5-table-header-row');
  const selection = host.querySelector('ui5-table-selection-multi');
  const controller = new AbortController();
  const options = {signal:controller.signal};
  let pageRows = [];
  let activeRun = null;
  for (const label of ['Eligibility', 'Customer', 'Seller', 'Invoice reference', 'Amount', 'Due date', 'Recommendation', 'Details']) {
    const cell = document.createElement('ui5-table-header-cell');
    cell.textContent = label;
    header.append(cell);
  }
  selection.addEventListener('change', () => onSelection(pageRows.map(row => row.row_id), selection.getSelectedAsSet()), options);
  host.querySelector('[data-prev]').addEventListener('click', () => onPage(-1), options);
  host.querySelector('[data-next]').addEventListener('click', () => onPage(1), options);
  /** Render a server page, using source IDs even when invoice references repeat. */
  function render(page, selectedIds, offset=0, pageSize=50) {
    pageRows = page.items;
    table.querySelectorAll('ui5-table-row').forEach(row => row.remove());
    for (const row of pageRows) {
      const element = document.createElement('ui5-table-row');
      element.rowKey = row.row_id;
      const eligibility = document.createElement('ui5-tag');
      eligibility.design = row.eligible ? 'Positive' : 'Critical';
      eligibility.textContent = row.eligibility_source === 'upstream' ? 'Approved upstream' : row.eligible ? 'Eligible' : 'Not eligible';
      const invoice = row.invoice;
      const inspect = document.createElement('ui5-button');
      inspect.design = 'Transparent';
      inspect.icon = 'inspect';
      inspect.textContent = 'View';
      inspect.accessibleName = `Inspect invoice ${invoice.invoice_ref}`;
      inspect.addEventListener('click', () => onInspect(row), {once:false});
      const fields = [eligibility, invoice.debtor_name || invoice.debtor_id,
        invoice.seller_name || invoice.seller_id, invoice.invoice_ref, invoiceAmount(invoice),
        formatInvoiceDate(invoice.due_date), outcomeTag(row.row_id), inspect];
      for (const value of fields) {
        const cell = document.createElement('ui5-table-cell');
        if (value instanceof Node) cell.append(value); else cell.textContent = value;
        element.append(cell);
      }
      table.append(element);
    }
    selection.setSelectedAsSet(new Set(pageRows.filter(row => selectedIds.has(row.row_id)).map(row => row.row_id)));
    host.querySelector('[data-page-label]').textContent = page.total ?
      `${offset + 1}–${Math.min(offset + pageSize, page.total)} of ${page.total} invoices` : 'No matching invoices';
    host.querySelector('[data-prev]').disabled = offset === 0;
    host.querySelector('[data-next]').disabled = offset + pageSize >= page.total;
  }
  /** Keep saved recommendation membership separate from eligibility and display filters. */
  function outcomeTag(id) {
    const outcome=recommendationOutcome(id,activeRun);
    const tag=document.createElement('ui5-tag');tag.design=outcome.design;tag.textContent=outcome.label;return tag;
  }
  /** Refresh recommendation tags without changing the user's row selection. */
  function setRun(run) {
    activeRun=run;
    table.querySelectorAll('ui5-table-row').forEach(row=>{row.querySelectorAll('ui5-table-cell')[6].replaceChildren(outcomeTag(row.rowKey));});
  }
  /** Release listeners and rendered rows on route changes. */
  function destroy() { controller.abort(); host.replaceChildren(); }
  return {render, destroy, setRun};
}
