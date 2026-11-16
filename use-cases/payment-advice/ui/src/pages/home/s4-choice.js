/**
 * Helpers of the "S/4 posting" tab that explain how each advice line was matched to an S/4 document
 * and let the reviewer choose the document when the check could not decide on its own.
 *
 * Every key read here is optional: an advice checked before these keys existed renders exactly as before
 * (empty "Open in S/4" / "How it was matched" cells, no overview list, no choice tables).
 */
import { node, value, keyValueList, button } from "./dom.js";

/** S/4 field that found the document -> short wording for the reviewer. */
const foundByLabels = {DocumentReferenceID: "Reference", AccountingDocument: "Document number", BillingDocument: "Billing document"};
/** State of an S/4 document offered for a choice -> wording used in the lines table too. */
const documentStates = {open: "Open", cleared: "Already cleared", reversed: "Reversed in S/4"};
/** Rule the check used to narrow several S/4 documents down to one -> plain words. */
const ruleLabels = {company_code: "company code", advice_company_code: "company code of the other lines", customer: "customer",
  sign: "invoice or credit", not_reversed: "not reversed", open: "still open", normal_item: "normal item",
  field_priority: "reference before document number", open_amount: "open amount", original_amount: "document amount",
  item_amount: "item amount", document_total: "document total"};

/**
 * Explain how the check picked the S/4 document of one line ("How it was matched" column).
 * @param {object|null|undefined} resolution `line.resolution` = `{method, steps: [{rule, kept, dropped}], warning}`.
 * @returns {string} Plain sentence, or "" for an old check / a line without an S/4 document.
 */
export function howMatched(resolution) {
  let text = "";
  if (resolution?.method === "unique") text = "Only document with this reference";
  else if (resolution?.method === "reviewer") text = "Chosen by the reviewer";
  else if (resolution?.method === "auto") {
    const steps = resolution.steps || [], first = steps[0];
    // The first step started from every document S/4 returned, so kept + dropped is the number compared.
    const among = first ? " among " + (Number(first.kept || 0) + Number(first.dropped || 0)) : "";
    const rules = steps.map(step => ruleLabels[step.rule] || step.rule).join(", ");
    text = "Chosen" + among + (rules ? " by " + rules : "");
  }
  // A warning means the rules picked one document but the reviewer should still look at it.
  if (text && resolution.warning) text += " - please confirm";
  return text;
}

/**
 * Name an S/4 document the way the check messages do: "company code/document/year".
 * @param {object} item S/4 item (`company_code` is missing on a check stored before it was kept).
 * @returns {string} e.g. "1000/1800000101/2026".
 */
function documentName(item) {
  return (item.company_code ? item.company_code + "/" : "") + item.accounting_document + "/" + item.fiscal_year;
}

/**
 * Text of the "S/4 document" cell of the lines table. The company code comes first, because one advice
 * may pay documents of several company codes.
 * @param {object} line One entry of `s4.lines`.
 * @returns {string} "company code/doc/year (field)", every item when one line pays several items of a document,
 *   "deducts from <document>" for a deduction quoting one of our invoices, or "".
 */
export function documentCell(line) {
  const item = line.s4_item || {};
  if (line.s4_items?.length) {
    const documents = line.s4_items.map(part => documentName(part) +
      (part.accounting_document_item ? " item " + part.accounting_document_item : ""));
    return documents.join(", ") + " (" + line.matched_by + ")";
  }
  if (item.accounting_document) return documentName(item) + " (" + line.matched_by + ")";
  if (line.kind === "deduction" && line.related_document) return "deducts from " + line.related_document;
  return "";
}

/**
 * Match overview of the whole check, shown right after the derived header values.
 * @param {object|undefined} summary `s4.summary` (missing on an old check).
 * @returns {HTMLElement|null} Key/value list, or null when the check has no summary.
 */
export function matchOverview(summary) {
  if (!summary) return null;
  const found = summary.matched_by || {}, decided = summary.resolution || {};
  // Only the fields that actually found a document, e.g. "reference 7, document number 2".
  const matchedBy = Object.entries(found).filter(([, count]) => count)
    .map(([field, count]) => (foundByLabels[field] || field).toLowerCase() + " " + count).join(", ");
  // Matched lines per company code, e.g. "1000: 12, 2000: 3"; only worth a row when there is more than one.
  const codes = Object.entries(summary.company_codes || {});
  const spread = codes.length > 1 ? {"matched lines per company code": codes.map(([code, count]) => code + ": " + count).join(", ")} : {};
  return keyValueList({"lines matched to S/4": value(summary.matched) + " of " + value(summary.lines),
    "matched by": matchedBy || "none", ...spread, "matched without a choice": decided.unique, "chosen by rules": decided.auto,
    "chosen by the reviewer": decided.reviewer, deductions: summary.deductions, "posted as open item": summary.residuals,
    "not resolved": summary.unresolved});
}

/**
 * Notes in the issues list for lines whose S/4 document the reviewer chose, each with "Undo choice".
 * @param {object} s4 Stored S/4 check.
 * @param {object} advice Current advice (for its revision).
 * @param {{post: Function, onError: Function}} actions Mutation callback and error handler.
 * @returns {HTMLElement[]} One `li` per chosen line (none once the advice is posted).
 */
export function reviewerChoiceNotes(s4, advice, {post, onError}) {
  if (advice.status === "posted") return [];
  return (s4.lines || []).filter(line => line.resolution?.method === "reviewer").map(line => {
    const item = line.s4_item || {};
    const note = node("li", lineLabel(line) + " uses S/4 document " + documentName(item) +
      ", chosen by the reviewer.", "s4-info");
    if (line.row_id) note.append(button("Undo choice", () => post("/edit", {revision: advice.revision, target: line.row_id,
      field: "s4_document", value: null, reason: "Undo S/4 document choice"}), onError, "inline-action"));
    return note;
  });
}

/**
 * One "choose the S/4 document" block per line the check could not decide (non-empty `candidates`). A matched
 * line has candidates when its document lies in another company code and the reviewer has to confirm it.
 * @param {object} s4 Stored S/4 check.
 * @param {object} advice Current advice (revision and current rows).
 * @param {{post: Function, onError: Function}} actions Mutation callback and error handler.
 * @returns {HTMLElement[]} Sections to append under the issues list (none once the advice is posted).
 */
export function choiceBlocks(s4, advice, actions) {
  if (advice.status === "posted") return [];
  return (s4.lines || []).filter(line => line.row_id && line.candidates?.length)
    .map(line => choiceBlock(line, advice, actions));
}

/**
 * Name a line the way every message of this tab does.
 * @param {object} line Check line (`index` is zero-based).
 * @returns {string} e.g. "Line 3 (INV-1)".
 */
function lineLabel(line) {
  return "Line " + (line.index + 1) + " (" + (line.reference || "no reference") + ")";
}

/**
 * Table of the S/4 documents offered for one line, each with a "Use this document" button.
 * @param {object} line Check line with `candidates`.
 * @param {object} advice Current advice.
 * @param {{post: Function, onError: Function}} actions Mutation callback and error handler.
 * @returns {HTMLElement} Section with a heading and a scrollable table.
 */
function choiceBlock(line, advice, actions) {
  const section = node("section", "", "s4-choice"), label = lineLabel(line);
  section.append(node("h4", label + (line.kind === "matched" ? ": confirm the S/4 document" : ": choose the S/4 document")));
  const scroll = node("div", "", "inbox-table-scroll"); scroll.tabIndex = 0;
  scroll.setAttribute("aria-label", "S/4 documents to choose from for line " + (line.index + 1) + ", horizontally scrollable");
  const table = node("table", "", "s4-lines s4-choices"), head = node("tr"), body = node("tbody");
  ["Document", "Customer", "Amount", "Open in S/4", "Status", "Found by", ""].forEach(h => head.append(node("th", h)));
  // When all offered items belong to one document, only the item number tells them apart.
  const oneDocument = new Set(line.candidates.map(c => c.company_code + "/" + c.accounting_document + "/" + c.fiscal_year)).size === 1;
  for (const candidate of line.candidates) {
    const tr = node("tr"), docText = candidate.company_code + "/" + candidate.accounting_document + "/" + candidate.fiscal_year +
      (oneDocument && candidate.accounting_document_item ? " item " + candidate.accounting_document_item : "");
    const status = (documentStates[candidate.state] || candidate.state || "") + (candidate.special_gl ? ", special G/L item" : "");
    const foundBy = (foundByLabels[candidate.matched_by] || candidate.matched_by || "") + (candidate.variant ? " as " + candidate.variant : "");
    [docText, candidate.customer, amountText(candidate.amount, candidate.currency), amountText(candidate.open_amount, candidate.currency),
     status, foundBy].forEach(cell => tr.append(node("td", value(cell))));
    const cell = node("td");
    cell.append(useButton(line, candidate, docText, label, advice, actions));
    tr.append(cell); body.append(tr);
  }
  const thead = node("thead"); thead.append(head); table.append(thead, body); scroll.append(table);
  section.append(scroll);
  return section;
}

/**
 * Format an amount with its currency.
 * @param {string|null|undefined} amount Amount as text.
 * @param {string|undefined} currency Currency code.
 * @returns {string} e.g. "1250.00 EUR", or "" when the amount is missing.
 */
function amountText(amount, currency) {
  return amount == null || amount === "" ? "" : amount + " " + (currency || "");
}

/**
 * "Use this document": saves the reviewer's choice as one audited batch on the advice row. A row posted as
 * residual is switched back in the same batch, because a chosen document means the line pays that invoice.
 * @param {object} line Check line the document is chosen for (`row_id` is the edit target).
 * @param {object} candidate Offered S/4 document; `candidate.pick` is the value the server expects.
 * @param {string} docText Document as shown in the table, repeated in the confirmation.
 * @param {string} label "Line N (reference)".
 * @param {object} advice Current advice (revision and current rows).
 * @param {{post: Function, onError: Function}} actions Mutation callback and error handler.
 * @returns {HTMLButtonElement} The button.
 */
function useButton(line, candidate, docText, label, advice, {post, onError}) {
  return button("Use this document", async () => {
    if (!window.confirm("Use S/4 document " + docText + " for " + label + "? Then check against S/4 again.")) return;
    const edits = [{target: line.row_id, field: "s4_document", value: candidate.pick}];
    const row = (advice.result?.line_items || []).find(item => item.row_id === line.row_id);
    if (row?.s4_treatment === "residual") edits.push({target: line.row_id, field: "s4_treatment", value: null});
    await post("/edits", {revision: advice.revision, reason: "S/4 document chosen by the reviewer", edits});
  }, onError, "inline-action");
}
