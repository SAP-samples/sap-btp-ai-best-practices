/**
 * Line-item table of the advice review pane, editable in place.
 *
 * Each editable cell is a native `contenteditable="plaintext-only"` element. Typing only records an
 * unsaved draft (kept in this module, so the 4-second inbox refresh and tab switches do not lose it);
 * nothing reaches the server until "Accept Changes" sends every draft as one atomic batch
 * (POST /edits). "Undo Changes" discards drafts and, when corrections were already accepted,
 * restores the values produced by extraction and interpretation (POST /revert).
 */
import { node, value, itemList, button } from "./dom.js";

/** Displayed columns: [result field, heading]. Every one of them is editable. */
const columns = [["invoice_reference","Invoice ref"],["customer_document_reference","Doc ref"],["customer_account","Customer account"],["invoice_date","Invoice date"],
  ["gross_amount","Gross"],["discount_amount","Discount"],["net_amount","Net"],["document_nature","Nature"],
  ["reason_code","Reason code"],["flags","Flags"],["rationale","Rationale"]];
const amountFields = new Set(["gross_amount","discount_amount","net_amount"]);
/** Unsaved edits: advice id -> Map("rowId|field" -> {text, base}); base is the value the edit started from. */
const tableDrafts = new Map();

/** Split flag text into one trimmed flag per non-empty line. */
function flagList(text) { return text.split("\n").map(flag => flag.trim()).filter(Boolean); }
/** Canonical editable text of one cell, used both for display and for change detection. */
function cellText(line, field) { return field === "flags" ? (line.flags || []).join("\n") : value(line[field]); }
/** Read what the reviewer typed, normalised the same way as cellText. */
function typedText(cell, field) { return field === "flags" ? flagList(cell.innerText).join("\n") : cell.innerText.trim(); }

/** Convert draft text to the typed value the API expects; throws on a non-numeric amount. */
function typedValue(field, text, label) {
  if (field === "flags") return flagList(text);
  if (!text) return null;
  if (!amountFields.has(field)) return text;
  const number = Number(text);
  if (!Number.isFinite(number)) throw new Error(label + ': "' + text + '" is not a number');
  return number;
}

/** Fill one cell: flags render as a bullet list, everything else as plain text. */
function fillCell(cell, field, text) {
  cell.replaceChildren();
  if (field === "flags") { const flags = flagList(text); if (flags.length) cell.append(itemList(flags)); }
  else cell.textContent = text;
}

/**
 * Render the line-item table into `host` and, when editable, Accept/Undo buttons into `actions`.
 *
 * Inputs: `advice` (persisted advice with id, revision, result, corrections), `result` (the result to
 * display), `editable` (false while processing, posted or failed, or when only the raw extraction
 * exists), and handlers `{post(path, body), rerender(), onError(error)}`.
 * Output: none; DOM is appended to `host` and `actions`.
 */
export function renderAdviceTable(host, actions, advice, result, {editable, post, rerender, onError}) {
  const lines = result?.line_items || [];
  const drafts = tableDrafts.get(advice.id) || new Map();
  const scroll = node("div", "", "inbox-table-scroll"); scroll.tabIndex = 0; scroll.setAttribute("aria-label","Extracted line items, horizontally scrollable");
  const table = node("table", "", "advice-lines" + (editable ? " editable" : "")), thead = node("thead"), headings = node("tr"), tbody = node("tbody");
  columns.forEach(([,label]) => headings.append(node("th",label))); thead.append(headings);

  let accept = null, undo = null;
  /** Enable Accept only with unsaved edits; Undo also when accepted corrections exist. */
  function syncButtons() {
    if (!accept) return;
    accept.disabled = !drafts.size;
    undo.disabled = !drafts.size && !advice.corrections?.length;
  }
  /** Record or clear one cell's draft after the reviewer types in it. */
  function recordDraft(cell, key, field, base) {
    const text = typedText(cell, field);
    if (text === base) drafts.delete(key); else drafts.set(key, {text, base});
    if (drafts.size) tableDrafts.set(advice.id, drafts); else tableDrafts.delete(advice.id);
    cell.className = (field === "flags" ? "flags-cell" : "") + (drafts.has(key) ? " changed" : "");
    syncButtons();
  }

  for (const line of lines) {
    const tr = node("tr", "", line.flags?.length ? "flagged" : "");
    tr.dataset.rowId = line.row_id;
    columns.forEach(([field, label]) => {
      const key = line.row_id + "|" + field, base = cellText(line, field), draft = drafts.get(key);
      const cell = node("td", "", (field === "flags" ? "flags-cell" : "") + (draft ? " changed" : ""));
      fillCell(cell, field, draft ? draft.text : base);
      // Show which documented customer rule decided the account (e.g. "Rg equals 'A' (priority 3)").
      if (field === "customer_account" && line.customer_account_rule) cell.title = "Rule: " + line.customer_account_rule;
      if (editable && line.row_id) {
        cell.contentEditable = "plaintext-only";
        cell.setAttribute("aria-label", label + " of " + (line.invoice_reference || "row"));
        cell.oninput = () => recordDraft(cell, key, field, base);
        // Enter finishes a single-value cell; in Flags it starts the next flag.
        cell.onkeydown = event => { if (event.key === "Enter" && field !== "flags") { event.preventDefault(); cell.blur(); } };
      }
      tr.append(cell);
    });
    tbody.append(tr);
  }
  table.append(thead,tbody); scroll.append(table);
  if (editable && lines.length) host.append(node("p", "Click any cell to correct it, then Accept Changes. Undo Changes restores the original values."));
  host.append(scroll);
  if (!editable) return;

  /** Send every draft as one batch bound to the displayed revision; drop drafts the server overtook. */
  async function acceptChanges() {
    const current = new Map(lines.map(line => [line.row_id, line]));
    const labels = Object.fromEntries(columns);
    const stale = [...drafts].filter(([key, {base}]) => {
      const [rowId, field] = key.split("|");
      return !current.has(rowId) || cellText(current.get(rowId), field) !== base;
    });
    if (stale.length) {
      stale.forEach(([key]) => drafts.delete(key));
      if (!drafts.size) tableDrafts.delete(advice.id);
      await rerender();
      throw new Error(stale.length + " edited cell(s) changed on the server since you edited them and were discarded; review and re-enter them.");
    }
    const edits = [...drafts].map(([key, {text}]) => {
      const [target, field] = key.split("|");
      return {target, field, value: typedValue(field, text, labels[field] + " of " + (current.get(target).invoice_reference || "row"))};
    });
    await post("/edits", {revision:advice.revision, edits});
    tableDrafts.delete(advice.id);
    await rerender();
  }
  /** Discard drafts; with accepted corrections, confirm and restore the original values on the server. */
  async function undoChanges() {
    if (advice.corrections?.length) {
      if (!window.confirm("Discard unsaved edits and restore every value to the original extraction? All accepted corrections on this advice are removed; the audit history is kept.")) return;
      tableDrafts.delete(advice.id);
      await post("/revert", {revision:advice.revision});
    } else tableDrafts.delete(advice.id);
    await rerender();
  }
  const group = node("div", "", "table-edit-actions");
  accept = button("Accept Changes", acceptChanges, onError, "primary");
  undo = button("Undo Changes", undoChanges, onError);
  group.append(accept, undo); actions.append(group);
  syncButtons();
}
