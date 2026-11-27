import { MATRIX_MODE_LABELS } from "./vocabulary.js";

/**
 * Create a DOM element and assign untrusted content only through textContent.
 *
 * The JSDoc types let `npm run typecheck` know the exact element type, so
 * mistakes such as chaining on `append()` (which returns undefined) are caught.
 *
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tag Element tag name, for example "table" or "p".
 * @param {unknown} [text] Optional text; objects are shown as JSON.
 * @param {string} [className] Optional CSS class list.
 * @returns {HTMLElementTagNameMap[K]} The new, detached element.
 */
export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null)
    node.textContent =
      typeof text === "object" ? JSON.stringify(text) : String(text);
  if (className) node.className = className;
  return node;
}

/**
 * Build a read-only table with a header row and one body row per record.
 *
 * Use it instead of assembling <thead>/<tbody> by hand: `Element.append()`
 * returns undefined, so `table.append(thead).append(row)` throws at runtime.
 *
 * @param {Array<[string, string]>} columns `[recordKey, headerLabel]` pairs, in display order.
 * @param {Array<Record<string, unknown>>} rows Records; missing values show as "—".
 * @returns {HTMLTableElement} Detached table element.
 */
export function dataTable(columns, rows) {
  const table = el("table");
  const thead = el("thead");
  const head = el("tr");
  for (const [, label] of columns) head.append(el("th", label));
  thead.append(head);
  const body = el("tbody");
  for (const record of rows) {
    const row = el("tr");
    for (const [key] of columns) row.append(el("td", display(record?.[key])));
    body.append(row);
  }
  table.append(thead, body);
  return table;
}

/** Format a value without interpreting HTML. */
export function display(value) {
  return value == null
    ? "—"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
}

/** Render JSON evidence with line breaks and safe text insertion. */
export function evidence(target, value) {
  const text = JSON.stringify(value, null, 2)
    .replace(/j_ch/gi, "Changeover")
    // Keep the stored compatibility mode visible next to its readable name.
    .replace(/"matrix_mode": "(\w+)"/g, (match, mode) => MATRIX_MODE_LABELS[mode] ? `"matrix_mode": "${mode} (${MATRIX_MODE_LABELS[mode]})"` : match);
  target.replaceChildren(el("pre", text, "evidence"));
}

/** Wrap an async event action with busy state and a shared visible error handler. */
export function action(button, callback, fail) {
  let pending = false;
  button.addEventListener("click", async () => {
    if (pending) return;
    pending = true;
    button.setAttribute("aria-busy", "true");
    try {
      await callback();
    } catch (error) {
      fail(error);
    } finally {
      pending = false;
      button.removeAttribute("aria-busy");
    }
  });
}

/** Populate a select preserving a valid selected value, using safe option text. */
export function options(select, entries, selected) {
  select.replaceChildren(
    ...entries.map(([value, label]) => {
      const option = el("option", label);
      option.value = value;
      return option;
    }),
  );
  if (entries.some(([value]) => String(value) === String(selected)))
    select.value = selected;
}

/** Re-export shared business labels for existing table and comparison consumers. */
export { fieldLabel } from "./vocabulary.js";
