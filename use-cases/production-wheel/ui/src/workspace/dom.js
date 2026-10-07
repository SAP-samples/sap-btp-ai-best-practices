/** Create a DOM element and assign untrusted content only through textContent. */
export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null)
    node.textContent =
      typeof text === "object" ? JSON.stringify(text) : String(text);
  if (className) node.className = className;
  return node;
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
  target.replaceChildren(el("pre", JSON.stringify(value, null, 2).replace(/j_ch/gi, "Changeover"), "evidence"));
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
