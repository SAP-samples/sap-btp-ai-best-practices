/** Inert DOM helpers shared by the inbox page and its panels; source values only ever reach textContent. */

/** Construct an inert DOM node; source values always go through textContent. */
export function node(tag, text = "", cls = "") {
  const element = document.createElement(tag);
  element.textContent = text ?? "";
  element.className = cls;
  return element;
}
/** Render readable primitive or structured field values. */
export function value(v) { return v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v); }
/** Render one object as a semantic key/value list without activating source markup. */
export function keyValueList(items) {
  const list = node("ul", "", "structured-list");
  Object.entries(items || {}).forEach(([key, item]) => {
    const entry = node("li");
    entry.append(node("span", key.replaceAll("_", " "), "structured-label"), node("span", value(item)));
    list.append(entry);
  });
  return list;
}
/** Render non-empty strings as a compact semantic list. */
export function itemList(items) {
  const list = node("ul", "", "structured-list");
  (items || []).forEach(item => list.append(node("li", value(item))));
  return list;
}
/** Create a button with a safely caught asynchronous action. */
export function button(text, action, onError, cls = "") {
  const b = node("button", text, cls); b.type = "button";
  b.addEventListener("click", async () => {
    b.disabled = true;
    try { await action(); } catch (error) { onError(error); } finally { b.disabled = false; }
  });
  return b;
}
