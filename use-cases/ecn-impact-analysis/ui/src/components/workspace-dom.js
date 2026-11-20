/** Create a safe text node element; source data is never interpreted as markup. */
export function element(tag, text = "", className = "") {
  const node = document.createElement(tag); node.textContent = text; node.className = className; return node;
}
/** Create an accessible action with a plain-text label. */
export function button(label, action) {
  const node = element("button", label); node.type = "button"; node.addEventListener("click", action); return node;
}
/** Render parallel business data using safe text-only table cells. */
export function table(headers, rows) {
  const wrap = element("div", "", "table-scroll"); const node = element("table");
  const head = element("thead"); const hr = element("tr");
  headers.forEach(h => hr.append(element("th", h))); head.append(hr);
  const body = element("tbody");
  rows.forEach(values => { const row = element("tr"); values.forEach((value, index) => { const cell = element("td"); cell.dataset.label = headers[index]; if (value instanceof Node) cell.append(value); else cell.textContent = value ?? "Unavailable"; row.append(cell); }); body.append(row); });
  node.append(head, body); wrap.append(node); return wrap;
}
