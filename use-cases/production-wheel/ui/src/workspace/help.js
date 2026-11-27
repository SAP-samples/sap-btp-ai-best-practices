import { request } from "../services/api.js";
import { el } from "./dom.js";

/**
 * Attach accessible help buttons without discarding controls or supporting content.
 * @param {Element} target Mounted page containing a .configuration section.
 * @param {string} endpoint API path returning help entries keyed by control ID.
 * @returns {Promise<void>} Resolves after decorating the connected page's fields.
 */
export async function mountConfigurationHelp(target, endpoint = "/api/configuration-help") {
  const owner = target.querySelector(".configuration");
  const guide = await request(endpoint);
  if (!owner?.isConnected) return;
  const dialog = el("dialog", null, "configuration-help");
  const title = el("h2");
  title.id = "configuration-help-title";
  dialog.setAttribute("aria-labelledby", title.id);
  const content = el("div");
  const close = el("button", "Close", "history-action");
  close.addEventListener("click", () => dialog.close());
  dialog.append(title, content, close);
  target.append(dialog);
  for (const [id, entry] of Object.entries(guide)) {
    const control = target.querySelector(`#${id}`);
    if (!control) continue;
    const button = el("button", "ⓘ", "info-button");
    button.type = "button";
    button.setAttribute("aria-label", `About ${entry.title}`);
    button.setAttribute("aria-haspopup", "dialog");
    button.addEventListener("click", (event) => {
      event.preventDefault();
      title.textContent = entry.title;
      content.replaceChildren(el("p", entry.text));
      for (const [field, text] of Object.entries(entry.fields || {})) {
        content.append(el("h3", field), el("p", text));
      }
      dialog.showModal();
    });
    const originalLabel = control.closest("label");
    // Section headings (no wrapping label) get the info button inline.
    if (!originalLabel) {
      control.append(" ", button);
      continue;
    }
    const field = el("div", null, "setting-field");
    const heading = el("div", null, "setting-heading");
    const labelNodes = [...originalLabel.childNodes];
    // A checkbox can precede its label text, so read all direct text nodes.
    const label = el("label", labelNodes
      .filter((node) => node.nodeType === Node.TEXT_NODE)
      .map((node) => node.textContent).join("").trim());
    label.htmlFor = control.id;
    heading.append(label, button);
    originalLabel.replaceWith(field);
    // Move every element, including hints such as #matrix-default. Keeping the
    // original nodes also preserves control state and already-attached listeners.
    field.append(heading, ...labelNodes.filter((node) => node.nodeType !== Node.TEXT_NODE));
  }
}
