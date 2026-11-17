import { request } from "../services/api.js";
import { el } from "./dom.js";

/** Attach keyboard-accessible info buttons and a dismissible help dialog to settings. */
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
    const field = el("div", null, "setting-field");
    const heading = el("div", null, "setting-heading");
    const label = el("label", originalLabel.firstChild.textContent.trim());
    label.htmlFor = control.id;
    heading.append(label, button);
    originalLabel.replaceWith(field);
    field.append(heading, control);
  }
}
