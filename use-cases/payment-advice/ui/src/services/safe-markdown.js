/** Render Markdown through an inert template and strict tag/attribute allowlist. */
import { marked } from "marked";
export function renderSafeMarkdown(host, source) {
  const template = document.createElement("template");
  template.innerHTML = marked.parse(source);
  const allowed = new Set(["P","BR","STRONG","EM","UL","OL","LI","PRE","CODE","BLOCKQUOTE","TABLE","THEAD","TBODY","TR","TH","TD","H1","H2","H3","H4","HR"]);
  for (const element of [...template.content.querySelectorAll("*")]) {
    if (!allowed.has(element.tagName)) { element.replaceWith(document.createTextNode(element.textContent)); continue; }
    for (const attribute of [...element.attributes]) element.removeAttribute(attribute.name);
  }
  host.replaceChildren(template.content);
}
