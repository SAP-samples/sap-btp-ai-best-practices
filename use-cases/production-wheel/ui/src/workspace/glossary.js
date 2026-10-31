import { el } from "./dom.js";
import { fieldLabel, fieldDescription } from "./vocabulary.js";

/** Render accessible column definitions for the current view using safe text nodes. */
export function renderGlossary(target, columns) {
  const list = el("dl", null, "column-glossary");
  for (const column of columns) {
    list.append(el("dt", fieldLabel(column)));
    list.append(el("dd", fieldDescription(column) || "Source or calculated field; consult the extraction evidence for its context."));
  }
  target.replaceChildren(
    el("p", "FINI = finished product; SEFI = semi-finished product; PCK = packaging code; PV = production version; DC = distribution centre. Baseline refers to the reference grouping used for comparison."),
    list,
  );
}
