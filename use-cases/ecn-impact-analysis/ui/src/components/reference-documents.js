import { validity } from "./material-dialog.js";
import { element } from "./workspace-dom.js";
import { createLatestRequestGate } from "../modules/request-generation.js";
import { buildPlantNameIndex, plantLabel } from "../modules/plant-display.js";

/**
 * Reference documents: the ECN's own document records and the BOM rows used,
 * one line per record.
 *
 * "ECN documents" are the resulting objects of the ECN JSON (part records,
 * drawings, CAD files with their revisions). They are references to the
 * document versions the ECN released, not the documents themselves.
 */
export const REFERENCE_SECTIONS = Object.freeze([
  ["added_materials", "Released ECN Records"],
  ["ebom_matches", "EBOM Evidence"],
  ["mbom_matches", "MBOM Evidence"],
]);

/** Build one compact line from a business evidence record. */
export function referenceLine(item, names = buildPlantNameIndex([item])) {
  const source = item.sheet
    ? [item.file_name, item.sheet, item.row != null ? `row ${item.row}` : null].filter(Boolean).join(" · ")
    : (item.sources || []).map(s => [s.file_name, s.sheet_name, s.source_row_number ? `row ${s.source_row_number}` : null].filter(Boolean).join(" · ")).filter(Boolean)[0];
  return [
    item.number || item.material_number || item.label,
    item.types?.join(" / "),
    item.description,
    item.revision && `revision ${item.revision}`,
    item.state,
    item.parent && `in ${item.parent}`,
    item.position && `position ${item.position}`,
    item.quantity != null && `qty ${item.quantity}`,
    item.effectivity || (item.valid_from || item.valid_to ? `${validity(item.valid_from)} – ${validity(item.valid_to)}` : null),
    (item.plant_code || item.plant) && `plant ${plantLabel(item.plant_code || item.plant, names)}`,
    source,
  ].filter(Boolean).join(" · ");
}

/**
 * Mount lazily loaded, paginated sections into `host`.
 *
 * Inputs: host element, `listBusinessEvidence(id, section, options)`, the ECN
 * row ID and the page abort signal (also removes listeners). Returns nothing.
 */
export function mountReferenceDocuments(host, listBusinessEvidence, rowId, signal, snapshotRequest = null) {
  host.replaceChildren();
  for (const [key, title] of REFERENCE_SECTIONS) {
    const panel = element("details");
    panel.dataset.evidenceSection = key;
    const summary = element("summary");
    summary.append(element("strong", title));
    const list = element("ul", "", "reference-lines");
    const pages = element("div", "", "evidence-pagination");
    const previous = element("button", "Previous"); previous.type = "button"; previous.disabled = true;
    const label = element("span");
    const next = element("button", "Next"); next.type = "button"; next.disabled = true;
    pages.append(previous, label, next);
    panel.append(summary, list, pages);
    const gate = createLatestRequestGate(signal);
    let page = 1, loaded = false;
    const load = async (target) => {
      const request = gate.next();
      list.replaceChildren(element("li", "Loading…"));
      try {
        const snapshot = snapshotRequest ? await snapshotRequest.catch(() => null) : null;
        let result;
        if (snapshot?.reference_documents) {
          const records = key === "added_materials" ? snapshot.reference_documents
            : (snapshot.programs || []).flatMap(program => program[key === "ebom_matches" ? "ebom" : "mbom"]);
          result = { items: records.slice((target - 1) * 25, target * 25), page: target,
            total_items: records.length, total_pages: Math.max(1, Math.ceil(records.length / 25)) };
        } else {
          result = await listBusinessEvidence(rowId, key, { page: target, pageSize: 25, includeTechnical: false, signal: request.signal });
        }
        if (!request.isCurrent()) return;
        page = result.page; loaded = true;
        const names = buildPlantNameIndex([...(snapshot?.plant_impact?.plants || []), ...result.items]);
        list.replaceChildren(...(result.items.length ? result.items.map(item => element("li", referenceLine(item, names))) : [element("li", "No records in this section.")]));
        previous.disabled = result.page <= 1;
        next.disabled = result.page >= result.total_pages;
        label.textContent = result.total_pages > 1 ? `Page ${result.page} of ${result.total_pages}` : "";
      } catch (error) {
        if (error.name !== "AbortError" && request.isCurrent()) list.replaceChildren(element("li", error.message, "evidence-error"));
      }
    };
    panel.addEventListener("toggle", () => { if (panel.open && !loaded) void load(1); }, { signal });
    previous.addEventListener("click", () => void load(page - 1), { signal });
    next.addEventListener("click", () => void load(page + 1), { signal });
    host.append(panel);
  }
}
