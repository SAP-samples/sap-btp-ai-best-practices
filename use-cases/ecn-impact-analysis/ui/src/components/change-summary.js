import { button, element } from "./workspace-dom.js";
import { readable, validity } from "./material-dialog.js";
import { plantsLabel } from "./plant-impact-table.js";

const ACTION_LABELS = {
  PHASE_IN: "New part", PHASE_OUT: "Phase-out", REVISION: "Revised", REVISED: "Revised",
  RELEASE: "Released", UNCHANGED: "Unchanged", REFERENCED: "Referenced", REVIEW: "Needs review", UNRESOLVED: "Unresolved",
};

// Deferred decision-irrelevant chart mappings. Keep the palette and helper
// ready for a future analytics view without loading ECharts for this page.
if (false) {
const ACTION_COLORS = {
  PHASE_IN: C.newPart,       // blue
  PHASE_OUT: C.phaseOut,     // red
  REVISION: C.revised,       // brown
  REVISED: C.revised,        // brown (legacy alias)
  RELEASE: C.positive,       // green
  UNCHANGED: C.referenced,   // grey
  REFERENCED: C.neutral,     // slate
  REVIEW: "#e9730c",         // orange — deliberately distinct from Revised brown
  UNRESOLVED: "#7858a8",     // purple
};
const FALLBACK_COLORS = ["#0a6ed1", "#188918", "#df6e0c", "#5d36ff", "#0f828f", "#c0399f"];

/** Deterministic palette colour for an unmapped label, stable across ECNs. */
function fallbackColor(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  return FALLBACK_COLORS[hash % FALLBACK_COLORS.length];
}

/** Count parts per change action and map each to a labelled, coloured slice. */
function donutData(items) {
  const byAction = {};
  for (const item of items) byAction[item.action] = (byAction[item.action] || 0) + 1;
  return Object.entries(byAction).map(([action, value]) => {
    const name = ACTION_LABELS[action] || readable(action);
    return { name, value, color: ACTION_COLORS[action] || fallbackColor(name) };
  });
}
}

/**
 * Expandable lists under the Change summary card.
 *
 * Inputs:
 *   host: element to render into.
 *   snapshot: assessment snapshot (counts, programs, questions).
 *   changes: ECN part actions from GET /changes (may be empty for old snapshots).
 *   attentionHref: link to the "What needs attention" section.
 *   inspect(number): opens one part.
 * Output: host re-rendered; nothing is returned.
 */
export function renderChangeLists(host, { snapshot, changes = [], attentionHref, inspect }) {
  host.replaceChildren();

  const items = (snapshot.parts?.length ? snapshot.parts : changes) || [];
  const attention = snapshot.counts.attention_items ?? snapshot.questions?.length ?? 0;

  // Deferred: a distribution chart describes record volume, not the ECN decision.
  // Keep the chart helpers and action mapping intact for a future analytics view.
  // const data = donutData(items);
  // if (data.length) { ...initChart(donutHost, partsDonutOption(data, total ?? items.length)); }

  host.append(technicalObjects(snapshot, items, inspect), bomMatches(snapshot.programs || [], inspect));

  // What-needs-attention call to action.
  const row = element("p", "", "attention-count");
  const link = element("a", `Items needing attention: ${attention}`);
  link.href = attentionHref;
  row.append(link);
  host.append(row);
}

/** One line per ECN part with its action and revision; count = snapshot technical objects. */
function technicalObjects(snapshot, changes, inspect) {
  const panel = element("details", "", "summary-list");
  panel.append(element("summary", `Parts in This ECN (${snapshot.counts.technical_objects})`));
  panel.append(element("p", "Every component on this ECN with its change action, revision, and classification. Select a component to open its source records.", "helper-text"));
  const list = element("ul");
  for (const change of changes) {
    const item = element("li");
    const description = change.description || change.number;
    item.append(button(description, () => void inspect(change.number)));
    const revision = change.revision_from && change.revision_to && change.revision_from !== change.revision_to
      ? ` · revision ${change.revision_from} → ${change.revision_to}` : "";
    const action = ACTION_LABELS[change.action] || readable(change.action);
    const version = revision || (change.revisions?.length ? ` · revision ${change.revisions.join(" / ")}` : "");
    item.append(document.createTextNode(` · ${action}${version}`));
    if (change.types?.length) item.append(document.createTextNode(` · ${change.types.join(" / ")}`));
    if (change.counterparts?.length) item.append(document.createTextNode(` · paired with ${change.counterparts.join(", ")}`));
    list.append(item);
  }
  panel.append(changes.length ? list : element("p", "Part actions are not available for this assessment.", "helper-text"));
  return panel;
}

/** BOM rows of the ECN parts, split by EBOM/MBOM, for one selected program (model range). */
function bomMatches(programs, inspect) {
  const total = programs.reduce((sum, program) => sum + program.ebom.length + program.mbom.length, 0);
  const panel = element("details", "", "summary-list");
  panel.append(element("summary", `BOM Matches (${total})`));
  if (!programs.length) {
    panel.append(element("p", "No program is declared on this ECN.", "helper-text"));
    return panel;
  }
  panel.append(element("p", "Where these parts appear in the engineering (EBOM) and manufacturing (MBOM) bills of material. Pick a model range to switch programs.", "helper-text"));
  const select = element("select");
  select.setAttribute("aria-label", "Program (model range)");
  for (const [index, program] of programs.entries()) {
    const rows = program.ebom.length + program.mbom.length;
    const option = element("option", program.bom_uploaded ? `${program.model_range} (${rows} rows)` : `${program.model_range} (no BOM uploaded)`);
    option.value = String(index);
    select.append(option);
  }
  const label = element("label", "Program ");
  label.append(select);
  const body = element("div");
  const render = () => {
    const program = programs[Number(select.value)];
    body.replaceChildren();
    if (!program.bom_uploaded) {
      body.append(element("p", `No BOM uploaded for ${program.model_range}. Publish its BOM in Settings and reassess.`, "helper-text"));
      return;
    }
    for (const [key, title] of [["ebom", "Engineering BOM (EBOM)"], ["mbom", "Manufacturing BOM (MBOM)"]]) {
      body.append(element("h4", `${title}: ${program[key].length}`));
      const list = element("ul");
      for (const row of program[key]) {
        const item = element("li");
        item.append(button(row.number, () => void inspect(row.number)));
        const facts = [
          row.parent && `in ${row.parent.split("_")[0]}`,
          row.position && `position ${row.position}`,
          row.quantity && `qty ${row.quantity}`,
          `${validity(row.valid_from)} – ${validity(row.valid_to)}`,
          row.plants?.length && `plants ${plantsLabel(row.plants)}`,
        ].filter(Boolean).join(" · ");
        item.append(document.createTextNode(` · ${facts}`));
        list.append(item);
      }
      body.append(program[key].length ? list : element("p", "No rows.", "helper-text"));
    }
  };
  select.addEventListener("change", render);
  panel.append(label, body);
  render();
  return panel;
}
