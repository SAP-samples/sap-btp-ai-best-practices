import * as api from "../services/workspace-api.js";
import { button, element, table } from "./workspace-dom.js";
import { materialDialog } from "./material-dialog.js";
import { renderReadiness } from "./readiness-list.js";
import { setChatContext } from "../modules/shell-chat.js";
import { mountAuditLog } from "./audit-log.js";

/**
 * Mount the snapshot-backed parts of one ECN page into their view hosts.
 *
 * Inputs:
 *   root: the ECN page element; hosts are found by ID (#object-inventory,
 *     #audit-log, #snapshot-followups). Missing hosts are skipped.
 *   id: ECN row ID. signal: page abort signal.
 *   options.graphReady: whether chat may send questions.
 *   options.snapshotRequest: shared snapshot promise (fetched here if absent).
 *   options.showSourceData(): opens the source tools in the Audit tab.
 * Returns { inspect, ready, reviewApi, updateAnalysisDescriptions }:
 *   `inspect(number)` opens a part,
 *   `ready` settles after the first visible render,
 *   `reviewApi` is a Promise<{accept, reject, setReviewerName}|null> for the footer.
 */
export function snapshotWorkspace(root, id, signal, { graphReady = true, snapshotRequest, showSourceData = () => {}, auditDetail = null } = {}) {
  snapshotRequest ||= api.getSnapshot(id, signal);
  const error = element("p"); error.setAttribute("role", "alert");
  const inventory = element("section", "", "surface");
  inventory.append(element("h2", "Parts and Documents in This ECN"));
  const rows = element("div"); const pageLabel = element("span");
  const search = element("input"); search.placeholder = "Identifier or description"; search.setAttribute("aria-label", "Search referenced objects");
  const kind = element("select"); kind.setAttribute("aria-label", "Object category");
  [["true", "Parts referenced in the ECN"], ["false", "Documents and other objects"], ["bom", "BOM rows carrying this ECN"]].forEach(([value, label]) => { const option = element("option", label); option.value = value; kind.append(option); });
  let page = 1, ancestor = null, readNumber = 0, browserLoaded = false;
  const ancestorLabel = element("p", "", "helper-text");
  const resetAncestor = button("Show all parts", () => { ancestor = null; page = 1; void load(); }); resetAncestor.hidden = true;
  /** Page objects without changing their source types, roles or observed levels. */
  async function load() {
    browserLoaded = true;
    const sequence = ++readNumber;
    try {
      const params = { page, page_size: 25, query: search.value, technical: kind.value !== "false", origin: kind.value === "bom" ? "bom" : "ecn" }; if (ancestor) params.ancestor = ancestor;
      const data = await api.listObjects(id, params, signal); if (signal.aborted || sequence !== readNumber) return;
      rows.replaceChildren(table(["Object", "Description", "Type", "Role in ECN", "BOM levels", "BOM rows"], data.items.map(o => [
        button(o.number || "Unnumbered object", () => void inspect(o.number || o.object_key, o.technical)), o.description,
        o.types.join(", "), o.roles.map(r => ({ RESULTING: "Released by ECN", AFFECTED: "Changed by ECN", BOM_ECN_REFERENCE: "BOM carries ECN" }[r] || r)).join(" / "),
        o.observed_levels.map(l => ({ "1": "Main", "2": "Sub", "3": "VC" }[l] || `Level ${l}`)).join(", ") || "Not found in BOM",
        o.occurrence_count,
      ])));
      pageLabel.textContent = `Page ${page} · ${data.total} objects`; previous.disabled = page <= 1; next.disabled = page * 25 >= data.total;
      ancestorLabel.textContent = ancestor ? `Parts found under ${ancestor}. Other parts below it are not marked as changed.` : "";
      resetAncestor.hidden = !ancestor; error.textContent = "";
    } catch (failure) { if (failure.name !== "AbortError") error.textContent = failure.message; }
  }
  const previous = button("Previous objects", () => { page--; void load(); }); const next = button("Next objects", () => { page++; void load(); });
  const controls = element("div", "", "workspace-controls"); controls.append(search, kind, button("Search objects", () => { page = 1; void load(); }));
  kind.addEventListener("change", () => { page = 1; ancestor = null; void load(); });
  const pages = element("nav", "", "pagination"); pages.append(previous, pageLabel, next);
  inventory.append(controls, ancestorLabel, resetAncestor, error, rows, pages);
  root.querySelector("#object-inventory")?.append(inventory);

  const browser = root.querySelector("#object-browser");
  browser?.addEventListener("toggle", () => { if (!browserLoaded) void load(); }, { signal });

  const inspect = materialDialog(id, signal, failure => { error.textContent = failure.message; });

  setChatContext({ id, signal, inspect, graphReady });

  const auditHost = root.querySelector("#audit-log");
  const audit = auditHost
    ? mountAuditLog(auditHost, id, snapshotRequest, signal, auditDetail, root.querySelector(".detail-badges"))
    : null;
  const reviewApi = Promise.resolve(audit?.reviewApi || null);

  const followups = root.querySelector("#snapshot-followups");
  let readySnapshot = null;
  let descriptions = [];

  const showDatasets = () => {
    showSourceData();
    const data = root.querySelector("#assessment-inputs");
    if (data) {
      data.open = true;
      data.scrollIntoView?.({ block: "start", behavior: "smooth" });
    }
  };

  const ready = (async () => {
    try {
      const snapshot = await snapshotRequest; if (signal.aborted) return;
      if (snapshot.datasets?.some(d => d.environment === "SYNTHETIC")) {
        // Insert synthetic data notice before the KPI header
        const kpiHost = root.querySelector("#ecn-kpi-header");
        kpiHost?.before(element("p", "Synthetic demonstration assessment", "snapshot-notice semantic-pill tone-neutral"));
      }
      readySnapshot = snapshot;
      if (followups) renderReadiness(followups, snapshot, inspect, showDatasets, descriptions);
      await reviewApi;
    } catch (failure) {
      if (failure.name !== "AbortError") {
        error.textContent = failure.message;
        followups?.replaceChildren(element("p", "Assessment findings are unavailable. Reload to try again.", "evidence-error"));
      }
    }
  })();

  return {
    inspect,
    ready,
    reviewApi,
    updateAuditDetail(detail) { audit?.updateDetail(detail); },
    /** Replace fixed finding explanations with AI descriptions once available. */
    updateAnalysisDescriptions(values) {
      descriptions = values || [];
      if (readySnapshot && followups) renderReadiness(followups, readySnapshot, inspect, showDatasets, descriptions);
    },
  };
}
