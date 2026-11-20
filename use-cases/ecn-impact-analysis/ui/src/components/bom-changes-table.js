import { button, element } from "./workspace-dom.js";
import { plantsLabel } from "./plant-impact-table.js";
import { acceptanceMaterialAction, blockedMaterialIssues } from "./acceptance-materials.js";

const ECN_STATUS = {
  NEW: ["New", "success"],
  REMOVED: ["Removed", "error"],
  UNCHANGED: ["Unchanged", "neutral"],
};

/**
 * One compact, material-oriented table of ECN-specific BOM evidence.
 *
 * Input: host element, `snapshot.bom_changes` ({lines, materials, counts}) and
 * `inspect(number)` to open a part. Review state supplies the server acceptance
 * plan; snapshot supplies the source-backed context for required actions.
 * Output: host re-rendered; returns nothing.
 */
export function renderBomChanges(host, bomChanges, inspect, { reviewState = {}, snapshot = {} } = {}) {
  const previousScroll = host.querySelector(".table-scroll");
  const scrollTop = previousScroll?.scrollTop || 0, scrollLeft = previousScroll?.scrollLeft || 0;
  host.replaceChildren(element("h2", "BOM Change Evidence"));
  const plan = reviewState.acceptancePlan;
  const issues = new Map(blockedMaterialIssues(plan, { ...snapshot, bom_changes: bomChanges }).map(issue => [issue.number, issue]));
  const pricedMaterials = new Set((plan?.parts || []).filter(part => part.price === "HAS_PRICE").map(part => part.number));
  const exemptMaterials = new Set((plan?.parts || []).filter(part => part.price === "NOT_NEEDED").map(part => part.number));
  const summary = reviewState.decided ? "An engineering decision is already recorded."
    : reviewState.blocked ? reviewState.blockReason || "Acceptance is blocked by the current review preconditions."
      : reviewState.acceptanceStatus === "unavailable" || plan?.available === false ? "Acceptance status is unavailable."
        : "";
  if (summary) host.append(element("p", summary, "bom-acceptance-summary"));
  const lines = bomChanges?.lines || [];
  if (!lines.length) {
    host.append(element("p", "No part row in the uploaded BOMs carries this ECN's change number. Document rows are not BOM changes.", "helper-text"));
    if (!issues.size) return;
  }
  const material = (side) => {
    if (!side?.number) return "Not supplied";
    const cell = element("div", "", "bom-change-part");
    const issue = issues.get(side.number);
    const link = issue ? acceptanceMaterialAction(issue, inspect)
      : button(side.number, () => void inspect(side.number));
    (issue ? link.querySelector("button") : link).className = "bom-part-link";
    cell.append(link);
    return cell;
  };

  const blockedStatus = (row) => {
    const cell = element("div", "", "bom-blocked-status");
    const issue = issues.get(row.side.number);
    if (issue) {
      cell.append(element("span", "Blocked", "semantic-pill tone-error"));
      cell.append(element("small", "Missing purchasing entry"));
    } else if (pricedMaterials.has(row.side.number)) {
      cell.append(element("span", "Not blocked", "semantic-pill tone-success"));
      cell.title = "A purchasing entry is available for this material.";
    } else if (exemptMaterials.has(row.side.number) || row.status !== "NEW") {
      cell.append(element("span", "Not required", "semantic-pill tone-neutral"));
      cell.title = "This material does not require new-part price evidence for acceptance.";
    } else {
      cell.append(element("span", reviewState.acceptanceStatus === "loading" ? "Checking…" : "Not assessed", "semantic-pill tone-neutral"));
    }
    return cell;
  };

  const ecnChangeStatus = (row) => {
    const cell = element("div", "", "bom-change-lifecycle");
    const [label, tone] = ECN_STATUS[row.status];
    cell.append(element("span", label, `semantic-pill tone-${tone}`));
    return cell;
  };

  const assembly = (line) => {
    const cell = element("div", "", "bom-change-assembly");
    cell.append(element("strong", line.assembly_description || (line.assembly ? `Assembly ${line.assembly}` : "Assembly not supplied")));
    const identifiers = [line.model_range, line.assembly].filter(Boolean).join(" · ");
    if (line.assembly_description && identifiers) cell.append(element("small", identifiers));
    else if (line.model_range) cell.append(element("small", line.model_range));
    return cell;
  };

  const materialRows = lines.flatMap((line) => {
    const outgoing = line.outgoing && { line, side: line.outgoing, status: "REMOVED" };
    const incoming = line.incoming && { line, side: line.incoming, status: "NEW" };
    if (outgoing && incoming && outgoing.side.number === incoming.side.number) {
      return [{
        line,
        side: incoming.side,
        status: "UNCHANGED",
      }];
    }
    return [outgoing, incoming].filter(Boolean);
  });

  // The full occurrence key prevents the same normalized assembly number in
  // different configurations from being visually merged. Input order is kept.
  const groups = new Map();
  materialRows.forEach((row) => {
    const { line } = row;
    const key = [line.scope_id, line.model_range, line.parent || line.assembly, line.assembly, line.alternative, line.sequence]
      .map((value) => value ?? "")
      .join("\u001f");
    const group = groups.get(key);
    if (group) group.rows.push(row);
    else groups.set(key, { line, rows: [row] });
  });

  const represented = new Set(materialRows.map(row => row.side.number));
  const unplaced = [...issues.values()].filter(issue => !represented.has(issue.number));
  if (unplaced.length) groups.set("unplaced", {
    line: {}, unplaced: true,
    rows: unplaced.map(issue => ({ line: {}, side: issue, status: "NEW", unplaced: true })),
  });

  const headers = ["Material", "Description", "ECN change status", "Parent assembly", "Affected plants", "Blocked status"];
  const evidenceTable = element("div", "", "table-scroll");
  const node = element("table");
  const head = element("thead");
  const headerRow = element("tr");
  headers.forEach((header) => { const th = element("th", header); th.scope = "col"; headerRow.append(th); });
  head.append(headerRow);
  node.append(head);

  for (const { line, rows, unplaced } of groups.values()) {
    const body = element("tbody", "", "bom-assembly-group");
    const groupRow = element("tr", "", "bom-assembly-group-header");
    const groupHeader = element("th");
    groupHeader.scope = "rowgroup";
    groupHeader.colSpan = headers.length;
    const title = unplaced ? "Materials without BOM placement" : line.assembly_description || (line.assembly ? `Assembly ${line.assembly}` : "Assembly not supplied");
    const identifiers = [line.model_range, line.assembly].filter(Boolean).join(" · ");
    groupHeader.append(element("strong", title, "bom-assembly-group-title"));
    if (identifiers) groupHeader.append(element("span", identifiers, "bom-assembly-group-identifiers"));
    groupHeader.append(element("span", `${rows.length} affected material ${rows.length === 1 ? "row" : "rows"}`, "bom-assembly-group-count"));
    groupRow.append(groupHeader);
    body.append(groupRow);

    rows.forEach((row) => {
      const materialRow = element("tr", "", "bom-change-row");
      [
        material(row.side),
        row.side.description || "Not supplied",
        ecnChangeStatus(row),
        row.unplaced ? "BOM placement not supplied" : assembly(row.line),
        row.unplaced ? row.side.plants : plantsLabel(row.line.plants),
        blockedStatus(row),
      ].forEach((value, index) => {
        const cell = element("td");
        cell.dataset.label = headers[index];
        if (value instanceof Node) cell.append(value);
        else cell.textContent = value ?? "Unavailable";
        materialRow.append(cell);
      });
      body.append(materialRow);
    });
    node.append(body);
  }

  evidenceTable.append(node);
  // Long ECNs keep their full, in-place evidence while the card remains bounded.
  evidenceTable.tabIndex = 0;
  evidenceTable.setAttribute("role", "region");
  evidenceTable.setAttribute("aria-label", "BOM change evidence table. Scroll for additional rows.");
  host.append(evidenceTable);
  evidenceTable.scrollTop = scrollTop;
  evidenceTable.scrollLeft = scrollLeft;
}
