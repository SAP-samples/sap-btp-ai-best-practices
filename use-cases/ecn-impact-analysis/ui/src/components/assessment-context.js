import * as api from "../services/workspace-api.js";
import { button, element, table } from "./workspace-dom.js";
import { validity, readable } from "./material-dialog.js";

/** Mount date, history and reassessment below the source summary; retain immutable versions. */
export function mountAssessmentContext(host, assessmentId, snapshotRequest, signal) {
  const pane = element("section", "", "assessment-context");
  const assessed = element("p", "Loading assessment date…"); pane.append(assessed);
  snapshotRequest.then(snapshot => {
    if (!signal.aborted) assessed.textContent = `BOM usage assessed as of: ${validity(snapshot.assessment_date)}`;
  }).catch(() => { if (!signal.aborted) assessed.textContent = "Assessment date unavailable."; });

  const history = element("details"); history.append(element("summary", "Assessment Versions"));
  const versions = element("ol"); history.append(versions);
  api.versions(assessmentId, signal).then(values => {
    if (signal.aborted) return;
    if (!values.length) versions.append(element("li", "No other assessments available."));
    for (const version of values) {
      const current = version.ecn_row_id === assessmentId;
      const link = element("a", `Created ${new Date(version.created_at).toLocaleString()}${current ? " · Viewing" : ""}`);
      link.href = `/ecns/${encodeURIComponent(version.ecn_row_id)}`;
      if (current) link.setAttribute("aria-current", "page");
      const row = element("li"); row.append(link); versions.append(row);
    }
  }).catch(error => { if (error.name !== "AbortError") versions.append(element("li", error.message)); });

  const reassessment = element("details"); reassessment.append(element("summary", "Create New Assessment"),
    element("p", "Create a new immutable assessment using the current published datasets. Earlier assessments and reviews stay available."));
  const date = element("input"); date.type = "date"; date.id = "new-assessment-date";
  const label = element("label", "Assess BOM usage as of (optional)"); label.htmlFor = date.id;
  const status = element("p"); status.setAttribute("role", "status"); let pending;
  const create = button("Create new assessment", async () => {
    if (!date.reportValidity()) return;
    if (!pending || pending.date !== date.value) pending = { date: date.value, id: crypto.randomUUID() };
    create.disabled = true; date.disabled = true; status.textContent = "Preparing the new assessment…";
    try {
      const result = await api.reassess(assessmentId, pending.id, signal, pending.date || null);
      if (signal.aborted) return;
      const link = element("a", "Open new assessment"); link.href = `/ecns/${encodeURIComponent(result.ecn_row_id)}`;
      status.replaceChildren(link);
    } catch (error) {
      if (error.name !== "AbortError") { status.textContent = error.message; create.disabled = false; date.disabled = false; }
    }
  });
  reassessment.append(label, date, element("p", "Leave blank to use today's date.", "helper-text"), create, status);
  pane.append(history, reassessment); host.append(pane);
}

/** Display every pinned dataset's readable scope and extraction date without hashes. */
export function renderDatasets(host, snapshot) {
  if (!host) return;
  host.replaceChildren();
  if (snapshot.datasets?.length) host.append(table(["Source", "Scope", "Extracted", "Use"], snapshot.datasets.map(d => [
    `${d.file_name || "Source file"}\n${readable(d.dataset_type)}`,
    d.model_range || "Model not supplied",
    validity(d.snapshot_date), d.comparison ? "Historical comparison" : "Assessment data",
  ])));
  const limits = element("details"); limits.append(element("summary", "Scope and Limitations"));
  for (const limitation of snapshot.limitations || []) limits.append(element("p", limitation, "helper-text"));
  host.append(limits);
}
