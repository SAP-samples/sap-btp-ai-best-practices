/**
 * Renderers for the evidence panels of one selected run in the Optimizer workspace:
 *
 * - the failure diagnostics panel of a failed run (GET /api/runs/{id}/failure-diagnostics);
 * - one page of the run's frozen compatibility matrix (GET /api/runs/{id}/matrix).
 *
 * They only build DOM nodes from API payloads; fetching, selection and polling stay
 * in optimizer.js. Keeping them here lets tests/run_diagnostics.test.js render the
 * rarely seen failure paths without mounting the whole page.
 */
import { dataTable, display, el } from "./dom.js";

/** @typedef {Array<[string, string]>} Columns `[payload key, header label]` pairs for dataTable. */

/** Columns of the affected-block table. @type {Columns} */
const AFFECTED_BLOCK_COLUMNS = [
  ["plant", "Plant"],
  ["sefi", "SEFI"],
  ["candidate_count", "Candidates"],
  ["error_type", "Failure type"],
  ["error_message", "Message"],
];

/** Columns of the missing-volume table of an incomplete plant profile matrix. @type {Columns} */
const MISSING_VOLUME_COLUMNS = [
  ["volume", "Volume (L)"],
  ["member_count", "Products"],
];

/** Columns of one frozen compatibility matrix page. @type {Columns} */
const MATRIX_COLUMNS = [
  ["volume_a", "Volume A"],
  ["volume_b", "Volume B"],
  ["status", "Status"],
];

/**
 * Append the solve process's own failure report and saved output tail.
 *
 * @param {HTMLElement} panel Failure diagnostics panel to append to.
 * @param {object} diagnostics Response of /api/runs/{id}/failure-diagnostics.
 * @returns {void}
 */
export function renderWorkerFailure(panel, diagnostics) {
  const failure = diagnostics.worker_failure;
  if (failure) {
    panel.append(el("h4", "Cause reported by the solve process"));
    panel.append(el("p", `${failure.exception_type || "Error"} during stage ${failure.stage || "unknown"}: ${failure.message || ""}`));
    const missing = failure.matrix_coverage?.missing_volumes || [];
    if (missing.length) {
      panel.append(
        el("p", "Package volumes missing from the plant profile volume matrix. Add them in Settings or ask the Assistant to propose them.", "muted"),
        dataTable(MISSING_VOLUME_COLUMNS, missing),
      );
    }
    if ((failure.traceback || []).length) {
      const details = el("details");
      details.append(
        el("summary", "Technical trace"),
        el("pre", failure.traceback.map((frame) => `${frame.file}:${frame.line} in ${frame.function}\n    ${frame.code || ""}`).join("\n")),
      );
      panel.append(details);
    }
  }
  if (diagnostics.worker_log_tail) {
    const details = el("details");
    details.append(el("summary", "Solve process output (last lines)"), el("pre", diagnostics.worker_log_tail));
    panel.append(details);
  }
}

/**
 * Replace the panel content with the failed run's bounded evidence and show it.
 *
 * Content, in order: the public error, summary fields, the solve process report
 * (see renderWorkerFailure) and a sample of the blocks that had no valid result.
 *
 * @param {HTMLElement} panel The #run-failure-diagnostics element.
 * @param {object} diagnostics Response of /api/runs/{id}/failure-diagnostics.
 * @returns {void}
 */
export function renderFailureDiagnostics(panel, diagnostics) {
  const fields = [
    ["Failure type", (diagnostics.failure_types || []).join(", ") || "Not recorded"],
    ["Affected blocks", diagnostics.affected_block_count],
    ["Candidates in affected blocks", diagnostics.candidate_count],
    ["Checkpoint available", diagnostics.checkpoint_available ? "Yes" : "No"],
    ["Solve process report", diagnostics.worker_report_available ? "Saved" : "Not saved"],
    ["Completion", diagnostics.complete == null ? "Not recorded" : diagnostics.complete ? "Complete" : "Incomplete"],
    ["Integrity", diagnostics.integrity_valid == null ? "Not recorded" : diagnostics.integrity_valid ? "Valid" : "Invalid"],
  ];
  const fieldList = el("dl", null, "diagnostic-fields");
  for (const [label, value] of fields) {
    const row = el("div");
    row.append(el("dt", label), el("dd", display(value)));
    fieldList.append(row);
  }
  panel.replaceChildren(
    el("h3", "Failure diagnostics"),
    diagnostics.public_error ? el("p", diagnostics.public_error) : el("p", "The saved checkpoint provides the evidence below."),
    fieldList,
  );
  renderWorkerFailure(panel, diagnostics);
  const blocks = diagnostics.affected_blocks || [];
  if (blocks.length) {
    panel.append(dataTable(AFFECTED_BLOCK_COLUMNS, blocks));
    if (diagnostics.affected_blocks_truncated)
      panel.append(el("p", "Only the first affected blocks are shown. Use scoped diagnostics or result queries for more evidence.", "muted"));
  }
  panel.hidden = false;
}

/**
 * Build the summary sentence and table for one frozen compatibility matrix page.
 *
 * @param {{total: number, offset: number, rows: Array<object>}} page Response of /api/runs/{id}/matrix.
 * @returns {{summary: string, table: HTMLTableElement}} Text for #run-matrix-summary and the table.
 */
export function matrixPageView(page) {
  const first = page.total ? page.offset + 1 : 0;
  const last = Math.min(page.offset + page.rows.length, page.total);
  return {
    summary: page.total
      ? `Showing ${first}–${last} of ${page.total} frozen compatibility pairs.`
      : "No frozen compatibility pairs match this filter.",
    table: dataTable(MATRIX_COLUMNS, page.rows),
  };
}
