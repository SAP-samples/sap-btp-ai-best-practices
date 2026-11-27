/**
 * Run evidence renderer contracts (failure diagnostics and frozen matrix pages).
 *
 * Run from ui/:
 *   node --test tests/run_diagnostics.test.js
 *   npm test                      # typecheck plus every unit test
 *
 * These paths only render for failed runs or on an explicit "Load matrix evidence"
 * click, so they went unnoticed when they threw "Cannot read properties of undefined
 * (reading 'append')". The payloads below follow the API response shapes.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

// The renderers create nodes through the global document, as in the browser.
const { window } = new JSDOM("<!doctype html><body><div id='panel' hidden></div></body>");
globalThis.window = window;
globalThis.document = window.document;
globalThis.Node = window.Node;

const { dataTable } = await import("../src/workspace/dom.js");
const { matrixPageView, renderFailureDiagnostics } = await import("../src/workspace/run_diagnostics.js");

/**
 * Return the header labels and body cell texts of a rendered table.
 * Input: HTMLTableElement. Output: {headers: string[], rows: string[][]}.
 */
function tableText(table) {
  return {
    headers: [...table.querySelectorAll("thead th")].map((cell) => cell.textContent),
    rows: [...table.querySelectorAll("tbody tr")].map((row) => [...row.children].map((cell) => cell.textContent)),
  };
}

/** A fresh, hidden diagnostics panel like #run-failure-diagnostics. */
function panel() {
  const node = document.createElement("div");
  node.hidden = true;
  return node;
}

/** Failure diagnostics of a run whose publication failed because blocks had no result. */
const blockFailure = {
  run_id: "run-1",
  status: "failed",
  stage: "publication_failed",
  public_error: "Solve failed: ValueError: result bundle is incomplete or failed independent validation See failure diagnostics.",
  checkpoint_available: true,
  complete: false,
  integrity_valid: false,
  failure_types: ["RuntimeError", "invalid_input"],
  affected_block_count: 2,
  candidate_count: 316,
  affected_blocks: [
    { plant: "P1", sefi: "S1", candidate_count: 130, error_type: "RuntimeError", error_message: "greenfield anchor produced no incumbent" },
    { plant: "P1", sefi: "S2", candidate_count: 186, error_type: "RuntimeError", error_message: "greenfield anchor produced no incumbent" },
  ],
  affected_blocks_truncated: true,
  worker_report_available: true,
  worker_failure: {
    failure_type: "invalid_input",
    exception_type: "ValueError",
    message: "result bundle is incomplete or failed independent validation",
    stage: "persisting",
    traceback: [{ file: "app/workspace/worker.py", line: 190, function: "execute_run", code: "service.publish_results(...)" }],
    matrix_coverage: null,
  },
  worker_log_available: true,
  worker_log_tail: "block_frontier: 100%",
};

test("dataTable builds one header row and one body row per record", () => {
  const table = dataTable([["a", "A"], ["b", "B"]], [{ a: 1, b: "x" }, { a: 2 }]);
  assert.equal(table.tagName, "TABLE");
  assert.deepEqual(tableText(table), { headers: ["A", "B"], rows: [["1", "x"], ["2", "—"]] });
});

test("a failed run with affected blocks renders the block table and shows the panel", () => {
  const target = panel();
  renderFailureDiagnostics(target, blockFailure);
  assert.equal(target.hidden, false);
  assert.match(target.querySelector("p").textContent, /result bundle is incomplete/);
  const tables = target.querySelectorAll("table");
  assert.equal(tables.length, 1);
  assert.deepEqual(tableText(tables[0]), {
    headers: ["Plant", "SEFI", "Candidates", "Failure type", "Message"],
    rows: [
      ["P1", "S1", "130", "RuntimeError", "greenfield anchor produced no incumbent"],
      ["P1", "S2", "186", "RuntimeError", "greenfield anchor produced no incumbent"],
    ],
  });
  assert.match(target.textContent, /Only the first affected blocks are shown/);
  assert.match(target.textContent, /Cause reported by the solve process/);
  assert.match(target.textContent, /worker\.py:190 in execute_run/);
  assert.match(target.textContent, /Solve process output/);
});

test("an incomplete profile matrix renders the missing volumes table", () => {
  const target = panel();
  renderFailureDiagnostics(target, {
    failure_types: ["profile_matrix_incomplete"],
    affected_blocks: [],
    worker_report_available: true,
    worker_failure: {
      failure_type: "profile_matrix_incomplete",
      exception_type: "ProfileMatrixCoverageError",
      message: "volumes missing",
      stage: "optimizing",
      traceback: [],
      matrix_coverage: { missing_volumes: [{ volume: "0.465", member_count: 12 }, { volume: "2.5", member_count: 3 }] },
    },
  });
  const tables = target.querySelectorAll("table");
  assert.equal(tables.length, 1);
  assert.deepEqual(tableText(tables[0]), {
    headers: ["Volume (L)", "Products"],
    rows: [["0.465", "12"], ["2.5", "3"]],
  });
  assert.match(target.textContent, /Package volumes missing/);
});

test("diagnostics without checkpoint or report still render their summary", () => {
  const target = panel();
  renderFailureDiagnostics(target, { checkpoint_available: false, worker_report_available: false });
  assert.equal(target.hidden, false);
  assert.equal(target.querySelectorAll("table").length, 0);
  assert.match(target.textContent, /Not recorded/);
  assert.match(target.textContent, /The saved checkpoint provides the evidence below/);
});

test("a frozen matrix page renders its rows and the visible range", () => {
  const view = matrixPageView({
    total: 120,
    offset: 50,
    rows: [{ volume_a: "0.125", volume_b: "0.25", status: "AVOID" }, { volume_a: "0.125", volume_b: "0.125", status: "Y" }],
  });
  assert.equal(view.summary, "Showing 51–52 of 120 frozen compatibility pairs.");
  assert.deepEqual(tableText(view.table), {
    headers: ["Volume A", "Volume B", "Status"],
    rows: [["0.125", "0.25", "AVOID"], ["0.125", "0.125", "Y"]],
  });
});

test("an empty matrix filter explains that nothing matches", () => {
  const view = matrixPageView({ total: 0, offset: 0, rows: [] });
  assert.equal(view.summary, "No frozen compatibility pairs match this filter.");
  assert.equal(tableText(view.table).rows.length, 0);
});
