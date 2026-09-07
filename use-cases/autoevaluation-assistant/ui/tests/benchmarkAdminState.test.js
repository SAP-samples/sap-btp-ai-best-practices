import assert from "node:assert/strict";
import test from "node:test";

import {
  applyBenchmarkValidation,
  beginBenchmarkHistoryLoad,
  beginBenchmarkMutation,
  benchmarkHistoryView,
  buildBenchmarkImportForm,
  canActivateBenchmark,
  createBenchmarkAdminState,
  completeBenchmarkHistoryLoad,
  formatBenchmarkTimestamp,
  selectBenchmarkWorkbook
} from "../src/pages/admin-documents/benchmarkAdminState.js";

test("activation requires successful validation for the currently selected workbook", () => {
  const firstFile = { name: "first.xlsx", size: 10 };
  const secondFile = { name: "second.xlsx", size: 20 };
  let state = selectBenchmarkWorkbook(createBenchmarkAdminState(), firstFile);

  assert.equal(canActivateBenchmark(state), false);
  state = applyBenchmarkValidation(state, {
    success: true,
    source_sha256: "abc",
    source_filename: firstFile.name
  });
  assert.equal(canActivateBenchmark(state), true);

  state = selectBenchmarkWorkbook(state, secondFile);
  assert.equal(canActivateBenchmark(state), false);
  assert.equal(state.validation, null);
  assert.equal(state.error, null);
});

test("failed validation never enables activation", () => {
  const file = { name: "invalid.xlsx", size: 10 };
  let state = selectBenchmarkWorkbook(createBenchmarkAdminState(), file);
  state = applyBenchmarkValidation(state, {
    success: false,
    source_sha256: "invalid",
    source_filename: file.name
  });

  assert.equal(canActivateBenchmark(state), false);
});

test("a late validation response cannot authorize a newer selection", () => {
  let state = selectBenchmarkWorkbook(createBenchmarkAdminState(), {
    name: "first.xlsx",
    size: 10
  });
  const firstRevision = state.selectionRevision;
  state = selectBenchmarkWorkbook(state, { name: "second.xlsx", size: 10 });
  state = applyBenchmarkValidation(
    state,
    { success: true, source_sha256: "late-first" },
    firstRevision
  );

  assert.equal(state.validation, null);
  assert.equal(canActivateBenchmark(state), false);
});

test("selection tokens are never reused across route remounts", () => {
  const firstMount = selectBenchmarkWorkbook(createBenchmarkAdminState(), {
    name: "first.xlsx",
    size: 10
  });
  const secondMount = selectBenchmarkWorkbook(createBenchmarkAdminState(), {
    name: "second.xlsx",
    size: 10
  });

  assert.notEqual(firstMount.selectionRevision, secondMount.selectionRevision);
  const afterLateResponse = applyBenchmarkValidation(
    secondMount,
    { success: true, source_sha256: "late-first" },
    firstMount.selectionRevision
  );
  assert.equal(afterLateResponse.validation, null);
  assert.equal(canActivateBenchmark(afterLateResponse), false);
});

test("dry-run and activation use the same workbook field with an explicit write flag", () => {
  class RecordingFormData {
    constructor() {
      this.values = new Map();
    }

    append(key, value, filename) {
      this.values.set(key, { value, filename });
    }
  }

  const file = { name: "peers.xlsx" };
  const dryRun = buildBenchmarkImportForm(file, false, RecordingFormData);
  const write = buildBenchmarkImportForm(file, true, RecordingFormData);

  assert.deepEqual(dryRun.values.get("workbook"), { value: file, filename: "peers.xlsx" });
  assert.equal(dryRun.values.get("write").value, "false");
  assert.deepEqual(write.values.get("workbook"), { value: file, filename: "peers.xlsx" });
  assert.equal(write.values.get("write").value, "true");
});

test("active import history exposes explicit unavailable and active states", () => {
  assert.deepEqual(benchmarkHistoryView({ available: false, reason: "no_active_dataset" }), {
    kind: "unavailable",
    reason: "no_active_dataset",
    active: null
  });
  const active = { import_id: "import-1", status: "active", source_filename: "peers.xlsx" };
  assert.deepEqual(benchmarkHistoryView({ available: true, active }), {
    kind: "active",
    reason: null,
    active
  });
});

test("a late history response cannot clear an activation mutation", () => {
  const selected = selectBenchmarkWorkbook(createBenchmarkAdminState(), {
    name: "peers.xlsx",
    size: 10
  });
  const historyRequest = beginBenchmarkHistoryLoad(selected);
  const activating = beginBenchmarkMutation(historyRequest.state, "activation");
  const afterLateHistory = completeBenchmarkHistoryLoad(
    activating,
    { available: true, active: { import_id: "older-view" } },
    historyRequest.requestRevision
  );

  assert.equal(afterLateHistory.mutationLoading, "activation");
  assert.equal(afterLateHistory.historyLoading, false);
  assert.equal(canActivateBenchmark(afterLateHistory), false);
});

test("active import timestamps follow the selected English or Italian locale", () => {
  const timestamp = "2026-07-13T14:30:00Z";
  const english = formatBenchmarkTimestamp(timestamp, "en");
  const italian = formatBenchmarkTimestamp(timestamp, "it");

  assert.equal(english, new Date(timestamp).toLocaleString("en"));
  assert.equal(italian, new Date(timestamp).toLocaleString("it"));
  assert.notEqual(italian, english);
  assert.doesNotMatch(italian, /\b(?:AM|PM)\b/i);
});
