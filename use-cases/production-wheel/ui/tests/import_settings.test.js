/**
 * Extraction settings helpers for the Datasets page and the assistant's cards.
 * Run with: npm test
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  importDecisionNote,
  modeledCodes,
  modeledRowCount,
  pendingImportContext,
  settingLines,
} from "../src/workspace/import_settings.js";

const inventory = {
  codes: [{ code: "01W", rows: 4 }, { code: "02W", rows: 3 }, { code: "16W", rows: 5 }],
  unrecognized: [{ value: "999.Not possible to determine", rows: 2 }],
  total_rows: 14,
};

test("modeled codes follow defaults, explicit lists and the all-available keyword", () => {
  assert.deepEqual(modeledCodes({}, inventory), ["01W", "02W"]);
  assert.deepEqual(modeledCodes({ modeled_frequencies: ["04W"] }, inventory), ["04W"]);
  assert.deepEqual(modeledCodes({ modeled_frequencies: "all_available" }, inventory), ["01W", "02W", "16W"]);
  assert.deepEqual(modeledRowCount({ modeled_frequencies: "all_available" }, inventory), { modeled: 12, total: 14 });
  assert.deepEqual(modeledRowCount({}, inventory), { modeled: 7, total: 14 });
});

test("setting lines show defaults and applied values in plain words", () => {
  const lines = settingLines({ modeled_frequencies: "all_available", demand_days: 240 }, inventory);
  assert.equal(lines[0], "Production frequencies: all 3 codes in the workbook (01W, 02W, 16W) (applied)");
  assert.equal(lines[1], "Demand days: 240 (applied)");
  assert.equal(settingLines({ demand_days: 240 }, null, "proposed")[1], "Demand days: 240 (proposed)");
  assert.equal(lines[3], "Canonical factor: 0.90 (default)");
  assert.equal(settingLines({})[0], "Production frequencies: 01W / 02W (default)");
});

test("the pending import reaches the assistant only from the Datasets page", () => {
  const pending = { preview_id: "p", primary_filename: "wheel.xlsx" };
  assert.equal(pendingImportContext({ page: "datasets" }, pending), pending);
  assert.equal(pendingImportContext({ page: "workspace" }, pending), null);
  assert.equal(pendingImportContext({ page: "datasets" }, null), null);
});

test("decision notes tell the assistant what the planner did", () => {
  const proposal = { primary_filename: "wheel.xlsx", overrides: { demand_days: 240 } };
  assert.match(importDecisionNote(proposal, true), /^\[Applied in UI\] .*"demand_days":240/);
  assert.match(importDecisionNote(proposal, false), /^\[Rejected in UI\]/);
});
