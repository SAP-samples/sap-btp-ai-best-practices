/** SAP settings panel helpers: node --test tests/sap_settings.test.js */
import test from "node:test";
import assert from "node:assert/strict";
import { kpiLine, SAP_VIEWS, selectionLine, tabStatusRows } from "../src/workspace/sap_settings.js";

const SUMMARY = {
  tabs: {
    cycle_stock: { status: "ready", rows: 12, message: "" },
    planning_calendar: { status: "blocked", rows: 0, message: "Set calendar_odd_profile" },
    central_dc_lot_size: { status: "not_applicable", rows: 0, message: "" },
  },
  wheel_roc_days: 18.8,
  kpi_changeovers_per_week: 35.2,
  point_j_ch: 35.21,
  runner_adjusted_fini: 3,
};

test("tab statuses keep order, business labels and messages", () => {
  assert.deepEqual(tabStatusRows(SUMMARY).map((row) => [row.label, row.statusLabel, row.rows]), [
    ["Cycle stock", "Ready", 12],
    ["Planning calendar", "Blocked", 0],
    ["Central DC lot size", "Not applicable", 0],
  ]);
  assert.equal(tabStatusRows(SUMMARY)[1].message, "Set calendar_odd_profile");
  assert.deepEqual(tabStatusRows(null), []);
});

test("KPI line shows the template changeovers next to the optimizer J_CH", () => {
  assert.equal(kpiLine(SUMMARY),
    "Wheel RoC 18.80 days · 35.20 changeovers/week (optimizer: 35.21) · 3 FINIs with runner-adjusted RoC");
  assert.equal(kpiLine(null), "");
  assert.match(kpiLine({ ...SUMMARY, runner_multiple_subgroups: 2, kpi_single_changeovers_per_week: 40 }),
    / · 2 subgroups with runner multiples \(single RoC would give 40\.00 changeovers\/week\)$/);
});

test("selection line guides the user until a scenario is chosen", () => {
  assert.match(selectionLine(null), /Select a decision scenario/);
  assert.equal(selectionLine({ point_index: 5, decision_label: "CO1" }), "Selected scenario: CO1 (point 5).");
  assert.equal(SAP_VIEWS.length, 7);
});
