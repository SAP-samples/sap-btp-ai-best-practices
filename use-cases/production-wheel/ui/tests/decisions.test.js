/** Decision-scenario chart and table helpers: node --test tests/decisions.test.js */
import test from "node:test";
import assert from "node:assert/strict";
import {
  CHANGEOVER_FOOTNOTE,
  frontierDatasets,
  frontierNote,
  keyPoints,
  pointIndexAt,
  scenarioRows,
} from "../src/workspace/decisions.js";

/** Nine frontier points: CO2..INV3 labelled around the knee at point 3, two unlabelled. */
function scenarios(selection = null) {
  const labels = { 1: ["CO2", "changeover_oriented"], 2: ["CO1", "changeover_oriented"],
    3: ["B", "balanced"], 4: ["INV1", "inventory_oriented"], 5: ["INV2", "inventory_oriented"],
    6: ["INV3", "inventory_oriented"] };
  return {
    knee_point_index: 3,
    selection,
    frontier_shortfall: false,
    frontier_points_retained: 9,
    points: Array.from({ length: 8 }, (_, offset) => {
      const index = offset + 1;
      const [label, kind] = labels[index] || [null, null];
      return { point_index: index, j_ch: index, coverage_days: 9 / index,
        sefi_batches_per_week: 10 + index / 3, group_count: 20 + index,
        decision_label: label, decision_class: kind };
    }),
  };
}

test("only the key points are shown by default", () => {
  assert.deepEqual(keyPoints(scenarios()).map((p) => p.decision_label),
    ["CO2", "CO1", "B", "INV1", "INV2", "INV3"]);
  assert.equal(scenarioRows(scenarios(), false).length, 6);
  assert.equal(scenarioRows(scenarios(), true).length, 8);
});

test("chart datasets split by class and add the rest only when showing all", () => {
  const key = frontierDatasets(scenarios(), false);
  assert.deepEqual(key.map((d) => d.label),
    ["Site / changeover-oriented", "Balanced", "Inventory-reduction-oriented"]);
  assert.equal(key[1].data[0].point_index, 3);
  assert.equal(key[1].pointStyle, "rectRot");
  assert.equal(frontierDatasets(scenarios(), true).at(-1).label, "Other frontier points");
});

test("a chart click maps through its own dataset, not a flat point list", () => {
  const datasets = frontierDatasets(scenarios(), false);
  assert.equal(pointIndexAt(datasets, { datasetIndex: 2, index: 1 }), 5);
  assert.equal(pointIndexAt(datasets, { datasetIndex: 9, index: 0 }), undefined);
});

test("rows carry rounded metrics, the selection flag and the changeover footnote", () => {
  const rows = scenarioRows(scenarios({ point_index: 4 }), false);
  const inventory = rows.find((row) => row.point_index === 4);
  assert.equal(inventory.label, "INV1");
  assert.equal(inventory.coverage_days, 2.3);
  assert.equal(inventory.sefi_batches_per_week, 11.3);
  assert.equal(inventory.selected, true);
  assert.equal(rows.filter((row) => row.selected).length, 1);
  assert.match(CHANGEOVER_FOOTNOTE, /within a group/);
});

test("a shortfall note appears only when the frontier missed its target", () => {
  assert.equal(frontierNote(scenarios()), "");
  assert.match(frontierNote({ ...scenarios(), frontier_shortfall: true, frontier_points_retained: 5 }), /only 5 distinct/);
});

test("cycle stock group counts appear only for points that use runner multiples", () => {
  const data = scenarios();
  assert.equal(scenarioRows(data, true)[0].cycle_stock, null);
  data.points[0] = { ...data.points[0], single_roc_group_count: 18, runner_multiple_group_count: 3 };
  data.points[1] = { ...data.points[1], single_roc_group_count: 22, runner_multiple_group_count: 0 };
  const rows = scenarioRows(data, true);
  assert.equal(rows[0].cycle_stock, "18 single · 3 multiple");
  assert.equal(rows[1].cycle_stock, null);
});
