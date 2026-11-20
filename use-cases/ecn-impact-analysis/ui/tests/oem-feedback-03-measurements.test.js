import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { assessmentSummary, materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";

vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));
let controller;
beforeEach(() => {
  document.body.replaceChildren(); vi.resetAllMocks(); controller = new AbortController();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});
afterEach(() => controller.abort());

function lengthCriterion(totalLengthMm, minimumMm = "200", maximumMm = "1500", status = "Pass") {
  return { id: "tube_length", title: "Tube length", status, explanation: "Synthetic calculator result", calculator: {
    tube_length: { diameterMm: "10", wallThicknessMm: "1.00", totalLengthMm, minimumMm, maximumMm, status },
  } };
}
function storedReport(criteria) {
  return { overall_result: "Inconclusive", candidate_reports: [{ id: "synthetic", label: "Synthetic primary", completeness: "partial", result: "Inconclusive", criteria, limitations: [] }] };
}
async function open(report, extra = {}) {
  api.getObject.mockResolvedValue({ number: "SIM-PIPE", description: "Synthetic pipe", types: [], material_brief: { sourcing: [] }, manufacturing_feasibility: report, ...extra });
  await materialDialog("synthetic-ecn", controller.signal, vi.fn())("SIM-PIPE");
  return document.querySelector("dialog");
}
function row(host, label, selector = ".drawing-feasibility-calculations") {
  return [...host.querySelectorAll(`${selector} tbody tr`)].find(node => node.cells[0].textContent === label);
}

it.each([
  ["1000", "200", "1500", "1000 mm", "200–1500 mm"],
  ["629.6", "200", "1500", "629.6 mm", "200–1500 mm"],
  ["1781.700", "200.00", "3000.000", "1781.700 mm", "200.00–3000.000 mm"],
  [0, 0, 1500, "0 mm", "0–1500 mm"],
  ["0.000", "-1000.00", "+1500.00", "0.000 mm", "-1000.00–+1500.00 mm"],
  ["-1000.50", "-1500", "+1500", "-1000.50 mm", "-1500–+1500 mm"],
  ["+1000.5000", "200", "1500", "+1000.5000 mm", "200–1500 mm"],
  ["12345678901234567890.123456789", "0", "99999999999999999999", "12345678901234567890.123456789 mm", "0–99999999999999999999 mm"],
  ["1.2300e+3", "0", "3e3", "1.2300e+3 mm", "0–3e3 mm"],
])("keeps summary and detailed mm text identical for %s", async (value, minimum, maximum, measurement, range) => {
  const criterion = lengthCriterion(value, minimum, maximum);
  const expected = `${measurement} meets the documented ${range} range.`;
  expect(assessmentSummary([criterion]).find(item => item.title === "Tube Length").message).toBe(expected);
  const dialog = await open(storedReport([criterion]));
  const summary = row(dialog, "Tube Length", ".drawing-feasibility-summary-table");
  const length = row(dialog, "Total Tube Length");
  expect(summary.cells[2].textContent).toBe(expected);
  expect(length.cells[1].textContent).toBe(measurement);
  expect(length.cells[2].textContent).toBe(range);
  expect(length.querySelector("ui5-tag").textContent).toBe("Pass");
  expect(row(dialog, "Wall Thickness").cells[1].textContent).toBe("1.00 mm");
  expect(dialog.querySelector(".drawing-feasibility-overall-tag").textContent).toBe("Inconclusive");
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});

it("formats a failing finding without recalculating its status", async () => {
  const criterion = lengthCriterion("1781.7", "200", "1500", "Fail");
  const dialog = await open(storedReport([criterion]));
  expect(row(dialog, "Tube Length", ".drawing-feasibility-summary-table").cells[2].textContent)
    .toBe("1781.7 mm does not meet the documented 200–1500 mm range.");
  expect(row(dialog, "Total Tube Length").querySelector("ui5-tag").textContent).toBe("Fail");
});

it.each([null, undefined, "", "unknown", "NaN", "Infinity", NaN, Infinity, {}, [], false, "12.3.4", "1,000", "1000 mm"].map(value => [value]))(
  "renders missing or invalid mm input %j truthfully without guessing", async value => {
    const criterion = lengthCriterion(value, "200", "1500", "Cannot assess");
    expect(assessmentSummary([criterion]).find(item => item.title === "Tube Length").message).toBe("Tube length cannot be assessed.");
    const dialog = await open(storedReport([criterion]));
    expect(row(dialog, "Total Tube Length").cells[1].textContent).toBe("Not available");
    expect(row(dialog, "Total Tube Length").querySelector("ui5-tag").textContent).toBe("Cannot assess");
  },
);
it.each([[undefined, "1500"], ["200", null], ["invalid", "1500"]])("does not invent a range from %s and %s", async (minimum, maximum) => {
  const criterion = lengthCriterion("1000", minimum, maximum, "Cannot assess");
  // Override the helper's defaults to represent genuinely absent endpoints.
  Object.assign(criterion.calculator.tube_length, { minimumMm: minimum, maximumMm: maximum });
  const dialog = await open(storedReport([criterion]));
  expect(row(dialog, "Total Tube Length").cells[1].textContent).toBe("1000 mm");
  expect(row(dialog, "Total Tube Length").cells[2].textContent).toBe("Not available");
  expect(row(dialog, "Tube Length", ".drawing-feasibility-summary-table").cells[2].textContent).toBe("Tube length cannot be assessed.");
});

it("covers mapped crimp, derived L3 and robot requirements without modifying formulas, lists, money or report data", async () => {
  const formula = "L3 = a + b; lookup(1,000, 2,000)";
  const criteria = [lengthCriterion("1000"), {
    id: "automatic_crimp_dimensions", title: "Automatic crimp", status: "Fail", calculator: {
      schedule_mapping: { l1: { sourcePoint: "P0", sourceField: "STRAIGHT PART" }, l3: { l3Mm: "1178.53981633974483", formula }, theta1: { valueDeg: "1000", sourcePoint: "B1", sourceField: "ANGLE" } },
      auto_crimp: { overallStatus: "Fail", results: {
        l1: { designValueMm: "1000.00", requiredMinimumMm: "1500.00", formulaOrLookup: formula, status: "Fail" },
        l2: { segments: [{ segmentIndex: 1, designValueMm: "1000", requiredMinimumMm: "2000", status: "Fail" }] },
        l3: { designValueMm: "1178.53981633974483", requiredMinimumMm: "1000.1", status: "Pass" },
        l4: { designValueMm: 0, requiredMinimumMm: "0.00", status: "Pass" },
      } },
    },
  }, { id: "robot_pickup", title: "Robot pickup", status: "Met", calculator: { robot_pickup: {
    status: "Met", l1Branch: { designValueMm: "2000.000", requiredMinimumMm: "1000.00", status: "Pass" },
    l2Branch: { designValueMm: 0, requiredMinimumMm: 0, status: "Pass" },
  } } }, { id: "end_configuration", title: "End", status: "Warning", explanation: "Compatible A, incompatible B, C." }];
  const report = storedReport(criteria);
  const original = JSON.stringify(report);
  function freeze(value) { if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); } }
  freeze(report);
  const dialog = await open(report, { material_brief: { sourcing: [{ procurement_type_labels: ["External procurement"], source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", currency: "USD", net_price: "1,000.50", price_per: "1", order_price_unit: "EA" } }] } });
  expect(row(dialog, "Automatic-Crimp L1").cells[1].textContent).toBe("1000.00 mm (P0 STRAIGHT PART)");
  expect(row(dialog, "Automatic-Crimp L1").cells[2].textContent).toBe(`≥ 1500.00 mm — ${formula}`);
  expect(row(dialog, "Automatic-Crimp L2.1").cells[2].textContent).toBe("≥ 2000 mm");
  expect(row(dialog, "Automatic-Crimp L3").cells[1].textContent).toBe("1178.53981633974483 mm");
  expect(row(dialog, "L3 Derivation").cells[2].textContent).toBe(formula);
  expect(row(dialog, "Automatic-Crimp L4").cells[1].textContent).toBe("0 mm");
  expect(row(dialog, "Robot-Pickup L1 Branch").cells[1].textContent).toBe("2000.000 mm");
  expect(row(dialog, "Robot-Pickup L1 Branch").cells[2].textContent).toBe("≥ 1000.00 mm");
  expect(row(dialog, "Robot-Pickup L2 Branch").cells[2].textContent).toBe("≥ 0 mm");
  expect(row(dialog, "Schedule Angle θ1").cells[1].textContent).toBe("1,000° (B1 ANGLE)");
  expect(dialog.textContent).toContain("L1, L2.1 are below the PS4 minimum.");
  expect(dialog.textContent).toContain("Compatible A, incompatible B, C.");
  expect(dialog.textContent).toContain("USD 1,000.50 / 1 EA");
  expect(JSON.stringify(report)).toBe(original);
  expect(dialog.querySelector(".drawing-feasibility-summary-table").getAttribute("aria-label")).toBe("PS4 Decision Summary for Synthetic primary.");
  expect(dialog.querySelector(".drawing-feasibility-calculations").open).toBe(false);
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});
