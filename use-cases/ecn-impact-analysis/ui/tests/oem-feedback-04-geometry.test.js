import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";

vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));
let controller;
beforeEach(() => {
  document.body.replaceChildren(); vi.resetAllMocks(); controller = new AbortController();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});
afterEach(() => controller.abort());

function criteria(status = "Pass") {
  const dimension = { designValueMm: "1000.00", requiredMinimumMm: "200.00", formulaOrLookup: "PS4-SPEC-TEST lookup(1,000, 2,000)", status: "Pass" };
  return [
    { id: "raw_material", title: "Raw material", status, calculator: { raw_material: { designation: "SYNTHETIC-PS4-SUFFIX", guidelineCoverage: "listed" } } },
    { id: "tube_length", title: "Tube length", status, calculator: { tube_length: { diameterMm: "10", wallThicknessMm: "1.00", totalLengthMm: "629.6", minimumMm: "200", maximumMm: "1500", status: "Pass" } } },
    { id: "bend_angles", title: "Bend angles", status, calculator: { bend_angles: { results: [{ angleDeg: "90", status: "Pass" }, { angleDeg: "120", status: "Pass" }] } } },
    { id: "automatic_crimp_dimensions", title: "Automatic crimp", status, calculator: {
      schedule_mapping: { l1: { sourcePoint: "P0", sourceField: "STRAIGHT PART" }, l2Segments: [{ sourcePoint: "B1", sourceField: "STRAIGHT PART" }], l3: { l3Mm: "1178.53981633974483", formula: "L3 = terminal PS4-SPEC-TEST + radius × θ4(rad)" }, l4: { sourcePoint: "B2", sourceField: "STRAIGHT PART" }, theta1: { valueDeg: "90", sourcePoint: "B1", sourceField: "ANGLE" }, theta4: { valueDeg: "120", sourcePoint: "B2", sourceField: "ANGLE" } },
      auto_crimp: { overallStatus: "Pass", results: { l1: dimension, l2: { segments: [dimension] }, l3: dimension, l4: dimension } },
    } },
    { id: "robot_pickup", title: "Robot pickup", status, calculator: { robot_pickup: { status: "Met", l1Branch: { designValueMm: "1000", requiredMinimumMm: "200", status: "Pass" }, l2Branch: { designValueMm: "1500.00", requiredMinimumMm: "1000.00", status: "Pass" } } } },
  ];
}
function report(items) {
  return { overall_result: "Feasible against the assessed PS4 requirements", candidate_reports: [{ id: "synthetic", label: "Synthetic primary", completeness: "complete", result: "Feasible against the assessed PS4 requirements", criteria: items, limitations: [] }] };
}
async function open(value) {
  api.getObject.mockResolvedValue({ number: "SIM-PIPE", description: "Synthetic pipe", types: [], material_brief: { sourcing: [] }, manufacturing_feasibility: value });
  await materialDialog("synthetic-ecn", controller.signal, vi.fn())("SIM-PIPE");
  return document.querySelector("dialog");
}
function groups(dialog) { return [...dialog.querySelectorAll(".drawing-feasibility-calculation-group")]; }
function freeze(value) { if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); } return value; }

it("collapses outer trace and all five native groups while retaining named status headers and heading hierarchy", async () => {
  const dialog = await open(report(criteria()));
  const outer = dialog.querySelector(".drawing-feasibility-calculations");
  expect(outer.tagName).toBe("DETAILS"); expect(outer.open).toBe(false);
  expect(outer.firstElementChild.tagName).toBe("SUMMARY");
  expect(outer.firstElementChild.querySelector("h4").textContent).toBe("Geometry and Plant A calculation trace");
  expect(groups(dialog)).toHaveLength(5);
  expect(groups(dialog).map(group => group.querySelector("h5").firstChild.textContent)).toEqual([
    "Material Designation", "Tube Length", "Bend Angles", "Automatic-Crimp Geometry", "Robot-Pickup Geometry",
  ]);
  for (const group of groups(dialog)) {
    expect(group.tagName).toBe("DETAILS"); expect(group.open).toBe(false); expect(group.hasAttribute("open")).toBe(false);
    const summary = group.firstElementChild;
    expect(summary.tagName).toBe("SUMMARY"); expect(summary.hasAttribute("role")).toBe(false);
    expect(summary.querySelector("h5")).not.toBeNull(); expect(summary.textContent).toContain("Result: Pass");
    const tag = summary.querySelector("ui5-tag");
    expect(tag.textContent).toBe("Pass"); expect(tag.getAttribute("design")).toBe("Positive");
    expect(tag.interactive).toBe(false); expect(tag.getAttribute("aria-hidden")).toBe("true");
    expect(summary.querySelector("button, a, input, [tabindex]")).toBeNull();
    expect(group.querySelector('[role="region"]').getAttribute("aria-label")).toContain("Plant A Calculation Trace.");
  }
  expect(dialog.querySelector(".drawing-feasibility-overall-tag").textContent).toBe("Feasible");
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});

it.each([["Pass", "Positive"], ["Fail", "Negative"], ["Cannot assess", "Critical"], ["Warning", "Critical"], ["Met", "Positive"], ["Not met", "Negative"], ["Unknown synthetic status", "Critical"], [undefined, "Critical"]])(
  "uses criterion status %s rather than inferring a result from passing calculator rows", async (status, design) => {
    const items = criteria(); items.forEach(item => { item.status = status; });
    const dialog = await open(report(items));
    for (const group of groups(dialog)) {
      const tag = group.firstElementChild.querySelector("ui5-tag");
      expect(tag.textContent).toBe(status || "Cannot assess"); expect(tag.getAttribute("design")).toBe(design);
      expect(group.open).toBe(false);
    }
    // Calculator rows remain exactly their own source results even if the criterion disagrees.
    const row = [...dialog.querySelectorAll(".drawing-feasibility-calculations tbody tr")].find(row => row.cells[0].textContent === "Total Tube Length");
    expect(row.querySelector("ui5-tag").textContent).toBe("Pass");
    expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
  },
);

it("expands only requested evidence without IO, row loss, source rewrites or report mutation", async () => {
  const value = freeze(report(criteria())); const before = JSON.stringify(value);
  const dialog = await open(value); const outer = dialog.querySelector(".drawing-feasibility-calculations");
  outer.open = true;
  const expectedRows = [
    ["PS4 Raw-Material Designation"], ["Outer Diameter", "Wall Thickness", "Total Tube Length"], ["Bend Angle 1", "Bend Angle 2"],
    ["Automatic-Crimp L1", "Automatic-Crimp L2.1", "Automatic-Crimp L3", "Automatic-Crimp L4", "Schedule Angle θ1", "Schedule Angle θ4", "L3 Derivation"],
    ["Robot-Pickup L1 Branch", "Robot-Pickup L2 Branch"],
  ];
  groups(dialog).forEach((group, index) => {
    const evidenceBefore = group.querySelector("table").innerHTML;
    group.open = true;
    expect(group.querySelector("table").innerHTML).toBe(evidenceBefore);
    expect([...group.querySelectorAll("tbody tr")].map(row => row.cells[0].textContent)).toEqual(expectedRows[index]);
    expect([...group.querySelectorAll("th")].map(cell => cell.textContent)).toEqual(["Measurement", "Drawing / Calculated Value", "Plant A Calculation Requirements", "Result"]);
    expect(group.firstElementChild.querySelector("ui5-tag").textContent).toBe("Pass");
    group.open = false;
  });
  expect(groups(dialog).every(group => !group.open)).toBe(true);
  for (const text of ["SYNTHETIC-PS4-SUFFIX", "Listed in the PS4 raw-material guideline", "629.6 mm", "200–1500 mm", "1000.00 mm", "P0 STRAIGHT PART", "B1 STRAIGHT PART", "B2 STRAIGHT PART", "90°", "120°", "PS4-SPEC-TEST lookup(1,000, 2,000)", "L3 = terminal PS4-SPEC-TEST + radius × θ4(rad)", "1178.53981633974483 mm", "≥ 1000.00 mm"]) expect(outer.textContent).toContain(text);
  expect(dialog.querySelector(".drawing-feasibility > h3").textContent).toContain("Plant A Manufacturing Feasibility");
  expect(value.candidate_reports[0].result).toBe("Feasible against the assessed PS4 requirements");
  expect(JSON.stringify(value)).toBe(before);
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});

it("resets disclosures and statuses for a locally selected configuration without fetching", async () => {
  const value = report(criteria());
  value.candidate_reports.push({ ...report(criteria("Fail")).candidate_reports[0], id: "second", label: "Synthetic second", result: "Not feasible against the PS4 requirements" });
  const dialog = await open(value); dialog.querySelector(".drawing-feasibility-calculations").open = true; groups(dialog)[0].open = true;
  const selector = dialog.querySelector('select[aria-label="Drawing Configuration"]'); selector.value = "1"; selector.dispatchEvent(new Event("change"));
  expect(dialog.querySelector(".drawing-feasibility-calculations").open).toBe(false);
  expect(groups(dialog).every(group => !group.open && group.firstElementChild.querySelector("ui5-tag").textContent === "Fail")).toBe(true);
  expect(dialog.querySelector(".drawing-feasibility-overall-tag").textContent).toBe("Not feasible");
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});

it.each([[[]], [[{ id: "tube_length", title: "Tube length", status: "Cannot assess" }]]])("does not fabricate geometry when calculators are absent", async items => {
  const dialog = await open(report(items));
  expect(dialog.querySelector(".drawing-feasibility-calculations")).toBeNull(); expect(groups(dialog)).toHaveLength(0);
  if (!items.length) expect(dialog.querySelector(".drawing-feasibility-overall-tag").textContent).toBe("Not assessed");
  expect(api.assessDrawing).not.toHaveBeenCalled();
});

it("keeps the native disclosure caret and keyboard semantics instead of custom replacement controls", () => {
  const css = readFileSync(resolve("src/style.css"), "utf8");
  expect(css).toContain(".drawing-feasibility-calculation-group > summary > h5 { display: inline;");
  expect(css).not.toMatch(/drawing-feasibility[^}]*summary[^}]*display:\s*(?:flex|grid|block|none)/);
  expect(css).not.toMatch(/drawing-feasibility[^}]*::(?:marker|-webkit-details-marker)[^}]*display:\s*none/);
});
