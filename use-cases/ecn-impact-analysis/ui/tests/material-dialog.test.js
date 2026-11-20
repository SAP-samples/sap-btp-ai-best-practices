import { beforeEach, expect, it, vi } from "vitest";
import {
  assessmentConclusion,
  assessmentDesign,
  assessmentSummary,
  endDateLine,
  hierarchyNodes,
  materialDialog,
  sourceLabel,
  validity,
} from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";
vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));


beforeEach(() => {
  document.body.replaceChildren(); vi.clearAllMocks();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});

it("renders canonical overall results with their semantic tag designs", () => {
  expect(assessmentDesign("Feasible against the assessed PS4 requirements")).toBe("Positive");
  expect(assessmentDesign("Not feasible against the PS4 requirements")).toBe("Negative");
  expect(assessmentDesign("Warning")).toBe("Critical");
});

it("uses the highest-priority assessed finding for the one-line conclusion", () => {
  expect(assessmentConclusion("Not feasible against the PS4 requirements", [
    { title: "Tube length", status: "Pass", message: "1000 mm meets the documented 200–1500 mm range." },
    { title: "End type", status: "Fail", message: "Unsupported end configuration specified." },
  ])).toBe("Not feasible: Unsupported end configuration specified.");
  expect(assessmentConclusion("Inconclusive", [
    { title: "Material designation", status: "Cannot assess", message: "Material designation cannot be confirmed." },
  ])).toBe("Inconclusive: Material designation cannot be confirmed.");
});

it("keeps a compatible end option as a warning rather than a failure", () => {
  const compatibilityDetail = "Compatible option: CAST B4. Non-compatible options: Parker EO2, Parker EO, CAST B3.";
  const summary = assessmentSummary([
    { id: "end_configuration", title: "End configuration", status: "Warning", explanation: compatibilityDetail },
    { id: "prohibited_EO-2 soft seal", title: "Prohibited feature: EO-2 soft seal", status: "Warning", explanation: "" },
    { id: "prohibited_metal seal", title: "Prohibited feature: metal seal", status: "Warning", explanation: "" },
  ]);

  expect(summary).toEqual(expect.arrayContaining([
    expect.objectContaining({ title: "End Type", status: "Warning", message: compatibilityDetail }),
    expect.objectContaining({ title: "Prohibited Features", status: "Warning", message: "See End type for the compatible and non-compatible end options." }),
  ]));
  expect(assessmentConclusion("Feasible against the assessed PS4 requirements", summary))
    .toBe(`Warning: ${compatibilityDetail}`);
});

it("groups detailed PS4 criteria into clear decision-summary categories", () => {
  const summary = assessmentSummary([
    { id: "raw_material", title: "Raw material", status: "Cannot assess", explanation: "", calculator: { raw_material: { designation: "SFS-EN_10305-4_-_E235", guidelineCoverage: "not_listed" } } },
    { id: "tube_length", title: "Tube length", status: "Pass", explanation: "", calculator: { tube_length: { diameterMm: "10", wallThicknessMm: "1", totalLengthMm: "1000", minimumMm: "200", maximumMm: "1500" } } },
    { id: "end_configuration", title: "End configuration", status: "Fail", explanation: "" },
    { id: "prohibited_banjo screw", title: "Prohibited feature: banjo screw", status: "Pass", explanation: "" },
    { id: "prohibited_metal seal", title: "Prohibited feature: metal seal", status: "Fail", explanation: "" },
    { id: "bend_angles", title: "Bend angles", status: "Pass", explanation: "", calculator: { bend_angles: { results: [{ angleDeg: "90" }, { angleDeg: "120" }] } } },
    { id: "automatic_crimp_dimensions", title: "Automatic-crimp dimensions", status: "Warning", explanation: "" },
    { id: "automatic_crimp_objective", title: "Automatic-crimp objective", status: "Warning", explanation: "" },
    { id: "robot_pickup", title: "Robot pickup", status: "Met", explanation: "" },
  ]);

  expect(summary).toEqual(expect.arrayContaining([
    expect.objectContaining({ title: "Material Designation", status: "Cannot assess", message: "SFS-EN_10305-4_-_E235 is not listed; suitability cannot be confirmed." }),
    expect.objectContaining({ title: "Tube Length", status: "Pass", message: "1000 mm meets the documented 200–1500 mm range." }),
    expect.objectContaining({ title: "End Type", status: "Fail", message: "Unsupported end configuration specified." }),
    expect.objectContaining({ title: "Prohibited Features", status: "Fail", message: "Metal seal specified." }),
    expect.objectContaining({ title: "Minimum Straight Lengths", status: "Warning", message: "Automatic-crimp geometry cannot be fully assessed." }),
    expect.objectContaining({ title: "Automatic-Crimp Objective", status: "Warning", message: "Automatic-crimp geometry cannot be fully assessed." }),
  ]));
});

it("formats calendar dates and source indices without timezone shifts or missing-value guesses", () => {
  expect(validity("2026-03-13")).toBe("13 Mar 2026");
  expect(validity("9999-12-31")).toBe("No planned end");
  expect(validity(null, "MISSING", null)).toBe("Not supplied");
  expect(validity(null, "INVALID", "03/04/xx")).toBe("Could not interpret");
  expect(validity(null, "LEGACY_PLACEHOLDER", "01.01.9999")).toBe("Legacy placeholder");
  expect(sourceLabel("$[0].ImplementationPlan[0].ResultingLinks[18].ResultingObjects[0]")).toBe("Implementation plan 1 › Resulting link 19 › Object 1");
  expect(sourceLabel("$.ImplementationPlan[0].ResultingLinks[1].ResultingObjects[0]")).toBe("Implementation plan 1 › Resulting link 2 › Object 1");
  expect(sourceLabel("$[1].CNAffectLinks[0].AffectedObjects[1]")).toBe("Affected link 1 › Object 2");
});

it("renders observed BOM coverage without raw source rows or inferred mappings", async () => {
  const path = "$[0].ImplementationPlan[0].ResultingLinks[18].ResultingObjects[0]";
  api.getObject.mockImplementation(async (_, params) => ({ number: "SIM-PART", description: "Long synthetic description", types: [], assessment_date: "2026-03-15",
    locations: [{ sheet: "EBOM", row: params.page === 1 ? 42 : 67, path: [], valid_from_date: "2026-03-13", valid_to_date: "9999-12-31" }], location_total: 26,
    material_brief: {
      bom_coverage: {
        engineering_occurrences: 1, manufacturing_occurrences: 1, coverage: "OBSERVED_IN_BOTH", plants: [], effective_interval_count: 1,
        occurrences: [
          { source: "Enriched EBOM", kind: "ENRICHED_EBOM", path: [{ number: "Q900V", description: "Q900V tractor" }, { number: "SIM-PART", description: "Long synthetic description" }], item_or_position: "0010", quantity: "2", unit: "PC", valid_from_date: "2026-03-13", valid_to_date: "9999-12-31", change_number: "ECN1" },
          { source: "130% MBOM", kind: "MBOM_130", path: [{ number: "Q900V", description: "Q900V tractor" }, { number: "SIM-PART", description: "Long synthetic description" }], item_or_position: "10", quantity: "2", unit: null, valid_from_date: "2026-03-13", valid_to_date: "9999-12-31", change_number: "ECN1" },
        ],
      },
      sourcing: [{ plant_code: "PK01", plant_name: "Plant B", procurement_type_labels: ["In-house and external procurement allowed"], source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", net_price: "0.85", currency: "EUR", price_per: "1", order_price_unit: "PC", matching_source_row_count: 1 } }],
      cost_assumption: "Cost values are illustrative demonstration data.",
    },
    reference_records: [], reference_total: 0, source_references: params.page === 1 ? [{ role: "RESULTING", path, revision: "1", state: "APPROVEDFORSERIES" }] : [], source_reference_total: 1 }));
  const opener = document.createElement("button"); document.body.append(opener); opener.focus();
  const controller = new AbortController(); const inspect = materialDialog("id", controller.signal, vi.fn()); await inspect("SIM-PART");
  const dialog = document.querySelector("dialog");
  expect(api.getObject).toHaveBeenCalledTimes(1);
  expect(dialog.textContent).toContain("Observed BOM Coverage");
  expect(dialog.querySelector("td[data-label='BOM presence']").textContent).toBe("Found in EBOM and MBOM");
  expect(dialog.textContent).toContain("Enriched EBOM: 1");
  expect(dialog.textContent).toContain("130% MBOM: 1");
  expect(dialog.textContent).toContain("Show 2 observed BOM occurrences");
  expect(dialog.querySelector(".bom-coverage-table td[data-label='Plant']").textContent).toBe("Plant not supplied");
  expect(dialog.textContent).not.toContain("Resulting link 19");
  expect(dialog.querySelector(".material-evidence")).toBeNull();
  expect(dialog.textContent).toContain("Make or buy allowed");
  expect(dialog.textContent).toContain("Source Purchase Price");
  expect(dialog.textContent).toContain("Lead Time");
  expect(dialog.textContent).toContain("Not Supplied");
  expect(dialog.textContent).toContain("€0.85 / 1 PC");
  expect(dialog.textContent).not.toContain("Mapped");
  expect(dialog.textContent).not.toContain("Split");
  expect(dialog.textContent).not.toContain("illustrative demonstration data");
  expect(dialog.textContent).not.toContain("No material, purchasing or stock records");
  expect(dialog.textContent).not.toContain("row 42");
  expect(dialog.textContent).not.toContain("Technical Source Details");
  expect(dialog.textContent).not.toContain(path);
  expect(dialog.textContent).not.toContain("APPROVEDFORSERIES");
  document.querySelector("footer button").click(); expect(document.activeElement).toBe(opener); controller.abort();
});

it("shows why procurement permission is unavailable rather than hiding the section", async () => {
  api.getObject.mockResolvedValue({
    number: "MAT-UNMAPPED", description: "No material-master source", types: [],
    material_brief: { bom_coverage: { occurrences: [] }, sourcing: [] },
    locations: [], location_total: 0, reference_records: [], reference_total: 0,
    source_references: [], source_reference_total: 0,
  });

  const controller = new AbortController();
  await materialDialog("id", controller.signal, vi.fn())("MAT-UNMAPPED");
  const text = document.querySelector("dialog").textContent;
  expect(text).toContain("Procurement Permission");
  expect(text).toContain("No source-backed procurement permission is available for this material.");
  controller.abort();
});

it("renders source-price review states without choosing an ambiguous price", async () => {
  api.getObject.mockResolvedValue({
    number: "PIPE-1", description: "Synthetic pipe", types: [],
    material_brief: {
      bom_coverage: { occurrences: [] },
      sourcing: [
        { plant_code: "PK01", procurement_type_labels: ["External procurement"], source_purchase_price: { status: "MULTIPLE_SOURCE_PRICES_REVIEW", matching_source_row_count: 2 } },
        { plant_code: "PL10", procurement_type_labels: ["External procurement"], source_purchase_price: { status: "ZERO_PRICE_REVIEW", matching_source_row_count: 1 } },
        { plant_code: "DE01", procurement_type_labels: ["External procurement"], source_purchase_price: { status: "NOT_SUPPLIED", matching_source_row_count: 0 } },
        { plant_code: "ES01", procurement_type_labels: ["In-house production"], source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", net_price: "9.50", currency: "EUR", price_per: "1", order_price_unit: "PC" } },
      ],
    },
    locations: [], location_total: 0, reference_records: [], reference_total: 0,
    source_references: [], source_reference_total: 0,
  });

  const controller = new AbortController();
  await materialDialog("id", controller.signal, vi.fn())("PIPE-1");
  const text = document.querySelector("dialog").textContent;
  expect(text).toContain("Multiple Source Prices — Review");
  expect(text).toContain("Zero Source Price — Review");
  expect(text).toContain("Not Supplied");
  expect(text).toContain("Not Allowed");
  expect(text).not.toContain("€9.50 / 1 PC");
  controller.abort();
});

it("removes the entire evidence section, including source and supporting-object tables", async () => {
  api.getObject.mockResolvedValue({
    number: "MAT-1", description: "Source-record test material", types: [],
    material_brief: { bom_coverage: { occurrences: [] }, sourcing: [] },
    locations: [], location_total: 0,
    reference_records: [{
      dataset_type: "PURCHASE_INFO", file_name: "Purchase Info.xlsx", sheet_name: "Sheet1", source_row_number: 66,
      plant_code: "PK01", plant_status: "ACTIVE", supplier_name: "Example Supplier", unrestricted_quantity: "12",
      storage_location: "A001", snapshot_date: "2026-03-15", procurement_type: "External procurement",
    }],
    reference_total: 1,
    source_references: [{ role: "RESULTING", path: "$.ImplementationPlan[0].ResultingLinks[39].ResultingObjects[0]", revision: "1", state: "APPROVEDFORSERIES" }],
    source_reference_total: 1,
    change: { supporting_objects: [{ number: "MAT-1.ASM", type: "Assembly", revision: "1", state: "APPROVEDFORSERIES" }] },
  });

  const controller = new AbortController();
  await materialDialog("id", controller.signal, vi.fn())("MAT-1");
  const dialog = document.querySelector("dialog");

  expect(dialog.querySelector(".material-evidence")).toBeNull();
  expect(dialog.textContent).not.toContain("Evidence");
  expect(dialog.textContent).not.toContain("Source Records");
  expect(dialog.textContent).not.toContain("ECN Source References");
  expect(dialog.textContent).not.toContain("Supporting Objects");
  expect(dialog.textContent).not.toContain("Purchase Info.xlsx");
  expect(dialog.textContent).not.toContain("Example Supplier");
  expect(dialog.textContent).not.toContain("Resulting link 40");
  expect(dialog.textContent).not.toContain("MAT-1.ASM");
  controller.abort();
});

it("renders an automatically stored drawing report without requiring a manual upload", async () => {
  api.getObject.mockResolvedValue({
    number: "QKX5525860", description: "Automated pipe", types: [],
    material_brief: { bom_coverage: { occurrences: [] }, sourcing: [] },
    locations: [], location_total: 0, reference_records: [], reference_total: 0,
    source_references: [], source_reference_total: 0,
    manufacturing_feasibility: {
      overall_result: "Feasible against the assessed PS4 requirements",
      scope_notice: "Assessment scope",
      candidate_reports: [{
        id: "fixture", label: "Primary", completeness: "complete",
        result: "Feasible against the assessed PS4 requirements", criteria: [], limitations: [],
      }],
    },
  });

  const controller = new AbortController();
  await materialDialog("id", controller.signal, vi.fn())("QKX5525860");
  const dialog = document.querySelector("dialog");
  expect(dialog.textContent).toContain("Plant A Manufacturing Feasibility");
  expect(dialog.textContent).toContain("Not assessed: No assessment criteria are available for this configuration.");
  expect(dialog.querySelector(".drawing-feasibility-overall-tag").textContent).toBe("Not assessed");
  expect(dialog.querySelector('input[aria-label="Pipe Drawing File"]')).toBeNull();
  expect(dialog.querySelector(".drawing-preview-input")).not.toBeNull();
  expect(dialog.querySelector(".engineering-drawing-preview").textContent).not.toContain("No viewable drawing");
  expect(api.assessDrawing).not.toHaveBeenCalled();
  controller.abort();
});

it("uploads a transient drawing inside the material popup and renders the controlled report as text", async () => {
  api.getObject.mockResolvedValue({
    number: "PIPE-1", description: "Synthetic pipe", types: [], material_brief: { bom_coverage: { engineering_occurrences: 0, manufacturing_occurrences: 0, coverage: "NOT_OBSERVED", plants: [], effective_interval_count: 0, occurrences: [] }, sourcing: [] },
    locations: [], location_total: 0, reference_records: [], reference_total: 0, source_references: [], source_reference_total: 0,
  });
  api.assessDrawing.mockResolvedValue({
    overall_result: "Inconclusive", scope_notice: "Assessment only <img src=x onerror=alert(1)>", extraction_limitations: ["Unclear annotation"], guideline_limitations: ["Guideline disclaimer"],
    candidate_reports: [{ id: "c1", label: "Primary", completeness: "partial", result: "Inconclusive", limitations: ["Configuration-dependent end compatibility"], criteria: [
      { id: "tube_length", title: "Tube length", status: "Fail", explanation: "Calculator-backed length assessment", calculator: { tube_length: { diameterMm: "10", wallThicknessMm: "1", totalLengthMm: "1781.7", minimumMm: "200", maximumMm: "1500", status: "Fail" } }, evidence: [{ page: "1", source_label: "Length callout <script>bad()</script>", raw_text: "SECRET DRAWING TEXT" }] },
      { id: "automatic_crimp_dimensions", title: "Automatic-crimp dimensions", status: "Pass", explanation: "Calculator-backed geometry assessment", calculator: {
        schedule_mapping: {
          l1: { sourcePoint: "P0", sourceField: "STRAIGHT PART", valueMm: "100" },
          l2Segments: [{ sourcePoint: "B1", sourceField: "STRAIGHT PART", valueMm: "100" }],
          l3: { l3Mm: "178.53981633974483", formula: "L3 = terminal straight part + terminal bend radius × θ4(rad)" },
          l4: { sourcePoint: "B2", sourceField: "STRAIGHT PART", valueMm: "100" },
          theta1: { sourcePoint: "B1", sourceField: "ANGLE", valueDeg: "90" },
          theta4: { sourcePoint: "B2", sourceField: "ANGLE", valueDeg: "90" },
        },
        auto_crimp: { overallStatus: "Pass", results: {
          l1: { designValueMm: "100", requiredMinimumMm: "32.9", formulaOrLookup: "table L1 minimum for θ1 ≤ 90", status: "Pass" },
          l2: { segments: [{ designValueMm: "100", requiredMinimumMm: "15", formulaOrLookup: "table L2 minimum", status: "Pass" }] },
          l3: { designValueMm: "178.53981633974483", requiredMinimumMm: "96.4", formulaOrLookup: "table L3 minimum for 85 < θ4 < 180", status: "Pass" },
          l4: { designValueMm: "100", requiredMinimumMm: "33.9", formulaOrLookup: "table L4 minimum for θ4 ≤ 120", status: "Pass" },
        } },
      }, evidence: [] },
      { id: "robot_pickup", title: "Robot pickup", status: "Met", explanation: "Calculator-backed pickup assessment", calculator: { robot_pickup: {
        status: "Met", l1Branch: { designValueMm: "100", requiredMinimumMm: "66.9", status: "Pass" }, l2Branch: { designValueMm: "100", requiredMinimumMm: "34", status: "Pass" },
      } }, evidence: [] },
    ] }],
  });
  const controller = new AbortController(); await materialDialog("id", controller.signal, vi.fn())("PIPE-1");
  const dialog = document.querySelector("dialog");
  const input = dialog.querySelector(".drawing-preview-input");
  const file = new File(["drawing"], "pipe.png", { type: "image/png" });
  Object.defineProperty(input, "files", { configurable: true, value: [file] });
  input.dispatchEvent(new Event("change"));
  [...dialog.querySelectorAll("button")].find(button => button.textContent === "Analyze Drawing").click();
  await vi.waitFor(() => expect(api.assessDrawing).toHaveBeenCalledWith(file, api.getObject.mock.calls[0][2]));
  await vi.waitFor(() => expect(dialog.textContent).toContain("Assessment complete."));

  expect(dialog.textContent).toContain("Plant A Manufacturing Feasibility");
  const feasibilityHeadingIcon = dialog.querySelector(".drawing-feasibility > h3 .ai-attribution-icon");
  expect(feasibilityHeadingIcon?.getAttribute("accessible-name")).toBe("AI-assisted drawing assessment");
  expect(dialog.textContent).toContain("Inconclusive");
  expect(dialog.querySelector('ui5-tag[size="L"]')).toBeNull();
  expect(dialog.querySelector(".drawing-feasibility-assessment-summary").textContent).toBe("Inconclusive: Material designation cannot be confirmed.");
  expect(dialog.querySelector('.drawing-feasibility-candidates > ui5-tag[aria-label^="Configuration Primary"]')).toBeNull();
  expect([...dialog.querySelectorAll("h4")].some(node => node.textContent === "Primary")).toBe(false);
  const calculationTrace = [...dialog.querySelectorAll("details")]
    .find(details => details.querySelector("summary")?.textContent === "Geometry and Plant A calculation trace");
  expect(calculationTrace.open).toBe(false);
  expect(calculationTrace.querySelector("summary .ai-attribution-icon")?.getAttribute("accessible-name"))
    .toBe("AI-extracted drawing inputs and calculator-derived Plant A results");
  expect(calculationTrace.textContent).not.toContain("AI extracts the drawing inputs;");
  const tubeRows = [...calculationTrace.querySelectorAll("tr")]
    .filter(row => row.textContent.includes("Outer Diameter") || row.textContent.includes("Wall Thickness") || row.textContent.includes("Total Tube Length"));
  expect(tubeRows.map(row => row.querySelector("ui5-tag").textContent)).toEqual(["Input", "Input", "Fail"]);
  expect(tubeRows[2].textContent).toContain("1781.7 mm");

  expect(calculationTrace.textContent).toContain("200–1500 mm");
  expect(calculationTrace.textContent).toContain("Plant A Calculation Requirements");
  expect(calculationTrace.textContent).toContain("Automatic-Crimp L1");
  expect(calculationTrace.textContent).toContain("≥ 32.9 mm");
  expect(calculationTrace.textContent).toContain("P0 STRAIGHT PART");
  expect(calculationTrace.textContent).toContain("L3 = terminal straight part + terminal bend radius × θ4(rad)");
  expect(calculationTrace.textContent).toContain("Robot-Pickup L2 Branch");
  expect(calculationTrace.textContent).toContain("≥ 34 mm");
  expect(dialog.textContent).not.toContain("Length callout <script>bad()</script>");
  expect(dialog.textContent).not.toContain("Configuration-dependent end compatibility");
  expect(dialog.textContent).not.toContain("Unclear annotation");
  expect(dialog.textContent).not.toContain("Guideline disclaimer");
  expect(dialog.querySelector(".drawing-feasibility-evidence")).toBeNull();
  expect(dialog.querySelector(".drawing-feasibility-summary-table")).not.toBeNull();
  expect(dialog.querySelector(".drawing-feasibility-summary-table").getAttribute("aria-label")).toBe("PS4 Decision Summary for Primary.");
  const summaryHeaderIcons = [...dialog.querySelectorAll(".drawing-feasibility-summary-table thead .ai-attribution-icon")];
  expect(summaryHeaderIcons.map(icon => icon.getAttribute("accessible-name"))).toEqual([
    "AI-assisted PS4 result",
    "AI-assisted PS4 key finding",
  ]);
  expect(dialog.querySelectorAll(".drawing-feasibility-summary-table tbody .ai-attribution-icon")).toHaveLength(0);
  const summaryHeaders = [...dialog.querySelectorAll(".drawing-feasibility-summary-table th")]
    .map(header => header.textContent);
  expect(summaryHeaders).toEqual(["PS4 Check", "Result", "Key Finding"]);
  expect(dialog.querySelector(".drawing-feasibility-status-label")).toBeNull();
  expect(dialog.querySelector(".drawing-feasibility-overall ui5-tag")).toBeNull();
  expect([...dialog.querySelectorAll("button")].some(button => button.textContent === "Analyze Drawing")).toBe(true);
  const calculationHeaders = [...calculationTrace.querySelectorAll("table")]
    .map(table => [...table.querySelectorAll("th")].map(header => header.textContent));
  expect(calculationHeaders.every(headers => JSON.stringify(headers) === JSON.stringify([
    "Measurement", "Drawing / Calculated Value", "Plant A Calculation Requirements", "Result",
  ]))).toBe(true);
  const calculationHeaderIcons = [...calculationTrace.querySelectorAll("thead .ai-attribution-icon")];
  expect(calculationHeaderIcons.map(icon => icon.getAttribute("accessible-name"))).toEqual(
    Array(calculationHeaders.length).fill("AI-extracted drawing input or calculator-derived value"),
  );
  expect(calculationTrace.querySelectorAll("tbody .ai-attribution-icon")).toHaveLength(0);
  expect(dialog.querySelector("script")).toBeNull();
  expect(dialog.querySelector("img")).toBeNull();
  expect(dialog.textContent).not.toContain("SECRET DRAWING TEXT");
  controller.abort();
});

it("keeps drawing-assessment failures local to the material popup", async () => {
  api.getObject.mockResolvedValue({ number: "PIPE-2", description: "Synthetic pipe", types: [], material_brief: { bom_coverage: { engineering_occurrences: 0, manufacturing_occurrences: 0, coverage: "NOT_OBSERVED", plants: [], effective_interval_count: 0, occurrences: [] }, sourcing: [] }, locations: [], location_total: 0, reference_records: [], reference_total: 0, source_references: [], source_reference_total: 0 });
  api.assessDrawing.mockRejectedValue(new Error("Drawing analysis unavailable"));
  const onError = vi.fn(); const controller = new AbortController(); await materialDialog("id", controller.signal, onError)("PIPE-2");
  const dialog = document.querySelector("dialog"); const input = dialog.querySelector(".drawing-preview-input");
  Object.defineProperty(input, "files", { configurable: true, value: [new File(["drawing"], "pipe.png", { type: "image/png" })] });
  input.dispatchEvent(new Event("change"));
  [...dialog.querySelectorAll("button")].find(button => button.textContent === "Analyze Drawing").click();
  await vi.waitFor(() => expect(dialog.textContent).toContain("Drawing analysis unavailable"));
  expect(onError).toHaveBeenCalledTimes(1);
  expect(input.disabled).toBe(false);
  controller.abort();
});

it("tells the material, ECN change, observed coverage, and sourcing story before optional evidence", async () => {
  api.getObject.mockResolvedValue({
    number: "NEW-DECAL", description: "3-point control warning decal", types: [],
    change: {
      action: "PHASE_IN", usage: { label: "Effective usage observed in supplied BOMs" },
      counterpart_details: [{ number: "OLD-DECAL", description: "Previous 3-point control warning decal" }],
      counterparts: ["OLD-DECAL"], qualifiers: [],
    },
    material_brief: {
      bom_coverage: {
        engineering_occurrences: 0, manufacturing_occurrences: 1, coverage: "MANUFACTURING_ONLY", plants: [], effective_interval_count: 1,
        occurrences: [{ source: "130% MBOM", kind: "MBOM_130", path: [{ number: "PANEL-1", description: "Control-panel assembly" }, { number: "NEW-DECAL", description: "3-point control warning decal" }], item_or_position: "0030", quantity: "2", unit: null, valid_from_date: "2026-05-19", valid_to_date: "9999-12-31", change_number: "ECN345274" }],
      },
      sourcing: [{ plant_code: "PL10", plant_name: "Plant A", procurement_type_labels: ["External procurement"], purchase_prices: [] }],
      cost_assumption: null,
    },
    locations: [], location_total: 0, reference_records: [], reference_total: 0, source_references: [], source_reference_total: 0,
  });

  const controller = new AbortController(); await materialDialog("id", controller.signal, vi.fn())("NEW-DECAL");
  const dialog = document.querySelector("dialog");

  expect(dialog.querySelector("h2").textContent).toBe("3-point control warning decal");
  expect(dialog.textContent).toContain("Material: NEW-DECAL");
  expect(dialog.textContent).toContain("Phase in this material; replaces Previous 3-point control warning decal (OLD-DECAL).");
  const coverageSection = [...dialog.querySelectorAll(".material-story-block")]
    .find(section => section.querySelector("h3")?.textContent === "Observed BOM Coverage");
  const coverageTable = coverageSection.querySelector(".bom-coverage-table");
  expect(coverageTable).not.toBeNull();
  expect(coverageTable.tabIndex).toBe(0);
  expect(coverageTable.getAttribute("role")).toBe("region");
  expect(coverageTable.getAttribute("aria-label")).toBe("Observed BOM coverage table.");
  expect(coverageSection.textContent).toContain("130% MBOM: 1");
  expect(coverageTable.querySelector("td[data-label='BOM presence']").textContent).toBe("MBOM only");
  expect(coverageSection.textContent).toContain("2");
  expect(coverageSection.textContent).toContain("19 May 2026");
  expect(coverageSection.textContent).not.toContain("Plant A");
  const occurrenceTable = coverageSection.querySelector(".observed-occurrences-table");
  expect(occurrenceTable.tabIndex).toBe(0);
  expect(occurrenceTable.getAttribute("role")).toBe("region");
  expect(occurrenceTable.getAttribute("aria-label")).toBe("Observed BOM occurrences. Scroll for additional columns or rows.");
  expect(dialog.textContent).not.toContain("Usage & Validity");
  expect(dialog.textContent).not.toContain("Technical Source Details");
  expect(dialog.textContent).toContain("Buy allowed");
  expect(dialog.textContent).not.toContain("Not allowed");
  expect(dialog.textContent).not.toContain("Not supplied");
  expect(dialog.textContent).not.toContain("Evidence");
  controller.abort();
});

/* MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: curated Make-vs-Buy radar test.
it("renders a sourced Make vs Buy profile with two scored options", async () => {
  const assessment = {
    plant_code: "PL10", plant_name: "Plant A", recommendation: "BUY_PENDING_QUOTE",
    recommendation_summary: "A supplier quote is required before release.", make_cost: "",
    currency: "EUR", price_per: "1", unit_of_measure: "PC", buy_cost: null,
    impacted_plant_count: 1, comparable_history: "Comparable decals are externally sourced for Plant A.",
    tooling: "The station can apply the new decal; no in-house print tooling is available.",
    capacity: "Application capacity is available after a short reprogramming changeover.",
  };
  api.getObject.mockResolvedValue({
    number: "NEW-DECAL", description: "3-point control warning decal", types: [],
    material_brief: { linked_use: [], sourcing: [], make_buy_assessments: [assessment] },
    locations: [], location_total: 0, reference_records: [], reference_total: 0, source_references: [], source_reference_total: 0,
  });

  const controller = new AbortController(); await materialDialog("id", controller.signal, vi.fn())("NEW-DECAL");
  const scores = { make: [2, 4, 5, 2, 1], buy: [3, 3, 5, 5, 4] };

  expect(makeBuyScores(assessment)).toEqual(scores);
  expect(document.querySelector(".make-buy-radar").getAttribute("aria-label")).toContain("Make versus buy comparison for Plant A (PL10)");
  expect(document.querySelector("dialog").textContent).toContain("Buy pending quote: A supplier quote is required before release.");
  expect(charts.makeBuyRadarOption).toHaveBeenCalledWith(scores);
  expect(charts.initChart).toHaveBeenCalledTimes(1);
  controller.abort();
});
MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: curated Make-vs-Buy radar test. */

const node = (number, label, level = null) => ({ number, label, level });
const ebomPath = [node("Q900V", "Main", "1"), node("Q900VS", "Sub", "2"), node("VC1", "VC", "3"), node("L4", "Level 4", "4"), node("PART", "Object", "5")];

it("preserves every source hierarchy node without derived configuration tags", () => {
  expect(hierarchyNodes({ kind: "ENRICHED_EBOM", path: ebomPath })).toEqual(ebomPath.map(node => ({ node, tag: "" })));
  const mbom = [node("PARENT", "Main"), node("PART", "Object")];
  expect(hierarchyNodes({ kind: "MBOM_130", path: mbom })).toEqual(mbom.map(node => ({ node, tag: "" })));
});

it("omits the end date when there is no planned end and words the start and end dates plainly", () => {
  expect(endDateLine({ valid_to_date: "9999-12-31", valid_to: "9999-12-31T00:00:00" })).toBe("");
  expect(endDateLine({ valid_to_date: null, valid_to_quality: "MISSING", valid_to: null })).toBe("");
  expect(endDateLine({ valid_to_date: "2022-04-20", valid_to_quality: "VALID", valid_to: "2022-04-20" })).toBe("End date: 20 Apr 2022");
  expect(endDateLine({ valid_to_date: null, valid_to_quality: "INVALID", valid_to: "xx" })).toBe("End date: Could not interpret");
});

it("renders static source hierarchy and effective dates without technical source details", async () => {
  api.getObject.mockResolvedValue({ number: "PART", description: "d", types: [], assessment_date: "2026-03-15",
    material_brief: { bom_coverage: {
      engineering_occurrences: 1, manufacturing_occurrences: 0, coverage: "ENGINEERING_ONLY", plants: [], effective_interval_count: 1,
      occurrences: [{ source: "Enriched EBOM", kind: "ENRICHED_EBOM", path: ebomPath, item_or_position: "0010", quantity: "1", unit: "PC", valid_from_date: "2022-02-25", valid_to_date: "9999-12-31", change_number: "ECN345274" }],
    }, sourcing: [], make_buy_assessments: [] },
    locations: [{ kind: "ENRICHED_EBOM", sheet: "EBOM", row: 1, path: ebomPath, valid_from_date: "2022-02-25", valid_from_quality: "VALID", valid_to_date: "9999-12-31" }], location_total: 1,
    reference_records: [], reference_total: 0, source_references: [], source_reference_total: 0 });
  const controller = new AbortController(); await materialDialog("id", controller.signal, vi.fn())("PART");
  const dialog = document.querySelector("dialog");
  const text = dialog.textContent;
  expect(text).toContain("25 Feb 2022");
  expect(text).toContain("No planned end");
  expect(dialog.querySelector("td[data-label='BOM presence']").textContent).toBe("EBOM only");
  expect(text).not.toMatch(/Start date|Valid from|Valid to|Technical Source Details|row 1/);
  expect([...document.querySelectorAll(".hierarchy-line")].map(line => line.textContent)).toEqual(["Q900V", "Q900VS", "VC1", "L4", "PART"]);
  expect(document.querySelectorAll(".hierarchy-line button")).toHaveLength(0);
  expect(dialog.open).toBe(true);
  controller.abort();
});
