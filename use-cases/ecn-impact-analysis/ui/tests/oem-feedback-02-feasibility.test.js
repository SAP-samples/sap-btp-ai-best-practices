import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { assessmentConclusion, materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";

vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));
const feasible = "Feasible against the assessed PS4 requirements";
const notFeasible = "Not feasible against the PS4 requirements";
const criteria = () => [
  { id: "raw_material", title: "Raw material", status: "Pass", calculator: { raw_material: { designation: "SYNTHETIC", guidelineCoverage: "listed" } } },
  { id: "tube_length", title: "Tube length", status: "Pass", calculator: { tube_length: { totalLengthMm: "1000", minimumMm: "200", maximumMm: "1500", status: "Pass" } } },
  { id: "end_configuration", title: "End configuration", status: "Pass" },
  { id: "prohibited_fixture", title: "Prohibited feature: fixture", status: "Pass" },
  { id: "bend_angles", title: "Bend angles", status: "Pass" },
  { id: "automatic_crimp_dimensions", title: "Automatic crimp", status: "Pass" },
  { id: "automatic_crimp_objective", title: "Automatic crimp objective", status: "Pass" },
  { id: "robot_pickup", title: "Robot pickup", status: "Met" },
];
const candidate = (result, label = "Synthetic primary") => ({ id: label, label, result, criteria: criteria(), completeness: "complete", limitations: [] });
const report = (...candidates) => ({ overall_result: "Configuration-dependent", candidate_reports: candidates });
let controller;
beforeEach(() => {
  document.body.replaceChildren(); vi.resetAllMocks(); controller = new AbortController();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});
afterEach(() => controller.abort());
async function open(manufacturing_feasibility) {
  api.getObject.mockResolvedValue({ number: "SIM-PIPE", description: "Synthetic pipe", types: [], material_brief: { sourcing: [] }, manufacturing_feasibility });
  const onError = vi.fn();
  await materialDialog("synthetic-ecn", controller.signal, onError)("SIM-PIPE");
  return { dialog: document.querySelector("dialog"), onError };
}
const tag = () => document.querySelector(".drawing-feasibility-overall-tag");
const meaning = () => document.querySelector(".drawing-feasibility-status-description").textContent;

it.each([
  [feasible, "Feasible", "Positive"],
  [notFeasible, "Not feasible", "Negative"],
  ["Inconclusive", "Inconclusive", "Critical"],
  ["Configuration-dependent", "Configuration-dependent", "Information"],
  ["Unrecognized result", "Unknown", "Critical"],
  [undefined, "Unknown", "Critical"],
])("shows canonical selected result %s as a read-only heading tag", async (result, text, design) => {
  const { dialog } = await open(report(candidate(result)));
  expect(tag().textContent).toBe(text);
  expect(tag().getAttribute("design")).toBe(design);
  expect(tag().closest("h3")).toBe(dialog.querySelector(".drawing-feasibility > h3"));
  expect(tag().interactive).toBe(false);
  expect(tag().hasAttribute("interactive")).toBe(false);
  expect(tag().hasAttribute("tabindex")).toBe(false);
  expect(tag().getAttribute("aria-hidden")).toBe("true");
  expect(tag().getAttribute("wrapping-type")).toBe("Normal");
  expect(meaning()).toContain("Overall manufacturing feasibility for Synthetic primary:");
  expect(meaning()).toContain("does not replace manufacturing approval");
  if (text === "Unknown") expect(meaning()).toContain("no recognized assessment result supplied");
  else expect(meaning()).toContain(result);
  const status = tag().parentElement;
  expect(status.getAttribute("role")).toBe("status");
  expect(status.getAttribute("aria-live")).toBe("polite");
  expect(status.getAttribute("aria-atomic")).toBe("true");
  expect(dialog.querySelectorAll(".drawing-feasibility-summary-table tbody tr")).toHaveLength(8);
  tag().click();
  expect(api.getObject).toHaveBeenCalledTimes(1);
  expect(api.assessDrawing).not.toHaveBeenCalled();
  expect(dialog.querySelector(".drawing-feasibility > h3 .ai-attribution-icon").getAttribute("accessible-name"))
    .toBe("AI-assisted drawing assessment");
});

it.each([undefined, null, {}, { overall_result: feasible, candidate_reports: [] }, report({ ...candidate(feasible), criteria: [] }), report({ ...candidate(feasible), criteria: undefined })])(
  "does not imply pass for missing or empty reports (%j)", async value => {
    const { dialog } = await open(value);
    expect(tag().textContent).toBe("Not assessed");
    expect(tag().getAttribute("design")).not.toBe("Positive");
    expect(dialog.textContent).not.toContain("All assessed PS4 feasibility checks pass.");
    expect(api.assessDrawing).not.toHaveBeenCalled();
  },
);

it("updates the heading and summary together when changing configurations without IO", async () => {
  const { dialog } = await open(report(candidate(feasible, "Synthetic A"), candidate(notFeasible, "Synthetic B"), candidate("Inconclusive", "Synthetic C")));
  const selector = dialog.querySelector('select[aria-label="Drawing Configuration"]');
  expect(tag().textContent).toBe("Feasible");
  for (const [index, text, label] of [[1, "Not feasible", "Synthetic B"], [2, "Inconclusive", "Synthetic C"], [0, "Feasible", "Synthetic A"]]) {
    selector.value = String(index); selector.dispatchEvent(new Event("change"));
    expect(tag().textContent).toBe(text); expect(meaning()).toContain(label);
    expect(dialog.querySelector(".drawing-feasibility-summary-table").getAttribute("aria-label")).toContain(label);
    expect(dialog.querySelectorAll(".drawing-feasibility-overall-tag")).toHaveLength(1);
  }
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});

it("never uses unknown overall results to announce all checks pass", () => {
  expect(assessmentConclusion(undefined, [])).not.toContain("checks pass");
  expect(assessmentConclusion("<img src=x>", [])).toContain("unknown");
});

function chooseSyntheticFile(dialog) {
  const input = dialog.querySelector(".drawing-preview-input");
  const file = new File(["synthetic fixture"], "synthetic.png", { type: "image/png" });
  Object.defineProperty(input, "files", { configurable: true, value: [file] });
  input.dispatchEvent(new Event("change"));
  const submit = [...dialog.querySelectorAll("button")].find(node => node.textContent === "Analyze Drawing");
  return { input, file, submit };
}

it("updates Not assessed → Analyzing → selected report and clears stale results on retry (mock only)", async () => {
  const { dialog, onError } = await open();
  const { input, file, submit } = chooseSyntheticFile(dialog);
  let resolve;
  api.assessDrawing.mockImplementationOnce(() => new Promise(done => { resolve = done; }));
  expect(tag().textContent).toBe("Not assessed");
  submit.click();
  expect(tag().textContent).toBe("Analyzing…"); expect(input.disabled).toBe(true); expect(submit.disabled).toBe(true);
  resolve(report(candidate(feasible)));
  await vi.waitFor(() => expect(tag().textContent).toBe("Feasible"));
  expect(api.assessDrawing).toHaveBeenCalledWith(file, api.getObject.mock.calls[0][2]);
  expect(input.disabled).toBe(false); expect(submit.disabled).toBe(false);
  api.assessDrawing.mockRejectedValueOnce(new Error("Synthetic assessment unavailable"));
  submit.click();
  expect(tag().textContent).toBe("Analyzing…");
  expect(dialog.querySelector(".drawing-feasibility-summary-table")).toBeNull();
  await vi.waitFor(() => expect(tag().textContent).toBe("Assessment unavailable"));
  expect(dialog.querySelector('.drawing-feasibility [role="alert"]').textContent).toBe("Synthetic assessment unavailable");
  expect(onError).toHaveBeenCalledTimes(1); expect(input.disabled).toBe(false);
  expect(api.getObject).toHaveBeenCalledTimes(1);
});

it("keeps missing-file validation Not assessed without running analysis", async () => {
  const { dialog } = await open();
  [...dialog.querySelectorAll("button")].find(node => node.textContent === "Analyze Drawing").click();
  expect(tag().textContent).toBe("Not assessed");
  expect(dialog.querySelector('.drawing-feasibility [role="alert"]').textContent).toContain("Select a drawing file");
  expect(api.assessDrawing).not.toHaveBeenCalled();
});

it("clears the transient report when the shared drawing selection changes", async () => {
  const { dialog } = await open();
  const { input, submit } = chooseSyntheticFile(dialog);
  api.assessDrawing.mockResolvedValue(report(candidate(feasible)));
  submit.click();
  await vi.waitFor(() => expect(tag().textContent).toBe("Feasible"));
  Object.defineProperty(input, "files", { configurable: true, value: [new File(["replacement"], "replacement.png", { type: "image/png" })] });
  input.dispatchEvent(new Event("change"));
  expect(dialog.querySelectorAll('input[type="file"]')).toHaveLength(1);
  expect(tag().textContent).toBe("Not assessed");
  expect(dialog.querySelector(".drawing-feasibility-summary-table")).toBeNull();
  expect(api.assessDrawing).toHaveBeenCalledTimes(1);
});

it("rejects an invalid file for both preview and analysis", async () => {
  const { dialog } = await open();
  const { input, submit } = chooseSyntheticFile(dialog);
  Object.defineProperty(input, "files", { configurable: true, value: [new File(["invalid"], "drawing.svg", { type: "image/svg+xml" })] });
  input.dispatchEvent(new Event("change"));
  submit.click();
  expect(dialog.querySelector(".drawing-preview-open").disabled).toBe(true);
  expect(tag().textContent).toBe("Not assessed");
  expect(api.assessDrawing).not.toHaveBeenCalled();
});

it("renders a mocked empty upload response as Not assessed rather than a pass", async () => {
  const { dialog } = await open(); const { submit } = chooseSyntheticFile(dialog);
  api.assessDrawing.mockResolvedValue({ overall_result: feasible, candidate_reports: [] });
  submit.click();
  await vi.waitFor(() => expect(tag().textContent).toBe("Not assessed"));
  expect(dialog.textContent).toContain("No drawing configuration report is available");
  expect(dialog.textContent).not.toContain("checks pass");
});

it("does not insert untrusted result or configuration labels as HTML", async () => {
  const { dialog } = await open(report(candidate("<img src=x>", "<script>synthetic</script>")));
  expect(tag().textContent).toBe("Unknown");
  expect(meaning()).toContain("<script>synthetic</script>");
  expect(dialog.querySelector("script, img")).toBeNull();
});
