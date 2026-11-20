import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import html from "../src/pages/ecn-detail/ecn-detail.html?raw";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const css = readFileSync(resolve("src/pages/ecn-detail/ecn-detail.css"), "utf8");
import { createEcnDetailPage } from "../src/pages/ecn-detail/ecn-detail.js";
import * as workspaceApi from "../src/services/workspace-api.js";
import { detailEnvelope } from "./fixtures/api-responses.js";

vi.mock("../src/services/workspace-api.js", () => ({
  getSnapshot: vi.fn(), getAnalysisStages: vi.fn(), generateSummaries: vi.fn(),
  listReviews: vi.fn(), saveReview: vi.fn(), getAcceptancePlan: vi.fn(), versions: vi.fn(),
  getObject: vi.fn(),
}));

const rowId = "11111111-1111-4111-8111-111111111111";
const snapshot = Object.freeze({
  evidence_hash: "synthetic-evidence", version: "synthetic-v1", assessment_date: "2026-09-14",
  counts: { technical_objects: 0, attention_items: 1 },
  questions: [{ code: "MBOM_NOT_UPDATED" }], parts: [], programs: [],
  bom_changes: { lines: [], materials: [] }, plant_impact: { plants: [] },
});
let controller;
let cleanup;
let originalShowModal;
let originalClose;
const select = (selector) => document.querySelector(selector);

beforeEach(() => {
  vi.clearAllMocks();
  controller = new AbortController();
  workspaceApi.listReviews.mockResolvedValue([]);
  workspaceApi.getAcceptancePlan.mockResolvedValue({ available: true, blocked: false });
  workspaceApi.versions.mockResolvedValue([]);
  workspaceApi.saveReview.mockResolvedValue([]);
  vi.spyOn(window, "confirm").mockReturnValue(true);
  originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "showModal");
  originalClose = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "close");
  Object.defineProperty(HTMLDialogElement.prototype, "showModal", { configurable: true, writable: true, value() { this.open = true; } });
  Object.defineProperty(HTMLDialogElement.prototype, "close", { configurable: true, writable: true, value() { this.open = false; this.dispatchEvent(new Event("close")); } });
});
afterEach(() => {
  controller.abort();
  cleanup?.();
  cleanup = null;
  document.body.replaceChildren();
  vi.restoreAllMocks();
  if (originalShowModal) Object.defineProperty(HTMLDialogElement.prototype, "showModal", originalShowModal);
  else delete HTMLDialogElement.prototype.showModal;
  if (originalClose) Object.defineProperty(HTMLDialogElement.prototype, "close", originalClose);
  else delete HTMLDialogElement.prototype.close;
});

async function mount() {
  document.body.innerHTML = `<main>${html}</main>`;
  // Deterministic native-dialog seam; real modal focus is a browser review gate.
  const dialog = select("#send-back-dialog");
  dialog.showModal = () => { dialog.open = true; };
  dialog.close = () => { dialog.open = false; };
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope()),
    getSnapshot: vi.fn().mockResolvedValue(snapshot), retryAnalysis: vi.fn(),
  };
  cleanup = await createEcnDetailPage({ api, startPolling: () => vi.fn() })({
    container: select("main"), params: { ecn_row_id: rowId }, query: new URLSearchParams(),
    signal: controller.signal, navigate: vi.fn(),
  });
  await vi.waitFor(() => expect(select("#accept-ecn-footer").accessibleDescription).toContain("Acceptance rules"));
  return api;
}

function expectUnblocked() {
  expect(select("#accept-ecn-footer").disabled).toBe(false);
  expect(select("#accept-ecn-explanation")).toBeNull();
  expect(select("#accept-ecn-reason").closest("#acceptance-rules")).not.toBeNull();
  expect(select("#accept-ecn-footer").accessibleDescription).not.toContain("ready");
  expect(select("#decision-readiness")).toBeNull();
}

describe("feedback 08 — acceptance clarification without rule changes", () => {
  it("offers a native rules dialog from the footer, with a focusable scroll region", async () => {
    const api = await mount();
    const dialog = select("#acceptance-rules");
    const trigger = select("#acceptance-rules-open");
    expect(dialog.tagName).toBe("UI5-DIALOG");
    expect(dialog.open).toBe(false);
    expect(dialog.getAttribute("header-text")).toBe("Acceptance rules");
    expect(dialog.closest("#decision-bar")).toBeNull();
    expect(trigger.closest("#decision-bar")).not.toBeNull();
    expect(select("#accept-ecn-explanation")).toBeNull();
    expect(select("#accept-ecn-reason").closest("#acceptance-rules")).toBe(dialog);
    const region = dialog.querySelector('[role="region"]');
    expect(region.getAttribute("aria-label")).toBe("Acceptance rules explanation");
    expect(region.tabIndex).toBe(0);
    expect(css).toContain(".acceptance-rules-content");
    expect(css).toContain("max-height: min(28rem, 60vh)");
    expect(css).toContain("overflow-y: auto");
    expect(css).not.toMatch(/\.acceptance-rules-content\s*\{[^}]*position:\s*fixed/);
    expect(trigger.accessibilityAttributes).toEqual({ controls: "acceptance-rules", expanded: "false", hasPopup: "dialog" });
    trigger.click();
    expect(dialog.open).toBe(true);
    expect(trigger.accessibilityAttributes.expanded).toBe("true");
    select("#acceptance-rules-close").click();
    expect(dialog.open).toBe(false);
    dialog.dispatchEvent(new Event("close"));
    expect(trigger.accessibilityAttributes.expanded).toBe("false");
    expect(api.retryAnalysis).not.toHaveBeenCalled();
    expect(workspaceApi.saveReview).not.toHaveBeenCalled();
    expect(workspaceApi.getAcceptancePlan).toHaveBeenCalledTimes(1);
  });

  it("closes open rules on abort and cleans handlers without reopening or retaining ARIA state", async () => {
    await mount();
    const dialog = select("#acceptance-rules"), trigger = select("#acceptance-rules-open");
    trigger.click();
    expect(dialog.open).toBe(true);
    controller.abort();
    expect(dialog.open).toBe(false);
    expect(trigger.accessibilityAttributes).toEqual({});
    cleanup();
    trigger.click();
    dialog.dispatchEvent(new Event("close"));
    expect(dialog.open).toBe(false);
    expect(trigger.accessibilityAttributes).toEqual({});
    expect(workspaceApi.saveReview).not.toHaveBeenCalled();
  });

  it("explains source membership, old-part exemption, confirmation, snapshot identity and simulation truthfully", async () => {
    await mount();
    const text = select("#acceptance-rules").textContent;
    expect(text).toContain("Incoming (new) parts must be listed in purchase-info-record data");
    expect(text).toContain("Missing incoming entries block acceptance");
    expect(text).toContain("current surrogate for price evidence");
    expect(text).toContain("does not guarantee a numeric or current price");
    expect(text).toContain("Outgoing (old) material does not require price evidence");
    expect(text).toContain("reviewer name");
    expect(text).toContain("confirm the decision");
    expect(text).toContain("snapshot hash and version still match");
    expect(text).toContain("checks acceptance preconditions again when saving");
    expect(text).toContain("reload it before submitting");
    expect(text).toContain("SAP update is simulated");
    expect(text).toContain("not an actual SAP update");
    expect(text).toContain("loading or unavailable");
    expect(text).not.toMatch(/geometry|feasibility|source Released/);
  });

  it("preserves a literal server blocker and BOM evidence path, including public button description", async () => {
    const reason = "Can't accept: 2 new parts have no price. <img src=x onerror=alert(1)>";
    workspaceApi.getAcceptancePlan.mockResolvedValue(Object.freeze({ available: true, blocked: true, block_reason: reason }));
    await mount();
    await vi.waitFor(() => expect(select("#accept-ecn-reason").textContent).toBe(reason));
    expect(select("#accept-ecn-reason img")).toBeNull();
    expect(select("#accept-ecn-footer").disabled).toBe(true);
    expect(select("#send-back-footer").disabled).toBe(false);
    expect(select("#accept-ecn-explanation")).toBeNull();
    expect(select("#accept-ecn-footer").accessibleDescription).toContain(reason);
    expect(select("#bom-changes .bom-acceptance-summary").textContent).toBe(reason);
    expect(select("#bom-changes .bom-acceptance-summary img")).toBeNull();
    select('[data-section-target="bom"]').click();
    expect(select('[data-section-target="bom"]').getAttribute("aria-current")).toBe("true");
  });

  it.each(["ACCEPTED", "CHANGES_REQUESTED"])("keeps a recorded %s decision disabled with its Audit Log path", async (outcome) => {
    workspaceApi.listReviews.mockResolvedValue([{ outcome, reviewer: "Synthetic reviewer", comments: "", created_at: "2026-09-14T09:00:00Z" }]);
    await mount();
    await vi.waitFor(() => expect(select("#accept-ecn-reason").textContent).toBe("An engineering decision is already recorded."));
    expect(select("#accept-ecn-footer").disabled).toBe(true);
    expect(select("#send-back-footer").disabled).toBe(true);
    expect(select("#reviewer-name-footer").hidden).toBe(false);
    expect(select("#accept-ecn-explanation")).toBeNull();
    select('[data-section-target="audit"]').click();
    expect(select('[data-section-target="audit"]').getAttribute("aria-current")).toBe("true");
  });

  it("includes unplaced blocked materials in the table and opens their required action", async () => {
    workspaceApi.getAcceptancePlan.mockResolvedValue({ available: true, blocked: true,
      block_reason: "Can't accept: 2 new parts have no price.", missing_price: ["PART-A", "PART-B"],
      parts: [
        { number: "PART-A", description: "New pipe", role: "NEW", price: "MISSING", master_data: "Create and release in PL10" },
        { number: "PART-B", description: "Bushing", role: "NEW", price: "MISSING", master_data: "Release (plant not known)" },
        { number: "PART-PRICED", role: "NEW", price: "HAS_PRICE" },
        { number: "PART-OLD", role: "OLD", price: "NOT_NEEDED" },
      ],
    });
    workspaceApi.getObject.mockResolvedValue({ number: "PART-A", description: "New pipe", types: [], material_brief: {} });
    await mount();
    expect(select("#review-blocked-materials")).toBeNull();
    const table = select("#bom-changes");
    expect(table.querySelectorAll(".bom-change-row")).toHaveLength(2);
    expect(table.textContent).toContain("New pipe");
    expect(table.textContent).not.toContain("PART-PRICED");
    expect(table.textContent).toContain("Materials without BOM placement");
    const open = table.querySelector(".bom-part-link");
    open.focus();
    open.click();
    await vi.waitFor(() => expect(select(".material-acceptance-issue")).not.toBeNull());
    const task = select(".material-acceptance-issue");
    expect(task.textContent).toContain("Add a purchase-info-record entry for PART-A");
    expect(task.textContent).toContain("Create and release in PL10");
    expect(task.textContent).toContain("Plant not determined");
    expect(workspaceApi.getObject).toHaveBeenCalledWith(rowId, { number: "PART-A", technical: true, page: 1, page_size: 25 }, expect.any(AbortSignal));
    select(".object-dialog-footer button").click();
    expect(document.activeElement).toBe(open);
    expect(workspaceApi.saveReview).not.toHaveBeenCalled();
  });

  it("shows a material load failure beside its material number and allows a retry", async () => {
    workspaceApi.getAcceptancePlan.mockResolvedValue({ available: true, blocked: true,
      block_reason: "Can't accept: 1 new part has no price.", missing_price: ["PART-A"] });
    workspaceApi.getObject.mockRejectedValueOnce(new Error("Source details unavailable"))
      .mockResolvedValueOnce({ number: "PART-A", description: "New pipe", types: [], material_brief: {} });
    await mount();
    const open = select(".bom-part-link");
    open.click();
    await vi.waitFor(() => expect(select(".bom-change-part .acceptance-material-error").textContent).toContain("Could not open PART-A"));
    expect(open.disabled).toBe(false);
    open.click();
    await vi.waitFor(() => expect(select(".object-dialog").open).toBe(true));
  });

  it("does not turn source follow-ups into acceptance blockers", async () => {
    await mount();
    expectUnblocked();
  });

  it("preserves missing-plan behavior without calling it ready", async () => {
    workspaceApi.getAcceptancePlan.mockResolvedValue({ available: false, blocked: false });
    await mount();
    expectUnblocked();
    expect(select("#accept-ecn-reason").textContent).toBe("Acceptance status is unavailable.");
  });

  it("preserves pending-plan and failed-plan behavior while keeping help available", async () => {
    let failPlan;
    workspaceApi.getAcceptancePlan.mockReturnValue(new Promise((_, reject) => { failPlan = reject; }));
    await mount();
    expectUnblocked();
    expect(select("#accept-ecn-reason").textContent).toBe("Checking acceptance status…");
    failPlan(new Error("Synthetic plan unavailable"));
    await vi.waitFor(() => expect(select("#accept-ecn-reason").textContent).toBe("Acceptance status is unavailable."));
    expectUnblocked();
    expect(select("#acceptance-rules").hidden).toBe(false);
  });

  it("keeps the reviewer field available before actions and after an exact-snapshot accepted save", async () => {
    await mount();
    const button = select("#accept-ecn-footer");
    const name = select("#reviewer-name-footer");
    expect(name.hidden).toBe(false);
    button.click();
    expect(name.hidden).toBe(false);
    expect(workspaceApi.saveReview).not.toHaveBeenCalled();
    name.value = "  Synthetic reviewer  ";
    window.confirm.mockReturnValueOnce(false);
    button.click();
    expect(window.confirm).toHaveBeenCalledWith("Accept this ECN? The SAP update is simulated.");
    expect(workspaceApi.saveReview).not.toHaveBeenCalled();
    workspaceApi.saveReview.mockResolvedValueOnce([{ outcome: "ACCEPTED", reviewer: "Synthetic reviewer", created_at: "2026-09-14T09:00:00Z" }]);
    button.click();
    await vi.waitFor(() => expect(workspaceApi.saveReview).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(button.disabled).toBe(true));
    expect(name.hidden).toBe(false);
    expect(select("#accept-ecn-reason").textContent).toBe("An engineering decision is already recorded.");
    expect(select("#accept-ecn-explanation")).toBeNull();
    expect(workspaceApi.saveReview).toHaveBeenCalledWith(rowId, {
      request_id: expect.any(String), reviewer: "Synthetic reviewer", outcome: "ACCEPTED", comments: "",
      evidence_hash: snapshot.evidence_hash, snapshot_version: snapshot.version,
    }, controller.signal);
  });

  it("retains existing failed-save behavior without changing the plan or decision state", async () => {
    workspaceApi.saveReview.mockRejectedValueOnce(new Error("Synthetic snapshot mismatch"));
    await mount();
    select("#reviewer-name-footer").value = "Synthetic reviewer";
    select("#accept-ecn-footer").click();
    await vi.waitFor(() => expect(workspaceApi.saveReview).toHaveBeenCalledTimes(1));
    expectUnblocked();
    expect(select("#acceptance-rules").textContent).toContain("reload it before submitting");
    // Existing Audit Log save path swallows errors; no new error handling in task 08.
  });

  it("keeps send-back comments/name validation and changes-requested save intact", async () => {
    await mount();
    select("#send-back-footer").click();
    select("#send-back-confirm").click();
    expect(select("#send-back-error").textContent).toBe("Say what must change.");
    select("#send-back-comments").value = "  Confirm the source evidence.  ";
    select("#send-back-confirm").click();
    expect(select("#send-back-error").textContent).toBe("Enter your name in the decision footer.");
    expect(workspaceApi.saveReview).not.toHaveBeenCalled();
    select("#reviewer-name-footer").value = "Synthetic reviewer";
    select("#send-back-confirm").click();
    await vi.waitFor(() => expect(workspaceApi.saveReview).toHaveBeenCalledTimes(1));
    expect(workspaceApi.saveReview).toHaveBeenCalledWith(rowId, expect.objectContaining({
      outcome: "CHANGES_REQUESTED", comments: "Confirm the source evidence.",
      reviewer: "Synthetic reviewer", evidence_hash: snapshot.evidence_hash, snapshot_version: snapshot.version,
    }), controller.signal);
    expect(window.confirm).not.toHaveBeenCalled();
  });
});
