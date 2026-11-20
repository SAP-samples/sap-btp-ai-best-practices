import { beforeEach, expect, it, vi } from "vitest";
import { renderBomChanges } from "../src/components/bom-changes-table.js";

let host;
beforeEach(() => {
  document.body.replaceChildren();
  host = document.createElement("section");
  document.body.append(host);
});
const line = (number, assembly = "A1") => ({ assembly, incoming: { number }, plants: [] });
const rows = () => [...host.querySelectorAll(".bom-change-row")];
const status = row => row.querySelector(".bom-blocked-status .semantic-pill").textContent;

it("distinguishes purchasing blockers from priced, removed, unchanged and unassessed materials", () => {
  const inspect = vi.fn();
  renderBomChanges(host, { lines: [
    line("BLOCKED"), line("PRICED"),
    { assembly: "A1", outgoing: { number: "OLD" } },
    { ...line("SAME"), outgoing: { number: "SAME" } },
    line("UNKNOWN"),
  ] }, inspect, { reviewState: { acceptanceStatus: "ready", acceptancePlan: {
    available: true, blocked: true, missing_price: ["BLOCKED"], parts: [
      { number: "BLOCKED", role: "NEW", price: "MISSING" },
      { number: "PRICED", role: "NEW", price: "HAS_PRICE" },
      { number: "OLD", role: "OLD", price: "NOT_NEEDED" },
    ],
  } } });
  expect(rows().map(status)).toEqual(["Blocked", "Not blocked", "Not required", "Not required", "Not assessed"]);
  expect(rows()[0].querySelector(".bom-blocked-status").textContent).toContain("Missing purchasing entry");
  expect(rows().every(row => !row.querySelector(".bom-blocked-status button, .bom-blocked-status a"))).toBe(true);
});

it("shows the exact blocker consistently across assemblies and opens its combined source context", () => {
  const inspect = vi.fn();
  const bom = { lines: [
    { ...line("PART", "A1"), assembly_description: "Supply", plants: [{ plant_code: "PL10", plant_name: "Plant A" }] },
    { ...line("PART", "A2"), assembly_description: "Steering", plants: [{ plant_code: "PK01", plant_name: "Plant B" }] },
  ] };
  renderBomChanges(host, bom, inspect, { reviewState: { acceptancePlan: {
    missing_price: ["PART", "PART"], parts: [{ number: "PART", price: "MISSING", master_data: "Create and release" }],
  } } });
  expect(rows()).toHaveLength(2);
  expect(rows().map(status)).toEqual(["Blocked", "Blocked"]);
  expect(host.textContent).not.toContain("Materials without BOM placement");
  rows()[1].querySelector(".bom-part-link").click();
  expect(inspect).toHaveBeenCalledWith("PART", true, { propagateError: true, acceptanceIssue: expect.objectContaining({
    number: "PART", plants: "Plant A (PL10), Plant B (PK01)", assemblies: ["Supply", "Steering"],
  }) });
  rows()[0].querySelector(".bom-part-link").click();
  expect(inspect).toHaveBeenLastCalledWith("PART", true, expect.objectContaining({ acceptanceIssue: expect.objectContaining({ number: "PART" }) }));
});

it("keeps unplaced blockers visible with no BOM lines and uses only the server's blocker identities", () => {
  renderBomChanges(host, { lines: [] }, vi.fn(), { reviewState: { blocked: true, blockReason: "Missing entries", acceptancePlan: {
    missing_price: ["UNPLACED", "UNPLACED"], parts: [{ number: "PRICED", price: "HAS_PRICE" }],
  } }, snapshot: { questions: [{ subjects: [{ number: "UNRELATED" }] }] } });
  expect(rows()).toHaveLength(1);
  expect(rows()[0].textContent).toContain("UNPLACED");
  expect(rows()[0].textContent).toContain("BOM placement not supplied");
  expect(rows()[0].textContent).toContain("Plant not determined");
  expect(status(rows()[0])).toBe("Blocked");
  expect(host.textContent).toContain("Materials without BOM placement");
  expect(host.textContent).not.toContain("PRICED");
  expect(host.textContent).not.toContain("UNRELATED");
});

it("uses server exemptions and keeps a material's incoming price evidence across outgoing occurrences", () => {
  renderBomChanges(host, { lines: [
    line("EXEMPT"), line("BOTH"), { assembly: "A2", outgoing: { number: "BOTH" } },
  ] }, vi.fn(), { reviewState: { acceptanceStatus: "ready", acceptancePlan: { parts: [
    { number: "EXEMPT", role: "OLD", price: "NOT_NEEDED" },
    { number: "BOTH", role: "NEW", price: "HAS_PRICE" },
    { number: "BOTH", role: "OLD", price: "NOT_NEEDED" },
  ] } } });
  expect(rows().map(status)).toEqual(["Not required", "Not blocked", "Not blocked"]);
});

it("updates loading status when the plan arrives and retains table scroll position", () => {
  const bom = { lines: [line("PART")] };
  renderBomChanges(host, bom, vi.fn(), { reviewState: { acceptanceStatus: "loading" } });
  expect(status(rows()[0])).toBe("Checking…");
  const scroll = host.querySelector(".table-scroll");
  scroll.scrollTop = 80;
  scroll.scrollLeft = 240;
  renderBomChanges(host, bom, vi.fn(), { reviewState: { acceptanceStatus: "ready", acceptancePlan: {
    available: true, parts: [{ number: "PART", price: "HAS_PRICE" }],
  } } });
  expect(status(rows()[0])).toBe("Not blocked");
  expect(host.querySelector(".table-scroll").scrollTop).toBe(80);
  expect(host.querySelector(".table-scroll").scrollLeft).toBe(240);
  renderBomChanges(host, bom, vi.fn(), { reviewState: { acceptanceStatus: "unavailable" } });
  expect(status(rows()[0])).toBe("Not assessed");
  expect(host.querySelector(".bom-acceptance-summary").textContent).toBe("Acceptance status is unavailable.");
});
