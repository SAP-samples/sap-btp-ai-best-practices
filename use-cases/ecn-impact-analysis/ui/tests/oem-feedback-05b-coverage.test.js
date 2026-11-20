import { beforeEach, expect, it, vi } from "vitest";
import { materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";

vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));

beforeEach(() => {
  document.body.replaceChildren();
  vi.clearAllMocks();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});

function occurrence(source, overrides = {}) {
  return {
    source, path: [{ number: "SIM-ASSEMBLY", description: "Synthetic assembly" }, { number: "SIM-PART", description: "Synthetic part" }],
    item_or_position: "0010", quantity: "2", unit: "PC",
    valid_from_date: "2026-03-13", valid_to_date: "9999-12-31", change_number: "SIM-ECN", ...overrides,
  };
}

function coverage(overrides = {}) {
  return {
    coverage: "OBSERVED_IN_BOTH", engineering_occurrences: 1, manufacturing_occurrences: 1,
    plants: [], effective_interval_count: 1,
    occurrences: [occurrence("Enriched EBOM"), occurrence("130% MBOM")], ...overrides,
  };
}

function freeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

async function openCoverage(bomCoverage) {
  const obj = freeze({
    number: "SIM-PART", description: "Synthetic part", types: [],
    material_brief: { bom_coverage: bomCoverage, sourcing: [
      { plant_code: "SIM-BUY", procurement_type_labels: ["External procurement"], source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", currency: "EUR", net_price: "4.70", price_per: "1", order_price_unit: "EA" } },
      { plant_code: "SIM-MAKE", procurement_type_labels: ["In-house production"], source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", currency: "EUR", net_price: "9.00" } },
    ] },
  });
  const original = JSON.stringify(obj);
  api.getObject.mockResolvedValue(obj);
  const controller = new AbortController();
  const onError = vi.fn();
  await materialDialog("synthetic-ecn", controller.signal, onError)(obj.number);
  const dialog = document.querySelector("dialog");
  expect(onError).not.toHaveBeenCalled();
  expect(api.getObject).toHaveBeenCalledTimes(1);
  expect(api.assessDrawing).not.toHaveBeenCalled();
  expect(JSON.stringify(obj)).toBe(original);
  // Prior currency and permission changes remain independent of BOM evidence.
  expect([...dialog.querySelectorAll("td[data-label='Source Purchase Price']")].map(cell => cell.textContent)).toEqual(["€4.70 / 1 EA", "Not Allowed"]);
  return { dialog, controller, table: dialog.querySelector(".bom-coverage-table") };
}

it.each([
  ["OBSERVED_IN_BOTH", "Found in EBOM and MBOM"],
  ["ENGINEERING_ONLY", "EBOM only"],
  ["MANUFACTURING_ONLY", "MBOM only"],
  ["NOT_OBSERVED", "Not observed in supplied BOM sources"],
  [undefined, "BOM presence not supplied"],
  [null, "BOM presence not supplied"],
  ["", "BOM presence not supplied"],
  ["FUTURE_STATUS", "BOM presence not supplied"],
  ["toString", "BOM presence not supplied"],
])("uses only the supplied %s enum, never occurrence or price inference", async (status, expected) => {
  const { table, controller } = await openCoverage(coverage({ coverage: status }));
  expect(table.querySelector("td[data-label='BOM presence']").textContent).toBe(expected);
  expect([...table.querySelectorAll("th")].map(node => node.textContent)).toEqual(["Component", "Engineering occurrences", "Manufacturing occurrences", "Effective interval", "Plant", "BOM presence"]);
  expect(table.querySelector("td[data-label='Plant']").textContent).toBe("Plant not supplied");
  controller.abort();
});

it("retains source counts, hierarchy, quantities, dates and change references", async () => {
  const rows = [occurrence("Enriched EBOM"), occurrence("Enriched EBOM"), occurrence("130% EBOM"), occurrence("130% MBOM", { quantity: 0, change_number_to: "SIM-END" })];
  const { dialog, table, controller } = await openCoverage(coverage({ occurrences: rows, engineering_occurrences: 3, plants: ["SIM-A", "SIM-B"], effective_interval_count: 2 }));
  expect(table.querySelector("td[data-label='Engineering occurrences']").textContent).toBe("Enriched EBOM: 2; 130% EBOM: 1");
  expect(table.querySelector("td[data-label='Manufacturing occurrences']").textContent).toBe("130% MBOM: 1");
  expect(table.querySelector("td[data-label='Plant']").textContent).toBe("SIM-A, SIM-B");
  expect(table.querySelector("td[data-label='Effective interval']").textContent).toBe("Multiple intervals (2)");
  const details = dialog.querySelector(".observed-occurrences-details");
  expect(details.open).toBe(false);
  expect(details.querySelector("summary").textContent).toBe("Show 4 observed BOM occurrences");
  expect(details.querySelectorAll("tbody tr")).toHaveLength(4);
  const last = [...details.querySelectorAll("tbody tr")].at(-1);
  expect([...last.querySelectorAll(".hierarchy-node")].map(node => node.textContent)).toEqual(["Synthetic assembly (SIM-ASSEMBLY)", "Synthetic part (SIM-PART)"]);
  expect(last.querySelector("td[data-label='Item or position']").textContent).toBe("0010");
  expect(last.querySelector("td[data-label='Quantity']").textContent).toBe("0 PC");
  expect(last.querySelector("td[data-label='Effective interval']").textContent).toBe("13 Mar 2026 – No planned end");
  expect(last.querySelector("td[data-label='Change reference']").textContent).toBe("SIM-ECN · SIM-END");
  controller.abort();
});

it("explains source scope visibly and associates help with the existing focusable region", async () => {
  const opener = document.createElement("button");
  document.body.append(opener); opener.focus();
  const { dialog, table, controller } = await openCoverage(coverage());
  const help = dialog.querySelector(".bom-presence-help");
  expect(help.textContent).toBe("BOM presence shows observations in the supplied engineering (EBOM) and manufacturing (MBOM) sources. It does not confirm complete or approved rollout, or approval for any plant or date.");
  expect(table.getAttribute("aria-describedby")).toBe(help.id);
  expect(table.getAttribute("aria-label")).toBe("Observed BOM coverage table.");
  expect(table.getAttribute("role")).toBe("region");
  expect(table.tabIndex).toBe(0);
  expect(table.closest("section").querySelector("h3").textContent).toBe("Observed BOM Coverage");
  expect(table.querySelector("button, a, input")).toBeNull();
  expect(table.querySelector("td[data-label='Effective interval']").textContent).toBe("13 Mar 2026 – No planned end");
  dialog.querySelector("footer button").click();
  expect(document.activeElement).toBe(opener);
  controller.abort();
});

it.each([[], undefined, null])("shows honest empty occurrence evidence for %s without a zero-row disclosure", async occurrences => {
  const { dialog, table, controller } = await openCoverage(coverage({ coverage: "NOT_OBSERVED", occurrences, plants: undefined, effective_interval_count: 1 }));
  expect(table.querySelector("td[data-label='BOM presence']").textContent).toBe("Not observed in supplied BOM sources");
  expect(table.querySelector("td[data-label='Plant']").textContent).toBe("Plant not supplied");
  expect(table.querySelector("td[data-label='Effective interval']").textContent).toBe("Not supplied");
  expect(table.querySelector("td[data-label='Engineering occurrences']").textContent).toBe("None");
  expect(table.querySelector("td[data-label='Manufacturing occurrences']").textContent).toBe("None");
  expect(dialog.textContent).toContain("No observed BOM occurrences were supplied for this material.");
  expect(dialog.querySelector(".observed-occurrences-details")).toBeNull();
  controller.abort();
});

it("does not invent missing occurrences from a known presence enum", async () => {
  const { table, dialog, controller } = await openCoverage(coverage({ coverage: "ENGINEERING_ONLY", occurrences: [], effective_interval_count: 0 }));
  expect(table.querySelector("td[data-label='BOM presence']").textContent).toBe("EBOM only");
  expect(dialog.textContent).toContain("No observed BOM occurrences were supplied");
  controller.abort();
});

it.each([undefined, null])("keeps an absent coverage object absent without inferring from pricing (%s)", async value => {
  const { dialog, table, controller } = await openCoverage(value);
  expect(table).toBeNull();
  expect(dialog.querySelector(".bom-presence-help")).toBeNull();
  controller.abort();
});

it("renders source strings literally without HTML interpretation", async () => {
  const { table, dialog, controller } = await openCoverage(coverage({ coverage: "<img src=x>", plants: ["<script>synthetic</script>"], occurrences: [occurrence("Enriched EBOM", { path: [{ description: "<img src=x>" }] })] }));
  expect(table.querySelector("td[data-label='BOM presence']").textContent).toBe("BOM presence not supplied");
  expect(table.querySelector("td[data-label='Plant']").textContent).toBe("<script>synthetic</script>");
  expect(dialog.querySelector(".hierarchy-node").textContent).toBe("<img src=x>");
  expect(dialog.querySelector("script, img")).toBeNull();
  controller.abort();
});

it("does not invent presence during the existing pending read or read-error path", async () => {
  let resolve;
  api.getObject.mockImplementation(() => new Promise(done => { resolve = done; }));
  const controller = new AbortController();
  const onError = vi.fn();
  const inspect = materialDialog("synthetic-ecn", controller.signal, onError);
  const pending = inspect("SIM-PART");
  expect(document.querySelector("dialog").open).toBe(true);
  expect(document.querySelector(".bom-coverage-table")).toBeNull();
  resolve({ number: "SIM-PART", material_brief: { bom_coverage: coverage() } });
  await pending;
  expect(document.querySelector("dialog").open).toBe(true);
  controller.abort();

  const error = new Error("Synthetic BOM object read unavailable");
  api.getObject.mockRejectedValue(error);
  const errorController = new AbortController();
  await materialDialog("synthetic-ecn", errorController.signal, onError)("SIM-PART");
  expect(onError).toHaveBeenCalledWith(error);
  expect(document.querySelector("dialog").open).toBe(true);
  expect(document.querySelector(".bom-coverage-table")).toBeNull();
  expect(api.assessDrawing).not.toHaveBeenCalled();
  errorController.abort();
});
