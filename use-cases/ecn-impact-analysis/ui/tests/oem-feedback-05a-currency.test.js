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

function source(overrides = {}) {
  return { status: "SOURCE_PRICE_AVAILABLE", currency: "EUR", net_price: "4.70", price_per: "1", order_price_unit: "EA", ...overrides };
}

async function openPrice(price, permission = "External procurement") {
  const obj = {
    number: "SIM-CURRENCY", description: "Synthetic currency material", types: [],
    material_brief: { sourcing: [{ plant_code: "SIM", procurement_type_labels: [permission], source_purchase_price: price }] },
  };
  const original = JSON.stringify(obj);
  Object.freeze(obj.material_brief.sourcing[0].source_purchase_price);
  api.getObject.mockResolvedValue(obj);
  const controller = new AbortController();
  const onError = vi.fn();
  await materialDialog("synthetic-ecn", controller.signal, onError)(obj.number);
  const dialog = document.querySelector("dialog");
  const section = [...dialog.querySelectorAll(".material-story-block")].find(node => node.querySelector("h3")?.textContent === "Procurement Permission");
  const cell = section.querySelector("td[data-label='Source Purchase Price']");
  expect(onError).not.toHaveBeenCalled();
  expect(api.getObject).toHaveBeenCalledTimes(1);
  expect(api.assessDrawing).not.toHaveBeenCalled();
  expect(JSON.stringify(obj)).toBe(original);
  return { dialog, section, cell, controller };
}

it.each([
  ["screenshot EUR basis", source(), "€4.70 / 1 EA"],
  ["EUR without basis", source({ price_per: null, order_price_unit: undefined }), "€4.70"],
  ["original decimal precision and price quantity", source({ net_price: "004.7000", price_per: "100.00", order_price_unit: "PC" }), "€004.7000 / 100.00 PC"],
  ["USD unchanged", source({ currency: "USD", net_price: "1,000.50" }), "USD 1,000.50 / 1 EA"],
  ["other currency code unchanged", source({ currency: "GBP" }), "GBP 4.70 / 1 EA"],
  ["numeric zero", source({ net_price: 0 }), "€0 / 1 EA"],
  ["string zero precision", source({ net_price: "0.00" }), "€0.00 / 1 EA"],
  ["non-EUR numeric zero", source({ currency: "USD", net_price: 0 }), "USD 0 / 1 EA"],
  ["zero price basis retained", source({ price_per: 0 }), "€4.70 / 0 EA"],
  ["missing amount", source({ net_price: undefined }), "EUR / 1 EA"],
  ["null amount", source({ net_price: null }), "EUR / 1 EA"],
  ["empty amount", source({ net_price: "" }), "EUR / 1 EA"],
  ["missing currency", source({ currency: undefined }), "4.70 / 1 EA"],
  ["null currency", source({ currency: null }), "4.70 / 1 EA"],
  ["no amount or currency", source({ currency: null, net_price: null, price_per: null, order_price_unit: null }), "Source price supplied"],
  ["no amount or currency with basis", source({ currency: null, net_price: null }), "Source price supplied / 1 EA"],
  ["missing source", undefined, "Not Supplied"],
  ["null source", null, "Not Supplied"],
  ["missing status", {}, "Not Supplied"],
  ["not supplied status even with an amount", source({ status: "NOT_SUPPLIED" }), "Not Supplied"],
  ["unknown status", source({ status: "UNKNOWN" }), "Not Supplied"],
  ["zero review overrides amount", source({ status: "ZERO_PRICE_REVIEW", net_price: 0 }), "Zero Source Price — Review"],
  ["multiple review overrides amount", source({ status: "MULTIPLE_SOURCE_PRICES_REVIEW" }), "Multiple Source Prices — Review"],
])("renders %s without conversion or invented values", async (_, price, expected) => {
  const { cell, controller } = await openPrice(price);
  expect(cell.textContent).toBe(expected);
  expect(cell.dataset.label).toBe("Source Purchase Price");
  expect(cell.querySelector("button, a, input")).toBeNull();
  controller.abort();
});

it.each(["In-house production", "Permission not supplied"])("keeps Not Allowed for %s even with a EUR price", async permission => {
  const { cell, dialog, controller } = await openPrice(source(), permission);
  expect(cell.textContent).toBe("Not Allowed");
  expect(dialog.textContent).not.toContain("€4.70");
  expect(dialog.querySelector(".make-buy-radar")).toBeNull();
  controller.abort();
});

it("keeps source strings as literal text, table headers and dialog focus behavior", async () => {
  const opener = document.createElement("button");
  document.body.append(opener); opener.focus();
  const { dialog, section, cell, controller } = await openPrice(source({ net_price: "<img src=x onerror=alert(1)>" }));
  expect(cell.textContent).toBe("€<img src=x onerror=alert(1)> / 1 EA");
  expect(cell.querySelector("img")).toBeNull();
  expect([...section.querySelectorAll("th")].map(node => node.textContent)).toEqual(["Plant", "Permitted", "Lead Time", "Source Purchase Price"]);
  expect(dialog.getAttribute("aria-labelledby")).toBe("material-dialog-title");
  dialog.querySelector("footer button").click();
  expect(document.activeElement).toBe(opener);
  controller.abort();
});

it("does not display a success price before the mocked object read completes", async () => {
  let resolve;
  api.getObject.mockImplementation(() => new Promise(done => { resolve = done; }));
  const controller = new AbortController();
  const pending = materialDialog("synthetic-ecn", controller.signal, vi.fn())("SIM-CURRENCY");
  expect(document.querySelector("dialog").open).toBe(true);
  expect(document.querySelector("dialog").textContent).not.toContain("€");
  resolve({ number: "SIM-CURRENCY", types: [], material_brief: { sourcing: [{ procurement_type_labels: ["External procurement"], source_purchase_price: source() }] } });
  await pending;
  expect(document.querySelector("dialog").textContent).toContain("€4.70 / 1 EA");
  expect(api.getObject).toHaveBeenCalledTimes(1);
  controller.abort();
});

it("preserves the existing read-error path without inventing a price", async () => {
  const error = new Error("Synthetic object read unavailable");
  api.getObject.mockRejectedValue(error);
  const controller = new AbortController();
  const onError = vi.fn();
  await materialDialog("synthetic-ecn", controller.signal, onError)("SIM-CURRENCY");
  expect(onError).toHaveBeenCalledWith(error);
  expect(document.querySelector("dialog").textContent).not.toContain("€");
  expect(document.querySelector("dialog").open).toBe(true);
  expect(api.assessDrawing).not.toHaveBeenCalled();
  controller.abort();
});
