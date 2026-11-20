import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";

vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));
let controller, opener, inspect, onError;
const object = number => ({ number, description: `${number} description`, material_brief: { sourcing: [] } });
function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  document.body.replaceChildren(); vi.resetAllMocks();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
  controller = new AbortController();
  opener = document.createElement("button"); document.body.append(opener); opener.focus();
  onError = vi.fn(); inspect = materialDialog("id", controller.signal, onError);
});
afterEach(() => controller.abort());

it("opens before a slow read, exposes only the requested identity, and keeps Close focused on completion", async () => {
  const read = deferred(); api.getObject.mockReturnValue(read.promise);
  const pending = inspect("PART-A");
  const dialog = document.querySelector("dialog");
  expect(dialog.open).toBe(true);
  expect(dialog.querySelector(".material-identifier").textContent).toBe("Material: PART-A");
  expect(dialog.querySelector('[role="status"]').textContent).toBe("Loading material details…");
  expect(dialog.querySelector('[aria-busy="true"]')).not.toBeNull();
  expect(dialog.querySelector("table, input")).toBeNull();
  const close = dialog.querySelector("footer button"); close.focus();
  read.resolve(object("PART-A")); await pending;
  expect(dialog.querySelector('[aria-busy="true"]')).toBeNull();
  expect(dialog.querySelector("h2").textContent).toBe("PART-A description");
  expect(document.activeElement).toBe(close);
});

it.each(["close", "escape", "teardown"])("cancels on %s and ignores a late successful read", async mode => {
  const read = deferred(); api.getObject.mockReturnValue(read.promise);
  const pending = inspect("PART-A");
  const child = api.getObject.mock.calls[0][2];
  expect(child).not.toBe(controller.signal); expect(child.aborted).toBe(false);
  const dialog = document.querySelector("dialog");
  if (mode === "close") dialog.querySelector("footer button").click();
  if (mode === "escape") { dialog.dispatchEvent(new Event("cancel")); dialog.close(); }
  if (mode === "teardown") controller.abort();
  expect(child.aborted).toBe(true);
  read.resolve(object("PART-A")); await pending;
  if (mode === "teardown") expect(dialog.isConnected).toBe(false);
  else { expect(dialog.open).toBe(false); expect(document.activeElement).toBe(opener); }
  expect(dialog.querySelector(".engineering-drawing-preview")).toBeNull();
  expect(onError).not.toHaveBeenCalled();
});

it("clears old details on switching and ignores superseded reads and errors", async () => {
  api.getObject.mockResolvedValueOnce(object("OLD")); await inspect("OLD");
  const first = deferred(), second = deferred();
  api.getObject.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
  const pendingA = inspect("PART-A", true, { propagateError: true });
  const signalA = api.getObject.mock.calls[1][2];
  expect(document.querySelector("dialog").textContent).not.toContain("OLD description");
  const pendingB = inspect("PART-B");
  expect(signalA.aborted).toBe(true);
  second.resolve(object("PART-B")); await pendingB;
  first.reject(new Error("Old request failed")); await pendingA;
  expect(document.querySelector("h2").textContent).toBe("PART-B description");
  expect(onError).not.toHaveBeenCalled();
});

it("ignores late data after closing and reopening for another material", async () => {
  const first = deferred(), second = deferred();
  api.getObject.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
  const pendingA = inspect("PART-A"); document.querySelector("footer button").click();
  opener.focus(); const pendingB = inspect("PART-B");
  // A queued close from the first opening must not cancel the new opening.
  document.querySelector("dialog").dispatchEvent(new Event("close"));
  first.resolve(object("PART-A")); await pendingA;
  expect(document.querySelector(".material-identifier").textContent).toBe("Material: PART-B");
  second.resolve(object("PART-B")); await pendingB;
  expect(document.querySelector("h2").textContent).toBe("PART-B description");
});

it("offers inline retry with the same material, technical flag, and acceptance task", async () => {
  const failure = new Error("Source details unavailable");
  api.getObject.mockRejectedValueOnce(failure).mockResolvedValueOnce(object("PART-A"));
  const acceptanceIssue = { number: "PART-A", required: "Add purchasing entry for PART-A", plants: "Plant A", assemblies: [], masterData: "Create" };
  await expect(inspect("PART-A", false, { acceptanceIssue, propagateError: true })).rejects.toBe(failure);
  const dialog = document.querySelector("dialog");
  expect(dialog.open).toBe(true); expect(dialog.querySelector("table, input")).toBeNull();
  expect(dialog.querySelector('[role="alert"]').textContent).toContain(failure.message);
  expect(onError).toHaveBeenCalledExactlyOnceWith(failure);
  [...dialog.querySelectorAll("button")].find(node => node.textContent === "Retry").click();
  await vi.waitFor(() => expect(dialog.querySelector(".material-acceptance-issue")).not.toBeNull());
  expect(api.getObject.mock.calls[1][1]).toMatchObject({ number: "PART-A", technical: false });
  expect(dialog.querySelector(".material-acceptance-issue").textContent).toContain(acceptanceIssue.required);
  expect(document.activeElement).toBe(dialog.querySelector(".material-acceptance-issue"));
});

it("aborts drawing analysis along with the material session", async () => {
  api.getObject.mockResolvedValue(object("PART-A")); await inspect("PART-A");
  const read = deferred(); api.assessDrawing.mockReturnValue(read.promise);
  const dialog = document.querySelector("dialog"), input = dialog.querySelector('input[type="file"]');
  Object.defineProperty(input, "files", { value: [new File(["drawing"], "part.png", { type: "image/png" })] });
  input.dispatchEvent(new Event("change"));
  [...dialog.querySelectorAll("button")].find(node => node.textContent === "Analyze Drawing").click();
  const child = api.assessDrawing.mock.calls[0][1];
  dialog.querySelector("footer button").click(); expect(child.aborted).toBe(true);
  read.reject(new DOMException("Cancelled", "AbortError"));
  await Promise.resolve(); expect(onError).not.toHaveBeenCalled();
});
