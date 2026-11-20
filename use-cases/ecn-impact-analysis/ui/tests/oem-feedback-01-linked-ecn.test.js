import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import html from "../src/pages/ecn-detail/ecn-detail.html?raw";
import { createEcnDetailPage } from "../src/pages/ecn-detail/ecn-detail.js";
import { detailEnvelope } from "./fixtures/api-responses.js";

// Isolate this presentation from workspace reads, chat and customer services.
vi.mock("../src/components/snapshot-workspace.js", () => ({
  snapshotWorkspace: vi.fn(() => ({ updateAnalysisDescriptions: vi.fn(), updateAuditDetail: vi.fn() })),
}));
let cleanup;
let originalShowModal;
let originalClose;
beforeEach(() => {
  vi.stubEnv("DEV", true);
  document.body.innerHTML = `<main>${html}</main>`;
  originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "showModal");
  originalClose = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "close");
  Object.defineProperty(HTMLDialogElement.prototype, "showModal", { configurable: true, writable: true,
    value: vi.fn(function () { this.open = true; }) });
  Object.defineProperty(HTMLDialogElement.prototype, "close", { configurable: true, writable: true,
    value: vi.fn(function () { this.open = false; this.dispatchEvent(new Event("close")); }) });
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Unexpected IO in synthetic linked-ECN test"); }));
});
afterEach(() => {
  cleanup?.();
  cleanup = null;
  if (originalShowModal) Object.defineProperty(HTMLDialogElement.prototype, "showModal", originalShowModal);
  else delete HTMLDialogElement.prototype.showModal;
  if (originalClose) Object.defineProperty(HTMLDialogElement.prototype, "close", originalClose);
  else delete HTMLDialogElement.prototype.close;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});
async function mount(query = "") {
  const controller = new AbortController();
  const api = {
    getEcn: vi.fn().mockResolvedValue(detailEnvelope()),
    retryAnalysis: vi.fn(),
    listEcns: vi.fn(),
    generateSummaries: vi.fn(),
  };
  cleanup = await createEcnDetailPage({ api, startPolling: () => vi.fn() })({
    container: document.querySelector("main"),
    params: { ecn_row_id: "11111111-1111-4111-8111-111111111111" },
    query: new URLSearchParams(`query=synthetic&sort=warning_count&page=2&page_size=25&${query}`),
    signal: controller.signal,
  });
  return { api, controller };
}
async function demo() {
  const mounted = await mount("demo-linked-ecn=1");
  await vi.waitFor(() => expect(document.querySelector("[data-linked-ecn-demo]")).not.toBeNull());
  return mounted;
}
it("places a disabled linked ECN button immediately after Release date, preserving source and Back fields", async () => {
  const { api } = await mount();
  const linked = document.querySelector("#linked-ecn-field");
  expect(linked.querySelector("dt").textContent).toBe("Linked ECN");
  expect(linked.querySelector("dd button").textContent).toBe("No linked ECN");
  expect(linked.querySelector("dd button").disabled).toBe(true);
  expect(linked.previousElementSibling.querySelector("dt").textContent).toBe("Release date");
  expect(document.querySelector("#source-record-fields").textContent).toContain("Released");
  expect(document.querySelector("#back-to-list").getAttribute("href")).toContain("query=synthetic");
  expect(document.querySelector("#back-to-list").getAttribute("href")).not.toContain("demo-linked-ecn");
  expect(document.querySelector("#linked-ecn-demo-dialog")).toBeNull();
  expect(api.getEcn).toHaveBeenCalledTimes(1);
});
it.each(["demo-linked-ecn=0", "demo-linked-ecn=true", "demo-linked-ecn="])("requires exact explicit opt-in: %s", async (query) => {
  await mount(query);
  expect(document.querySelector("#linked-ecn-field dd button").disabled).toBe(true);
  expect(document.querySelector("#linked-ecn-demo-dialog")).toBeNull();
});
it("opens an explicitly fictional local dialog with zero additional IO and returns focus on close/Escape", async () => {
  const { api } = await demo();
  const link = document.querySelector("[data-linked-ecn-demo]");
  const dialog = document.querySelector("#linked-ecn-demo-dialog");
  expect(document.querySelector("#linked-ecn-field").textContent).toContain("Simulated demo relationship");
  expect(link.textContent).toMatch(/^ECN-DEMO-/);
  expect(link.getAttribute("aria-haspopup")).toBe("dialog");
  expect(dialog.getAttribute("aria-describedby")).toBe("linked-ecn-demo-notice");
  link.focus();
  link.click(); // Native anchor Enter activation uses the same click path.
  expect(dialog.open).toBe(true);
  expect(dialog.textContent).toContain("No source relationship is supplied");
  expect(dialog.textContent).toContain("protective sleeve");
  expect(document.activeElement).toBe(dialog.querySelector("button"));
  dialog.querySelector("button").click();
  expect(dialog.open).toBe(false);
  expect(document.activeElement).toBe(link);
  link.click();
  dialog.dispatchEvent(new Event("cancel", { cancelable: true }));
  expect(dialog.open).toBe(false);
  expect(document.activeElement).toBe(link);
  expect(api.getEcn).toHaveBeenCalledTimes(1);
  expect(api.listEcns).not.toHaveBeenCalled();
  expect(api.generateSummaries).not.toHaveBeenCalled();
  expect(fetch).not.toHaveBeenCalled();
});
it("keeps a single listener/dialog through source re-render and returns focus to the replacement link", async () => {
  const api = { getEcn: vi.fn().mockResolvedValue(detailEnvelope({ job: { state: "RUNNING" } })) };
  let update;
  const controller = new AbortController();
  cleanup = await createEcnDetailPage({ api, startPolling: ({ onUpdate }) => { update = onUpdate; return vi.fn(); } })({
    container: document.querySelector("main"), params: { ecn_row_id: "synthetic-row" },
    query: new URLSearchParams("demo-linked-ecn=1"), signal: controller.signal,
  });
  await vi.waitFor(() => expect(document.querySelector("[data-linked-ecn-demo]")).not.toBeNull());
  document.querySelector("[data-linked-ecn-demo]").click();
  update(detailEnvelope());
  update(detailEnvelope());
  expect(document.querySelectorAll("#linked-ecn-demo-dialog")).toHaveLength(1);
  const dialog = document.querySelector("#linked-ecn-demo-dialog");
  dialog.querySelector("button").click();
  const currentLink = document.querySelector("[data-linked-ecn-demo]");
  expect(document.activeElement).toBe(currentLink);
  currentLink.click();
  expect(dialog.showModal).toHaveBeenCalledTimes(2);
  controller.abort();
});
it("aborts an open preview, removes listeners, and never refocuses the departed route", async () => {
  const { controller } = await demo();
  const link = document.querySelector("[data-linked-ecn-demo]");
  const dialog = document.querySelector("#linked-ecn-demo-dialog");
  link.click();
  const focus = vi.spyOn(link, "focus");
  controller.abort();
  expect(dialog.open).toBe(false);
  expect(dialog.isConnected).toBe(false);
  expect(focus).not.toHaveBeenCalled();
  link.click();
  expect(dialog.showModal).toHaveBeenCalledTimes(1);
  cleanup(); // Idempotent after abort.
});
it("ignores opt-in in production and keeps synthetic literals/markup in a DEV-only import graph", async () => {
  vi.stubEnv("DEV", false);
  await mount("demo-linked-ecn=1");
  expect(document.querySelector("#linked-ecn-field dd button").disabled).toBe(true);
  expect(document.querySelector("#linked-ecn-demo-dialog")).toBeNull();
  const source = readFileSync(resolve("src/pages/ecn-detail/ecn-detail.js"), "utf8");
  expect(source).toMatch(/if \(import\.meta\.env\.DEV && context\.query\.get\("demo-linked-ecn"\) === "1"\) \{\s*\/\/[^\n]+\s*void import\("\.\.\/\.\.\/components\/demo\/linked-ecn-demo\.js"\)/);
  expect(source).not.toContain("ECN-DEMO-");
  expect(html).not.toContain("linked-ecn-demo-dialog");
});
