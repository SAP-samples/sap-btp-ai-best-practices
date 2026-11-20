import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { drawingPreviewError, engineeringDrawingPreview, MAX_DRAWING_PREVIEW_BYTES } from "../src/components/engineering-drawing-preview.js";
import { materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";
import { mountPdfDrawing } from "../src/components/pdf-drawing-viewer.js";
vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn(), getEngineeringDrawing: vi.fn() }));
vi.mock("../src/components/pdf-drawing-viewer.js", () => ({ mountPdfDrawing: vi.fn((host) => {
  const canvas = document.createElement("canvas"); canvas.className = "pdf-drawing-page"; host.append(canvas);
  return { ready: Promise.resolve(), dispose: () => host.replaceChildren() };
}) }));

let controller, create, revoke;
beforeEach(() => {
  document.body.replaceChildren(); vi.clearAllMocks();
  controller = new AbortController();
  create = vi.fn().mockReturnValue("blob:synthetic-local-drawing"); revoke = vi.fn();
  vi.stubGlobal("URL", class extends URL { static createObjectURL = create; static revokeObjectURL = revoke; });
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});
afterEach(() => { controller.abort(); vi.unstubAllGlobals(); });
function mount(options) {
  const host = document.createElement("div"); document.body.append(host);
  const { dispose } = engineeringDrawingPreview(host, controller.signal, options);
  return { host, dispose };
}
function select(host, file) {
  const input = host.querySelector(".drawing-preview-input");
  Object.defineProperty(input, "files", { configurable: true, value: file ? [file] : [] });
  input.dispatchEvent(new Event("change"));
}
const file = (name = "synthetic.png", type = "image/png") => new File(["synthetic drawing bytes"], name, { type });
const open = host => host.querySelector(".drawing-preview-open").click();
const close = host => host.querySelector(".drawing-preview-viewer button").click();
const object = report => ({ number: "SIM-PART", description: "Synthetic material", material_brief: { sourcing: [] }, manufacturing_feasibility: report });

describe("local drawing preview only", () => {
  it.each([
    ["drawing.pdf", "application/pdf", "canvas"],
    ["drawing.png", "image/png", "img"],
    ["drawing.JPG", "image/jpeg", "img"],
    ["drawing.jpeg", "image/jpeg", "img"],
  ])("selects %s without reading/uploading and opens only on View", async (name, type, tag) => {
    const { host } = mount(); const chosen = file(name, type);
    chosen.arrayBuffer = vi.fn(); chosen.text = vi.fn();
    expect(host.textContent).not.toContain("No viewable drawing");
    expect(host.textContent).not.toContain("Local preview");
    expect(host.querySelector("label").textContent).toContain("Choose drawing");
    expect(host.querySelector("label input").getAttribute("aria-describedby")).toContain("-help");
    expect(host.querySelector(".drawing-preview-open").disabled).toBe(true);
    select(host, chosen);
    expect(create).not.toHaveBeenCalled(); expect(chosen.arrayBuffer).not.toHaveBeenCalled(); expect(chosen.text).not.toHaveBeenCalled();
    expect(api.assessDrawing).not.toHaveBeenCalled(); expect(api.getObject).not.toHaveBeenCalled();
    open(host);
    await vi.waitFor(() => expect(host.querySelector(tag)).not.toBeNull());
    expect(create).toHaveBeenCalledExactlyOnceWith(chosen);
    const media = host.querySelector(tag);
    expect(media).not.toBeNull();
    expect(type === "application/pdf" ? host.querySelector("a[download]").href : media.src).toBe("blob:synthetic-local-drawing");
    expect(host.querySelector(".drawing-preview-status").textContent).toBe(name);
    expect(host.querySelector(".drawing-preview-content").tabIndex).toBe(0);
    media.dispatchEvent(new Event("load"));
    expect(host.querySelector(".drawing-preview-status").textContent).toBe(name);
    expect(document.activeElement).toBe(host.querySelector(".drawing-preview-viewer button"));
    close(host); expect(revoke).toHaveBeenCalledExactlyOnceWith("blob:synthetic-local-drawing");
    expect(document.activeElement).toBe(host.querySelector(".drawing-preview-open"));
    expect(host.querySelector(".drawing-preview-open").getAttribute("aria-expanded")).toBe("false");
    expect(host.querySelector(".drawing-preview-viewer").hidden).toBe(true);
    expect(host.querySelector(tag)).toBeNull();
  });

  it.each([
    [null, "Choose a drawing"],
    [file("drawing.svg", "image/svg+xml"), "SVG and HTML"],
    [file("drawing.html", "text/html"), "SVG and HTML"],
    [file("drawing.png", "text/html"), "matching file extension"],
    [file("drawing.pdf", "image/png"), "matching file extension"],
    [file("drawing.png", ""), "matching file extension"],
    [new File([], "drawing.pdf", { type: "application/pdf" }), "empty"],
    [{ name: "javascript:evil.pdf", type: "application/pdf", size: 1 }, "Choose a drawing"],
  ])("rejects missing/invalid declared file metadata %#", (chosen, message) => {
    const { host } = mount(); select(host, chosen); open(host);
    expect(host.querySelector('[role="alert"]').textContent).toContain(message);
    expect(host.querySelector(".drawing-preview-open").disabled).toBe(true);
    expect(create).not.toHaveBeenCalled(); expect(api.assessDrawing).not.toHaveBeenCalled();
  });

  it("enforces a finite 20 MiB limit including the exact boundary", () => {
    const chosen = file(); Object.defineProperty(chosen, "size", { configurable: true, value: MAX_DRAWING_PREVIEW_BYTES });
    expect(drawingPreviewError(chosen)).toBe("");
    Object.defineProperty(chosen, "size", { value: MAX_DRAWING_PREVIEW_BYTES + 1 });
    const { host } = mount(); select(host, chosen);
    expect(host.querySelector('[role="alert"]').textContent).toContain("20 MiB"); expect(create).not.toHaveBeenCalled();
  });

  it("revokes on replacement, invalid replacement, Escape and disposal without double revoke", () => {
    const { host, dispose } = mount(); select(host, file()); open(host); open(host);
    expect(create).toHaveBeenCalledTimes(1);
    const staleMedia = host.querySelector("img");
    select(host, file("replacement.jpg", "image/jpeg")); expect(revoke).toHaveBeenCalledTimes(1);
    staleMedia.dispatchEvent(new Event("load")); staleMedia.dispatchEvent(new Event("error"));
    expect(host.querySelector(".drawing-preview-status").textContent).toBe("replacement.jpg");
    expect(host.querySelector('[role="alert"]').textContent).toBe("");
    open(host);
    const escape = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    host.querySelector(".drawing-preview-viewer button").dispatchEvent(escape);
    expect(escape.defaultPrevented).toBe(true); expect(revoke).toHaveBeenCalledTimes(2);
    expect(document.activeElement).toBe(host.querySelector(".drawing-preview-open"));
    open(host); select(host, file("bad.svg", "image/svg+xml")); expect(revoke).toHaveBeenCalledTimes(3);
    select(host, file()); open(host); dispose(); dispose(); controller.abort();
    expect(revoke).toHaveBeenCalledTimes(4); expect(host.querySelector("img")).toBeNull();
    expect(host.querySelector("input").disabled).toBe(true);
  });

  it("revokes on abort and never opens or processes selections after disposal", () => {
    const { host } = mount(); select(host, file()); open(host); controller.abort();
    expect(revoke).toHaveBeenCalledTimes(1); select(host, file()); open(host);
    expect(create).toHaveBeenCalledTimes(1); expect(host.querySelector("img")).toBeNull();
  });

  it("reports rendering/URL failures locally and clears stale errors on valid selection", () => {
    const { host } = mount(); select(host, file()); open(host);
    host.querySelector("img").dispatchEvent(new Event("error"));
    expect(host.querySelector('[role="alert"]').textContent).toContain("could not be displayed"); expect(revoke).toHaveBeenCalledTimes(1);
    select(host, file()); expect(host.querySelector('[role="alert"]').textContent).toBe("");
    create.mockImplementation(() => { throw new Error("synthetic unavailable"); }); open(host);
    expect(host.querySelector('[role="alert"]').textContent).toContain("Could not open the drawing");
    create.mockReturnValue("javascript:evil"); open(host);
    expect(host.querySelector("img")).toBeNull(); expect(revoke).toHaveBeenLastCalledWith("javascript:evil");
    expect(api.assessDrawing).not.toHaveBeenCalled();
  });

  it("renders a hostile filename as text, never markup or an arbitrary source URL", () => {
    const { host } = mount(); select(host, file('<img src=x onerror=evil>.png'));
    expect(host.querySelector("img")).toBeNull(); expect(host.textContent).toContain('<img src=x onerror=evil>.png');
    open(host); expect(host.querySelector("img").getAttribute("src")).toBe("blob:synthetic-local-drawing");
  });
});

describe("material workspace integration", () => {
  it("returns focus when closing a PDF during rendering and can reopen without stale errors", async () => {
    let finishRendering;
    const disposePdf = vi.fn();
    mountPdfDrawing.mockImplementationOnce(() => ({
      ready: new Promise(resolve => { finishRendering = resolve; }), dispose: disposePdf,
    }));
    const { host } = mount(); select(host, file("drawing.pdf", "application/pdf"));
    open(host);
    await vi.waitFor(() => expect(finishRendering).toBeTypeOf("function"));
    expect(host.querySelector(".drawing-preview-open").disabled).toBe(true);
    close(host);
    expect(disposePdf).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(host.querySelector(".drawing-preview-open"));
    finishRendering();
    await Promise.resolve();
    open(host);
    await vi.waitFor(() => expect(host.querySelector("canvas")).not.toBeNull());
    expect(host.querySelector('[role="alert"]').textContent).toBe("");
  });

  it("shows the attached sample without file inputs and retrieves its bytes only on View", async () => {
    const source = { ...object(null), number: "QKX5525860", engineering_drawing: {
      file_name: "QKX5525860_2_AFS.pdf", media_type: "application/pdf",
    } };
    api.getObject.mockResolvedValue(source);
    api.getEngineeringDrawing.mockResolvedValue(new Blob(["%PDF-1.7 sample"], { type: "application/pdf" }));
    await materialDialog("synthetic-id", controller.signal, vi.fn())("QKX5525860");
    const dialog = document.querySelector("dialog");
    expect(dialog.querySelector('input[type="file"]')).toBeNull();
    expect(dialog.textContent).not.toContain("No viewable drawing");
    expect(dialog.textContent).not.toContain("Local preview");
    expect(api.getEngineeringDrawing).not.toHaveBeenCalled();
    open(dialog);
    await vi.waitFor(() => expect(dialog.querySelector("canvas")).not.toBeNull());
    expect(api.getEngineeringDrawing).toHaveBeenCalledExactlyOnceWith("synthetic-id", "QKX5525860", api.getObject.mock.calls[0][2]);
    expect(dialog.querySelector("a[download]").href).toBe("blob:synthetic-local-drawing");
    close(dialog);
    expect(revoke).toHaveBeenCalledTimes(1);
    expect(dialog.querySelector('input[type="file"]')).toBeNull();
    expect(api.assessDrawing).not.toHaveBeenCalled();
  });

  it("keeps the attached drawing retryable after load failure and ignores a late load after disposal", async () => {
    const drawing = { file_name: "sample.pdf", media_type: "application/pdf" };
    let resolveLoad;
    const loadDrawing = vi.fn().mockRejectedValueOnce(new Error("Unavailable"))
      .mockImplementationOnce(() => new Promise(resolve => { resolveLoad = resolve; }));
    const { host, dispose } = mount({ drawing, loadDrawing });
    open(host);
    await vi.waitFor(() => expect(host.querySelector('[role="alert"]').textContent).toContain("Could not open"));
    expect(host.querySelector(".drawing-preview-open").disabled).toBe(false);
    expect(host.querySelector('input[type="file"]')).toBeNull();
    open(host);
    expect(host.querySelector(".drawing-preview-open").disabled).toBe(true);
    dispose();
    resolveLoad(new Blob(["%PDF-1.7 sample"], { type: "application/pdf" }));
    await Promise.resolve();
    expect(create).not.toHaveBeenCalled();
    expect(host.querySelector("canvas")).toBeNull();
  });

  it("leaves stored reports without bytes unattached and immutable with no extra API calls", async () => {
    const report = { drawing_file_name: "stored.pdf", drawing_sha256: "synthetic-hash", drawing_url: "javascript:evil", candidate_reports: [{ label: "Synthetic", result: "Inconclusive", criteria: [] }] };
    const source = object(report); const before = JSON.stringify(source); api.getObject.mockResolvedValue(source);
    const inspect = materialDialog("synthetic-id", controller.signal, vi.fn()); await inspect("SIM-PART");
    const dialog = document.querySelector("dialog"); const reportHost = dialog.querySelector(".drawing-feasibility-report-host");
    const savedReport = reportHost.textContent, reportNode = reportHost.firstElementChild;
    expect(dialog.querySelector(".engineering-drawing-preview").textContent).not.toContain("No viewable drawing");
    expect(dialog.querySelector('[aria-label="Pipe Drawing File"]')).toBeNull(); expect(create).not.toHaveBeenCalled();
    select(dialog, file("local.pdf", "application/pdf")); open(dialog);
    await vi.waitFor(() => expect(dialog.querySelector("canvas")).not.toBeNull());
    close(dialog);
    expect(reportHost.textContent).toBe(savedReport);
    expect(reportHost.firstElementChild).toBe(reportNode);
    expect(JSON.stringify(source)).toBe(before); expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
  });

  it("cleans up on material replacement and outer Close and restores the original opener", async () => {
    api.getObject.mockImplementation(async (_, params) => ({ ...object(null), number: params.number }));
    const opener = document.createElement("button"); document.body.append(opener); opener.focus();
    const inspect = materialDialog("synthetic-id", controller.signal, vi.fn()); await inspect("FIRST");
    const dialog = document.querySelector("dialog"); select(dialog, file()); open(dialog);
    await inspect("SECOND"); expect(revoke).toHaveBeenCalledTimes(1); expect(dialog.querySelector(".drawing-preview-open").disabled).toBe(true);
    select(dialog, file()); open(dialog); dialog.querySelector(".object-dialog-footer button").click();
    expect(revoke).toHaveBeenCalledTimes(2); expect(document.activeElement).toBe(opener);
    controller.abort(); expect(revoke).toHaveBeenCalledTimes(2); expect(document.querySelector("dialog")).toBeNull();
  });

  it("shares one validated file between preview and explicit analysis", async () => {
    api.getObject.mockResolvedValue(object(null)); await materialDialog("synthetic-id", controller.signal, vi.fn())("SIM-PART");
    const dialog = document.querySelector("dialog"), chosen = file(); select(dialog, chosen); open(dialog);
    expect(dialog.querySelectorAll('input[type="file"]')).toHaveLength(1);
    expect(api.assessDrawing).not.toHaveBeenCalled(); expect(api.getObject).toHaveBeenCalledTimes(1);
    api.assessDrawing.mockResolvedValue({ candidate_reports: [] });
    [...dialog.querySelectorAll("button")].find(node => node.textContent === "Analyze Drawing").click();
    await vi.waitFor(() => expect(api.assessDrawing).toHaveBeenCalledExactlyOnceWith(chosen, api.getObject.mock.calls[0][2]));
    expect(dialog.querySelectorAll('input[type="file"]')).toHaveLength(1);
  });

  it("bounds the viewer and uses theme-token focus styles without private UI5 selectors", () => {
    // Resolve from test cwd; import.meta.url and CSS raw transforms are unreliable in this harness.
    const css = readFileSync(resolve(process.cwd(), "src/style.css"), "utf8");
    expect(css).toContain(".drawing-preview-viewer[hidden] { display: none; }");
    expect(css).toContain("max-height: min(32rem, 55vh)"); expect(css).toContain("overscroll-behavior: contain");
    expect(css).toContain(".engineering-drawing-preview button:focus-visible"); expect(css).toContain("outline: 2px solid var(--sapContent_FocusColor)");
  });
});
