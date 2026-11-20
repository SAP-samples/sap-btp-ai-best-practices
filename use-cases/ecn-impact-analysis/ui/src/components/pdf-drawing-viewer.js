import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
// Bundle as a .js worker: Staticfile servers may serve raw .mjs as binary.
import workerUrl from "pdfjs-dist/build/pdf.worker.mjs?worker&url";
import { button, element } from "./workspace-dom.js";

GlobalWorkerOptions.workerSrc = workerUrl;

/** Render drawing pages without relying on the browser's native PDF plugin. */
export function mountPdfDrawing(host, url) {
  const task = getDocument({ url, isEvalSupported: false });
  let documentPdf, renderTask, disposed = false, rendering = false, pageNumber = 1, zoom = 1;
  const toolbar = element("nav", "", "pdf-drawing-controls");
  toolbar.setAttribute("aria-label", "Drawing pages and zoom");
  const pageLabel = element("span"); pageLabel.setAttribute("role", "status");
  const pageHost = element("div", "", "pdf-drawing-page-host");
  const failure = element("p", "", "drawing-preview-error"); failure.setAttribute("role", "alert");
  const previous = button("Previous page", () => changeView(() => pageNumber--));
  const next = button("Next page", () => changeView(() => pageNumber++));
  const smaller = button("Zoom out", () => changeView(() => { zoom = Math.max(.5, zoom - .25); }));
  const larger = button("Zoom in", () => changeView(() => { zoom = Math.min(3, zoom + .25); }));
  for (const control of [previous, next, smaller, larger]) control.disabled = true;
  toolbar.append(previous, pageLabel, next, smaller, larger);
  host.append(toolbar, failure, pageHost);

  async function renderPage() {
    if (disposed) return;
    const page = await documentPdf.getPage(pageNumber);
    if (disposed) return;
    const base = page.getViewport({ scale: 1 });
    const width = Math.max(240, host.clientWidth || 900);
    const viewport = page.getViewport({ scale: width / base.width * zoom });
    const ratio = Math.min(devicePixelRatio || 1, 2, Math.sqrt(8_000_000 / (viewport.width * viewport.height)));
    const canvas = element("canvas", "", "pdf-drawing-page");
    canvas.width = Math.ceil(viewport.width * ratio); canvas.height = Math.ceil(viewport.height * ratio);
    canvas.style.width = `${viewport.width}px`; canvas.style.height = `${viewport.height}px`;
    canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", `Drawing page ${pageNumber}`);
    pageHost.replaceChildren(canvas);
    renderTask = page.render({ canvasContext: canvas.getContext("2d"), viewport,
      transform: ratio === 1 ? null : [ratio, 0, 0, ratio, 0, 0] });
    await renderTask.promise;
    renderTask = null;
    if (disposed) return;
    canvas.dataset.rendered = "true";
    pageLabel.textContent = `Page ${pageNumber} of ${documentPdf.numPages} · ${Math.round(zoom * 100)}%`;
    previous.disabled = pageNumber <= 1; next.disabled = pageNumber >= documentPdf.numPages;
    smaller.disabled = zoom <= .5; larger.disabled = zoom >= 3;
    page.cleanup();
  }
  async function redraw() {
    rendering = true;
    failure.textContent = "";
    try { await renderPage(); }
    catch { if (!disposed) failure.textContent = "Could not display this drawing page. Try again."; }
    finally { rendering = false; }
  }
  function changeView(change) {
    if (disposed || rendering || !documentPdf) return;
    change(); void redraw();
  }
  const ready = (async () => {
    documentPdf = await task.promise;
    await renderPage();
  })();
  function dispose() {
    if (disposed) return;
    disposed = true; renderTask?.cancel();
    void task.destroy().catch(() => {});
    host.replaceChildren();
  }
  return { ready, dispose };
}
