import { button, element } from "./workspace-dom.js";

export const MAX_DRAWING_PREVIEW_BYTES = 20 * 1024 * 1024;
const supported = { "application/pdf": /\.pdf$/i, "image/png": /\.png$/i, "image/jpeg": /\.jpe?g$/i };

/** Check declared metadata only; no file bytes are read or sent for selection. */
export function drawingPreviewError(file) {
  if (!(file instanceof File)) return "Choose a drawing file to preview.";
  if (!Object.hasOwn(supported, file.type) || !supported[file.type].test(file.name)) return "Choose a PDF, PNG, or JPEG file with a matching file extension. SVG and HTML are not supported.";
  if (!file.size) return "The selected file is empty. Choose a non-empty drawing.";
  if (file.size > MAX_DRAWING_PREVIEW_BYTES) return "The selected file exceeds the 20 MiB drawing limit.";
  return "";
}

/** View a drawing and expose its validated selection for explicit analysis. */
export function engineeringDrawingPreview(host, signal, { drawing = null, loadDrawing } = {}) {
  const attached = drawing && Object.hasOwn(supported, drawing.media_type)
    && supported[drawing.media_type].test(drawing.file_name || "");
  const section = element("section", "", "material-story-block engineering-drawing-preview");
  section.append(element("h3", "Engineering drawing"));
  const label = element("label", "Choose drawing");
  const input = document.createElement("input");
  input.type = "file"; input.accept = "application/pdf,image/png,image/jpeg";
  input.className = "drawing-preview-input";
  label.append(input);
  const help = element("p", "PDF, PNG, or JPEG, up to 20 MiB.", "helper-text");
  // IDs are unique even if multiple material workspaces are mounted.
  const key = `drawing-preview-${++engineeringDrawingPreview.sequence}`;
  help.id = `${key}-help`;
  input.setAttribute("aria-describedby", help.id);
  const status = element("p", attached ? drawing.file_name : "", "helper-text drawing-preview-status");
  status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
  const error = element("p", "", "drawing-preview-error"); error.setAttribute("role", "alert");
  const viewer = element("section", "", "drawing-preview-viewer"); viewer.id = `${key}-viewer`; viewer.hidden = true;
  viewer.setAttribute("role", "region"); viewer.setAttribute("aria-label", "Engineering drawing preview");
  const content = element("div", "", "drawing-preview-content");
  content.tabIndex = 0; content.setAttribute("role", "region");
  content.setAttribute("aria-label", "Drawing viewport. Scroll to inspect the drawing.");
  const close = button("Close preview / return", () => endView(true));
  const view = button("View engineering drawing", openView); view.className = "drawing-preview-open"; view.disabled = !attached;
  view.setAttribute("aria-controls", viewer.id); view.setAttribute("aria-expanded", "false");
  viewer.append(close, content);
  if (!attached) section.append(label, help);
  section.append(status, error, view, viewer); host.append(section);
  let selected = null, objectUrl = null, disposed = false, loading = false, viewSequence = 0, disposePdf = null;

  function endView(returnFocus = false) {
    viewSequence++;
    disposePdf?.(); disposePdf = null;
    loading = false; view.disabled = disposed || (!attached && !selected);
    content.replaceChildren();
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    viewer.hidden = true; view.setAttribute("aria-expanded", "false");
    if (selected) status.textContent = selected.name;
    if (returnFocus && view.isConnected) view.focus();
  }
  function selectFile() {
    if (disposed) return;
    endView(); selected = null; view.disabled = true;
    const file = input.files?.[0];
    error.textContent = drawingPreviewError(file);
    status.textContent = "";
    if (error.textContent) { input.value = ""; return; }
    selected = file; view.disabled = false;
    status.textContent = file.name;
  }
  async function openView() {
    if (disposed || loading || (!selected && !attached) || objectUrl) return;
    error.textContent = "";
    const request = ++viewSequence;
    try {
      if (!selected && attached) {
        loading = true; view.disabled = true; status.textContent = "Loading drawing…";
        const blob = await loadDrawing();
        if (disposed || request !== viewSequence) return;
        selected = new File([blob], drawing.file_name, { type: blob.type });
        if (drawingPreviewError(selected) || selected.type !== drawing.media_type) {
          selected = null;
          throw new Error("Invalid drawing response");
        }
      }
      objectUrl = URL.createObjectURL(selected);
      // Only file bytes are embedded; never consume an arbitrary source URL.
      if (!objectUrl.startsWith("blob:")) throw new Error("Invalid drawing preview URL");
      viewer.hidden = false; view.setAttribute("aria-expanded", "true");
      status.textContent = selected.name;
      close.focus();
      if (selected.type === "application/pdf") {
        loading = true; view.disabled = true;
        const download = element("a", "Download drawing"); download.href = objectUrl; download.download = selected.name;
        const pdfHost = element("div", "", "pdf-drawing-viewer");
        content.append(download, pdfHost);
        const { mountPdfDrawing } = await import("./pdf-drawing-viewer.js");
        if (disposed || request !== viewSequence) return;
        const pdf = mountPdfDrawing(pdfHost, objectUrl);
        disposePdf = pdf.dispose;
        await pdf.ready;
        return;
      }
      const media = document.createElement("img"); media.src = objectUrl; media.alt = `Drawing: ${selected.name}`;
      media.className = "drawing-preview-media";
      media.addEventListener("load", () => {
        if (!disposed && content.contains(media)) status.textContent = selected.name;
      }, { once: true });
      media.addEventListener("error", () => {
        if (disposed || !content.contains(media)) return;
        endView(true); error.textContent = "The drawing could not be displayed. Try again.";
      }, { once: true });
      content.append(media);
    } catch {
      if (!disposed && request === viewSequence) { endView(); error.textContent = "Could not open the drawing. Try again."; }
    } finally {
      if (request === viewSequence) { loading = false; view.disabled = disposed || (!attached && !selected); }
    }
  }
  function escapeView(event) {
    if (event.key !== "Escape" || viewer.hidden) return;
    event.preventDefault(); event.stopPropagation(); endView(true);
  }
  input.addEventListener("change", selectFile);
  section.addEventListener("keydown", escapeView);
  function dispose() {
    if (disposed) return;
    disposed = true; endView(); selected = null; input.value = ""; input.disabled = true; view.disabled = true;
    status.textContent = "";
    input.removeEventListener("change", selectFile); section.removeEventListener("keydown", escapeView);
    signal.removeEventListener("abort", dispose);
  }
  signal.addEventListener("abort", dispose, { once: true });
  if (signal.aborted) dispose();
  return { dispose, input: attached ? null : input, getSelectedFile: () => selected };
}
engineeringDrawingPreview.sequence = 0;
