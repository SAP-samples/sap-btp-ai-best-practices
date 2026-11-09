/* Chat UI5 components */
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/SuggestionItem.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Link.js";
import "@ui5/webcomponents/dist/Bar.js";
import "@ui5/webcomponents-icons/dist/navigation-right-arrow.js";
import "@ui5/webcomponents-icons/dist/decline.js";
import "@ui5/webcomponents-icons/dist/business-objects-experience.js";
import "@ui5/webcomponents/dist/TabContainer.js";
import "@ui5/webcomponents/dist/Tab.js";
import "@ui5/webcomponents-icons/dist/search.js";
import "@ui5/webcomponents-icons/dist/people-connected.js";
import "@ui5/webcomponents/dist/List.js";
import "@ui5/webcomponents/dist/ListItemStandard.js";
import "@ui5/webcomponents/dist/ListItemCustom.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Menu.js";
import "@ui5/webcomponents/dist/MenuItem.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/illustrations/NoSearchResults.js";
import "@ui5/webcomponents-fiori/dist/illustrations/NoActivities.js";
import "@ui5/webcomponents-icons/dist/overflow.js";
import "@ui5/webcomponents-icons/dist/delete.js";
import "@ui5/webcomponents-icons/dist/competitor.js";
import "@ui5/webcomponents-icons/dist/home.js";
import "@ui5/webcomponents-icons/dist/locked.js";
import "@ui5/webcomponents-icons/dist/copy.js";
import "@ui5/webcomponents-icons/dist/accept.js";
import "@ui5/webcomponents-icons/dist/share-2.js";
import "@ui5/webcomponents-icons/dist/hint.js";
import "@ui5/webcomponents-icons/dist/information.js";
import "@ui5/webcomponents-icons/dist/attachment.js";
import "@ui5/webcomponents-icons/dist/pdf-attachment.js";
import "@ui5/webcomponents-icons/dist/doc-attachment.js";
import "@ui5/webcomponents-icons/dist/ppt-attachment.js";
import "@ui5/webcomponents-icons/dist/excel-attachment.js";
import "@ui5/webcomponents-icons/dist/email.js";
import "@ui5/webcomponents-icons/dist/document-text.js";
import "@ui5/webcomponents-icons/dist/syntax.js";
import "@ui5/webcomponents-icons/dist/upload.js";
import "@ui5/webcomponents-icons/dist/edit.js";
import "@ui5/webcomponents-icons/dist/stop.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";
import { marked } from "marked";
import dayjs from "dayjs";
import calendar from "dayjs/plugin/calendar.js";

import { request } from "../../services/api.js";
import { getAuthHeaders, API_BASE_URL } from "../../services/api.js";
import { getUserInfo, hasCompanyEmail } from "../../services/auth.js";
import { evaluateSolution, submitSolution, getStoredEvaluation, getStoredEvaluations } from "../../services/evaluation.js";
import { getCompanyLeaderboard, openLeaderboardDialog } from "../../services/leaderboard.js";
import { createShare, getShares, revokeShare, removeMember } from "../../services/sharing.js";


const A2A_ENDPOINT = "/api/ideation-agent/a2a";
const UPLOAD_ENDPOINT = "/api/ideation-agent/upload";
const CONVERSATIONS_ENDPOINT = "/api/ideation-agent/conversations";
const DEFAULT_PLACEHOLDER = "Describe the challenge, idea, or use case you want to make real with AI...";
const CLARIFICATION_PLACEHOLDER = "Write your own answer here, or choose one of the options above...";
const REFINEMENT_PLACEHOLDER = "Share feedback or request changes...";
const DEFAULT_COMPOSER_NOTE = "Powered by AI. Verify results.";

let rpcId = 1;
let contextId = null;
let _inflightMessageId = null; // messageId of the in-flight turn (for the Stop/cancel endpoint)
let companyNameSent = false;
let activeEventId = null; // Set when navigating from an event workspace
let activeEventCode = null; // The event's join code for URL persistence
let conversationVariant = "provocateur"; // Critical Mode is the default

// Sharing state (module-level so module-level functions can access)
let _isReadOnly = false;
let _isSharedConversation = false;
let _handoverCardShown = false;

// Research drawer state
const ENRICHMENT_ENDPOINT = "/api/ideation-agent/enrichment";
const SUGGESTIONS_ENDPOINT = "/api/ideation-agent/company-suggestions";
const POLL_INTERVAL_MS = 3000;
let enrichmentStatus = "idle";
let enrichmentResult = null;
let perplexityStatus = "idle";
let perplexityResult = null;
let perplexityProblemStatus = "idle";
let perplexityProblemResult = null;
let linkedinStatus = "idle";
let linkedinResult = null;
let enrichmentPollTimer = null;
let drawerOpen = false;
let drawerAutoOpened = false;
let conversationComplete = false;

// Strategic reframe state
let reframeStatus = "idle";
let reframeResult = null;
let reframePollTimer = null;

// SAP AI Catalog matches state
let catalogMatchStatus = "idle";
let catalogMatchResult = null;
let catalogMatchPollTimer = null;

// Evaluation & leaderboard state (per-message)
// Map<msgIndex, {result, loading, ui, isOnLeaderboard, delta, rank, animate}>
const evaluations = new Map();
let activeLoadingTimers = new Map(); // msgIndex → intervalId for loading text progression
let _renderMessages = () => {}; // set by initIdeationAgentPage, called from module-level eval functions
let _submitUserMessage = () => {}; // set by initIdeationAgentPage, called from eval chip click handlers
const conversation = []; // shared conversation array (populated inside initIdeationAgentPage)

// File attachment state
const MAX_PENDING_FILES = 5;
let _pendingFiles = []; // { file: File, previewUrl: string, id: string }[]

function makeMessageId() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function extractTextParts(parts) {
    if (!Array.isArray(parts)) return "";
    return parts
        .filter((part) => part && part.kind === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n")
        .trim();
}

function extractAgentText(task) {
    const statusText = extractTextParts(task?.status?.message?.parts);
    if (statusText) return statusText;

    const artifacts = Array.isArray(task?.artifacts) ? task.artifacts : [];
    for (const artifact of artifacts) {
        const artifactText = extractTextParts(artifact?.parts);
        if (artifactText) return artifactText;
    }
    return "";
}

function extractContextId(task) {
    return task?.contextId || task?.status?.message?.contextId || null;
}

function extractUiMetadata(task) {
    return task?.status?.message?.metadata?.ui || task?.metadata?.ui || null;
}

function parseA2AError(response) {
    const err = response?.error;
    if (!err || typeof err !== "object") return "Unknown A2A error";
    const message = typeof err.message === "string" ? err.message : "Unknown A2A error";
    const detail = err?.data?.detail;
    return detail ? `${message}: ${detail}` : message;
}

function logFinalResultFields(ui) {
    if (!ui || ui.kind !== "final") {
        return;
    }

    console.log("Ideation Agent final result fields", {
        industry: ui.industry || "",
        title: ui.title || "",
        problem: ui.problem || "",
        selected_ai_capabilities: Array.isArray(ui.selected_ai_capabilities)
            ? ui.selected_ai_capabilities
            : [],
        solution: ui.solution || ""
    });
}

async function postA2AMessage(userText, companyName = "", displayText = "", uploadedFiles = [], signal = undefined, messageId = null) {
    const parts = [];
    if (userText) {
        parts.push({ kind: "text", text: userText });
    }
    // Add file parts (file_id references — server resolves to content blocks)
    for (const f of uploadedFiles) {
        parts.push({
            kind: "file",
            file_id: f.file_id,
            context_id: contextId,
            filename: f.filename,
            num_outputs: f.num_outputs || 1,
            original_type: f.original_type || "",
        });
    }
    // Must have at least one part
    if (parts.length === 0) {
        parts.push({ kind: "text", text: "" });
    }

    const mid = messageId || makeMessageId();
    _inflightMessageId = mid; // published so Stop can target this turn via the cancel endpoint
    const message = {
        role: "user",
        parts,
        messageId: mid
    };
    if (contextId) {
        message.contextId = contextId;
    }
    if (displayText) {
        message.metadata = { displayText };
    }

    const params = { message };
    params.metadata = { variant: "provocateur" };
    if (companyName) params.metadata.company_name = companyName;
    if (activeEventId) params.metadata.event_id = activeEventId;

    const payload = {
        jsonrpc: "2.0",
        id: rpcId++,
        method: "message/send",
        params
    };

    const response = await request(A2A_ENDPOINT, "POST", payload, {}, signal);
    if (response?.error) {
        throw new Error(parseA2AError(response));
    }

    const task = response?.result;
    if (!task || typeof task !== "object") {
        throw new Error("A2A response did not include a task result");
    }

    const nextContextId = extractContextId(task);
    if (nextContextId) {
        const isFirstAssignment = !contextId;
        contextId = nextContextId;
        // Persist new conversation ID in URL on first assignment
        if (isFirstAssignment) {
            const url = new URL(window.location);
            url.searchParams.set("c", contextId);
            if (activeEventId) {
                url.searchParams.set("event_id", activeEventId);
            }
            if (activeEventCode) {
                url.searchParams.set("event_code", activeEventCode);
            }
            history.replaceState(null, "", url);
        }
    }

    const finalText = extractAgentText(task);
    if (!finalText) {
        throw new Error("A2A response did not include text output");
    }

    return {
        text: finalText,
        ui: extractUiMetadata(task),
        contextId,
        generatedTitle: task?.metadata?.generatedTitle || null,
    };
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

/** Strip inline markdown bold/italic markers for plain-text display. */
function stripInlineMarkdown(str) {
    return (str || "")
        .replace(/\*\*(.+?)\*\*/g, "$1")
        .replace(/__(.+?)__/g, "$1")
        .replace(/\*(.+?)\*/g, "$1")
        .replace(/_(.+?)_/g, "$1");
}

// --- Agent loading messages (typewriter animation) ---
const _AGENT_LOADING_MESSAGES = [
    "Thinking",
    "Analyzing your request",
    "Understanding the context",
    "Exploring possibilities",
    "Connecting the dots",
    "Considering different angles",
    "Identifying key patterns",
    "Building on your input",
    "Evaluating trade-offs",
    "Looking at this from another perspective",
    "Refining the approach",
    "Weighing the options",
    "Digging deeper",
    "Structuring the response",
    "Synthesizing insights",
    "Challenging assumptions",
    "Mapping out dependencies",
    "Thinking through edge cases",
    "Crafting a response",
    "Putting it all together",
    "Polishing the details",
    "Almost there",
];
let _loadingMsgIndex = 0;
let _loadingTypeTimer = null;
let _loadingRotateTimer = null;

function _getLoadingMessage() {
    return _AGENT_LOADING_MESSAGES[_loadingMsgIndex % _AGENT_LOADING_MESSAGES.length];
}

function _typeLoadingMessage(el) {
    if (!el) return;
    const text = _getLoadingMessage();
    let charIndex = 0;
    el.textContent = "";

    clearInterval(_loadingTypeTimer);
    _loadingTypeTimer = setInterval(() => {
        if (charIndex <= text.length) {
            el.textContent = text.slice(0, charIndex) + "…";
            charIndex++;
        } else {
            clearInterval(_loadingTypeTimer);
        }
    }, 30);
}

function _startLoadingMessages() {
    _loadingMsgIndex = 0;
    clearInterval(_loadingRotateTimer);
    _loadingRotateTimer = setInterval(() => {
        _loadingMsgIndex++;
        const hint = document.querySelector(".loader-hint");
        if (hint) _typeLoadingMessage(hint);
    }, 3500);
}

function _stopLoadingMessages() {
    clearInterval(_loadingTypeTimer);
    clearInterval(_loadingRotateTimer);
    _loadingTypeTimer = null;
    _loadingRotateTimer = null;
    _loadingMsgIndex = 0;
}

function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

// --- File attachment helpers (module-level for reuse) -------------------------

function _addPendingFiles(files) {
    const remaining = MAX_PENDING_FILES - _pendingFiles.length;
    if (remaining <= 0) {
        _showFileToast(`Maximum ${MAX_PENDING_FILES} files allowed.`);
        return;
    }
    const toAdd = files.slice(0, remaining);
    for (const file of toAdd) {
        _pendingFiles.push({
            file,
            previewUrl: URL.createObjectURL(file),
            id: crypto.randomUUID ? crypto.randomUUID() : `f-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        });
    }
    if (files.length > remaining) {
        _showFileToast(`Only ${remaining} more file(s) allowed. ${files.length - remaining} skipped.`);
    }
    _renderAttachmentStrip();
    _updateComposerForFiles();
}

function _removePendingFile(id) {
    const idx = _pendingFiles.findIndex(f => f.id === id);
    if (idx >= 0) {
        URL.revokeObjectURL(_pendingFiles[idx].previewUrl);
        _pendingFiles.splice(idx, 1);
    }
    _renderAttachmentStrip();
    _updateComposerForFiles();
}

function _clearPendingFiles() {
    for (const pf of _pendingFiles) URL.revokeObjectURL(pf.previewUrl);
    _pendingFiles = [];
    _renderAttachmentStrip();
    _updateComposerForFiles();
}

function _renderAttachmentStrip() {
    const strip = document.getElementById("attachment-strip");
    if (!strip) return;
    strip.innerHTML = "";
    strip.hidden = _pendingFiles.length === 0;

    for (const pf of _pendingFiles) {
        const fType = pf.file.type || "";
        const isPdf = fType === "application/pdf";
        const isImage = fType.startsWith("image/");

        // Only images and PDFs get a clickable preview
        let clickHandler = null;
        if (isPdf) {
            clickHandler = () => _openLightboxPdf(pf.previewUrl);
        } else if (isImage) {
            clickHandler = () => _openLightbox(pf.previewUrl);
        }

        const chip = _createFileChip({
            filename: pf.file.name,
            fileType: fType,
            onClick: clickHandler,
        });
        chip.classList.add("attachment-chip-pending");

        const removeBtn = document.createElement("ui5-button");
        removeBtn.setAttribute("icon", "decline");
        removeBtn.setAttribute("design", "Transparent");
        removeBtn.setAttribute("tooltip", "Remove");
        removeBtn.className = "attachment-chip-remove";
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            _removePendingFile(pf.id);
        });
        chip.appendChild(removeBtn);

        strip.appendChild(chip);
    }
}

function _updateComposerForFiles() {
    // Re-evaluate send button state (files alone can enable send)
    const input = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    if (!sendButton) return;
    const hasText = (input?.value || "").trim().length > 0;
    const hasFiles = _pendingFiles.length > 0;
    sendButton.disabled = !(hasText || hasFiles);
}

async function _uploadPendingFiles(filesToUpload, uploadContextId) {
    /** Upload files to Object Store. Returns array of file references. */
    const results = [];
    const headers = await getAuthHeaders();

    await Promise.all(filesToUpload.map(async (pf) => {
        const formData = new FormData();
        formData.append("file", pf.file);
        try {
            const response = await fetch(
                `${API_BASE_URL}${UPLOAD_ENDPOINT}?context_id=${encodeURIComponent(uploadContextId)}`,
                { method: "POST", headers, body: formData }
            );
            if (!response.ok) {
                const detail = await response.json().catch(() => ({}));
                console.warn(`Upload failed for ${pf.file.name}:`, detail.detail || response.status);
                return;
            }
            const data = await response.json();
            results.push({
                file_id: data.file_id,
                filename: data.filename,
                num_outputs: data.num_outputs,
                original_type: data.original_type || "",
                dimensions: data.dimensions,
                _localPreviewUrl: pf.previewUrl,
            });
        } catch (err) {
            console.warn(`Upload error for ${pf.file.name}:`, err);
        }
    }));

    return results;
}

function _showFileToast(msg) {
    const toast = document.getElementById("file-toast");
    if (toast) {
        toast.textContent = msg;
        toast.open = true;
    }
}

function _openLightbox(src) {
    const lb = document.getElementById("image-lightbox");
    const img = document.getElementById("lightbox-img");
    const pdf = document.getElementById("lightbox-pdf");
    const loader = document.getElementById("lightbox-loader");
    if (!lb || !img) return;
    if (pdf) { pdf.src = ""; pdf.hidden = true; }
    if (loader) loader.hidden = true;
    img.hidden = false;
    img.src = src;
    lb.hidden = false;
}

/**
 * Determine the appropriate icon name for a file based on its MIME type or extension.
 * @param {string} fileType - MIME type
 * @param {string} filename - Original filename
 * @returns {string} UI5 icon name
 */
function _getFileIcon(fileType, filename) {
    const type = (fileType || "").toLowerCase();
    const ext = (filename || "").split(".").pop()?.toLowerCase() || "";

    // PDF
    if (type === "application/pdf" || ext === "pdf") return "pdf-attachment";

    // Word / RTF / ODT
    if (type.includes("wordprocessingml") || type === "application/msword" ||
        type === "application/rtf" || type === "application/vnd.oasis.opendocument.text" ||
        ["doc", "docx", "rtf", "odt"].includes(ext)) return "doc-attachment";

    // PowerPoint / ODP
    if (type.includes("presentationml") || type === "application/vnd.ms-powerpoint" ||
        type === "application/vnd.oasis.opendocument.presentation" ||
        ["ppt", "pptx", "odp"].includes(ext)) return "ppt-attachment";

    // Excel / ODS / CSV / TSV
    if (type.includes("spreadsheetml") || type === "application/vnd.ms-excel" ||
        type === "application/vnd.oasis.opendocument.spreadsheet" ||
        type === "text/csv" || type === "text/tab-separated-values" ||
        ["xls", "xlsx", "ods", "csv", "tsv"].includes(ext)) return "excel-attachment";

    // Email / MHTML
    if (type === "message/rfc822" || ["eml", "mht", "mhtml"].includes(ext)) return "email";

    // Code files
    if (type === "application/json" || type === "text/xml" || type === "text/yaml" ||
        type === "text/x-python" || type === "text/javascript" || type === "text/css" ||
        type === "application/sql" || type === "text/html" ||
        ["json", "xml", "yaml", "yml", "py", "js", "ts", "html", "htm", "css", "sql"].includes(ext)) return "syntax";

    // Plain text / Markdown
    if (type === "text/plain" || type === "text/markdown" ||
        ["txt", "md", "markdown", "text"].includes(ext)) return "document-text";

    // Images (fallback)
    if (type.startsWith("image/")) return "picture";

    // Generic fallback
    return "document-text";
}

/** Set of accepted file extensions for drag-and-drop (matches the file input accept attribute). */
const _ACCEPTED_EXTENSIONS = new Set([
    "png", "jpg", "jpeg", "gif", "webp",
    "pdf",
    "doc", "docx", "ppt", "pptx", "xls", "xlsx",
    "csv", "tsv",
    "odt", "odp", "ods",
    "rtf",
    "eml", "mht", "mhtml",
    "txt", "md", "json", "xml", "yaml", "yml",
    "py", "js", "ts", "html", "htm", "css", "sql",
]);

/**
 * Check if a File object is an accepted type for upload.
 * @param {File} file
 * @returns {boolean}
 */
function _isAcceptedFileType(file) {
    // Check MIME type first
    if (file.type.startsWith("image/")) return true;
    if (file.type === "application/pdf") return true;
    if (file.type.startsWith("text/")) return true;
    if (file.type === "application/json") return true;
    if (file.type.includes("officedocument") || file.type.includes("opendocument")) return true;
    if (file.type === "application/msword" || file.type === "application/vnd.ms-powerpoint" || file.type === "application/vnd.ms-excel") return true;
    if (file.type === "application/rtf" || file.type === "message/rfc822") return true;

    // Fallback: check extension
    const ext = (file.name || "").split(".").pop()?.toLowerCase() || "";
    return _ACCEPTED_EXTENSIONS.has(ext);
}

/**
 * Create a file chip element (icon + filename). Used in both the attachment
 * strip and chat bubbles for consistent rendering.
 * @param {object} opts
 * @param {string} opts.filename - Display name
 * @param {function} opts.onClick - Click handler (opens preview)
 * @returns {HTMLDivElement}
 */
function _createFileChip({ filename, onClick, fileType, isDownload }) {
    const chip = document.createElement("div");
    chip.className = "bubble-file-chip";
    chip.title = filename || "Attached file";

    const iconName = _getFileIcon(fileType, filename);
    const icon = document.createElement("ui5-icon");
    icon.setAttribute("name", iconName);
    chip.appendChild(icon);

    const nameEl = document.createElement("span");
    nameEl.className = "bubble-file-name";
    nameEl.textContent = filename || "File";
    chip.appendChild(nameEl);

    if (onClick) {
        chip.addEventListener("click", (e) => {
            if (e.target.closest(".attachment-chip-remove")) return;
            if (isDownload) {
                chip.classList.add("chip-downloading");
                Promise.resolve(onClick()).finally(() => {
                    chip.classList.remove("chip-downloading");
                });
            } else {
                onClick();
            }
        });
    }

    return chip;
}

async function _loadAuthImage(imgEl, url) {
    /** Fetch an image with auth headers and set it as a blob URL on the img element. */
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(url, { headers });
        if (!response.ok) return;
        const blob = await response.blob();
        imgEl.src = URL.createObjectURL(blob);
    } catch (err) {
        console.warn("Failed to load auth image:", err);
    }
}

async function _openLightboxAuth(url) {
    /** Open lightbox with an authenticated image fetch. */
    const lb = document.getElementById("image-lightbox");
    const img = document.getElementById("lightbox-img");
    const pdf = document.getElementById("lightbox-pdf");
    const loader = document.getElementById("lightbox-loader");
    if (!lb || !img) return;
    if (pdf) { pdf.src = ""; pdf.hidden = true; }
    img.src = "";
    img.hidden = true;
    if (loader) loader.hidden = false;
    lb.hidden = false;
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(url, { headers });
        if (!response.ok) {
            lb.hidden = true;
            return;
        }
        const blob = await response.blob();
        img.src = URL.createObjectURL(blob);
        img.onload = () => {
            img.hidden = false;
            if (loader) loader.hidden = true;
        };
    } catch (err) {
        console.warn("Failed to load lightbox image:", err);
        lb.hidden = true;
    }
}

function _closeLightbox() {
    const lb = document.getElementById("image-lightbox");
    const img = document.getElementById("lightbox-img");
    const pdf = document.getElementById("lightbox-pdf");
    if (lb) lb.hidden = true;
    if (img) img.src = "";
    if (pdf) { pdf.src = ""; pdf.hidden = true; }
}

function _openLightboxPdf(src) {
    const lb = document.getElementById("image-lightbox");
    const img = document.getElementById("lightbox-img");
    const pdf = document.getElementById("lightbox-pdf");
    const loader = document.getElementById("lightbox-loader");
    if (!lb || !pdf) return;
    if (img) { img.src = ""; img.hidden = true; }
    if (loader) loader.hidden = true;
    pdf.src = src;
    pdf.hidden = false;
    lb.hidden = false;
}

async function _openLightboxPdfAuth(url) {
    /** Open lightbox PDF viewer with an authenticated fetch. */
    const lb = document.getElementById("image-lightbox");
    const img = document.getElementById("lightbox-img");
    const pdf = document.getElementById("lightbox-pdf");
    const loader = document.getElementById("lightbox-loader");
    if (!lb || !pdf) return;
    if (img) { img.src = ""; img.hidden = true; }
    pdf.hidden = true;
    if (loader) loader.hidden = false;
    lb.hidden = false;
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(url, { headers });
        if (!response.ok) { lb.hidden = true; return; }
        const blob = await response.blob();
        pdf.src = URL.createObjectURL(blob);
        pdf.hidden = false;
        if (loader) loader.hidden = true;
    } catch (err) {
        console.warn("Failed to load lightbox PDF:", err);
        lb.hidden = true;
    }
}

async function _downloadFileAuth(url, filename) {
    /** Download a file via authenticated fetch + programmatic <a> click. */
    try {
        const headers = await getAuthHeaders();
        const res = await fetch(url, { headers });
        if (!res.ok) {
            console.warn("Download failed:", res.status);
            return;
        }
        const blob = await res.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = blobUrl;
        a.download = filename || "download";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    } catch (err) {
        console.warn("Download error:", err);
    }
}

function showResearchBadge() {}

function hideResearchBadge() {}

function openDrawer() {
    const drawer = document.getElementById("research-drawer");
    const backdrop = document.getElementById("research-backdrop");
    drawerOpen = true;
    hideResearchBadge();
    drawer?.removeAttribute("hidden");
    backdrop?.removeAttribute("hidden");
    requestAnimationFrame(() => {
        drawer?.classList.add("open");
        backdrop?.classList.add("open");
    });
}

function closeDrawer() {
    const drawer = document.getElementById("research-drawer");
    const backdrop = document.getElementById("research-backdrop");
    drawerOpen = false;
    drawer?.classList.remove("open");
    backdrop?.classList.remove("open");
    setTimeout(() => {
        if (!drawerOpen) {
            drawer?.setAttribute("hidden", "");
            backdrop?.setAttribute("hidden", "");
        }
    }, 300);
}

// ── Evaluation & Leaderboard ───────────────────────────────────────────────

function getScoreColor(score) {
    if (score >= 800) return "#1d7d39"; // green
    if (score >= 600) return "#0070f2"; // blue
    if (score >= 400) return "#e76500"; // orange
    return "#bb0000"; // red
}

// ── Score-criteria popover ("How this score is calculated") ────────────────
// Content mirrors the evaluator rubric. SOURCE OF TRUTH:
// api/app/services/evaluation.py (SYSTEM_PROMPT, ~lines 20-104). Keep in sync.
// Band chip tones follow getScoreColor()'s thresholds so the popover never
// contradicts the color of the score box.
const SCORE_DIMENSIONS = [
    { name: "Process specificity", desc: "Triggers, steps, data flows, outputs", weight: "25%" },
    { name: "Problem–solution fit", desc: "Does the AI approach fit the problem", weight: "20%" },
    { name: "Integration context", desc: "Named systems &amp; data sources", weight: "20%" },
    { name: "Decision &amp; edge cases", desc: "Branches, failure paths, routing logic", weight: "15%" },
    { name: "User roles &amp; personas", desc: "Who triggers, approves, benefits", weight: "10%" },
    { name: "Business context", desc: "Enough context to scope the work", weight: "10%" },
];
const SCORE_BANDS = [
    { range: "900–1000", label: "Joule-ready", tone: "positive" },
    { range: "800–899", label: "Complete intent", tone: "positive" },
    { range: "650–799", label: "Strong foundation", tone: "informative" },
    { range: "450–649", label: "Solid concept", tone: "critical" },
    { range: "250–449", label: "Underdeveloped", tone: "negative" },
    { range: "1–249", label: "Off-target", tone: "negative" },
];
const SCORE_CAPS = [
    { cond: "No failure paths", max: "max 780" },
    { cond: "Only one failure path", max: "max 830" },
    { cond: "No escalation lifecycle", max: "max 870" },
];

function buildCriteriaHtml() {
    const dims = SCORE_DIMENSIONS.map((d) => `
        <div class="crit-dim">
            <span class="crit-dim-name">${d.name}</span>
            <span class="crit-dim-desc">${d.desc}</span>
            <span class="crit-dim-weight">${d.weight}</span>
        </div>`).join("");
    const bands = SCORE_BANDS.map((b) => `
        <div class="crit-band">
            <span class="crit-band-chip crit-tone-${b.tone}"></span>
            <span class="crit-band-range">${b.range}</span>
            <span class="crit-band-label">${b.label}</span>
        </div>`).join("");
    const caps = SCORE_CAPS.map((c) => `
        <div class="crit-cap">
            <span class="crit-cap-cond">${c.cond}</span>
            <span class="crit-cap-arrow">→</span>
            <span class="crit-cap-max">${c.max}</span>
        </div>`).join("");
    return `
        <div class="crit-body">
            <p class="crit-intro">This measures how <strong>ready your proposal is for Joule to turn into a build</strong> — not whether the idea itself is good. A reviewer AI grades it from 1–1000 across six weighted dimensions.</p>
            <div class="crit-columns">
                <div class="crit-col">
                    <div class="crit-section-label">What's weighted</div>
                    <div class="crit-dims">${dims}</div>
                </div>
                <div class="crit-col">
                    <div class="crit-section-label">Score bands</div>
                    <div class="crit-bands">${bands}</div>
                    <div class="crit-section-label crit-section-label--spaced">What caps a score</div>
                    <div class="crit-caps">${caps}</div>
                    <p class="crit-median">A well-formed first draft usually lands <strong>680–720</strong>.</p>
                </div>
            </div>
        </div>`;
}

let _criteriaPopover = null;
let _criteriaCloseTimer = null;
let _criteriaPinned = false;

function ensureCriteriaPopover() {
    if (_criteriaPopover) return _criteriaPopover;
    const pop = document.createElement("ui5-popover");
    pop.setAttribute("placement", "Top");
    pop.setAttribute("header-text", "How this score is calculated");
    pop.setAttribute("accessible-name", "How this score is calculated");
    // Read-only panel: don't auto-focus content on open (avoids a focus ring on
    // the content region and keeps focus on the trigger button).
    pop.setAttribute("prevent-initial-focus", "");
    pop.className = "eval-criteria-popover";
    pop.innerHTML = buildCriteriaHtml();
    // Stay open while the pointer is inside; close shortly after it leaves.
    pop.addEventListener("mouseenter", cancelCriteriaClose);
    pop.addEventListener("mouseleave", () => { if (!_criteriaPinned) scheduleCriteriaClose(); });
    pop.addEventListener("close", () => { _criteriaPinned = false; });
    document.body.appendChild(pop);
    _criteriaPopover = pop;
    return pop;
}

function cancelCriteriaClose() {
    if (_criteriaCloseTimer) {
        clearTimeout(_criteriaCloseTimer);
        _criteriaCloseTimer = null;
    }
}

function scheduleCriteriaClose() {
    cancelCriteriaClose();
    _criteriaCloseTimer = setTimeout(() => {
        if (!_criteriaPinned && _criteriaPopover) _criteriaPopover.open = false;
    }, 200);
}

// Show the criteria popover anchored to `opener`. Pass { pin: true } for a
// click (stays open until dismissed); omit for hover (auto-closes on leave).
function openCriteria(opener, { pin = false } = {}) {
    const pop = ensureCriteriaPopover();
    cancelCriteriaClose();
    pop.opener = opener;
    pop.open = true;
    if (pin) _criteriaPinned = true;
}

function closeCriteriaPopover() {
    cancelCriteriaClose();
    _criteriaPinned = false;
    if (_criteriaPopover) _criteriaPopover.open = false;
}

const LOADING_MESSAGES = [
    "Evaluating your solution...",
    "Checking workflow completeness...",
    "Analyzing decision logic...",
    "Reviewing failure paths...",
    "Scoring against rubric...",
    "Crafting suggestions...",
    "Wrapping up...",
];

function startLoadingProgression(msgIndex) {
    let step = 0;
    const interval = setInterval(() => {
        step++;
        const evalState = evaluations.get(msgIndex);
        if (!evalState || !evalState.loading) {
            clearInterval(interval);
            activeLoadingTimers.delete(msgIndex);
            return;
        }
        evalState.loadingText = LOADING_MESSAGES[Math.min(step, LOADING_MESSAGES.length - 1)];
        // Re-render just the loading text if card exists
        const card = document.querySelector(`.evaluation-card[data-msg-index="${msgIndex}"] .eval-loading-text`);
        if (card) card.textContent = evalState.loadingText;
    }, 2500);
    activeLoadingTimers.set(msgIndex, interval);
}

function triggerAutoEvaluation(ui, msgIndex) {
    // Capture context at trigger time — if user switches conversation mid-flight,
    // the result must not pollute the new conversation's state.
    const triggerContextId = contextId;
    const triggerEventId = activeEventId;

    // Compute delta from previous evaluation if any
    let previousScore = null;
    for (const [idx, evalState] of evaluations) {
        if (idx < msgIndex && evalState.result?.success) {
            previousScore = evalState.result.score;
        }
    }

    evaluations.set(msgIndex, {
        loading: true,
        result: null,
        ui,
        isOnLeaderboard: false,
        delta: null,
        rank: null,
        animate: true,
        loadingText: LOADING_MESSAGES[0],
    });
    _renderMessages();
    startLoadingProgression(msgIndex);

    evaluateSolution({
        industry: ui.industry || "",
        title: ui.title || "",
        problem: ui.problem || "",
        solution: ui.solution || "",
        app_type: ui.app_type || "Agent",
        company_context: ui.companyContext || null,
        context_id: triggerContextId || null,
    }).then((result) => {
        // Clear loading timer
        const timer = activeLoadingTimers.get(msgIndex);
        if (timer) { clearInterval(timer); activeLoadingTimers.delete(msgIndex); }

        // If user switched conversation, discard the UI update but still submit
        if (contextId !== triggerContextId) {
            // Still save to leaderboard under the correct (original) context
            if (result.success) {
                submitSolution({
                    industry: ui.industry || "",
                    title: ui.title || "",
                    problem: ui.problem || "",
                    solution: ui.solution || "",
                    app_type: ui.app_type || "Agent",
                    company_context: ui.companyContext || null,
                    sources: ui.sources || null,
                    evaluator_feedback: result.feedback,
                    evaluator_detailed_analysis: result.detailedAnalysis,
                    evaluator_suggestions: result.suggestions,
                    appropriateness_level: result.appropriatenessLevel,
                    score: result.score,
                    rating: 0,
                    context_id: triggerContextId || null,
                    message_index: msgIndex,
                    ...(triggerEventId && { event_id: triggerEventId }),
                }).catch(() => {});
            }
            return;
        }

        const delta = previousScore !== null && result.success ? result.score - previousScore : null;

        // Mark all earlier cards as NOT on leaderboard (latest wins)
        if (result.success) {
            for (const [idx, es] of evaluations) {
                if (idx < msgIndex) es.isOnLeaderboard = false;
            }
        }

        evaluations.set(msgIndex, {
            loading: false,
            result,
            ui,
            isOnLeaderboard: result.success,
            delta,
            rank: null,
            animate: true,
        });

        _renderMessages();

        // Auto-submit to leaderboard (replaces previous for same context)
        if (result.success) {
            submitSolution({
                industry: ui.industry || "",
                title: ui.title || "",
                problem: ui.problem || "",
                solution: ui.solution || "",
                app_type: ui.app_type || "Agent",
                company_context: ui.companyContext || null,
                sources: ui.sources || null,
                evaluator_feedback: result.feedback,
                evaluator_detailed_analysis: result.detailedAnalysis,
                evaluator_suggestions: result.suggestions,
                appropriateness_level: result.appropriatenessLevel,
                score: result.score,
                rating: 0,
                context_id: triggerContextId || null,
                message_index: msgIndex,
                ...(triggerEventId && { event_id: triggerEventId }),
            }).then(() => {
                // If user switched away during submit, don't update UI
                if (contextId !== triggerContextId) return;

                // Fetch rank preview from the appropriate leaderboard
                if (triggerEventId) {
                    // Event mode: use event leaderboard
                    return import("../../services/events.js").then(({ getEventLeaderboard }) =>
                        getEventLeaderboard(triggerEventId, 200)
                    ).then((entries) => {
                        if (contextId !== triggerContextId) return;
                        if (entries?.length) {
                            const myEntry = entries.find((e) => e.context_id === triggerContextId);
                            if (myEntry) {
                                const evalState = evaluations.get(msgIndex);
                                if (evalState) {
                                    evalState.rank = myEntry.rank;
                                    evalState.rankTotal = entries.length;
                                    _renderMessages();
                                }
                            }
                        }
                    }).catch(() => {});
                } else if (hasCompanyEmail()) {
                    // Personal mode: use company leaderboard
                    return getCompanyLeaderboard(200).then((data) => {
                        if (contextId !== triggerContextId) return;
                        if (data?.entries?.length) {
                            const myEntry = data.entries.find((e) => e.contextId === triggerContextId);
                            if (myEntry) {
                                const evalState = evaluations.get(msgIndex);
                                if (evalState) {
                                    evalState.rank = myEntry.rank;
                                    _renderMessages();
                                }
                            }
                        }
                    }).catch(() => {});
                }
            }).catch((err) => console.error("Auto-submit to leaderboard failed:", err));

        }
    }).catch((err) => {
        console.error("Evaluation failed:", err);
        const timer = activeLoadingTimers.get(msgIndex);
        if (timer) { clearInterval(timer); activeLoadingTimers.delete(msgIndex); }

        // If user switched conversation, don't pollute the new context
        if (contextId !== triggerContextId) return;

        evaluations.set(msgIndex, {
            loading: false,
            result: { score: 0, feedback: "Evaluation unavailable", detailedAnalysis: "", suggestions: [], appropriatenessLevel: "fair", success: false, error: err.message },
            ui,
            isOnLeaderboard: false,
            delta: null,
            rank: null,
            animate: false,
        });
        _renderMessages();
    });
}

function handleUseThisScore(msgIndex) {
    const evalState = evaluations.get(msgIndex);
    if (!evalState?.result?.success || !evalState.ui) return;

    const ui = evalState.ui;
    const result = evalState.result;

    // Mark all as not on leaderboard, then mark this one
    for (const [, es] of evaluations) es.isOnLeaderboard = false;
    evalState.isOnLeaderboard = true;

    submitSolution({
        industry: ui.industry || "",
        title: ui.title || "",
        problem: ui.problem || "",
        solution: ui.solution || "",
        app_type: ui.app_type || "Agent",
        company_context: ui.companyContext || null,
        sources: ui.sources || null,
        evaluator_feedback: result.feedback,
        evaluator_detailed_analysis: result.detailedAnalysis,
        evaluator_suggestions: result.suggestions,
        appropriateness_level: result.appropriatenessLevel,
        score: result.score,
        rating: 0,
        context_id: contextId || null,
        message_index: msgIndex,
        ...(activeEventId && { event_id: activeEventId }),
    }).then(() => {
        _renderMessages();
    }).catch((err) => console.error("Override submit failed:", err));

    _renderMessages(); // optimistic UI update
}

/**
 * Load stored evaluations from HANA on conversation reload.
 * Uses the bulk /all endpoint to fetch ALL stored evals by message_index,
 * avoiding unnecessary LLM re-evaluation calls.
 */
function loadStoredEvaluations() {
    if (!contextId) return;
    const loadContextId = contextId; // capture to detect stale callbacks

    getStoredEvaluations(contextId).then(({ entries }) => {
        // If user switched conversation, discard
        if (contextId !== loadContextId) return;

        if (!entries?.length) {
            // No stored evals at all — trigger fresh evaluations for all final messages
            _evalMissingFinals();
            return;
        }

        // Build a set of stored message indexes for quick lookup
        const storedIndexes = new Set();
        let previousScore = null;

        for (const entry of entries) {
            let matchIndex = entry.messageIndex;

            // Legacy rows (null messageIndex) → attach to last final
            if (matchIndex == null) {
                for (let i = conversation.length - 1; i >= 0; i--) {
                    if (conversation[i].ui?.kind === "final") {
                        matchIndex = i;
                        break;
                    }
                }
            }
            if (matchIndex == null || matchIndex >= conversation.length) continue;

            storedIndexes.add(matchIndex);
            const delta = previousScore !== null ? entry.score - previousScore : null;
            previousScore = entry.score;

            evaluations.set(matchIndex, {
                loading: false,
                result: {
                    score: entry.score,
                    feedback: entry.feedback,
                    detailedAnalysis: entry.detailedAnalysis,
                    suggestions: entry.suggestions,
                    appropriatenessLevel: entry.appropriatenessLevel,
                    success: true,
                },
                ui: conversation[matchIndex].ui,
                isOnLeaderboard: entry.isOnLeaderboard ?? false,
                delta,
                rank: null,
                animate: false, // no animation on reload
            });
        }

        // Fallback for old data without IS_ON_LEADERBOARD: if none are marked,
        // mark the last loaded entry as on leaderboard.
        const anyMarked = [...evaluations.values()].some(e => e.isOnLeaderboard);
        if (!anyMarked) {
            const loadedIndexes = [...storedIndexes].sort((a, b) => a - b);
            const lastLoadedIdx = loadedIndexes.length ? loadedIndexes[loadedIndexes.length - 1] : null;
            if (lastLoadedIdx != null) {
                const lastEval = evaluations.get(lastLoadedIdx);
                if (lastEval) lastEval.isOnLeaderboard = true;
            }
        }

        _renderMessages();

        // Fetch rank for the active leaderboard entry
        const activeEntry = [...evaluations.entries()].find(([, es]) => es.isOnLeaderboard);
        if (activeEntry) {
            const [, activeEval] = activeEntry;
            if (activeEventId) {
                // Event mode: use event leaderboard
                import("../../services/events.js").then(({ getEventLeaderboard }) =>
                    getEventLeaderboard(activeEventId, 200)
                ).then((entries) => {
                    if (contextId !== loadContextId) return;
                    if (entries?.length) {
                        const myEntry = entries.find((e) => e.context_id === loadContextId);
                        if (myEntry) {
                            activeEval.rank = myEntry.rank;
                            activeEval.rankTotal = entries.length;
                        }
                    }
                    // Always re-render: if entry not found, spinner disappears via rank=-1
                    if (!activeEval.rank) activeEval.rank = -1;
                    _renderMessages();
                }).catch(() => {
                    if (contextId !== loadContextId) return;
                    activeEval.rank = -1;
                    _renderMessages();
                });
            } else if (hasCompanyEmail()) {
                // Personal mode: use company leaderboard
                getCompanyLeaderboard(200).then((data) => {
                    if (contextId !== loadContextId) return;
                    if (data?.entries?.length) {
                        // Match by contextId — the entry may be attributed to the owner
                        // or still under the editor's email depending on dedup
                        const myEntry = data.entries.find((e) => e.contextId === loadContextId);
                        if (myEntry) {
                            activeEval.rank = myEntry.rank;
                        }
                    }
                    if (!activeEval.rank) activeEval.rank = -1;
                    _renderMessages();
                }).catch(() => {
                    if (contextId !== loadContextId) return;
                    activeEval.rank = -1;
                    _renderMessages();
                });
            }
        }

        // Only evaluate finals that have NO stored eval (all stored ones are already in the Map)
        _evalMissingFinals();
    }).catch(() => {
        // Fallback: trigger fresh evals if bulk endpoint fails
        _evalMissingFinals();
    });
}

/**
 * Trigger evaluations for any kind:"final" messages that don't have an eval yet.
 */
function _evalMissingFinals() {
    // Don't trigger fresh evaluations for read-only users — show owner's existing ones
    if (_isReadOnly) return;
    for (let i = 0; i < conversation.length; i++) {
        const msg = conversation[i];
        if (msg.role === "assistant" && msg.ui?.kind === "final" && !evaluations.has(i)) {
            triggerAutoEvaluation(msg.ui, i);
        }
    }
}

/**
 * Shorten a suggestion to a concise chip label (~5 words).
 */
function shortenSuggestion(text) {
    const trimmed = text
        .replace(/^(Explore how|Consider (the|how)|How would you|Think about|What if)\s*/i, "")
        .replace(/[.?!]+$/, "");
    const words = trimmed.split(/\s+/);
    return words.slice(0, 5).join(" ") + (words.length > 5 ? "…" : "");
}

/**
 * Build an evaluation card DOM element for a given eval state.
 * Called inline from renderMessages() for each kind:"final" message.
 */
function buildEvaluationCard(evalState, msgIndex) {
    const card = document.createElement("div");
    card.className = "evaluation-card";
    card.setAttribute("data-msg-index", msgIndex);

    // Skip animation on reload (no-animate class)
    if (!evalState.animate) {
        card.classList.add("no-animate");
    }

    if (evalState.loading) {
        card.innerHTML = `
            <div class="eval-loading">
                <ui5-busy-indicator active size="S" delay="0"></ui5-busy-indicator>
                <span class="eval-loading-text">${evalState.loadingText || LOADING_MESSAGES[0]}</span>
            </div>
        `;
        return card;
    }

    const result = evalState.result;
    if (!result) return card;

    const scoreColor = getScoreColor(result.score);
    const analysisHtml = result.detailedAnalysis ? marked.parse(result.detailedAnalysis) : "";

    // Build suggestions callout with readable text (chips rendered outside card in renderMessages)
    let suggestionsHtml = "";
    if (result.suggestions?.length) {
        const textLines = result.suggestions.map(s => escapeHtml(stripInlineMarkdown(s))).join("<br>");
        suggestionsHtml = `
            <div class="eval-suggestions">
                <span class="eval-suggestions-label">To strengthen this use case:</span>
                <span class="eval-suggestions-text">${textLines}</span>
            </div>
        `;
    }

    // Delta badge — rendered OUTSIDE score div as a separate pill
    let deltaHtml = "";
    if (evalState.delta !== null && evalState.delta !== 0) {
        const cls = evalState.delta > 0 ? "positive" : "negative";
        const sign = evalState.delta > 0 ? "↑+" : "↓";
        deltaHtml = `<span class="eval-score-delta ${cls}">${sign}${Math.abs(evalState.delta)}</span>`;
    }

    // Footer: rank + leaderboard/override
    let footerHtml = "";
    const _isCorporate = hasCompanyEmail();
    const _isEvent = !!activeEventId;
    const _hasLeaderboard = _isEvent || _isCorporate; // event leaderboard always available; company only for corporate

    const rankText = _hasLeaderboard
        ? (evalState.rank && evalState.rank > 0
            ? `<span class="eval-rank">This places you <strong>#${evalState.rank}</strong>${_isEvent ? ` of ${evalState.rankTotal || "?"}` : " in your company"}</span>`
            : evalState.isOnLeaderboard && !evalState.rank
                ? `<ui5-busy-indicator active size="S" delay="0" class="eval-rank-loader"></ui5-busy-indicator>`
                : "")
        : "";
    const leaderboardBtn = _hasLeaderboard
        ? `<ui5-button class="eval-view-leaderboard" design="Transparent" icon="competitor">${_isEvent ? "Event Leaderboard" : "View Leaderboard"}</ui5-button>`
        : "";
    const useForLeaderboardBtn = _hasLeaderboard
        ? `<ui5-button class="eval-use-for-leaderboard" design="Transparent" data-action="use-this-score">Use for leaderboard</ui5-button>`
        : "";

    if (evalState.isOnLeaderboard) {
        footerHtml = `<div class="eval-footer">
            ${rankText}
            <div class="eval-footer-actions">
                ${leaderboardBtn}
            </div>
        </div>`;
    } else if (result.success && evaluations.size > 1) {
        // Older eval that is no longer the leaderboard entry — offer override
        footerHtml = `<div class="eval-footer">
            ${_hasLeaderboard ? `<span class="eval-not-on-leaderboard">Not on leaderboard</span>` : ""}
            <div class="eval-footer-actions">
                ${useForLeaderboardBtn}
            </div>
        </div>`;
    } else if (result.success) {
        // Single eval, not yet confirmed on leaderboard — still show View Leaderboard
        footerHtml = `<div class="eval-footer">
            ${rankText}
            <div class="eval-footer-actions">
                ${leaderboardBtn}
            </div>
        </div>`;
    }

    // Strategic reframe section (injected when result is available)
    let reframeHtml = "";
    if (reframeResult && reframeResult.status === "ok") {
        reframeHtml = `
            <div class="eval-reframe">
                <div class="eval-reframe-header">
                    <span class="eval-reframe-label">Strategic perspective:</span>
                    <ui5-button class="eval-reframe-regenerate" design="Transparent" icon="refresh" tooltip="Generate a new perspective"></ui5-button>
                </div>
                <div class="eval-reframe-sections">
                    <div class="eval-reframe-item">
                        <span class="eval-reframe-item-header"><ui5-icon name="initiative"></ui5-icon> Challenge</span>
                        <span class="eval-reframe-item-text">${escapeHtml(reframeResult.challenge)}</span>
                    </div>
                    <div class="eval-reframe-item">
                        <span class="eval-reframe-item-header"><ui5-icon name="trend-up"></ui5-icon> Alternative Angle</span>
                        <span class="eval-reframe-item-text">${escapeHtml(reframeResult.alternative_angle)}</span>
                    </div>
                    <div class="eval-reframe-item">
                        <span class="eval-reframe-item-header"><ui5-icon name="alert"></ui5-icon> Key Risk</span>
                        <span class="eval-reframe-item-text">${escapeHtml(reframeResult.risk)}</span>
                    </div>
                </div>
                <span class="eval-reframe-summary">${escapeHtml(reframeResult.reframe_summary)}</span>
            </div>
        `;
    } else if (reframeStatus === "pending") {
        reframeHtml = `
            <div class="eval-reframe-skeleton">
                <ui5-busy-indicator active size="S" delay="0"></ui5-busy-indicator>
                <span style="font-size: 11px; color: var(--sapContent_LabelColor);">Generating strategic perspective…</span>
            </div>
        `;
    }

    // SAP AI Catalog matches section
    let catalogMatchHtml = "";
    if (catalogMatchResult && catalogMatchResult.status === "ok" && catalogMatchResult.matches?.length) {
        const matchItems = catalogMatchResult.matches.map(m => {
            return `
                <div class="catalog-match-item">
                    <div class="catalog-match-header">
                        <a href="${escapeHtml(m.detail_page_url)}" target="_blank" rel="noopener" class="catalog-match-name">${escapeHtml(m.name)}</a>
                        <span class="catalog-match-product">${escapeHtml(m.product)}</span>
                    </div>
                    <span class="catalog-match-description">${escapeHtml(m.description)}</span>
                </div>
            `;
        }).join("");

        catalogMatchHtml = `
            <div class="catalog-matches">
                <div class="catalog-matches-header">
                    <span class="catalog-matches-label">SAP Standard AI Solutions</span>
                    <span class="catalog-matches-subtitle">Existing solutions that may cover this scenario</span>
                    <span class="catalog-matches-header-right">
                        <a href="https://discovery-center.cloud.sap/ai-catalog/" target="_blank" rel="noopener" class="catalog-matches-link">SAP Discovery Center</a>
                        <ui5-button class="catalog-matches-regenerate" design="Transparent" icon="refresh" tooltip="Re-generate matches"></ui5-button>
                    </span>
                </div>
                <div class="catalog-matches-list">
                    ${matchItems}
                </div>
            </div>
        `;
    } else if (catalogMatchStatus === "pending") {
        catalogMatchHtml = `
            <div class="catalog-matches-skeleton">
                <ui5-busy-indicator active size="S" delay="0"></ui5-busy-indicator>
                <span style="font-size: 11px; color: var(--sapContent_LabelColor);">Finding matching SAP AI solutions…</span>
            </div>
        `;
    }

    card.innerHTML = `
        <div class="eval-header">
            <div class="eval-score" style="background: ${scoreColor};"><span class="eval-score-num">${evalState.animate ? "0" : result.score}</span></div>
            ${deltaHtml}
            <div class="eval-headline">
                <span class="eval-feedback">${escapeHtml(result.feedback)}</span>
                <span class="eval-level">${result.appropriatenessLevel}</span>
            </div>
            <ui5-button class="eval-info-btn" design="Transparent" icon="information">How it's scored</ui5-button>
        </div>
        <div class="eval-analysis">${analysisHtml}</div>
        ${suggestionsHtml}
        ${reframeHtml}
        ${catalogMatchHtml}
        ${footerHtml}
    `;

    // Wire up actions
    setTimeout(() => {
        const viewBtn = card.querySelector(".eval-view-leaderboard");
        if (viewBtn) viewBtn.addEventListener("click", () => {
            if (activeEventCode) {
                // Event mode: navigate to event page (leaderboard tab)
                window.location.href = `/event/${activeEventCode}`;
            } else {
                openLeaderboardDialog({ contextId, score: result.score });
            }
        });

        const overrideLink = card.querySelector('[data-action="use-this-score"]');
        if (overrideLink) overrideLink.addEventListener("click", () => handleUseThisScore(msgIndex));

        // "How this score is calculated" — hover to preview, click to pin open
        const infoBtn = card.querySelector(".eval-info-btn");
        if (infoBtn) {
            infoBtn.addEventListener("mouseenter", () => { if (!_criteriaPinned) openCriteria(infoBtn); });
            infoBtn.addEventListener("mouseleave", () => { if (!_criteriaPinned) scheduleCriteriaClose(); });
            infoBtn.addEventListener("click", () => {
                const pop = _criteriaPopover;
                if (pop && pop.open && _criteriaPinned && pop.opener === infoBtn) {
                    pop.open = false; // toggle closed
                } else {
                    openCriteria(infoBtn, { pin: true });
                }
            });
        }

        // Reframe regenerate button
        const regenBtn = card.querySelector(".eval-reframe-regenerate");
        if (regenBtn) regenBtn.addEventListener("click", async () => {
            regenBtn.setAttribute("disabled", "");
            reframeStatus = "pending";
            reframeResult = null;
            _renderMessages(); // show skeleton
            try {
                await request(`/api/ideation-agent/reframe/${contextId}/regenerate`, "POST");
                startReframePoll();
            } catch (e) {
                console.warn("Reframe regenerate error:", e);
                reframeStatus = "failed";
                _renderMessages();
            }
        });

        // Catalog matches regenerate button
        const catalogRegenBtn = card.querySelector(".catalog-matches-regenerate");
        if (catalogRegenBtn) catalogRegenBtn.addEventListener("click", async () => {
            catalogRegenBtn.setAttribute("disabled", "");
            catalogMatchStatus = "pending";
            catalogMatchResult = null;
            _renderMessages();
            try {
                await request(`/api/ideation-agent/catalog-matches/${contextId}/regenerate`, "POST");
                startCatalogMatchPoll();
            } catch (e) {
                console.warn("Catalog match regenerate error:", e);
                catalogMatchStatus = "failed";
                _renderMessages();
            }
        });

        // Score count-up animation
        if (evalState.animate && result.score > 0) {
            const scoreEl = card.querySelector(".eval-score-num");
            if (scoreEl) {
                const target = result.score;
                const duration = 800;
                const start = performance.now();
                function animateScore(now) {
                    const elapsed = now - start;
                    const progress = Math.min(elapsed / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
                    scoreEl.textContent = Math.round(target * eased);
                    if (progress < 1) requestAnimationFrame(animateScore);
                }
                requestAnimationFrame(animateScore);
                evalState.animate = false; // only animate once
            }
        }
    }, 0);

    return card;
}

/**
 * Build evaluation context string for injection into API messages.
 */
function buildEvalContext(result) {
    const suggestionsList = result.suggestions
        .map((s, i) => `${i + 1}. ${s}`)
        .join("\n");
    return `[Previous evaluation scored ${result.score}/1000. Feedback: "${result.feedback}"]\n\nSuggested improvements:\n${suggestionsList}`;
}

// ── End Evaluation & Leaderboard ───────────────────────────────────────────

// Restore the research drawer to its pristine, empty state. Must run whenever
// the active conversation changes: the CSE tab renders append-only into the
// static panels below (each block guarded by `!panel.querySelector(...)`), and
// the Perplexity / LinkedIn tabs early-return while their status is "idle" — so
// none of the render functions clear stale DOM on their own. Without this, the
// previous conversation's research stays on screen, and the CSE guards even
// block the newly selected conversation's data from ever rendering.
function resetResearchDrawer() {
    // CSE tab: empty and re-hide the static panels; drop any "not found" card.
    for (const id of [
        "research-panel-company",
        "research-panel-analysis",
        "research-panel-stories",
        "research-panel-sources",
    ]) {
        const panel = document.getElementById(id);
        if (panel) {
            panel.innerHTML = "";
            panel.setAttribute("hidden", "");
        }
    }
    document.getElementById("research-cse-content")
        ?.querySelector(".research-cse-not-found")?.remove();
    // CSE status row (spinner + "Searching...") back to hidden.
    document.getElementById("research-status")?.setAttribute("hidden", "");
    document.getElementById("research-busy")?.removeAttribute("active");

    // Perplexity + LinkedIn tabs: restore the original placeholders and clear
    // the pending "●" indicator from the tab headers.
    const pxTab = document.getElementById("research-tab-perplexity");
    if (pxTab) {
        pxTab.removeAttribute("additional-text");
        pxTab.innerHTML = `
            <div class="research-tab-placeholder">
                <ui5-icon name="ai" class="placeholder-icon"></ui5-icon>
                <ui5-text class="placeholder-text">Market intelligence will appear here.</ui5-text>
            </div>`;
    }
    const liTab = document.getElementById("research-tab-linkedin");
    if (liTab) {
        liTab.removeAttribute("additional-text");
        liTab.innerHTML = `
            <div class="research-tab-placeholder">
                <ui5-icon name="people-connected" class="placeholder-icon"></ui5-icon>
                <ui5-text class="placeholder-text">LinkedIn insights coming soon.</ui5-text>
            </div>`;
    }
}

function renderDrawerContent() {
    const toggle = document.getElementById("research-toggle");
    const statusEl = document.getElementById("research-status");
    const busyEl = document.getElementById("research-busy");
    const companyPanel = document.getElementById("research-panel-company");
    const analysisPanel = document.getElementById("research-panel-analysis");
    const storiesPanel = document.getElementById("research-panel-stories");
    const sourcesPanel = document.getElementById("research-panel-sources");

    if (enrichmentStatus === "idle" && perplexityStatus === "idle") {
        if (toggle) toggle.style.display = "none";
        return;
    }

    // For not_found or failed CSE: hide CSE status, but keep toggle if other research is active
    if (
        (enrichmentStatus === "complete" && enrichmentResult?.status === "not_found")
        || enrichmentStatus === "failed"
    ) {
        if (busyEl) busyEl.removeAttribute("active");
        statusEl?.setAttribute("hidden", "");
        const otherActive = perplexityStatus === "pending" || perplexityProblemStatus === "pending"
            || linkedinStatus === "pending" || perplexityStatus === "complete" || linkedinStatus === "complete";
        if (!otherActive) {
            if (toggle) toggle.style.display = "none";
            toggle?.classList.remove("pulse");
            if (drawerOpen) closeDrawer();
            return;
        }
    }

    if (toggle) toggle.style.display = "";

    const anyPending = enrichmentStatus === "pending" || perplexityStatus === "pending"
        || perplexityProblemStatus === "pending" || linkedinStatus === "pending";
    if (anyPending) {
        toggle?.classList.add("pulse");
    } else {
        toggle?.classList.remove("pulse");
    }

    if (enrichmentStatus === "pending") {
        statusEl?.removeAttribute("hidden");
        if (busyEl) busyEl.setAttribute("active", "");
    } else {
        if (busyEl) busyEl.removeAttribute("active");
        statusEl?.setAttribute("hidden", "");
    }

    if (!enrichmentResult || enrichmentResult.status !== "ok") {
        // Show illustrated empty state when CSE search completed but company not found
        const cseContent = document.getElementById("research-cse-content");
        if (
            enrichmentStatus === "complete"
            && enrichmentResult?.status === "not_found"
            && cseContent
            && !cseContent.querySelector(".research-cse-not-found")
        ) {
            const msg = document.createElement("ui5-illustrated-message");
            msg.setAttribute("name", "NoSearchResults");
            msg.setAttribute("design", "Spot");
            msg.setAttribute("title-text", "Company not found");
            msg.setAttribute("subtitle-text",
                enrichmentResult.message
                || "Company profile not available. CSE covers publicly traded companies only.");
            msg.className = "research-cse-not-found";
            cseContent.appendChild(msg);
        }
        return;
    }

    const company = enrichmentResult.company;
    if (company && companyPanel && !companyPanel.querySelector(".research-company-card")) {
        companyPanel.removeAttribute("hidden");
        const card = document.createElement("div");
        card.className = "research-company-card";
        const ticker = company.ticker ? `${escapeHtml(company.ticker)}` : "";
        const exchange = company.exchange ? ` · ${escapeHtml(company.exchange)}` : "";
        card.innerHTML = `
            <div class="company-name">${escapeHtml(company.name || "")}</div>
            <div class="company-meta">${ticker}${exchange}</div>
        `;
        companyPanel.appendChild(card);
    }

    if (enrichmentResult.analysis_markdown && analysisPanel && !analysisPanel.querySelector(".research-panel-body")) {
        analysisPanel.removeAttribute("hidden");
        const body = document.createElement("div");
        body.className = "research-panel-body";
        body.innerHTML = marked.parse(enrichmentResult.analysis_markdown);
        analysisPanel.appendChild(body);
    }

    if (enrichmentResult.stories_markdown && storiesPanel && !storiesPanel.querySelector(".research-panel-body")) {
        storiesPanel.removeAttribute("hidden");
        const body = document.createElement("div");
        body.className = "research-panel-body";
        body.innerHTML = marked.parse(enrichmentResult.stories_markdown);
        storiesPanel.appendChild(body);
    }

    const links = [...(enrichmentResult.story_links || [])];
    if (enrichmentResult.report_link) links.unshift(enrichmentResult.report_link);
    if (links.length > 0 && sourcesPanel && !sourcesPanel.querySelector(".research-sources-list")) {
        sourcesPanel.removeAttribute("hidden");
        const list = document.createElement("div");
        list.className = "research-sources-list";
        for (const link of links) {
            const el = document.createElement("ui5-link");
            el.setAttribute("href", link);
            el.setAttribute("target", "_blank");
            el.textContent = link;
            list.appendChild(el);
        }
        sourcesPanel.appendChild(list);
    }
}

function renderSourcesList(sourceDetails) {
    if (!Array.isArray(sourceDetails) || sourceDetails.length === 0) return "";
    const items = sourceDetails.map(s => {
        const url = s.url || "";
        const title = s.title || url;
        return `<li><ui5-link href="${url}" target="_blank">${escapeHtml(title)}</ui5-link></li>`;
    }).join("");
    return `<div class="research-sources"><strong>Sources</strong><ul>${items}</ul></div>`;
}

function renderPerplexityTab() {
    const tab = document.getElementById("research-tab-perplexity");
    if (!tab) return;

    if (perplexityStatus === "idle" && perplexityProblemStatus === "idle") return;

    const hasCompanyData = perplexityResult && perplexityResult.status === "ok";
    const hasProblemData = perplexityProblemResult && perplexityProblemResult.status === "ok";

    const container = document.createElement("div");
    container.className = "perplexity-content research-tab-content";

    // Company Intelligence panel — always present once research has started
    const companyPanel = document.createElement("ui5-panel");
    companyPanel.setAttribute("header-text", "Company Intelligence");
    companyPanel.className = "research-panel";
    const companyBody = document.createElement("div");
    companyBody.className = "research-panel-body";

    if (hasCompanyData) {
        const companyMd = perplexityResult.company_markdown || perplexityResult.summary_markdown || "";
        if (companyMd) {
            companyBody.innerHTML = marked.parse(companyMd) + renderSourcesList(perplexityResult.source_details);
        } else {
            companyBody.innerHTML = `<ui5-text>No company data returned.</ui5-text>`;
        }
    } else if (perplexityStatus === "pending") {
        companyBody.className = "research-panel-body research-status";
        companyBody.innerHTML = `<ui5-busy-indicator active size="S" delay="0"></ui5-busy-indicator>
            <ui5-text class="research-status-text">Researching company intelligence...</ui5-text>`;
    } else {
        companyBody.innerHTML = `<ui5-text>Company research unavailable.</ui5-text>`;
    }
    companyPanel.appendChild(companyBody);
    container.appendChild(companyPanel);

    // Domain Research panel — always visible, shows state
    const problemPanel = document.createElement("ui5-panel");
    problemPanel.setAttribute("header-text", "Domain Research");
    problemPanel.className = "research-panel";
    const problemBody = document.createElement("div");
    problemBody.className = "research-panel-body";

    if (hasProblemData) {
        const problemMd = perplexityProblemResult.problem_markdown || "";
        if (problemMd) {
            problemBody.innerHTML = marked.parse(problemMd) + renderSourcesList(perplexityProblemResult.source_details);
        } else {
            problemBody.innerHTML = `<ui5-text>No domain data returned.</ui5-text>`;
        }
    } else if (perplexityProblemStatus === "pending") {
        problemBody.className = "research-panel-body research-status";
        problemBody.innerHTML = `<ui5-busy-indicator active size="S" delay="0"></ui5-busy-indicator>
            <ui5-text class="research-status-text">Researching problem domain...</ui5-text>`;
    } else if (perplexityProblemStatus === "idle") {
        problemBody.className = "research-panel-body research-status";
        if (conversationComplete) {
            problemBody.innerHTML = `<ui5-text class="research-status-text">Domain research was not triggered for this conversation.</ui5-text>`;
        } else {
            problemBody.innerHTML = `<ui5-text class="research-status-text">Waiting for more context to start domain research...</ui5-text>`;
        }
    } else {
        problemBody.innerHTML = `<ui5-text>Domain research unavailable.</ui5-text>`;
    }
    problemPanel.appendChild(problemBody);
    container.appendChild(problemPanel);

    tab.innerHTML = "";
    tab.appendChild(container);
}

async function pollEnrichment() {
    if (!contextId) return;
    const pollingContextId = contextId;
    try {
        const data = await request(`${ENRICHMENT_ENDPOINT}/${contextId}`);
        if (!data) return; // Auth failed or empty response — skip this poll cycle
        if (contextId !== pollingContextId) return;
        enrichmentStatus = data.status || "idle";
        if (data.result) enrichmentResult = data.result;

        // Perplexity enrichment
        if (data.perplexity) {
            perplexityStatus = data.perplexity.status || "idle";
            if (data.perplexity.result) perplexityResult = data.perplexity.result;
            perplexityProblemStatus = data.perplexity.problem_status || "idle";
            if (data.perplexity.problem_result) perplexityProblemResult = data.perplexity.problem_result;
        }

        // LinkedIn enrichment
        if (data.linkedin) {
            linkedinStatus = data.linkedin.status || "idle";
            if (data.linkedin.result) linkedinResult = data.linkedin.result;
        }

        renderDrawerContent();
        renderPerplexityTab();
        renderLinkedInTab();

        // Show loading indicator on Perplexity tab even when CSE tab is active
        const pxTab = document.getElementById("research-tab-perplexity");
        if (pxTab) {
            const anyPxPending = perplexityStatus === "pending" || perplexityProblemStatus === "pending";
            if (anyPxPending) {
                pxTab.setAttribute("additional-text", "●");
            } else {
                pxTab.removeAttribute("additional-text");
            }
        }

        // Show loading indicator on LinkedIn tab
        const liTab = document.getElementById("research-tab-linkedin");
        if (liTab) {
            if (linkedinStatus === "pending") {
                liTab.setAttribute("additional-text", "●");
            } else {
                liTab.removeAttribute("additional-text");
            }
        }

        const cseTerminal = enrichmentStatus === "complete" || enrichmentStatus === "failed" || enrichmentStatus === "disabled";
        const pxTerminal = perplexityStatus === "complete" || perplexityStatus === "failed" || perplexityStatus === "idle";
        const pxProblemDone = perplexityProblemStatus === "complete" || perplexityProblemStatus === "failed";
        const pxProblemTerminal = pxProblemDone || (perplexityProblemStatus === "idle" && cseTerminal && pxTerminal);
        const liTerminal = linkedinStatus === "complete" || linkedinStatus === "failed" || linkedinStatus === "idle";
        if (cseTerminal && pxTerminal && pxProblemTerminal && liTerminal) {
            stopEnrichmentPolling();
            const cseOk = enrichmentStatus === "complete" && enrichmentResult?.status === "ok";
            const pxOk = perplexityStatus === "complete" && perplexityResult?.status === "ok";
            if ((cseOk || pxOk) && !drawerAutoOpened) {
                drawerAutoOpened = true;
                showResearchBadge();
            }
        }
    } catch (err) {
        console.error("[Research] poll error:", err);
    }
}

function renderLinkedInTab() {
    const tab = document.getElementById("research-tab-linkedin");
    if (!tab) return;
    if (linkedinStatus === "idle") return;

    const container = document.createElement("div");
    container.className = "linkedin-content research-tab-content";

    if (linkedinResult && linkedinResult.status === "ok") {
        const techs = linkedinResult.top_technologies || [];
        if (techs.length > 0) {
            const techPanel = document.createElement("ui5-panel");
            techPanel.setAttribute("header-text", "Technology Signals");
            techPanel.className = "research-panel";
            const techBody = document.createElement("div");
            techBody.className = "research-panel-body";
            techBody.innerHTML = "<ul>" + techs.map(t =>
                `<li><strong>${escapeHtml(t.name)}</strong> (${t.count} mentions)${t.insight ? `<br><span style="color:var(--sapContent_LabelColor);font-size:13px">${escapeHtml(t.insight)}</span>` : ""}</li>`
            ).join("") + "</ul>";
            techPanel.appendChild(techBody);
            container.appendChild(techPanel);
        }

        const ai = linkedinResult.ai_signals || {};
        if (ai.ai_maturity_level) {
            const aiPanel = document.createElement("ui5-panel");
            aiPanel.setAttribute("header-text", "AI Maturity");
            aiPanel.className = "research-panel";
            const aiBody = document.createElement("div");
            aiBody.className = "research-panel-body";
            aiBody.innerHTML = `<p><strong>Level:</strong> ${escapeHtml(ai.ai_maturity_level)}</p>
                <p>${escapeHtml(ai.ai_maturity_reasoning || "")}</p>`;
            aiPanel.appendChild(aiBody);
            container.appendChild(aiPanel);
        }

        const roles = linkedinResult.top_business_vacancies || [];
        if (roles.length > 0) {
            const rolesPanel = document.createElement("ui5-panel");
            rolesPanel.setAttribute("header-text", "Active Hiring Areas");
            rolesPanel.className = "research-panel";
            const rolesBody = document.createElement("div");
            rolesBody.className = "research-panel-body";
            rolesBody.innerHTML = "<ul>" + roles.map(r =>
                `<li><strong>${escapeHtml(r.role_family)}</strong> (${r.count} positions)${r.insight ? `<br><span style="color:var(--sapContent_LabelColor);font-size:13px">${escapeHtml(r.insight)}</span>` : ""}</li>`
            ).join("") + "</ul>";
            rolesPanel.appendChild(rolesBody);
            container.appendChild(rolesPanel);
        }

        const insights = linkedinResult.ideation_insights || {};
        const hasInsights = (insights.bottlenecks?.length || 0) + (insights.automation_opportunities?.length || 0) > 0;
        if (hasInsights) {
            const insightsPanel = document.createElement("ui5-panel");
            insightsPanel.setAttribute("header-text", "Ideation Signals");
            insightsPanel.className = "research-panel";
            const insightsBody = document.createElement("div");
            insightsBody.className = "research-panel-body";
            let insightsHtml = "";
            if (insights.bottlenecks?.length) {
                insightsHtml += `<h3>Bottlenecks</h3><ul>${insights.bottlenecks.map(b => `<li>${escapeHtml(b)}</li>`).join("")}</ul>`;
            }
            if (insights.automation_opportunities?.length) {
                insightsHtml += `<h3>Automation Opportunities</h3><ul>${insights.automation_opportunities.map(a => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
            }
            insightsBody.innerHTML = insightsHtml;
            insightsPanel.appendChild(insightsBody);
            container.appendChild(insightsPanel);
        }

        const metaDiv = document.createElement("div");
        metaDiv.className = "linkedin-meta";
        metaDiv.innerHTML = `<ui5-text style="font-size:12px;color:var(--sapContent_LabelColor)">
            ${linkedinResult.rows_analyzed || 0} positions analyzed · ${escapeHtml(linkedinResult.time_window || "")} · Confidence: ${((linkedinResult.confidence || 0) * 100).toFixed(0)}%
        </ui5-text>`;
        container.appendChild(metaDiv);

        const links = linkedinResult.source_links || [];
        if (links.length > 0) {
            const srcPanel = document.createElement("ui5-panel");
            srcPanel.setAttribute("header-text", "Sources");
            srcPanel.setAttribute("collapsed", "");
            srcPanel.className = "research-panel";
            const srcBody = document.createElement("div");
            srcBody.className = "research-panel-body";
            srcBody.innerHTML = "<ul>" + links.map(url =>
                `<li><ui5-link href="${url}" target="_blank">${escapeHtml(url)}</ui5-link></li>`
            ).join("") + "</ul>";
            srcPanel.appendChild(srcBody);
            container.appendChild(srcPanel);
        }
    } else if (linkedinStatus === "pending") {
        container.innerHTML = `<div class="research-tab-placeholder">
            <ui5-busy-indicator active size="S" delay="0"></ui5-busy-indicator>
            <ui5-text class="research-status-text">Analyzing LinkedIn positions...</ui5-text>
        </div>`;
    } else if (linkedinResult && linkedinResult.status === "no_data") {
        container.innerHTML = `<div class="research-tab-placeholder">
            <ui5-text class="placeholder-text">No LinkedIn positions found for this company.</ui5-text>
        </div>`;
    } else if (linkedinStatus === "failed") {
        container.innerHTML = `<div class="research-tab-placeholder">
            <ui5-text class="placeholder-text">LinkedIn analysis unavailable.</ui5-text>
        </div>`;
    }

    tab.innerHTML = "";
    tab.appendChild(container);
}

function startEnrichmentPolling() {
    if (enrichmentPollTimer) return;
    enrichmentPollTimer = setInterval(pollEnrichment, POLL_INTERVAL_MS);
    pollEnrichment();
}

function stopEnrichmentPolling() {
    if (enrichmentPollTimer) {
        clearInterval(enrichmentPollTimer);
        enrichmentPollTimer = null;
    }
}

// ── Strategic Reframe Polling ─────────────────────────────────────────────
function startReframePoll() {
    if (reframePollTimer) return;
    reframeStatus = "pending";
    reframePollTimer = setInterval(pollReframe, 2000);
    pollReframe();
}

function stopReframePoll() {
    if (reframePollTimer) {
        clearInterval(reframePollTimer);
        reframePollTimer = null;
    }
}

async function pollReframe() {
    if (!contextId) return;
    const pollingContextId = contextId;
    try {
        const data = await request(`/api/ideation-agent/reframe/${contextId}`);
        if (!data || contextId !== pollingContextId) return;
        reframeStatus = data.status || "idle";
        if (data.result) reframeResult = data.result;

        if (reframeStatus === "complete" || reframeStatus === "failed") {
            stopReframePoll();
            _renderMessages(); // re-render to show the reframe section in eval card
        } else if (reframeStatus === "pending" && !reframePollTimer) {
            // Auto-triggered: endpoint started the task, begin polling for result
            reframePollTimer = setInterval(pollReframe, 2000);
            _renderMessages(); // show skeleton while waiting
        }
    } catch (e) {
        console.warn("Reframe poll error:", e);
    }
}

// ── SAP AI Catalog match polling ────────────────────────────────────────────
function startCatalogMatchPoll() {
    if (catalogMatchPollTimer) return;
    catalogMatchStatus = "pending";
    catalogMatchPollTimer = setInterval(pollCatalogMatches, 2500);
    pollCatalogMatches();
}

function stopCatalogMatchPoll() {
    if (catalogMatchPollTimer) {
        clearInterval(catalogMatchPollTimer);
        catalogMatchPollTimer = null;
    }
}

async function pollCatalogMatches() {
    if (!contextId) return;
    const pollingContextId = contextId;
    try {
        const data = await request(`/api/ideation-agent/catalog-matches/${contextId}`);
        if (!data || contextId !== pollingContextId) return;
        catalogMatchStatus = data.status || "idle";
        if (data.result) catalogMatchResult = data.result;

        if (catalogMatchStatus === "complete" || catalogMatchStatus === "failed") {
            stopCatalogMatchPoll();
            _renderMessages();
        } else if (catalogMatchStatus === "pending" && !catalogMatchPollTimer) {
            catalogMatchPollTimer = setInterval(pollCatalogMatches, 2500);
            _renderMessages();
        }
    } catch (e) {
        console.warn("Catalog match poll error:", e);
    }
}

export default function initIdeationAgentPage() {
    dayjs.extend(calendar);

    // ── Event mode detection ────────────────────────────────────────────
    const urlParams = new URLSearchParams(window.location.search);
    const eventId = urlParams.get("event_id") || null;
    const eventCode = urlParams.get("event_code") || null;
    const isEventMode = !!eventId;

    // Hide CSE tab for external (CDC) users — CSE is SAP-internal only
    const userInfo = getUserInfo();
    const isExternalUser = userInfo?.authProvider === "cdc";
    if (isExternalUser) {
        const cseTab = document.getElementById("research-tab-cse");
        if (cseTab) cseTab.remove();
        // Select Perplexity tab as default instead
        const perplexityTab = document.getElementById("research-tab-perplexity");
        if (perplexityTab) perplexityTab.setAttribute("selected", "");
    }

    const sendButton = document.getElementById("send-button");
    const chatTimeline = document.getElementById("chat-timeline");
    const chatSplash = document.getElementById("chat-splash");
    const input = document.getElementById("message-input");
    const composerNote = document.getElementById("composer-note");
    const companyInput = document.getElementById("company-input");

    let isSending = false;
    let _sendAbort = null;       // AbortController for the in-flight send (Stop button)
    let _stopRequested = false;  // set when the user pressed Stop for the current turn
    let _editingIndex = null;    // conversation index currently being edited inline, or null
    let _editingDraft = null;    // in-progress edit text, preserved across re-renders

    // File attachment bindings
    const attachButton = document.getElementById("attach-button");
    const fileInput = document.getElementById("file-input");
    const dragOverlay = document.getElementById("drag-overlay");

    if (attachButton && fileInput) {
        attachButton.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (e) => {
            _addPendingFiles(Array.from(e.target.files));
            fileInput.value = "";
        });
    }

    // Drag & drop on chat container
    const chatContainer = chatTimeline?.closest(".chat-container");
    if (chatContainer) {
        let dragCounter = 0;
        chatContainer.addEventListener("dragenter", (e) => {
            e.preventDefault();
            dragCounter++;
            if (dragOverlay) dragOverlay.hidden = false;
        });
        chatContainer.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                if (dragOverlay) dragOverlay.hidden = true;
            }
        });
        chatContainer.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
        });
        chatContainer.addEventListener("drop", (e) => {
            e.preventDefault();
            dragCounter = 0;
            if (dragOverlay) dragOverlay.hidden = true;
            const files = Array.from(e.dataTransfer.files).filter(f => _isAcceptedFileType(f));
            if (files.length > 0) _addPendingFiles(files);
        });
    }

    // Lightbox close handlers
    const lightbox = document.getElementById("image-lightbox");
    const lightboxClose = document.getElementById("lightbox-close");
    if (lightbox) {
        lightbox.addEventListener("click", (e) => {
            if (e.target === lightbox) _closeLightbox();
        });
    }
    if (lightboxClose) {
        lightboxClose.addEventListener("click", _closeLightbox);
    }
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") _closeLightbox();
    });

    // Research drawer controls
    const researchToggle = document.getElementById("research-toggle");
    const researchBackdrop = document.getElementById("research-backdrop");
    const researchClose = document.getElementById("research-close");

    researchToggle?.addEventListener("click", () => { drawerOpen ? closeDrawer() : openDrawer(); });
    researchClose?.addEventListener("click", closeDrawer);
    researchBackdrop?.addEventListener("click", closeDrawer);

    // Company autocomplete
    if (companyInput) {
        let lastSelectedValue = "";

        const fetchSuggestions = debounce(async (query) => {
            if (!query || query.length < 2 || query === lastSelectedValue) {
                if (!query || query.length < 2) companyInput.innerHTML = "";
                return;
            }
            try {
                const data = await request(`${SUGGESTIONS_ENDPOINT}?q=${encodeURIComponent(query)}`);
                const suggestions = data?.suggestions || [];
                companyInput.innerHTML = "";
                for (const item of suggestions) {
                    const el = document.createElement("ui5-suggestion-item");
                    el.setAttribute("text", item.name || "");
                    const meta = [item.ticker, item.exchange].filter(Boolean).join(" · ");
                    if (meta) {
                        el.setAttribute("additional-text", meta);
                    }
                    companyInput.appendChild(el);
                }
            } catch (err) {
                console.error("[Company] suggestion fetch error:", err);
            }
        }, 300);

        companyInput.addEventListener("input", (e) => {
            fetchSuggestions(e.target.value?.trim() || "");
        });

        companyInput.addEventListener("selection-change", (e) => {
            const selected = e.detail?.item;
            if (selected) {
                lastSelectedValue = selected.getAttribute("text") || "";
                companyInput.value = lastSelectedValue;
            }
        });
    }

    // ── Event mode setup ────────────────────────────────────────────────
    if (isEventMode) {
        activeEventId = eventId;
        activeEventCode = eventCode;

        // Change "New Conversation" button text
        const newConvBtn = document.getElementById("new-conversation-btn");
        if (newConvBtn) newConvBtn.textContent = "New Solution";
    }

    // ── Workspace switcher ─────────────────────────────────────────────
    const workspaceSelect = document.getElementById("workspace-select");
    const sidebarHeader = document.querySelector(".sidebar-header");
    const workspaceEventActions = document.getElementById("workspace-event-actions");
    const workspaceEventRank = document.getElementById("workspace-event-rank");

    if (workspaceSelect) {
        // Event mode — activate sidebar event state
        if (isEventMode) {
            sidebarHeader?.classList.add("event-active");
            if (workspaceEventActions) workspaceEventActions.hidden = false;

            // Leaderboard button → navigate to event page
            document.getElementById("workspace-event-link")?.addEventListener("click", () => {
                window.location.href = `/event/${eventCode}`;
            });

            // Fetch rank and event status asynchronously
            import("../../services/events.js").then(async ({ getEventByCode, getEventLeaderboard }) => {
                try {
                    const ev = await getEventByCode(eventCode);

                    // Disable new solution if event has ended
                    if (ev.status === "ended") {
                        const newConvBtn = document.getElementById("new-conversation-btn");
                        if (newConvBtn) {
                            newConvBtn.disabled = true;
                            newConvBtn.textContent = "Event Ended";
                        }
                        // Hide the composer entirely
                        const composerEl = input?.closest(".composer");
                        if (composerEl) composerEl.style.display = "none";
                        const companyWrapper = companyInput?.closest(".composer-company");
                        if (companyWrapper) companyWrapper.style.display = "none";
                        if (composerNote) composerNote.style.display = "none";

                        // Replace splash with ended illustration
                        if (chatSplash) {
                            chatSplash.innerHTML = `
                                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 2rem;">
                                    <ui5-illustrated-message name="NoActivities" design="Dialog" title-text="This event has ended" subtitle-text="You can still view your previous submissions from the sidebar."></ui5-illustrated-message>
                                </div>
                            `;
                        }
                    }

                    const entries = await getEventLeaderboard(ev.event_id);
                    const myEntry = entries?.find(e => e.is_current_user);
                    const total = entries?.length || 0;
                    if (myEntry && workspaceEventRank) {
                        workspaceEventRank.textContent = `#${myEntry.rank} of ${total}`;
                        if (myEntry.rank <= 3) workspaceEventRank.classList.add("top-rank");
                        workspaceEventRank.hidden = false;
                    }
                } catch (err) {
                    console.warn("Failed to load event rank:", err);
                }
            });
        }

        // Populate with user's events asynchronously
        import("../../services/events.js").then(async ({ getMyEvents }) => {
            try {
                const events = await getMyEvents();
                for (const ev of events) {
                    const option = document.createElement("ui5-option");
                    option.setAttribute("icon", "calendar");
                    option.setAttribute("value", ev.event_id);
                    option.setAttribute("data-code", ev.join_code);
                    option.textContent = ev.name;
                    if (ev.status === "ended") {
                        option.setAttribute("additional-text", "ended");
                    }
                    if (ev.event_id === eventId) {
                        option.setAttribute("selected", "");
                    }
                    workspaceSelect.appendChild(option);
                }
            } catch (err) {
                console.warn("Failed to load workspaces:", err);
            }
        });

        // Handle workspace change — navigate to apply new context
        workspaceSelect.addEventListener("change", () => {
            const selected = workspaceSelect.selectedOption;
            const selectedEventId = selected?.getAttribute("value") || "";
            if (!selectedEventId) {
                window.location.href = "/";
            } else {
                const code = selected.getAttribute("data-code") || "";
                window.location.href = `/?event_id=${selectedEventId}&event_code=${code}`;
            }
        });
    }

    function formatTimestamp(epochMs) {
        return dayjs(epochMs).calendar(null, {
            sameDay: "[Today] h:mm A",
            lastDay: "[Yesterday] h:mm A",
            lastWeek: "MMM D, YYYY h:mm A",
            sameElse: "MMM D, YYYY h:mm A"
        });
    }

    function setComposerState(disabled) {
        if (input) {
            input.disabled = disabled;
        }
        if (sendButton) {
            if (disabled) {
                // Morph the send button into an enabled Stop button while a turn runs.
                sendButton.setAttribute("icon", "stop");
                sendButton.setAttribute("design", "Negative");
                sendButton.setAttribute("tooltip", "Stop");
                sendButton.disabled = false;
            } else {
                sendButton.setAttribute("icon", "paper-plane");
                sendButton.setAttribute("design", "Emphasized");
                sendButton.setAttribute("tooltip", "Send");
                const hasText = (input?.value || "").trim().length > 0;
                const hasFiles = _pendingFiles.length > 0;
                sendButton.disabled = !hasText && !hasFiles;
            }
        }
        // Disable attach button while sending
        if (attachButton) attachButton.disabled = disabled;
    }

    function activeClarificationIndex() {
        for (let index = conversation.length - 1; index >= 0; index -= 1) {
            const msg = conversation[index];
            if (msg?.role !== "assistant" || msg?.content === "…") continue;
            if (msg?.ui?.kind !== "clarification") continue;
            const hasLaterUserAnswer = conversation.slice(index + 1).some((item) => item?.role === "user");
            if (!hasLaterUserAnswer) {
                return index;
            }
        }
        return -1;
    }

    function updateComposerGuidance(activeClarificationIdx) {
        const hasActiveClarification = activeClarificationIdx >= 0;
        const lastMsg = conversation[conversation.length - 1];
        const lastKind = lastMsg?.ui?.kind;
        const isRefinementPhase = lastKind === "refinement" || lastKind === "final";
        if (input) {
            if (isRefinementPhase) {
                input.placeholder = REFINEMENT_PLACEHOLDER;
            } else if (hasActiveClarification) {
                input.placeholder = CLARIFICATION_PLACEHOLDER;
            } else {
                input.placeholder = DEFAULT_PLACEHOLDER;
            }
        }
        if (composerNote) {
            composerNote.textContent = DEFAULT_COMPOSER_NOTE;
        }
    }

    function focusComposer() {
        if (!input) return;
        input.focus();
        requestAnimationFrame(() => {
            input.scrollIntoView({ block: "nearest", behavior: "smooth" });
        });
    }

    function renderMessages() {
        if (!chatTimeline) return;
        closeCriteriaPopover(); // dismiss score-criteria popover; its opener is about to be re-rendered
        chatTimeline.innerHTML = "";
        const latestClarificationIdx = activeClarificationIndex();
        const disableSuggestions = isSending;
        updateComposerGuidance(latestClarificationIdx);

        // Index of the last user message — the only one that gets the Edit affordance.
        let _lastUserIdx = -1;
        for (let i = conversation.length - 1; i >= 0; i -= 1) {
            if (conversation[i].role === "user") { _lastUserIdx = i; break; }
        }

        const hasRealMessages = conversation.some((m) => m.content !== "…");
        if (chatSplash) {
            if (!hasRealMessages) {
                chatSplash.classList.add("visible");
            } else {
                chatSplash.classList.remove("visible");
            }
        }

        conversation.forEach((msg, index) => {
            if (msg.content === "…") {
                const loadRow = document.createElement("div");
                loadRow.className = "loader-row";

                const loader = document.createElement("ui5-busy-indicator");
                loader.setAttribute("active", "");
                loader.setAttribute("size", "M");
                loader.setAttribute("delay", "0");
                loadRow.appendChild(loader);

                const hint = document.createElement("ui5-text");
                hint.className = "loader-hint";
                loadRow.appendChild(hint);

                chatTimeline.appendChild(loadRow);
                _typeLoadingMessage(hint);
                return;
            }

            // Thinking card detection (mini-reimagine / full reimagine)
            // Two formats:
            //   Two-call: "[thinking:direction]hint text" (card only, no question)
            //   Legacy:   "[thinking:direction]hint text\n\nNormal message content..."
            let messageContent = msg.content || "";
            const thinkingMatch = msg.role === "assistant" && messageContent.match(/^\[thinking:(direction|reframe)\]([\s\S]*)/);
            if (thinkingMatch) {
                const level = thinkingMatch[1];
                const remainder = thinkingMatch[2];

                // Split on double-newline if present (legacy combined format)
                const splitIdx = remainder.indexOf("\n\n");
                let thinkingText, questionText;
                if (splitIdx >= 0) {
                    thinkingText = remainder.slice(0, splitIdx).trim();
                    questionText = remainder.slice(splitIdx + 2);
                } else {
                    thinkingText = remainder.trim();
                    questionText = null;
                }

                // Render thinking card
                const thinkingRow = document.createElement("div");
                thinkingRow.className = "message-row assistant";
                const card = document.createElement("div");
                card.className = "thinking-card";
                const icon = document.createElement("ui5-icon");
                icon.setAttribute("name", "lightbulb");
                icon.className = "thinking-icon";
                card.appendChild(icon);
                const textEl = document.createElement("ui5-text");
                textEl.textContent = thinkingText;
                card.appendChild(textEl);
                thinkingRow.appendChild(card);
                chatTimeline.appendChild(thinkingRow);

                // If no question follows (two-call flow), skip bubble rendering
                if (!questionText) return;
                messageContent = questionText;
            }

            const row = document.createElement("div");
            row.className = `message-row ${msg.role}`;

            const bubble = document.createElement("div");
            bubble.className = `bubble ${msg.role}`;

            // Render file chips in user bubbles (icon + filename, click to preview)
            if (msg.role === "user" && msg.files && msg.files.length > 0) {
                const filesDiv = document.createElement("div");
                filesDiv.className = "bubble-files";
                for (const f of msg.files) {
                    const thumbCtx = f.context_id || contextId;
                    const fType = f.original_type || "";
                    const isPdf = fType === "application/pdf" || (f.filename || "").toLowerCase().endsWith(".pdf");
                    const isImage = fType.startsWith("image/");
                    const fileUrl = `${API_BASE_URL}/api/ideation-agent/files/${f.file_id}/full?context_id=${encodeURIComponent(thumbCtx)}`;

                    // Only images and PDFs get clickable preview
                    let clickHandler = null;
                    if (isPdf) {
                        clickHandler = () => {
                            if (f._localPreviewUrl) {
                                _openLightboxPdf(f._localPreviewUrl);
                            } else {
                                _openLightboxPdfAuth(fileUrl);
                            }
                        };
                    } else if (isImage) {
                        clickHandler = () => {
                            if (f._localPreviewUrl) {
                                _openLightbox(f._localPreviewUrl);
                            } else {
                                _openLightboxAuth(fileUrl);
                            }
                        };
                    } else if (f.file_id) {
                        // Non-previewable documents: download on click (only if uploaded)
                        clickHandler = () => _downloadFileAuth(fileUrl, f.filename);
                    }

                    const isDownloadable = !isPdf && !isImage;
                    const chip = _createFileChip({
                        filename: f.filename,
                        fileType: fType,
                        onClick: clickHandler,
                        isDownload: isDownloadable,
                    });
                    filesDiv.appendChild(chip);
                }
                bubble.appendChild(filesDiv);
            }

            // Add copy-to-clipboard button for final solution bubbles
            if (msg.role === "assistant" && msg?.ui?.kind === "final") {
                bubble.style.position = "relative";
                const bubbleCopyBtn = document.createElement("ui5-button");
                bubbleCopyBtn.className = "bubble-copy-btn";
                bubbleCopyBtn.setAttribute("design", "Transparent");
                bubbleCopyBtn.setAttribute("icon", "copy");
                bubbleCopyBtn.setAttribute("tooltip", "Copy solution for Joule Studio");
                bubbleCopyBtn.addEventListener("click", () => {
                    const payload = msg.ui;
                    const title = payload?.title || "Untitled Solution";
                    const industry = payload?.industry || "";
                    const problem = payload?.problem || "";
                    const solution = payload?.solution || "";
                    const appType = payload?.app_type || "Agent";
                    const companyContext = payload?.companyContext || "";

                    let md = `# ${title}\n\n`;
                    if (industry) md += `**Industry:** ${industry}\n\n`;
                    if (appType) md += `**Type:** ${appType}\n\n`;
                    if (companyContext) md += `**Company Context:** ${companyContext}\n\n`;
                    if (problem) md += `## Problem\n\n${problem}\n\n`;
                    if (solution) md += `## Solution\n\n${solution}\n`;

                    navigator.clipboard.writeText(md.trim()).then(() => {
                        bubbleCopyBtn.setAttribute("icon", "accept");
                        setTimeout(() => bubbleCopyBtn.setAttribute("icon", "copy"), 2000);
                    });
                });
                bubble.appendChild(bubbleCopyBtn);
            }

            if (msg.role === "user" && index === _editingIndex) {
                // Inline edit mode: render an editor in place of the message text.
                bubble.appendChild(_buildInlineEditor(index, msg.content || ""));
            } else {
                const messageHtml = marked.parse(messageContent);
                const msgContentDiv = document.createElement("div");
                msgContentDiv.innerHTML = messageHtml;
                bubble.appendChild(msgContentDiv);

                // Edit affordance on the last user message — same mechanism as the copy
                // button (inside the bubble, hover-revealed), positioned bottom-right.
                if (msg.role === "user" && index === _lastUserIdx && !isSending && !_isReadOnly && _editingIndex === null) {
                    bubble.style.position = "relative";
                    const editBtn = document.createElement("ui5-button");
                    editBtn.className = "bubble-edit-btn";
                    editBtn.setAttribute("design", "Transparent");
                    editBtn.setAttribute("icon", "edit");
                    editBtn.setAttribute("tooltip", "Edit");
                    editBtn.addEventListener("click", () => beginEditMessage(index));
                    bubble.appendChild(editBtn);
                }
            }

            const col = document.createElement("div");
            col.className = `bubble-col ${msg.role}`;
            if (msg.role === "user") {
                const tsRow = document.createElement("div");
                tsRow.className = "timestamp-row";
                const ts = document.createElement("ui5-text");
                ts.className = "timestamp";
                ts.textContent = formatTimestamp(msg.ts || Date.now());
                tsRow.appendChild(ts);
                chatTimeline.appendChild(tsRow);
            }
            col.appendChild(bubble);

            // Show sender attribution in shared conversations
            if (_isSharedConversation && msg.role === "user" && msg.senderName) {
                const currentUser = getUserInfo();
                const currentName = [currentUser?.givenName, currentUser?.familyName].filter(Boolean).join(" ");
                // Only show attribution if the sender is someone other than the current user
                if (msg.senderName !== currentName) {
                    const attribution = document.createElement("div");
                    attribution.className = "message-attribution";
                    attribution.textContent = `by ${msg.senderName}`;
                    col.appendChild(attribution);
                }
            }

            if (
                msg.role === "assistant"
                && msg?.ui?.kind === "clarification"
                && index === latestClarificationIdx
                && Array.isArray(msg?.ui?.suggestedReplies)
                && msg.ui.suggestedReplies.length === 3
                && !_isReadOnly
            ) {
                const questions = msg.ui.questions;
                const isMultiQuestion = Array.isArray(questions) && questions.length > 1;

                if (isMultiQuestion) {
                    // Multi-question card rendering: contained card with numbered questions + progress
                    const card = document.createElement("div");
                    card.className = "multi-q-card";
                    card.setAttribute("data-ui5-compact-size", "");

                    const body = document.createElement("div");
                    body.className = "multi-q-body";

                    const selectedAnswers = new Array(questions.length).fill(null);

                    const updateProgress = () => {
                        const answered = selectedAnswers.filter(a => a !== null).length;
                        progressText.textContent = `${answered} of ${questions.length} answered`;
                        if (answered === questions.length) {
                            submitBtn.removeAttribute("disabled");
                            submitBtn.classList.add("multi-q-submit-ready");
                        } else {
                            submitBtn.setAttribute("disabled", "");
                            submitBtn.classList.remove("multi-q-submit-ready");
                        }
                    };

                    questions.forEach((q, qIdx) => {
                        const group = document.createElement("div");
                        group.className = "multi-q-group";

                        // Header: badge + question label
                        const header = document.createElement("div");
                        header.className = "multi-q-group-header";

                        const badge = document.createElement("span");
                        badge.className = "multi-q-badge";
                        badge.textContent = String(qIdx + 1);
                        header.appendChild(badge);

                        const label = document.createElement("ui5-label");
                        label.className = "multi-q-question";
                        label.setAttribute("wrapping-type", "Normal");
                        label.textContent = q.question;
                        header.appendChild(label);

                        group.appendChild(header);

                        // Chip row
                        const chipRow = document.createElement("div");
                        chipRow.className = "multi-q-chips";

                        for (const reply of q.suggestedReplies) {
                            const button = document.createElement("ui5-button");
                            button.className = "clarification-chip";
                            button.setAttribute("design", "Transparent");
                            button.textContent = reply;
                            button.disabled = disableSuggestions;
                            button.addEventListener("click", () => {
                                if (disableSuggestions) return;
                                // If "Other" input is open, close it and restore "Other..." button
                                if (!otherInput.hidden) {
                                    otherInput.hidden = true;
                                    otherChip.hidden = false;
                                    input.value = "";
                                }
                                // Toggle: clicking already-selected deselects
                                if (selectedAnswers[qIdx] === reply) {
                                    selectedAnswers[qIdx] = null;
                                    button.classList.remove("selected");
                                    group.classList.remove("multi-q-answered");
                                } else {
                                    // Deselect siblings
                                    chipRow.querySelectorAll("ui5-button.clarification-chip").forEach(b => b.classList.remove("selected"));
                                    button.classList.add("selected");
                                    selectedAnswers[qIdx] = reply;
                                    group.classList.add("multi-q-answered");
                                }
                                updateProgress();
                            });
                            chipRow.appendChild(button);
                        }
                        group.appendChild(chipRow);

                        // "Other..." chip
                        const otherChip = document.createElement("ui5-button");
                        otherChip.className = "multi-q-other-chip";
                        otherChip.setAttribute("design", "Transparent");
                        otherChip.setAttribute("icon", "edit");
                        otherChip.textContent = "Other...";
                        otherChip.disabled = disableSuggestions;
                        group.appendChild(otherChip);

                        // Inline input row (hidden by default)
                        const otherInput = document.createElement("div");
                        otherInput.className = "multi-q-other-input";
                        otherInput.hidden = true;

                        const input = document.createElement("ui5-input");
                        input.setAttribute("placeholder", "Type your answer...");
                        input.className = "multi-q-input-field";
                        otherInput.appendChild(input);

                        group.appendChild(otherInput);

                        // "Other..." click: show input, deselect any chip
                        otherChip.addEventListener("click", () => {
                            if (disableSuggestions) return;
                            // Deselect any chip in this group
                            chipRow.querySelectorAll("ui5-button.clarification-chip").forEach(b => b.classList.remove("selected"));
                            selectedAnswers[qIdx] = null;
                            otherChip.hidden = true;
                            otherInput.hidden = false;
                            input.focus();
                            updateProgress();
                        });

                        // Live-update answer as user types (no Enter needed)
                        input.addEventListener("input", () => {
                            const value = (input.value || "").trim();
                            selectedAnswers[qIdx] = value || null;
                            if (value) {
                                group.classList.add("multi-q-answered");
                            } else {
                                group.classList.remove("multi-q-answered");
                            }
                            updateProgress();
                        });

                        body.appendChild(group);
                    });

                    card.appendChild(body);

                    // Footer: progress + submit
                    const footer = document.createElement("div");
                    footer.className = "multi-q-footer";

                    const progressText = document.createElement("ui5-text");
                    progressText.className = "multi-q-progress";
                    progressText.textContent = `0 of ${questions.length} answered`;
                    footer.appendChild(progressText);

                    const submitBtn = document.createElement("ui5-button");
                    submitBtn.className = "multi-q-submit";
                    submitBtn.setAttribute("design", "Emphasized");
                    submitBtn.setAttribute("disabled", "");
                    submitBtn.textContent = "Submit answers";
                    submitBtn.addEventListener("click", () => {
                        if (selectedAnswers.some(a => a === null)) return;
                        const answers = selectedAnswers.map((a, i) => `${i + 1}. ${a}`).join("\n");
                        submitUserMessage(answers);
                    });
                    footer.appendChild(submitBtn);

                    card.appendChild(footer);
                    col.appendChild(card);
                } else {
                    // Single-question: original behavior (click = instant send)
                    const suggestions = document.createElement("div");
                    suggestions.className = "clarification-actions";
                    suggestions.setAttribute("data-ui5-compact-size", "");

                    for (const reply of msg.ui.suggestedReplies) {
                        const button = document.createElement("ui5-button");
                        button.className = "clarification-chip";
                        button.setAttribute("design", "Transparent");
                        button.textContent = reply;
                        button.disabled = disableSuggestions;
                        button.addEventListener("click", () => {
                            if (!disableSuggestions) {
                                submitUserMessage(reply);
                            }
                        });
                        suggestions.appendChild(button);
                    }

                    col.appendChild(suggestions);
                }
            }

            // Inline evaluation card for "final" messages — rendered BEFORE action buttons
            let evalHasSuggestions = false;
            if (msg.role === "assistant" && msg?.ui?.kind === "final") {
                const evalState = evaluations.get(index);
                if (evalState) {
                    const card = buildEvaluationCard(evalState, index);
                    col.appendChild(card);

                    // Suggestion chips OUTSIDE the card (as quick-reply buttons)
                    if (evalState.result?.suggestions?.length
                        && index === conversation.length - 1
                        && !isSending
                        && !_isReadOnly) {
                        evalHasSuggestions = true;
                        const chipContainer = document.createElement("div");
                        chipContainer.className = "clarification-actions";
                        chipContainer.setAttribute("data-ui5-compact-size", "");

                        // "Improve all" button
                        const allBtn = document.createElement("ui5-button");
                        allBtn.className = "clarification-chip";
                        allBtn.setAttribute("design", "Emphasized");
                        allBtn.textContent = "Improve all";
                        allBtn.addEventListener("click", () => {
                            chipContainer.querySelectorAll("ui5-button").forEach(b => b.setAttribute("disabled", ""));
                            const evalContext = buildEvalContext(evalState.result);
                            const apiText = `${evalContext}\n\nI'd like to improve my solution by addressing ALL of the evaluator's suggestions above. Before updating the report, ask me clarifying questions (one per suggestion, using multi-question format) to understand how each applies to my specific scenario.`;
                            submitUserMessage(apiText, "Address all evaluation suggestions");
                        });
                        chipContainer.appendChild(allBtn);

                        // Individual suggestion chips
                        for (const suggestion of evalState.result.suggestions) {
                            const chip = document.createElement("ui5-button");
                            chip.className = "clarification-chip";
                            chip.setAttribute("design", "Transparent");
                            chip.textContent = stripInlineMarkdown(suggestion);
                            chip.addEventListener("click", () => {
                                chipContainer.querySelectorAll("ui5-button").forEach(b => b.setAttribute("disabled", ""));
                                const evalContext = buildEvalContext(evalState.result);
                                const apiText = `${evalContext}\n\nI'd like to improve my solution based on this evaluator suggestion: "${suggestion}"\n\nBefore updating the report, ask me a clarifying question to understand how this applies to my specific scenario.`;
                                submitUserMessage(apiText, stripInlineMarkdown(suggestion));
                            });
                            chipContainer.appendChild(chip);
                        }

                        // "Address strategic perspective" chip (purple accent)
                        if (reframeResult && reframeResult.status === "ok") {
                            const reframeBtn = document.createElement("ui5-button");
                            reframeBtn.className = "clarification-chip reframe-chip";
                            reframeBtn.setAttribute("design", "Emphasized");
                            reframeBtn.textContent = "Address strategic perspective";
                            reframeBtn.addEventListener("click", () => {
                                chipContainer.querySelectorAll("ui5-button").forEach(b => b.setAttribute("disabled", ""));
                                const reframeContext = [
                                    `Strategic challenge: ${reframeResult.challenge}`,
                                    `Alternative angle: ${reframeResult.alternative_angle}`,
                                    `Key risk: ${reframeResult.risk}`,
                                    `Summary: ${reframeResult.reframe_summary}`,
                                ].join("\n");
                                const apiText = `[REFRAME_CONTEXT]\n${reframeContext}\n[/REFRAME_CONTEXT]\n\nI'd like to address the strategic perspective raised above. Please ask me clarifying questions about how these concerns apply to my specific situation, then update the recommendation.`;
                                submitUserMessage(apiText, "Address strategic perspective");
                            });
                            chipContainer.appendChild(reframeBtn);
                        }

                        col.appendChild(chipContainer);
                    }
                } else if (contextId) {
                    // Skeleton placeholder while stored evals load
                    const skeleton = document.createElement("div");
                    skeleton.className = "evaluation-card eval-skeleton no-animate";
                    skeleton.innerHTML = `
                        <div class="eval-skeleton-header">
                            <div class="eval-skeleton-bar eval-skeleton-score"></div>
                            <div class="eval-skeleton-headline">
                                <div class="eval-skeleton-bar eval-skeleton-w80"></div>
                                <div class="eval-skeleton-bar eval-skeleton-w30"></div>
                            </div>
                        </div>
                        <div class="eval-skeleton-body">
                            <div class="eval-skeleton-bar eval-skeleton-w95"></div>
                            <div class="eval-skeleton-bar eval-skeleton-w80"></div>
                            <div class="eval-skeleton-bar eval-skeleton-w95"></div>
                            <div class="eval-skeleton-bar eval-skeleton-w60"></div>
                            <div class="eval-skeleton-bar eval-skeleton-w80"></div>
                            <div class="eval-skeleton-bar eval-skeleton-w50"></div>
                        </div>
                    `;
                    col.appendChild(skeleton);
                }
            }

            // "Go deeper" action: shown on the final report (latest message)
            // Suppressed when eval card has suggestion chips or evals are still loading
            let evalStillLoading = msg.role === "assistant" && msg?.ui?.kind === "final"
                && contextId && (!evaluations.has(index) || evaluations.get(index)?.loading);
            if (
                msg.role === "assistant"
                && msg?.ui?.kind === "final"
                && index === conversation.length - 1
                && !isSending
                && !_isReadOnly
                && !evalHasSuggestions
                && !evalStillLoading
            ) {
                const actions = document.createElement("div");
                actions.className = "clarification-actions";
                actions.setAttribute("data-ui5-compact-size", "");

                const button = document.createElement("ui5-button");
                button.className = "clarification-chip";
                button.setAttribute("design", "Transparent");
                button.textContent = "Go deeper";
                button.addEventListener("click", () => {
                    submitUserMessage("I'd like to go deeper on this");
                });
                actions.appendChild(button);
                col.appendChild(actions);
            }

            // "Next step" card: shown on the final report (latest message)
            if (
                msg.role === "assistant"
                && msg?.ui?.kind === "final"
                && index === conversation.length - 1
                && !isSending
                && !_isReadOnly
                && contextId
            ) {
                const handoverCard = document.createElement("div");
                handoverCard.className = "handover-prompt-card";
                if (_handoverCardShown) handoverCard.style.animation = "none";
                _handoverCardShown = true;
                handoverCard.innerHTML = `
                    <ui5-icon name="journey-arrive" class="handover-prompt-icon"></ui5-icon>
                    <div class="handover-prompt-text">
                        <strong>Ready for the next step?</strong>
                        <span>Prepare a handover package for the build team, or copy the solution for Joule Studio to start building.</span>
                        <div class="handover-prompt-actions">
                            <ui5-button design="Emphasized" class="handover-prompt-btn">Open Handover Package</ui5-button>
                            <ui5-button design="Default" icon="copy" class="joule-copy-btn">Copy for Joule Studio</ui5-button>
                        </div>
                    </div>
                `;
                // Handover Package button
                handoverCard.querySelector(".handover-prompt-btn").addEventListener("click", async () => {
                    const btn = handoverCard.querySelector(".handover-prompt-btn");
                    btn.setAttribute("loading", "");
                    btn.disabled = true;
                    try {
                        const { createHandover } = await import("../../services/handover.js");
                        const payload = {};
                        if (msg.ui?.title) payload.usecase_title = msg.ui.title;
                        if (msg.ui?.industry) payload.industry = msg.ui.industry;
                        const result = await createHandover({
                            linkedConversationId: contextId,
                            payload: Object.keys(payload).length ? payload : undefined,
                        });
                        if (result?.id) {
                            window.location.href = `/handover?id=${result.id}`;
                        }
                    } catch (err) {
                        console.error("Failed to create handover:", err);
                        btn.removeAttribute("loading");
                        btn.disabled = false;
                    }
                });
                // Copy for Joule Studio button
                handoverCard.querySelector(".joule-copy-btn").addEventListener("click", () => {
                    const btn = handoverCard.querySelector(".joule-copy-btn");
                    const ui = msg.ui || {};
                    const title = ui.title || "Untitled Solution";
                    const industry = ui.industry || "";
                    const problem = ui.problem || "";
                    const solution = ui.solution || "";
                    const appType = ui.app_type || "Agent";
                    const companyContext = ui.companyContext || "";

                    let md = `# ${title}\n\n`;
                    if (industry) md += `**Industry:** ${industry}\n\n`;
                    if (appType) md += `**Type:** ${appType}\n\n`;
                    if (companyContext) md += `**Company Context:** ${companyContext}\n\n`;
                    if (problem) md += `## Problem\n\n${problem}\n\n`;
                    if (solution) md += `## Solution\n\n${solution}\n`;

                    navigator.clipboard.writeText(md.trim()).then(() => {
                        btn.textContent = "Copied!";
                        btn.setAttribute("icon", "accept");
                        setTimeout(() => {
                            btn.textContent = "Copy for Joule Studio";
                            btn.setAttribute("icon", "copy");
                        }, 2000);
                    });
                });
                col.appendChild(handoverCard);
            }

            // Refinement action: shown after "refinement" kind only (latest message)
            if (
                msg.role === "assistant"
                && msg?.ui?.kind === "refinement"
                && index === conversation.length - 1
                && !isSending
                && !_isReadOnly
            ) {
                const actions = document.createElement("div");
                actions.className = "clarification-actions";
                actions.setAttribute("data-ui5-compact-size", "");

                const button = document.createElement("ui5-button");
                button.className = "clarification-chip";
                button.setAttribute("design", "Transparent");
                button.textContent = "Update report";
                button.addEventListener("click", () => {
                    submitUserMessage("[ACTION:UPDATE_REPORT]", "Update report");
                });
                actions.appendChild(button);
                col.appendChild(actions);
            }

            row.appendChild(col);

            chatTimeline.appendChild(row);
        });

        setTimeout(() => {
            chatTimeline.scrollTop = chatTimeline.scrollHeight;
        }, 10);
    }
    _renderMessages = renderMessages;
    _submitUserMessage = submitUserMessage;

    async function submitUserMessage(text, displayText, options = {}) {
        if (!text && _pendingFiles.length === 0) return;
        const rewindMessageId = options.rewindMessageId || null; // set on edit-resubmit

        // Capture company name on first message only
        let companyName = "";
        if (!companyNameSent && companyInput) {
            companyName = (companyInput.value || "").trim();
            companyNameSent = true;
            const companyWrapper = companyInput.closest(".composer-company");
            if (companyWrapper) {
                companyWrapper.hidden = true;
            }
            if (companyName) {
                enrichmentStatus = "pending";
                perplexityStatus = "pending";
                linkedinStatus = "pending";
                renderDrawerContent();
                renderPerplexityTab();
                renderLinkedInTab();
            }
        }

        // Capture pending files before clearing (keep preview URLs alive for bubble display)
        const filesToUpload = [..._pendingFiles];
        _pendingFiles = [];
        _renderAttachmentStrip();

        // Build file metadata immediately so chips show in the bubble right away
        const immediateFiles = filesToUpload.map(pf => ({
            file_id: null,
            filename: pf.file.name,
            context_id: null,
            num_outputs: 1,
            original_type: pf.file.type || "",
            _localPreviewUrl: pf.previewUrl,
        }));

        const _turnMessageId = makeMessageId();  // reused as this turn's server message id (for Stop/Edit addressing)
        const restoreText = text || "";           // returned to the composer if the user stops this turn
        conversation.push({ role: "user", content: displayText || text || "(images attached)", ts: Date.now(), msgId: _turnMessageId, files: immediateFiles.length > 0 ? immediateFiles : undefined });
        if (input) {
            input.value = "";
        }
        renderMessages();

        conversation.push({ role: "assistant", content: "…" });
        renderMessages();
        _startLoadingMessages();

        // Add sidebar entry immediately for new conversations (before awaiting API)
        const isNewConversation = !contextId;
        if (isNewConversation) {
            addOptimisticSidebarEntry("__pending__", companyName);
        }

        _stopRequested = false;
        _sendAbort = new AbortController();
        // Ensure a client-side contextId from the very first turn, so Stop can target
        // it via the cancel endpoint (otherwise turn 1 would only abort client-side).
        if (!contextId) {
            contextId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            const url = new URL(window.location);
            url.searchParams.set("c", contextId);
            if (activeEventId) url.searchParams.set("event_id", activeEventId);
            if (activeEventCode) url.searchParams.set("event_code", activeEventCode);
            history.replaceState(null, "", url);
        }
        isSending = true;
        setComposerState(true);
        const sendingContextId = contextId;
        let generatedTitle = null;
        try {
            // Edit-resubmit: rewind server history to before the edited message first.
            // The loading state is already on screen, so this runs "under" the spinner
            // instead of blocking the click with no feedback.
            if (rewindMessageId && contextId) {
                try {
                    await request(
                        `${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(contextId)}/rewind`,
                        "POST",
                        { messageId: rewindMessageId },
                    );
                } catch (e) {
                    console.error("Rewind failed:", e);
                }
            }
            // Upload files if any (need a context_id — use existing or generate one)
            let uploadedFiles = [];
            if (filesToUpload.length > 0) {
                // For new conversations, generate a context_id client-side for the upload path.
                // The A2A handler will create the conversation when the message arrives.
                if (!contextId) {
                    contextId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
                    // Update URL with new context_id
                    const url = new URL(window.location);
                    url.searchParams.set("c", contextId);
                    if (activeEventId) url.searchParams.set("event_id", activeEventId);
                    if (activeEventCode) url.searchParams.set("event_code", activeEventCode);
                    history.replaceState(null, "", url);
                }
                uploadedFiles = await _uploadPendingFiles(filesToUpload, contextId);
                if (uploadedFiles.length === 0 && filesToUpload.length > 0) {
                    _showFileToast("All file uploads failed. Sending message without attachments.");
                } else if (uploadedFiles.length < filesToUpload.length) {
                    _showFileToast(`${filesToUpload.length - uploadedFiles.length} file(s) failed to upload.`);
                }

                // Update conversation entry with file metadata for rendering
                const userMsg = conversation[conversation.length - 2];
                if (userMsg && userMsg.role === "user") {
                    userMsg.files = uploadedFiles.map(f => ({
                        file_id: f.file_id,
                        filename: f.filename,
                        context_id: contextId,
                        num_outputs: f.num_outputs,
                        original_type: f.original_type || "",
                        _localPreviewUrl: f._localPreviewUrl || null,
                    }));
                    delete userMsg._uploadingFiles;
                    renderMessages();
                }
            }

            let reply = await postA2AMessage(text || "", companyName, displayText || "", uploadedFiles, _sendAbort.signal, _turnMessageId);
            // Stale guard: discard if user switched to a different conversation
            if (sendingContextId && contextId !== sendingContextId) {
                return;
            }
            generatedTitle = reply.generatedTitle;
            logFinalResultFields(reply.ui);
            if (reply.ui?.kind === "confirmed") {
                conversationComplete = true;
            }
            if (!enrichmentPollTimer && contextId) {
                const problemDone = perplexityProblemStatus === "complete" || perplexityProblemStatus === "failed";
                const needsPoll = enrichmentStatus === "pending" || perplexityStatus === "pending" || perplexityProblemStatus === "pending" || !problemDone;
                if (needsPoll) startEnrichmentPolling();
            }
            const last = conversation[conversation.length - 1];
            if (last && last.role === "assistant") {
                last.content = reply.text;
                last.ui = reply.ui || null;
            } else {
                conversation.push({ role: "assistant", content: reply.text, ui: reply.ui || null });
            }

            // Auto-continue after thinking card — send follow-up to get the informed question
            if (reply.ui?.kind === "thinking") {
                renderMessages();
                chatTimeline.scrollTop = chatTimeline.scrollHeight;
                // Show "Thinking..." loader for the follow-up question
                conversation.push({ role: "assistant", content: "…" });
                renderMessages();
                chatTimeline.scrollTop = chatTimeline.scrollHeight;
                const followUp = await postA2AMessage("__continue__", "", "", [], _sendAbort?.signal);
                const lastAssistant = conversation[conversation.length - 1];
                if (lastAssistant && lastAssistant.role === "assistant") {
                    lastAssistant.content = followUp.text;
                    lastAssistant.ui = followUp.ui || null;
                } else {
                    conversation.push({ role: "assistant", content: followUp.text, ui: followUp.ui || null });
                }
                reply = followUp; // so downstream checks (final, etc.) use the real response
            }

            // Auto-evaluate when agent produces a final solution (after conversation is updated)
            if (reply.ui?.kind === "final" && !_isReadOnly) {
                const msgIndex = conversation.length - 1;
                triggerAutoEvaluation(reply.ui, msgIndex);
            }
            // Start polling for strategic reframe (background LLM generates it)
            if (reply.ui?.kind === "final") {
                startReframePoll();
                startCatalogMatchPoll();
            }
        } catch (error) {
            if (error?.name === "AbortError" || _stopRequested) {
                // Stopped by the user: discard this exchange (no partial answer exists,
                // since responses are atomic), and return the typed text to the composer.
                if (conversation.length && conversation[conversation.length - 1]?.content === "…") {
                    conversation.pop();
                }
                const tail = conversation[conversation.length - 1];
                if (tail && tail.role === "user") {
                    conversation.pop();
                    if (input && restoreText) input.value = restoreText;
                }
                renderMessages();
                return;
            }
            console.error("Error:", error);
            const last = conversation[conversation.length - 1];
            const message = `An error occurred: ${error.message}`;
            if (last && last.role === "assistant") {
                last.content = message;
                last.ui = { kind: "error", allowFreeform: true, clarificationCount: 0 };
            } else {
                conversation.push({
                    role: "assistant",
                    content: message,
                    ui: { kind: "error", allowFreeform: true, clarificationCount: 0 }
                });
            }
        } finally {
            isSending = false;
            _sendAbort = null;
            _inflightMessageId = null;
            _stopRequested = false;
            _stopLoadingMessages();
            setComposerState(false);
            renderMessages();
            renderPerplexityTab();
            // Update sidebar: replace placeholder with real contextId
            if (contextId && activeConversationId !== contextId) {
                activeConversationId = contextId;
                const pending = conversationList?.querySelector('[data-conv-id="__pending__"]');
                if (pending) {
                    pending.setAttribute("data-conv-id", contextId);
                } else {
                    addOptimisticSidebarEntry(contextId, companyName);
                }
            }
            // Reveal the subheader (title bar + research) for a new conversation.
            // Prefer the generated title, but fall back to the company name / default so
            // the bar still appears live even when LLM title generation is unavailable.
            if (isNewConversation && contextId) {
                const displayTitle = capitalizeFirst(generatedTitle || companyName || "New conversation");
                const item = conversationList?.querySelector(`[data-conv-id="${contextId}"]`);
                const titleSpan = item?.querySelector(".conv-item-title");
                if (titleSpan && generatedTitle) titleSpan.textContent = displayTitle;
                const subheaderTitle = document.getElementById("chat-subheader-title");
                const subheader = document.getElementById("chat-subheader");
                if (subheaderTitle) subheaderTitle.textContent = displayTitle;
                if (subheader) subheader.style.display = "";
                document.querySelector(".chat-main")?.classList.add("has-subheader");
            }
            if (input) {
                setTimeout(focusComposer, 10);
            }
        }
    }

    async function sendMessage() {
        const text = (input?.value || "").trim();
        if (!text && _pendingFiles.length === 0) return;
        return submitUserMessage(text);
    }

    // ── Stop generation ──────────────────────────────────────────────────
    function stopGeneration() {
        if (!isSending) return;
        _stopRequested = true;
        // Tell the server to discard the in-flight turn (best-effort, self-retrying).
        if (contextId && _inflightMessageId) {
            _cancelTurn(contextId, _inflightMessageId);
        }
        // Abort the client wait — the submitUserMessage catch handles UI cleanup.
        if (_sendAbort) {
            try { _sendAbort.abort(); } catch (_) { /* noop */ }
        }
    }

    async function _cancelTurn(ctxId, msgId) {
        const path = `${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(ctxId)}/cancel`;
        try {
            await request(path, "POST", { messageId: msgId });
        } catch (_) {
            // One retry so auto-discard is robust against a transient failure.
            try { await request(path, "POST", { messageId: msgId }); } catch (__) { /* give up */ }
        }
    }

    // ── Edit last user message (rewind & rerun) ──────────────────────────
    function beginEditMessage(index) {
        if (isSending) return;
        _editingIndex = index;
        _editingDraft = conversation[index]?.content || "";
        renderMessages();
    }

    function cancelEditMessage() {
        _editingIndex = null;
        _editingDraft = null;
        renderMessages();
    }

    async function submitEdit(index, newText) {
        const trimmed = (newText || "").trim();
        const target = conversation[index];
        if (!target || target.role !== "user") return;
        if (!trimmed) return;
        const rewindMessageId = target.msgId;
        _editingIndex = null;
        _editingDraft = null;
        // Truncate the local mirror to before the edited message, then resend. The
        // server-side rewind runs inside submitUserMessage once the loading state is
        // visible, so Save gives instant feedback instead of waiting on the round-trip.
        conversation.length = index;
        await submitUserMessage(trimmed, undefined, { rewindMessageId });
    }

    function _buildInlineEditor(index, text) {
        const wrap = document.createElement("div");
        wrap.className = "bubble-edit";
        const ta = document.createElement("ui5-textarea");
        ta.className = "bubble-edit-input";
        ta.value = _editingDraft != null ? _editingDraft : text;
        ta.setAttribute("rows", "3");
        ta.setAttribute("growing", "");
        ta.addEventListener("input", () => { _editingDraft = ta.value; });
        const actions = document.createElement("div");
        actions.className = "bubble-edit-actions";
        const cancelBtn = document.createElement("ui5-button");
        cancelBtn.setAttribute("design", "Transparent");
        cancelBtn.textContent = "Cancel";
        cancelBtn.addEventListener("click", cancelEditMessage);
        const saveBtn = document.createElement("ui5-button");
        saveBtn.setAttribute("design", "Emphasized");
        saveBtn.textContent = "Save";
        saveBtn.addEventListener("click", () => submitEdit(index, ta.value));
        actions.appendChild(cancelBtn);
        actions.appendChild(saveBtn);
        ta.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
                e.preventDefault();
                submitEdit(index, ta.value);
            } else if (e.key === "Escape") {
                e.preventDefault();
                cancelEditMessage();
            }
        });
        wrap.appendChild(ta);
        wrap.appendChild(actions);
        setTimeout(() => { try { ta.focus(); } catch (_) { /* noop */ } }, 0);
        return wrap;
    }

    if (sendButton) {
        sendButton.addEventListener("click", () => {
            if (isSending) {
                stopGeneration();
            } else {
                sendMessage();
            }
        });
        sendButton.disabled = true;
    }

    if (input) {
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                if (e.shiftKey) return;
                if (!e.metaKey && !e.ctrlKey) {
                    e.preventDefault();
                    sendMessage();
                }
            }
        });
        input.addEventListener("input", () => {
            setComposerState(false);
        });
    }

    if (companyInput) {
        companyInput.addEventListener("input", () => {
            setComposerState(false);
        });
        companyInput.addEventListener("selection-change", () => {
            setComposerState(false);
        });
    }

    renderMessages();

    // --- Conversation sidebar logic ---
    const conversationList = document.getElementById("conversation-list");
    const newConversationBtn = document.getElementById("new-conversation-btn");
    const contextMenu = document.getElementById("conversation-context-menu");
    let activeConversationId = null;
    let contextMenuTargetId = null;

    function formatConvDate(isoStr) {
        if (!isoStr) return "";
        const d = dayjs(isoStr);
        const now = dayjs();
        if (d.isSame(now, "day")) return d.format("h:mm A");
        if (d.isSame(now.subtract(1, "day"), "day")) return "Yesterday";
        if (d.isSame(now, "week")) return d.format("ddd");
        if (d.isSame(now, "year")) return d.format("MMM D");
        return d.format("MMM D, YYYY");
    }

    function capitalizeFirst(str) {
        if (!str) return str;
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    function addOptimisticSidebarEntry(convId, title) {
        if (!conversationList) return;
        // Don't duplicate if already present
        if (conversationList.querySelector(`[data-conv-id="${convId}"]`)) {
            // Just ensure it's selected
            conversationList.querySelectorAll("ui5-li-custom").forEach(el => {
                el.toggleAttribute("selected", el.getAttribute("data-conv-id") === convId);
            });
            return;
        }

        // Deselect existing items
        conversationList.querySelectorAll("ui5-li-custom[selected]").forEach(el => el.removeAttribute("selected"));

        const item = document.createElement("ui5-li-custom");
        item.setAttribute("type", "Active");
        item.setAttribute("data-conv-id", convId);
        item.setAttribute("selected", "");

        const row = document.createElement("div");
        row.className = "conv-item-row";

        const titleSpan = document.createElement("span");
        titleSpan.className = "conv-item-title";
        titleSpan.textContent = capitalizeFirst(title || "New conversation");
        row.appendChild(titleSpan);

        const date = document.createElement("span");
        date.className = "conv-item-date";
        date.textContent = "Now";
        row.appendChild(date);

        const btn = document.createElement("ui5-button");
        btn.setAttribute("icon", "overflow");
        btn.setAttribute("design", "Transparent");
        btn.setAttribute("tooltip", "More");
        btn.setAttribute("accessible-name", "More actions");
        btn.className = "conv-item-more";
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            contextMenuTargetId = item.getAttribute("data-conv-id");
            if (contextMenu) {
                contextMenu.opener = btn;
                contextMenu.open = true;
            }
        });
        row.appendChild(btn);

        item.appendChild(row);
        // Insert at top of list
        conversationList.prepend(item);
    }

    async function loadConversationList() {
        if (!conversationList) return;
        const sidebarBusy = document.getElementById("sidebar-busy");
        if (sidebarBusy) sidebarBusy.active = true;
        conversationList.setAttribute("no-data-text", "");
        try {
            const endpoint = isEventMode
                ? `${CONVERSATIONS_ENDPOINT}?event_id=${eventId}`
                : `${CONVERSATIONS_ENDPOINT}?include_shared=true`;
            const data = await request(endpoint);
            const conversations = data?.conversations || [];
            const sharedConversations = data?.shared_conversations || [];
            conversationList.innerHTML = "";
            for (const conv of conversations) {
                const item = document.createElement("ui5-li-custom");
                item.setAttribute("type", "Active");
                item.setAttribute("data-conv-id", conv.id);
                if (conv.id === activeConversationId) {
                    item.setAttribute("selected", "");
                }

                const row = document.createElement("div");
                row.className = "conv-item-row";

                const title = document.createElement("span");
                title.className = "conv-item-title";
                title.textContent = capitalizeFirst(conv.title || conv.company_name || "Untitled");
                row.appendChild(title);

                if (conv.shared_member_count > 0) {
                    const shareIcon = document.createElement("ui5-icon");
                    shareIcon.setAttribute("name", "share-2");
                    shareIcon.className = "conv-shared-icon";
                    shareIcon.setAttribute("title", `Shared with ${conv.shared_member_count} member${conv.shared_member_count > 1 ? "s" : ""}`);
                    row.appendChild(shareIcon);
                }

                const date = document.createElement("span");
                date.className = "conv-item-date";
                date.textContent = formatConvDate(conv.updated_at);
                row.appendChild(date);

                const btn = document.createElement("ui5-button");
                btn.setAttribute("icon", "overflow");
                btn.setAttribute("design", "Transparent");
                btn.setAttribute("tooltip", "More");
                btn.setAttribute("accessible-name", "More actions");
                btn.className = "conv-item-more";
                btn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    contextMenuTargetId = conv.id;
                    if (contextMenu) {
                        contextMenu.opener = btn;
                        contextMenu.open = true;
                    }
                });
                row.appendChild(btn);

                item.appendChild(row);
                conversationList.appendChild(item);
            }

            // Render "Shared with me" section if there are shared conversations
            if (sharedConversations.length > 0) {
                const separator = document.createElement("ui5-li-custom");
                separator.setAttribute("type", "Inactive");
                separator.innerHTML = `<div class="sidebar-separator">Shared with me</div>`;
                conversationList.appendChild(separator);

                for (const conv of sharedConversations) {
                    const item = document.createElement("ui5-li-custom");
                    item.setAttribute("type", "Active");
                    item.setAttribute("data-conv-id", conv.id);
                    item.setAttribute("data-shared", "true");
                    item.setAttribute("data-permission", conv.permission || "read");
                    if (conv.id === activeConversationId) {
                        item.setAttribute("selected", "");
                    }

                    const row = document.createElement("div");
                    row.className = "conv-item-row";

                    const titleWrap = document.createElement("div");
                    titleWrap.style.cssText = "display: flex; flex-direction: column; flex: 1; min-width: 0;";

                    const titleEl = document.createElement("span");
                    titleEl.className = "conv-item-title";
                    titleEl.textContent = conv.title || "Untitled";
                    titleWrap.appendChild(titleEl);

                    const ownerEl = document.createElement("span");
                    ownerEl.className = "shared-conv-owner";
                    const ownerName = conv.owner_name || "";
                    ownerEl.textContent = ownerName ? `from ${ownerName}` : "";
                    titleWrap.appendChild(ownerEl);

                    row.appendChild(titleWrap);

                    const badge = document.createElement("span");
                    badge.className = `shared-badge shared-badge--${conv.permission || "read"}`;
                    badge.textContent = conv.permission === "write" ? "edit" : "view";
                    row.appendChild(badge);

                    item.appendChild(row);
                    conversationList.appendChild(item);
                }
            }
        } catch (err) {
            console.error("[Sidebar] Failed to load conversations:", err);
        } finally {
            if (sidebarBusy) sidebarBusy.active = false;
            conversationList.setAttribute("no-data-text", "No conversations yet");
        }
    }

    async function selectConversation(convId) {
        if (!convId || convId === activeConversationId) return;
        activeConversationId = convId;
        contextId = convId;
        companyNameSent = true;
        conversationComplete = false;
        _handoverCardShown = false;

        // Clear any pending file uploads from previous context
        _clearPendingFiles();

        // Persist in URL so refresh restores this conversation
        const url = new URL(window.location);
        url.searchParams.set("c", convId);
        if (isEventMode) {
            url.searchParams.set("event_id", eventId);
            if (eventCode) url.searchParams.set("event_code", eventCode);
        }
        history.replaceState(null, "", url);

        // Reset evaluation state for the new context
        evaluations.clear();
        for (const [, timer] of activeLoadingTimers) clearInterval(timer);
        activeLoadingTimers.clear();

        // Reset enrichment state for the new context
        stopEnrichmentPolling();
        enrichmentStatus = "idle";
        enrichmentResult = null;
        perplexityStatus = "idle";
        perplexityResult = null;
        perplexityProblemStatus = "idle";
        perplexityProblemResult = null;
        linkedinStatus = "idle";
        linkedinResult = null;
        // Reset strategic reframe state
        stopReframePoll();
        reframeStatus = "idle";
        reframeResult = null;
        // Reset catalog match state
        stopCatalogMatchPoll();
        catalogMatchStatus = "idle";
        catalogMatchResult = null;
        drawerAutoOpened = true;
        resetResearchDrawer();
        renderDrawerContent();
        renderPerplexityTab();
        renderLinkedInTab();

        // Hide splash, show chat
        if (chatSplash) chatSplash.classList.remove("visible");
        const companyWrapper = companyInput?.closest(".composer-company");
        if (companyWrapper) companyWrapper.hidden = true;

        // Pre-hide composer for shared read-only conversations (avoid flash)
        const sidebarItem = conversationList?.querySelector(`ui5-li-custom[data-conv-id="${convId}"]`);
        if (sidebarItem?.getAttribute("data-shared") === "true" && sidebarItem?.getAttribute("data-permission") === "read") {
            _isReadOnly = true;
            _isSharedConversation = true;
            const composer = document.querySelector(".composer");
            const composerNote = document.getElementById("composer-note");
            if (composer) composer.style.display = "none";
            if (composerNote) composerNote.style.display = "none";
        }

        // Show loading state immediately
        conversation.length = 0;
        if (chatTimeline) {
            chatTimeline.innerHTML = "";
            const loader = document.createElement("ui5-busy-indicator");
            loader.setAttribute("active", "");
            loader.setAttribute("size", "M");
            loader.setAttribute("delay", "0");
            loader.style.display = "flex";
            loader.style.justifyContent = "center";
            loader.style.padding = "3rem 0";
            chatTimeline.appendChild(loader);
        }

        // Load messages from backend
        try {
            const sendingContextId = contextId;
            const data = await request(`${CONVERSATIONS_ENDPOINT}/${convId}/messages`);
            if (contextId !== sendingContextId) return;
            const messages = data?.messages || [];
            const accessLevel = data?.access_level || "owner";
            conversation.length = 0;
            for (const msg of messages) {
                // For user messages, prefer displayText metadata over raw API content
                const content = (msg.role === "user" && msg.metadata?.displayText)
                    ? msg.metadata.displayText
                    : (msg.content || "");
                const entry = {
                    role: msg.role === "agent" ? "assistant" : msg.role,
                    content,
                    ts: msg.created_at ? new Date(msg.created_at).getTime() : Date.now(),
                    msgId: msg.id,
                    seq: msg.sequence_num,
                };
                if (msg.metadata?.ui) {
                    entry.ui = msg.metadata.ui;
                }
                if (msg.metadata?.sender_name) {
                    entry.senderName = msg.metadata.sender_name;
                }
                // File attachments (Object Store references)
                if (msg.metadata?.files && Array.isArray(msg.metadata.files)) {
                    entry.files = msg.metadata.files;
                }
                conversation.push(entry);
            }

            // Set shared/read-only state BEFORE rendering so chips are correctly hidden
            _isSharedConversation = accessLevel !== "owner";
            _setReadOnlyMode(accessLevel === "read");

            renderMessages();

            if (accessLevel !== "read") {
                focusComposer();
            }
            if (contextId) startEnrichmentPolling();

            // Load stored evaluations for this conversation
            const hasFinal = conversation.some((m) => m.ui?.kind === "final");
            if (hasFinal) {
                loadStoredEvaluations();
                // Load cached reframe (single poll to fetch from HANA)
                pollReframe();
                // Load cached catalog matches (single poll to fetch from HANA)
                pollCatalogMatches();
            }

            // Auto-activate event mode for shared conversations that belong to an event
            if (!activeEventId && data?.event_id) {
                const convEventId = data.event_id;
                // Check workspace select for membership (options are loaded asynchronously)
                const _tryActivateEvent = () => {
                    const ws = document.getElementById("workspace-select");
                    if (!ws) return false;
                    for (const opt of ws.querySelectorAll("ui5-option")) {
                        if (opt.getAttribute("value") === convEventId) {
                            activeEventId = convEventId;
                            activeEventCode = opt.getAttribute("data-code") || "";
                            // Activate sidebar event UI
                            const sh = document.querySelector(".sidebar-header");
                            if (sh) sh.classList.add("event-active");
                            const ea = document.getElementById("workspace-event-actions");
                            if (ea) ea.hidden = false;
                            // Select this event in dropdown
                            opt.setAttribute("selected", "");
                            return true;
                        }
                    }
                    return false;
                };
                // Try immediately; if workspace options haven't loaded yet, retry after a short delay
                if (!_tryActivateEvent()) {
                    setTimeout(_tryActivateEvent, 500);
                }
            }
        } catch (err) {
            console.error("[Sidebar] Failed to load conversation messages:", err);
        }

        // Update sidebar selection
        if (conversationList) {
            for (const item of conversationList.querySelectorAll("ui5-li-custom")) {
                if (item.getAttribute("data-conv-id") === convId) {
                    item.setAttribute("selected", "");
                    // Update subheader with full title
                    const subheader = document.getElementById("chat-subheader");
                    const subheaderTitle = document.getElementById("chat-subheader-title");
                    const subheaderCompany = document.getElementById("chat-subheader-company");
                    const convTitle = item.querySelector(".conv-item-title")?.textContent || "";
                    if (subheader) subheader.style.display = "";
                    if (subheaderTitle) subheaderTitle.textContent = convTitle;
                    if (subheaderCompany) subheaderCompany.textContent = "";
                    document.querySelector(".chat-main")?.classList.add("has-subheader");
                } else {
                    item.removeAttribute("selected");
                }
            }
        }
    }

    function startNewConversation() {
        activeConversationId = null;
        contextId = null;
        companyNameSent = false;
        conversationComplete = false;

        // Clear any pending file uploads from previous context
        _clearPendingFiles();

        evaluations.clear();
        for (const [, timer] of activeLoadingTimers) clearInterval(timer);
        activeLoadingTimers.clear();
        enrichmentStatus = "idle";
        enrichmentResult = null;
        perplexityStatus = "idle";
        perplexityResult = null;
        perplexityProblemStatus = "idle";
        perplexityProblemResult = null;
        linkedinStatus = "idle";
        linkedinResult = null;
        stopEnrichmentPolling();
        reframeStatus = "idle";
        reframeResult = null;
        stopReframePoll();
        catalogMatchStatus = "idle";
        catalogMatchResult = null;
        stopCatalogMatchPoll();
        resetResearchDrawer();
        conversation.length = 0;

        // Reset read-only and shared state
        _isSharedConversation = false;
        _setReadOnlyMode(false);

        // Clear conversation from URL (but preserve event params)
        const url = new URL(window.location);
        url.searchParams.delete("c");
        if (isEventMode) {
            url.searchParams.set("event_id", eventId);
            if (eventCode) url.searchParams.set("event_code", eventCode);
        }
        history.replaceState(null, "", url);

        // Hide research panel
        if (drawerOpen) closeDrawer();
        if (researchToggle) researchToggle.style.display = "none";
        hideResearchBadge();

        // Hide subheader
        const subheader = document.getElementById("chat-subheader");
        if (subheader) subheader.style.display = "none";
        document.querySelector(".chat-main")?.classList.remove("has-subheader");

        // Show splash and company input
        const companyWrapper = companyInput?.closest(".composer-company");
        if (companyWrapper) companyWrapper.hidden = false;
        if (companyInput) companyInput.value = "";
        renderMessages();

        // Deselect sidebar
        if (conversationList) {
            for (const item of conversationList.querySelectorAll("ui5-li-custom")) {
                item.removeAttribute("selected");
            }
        }

        if (companyInput) {
            requestAnimationFrame(() => companyInput.focus());
        } else if (input) {
            requestAnimationFrame(() => input.focus());
        }
    }

    // Inline rename logic
    function startInlineRename(convId) {
        const item = conversationList?.querySelector(`ui5-li-custom[data-conv-id="${convId}"]`);
        if (!item) return;
        const titleEl = item.querySelector(".conv-item-title");
        if (!titleEl) return;
        const currentTitle = titleEl.textContent.trim();
        const input = document.createElement("ui5-input");
        input.value = currentTitle;
        input.style.width = "100%";
        input.setAttribute("data-renaming", convId);
        titleEl.replaceWith(input);
        requestAnimationFrame(() => input.focus());

        async function commitRename() {
            const newTitle = (input.value || "").trim();
            input.removeEventListener("change", commitRename);
            input.removeEventListener("keydown", handleKey);
            input.removeEventListener("focusout", commitRename);
            if (newTitle && newTitle !== currentTitle) {
                try {
                    await request(`${CONVERSATIONS_ENDPOINT}/${convId}`, "PATCH", { title: newTitle });
                } catch (err) {
                    console.error("[Sidebar] Rename failed:", err);
                }
            }
            loadConversationList();
        }

        function cancelRename() {
            input.removeEventListener("change", commitRename);
            input.removeEventListener("keydown", handleKey);
            input.removeEventListener("focusout", commitRename);
            loadConversationList();
        }

        function handleKey(e) {
            if (e.key === "Enter") { e.preventDefault(); commitRename(); }
            if (e.key === "Escape") { e.preventDefault(); cancelRename(); }
        }

        input.addEventListener("keydown", handleKey);
        input.addEventListener("focusout", commitRename);
    }

    async function deleteConversation(convId) {
        const item = conversationList?.querySelector(`ui5-li-custom[data-conv-id="${convId}"]`);
        const convTitle = item?.querySelector(".conv-item-title")?.textContent?.trim() || "this conversation";

        const confirmed = await new Promise((resolve) => {
            const dialog = document.createElement("ui5-dialog");
            dialog.setAttribute("accessible-name", "Delete Conversation");

            const header = document.createElement("ui5-bar");
            header.setAttribute("slot", "header");
            header.setAttribute("design", "Header");
            const headerIcon = document.createElement("ui5-icon");
            headerIcon.setAttribute("slot", "startContent");
            headerIcon.setAttribute("name", "delete");
            header.appendChild(headerIcon);
            const headerTitle = document.createElement("ui5-title");
            headerTitle.setAttribute("slot", "startContent");
            headerTitle.setAttribute("level", "H5");
            headerTitle.textContent = "Delete Conversation";
            header.appendChild(headerTitle);
            dialog.appendChild(header);

            const content = document.createElement("div");
            content.style.padding = "1rem";

            const msg = document.createElement("div");
            msg.style.display = "block";
            const msgPrefix = document.createElement("ui5-text");
            msgPrefix.textContent = "Are you sure you want to permanently delete \"";
            msg.appendChild(msgPrefix);
            const msgTitle = document.createElement("ui5-text");
            msgTitle.textContent = convTitle;
            msgTitle.style.fontWeight = "bold";
            msg.appendChild(msgTitle);
            const msgSuffix = document.createElement("ui5-text");
            msgSuffix.textContent = "\"?";
            msg.appendChild(msgSuffix);
            content.appendChild(msg);

            const warn = document.createElement("ui5-text");
            warn.textContent = "You cannot recover this conversation later.";
            warn.style.display = "block";
            warn.style.marginTop = "0.75rem";
            content.appendChild(warn);

            dialog.appendChild(content);

            const footer = document.createElement("ui5-bar");
            footer.setAttribute("slot", "footer");
            footer.setAttribute("design", "Footer");

            const confirmBtn = document.createElement("ui5-button");
            confirmBtn.setAttribute("slot", "endContent");
            confirmBtn.setAttribute("design", "Negative");
            confirmBtn.textContent = "Delete";
            confirmBtn.addEventListener("click", () => { dialog.open = false; resolve(true); });
            footer.appendChild(confirmBtn);

            const cancelBtn = document.createElement("ui5-button");
            cancelBtn.setAttribute("slot", "endContent");
            cancelBtn.setAttribute("design", "Transparent");
            cancelBtn.textContent = "Cancel";
            cancelBtn.addEventListener("click", () => { dialog.open = false; resolve(false); });
            footer.appendChild(cancelBtn);

            dialog.appendChild(footer);
            document.body.appendChild(dialog);
            dialog.addEventListener("close", () => { document.body.removeChild(dialog); });
            dialog.open = true;
        });

        if (!confirmed) return;
        try {
            await request(`${CONVERSATIONS_ENDPOINT}/${convId}`, "DELETE");
            if (activeConversationId === convId) {
                startNewConversation();
            }
            loadConversationList();
        } catch (err) {
            console.error("[Sidebar] Delete failed:", err);
        }
    }

    // ── Share Conversation Dialog ────────────────────────────────────────────

    let _activeShareConvId = null;

    function _setReadOnlyMode(readOnly) {
        _isReadOnly = readOnly;
        const composer = document.querySelector(".composer");
        const composerNote = document.getElementById("composer-note");
        let banner = document.querySelector(".read-only-banner");

        if (readOnly) {
            if (composer) composer.style.display = "none";
            if (composerNote) composerNote.style.display = "none";
            if (!banner) {
                banner = document.createElement("div");
                banner.className = "read-only-banner";
                banner.innerHTML = `<ui5-icon name="hint"></ui5-icon><ui5-text>You have view-only access to this conversation</ui5-text>`;
                const container = document.querySelector(".chat-container");
                if (container) container.appendChild(banner);
            }
            banner.style.display = "";
        } else {
            if (composer) composer.style.display = "";
            if (composerNote) composerNote.style.display = "";
            if (banner) banner.style.display = "none";
        }
    }

    async function openShareDialog(convId) {
        _activeShareConvId = convId;
        const dialog = document.getElementById("share-conversation-dialog");
        const codeEl = document.getElementById("share-code-text");
        const linkInput = document.getElementById("share-link-input");
        const membersSection = document.getElementById("share-members-section");
        const membersList = document.getElementById("share-members-list");

        if (!dialog) return;

        // Reset
        codeEl.textContent = "...";
        linkInput.value = "";
        membersSection.hidden = true;
        membersList.innerHTML = "";

        dialog.open = true;

        // Load existing share or create one — single link per conversation
        try {
            const data = await getShares(convId);
            const permToggle = document.getElementById("share-permission-toggle");

            // Use existing share (first active one), or create a new one
            let share = data.shares?.[0];

            if (!share) {
                const selectedPerm = permToggle?.querySelector("[pressed]")?.getAttribute("data-permission") || "read";
                share = await createShare(convId, selectedPerm);
            }

            // Sync the toggle to reflect the share's current permission
            if (permToggle) {
                permToggle.querySelectorAll("ui5-segmented-button-item").forEach(item => {
                    if (item.getAttribute("data-permission") === share.permission) {
                        item.setAttribute("pressed", "");
                    } else {
                        item.removeAttribute("pressed");
                    }
                });
            }

            codeEl.textContent = share.share_code;
            const link = `${window.location.origin}/shared/${share.share_code}`;
            linkInput.value = link;

            // Show members if any
            if (data.members && data.members.length > 0) {
                membersSection.hidden = false;
                membersList.innerHTML = data.members.map(m => `
                    <div class="share-member-row" data-uid="${m.uid}">
                        <span class="share-member-name">${_escapeHtml(m.display_name || m.email || m.uid)}</span>
                        <ui5-select class="share-member-perm-select" data-uid="${m.uid}">
                            <ui5-option value="read" ${m.permission === "read" ? "selected" : ""}>View only</ui5-option>
                            <ui5-option value="write" ${m.permission === "write" ? "selected" : ""}>Can edit</ui5-option>
                        </ui5-select>
                        <ui5-button class="share-remove-member-btn" icon="decline" design="Transparent"
                            tooltip="Remove" data-uid="${m.uid}"></ui5-button>
                    </div>
                `).join("");

                // Wire per-member permission dropdowns
                membersList.querySelectorAll(".share-member-perm-select").forEach(sel => {
                    sel.addEventListener("change", async (e) => {
                        const uid = sel.getAttribute("data-uid");
                        const selectedOption = e.detail?.selectedOption || sel.selectedOption;
                        const newPerm = selectedOption?.getAttribute("value") || "read";
                        try {
                            const { updateMemberPermission } = await import("../../services/sharing.js");
                            await updateMemberPermission(convId, uid, newPerm);
                            const toast = document.getElementById("share-toast");
                            if (toast) { toast.textContent = `Permission updated to "${newPerm === "write" ? "Can edit" : "View only"}"`; toast.open = true; }
                        } catch (err) {
                            console.error("[Share] Failed to update member permission:", err);
                            const toast = document.getElementById("share-toast");
                            if (toast) { toast.textContent = "Failed to update permission"; toast.open = true; }
                        }
                    });
                });

                // Wire remove buttons
                membersList.querySelectorAll(".share-remove-member-btn").forEach(btn => {
                    btn.addEventListener("click", async () => {
                        const uid = btn.getAttribute("data-uid");
                        try {
                            await removeMember(convId, uid);
                            btn.closest(".share-member-row")?.remove();
                            const remaining = membersList.querySelectorAll(".share-member-row");
                            if (remaining.length === 0) membersSection.hidden = true;
                            const toast = document.getElementById("share-toast");
                            if (toast) { toast.textContent = "Member removed"; toast.open = true; }
                        } catch (err) {
                            console.error("[Share] Failed to remove member:", err);
                            const toast = document.getElementById("share-toast");
                            if (toast) { toast.textContent = "Failed to remove member"; toast.open = true; }
                        }
                    });
                });
            }
        } catch (err) {
            console.error("[Share] Failed to load/create share:", err);
            codeEl.textContent = "Error";
        }
    }

    function _escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str || "";
        return div.innerHTML;
    }

    // Share dialog: close button
    document.getElementById("share-dialog-close-btn")?.addEventListener("click", () => {
        const dialog = document.getElementById("share-conversation-dialog");
        if (dialog) dialog.open = false;
    });

    // Share dialog: copy code
    document.getElementById("share-copy-code-btn")?.addEventListener("click", () => {
        const code = document.getElementById("share-code-text")?.textContent;
        if (code && code !== "...") {
            navigator.clipboard.writeText(code);
            const toast = document.getElementById("share-toast");
            if (toast) { toast.textContent = "Code copied!"; toast.open = true; }
        }
    });

    // Share dialog: copy link
    document.getElementById("share-copy-link-btn")?.addEventListener("click", () => {
        const link = document.getElementById("share-link-input")?.value;
        if (link) {
            navigator.clipboard.writeText(link);
            const toast = document.getElementById("share-toast");
            if (toast) { toast.textContent = "Link copied!"; toast.open = true; }
        }
    });

    // Share dialog: permission toggle change — update existing share's permission
    document.getElementById("share-permission-toggle")?.addEventListener("selection-change", async (e) => {
        if (!_activeShareConvId) return;
        const selectedItem = e.detail?.selectedItem || e.detail?.targetItem;
        const permission = selectedItem?.getAttribute("data-permission") || "read";

        try {
            const { updateSharePermission } = await import("../../services/sharing.js");
            await updateSharePermission(_activeShareConvId, permission);
            const toast = document.getElementById("share-toast");
            if (toast) { toast.textContent = `New members will get "${permission === "write" ? "Can edit" : "View only"}" access`; toast.open = true; }
        } catch (err) {
            console.error("[Share] Permission toggle error:", err);
            const toast = document.getElementById("share-toast");
            if (toast) { toast.textContent = "Failed to update permission"; toast.open = true; }
        }
    });

    // Share dialog: stop sharing (revoke all share links)
    document.getElementById("share-revoke-btn")?.addEventListener("click", async () => {
        if (!_activeShareConvId) return;
        try {
            const data = await getShares(_activeShareConvId);
            // Revoke all active share links
            for (const share of (data.shares || [])) {
                await revokeShare(_activeShareConvId, share.share_id);
            }
            // Close dialog and show toast
            const dialog = document.getElementById("share-conversation-dialog");
            if (dialog) dialog.open = false;
            const toast = document.getElementById("share-toast");
            if (toast) { toast.textContent = "Sharing stopped — link is no longer valid"; toast.open = true; }
        } catch (err) {
            console.error("[Share] Failed to revoke shares:", err);
            const toast = document.getElementById("share-toast");
            if (toast) { toast.textContent = "Failed to stop sharing"; toast.open = true; }
        }
    });

    // ── Wire sidebar events ─────────────────────────────────────────────────
    if (conversationList) {
        conversationList.addEventListener("selection-change", (e) => {
            const selected = e.detail?.selectedItems?.[0] || e.detail?.targetItem;
            const convId = selected?.getAttribute("data-conv-id");
            if (convId) selectConversation(convId);
        });
    }
    if (newConversationBtn) {
        newConversationBtn.addEventListener("click", startNewConversation);
    }
    if (contextMenu) {
        contextMenu.addEventListener("item-click", (e) => {
            const clickedItem = e.detail?.item;
            const text = clickedItem?.getAttribute("text") || clickedItem?.text;
            if (text === "Rename" && contextMenuTargetId) {
                startInlineRename(contextMenuTargetId);
            } else if (text === "Delete" && contextMenuTargetId) {
                deleteConversation(contextMenuTargetId);
            } else if (text === "Share" && contextMenuTargetId) {
                openShareDialog(contextMenuTargetId);
            }
            contextMenuTargetId = null;
        });
    }

    // Load conversation list on page init, then restore from URL if present
    loadConversationList().then(() => {
        const params = new URLSearchParams(window.location.search);
        const resumeId = params.get("c");
        if (resumeId) {
            selectConversation(resumeId);
        }
    });

    if (companyInput && !new URLSearchParams(window.location.search).get("c")) {
        requestAnimationFrame(() => companyInput.focus());
    }
}
