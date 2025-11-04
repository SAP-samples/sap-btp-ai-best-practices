import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Bar.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/Timeline.js";
import "@ui5/webcomponents-fiori/dist/TimelineItem.js";
import { marked } from "marked";
import { request, API_BASE_URL, API_KEY } from "../../services/api.js";
import {
  systemFromToolName,
  extractEntitiesFromArgs,
  summarizeToolCalls,
  tryParseJson,
  tryExtractJsonFromText,
  countRecordsInPayload,
  summarizeToolResult,
  iconForType,
  titleForType,
  stateForType,
  toDisplay,
  humanSubtitleFromText,
  assistantSubtitleFromText
} from "./timeline-summarizer.js";

// Helpers moved to ./timeline-summarizer.js

function renderTimelineFromMessages(timeline, messages) {
  if (!timeline) return;
  if (!Array.isArray(messages)) return;

  // Helper functions imported from summarizer module

  // Keep a queue of pending tool calls to associate with following Tool messages
  const pendingToolCalls = [];

  messages.forEach((m, idx) => {
    const { text, calls } = toDisplay(m?.content, m?.tool_calls);
    const item = document.createElement("ui5-timeline-item");
    item.setAttribute("title-text", titleForType(m?.type));
    item.setAttribute("icon", iconForType(m?.type));
    item.setAttribute("name", `Step ${idx + 1}`);
    const state = stateForType(m?.type);
    if (state) {
      item.setAttribute("state", state);
    }
    if (calls.length > 0) {
      // Track calls for pairing with subsequent Tool messages
      calls.forEach((c) => pendingToolCalls.push(c));
      const summary = summarizeToolCalls(calls);
      const subtitle = summary ? `Tool Calls (${calls.length}): ${summary}` : `Tool Calls (${calls.length})`;
      item.setAttribute("subtitle-text", subtitle);
    } else if (m?.type?.startsWith?.("Tool")) {
      // Pair this Tool message with the next pending tool call when available
      const related = pendingToolCalls.length > 0 ? pendingToolCalls.shift() : null;
      const subtitle = summarizeToolResult(related, m);
      if (subtitle) item.setAttribute("subtitle-text", subtitle);
    } else if (m?.type === "HumanMessage") {
      const subtitle = humanSubtitleFromText(text || "");
      item.setAttribute("subtitle-text", subtitle);
    } else if (m?.type === "AIMessage") {
      const subtitle = assistantSubtitleFromText(text || "");
      if (subtitle) item.setAttribute("subtitle-text", subtitle);
    }
    const textEl = document.createElement("ui5-text");
    textEl.style.whiteSpace = "pre-wrap";
    textEl.style.margin = "0.25rem 0 0";
    textEl.textContent = text || "";
    item.appendChild(textEl);
    if (calls.length > 0) {
      calls.forEach((tc) => {
        const block = document.createElement("ui5-text");
        block.style.whiteSpace = "pre-wrap";
        block.style.margin = "0.1rem 0 0.25rem";
        block.textContent = `${tc?.name || "tool"} ${JSON.stringify(tc?.args || {}, null, 2)}`;
        item.appendChild(block);
      });
    }
    timeline.appendChild(item);
  });
}

export default function initEmailAgentPage() {
  const runBtn = document.getElementById("run-email-agent");
  const resultDiv = document.getElementById("email-agent-result");
  const loadingRow = document.getElementById("email-agent-loading");
  const busy = document.getElementById("email-agent-busy");
  const emailSelect = document.getElementById("email-select");
  const timeline = document.getElementById("email-agent-timeline");
  const logPanel = document.getElementById("email-agent-log");
  const grid = document.getElementById("email-agent-grid");
  const empty = document.getElementById("email-agent-empty");

  // Preview elements
  const originalCard = document.getElementById("original-email-card");
  const previewSubject = document.getElementById("preview-subject");
  const previewFrom = document.getElementById("preview-from");
  const previewDate = document.getElementById("preview-date");
  const previewAttachments = document.getElementById("preview-attachments");
  const previewHtmlFrame = document.getElementById("preview-html");
  const previewText = document.getElementById("preview-text");
  const previewHeader = document.getElementById("preview-header");

  async function fetchEmailMetadata(path) {
    const qs = `?path=${encodeURIComponent(path)}`;
    return await request(`/api/email-files/metadata${qs}`);
  }

  async function fetchEmailContent(path) {
    const qs = `?path=${encodeURIComponent(path)}`;
    return await request(`/api/email-files/content${qs}`);
  }

  function clearPreview() {
    if (originalCard) originalCard.style.display = "none";
    if (previewHeader) {
      previewHeader.setAttribute("title-text", "");
      previewHeader.removeAttribute("subtitle-text");
      previewHeader.removeAttribute("additional-text");
    }
    if (previewAttachments) previewAttachments.innerHTML = "";
    if (previewHtmlFrame) {
      const doc = previewHtmlFrame.contentDocument || previewHtmlFrame.contentWindow?.document;
      if (doc) {
        doc.open();
        doc.write("");
        doc.close();
      }
    }
    if (previewText) {
      previewText.style.display = "none";
      previewText.textContent = "";
    }
  }

  function renderPreview(email, content) {
    if (!originalCard) return;
    originalCard.style.display = "block";
    if (previewHeader) {
      const subject = email?.subject || email?.name || email?.path || "";
      const from = email?.from || "";
      const date = email?.date || "";
      previewHeader.setAttribute("title-text", subject);
      previewHeader.setAttribute("subtitle-text", from);
      if (date) previewHeader.setAttribute("additional-text", date);
      else previewHeader.removeAttribute("additional-text");
    }

    // Attachments (as tags only, no label)
    if (previewAttachments) {
      previewAttachments.innerHTML = "";
      const names = email?.attachment_filenames || [];
      if (Array.isArray(names) && names.length > 0) {
        names.forEach((n, idx) => {
          const tag = document.createElement("ui5-tag");
          tag.setAttribute("design", "Set1");
          tag.setAttribute("color-scheme", "5");
          tag.textContent = n;
          tag.style.cursor = "pointer";
          tag.addEventListener("click", async (ev) => {
            ev.preventDefault();
            const url = `${API_BASE_URL}/api/email-files/attachment?path=${encodeURIComponent(email.path)}&filename=${encodeURIComponent(n)}`;
            try {
              const resp = await fetch(url, { method: "GET", headers: { "X-API-Key": API_KEY || "" } });
              if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
              const blob = await resp.blob();
              const objectUrl = URL.createObjectURL(blob);
              window.open(objectUrl, "_blank", "noopener,noreferrer");
              setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);
            } catch (e) {
              console.error("Failed to open attachment:", e);
            }
          });
          previewAttachments.appendChild(tag);
          if (idx < names.length - 1) {
            previewAttachments.appendChild(document.createTextNode(" "));
          }
        });
      }
    }

    const bodyHtml = content?.bodyHtml;
    const bodyText = content?.bodyText;
    if (previewHtmlFrame) {
      const doc = previewHtmlFrame.contentDocument || previewHtmlFrame.contentWindow?.document;
      if (doc) {
        doc.open();
        const html = bodyHtml || (bodyText ? `<pre>${escapeHtml(bodyText)}</pre>` : "");
        doc.write(html || "");
        doc.close();
      }
    }
    if (previewText) {
      if (!bodyHtml && bodyText) {
        previewText.style.display = "block";
        previewText.textContent = bodyText;
      } else {
        previewText.style.display = "none";
        previewText.textContent = "";
      }
    }
  }

  function escapeHtml(str) {
    return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  async function loadEmails() {
    if (!emailSelect) return;
    emailSelect.innerHTML = "";
    try {
      const files = await request(`/api/email-files/`);
      const options = files.map((f) => {
        const o = document.createElement("ui5-option");
        // Use absolute path as value; show filename and size
        o.value = f?.path || "";
        const name = f?.name || "(unknown)";
        o.textContent = name;
        return o;
      });
      options.forEach((o) => emailSelect.appendChild(o));
      if (options.length === 0) {
        const empty = document.createElement("ui5-option");
        empty.textContent = "No emails found";
        empty.disabled = true;
        emailSelect.appendChild(empty);
      }
      // Select first by default
      if (emailSelect.children.length > 0) {
        emailSelect.selectedIndex = 0;
      }
    } catch (e) {
      // In case of error, leave select empty
      console.error("Failed to load emails:", e);
      const errOpt = document.createElement("ui5-option");
      errOpt.textContent = "Failed to load emails";
      errOpt.disabled = true;
      emailSelect.appendChild(errOpt);
    }
  }

  async function runAutomation() {
    if (!runBtn || !resultDiv) return;
    runBtn.disabled = true;
    if (loadingRow) loadingRow.style.display = "block";
    if (busy) busy.active = true;
    resultDiv.innerHTML = "";
    if (timeline) timeline.innerHTML = "";
    if (logPanel) logPanel.style.display = "none";
    try {
      // Use selected .msg file path directly (from selectedOption.value)
      const emailPath = emailSelect?.selectedOption?.value || null;
      console.log("emailPath", emailPath);
      const data = await request(`/api/automation/run`, "POST", { email_path: emailPath });

      // Pretty-print JSON returned by backend (string) and parse for fields
      const jsonStr = data?.text || "";
      let prettyJson = "";
      let parsed = null;
      try {
        parsed = JSON.parse(jsonStr);
        prettyJson = JSON.stringify(parsed, null, 2);
      } catch (_) {
        prettyJson = jsonStr || "No JSON returned";
      }

      const replySubject = parsed?.replySubject || "";
      const replyBody = parsed?.replyBody || "";
      const decision = parsed?.decision ?? "";
      const decisionSummary = parsed?.decisionSummary || "";
      const moveToFolder = parsed?.moveToFolder ?? "";
      const needsHumanReview = parsed?.needsHumanReview;

      // Render: Subject, Body, and collapsible JSON panel
      resultDiv.innerHTML = "";

      const subjTitle = document.createElement("ui5-title");
      subjTitle.setAttribute("level", "H5");
      subjTitle.className = "email-subject";
      subjTitle.textContent = `${replySubject || "(unknown)"}`;

      const bodyBlock = document.createElement("div");
      bodyBlock.className = "email-body";
      bodyBlock.innerHTML = marked.parse(replyBody || "(no reply body)");

      // Panel 1: Result Email (subject + body)
      const resultEmailPanel = document.createElement("ui5-panel");
      resultEmailPanel.setAttribute("header-text", "Result Email");
      resultEmailPanel.className = "stack-panel result-email-panel";
      resultEmailPanel.appendChild(subjTitle);
      resultEmailPanel.appendChild(bodyBlock);

      // Meta table for decision/moveToFolder/needsHumanReview
      const metaBlock = document.createElement("div");
      const yesNo = (v) => (v === true ? "Yes" : v === false ? "No" : String(v ?? ""));
      metaBlock.innerHTML = `
        <table class="meta-table">
          <tbody>
            <tr><th>Summary</th><td>${decisionSummary || ""}</td></tr>
            <tr><th>Decision</th><td>${decision || ""}</td></tr>
            <tr><th>Move To Folder</th><td>${moveToFolder || ""}</td></tr>
            <tr><th>Needs Human Review</th><td>${yesNo(needsHumanReview)}</td></tr>
          </tbody>
        </table>
      `;

      // Panel 2: Table (metadata)
      const tablePanel = document.createElement("ui5-panel");
      tablePanel.setAttribute("header-text", "Decision Details");
      tablePanel.className = "stack-panel table-panel";
      tablePanel.appendChild(metaBlock);

      const jsonPanel = document.createElement("ui5-panel");
      jsonPanel.setAttribute("header-text", "Agent JSON");
      jsonPanel.setAttribute("collapsed", "");
      jsonPanel.className = "stack-panel json-panel";
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = prettyJson;
      pre.appendChild(code);
      jsonPanel.appendChild(pre);

      resultDiv.appendChild(resultEmailPanel);
      resultDiv.appendChild(tablePanel);
      resultDiv.appendChild(jsonPanel);

      if (grid) grid.style.display = "grid";
      if (empty) empty.style.display = "none";

      // Render timeline from raw messages (type, content, tool_calls)
      const messages = data?.messages || [];
      if (Array.isArray(messages) && messages.length > 0) {
        if (logPanel) logPanel.style.display = "block";
        renderTimelineFromMessages(timeline, messages);
      } else {
        if (logPanel) logPanel.style.display = "none";
      }
    } catch (e) {
      resultDiv.textContent = `Failed: ${e.message}`;
    } finally {
      runBtn.disabled = false;
      if (loadingRow) loadingRow.style.display = "none";
      if (busy) busy.active = false;
    }
  }

  if (runBtn) runBtn.addEventListener("click", runAutomation);

  async function previewSelectedEmail() {
    try {
      // Hide previous agent results and timeline when switching emails
      if (resultDiv) resultDiv.innerHTML = "";
      if (timeline) timeline.innerHTML = "";
      if (logPanel) logPanel.style.display = "none";
      if (grid) grid.style.display = "none";
      if (empty) empty.style.display = "none";

      const emailPath = emailSelect?.selectedOption?.value || (emailSelect?.children && emailSelect.children[0] && emailSelect.children[0].value) || null;
      if (!emailPath) {
        clearPreview();
        return;
      }
      // fetch metadata and content
      const [md, content] = await Promise.all([fetchEmailMetadata(emailPath), fetchEmailContent(emailPath)]);
      const email = {
        subject: md?.subject,
        from: md?.from,
        date: md?.date,
        name:
          md?.name ||
          (emailSelect?.selectedOption && emailSelect.selectedOption.textContent) ||
          (emailSelect?.children && emailSelect.children[0] && emailSelect.children[0].textContent) ||
          emailPath,
        path: emailPath,
        attachment_filenames: md?.attachment_filenames || []
      };
      renderPreview(email, content);
      // Keep results grid hidden until agent runs again
      if (grid) grid.style.display = "none";
      if (empty) empty.style.display = "none";
    } catch (e) {
      console.error("Failed to load preview:", e);
      clearPreview();
    }
  }

  // Preview on select change
  emailSelect?.addEventListener("change", previewSelectedEmail);

  // Initial load: load emails then trigger preview for the first selection
  (async () => {
    try {
      await loadEmails();
      if (emailSelect) {
        if (emailSelect.children && emailSelect.children.length > 0 && (emailSelect.selectedIndex == null || emailSelect.selectedIndex < 0)) {
          emailSelect.selectedIndex = 0;
        }
        // Give UI5 a brief moment to reflect selectedOption, then preview (with fallbacks inside)
        setTimeout(() => {
          previewSelectedEmail();
        }, 75);
      }
    } catch (e) {
      // ignore
    }
  })();
}
