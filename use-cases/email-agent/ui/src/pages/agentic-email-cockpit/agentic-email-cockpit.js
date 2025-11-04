/* UI5 components used in this page */
import "@ui5/webcomponents-fiori/dist/FlexibleColumnLayout.js";
import "@ui5/webcomponents/dist/List.js";
import "@ui5/webcomponents/dist/ListItem.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Toast.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/Bar.js";
import "@ui5/webcomponents/dist/Avatar.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/illustrations/NoData.js";

/* Icons */
import "@ui5/webcomponents-icons/dist/inbox.js";
import "@ui5/webcomponents-icons/dist/email.js";
import "@ui5/webcomponents-icons/dist/tag.js";
import "@ui5/webcomponents-icons/dist/slim-arrow-right.js";
import "@ui5/webcomponents-icons/dist/activate.js";
import "@ui5/webcomponents-icons/dist/reset.js";
import "@ui5/webcomponents-icons/dist/ai.js";

import { request, API_BASE_URL, API_KEY } from "../../services/api.js";
import { marked } from "marked";
async function fetchEmailFiles() {
  return await request(`/api/email-files/`);
}
async function fetchEmailMetadata(path) {
  const qs = `?path=${encodeURIComponent(path)}`;
  return await request(`/api/email-files/metadata${qs}`);
}
async function fetchEmailContent(path) {
  const qs = `?path=${encodeURIComponent(path)}`;
  return await request(`/api/email-files/content${qs}`);
}

export default function initEmailCockpitPage() {
  // Ensure FCL has all three slots present before first render
  const fcl = document.getElementById("email-fcl");
  if (fcl) {
    // Explicitly set separators to avoid undefined internal state
    try {
      fcl.separators = "All"; // valid enum: None | Begin | End | Both | All
    } catch {}
  }
  const tagsList = document.getElementById("tags-list");
  const emailsList = document.getElementById("emails-list");
  const emailsTitle = document.getElementById("emails-title");
  const emailsCount = document.getElementById("emails-count");
  const subjectEl = document.getElementById("email-subject");
  const agentResultDiv = document.getElementById("agent-result");
  const fromEl = document.getElementById("email-from");
  const dateEl = document.getElementById("email-date");
  const filenameEl = document.getElementById("email-filename");
  const attachmentsEl = document.getElementById("email-attachments");
  const emailHtmlFrame = document.getElementById("email-html");
  const emailTextEl = document.getElementById("email-text");
  const previewEmpty = document.getElementById("preview-empty");
  const previewContainer = document.getElementById("preview-container");
  const statusTitleEl = document.getElementById("email-status-title");
  const emailSelect = document.getElementById("email-select");
  const runAgentBtn = document.getElementById("run-agent");
  const toast = document.getElementById("action-toast");
  const globalBusy = document.getElementById("global-busy");
  const resetBtn = document.getElementById("reset-btn");

  // Client-only state
  let emailsState = [];
  let currentFolder = "Inbox";
  let currentEmailPath = null;

  function updateCurrentEmailPath(email) {
    currentEmailPath = email?.path || null;
  }

  // Always show reset (client-only)
  if (resetBtn) resetBtn.style.display = "inline-block";

  function clearAgentResult() {
    if (agentResultDiv) agentResultDiv.innerHTML = "";
  }

  function renderAgentResult(agent) {
    if (!agentResultDiv) return;
    if (!agent) {
      // Show inline CTA to run agent for the current email
      agentResultDiv.innerHTML = `
        <ui5-card class="stack-panel">
          <div style="padding: 1rem">
            <ui5-panel header-text="Agent Result">
              <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap: wrap">
                <ui5-label>No agent result yet for this email.</ui5-label>
                <ui5-button id="run-agent-inline" design="Emphasized" icon="ai">Run Agent</ui5-button>
              </div>
            </ui5-panel>
          </div>
        </ui5-card>
      `;
      const inlineBtn = agentResultDiv.querySelector("#run-agent-inline");
      inlineBtn?.addEventListener("click", async () => {
        if (!currentEmailPath) return;
        try {
          if (globalBusy) globalBusy.style.display = "flex";
          inlineBtn.disabled = true;
          const data = await request(`/api/automation/run`, "POST", { email_path: currentEmailPath });
          const jsonStr = data?.text || "";
          let parsed = {};
          try {
            parsed = JSON.parse(jsonStr || "{}");
          } catch {
            parsed = {};
          }
          const moveToFolder = parsed?.moveToFolder || "Inbox";
          // Update email state
          emailsState = emailsState.map((e) => (e.path === currentEmailPath ? { ...e, folder: moveToFolder, agentResult: parsed } : e));
          // Refresh folders, list, and preview
          setFolders();
          renderEmails(listForFolder(currentFolder));
          const selected = emailsState.find((e) => e.path === currentEmailPath);
          if (selected) renderPreview(selected);
          if (toast) {
            toast.textContent = `Moved to ${moveToFolder}`;
            toast.open = true;
          }
        } catch (e) {
          console.error("Inline automation failed:", e);
          if (toast) {
            toast.textContent = "Automation failed. Please try again.";
            toast.open = true;
          }
        } finally {
          inlineBtn.disabled = false;
          if (globalBusy) globalBusy.style.display = "none";
        }
      });
      return;
    }
    const replySubject = agent.replySubject || "";
    const replyBody = agent.replyBody || "";
    const decision = agent.decision || "";
    const moveToFolder = agent.moveToFolder || "";
    const needsHumanReview = agent.needsHumanReview;
    const prettyJson = (() => {
      try {
        return JSON.stringify(agent, null, 2);
      } catch (_) {
        return String(agent);
      }
    })();

    const yesNo = (v) => (v === true ? "Yes" : v === false ? "No" : String(v ?? ""));
    const bodyHtml = replyBody ? marked.parse(replyBody) : "(no reply body)";

    agentResultDiv.innerHTML = `
      <ui5-card class="stack-panel">
        <div style="padding: 1rem">
          <ui5-panel header-text="Agent Result">
            <div class="email-body">
              <ui5-label for="agent-reply-subject">Reply Subject</ui5-label>
              <ui5-input id="agent-reply-subject" style="width:100%" placeholder="Subject"></ui5-input>
              <div style="height: 0.5rem"></div>
              <ui5-label for="agent-reply-body">Reply Body (Markdown)</ui5-label>
              <ui5-textarea id="agent-reply-body" growing growing-max-rows="16" rows="10" style="width:100%" placeholder="Write the reply..."></ui5-textarea>
            </div>
          </ui5-panel>
          <ui5-panel header-text="Details" class="stack-panel">
            <table class="meta-table">
              <tbody>
                <tr><th style="text-align: left;">Decision</th><td>${escapeHtml(decision)}</td></tr>
                <tr><th style="text-align: left;">Move To Folder</th><td>${escapeHtml(moveToFolder)}</td></tr>
                <tr><th style="text-align: left;">Needs Human Review</th><td>${yesNo(needsHumanReview)}</td></tr>
              </tbody>
            </table>
          </ui5-panel>
          <ui5-panel header-text="Agent JSON" class="stack-panel json-panel" collapsed>
            <pre style="margin:0"><code>${escapeHtml(prettyJson)}</code></pre>
          </ui5-panel>
        </div>
      </ui5-card>
    `;

    // Bind editors to update state and JSON panel live
    const subjInput = agentResultDiv.querySelector("#agent-reply-subject");
    const bodyInput = agentResultDiv.querySelector("#agent-reply-body");
    // Initialize field values programmatically to preserve newlines and special chars
    if (subjInput) {
      try {
        subjInput.value = replySubject || "";
      } catch {}
    }
    if (bodyInput) {
      try {
        bodyInput.value = replyBody || "";
      } catch {}
    }
    const jsonCode = agentResultDiv.querySelector(".json-panel code");
    const updateJson = () => {
      try {
        if (jsonCode) jsonCode.textContent = JSON.stringify(agent, null, 2);
      } catch (_) {}
    };
    subjInput?.addEventListener("input", (e) => {
      const val = e?.target?.value ?? "";
      agent.replySubject = val;
      // update selected email state
      if (currentEmailPath) {
        emailsState = emailsState.map((em) => (em.path === currentEmailPath ? { ...em, agentResult: agent } : em));
      }
      updateJson();
    });
    bodyInput?.addEventListener("input", (e) => {
      const val = e?.target?.value ?? "";
      agent.replyBody = val;
      if (currentEmailPath) {
        emailsState = emailsState.map((em) => (em.path === currentEmailPath ? { ...em, agentResult: agent } : em));
      }
      updateJson();
    });
  }

  function clearPreview() {
    subjectEl.textContent = "";
    if (previewContainer) previewContainer.style.display = "none";
    if (previewEmpty) previewEmpty.style.display = "block";
    if (statusTitleEl) statusTitleEl.style.display = "none";
    clearAgentResult();
    // Clear current email path
    updateCurrentEmailPath(null);
  }

  function renderEmails(emails) {
    emailsList.innerHTML = "";
    // Update title and counter in header bar
    const titleText = currentFolder || "Inbox";
    if (emailsTitle) emailsTitle.textContent = titleText;
    if (emailsCount) emailsCount.textContent = String(emails.length);

    emails.forEach((e) => {
      const li = document.createElement("ui5-li");
      li.setAttribute("type", "Active");
      li.setAttribute("wrapping-type", "Normal");
      li.textContent = e.subject || e.name || e.path || "(unknown)";
      // Description: from + date if available
      const fromDesc = e.from || "";
      const dateDesc = e.date || "";
      const combinedDesc = [fromDesc, dateDesc].filter(Boolean).join("\n");
      if (combinedDesc) li.setAttribute("description", combinedDesc);
      li.dataset.path = e.path;
      emailsList.appendChild(li);
    });

    // Do not auto-select any email
    try {
      emailsList.selectedItems = [];
    } catch {}
    clearPreview();
  }

  function renderPreview(email) {
    if (!email) {
      clearPreview();
      return;
    }
    if (previewEmpty) previewEmpty.style.display = "none";
    if (previewContainer) previewContainer.style.display = "block";

    // Update current email ID
    updateCurrentEmailPath(email);
    subjectEl.textContent = email.subject || email.name || email.path || "";
    // Render original email meta and body
    fromEl.textContent = email.from || "";
    dateEl.textContent = email.date || "";
    if (filenameEl) filenameEl.textContent = email.name || email.path || "";
    // Attachments
    attachmentsEl.innerHTML = "";
    const names = email.attachment_filenames || [];
    if (Array.isArray(names) && names.length > 0) {
      const cap = document.createElement("ui5-label");
      cap.textContent = "Attachments:";
      cap.style.marginRight = "0.25rem";
      attachmentsEl.appendChild(cap);
      names.forEach((n) => {
        const tag = document.createElement("ui5-tag");
        tag.setAttribute("design", "Set1");
        tag.setAttribute("color-scheme", "5");
        tag.textContent = n;
        tag.style.cursor = "pointer";
        tag.addEventListener("click", async (ev) => {
          ev.preventDefault();
          const url = `${API_BASE_URL}/api/email-files/attachment?path=${encodeURIComponent(email.path)}&filename=${encodeURIComponent(n)}`;
          try {
            const resp = await fetch(url, {
              method: "GET",
              headers: { "X-API-Key": API_KEY || "" }
            });
            if (!resp.ok) {
              throw new Error(`HTTP ${resp.status}`);
            }
            const blob = await resp.blob();
            const objectUrl = URL.createObjectURL(blob);
            window.open(objectUrl, "_blank", "noopener,noreferrer");
            // Best-effort revoke later
            setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000);
          } catch (e) {
            console.error("Failed to open attachment:", e);
          }
        });
        attachmentsEl.appendChild(tag);
        attachmentsEl.appendChild(document.createTextNode(" "));
      });
    }
    // Body: prefer HTML in iframe; fallback to text
    if (emailHtmlFrame) {
      const doc = emailHtmlFrame.contentDocument || emailHtmlFrame.contentWindow?.document;
      if (doc) {
        doc.open();
        const html = email.bodyHtml || (email.bodyText ? `<pre>${escapeHtml(email.bodyText)}</pre>` : "");
        doc.write(html || "");
        doc.close();
      }
    }
    if (emailTextEl) {
      if (!email.bodyHtml && email.bodyText) {
        emailTextEl.style.display = "block";
        emailTextEl.textContent = email.bodyText;
      } else {
        emailTextEl.style.display = "none";
        emailTextEl.textContent = "";
      }
    }
    // Show prior agent result if any
    renderAgentResult(email.agentResult || null);
  }

  function escapeHtml(str) {
    return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  function setFolders() {
    // Determine unique folders in state
    const folders = new Map();
    // Ensure Inbox exists
    folders.set("Inbox", 0);
    emailsState.forEach((e) => {
      const folder = e.folder || "Inbox";
      folders.set(folder, (folders.get(folder) || 0) + 1);
    });

    // Rebuild list
    const inbox = tagsList?.querySelector("ui5-li[data-tag='']");
    tagsList.innerHTML = "";
    if (inbox) tagsList.appendChild(inbox);
    // Update Inbox count
    if (inbox) inbox.setAttribute("additional-text", String(folders.get("Inbox") || 0));

    // Add other folders
    for (const [name, count] of folders.entries()) {
      if (name === "Inbox") continue;
      const li = document.createElement("ui5-li");
      li.setAttribute("type", "Active");
      li.setAttribute("icon", "tag");
      li.textContent = name;
      li.dataset.tag = name;
      li.setAttribute("additional-text", String(count || 0));
      tagsList.appendChild(li);
    }

    // Maintain selection
    const items = Array.from(tagsList.querySelectorAll("ui5-li"));
    const toSelect = items.find((el) => (el.dataset.tag || "") === (currentFolder === "Inbox" ? "" : currentFolder));
    if (toSelect) {
      try {
        tagsList.selectedItems = [toSelect];
      } catch {}
    }
  }

  function listForFolder(folder) {
    const f = folder || currentFolder || "Inbox";
    return emailsState.filter((e) => (e.folder || "Inbox") === f);
  }

  // Removed priority/status helpers (not used in file-based flow)

  // Events
  resetBtn?.addEventListener("click", () => {
    try {
      if (globalBusy) globalBusy.style.display = "flex";
      // Reset all emails back to Inbox and clear agent results
      emailsState = emailsState.map((e) => ({ ...e, folder: "Inbox", agentResult: null }));
      currentFolder = "Inbox";
      setFolders();
      renderEmails(listForFolder(currentFolder));
      clearPreview();
      if (toast) {
        toast.textContent = "Reset to Inbox";
        toast.open = true;
      }
    } finally {
      if (globalBusy) globalBusy.style.display = "none";
    }
  });

  tagsList?.addEventListener("item-click", (ev) => {
    const li = ev.detail.item;
    const tag = li?.dataset?.tag || "";
    currentFolder = tag || "Inbox";
    try {
      tagsList.selectedItems = [li];
    } catch {}
    const list = listForFolder(currentFolder);
    renderEmails(list);
  });

  emailsList?.addEventListener("item-click", async (ev) => {
    const li = ev.detail.item;
    const path = li?.dataset?.path;
    let email = emailsState.find((e) => e.path === path);
    // If metadata/content missing, fetch and patch state
    try {
      if (email && (!email.subject || !email.from || !email.date)) {
        const md = await fetchEmailMetadata(path);
        email = Object.assign(email, {
          subject: md?.subject,
          from: md?.from,
          date: md?.date,
          attachment_filenames: md?.attachment_filenames || email.attachment_filenames
        });
      }
      if (email && !email.bodyHtml && !email.bodyText) {
        const content = await fetchEmailContent(path);
        email = Object.assign(email, {
          bodyHtml: content?.bodyHtml,
          bodyText: content?.bodyText
        });
      }
    } catch (e) {
      console.error("Failed to load email detail:", e);
    }
    renderPreview(email);
  });

  runAgentBtn?.addEventListener("click", async () => {
    try {
      const emailPath = emailSelect?.selectedOption?.value || null;
      if (!emailPath) return;
      if (globalBusy) globalBusy.style.display = "flex";
      runAgentBtn.disabled = true;
      const data = await request(`/api/automation/run`, "POST", { email_path: emailPath });
      const jsonStr = data?.text || "";
      let parsed = null;
      try {
        parsed = JSON.parse(jsonStr);
      } catch (_) {
        parsed = {};
      }
      const moveToFolder = parsed?.moveToFolder || "Inbox";
      // Update email state
      emailsState = emailsState.map((e) => (e.path === emailPath ? { ...e, folder: moveToFolder, agentResult: parsed } : e));
      // Refresh folders and list
      setFolders();
      const list = listForFolder(currentFolder);
      renderEmails(list);
      // If the processed email is currently selected in preview, update the result
      const selected = emailsState.find((e) => e.path === currentEmailPath);
      if (selected && selected.path === emailPath) {
        renderPreview(selected);
      }
      if (toast) {
        toast.textContent = `Moved to ${moveToFolder}`;
        toast.open = true;
      }
    } catch (e) {
      console.error("Automation failed:", e);
      if (toast) {
        toast.textContent = "Automation failed. Please try again.";
        toast.open = true;
      }
    } finally {
      runAgentBtn.disabled = false;
      if (globalBusy) globalBusy.style.display = "none";
    }
  });

  // Initial load
  (async () => {
    try {
      const files = await fetchEmailFiles();
      // Build state (filenames only first)
      emailsState = (files || []).map((f) => ({ id: f?.path || f?.name, name: f?.name || f?.path, path: f?.path || "", folder: "Inbox", agentResult: null }));
      // Prefetch basic metadata for list (best-effort, sequential to limit load)
      for (const e of emailsState) {
        try {
          const md = await fetchEmailMetadata(e.path);
          e.subject = md?.subject || e.name;
          e.from = md?.from || "";
          e.date = md?.date || "";
          e.attachment_filenames = md?.attachment_filenames || [];
        } catch (_) {
          // ignore failures; list will fall back to filename
        }
      }
      // Populate email-select
      if (emailSelect) {
        emailSelect.innerHTML = "";
        emailsState.forEach((e) => {
          const opt = document.createElement("ui5-option");
          opt.value = e.path;
          opt.textContent = e.name || e.path;
          emailSelect.appendChild(opt);
        });
        if (emailSelect.children.length > 0) {
          emailSelect.selectedIndex = 0;
        }
      }
      // Render folders and list
      setFolders();
      renderEmails(listForFolder(currentFolder));
      // Ensure Inbox is selected in folders list
      const inboxItem = tagsList?.querySelector("ui5-li[data-tag='']");
      if (inboxItem && tagsList) {
        try {
          tagsList.selectedItems = [inboxItem];
        } catch {}
      }
      if (emailsTitle) emailsTitle.textContent = "Inbox";
      if (emailsCount) emailsCount.textContent = String(listForFolder("Inbox").length);
    } catch (e) {
      console.error("Failed to load email files:", e);
    }
  })();
}
