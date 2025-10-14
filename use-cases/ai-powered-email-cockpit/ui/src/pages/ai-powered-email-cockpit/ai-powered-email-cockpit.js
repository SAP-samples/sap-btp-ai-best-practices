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
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/illustrations/NoData.js";

/* Icons */
import "@ui5/webcomponents-icons/dist/inbox.js";
import "@ui5/webcomponents-icons/dist/email.js";
import "@ui5/webcomponents-icons/dist/tag.js";
import "@ui5/webcomponents-icons/dist/slim-arrow-right.js";
import "@ui5/webcomponents-icons/dist/activate.js";
import "@ui5/webcomponents-icons/dist/reset.js";

import { request } from "../../services/api.js";

async function fetchTags() {
  return await request("/api/emails/tags");
}

async function fetchEmails(tag = null) {
  const qs = tag ? `?tag=${encodeURIComponent(tag)}` : "";
  return await request(`/api/emails/${qs}`);
}

async function classifyAllEmails() {
  return await request(`/api/emails/classify`, "POST");
}

async function resetEmails() {
  return await request(`/api/emails/reset`, "POST");
}

async function generateSummary(messageId) {
  return await request(`/api/emails/summary/${messageId}`, "POST");
}

async function generateResponse(messageId) {
  return await request(`/api/emails/respond/${messageId}`, "POST");
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
  const fromEl = document.getElementById("email-from");
  const dateEl = document.getElementById("email-date");
  const tagsEl = document.getElementById("email-tags");
  const bodyEl = document.getElementById("email-body");
  const previewEmpty = document.getElementById("preview-empty");
  const previewContainer = document.getElementById("preview-container");
  const avatarEl = document.getElementById("email-avatar");
  const statusTitleEl = document.getElementById("email-status-title");
  const priorityLabelEl = document.getElementById("email-priority-label");
  const priorityTagEl = document.getElementById("email-priority-tag");
  const classifyBtn = document.getElementById("classify-btn");
  const toast = document.getElementById("action-toast");
  const globalBusy = document.getElementById("global-busy");
  const resetBtn = document.getElementById("reset-btn");
  const priorityFilter = document.getElementById("priority-filter");

  let currentTag = null;
  let currentPriority = "all";
  let emailsCache = [];
  let allEmailsCache = [];
  let filteredEmailsCache = [];
  let currentEmailId = null;

  function updateCurrentEmailId(email) {
    currentEmailId = email?.messageId || null;
  }

  // Function to check if emails are classified
  function areEmailsClassified(emails) {
    return emails.some((email) => (email.tags && email.tags.length > 0) || (email.status && email.status.trim() !== "") || (email.priority && email.priority.trim() !== ""));
  }

  // Function to update reset button visibility
  function updateResetButtonVisibility(emails) {
    if (resetBtn) {
      const isClassified = areEmailsClassified(emails || allEmailsCache);
      resetBtn.style.display = isClassified ? "inline-block" : "none";
    }
  }

  // Function to show/hide loading states and content
  function setLoadingState(type, isLoading) {
    const loadingEl = document.getElementById(`${type}-loading`);
    const contentEl = document.getElementById(`${type}-content`);
    const errorEl = document.getElementById(`${type}-error`);

    if (isLoading) {
      if (loadingEl) loadingEl.style.display = "block";
      if (contentEl) contentEl.style.display = "none";
      if (errorEl) errorEl.style.display = "none";
    } else {
      if (loadingEl) loadingEl.style.display = "none";
    }
  }

  function showContent(type, content) {
    setLoadingState(type, false);
    const contentEl = document.getElementById(`${type}-content`);
    const errorEl = document.getElementById(`${type}-error`);

    if (contentEl && content) {
      contentEl.innerHTML = escapeHtml(content);
      contentEl.style.display = "block";
    }
    if (errorEl) errorEl.style.display = "none";
  }

  function showError(type, errorMessage = null) {
    setLoadingState(type, false);
    const contentEl = document.getElementById(`${type}-content`);
    const errorEl = document.getElementById(`${type}-error`);

    if (contentEl) contentEl.style.display = "none";
    if (errorEl) {
      if (errorMessage) {
        errorEl.querySelector("ui5-text").textContent = errorMessage;
      }
      errorEl.style.display = "block";
    }
  }

  // Automatically generate summary and response for an email
  async function generateEmailSummaryAndResponse(messageId) {
    if (!messageId) return;

    // Start both operations in parallel
    const summaryPromise = generateEmailSummary(messageId);
    const responsePromise = generateEmailResponse(messageId);

    // Handle summary generation
    summaryPromise
      .then((result) => {
        if (result.success) {
          showContent("summary", result.summary);
        } else {
          showError("summary", "Failed to generate summary");
        }
      })
      .catch((error) => {
        console.error("Summary generation failed:", error);
        showError("summary", "Failed to generate summary");
      });

    // Handle response generation
    responsePromise
      .then((result) => {
        if (result.success) {
          showContent("response", result.response);
        } else {
          showError("response", "Failed to generate response");
        }
      })
      .catch((error) => {
        console.error("Response generation failed:", error);
        showError("response", "Failed to generate response");
      });
  }

  async function generateEmailSummary(messageId) {
    setLoadingState("summary", true);
    try {
      return await generateSummary(messageId);
    } catch (error) {
      showError("summary");
      throw error;
    }
  }

  async function generateEmailResponse(messageId) {
    setLoadingState("response", true);
    try {
      return await generateResponse(messageId);
    } catch (error) {
      showError("response");
      throw error;
    }
  }

  function clearPreview() {
    subjectEl.textContent = "";
    fromEl.textContent = "";
    dateEl.textContent = "";
    tagsEl.innerHTML = "";
    bodyEl.innerHTML = "";
    if (previewContainer) previewContainer.style.display = "none";
    if (previewEmpty) previewEmpty.style.display = "block";
    if (statusTitleEl) statusTitleEl.style.display = "none";
    if (priorityLabelEl) priorityLabelEl.style.display = "none";
    if (priorityTagEl) priorityTagEl.style.display = "none";

    // Clear AI containers
    ["summary", "response"].forEach((type) => {
      const loadingEl = document.getElementById(`${type}-loading`);
      const contentEl = document.getElementById(`${type}-content`);
      const errorEl = document.getElementById(`${type}-error`);

      if (loadingEl) loadingEl.style.display = "none";
      if (contentEl) {
        contentEl.style.display = "none";
        contentEl.innerHTML = "";
      }
      if (errorEl) errorEl.style.display = "none";
    });

    // Clear current email ID
    updateCurrentEmailId(null);
  }

  function formatDate(iso) {
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso || "";
    }
  }

  function renderEmails(emails) {
    emailsList.innerHTML = "";
    // Update title and counter in header bar
    const titleText = currentTag || "Inbox";
    if (emailsTitle) emailsTitle.textContent = titleText;
    if (emailsCount) emailsCount.textContent = String(emails.length);

    emails.forEach((e) => {
      const li = document.createElement("ui5-li");
      li.setAttribute("type", "Active");
      li.setAttribute("wrapping-type", "Normal");
      li.textContent = e.subject || "(no subject)";

      // Description (from/date info)
      const fromDesc = e.from?.name && e.from?.email ? `${e.from.name} <${e.from.email}>` : e.from?.name || e.from?.email || "";
      const dateDesc = e.sentDate ? formatDate(e.sentDate) : "";
      const combinedDesc = [fromDesc, dateDesc].filter(Boolean).join("\n");
      if (combinedDesc) {
        li.setAttribute("description", combinedDesc);
      }

      // Status and priority info
      const statusMeta = getStatusMeta(e);
      if (statusMeta) {
        // Show status and priority in additional text with clean formatting
        const priorityLabel = getPriorityLabel(e.priority);
        const additionalText = priorityLabel ? `${statusMeta.label} | ${priorityLabel}` : statusMeta.label;
        li.setAttribute("additional-text", additionalText);
        li.setAttribute("additional-text-state", statusMeta.state);
        li.setAttribute("highlight", statusMeta.state);
      }

      li.dataset.messageId = e.messageId;
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
    updateCurrentEmailId(email);

    // Automatically start generating summary and response
    generateEmailSummaryAndResponse(email.messageId);
    subjectEl.textContent = email.subject || "";
    const fromName = email.from?.name || "";
    const fromEmail = email.from?.email || "";
    fromEl.textContent = fromName && fromEmail ? `${fromName} <${fromEmail}>` : fromName || fromEmail;
    if (avatarEl) {
      // Use first letter of name as initials fallback
      const initial = (fromName || fromEmail || "?").trim().charAt(0).toUpperCase();
      avatarEl.textContent = initial;
    }
    dateEl.textContent = formatDate(email.sentDate);
    // Display status and priority separately
    const statusMeta = getStatusMeta(email);

    // Status tag
    if (statusTitleEl) {
      if (statusMeta) {
        statusTitleEl.style.display = "inline-block";
        statusTitleEl.setAttribute("design", statusMeta.tagDesign);
        statusTitleEl.textContent = statusMeta.statusLabel;
      } else {
        statusTitleEl.style.display = "none";
      }
    }

    // Priority label and tag
    if (priorityLabelEl && priorityTagEl) {
      if (statusMeta && statusMeta.priority) {
        priorityLabelEl.style.display = "inline-block";
        priorityTagEl.style.display = "inline-block";
        priorityTagEl.setAttribute("design", statusMeta.priority.design);
        priorityTagEl.setAttribute("color-scheme", statusMeta.priority.colorScheme);
        priorityTagEl.textContent = statusMeta.priority.label;
      } else {
        priorityLabelEl.style.display = "none";
        priorityTagEl.style.display = "none";
      }
    }
    // Tags
    tagsEl.innerHTML = "";
    (email.tags || []).forEach((t) => {
      const tag = document.createElement("ui5-tag");
      tag.setAttribute("design", "Set2");
      tag.setAttribute("color-scheme", "5");
      tag.textContent = t;
      tagsEl.appendChild(tag);
      const spacer = document.createTextNode(" ");
      tagsEl.appendChild(spacer);
    });
    // Body: prefer HTML
    const html = email.body?.html;
    const text = email.body?.text;
    bodyEl.innerHTML = html || (text ? `<pre>${escapeHtml(text)}</pre>` : "");
  }

  function escapeHtml(str) {
    return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  function setTags(tags) {
    // Keep Inbox
    const inbox = tagsList.querySelector("ui5-li[data-tag='']");
    tagsList.innerHTML = "";
    if (inbox) tagsList.appendChild(inbox);
    // Compute counts from all emails
    const inboxCount = allEmailsCache.length;
    if (inboxCount && inbox) {
      inbox.setAttribute("additional-text", String(inboxCount));
    }
    const tagCounts = new Map();
    allEmailsCache.forEach((e) => {
      (e.tags || []).forEach((t) => {
        tagCounts.set(t, (tagCounts.get(t) || 0) + 1);
      });
    });
    tags.forEach((t) => {
      const li = document.createElement("ui5-li");
      li.setAttribute("type", "Active");
      li.setAttribute("icon", "tag");
      li.textContent = t;
      li.dataset.tag = t;
      const count = tagCounts.get(t) || 0;
      li.setAttribute("additional-text", String(count));
      tagsList.appendChild(li);
    });
  }

  function filterEmailsByPriority(emails, priority) {
    if (priority === "all") return emails;
    return emails.filter((email) => String(email.priority || "").toLowerCase() === priority);
  }

  async function loadAndRender(tag = null, priority = "all") {
    const emails = await fetchEmails(tag);
    emailsCache = emails;
    filteredEmailsCache = filterEmailsByPriority(emails, priority);
    renderEmails(filteredEmailsCache);
    clearPreview();
    updateResetButtonVisibility();
  }

  function getStatusMeta(email) {
    if (!email) {
      return null;
    }

    // Get priority information
    const priority = String(email.priority || "").toLowerCase();
    const priorityInfo = getPriorityInfo(priority);

    const normalized = String(email.status || "").toLowerCase();
    let statusInfo;

    switch (normalized) {
      case "action-needed":
        statusInfo = { label: "Action Needed", state: "Negative", tagDesign: "Negative", icon: "status-negative" };
        break;
      case "waiting-for-response":
        statusInfo = { label: "Waiting for Response", state: "Critical", tagDesign: "Critical", icon: "status-critical" };
        break;
      case "resolved":
        statusInfo = { label: "Resolved", state: "Positive", tagDesign: "Positive", icon: "status-positive" };
        break;
      default:
        statusInfo = { label: "New", state: "Neutral", tagDesign: "Neutral", icon: "status-inactive" };
    }

    return {
      ...statusInfo,
      label: statusInfo.label, // Keep status simple in list
      statusLabel: statusInfo.label,
      priority: priorityInfo
    };
  }

  function getPriorityInfo(priority) {
    switch (priority) {
      case "critical":
        return { label: "Critical", state: "Negative", design: "Set1", colorScheme: "1" };
      case "high":
        return { label: "High", state: "Critical", design: "Set1", colorScheme: "2" };
      case "medium":
        return { label: "Medium", state: "Warning", design: "Set1", colorScheme: "3" };
      case "low":
        return { label: "Low", state: "Success", design: "Set1", colorScheme: "4" };
      default:
        return { label: "Normal", state: "Neutral", design: "Set1", colorScheme: "10" };
    }
  }

  function getPriorityLabel(priority) {
    // Return clean priority labels for list display
    switch (String(priority || "").toLowerCase()) {
      case "critical":
        return "Critical 🚩";
      case "high":
        return "High";
      case "medium":
        return "Medium";
      case "low":
        return "Low";
      default:
        return null; // Don't show priority for normal/unset
    }
  }

  // Events
  classifyBtn?.addEventListener("click", async () => {
    try {
      if (globalBusy) globalBusy.style.display = "flex";
      classifyBtn.disabled = true;
      await classifyAllEmails();
      // Refresh data (all emails and tags)
      const [tags, emails] = await Promise.all([fetchTags(), fetchEmails(currentTag)]);
      allEmailsCache = await fetchEmails();
      setTags(tags);
      emailsCache = emails;
      filteredEmailsCache = filterEmailsByPriority(emails, currentPriority);
      renderEmails(filteredEmailsCache);
      updateResetButtonVisibility();
      if (toast) {
        toast.textContent = "Emails classified successfully";
        toast.open = true;
      }
    } catch (e) {
      console.error("Classification failed:", e);
      if (toast) {
        toast.textContent = "Classification failed. Please try again.";
        toast.open = true;
      }
    } finally {
      classifyBtn.disabled = false;
      if (globalBusy) globalBusy.style.display = "none";
    }
  });

  resetBtn?.addEventListener("click", async () => {
    try {
      if (globalBusy) globalBusy.style.display = "flex";
      resetBtn.disabled = true;
      await resetEmails();
      // Refresh all data after reset
      const [tags, emails] = await Promise.all([fetchTags(), fetchEmails(currentTag)]);
      allEmailsCache = await fetchEmails();
      setTags(tags);
      emailsCache = emails;
      filteredEmailsCache = filterEmailsByPriority(emails, currentPriority);
      renderEmails(filteredEmailsCache);
      clearPreview();
      updateResetButtonVisibility();
      if (toast) {
        toast.textContent = "Dataset reset to unclassified";
        toast.open = true;
      }
    } catch (e) {
      console.error("Reset failed:", e);
      if (toast) {
        toast.textContent = "Reset failed. Please try again.";
        toast.open = true;
      }
    } finally {
      resetBtn.disabled = false;
      if (globalBusy) globalBusy.style.display = "none";
    }
  });
  tagsList?.addEventListener("item-click", async (ev) => {
    const li = ev.detail.item;
    currentTag = li?.dataset?.tag || null;
    // Mark the clicked folder as selected
    try {
      tagsList.selectedItems = [li];
    } catch {}
    await loadAndRender(currentTag, currentPriority);
    // Update header title and count after folder change
    const baseEmails = currentTag ? allEmailsCache.filter((e) => (e.tags || []).includes(currentTag)) : allEmailsCache;
    const filteredCount = filterEmailsByPriority(baseEmails, currentPriority).length;
    if (emailsTitle) emailsTitle.textContent = currentTag || "Inbox";
    if (emailsCount) emailsCount.textContent = String(filteredCount);
  });

  priorityFilter?.addEventListener("change", (ev) => {
    currentPriority = ev.detail.selectedOption.value;
    // Re-filter current emails with new priority
    filteredEmailsCache = filterEmailsByPriority(emailsCache, currentPriority);
    renderEmails(filteredEmailsCache);

    // Update count display
    if (emailsCount) emailsCount.textContent = String(filteredEmailsCache.length);
    clearPreview();
  });

  emailsList?.addEventListener("item-click", (ev) => {
    const li = ev.detail.item;
    const id = li?.dataset?.messageId;
    const email = filteredEmailsCache.find((e) => e.messageId === id);
    renderPreview(email);
  });

  // Initial load
  (async () => {
    try {
      // Reset emails on page load to ensure clean state
      await resetEmails();

      const [tags, emails] = await Promise.all([fetchTags(), fetchEmails()]);
      allEmailsCache = emails;
      emailsCache = emails;
      filteredEmailsCache = filterEmailsByPriority(emails, currentPriority);
      setTags(tags);
      renderEmails(filteredEmailsCache);
      updateResetButtonVisibility();
      // Ensure initial header count and title are correct
      if (emailsTitle) emailsTitle.textContent = "Inbox";
      if (emailsCount) emailsCount.textContent = String(filteredEmailsCache.length);
      // Ensure Inbox is selected by default in folders list
      const inboxItem = tagsList?.querySelector("ui5-li[data-tag='']");
      if (inboxItem && tagsList) {
        try {
          tagsList.selectedItems = [inboxItem];
        } catch {}
      }
    } catch (e) {
      console.error("Failed to load emails:", e);
    }
  })();
}
