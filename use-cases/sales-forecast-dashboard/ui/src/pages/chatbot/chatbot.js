/* Chatbot UI5 Components */
import "@ui5/webcomponents-fiori/dist/Page.js";
import "@ui5/webcomponents/dist/Bar.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/List.js";
import "@ui5/webcomponents/dist/ListItemStandard.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";

/* Icons */
import "@ui5/webcomponents-icons/dist/da.js";
import "@ui5/webcomponents-icons/dist/paper-plane.js";
import "@ui5/webcomponents-icons/dist/chart-table-view.js";
import "@ui5/webcomponents-icons/dist/trend-up.js";
import "@ui5/webcomponents-icons/dist/business-objects-experience.js";
import "@ui5/webcomponents-icons/dist/compare.js";
import "@ui5/webcomponents-icons/dist/employee.js";
import "@ui5/webcomponents-icons/dist/pdf-attachment.js";
import "@ui5/webcomponents-icons/dist/excel-attachment.js";
import "@ui5/webcomponents-icons/dist/download.js";

/* API */
import { request, getFileUrl } from "../../services/api.js";

/* Markdown parsing */
import { marked } from "marked";

// Session state - persisted across messages
let sessionId = null;

export default function initChatbotPage() {
  console.log("Chatbot page initialized");

  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const chatMessages = document.getElementById("chat-messages");
  const suggestionsList = document.getElementById("suggestions-list");
  const busyIndicator = document.getElementById("chat-busy");

  // Send message on button click
  sendBtn.addEventListener("click", () => {
    sendMessage();
  });

  // Send message on Enter (without Shift)
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Handle suggestion clicks
  suggestionsList.addEventListener("item-click", (e) => {
    const question = e.detail.item.getAttribute("data-question");
    if (question) {
      chatInput.value = question;
      sendMessage();
    }
  });

  async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, "user");

    // Clear input
    chatInput.value = "";

    // Disable input while processing
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Show busy indicator
    busyIndicator.active = true;

    try {
      // Call the chatbot API
      const response = await request("/api/chatbot/chat", "POST", {
        message: message,
        session_id: sessionId,
      });

      // Store session ID for continuity
      sessionId = response.session_id;

      // Add assistant response with attachments
      addMessage(response.message, "assistant", response.attachments || []);

      // Log tool calls if any (for debugging)
      if (response.tool_calls && response.tool_calls.length > 0) {
        console.log("Agent used tools:", response.tool_calls.map(t => t.name).join(", "));
      }
      if (response.attachments && response.attachments.length > 0) {
        console.log("Attachments:", response.attachments.map(a => a.filename).join(", "));
      }
    } catch (error) {
      console.error("Chatbot API error:", error);
      addMessage(
        `I apologize, but I encountered an error: ${error.message}. Please try again.`,
        "assistant"
      );
    } finally {
      // Hide busy indicator
      busyIndicator.active = false;

      // Re-enable input
      chatInput.disabled = false;
      sendBtn.disabled = false;
      chatInput.focus();
    }
  }

  function addMessage(text, role, attachments = []) {
    const stickToBottom = isNearBottom();
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const avatarDiv = document.createElement("div");
    avatarDiv.className = "message-avatar";

    const icon = document.createElement("ui5-icon");
    icon.setAttribute("name", role === "assistant" ? "da" : "employee");
    avatarDiv.appendChild(icon);

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const textEl = document.createElement("div");
    textEl.className = "message-text";
    // Parse markdown to HTML (supports tables, bold, lists, etc.)
    textEl.innerHTML = marked.parse(text);
    contentDiv.appendChild(textEl);

    // Render attachments if present
    if (attachments && attachments.length > 0) {
      const attachmentsDiv = document.createElement("div");
      attachmentsDiv.className = "message-attachments";

      for (const attachment of attachments) {
        const fullUrl = getFileUrl(attachment.filename);

        if (attachment.file_type === "image") {
          // Image: display inline with click-to-enlarge
          const img = document.createElement("img");
          img.src = fullUrl;
          img.alt = attachment.filename;
          img.className = "attachment-image";
          img.loading = "lazy";
          img.addEventListener("click", () => openLightbox(fullUrl));
          attachmentsDiv.appendChild(img);
        } else {
          // CSV/PDF: download button
          const downloadLink = document.createElement("a");
          downloadLink.href = fullUrl;
          downloadLink.download = attachment.filename;
          downloadLink.className = "attachment-download";

          const downloadIcon = document.createElement("ui5-icon");
          downloadIcon.setAttribute(
            "name",
            attachment.file_type === "pdf" ? "pdf-attachment" : "excel-attachment"
          );
          downloadLink.appendChild(downloadIcon);

          const label = document.createElement("span");
          label.textContent = attachment.filename;
          downloadLink.appendChild(label);

          attachmentsDiv.appendChild(downloadLink);
        }
      }

      contentDiv.appendChild(attachmentsDiv);
    }

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);
    if (stickToBottom) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  // Only auto-scroll when the user is already at (or near) the bottom
  function isNearBottom() {
    const distanceFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
    return distanceFromBottom < 32;
  }

  // Lightbox functionality for image attachments
  const lightbox = document.getElementById("image-lightbox");
  const lightboxImage = document.getElementById("lightbox-image");
  const lightboxClose = document.getElementById("lightbox-close");

  function openLightbox(imgSrc) {
    lightboxImage.src = imgSrc;
    lightbox.classList.add("active");
  }

  function closeLightbox() {
    lightbox.classList.remove("active");
    lightboxImage.src = "";
  }

  // Close lightbox on backdrop click
  lightbox.addEventListener("click", closeLightbox);

  // Close lightbox on close button click
  lightboxClose.addEventListener("click", (e) => {
    e.stopPropagation();
    closeLightbox();
  });

  // Prevent closing when clicking the image itself
  lightboxImage.addEventListener("click", (e) => e.stopPropagation());

  // Close lightbox on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox.classList.contains("active")) {
      closeLightbox();
    }
  });

  // Helper function to show scenario context indicator
  function showScenarioContextIndicator(scenario) {
    const contextDiv = document.createElement("div");
    contextDiv.className = "scenario-context-indicator";
    contextDiv.innerHTML = `
      <div class="context-header">
        <ui5-icon name="simulate"></ui5-icon>
        <span>Analyzing Scenario: <strong>${scenario.scenario_name}</strong></span>
      </div>
      <div class="context-details">
        <span>Channel: ${scenario.channel}</span>
        <span>Horizon: Week ${scenario.time_window.horizon_start}-${scenario.time_window.horizon_end}</span>
        <span>Modifications: ${scenario.modifications.length}</span>
      </div>
    `;
    // Insert before chat messages
    chatMessages.parentNode.insertBefore(contextDiv, chatMessages);
  }

  // Check for pending scenario analysis from Scenario Maker
  const pendingScenarioData = sessionStorage.getItem("pendingScenarioAnalysis");
  if (pendingScenarioData) {
    try {
      const { scenario, prompt, timestamp } = JSON.parse(pendingScenarioData);

      // Only process if recent (within 5 minutes)
      if (Date.now() - timestamp < 5 * 60 * 1000) {
        // Clear storage immediately to prevent re-processing
        sessionStorage.removeItem("pendingScenarioAnalysis");

        // Show scenario context indicator
        showScenarioContextIndicator(scenario);

        // Set the prompt and trigger send
        chatInput.value = prompt;

        // Slight delay to ensure UI is ready
        setTimeout(() => {
          sendMessage();
        }, 100);
      } else {
        // Expired, clear it
        sessionStorage.removeItem("pendingScenarioAnalysis");
      }
    } catch (e) {
      console.error("Error parsing pending scenario:", e);
      sessionStorage.removeItem("pendingScenarioAnalysis");
    }
  }

  // Check for pending store query from dashboard
  if (window.pendingChatbotQuery && window.pendingChatbotQuery.storeId) {
    const storeId = window.pendingChatbotQuery.storeId;
    // Clear the pending query
    window.pendingChatbotQuery = null;
    // Set the input and send the message
    chatInput.value = `Show me information about store ${storeId}`;
    sendMessage();
  }
}
