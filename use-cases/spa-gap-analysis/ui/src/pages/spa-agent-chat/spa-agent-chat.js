/* SPA Agent Chat page - UI5 components (SAP Fiori style) */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/CheckBox.js";

import { request } from "../../services/api.js";

// State
let conversationId = null;
let messages = [];

// LocalStorage keys
const STORAGE_KEYS = {
  CONVERSATION_ID: 'spa_agent_chat_conversation_id',
  MESSAGES: 'spa_agent_chat_messages'
};

// Save state to localStorage
function saveState() {
  try {
    sessionStorage.setItem(STORAGE_KEYS.CONVERSATION_ID, conversationId || '');
    sessionStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(messages));
  } catch (error) {
    console.error('Failed to save chat state:', error);
  }
}

// Load state from localStorage
function loadState() {
  try {
    const savedConversationId = sessionStorage.getItem(STORAGE_KEYS.CONVERSATION_ID);
    const savedMessages = sessionStorage.getItem(STORAGE_KEYS.MESSAGES);

    if (savedConversationId) {
      conversationId = savedConversationId;
    }

    if (savedMessages) {
      messages = JSON.parse(savedMessages);
    }
  } catch (error) {
    console.error('Failed to load chat state:', error);
  }
}

// Clear state from localStorage
function clearState() {
  try {
    sessionStorage.removeItem(STORAGE_KEYS.CONVERSATION_ID);
    sessionStorage.removeItem(STORAGE_KEYS.MESSAGES);
  } catch (error) {
    console.error('Failed to clear chat state:', error);
  }
}

// API function
async function sendChatMessage(message, customerId = null, excludeUnknown = false) {
  return await request(
    "/api/spa/agent-chat",
    "POST",
    {
      message,
      conversation_id: conversationId,
      customer_id: customerId,
      exclude_unknown: excludeUnknown
    }
  );
}

async function clearConversation() {
  if (!conversationId) return;

  try {
    await request(
      `/api/spa/agent-chat/${conversationId}`,
      "DELETE",
      null
    );
    conversationId = null;
    messages = [];
    clearState(); // Clear localStorage
  } catch (error) {
    console.error("Failed to clear conversation:", error);
  }
}

export default function initSpaAgentChatPage() {
  console.log("SPA Agent Chat page initialized");

  const chatTimeline = document.getElementById("chat-timeline");
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const clearButton = document.getElementById("clear-conversation-button");
  const hasNamesCheckbox = document.getElementById("filter-has-names-agent");

  // Load saved state from sessionStorage
  loadState();

  // Initialize
  if (chatTimeline) {
    if (messages.length > 0) {
      // Restore messages from saved state
      restoreMessages();
    } else {
      renderEmptyState();
    }
  }

  // Send button handler
  if (sendButton && messageInput) {
    sendButton.addEventListener("click", async () => {
      await handleSendMessage();
    });
  }

  // Enter key handler (Shift+Enter for new line, Enter to send)
  if (messageInput) {
    messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
  }

  // Clear conversation handler
  if (clearButton) {
    clearButton.addEventListener("click", async () => {
      await clearConversation();
      chatTimeline.innerHTML = "";
      renderEmptyState();
    });
  }

  // Quick action button handlers
  const quickAction1 = document.getElementById("quick-action-1");
  const quickAction2 = document.getElementById("quick-action-2");
  const quickAction3 = document.getElementById("quick-action-3");
  const quickActionOnboarding = document.getElementById("quick-action-onboarding");
  const quickAction4 = document.getElementById("quick-action-4");

  if (quickAction1) {
    quickAction1.addEventListener("click", () => {
      messageInput.value = "Analyze customer <customer_id>";
      handleSendMessage();
    });
  }

  if (quickAction2) {
    quickAction2.addEventListener("click", () => {
      messageInput.value = "Show RFM distribution";
      handleSendMessage();
    });
  }

  if (quickAction3) {
    quickAction3.addEventListener("click", () => {
      messageInput.value = "Find similar customers to <customer_id>";
      handleSendMessage();
    });
  }

  if (quickActionOnboarding) {
    quickActionOnboarding.addEventListener("click", () => {
      messageInput.value = "Research onboarding for new customer: Demo Contractor Services, Demo City AZ";
      handleSendMessage();
    });
  }

  if (quickAction4) {
    quickAction4.addEventListener("click", () => {
      messageInput.value = "What can you help me with?";
      handleSendMessage();
    });
  }

  async function handleSendMessage() {
    const message = messageInput.value.trim();

    if (!message) return;

    // Disable input while processing
    sendButton.disabled = true;
    messageInput.disabled = true;

    // Add user message to timeline
    addUserMessage(message);
    messageInput.value = "";

    // Show loading indicator
    showLoadingIndicator();

    try {
      const excludeUnknown = hasNamesCheckbox ? hasNamesCheckbox.checked : false;
      console.log('[FILTER DEBUG] Checkbox state:', hasNamesCheckbox?.checked);
      console.log('[FILTER DEBUG] Sending exclude_unknown:', excludeUnknown);
      const response = await sendChatMessage(message, null, excludeUnknown);

      // Store conversation ID
      if (response.conversation_id) {
        conversationId = response.conversation_id;
        saveState(); // Save conversation ID
      }

      // Remove loading indicator
      hideLoadingIndicator();

      // Add assistant response
      console.log('[Agent Chat] Response entities:', response.data?.entities);
      addAssistantMessage(response.message, response.data?.entities);

      // Scroll to bottom
      scrollToBottom();

    } catch (error) {
      hideLoadingIndicator();
      addErrorMessage(error.message || "Failed to send message");
    } finally {
      sendButton.disabled = false;
      messageInput.disabled = false;
      messageInput.focus();
    }
  }

  function renderEmptyState() {
    chatTimeline.innerHTML = `
      <div id="empty-state" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem; gap: 1rem; color: var(--sapContent_LabelColor)">
        <ui5-icon name="discussion-2" style="font-size: 3rem; opacity: 0.3"></ui5-icon>
        <ui5-text>Start a conversation by typing a message or using a quick action</ui5-text>
      </div>
    `;
  }

  function restoreMessages() {
    // Clear timeline first
    chatTimeline.innerHTML = "";

    // Restore all messages from saved state
    messages.forEach(msg => {
      if (msg.role === "user") {
        const messageDiv = document.createElement("div");
        messageDiv.className = "message-wrapper";
        messageDiv.innerHTML = `
          <div style="display: flex; justify-content: flex-end; padding: 0.5rem 0; margin-bottom: 0.5rem">
            <div style="max-width: 70%; text-align: right">
              <div style="
                background: var(--sapInformationBackground);
                border-radius: 12px 12px 0 12px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.25rem;
                word-wrap: break-word;
              ">
                <ui5-text>${escapeHtml(msg.content)}</ui5-text>
              </div>
              <ui5-text style="font-size: 0.7rem; color: var(--sapContent_LabelColor)">${msg.timestamp}</ui5-text>
            </div>
          </div>
        `;
        chatTimeline.appendChild(messageDiv);
      } else if (msg.role === "assistant") {
        const formattedMessage = formatMessage(msg.content, msg.entities || []);
        const messageDiv = document.createElement("div");
        messageDiv.className = "message-wrapper";
        messageDiv.innerHTML = `
          <div style="display: flex; justify-content: flex-start; padding: 0.5rem 0; margin-bottom: 0.5rem">
            <div style="max-width: 70%">
              <div style="
                background: var(--sapList_Background);
                border: 1px solid var(--sapGroup_BorderColor);
                border-radius: 12px 12px 12px 0;
                padding: 0.75rem 1rem;
                margin-bottom: 0.25rem;
                word-wrap: break-word;
              ">
                ${formattedMessage}
              </div>
              <ui5-text style="font-size: 0.7rem; color: var(--sapContent_LabelColor)">${msg.timestamp}</ui5-text>
            </div>
          </div>
        `;
        chatTimeline.appendChild(messageDiv);
        // Attach click handlers for restored messages
        attachClickHandlers(messageDiv, msg.entities || []);
      }
    });

    scrollToBottom();
  }

  function removeEmptyState() {
    const emptyState = document.getElementById("empty-state");
    if (emptyState) {
      emptyState.remove();
    }
  }

  function addUserMessage(message) {
    removeEmptyState();

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const messageDiv = document.createElement("div");
    messageDiv.className = "message-wrapper";
    messageDiv.innerHTML = `
      <div style="display: flex; justify-content: flex-end; padding: 0.5rem 0; margin-bottom: 0.5rem">
        <div style="max-width: 70%; text-align: right">
          <div style="
            background: var(--sapInformationBackground);
            border-radius: 12px 12px 0 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.25rem;
            word-wrap: break-word;
          ">
            <ui5-text>${escapeHtml(message)}</ui5-text>
          </div>
          <ui5-text style="font-size: 0.7rem; color: var(--sapContent_LabelColor)">${timestamp}</ui5-text>
        </div>
      </div>
    `;

    chatTimeline.appendChild(messageDiv);
    messages.push({ role: "user", content: message, timestamp });
    saveState(); // Save after adding message
    scrollToBottom();
  }

  function addAssistantMessage(message, entities = []) {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Convert markdown-style formatting to HTML and make IDs clickable
    const formattedMessage = formatMessage(message, entities);

    const messageDiv = document.createElement("div");
    messageDiv.className = "message-wrapper";
    messageDiv.innerHTML = `
      <div style="display: flex; justify-content: flex-start; padding: 0.5rem 0; margin-bottom: 0.5rem">
        <div style="max-width: 70%">
          <div style="
            background: var(--sapList_Background);
            border: 1px solid var(--sapGroup_BorderColor);
            border-radius: 12px 12px 12px 0;
            padding: 0.75rem 1rem;
            margin-bottom: 0.25rem;
            word-wrap: break-word;
          ">
            ${formattedMessage}
          </div>
          <ui5-text style="font-size: 0.7rem; color: var(--sapContent_LabelColor)">${timestamp}</ui5-text>
        </div>
      </div>
    `;

    chatTimeline.appendChild(messageDiv);
    messages.push({ role: "assistant", content: message, timestamp, entities });
    saveState(); // Save after adding message

    // Attach click handlers for clickable IDs
    attachClickHandlers(messageDiv, entities);

    scrollToBottom();
  }

  function addErrorMessage(error) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message-wrapper";
    messageDiv.innerHTML = `
      <div style="padding: 0.5rem 0; margin-bottom: 0.5rem">
        <ui5-message-strip design="Negative" hide-close-button>
          <ui5-icon name="alert" slot="icon"></ui5-icon>
          <strong>Error:</strong> ${escapeHtml(error)}
        </ui5-message-strip>
      </div>
    `;
    chatTimeline.appendChild(messageDiv);
    scrollToBottom();
  }

  function showLoadingIndicator() {
    const loadingDiv = document.createElement("div");
    loadingDiv.id = "loading-indicator";
    loadingDiv.className = "message-wrapper";
    loadingDiv.innerHTML = `
      <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0; margin-bottom: 0.5rem">
        <ui5-busy-indicator active style="width: 2rem; height: 2rem"></ui5-busy-indicator>
        <ui5-text style="font-style: italic; color: var(--sapContent_LabelColor)">Analyzing...</ui5-text>
      </div>
    `;
    chatTimeline.appendChild(loadingDiv);
    scrollToBottom();
  }

  function hideLoadingIndicator() {
    const loadingIndicator = document.getElementById("loading-indicator");
    if (loadingIndicator) {
      loadingIndicator.remove();
    }
  }

  function scrollToBottom() {
    setTimeout(() => {
      chatTimeline.scrollTop = chatTimeline.scrollHeight;
    }, 100);
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function formatMessage(text, entities = []) {
    // Convert markdown-style formatting to HTML
    let formatted = escapeHtml(text);

    console.log('[formatMessage] Processing entities:', entities);
    console.log('[formatMessage] Text preview:', text.substring(0, 200));

    // Bold text: **text** -> <strong>text</strong>
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Make customer IDs and SPA IDs clickable
    if (entities && entities.length > 0) {
      entities.forEach(entity => {
        if (entity.type === 'customer' && entity.id) {
          console.log(`[formatMessage] Processing customer entity: ${entity.id} - ${entity.name}`);
          // Match "Customer 999001" or "ID: 999001" or "(ID: 999001)"
          const patterns = [
            new RegExp(`Customer ${entity.id}\\b`, 'g'),
            new RegExp(`ID: ${entity.id}\\b`, 'g'),
            new RegExp(`\\(ID: ${entity.id}\\)`, 'g')
          ];
          patterns.forEach(pattern => {
            const matches = formatted.match(pattern);
            if (matches) {
              console.log(`[formatMessage] Found ${matches.length} matches for pattern:`, pattern);
            }
            formatted = formatted.replace(pattern, (match) =>
              `<span class="clickable-customer-id" data-customer-id="${entity.id}" style="color: var(--sapLinkColor); cursor: pointer; text-decoration: underline;">${match}</span>`
            );
          });
        } else if (entity.type === 'spa' && entity.id) {
          // Match "SPA 40088797"
          const spaPattern = new RegExp(`SPA ${entity.id}\\b`, 'g');
          formatted = formatted.replace(spaPattern, (match) =>
            `<span class="clickable-spa-id" data-spa-id="${entity.id}" style="color: var(--sapLinkColor); cursor: pointer; text-decoration: underline;">${match}</span>`
          );
        }
      });
    } else {
      console.log('[formatMessage] No entities to process');
    }

    // Convert markdown to HTML (IMPORTANT: process in correct order!)

    // STEP 1: Headers first (before line breaks)
    // Headers: ## Header -> styled div with bottom border (normal size)
    formatted = formatted.replace(/^## (.+)$/gm, '<div style="margin-top: 1.25rem; margin-bottom: 0.5rem; font-size: 1.05em; font-weight: 600; color: var(--sapTextColor); border-bottom: 1px solid var(--sapGroup_BorderColor); padding-bottom: 0.25rem;">$1</div>');

    // Subheaders: ### Header -> styled div (normal size)
    formatted = formatted.replace(/^### (.+)$/gm, '<div style="margin-top: 0.75rem; margin-bottom: 0.35rem; font-size: 1em; font-weight: 600; color: var(--sapTextColor);">$1</div>');

    // STEP 2: Convert double+ line breaks to paragraph spacing BEFORE converting single line breaks
    // This preserves paragraph breaks as visible spacing
    formatted = formatted.replace(/\n\n+/g, '{{PARAGRAPH_BREAK}}');

    // STEP 3: Convert single line breaks to <br>
    formatted = formatted.replace(/\n/g, '<br>');

    // STEP 4: Convert paragraph break markers to actual spacing divs
    formatted = formatted.replace(/\{\{PARAGRAPH_BREAK\}\}/g, '<div style="margin-top: 0.75rem;"></div>');

    // STEP 5: Convert bullet points to proper formatted items with indentation
    // Handle: "- item" or "• item"
    formatted = formatted.replace(/<br>([-•]\s+.+?)(?=<br>|$)/g, (match, item) => {
      // Remove the dash/bullet and trim
      const content = item.replace(/^[-•]\s+/, '').trim();
      return `<div style="margin-left: 1.5rem; margin-top: 0.2rem; margin-bottom: 0.2rem; position: relative; line-height: 1.4;">
        <span style="position: absolute; left: -1rem; color: var(--sapTextColor);">•</span>
        <span>${content}</span>
      </div>`;
    });

    return formatted;
  }

  function attachClickHandlers(messageDiv, entities = []) {
    // Customer ID click handlers
    const customerLinks = messageDiv.querySelectorAll('.clickable-customer-id');
    customerLinks.forEach(link => {
      link.addEventListener('click', () => {
        const customerId = link.getAttribute('data-customer-id');
        // Navigate to Quick Lookup tab with customer ID
        const event = new CustomEvent('navigate-to-quick-lookup', {
          detail: { customerId }
        });
        document.dispatchEvent(event);
      });
    });

    // SPA ID click handlers (future: could open SPA details modal)
    const spaLinks = messageDiv.querySelectorAll('.clickable-spa-id');
    spaLinks.forEach(link => {
      link.addEventListener('click', () => {
        const spaId = link.getAttribute('data-spa-id');
        console.log('SPA clicked:', spaId);
        // TODO: Show SPA details in modal or navigate to SPA details page
      });
    });
  }
}
