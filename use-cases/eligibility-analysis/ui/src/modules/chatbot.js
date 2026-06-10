import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Icon.js";

import { sendA2AUserMessage } from "../services/a2a.js";

const DEFAULT_WELCOME = "Hello! I'm your eligibility assistant. Ask me about invoice rules, rejection reasons, or seller eligibility insights.";

function newContextId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createMessageElement(content, role, { isError = false } = {}) {
  const message = document.createElement("div");
  message.className = `chat-message ${role}${isError ? " error" : ""}`;

  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  message.appendChild(body);

  return message;
}

function createLoadingElement() {
  const message = document.createElement("div");
  message.className = "chat-message assistant loading";

  const body = document.createElement("div");
  body.className = "message-content";

  const indicator = document.createElement("ui5-busy-indicator");
  indicator.setAttribute("active", "");
  indicator.setAttribute("size", "Small");
  indicator.setAttribute("delay", "0");

  const label = document.createElement("span");
  label.className = "loading-text";
  label.textContent = "Working on it...";

  body.appendChild(indicator);
  body.appendChild(label);
  message.appendChild(body);

  return message;
}

export function initChatbot() {
  const panel = document.getElementById("chatbotPanel");
  const messagesContainer = document.getElementById("chatMessages");
  const input = document.getElementById("chatInput");
  const sendButton = document.getElementById("sendChatBtn");
  const clearButton = document.getElementById("clearChatBtn");
  const closeButton = document.getElementById("closeChatBtn");
  const assistantButton = document.getElementById("assistantBtn");

  if (!panel || !messagesContainer || !input || !sendButton) {
    console.warn("[Chatbot] Required elements not found; chatbot not initialized.");
    return;
  }

  let isOpen = false;
  let isLoading = false;
  let contextId = newContextId();

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function syncAssistantToggle() {
    if (assistantButton) {
      assistantButton.pressed = isOpen;
    }
  }

  function openPanel() {
    panel.classList.add("open");
    panel.setAttribute("aria-hidden", "false");
    isOpen = true;
    syncAssistantToggle();
    setTimeout(() => input.focus(), 100);
  }

  function closePanel() {
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
    isOpen = false;
    syncAssistantToggle();
  }

  function togglePanel() {
    if (isOpen) {
      closePanel();
    } else {
      openPanel();
    }
  }

  function setInputEnabled(enabled) {
    input.disabled = !enabled;
    sendButton.disabled = !enabled;
  }

  function addMessage(content, role, options = {}) {
    const message = createMessageElement(content, role, options);
    messagesContainer.appendChild(message);
    scrollToBottom();
    return message;
  }

  function addLoading() {
    const loading = createLoadingElement();
    messagesContainer.appendChild(loading);
    scrollToBottom();
    return loading;
  }

  function clearChat() {
    messagesContainer.innerHTML = "";
    addMessage(DEFAULT_WELCOME, "assistant");
    contextId = newContextId();
  }

  async function sendMessage() {
    if (isLoading) return;
    const text = (input.value || "").trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    const loadingEl = addLoading();
    isLoading = true;
    setInputEnabled(false);

    try {
      const response = await sendA2AUserMessage(text, { contextId });
      if (response?.contextId) {
        contextId = response.contextId;
      }

      loadingEl.remove();
      const answer = response?.text || "I couldn't generate a response. Please try again.";
      addMessage(answer, "assistant");
    } catch (error) {
      loadingEl.remove();
      const message = error?.message || "Something went wrong while contacting the assistant.";
      addMessage(message, "assistant", { isError: true });
      console.error("[Chatbot] A2A error:", error);
    } finally {
      isLoading = false;
      setInputEnabled(true);
      input.focus();
    }
  }

  sendButton.addEventListener("click", sendMessage);
  input.addEventListener("keypress", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  if (clearButton) {
    clearButton.addEventListener("click", clearChat);
  }

  if (closeButton) {
    closeButton.addEventListener("click", closePanel);
  }

  if (assistantButton) {
    assistantButton.addEventListener("click", togglePanel);
  }

  panel.setAttribute("aria-hidden", "true");
  clearChat();

  console.log("[Chatbot] Initialized");
}
