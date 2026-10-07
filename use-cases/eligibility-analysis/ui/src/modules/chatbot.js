import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Icon.js";

import {workspaceRequest} from "../services/workspace-api.js";

import {workspaceReferences} from "../services/assistant-context.js";
import {messageContent} from "./chat-markdown.js";

let workspaceContext = {};
/** Publish current source references only; never inject invoice content into the prompt. */
export function setAssistantWorkspaceContext(context) {
  workspaceContext=workspaceReferences(context);
  window.dispatchEvent(new CustomEvent("assistant-workspace-context",{detail:workspaceContext}));
}

import { sendA2AUserMessage } from "../services/a2a.js";

const DEFAULT_WELCOME = "Hello! Ask me about eligibility, historical patterns, credit capacity, or the assumptions behind your saved recommendation.";

function newContextId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/**
 * Build one chat bubble, rendering Markdown only for successful assistant content.
 * @param {unknown} content Message content.
 * @param {string} role Message author role.
 * @param {{isError?: boolean}} options Rendering flags.
 * @returns {HTMLElement} Detached message element ready for insertion.
 */
function createMessageElement(content, role, { isError = false } = {}) {
  const message = document.createElement("div");
  message.className = `chat-message ${role}${isError ? " error" : ""}`;

  const body = document.createElement("div");
  body.className = "message-content";
  const rendered=messageContent(content,role,{isError});
  if(rendered.mode==='html'){
    body.classList.add('markdown-content');
    body.innerHTML=rendered.value;
  }else{
    body.textContent=rendered.value;
  }
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
  let contextId = sessionStorage.getItem("receivables-chat-context") || newContextId();
  const contextLabel=document.createElement("div");contextLabel.className="chat-context-label";
  messagesContainer.before(contextLabel);
  messagesContainer.setAttribute("aria-live","polite");
  panel.setAttribute("role","complementary");panel.setAttribute("aria-label","Receivables assistant");
  window.addEventListener("assistant-workspace-context",event=>{
    const scope=event.detail;contextLabel.textContent=scope.analysis_id?`Offer ${scope.analysis_id.slice(0,8)}${scope.run_id?` · Run ${scope.run_id.slice(0,8)}`:""} · ${scope.row_ids?.length||0} invoices in scope`:"General assistance · Choose an offer for contextual answers";
  });

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
    assistantButton?.focus();
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
    if (clearButton) clearButton.disabled = !enabled;
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
    sessionStorage.setItem("receivables-chat-context",contextId);
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
      const response = await sendA2AUserMessage(text, { contextId, workspaceContext:structuredClone(workspaceContext) });
      if (response?.contextId) {
        contextId = response.contextId;
        sessionStorage.setItem("receivables-chat-context",contextId);
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
  // Restore persisted complete turns; browser storage holds only the conversation pointer.
  workspaceRequest(`/conversations/${encodeURIComponent(contextId)}`).then(saved=>{
    if(messagesContainer.children.length)return;
    for(const item of saved.messages)addMessage(typeof item.content==='string'?item.content:JSON.stringify(item.content),item.role);
    if(!saved.messages.length)addMessage(DEFAULT_WELCOME,"assistant");
  }).catch(()=>{if(!messagesContainer.children.length)addMessage(DEFAULT_WELCOME,"assistant");});
  panel.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();closePanel();}});

  console.log("[Chatbot] Initialized");
}
