/* Chat UI5 components */
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Avatar.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents-icons/dist/navigation-right-arrow.js";
import { marked } from "marked";

import { request } from "../../services/api.js";

// API for chat with history
async function postChatHistory(messages) {
  return await request(`/api/chat/openai/history`, "POST", { messages });
}

export default function initChatPage() {
  console.log("Chat page initialized");

  const sendButton = document.getElementById("send-button");
  const chatTimeline = document.getElementById("chat-timeline");
  const input = document.getElementById("message-input");
  const qa1 = document.getElementById("qa-1");
  const qa2 = document.getElementById("qa-2");
  const qa3 = document.getElementById("qa-3");

  const conversation = [];

  function renderMessages() {
    if (!chatTimeline) return;
    chatTimeline.innerHTML = "";
    for (const msg of conversation) {
      const row = document.createElement("div");
      row.className = `message-row ${msg.role}`;

      const avatar = document.createElement("ui5-avatar");
      avatar.shape = "Circle";
      avatar.interactive = false;
      avatar.icon = msg.role === "user" ? "person-placeholder" : "ai";

      const bubble = document.createElement("div");
      bubble.className = `bubble ${msg.role}`;
      if (msg.content === "…") {
        // typing bubble with spinner
        const dots = document.createElement("ui5-busy-indicator");
        dots.setAttribute("active", "");
        dots.setAttribute("size", "S");
        dots.setAttribute("delay", "0");
        bubble.appendChild(dots);
      } else {
        bubble.innerHTML = marked.parse(msg.content || "");
      }

      if (msg.role === "user") {
        row.appendChild(bubble);
        row.appendChild(avatar);
      } else {
        row.appendChild(avatar);
        row.appendChild(bubble);
      }

      chatTimeline.appendChild(row);
    }
    chatTimeline.scrollTop = chatTimeline.scrollHeight;
  }

  async function sendMessage() {
    const text = (input?.value || "").trim();
    if (!text) return;

    // push user message
    conversation.push({ role: "user", content: text });
    input.value = "";
    renderMessages();

    // show typing placeholder
    conversation.push({ role: "assistant", content: "…" });
    renderMessages();

    if (sendButton) sendButton.disabled = true;
    if (input) input.disabled = true;
    try {
      const data = await postChatHistory(conversation.filter((m) => m.content !== "…"));
      // replace typing with real answer
      const last = conversation[conversation.length - 1];
      if (last && last.role === "assistant") {
        last.content = data.text;
      } else {
        conversation.push({ role: "assistant", content: data.text });
      }
    } catch (error) {
      console.error("Error:", error);
      const last = conversation[conversation.length - 1];
      if (last && last.role === "assistant") {
        last.content = `An error occurred: ${error.message}`;
      } else {
        conversation.push({ role: "assistant", content: `An error occurred: ${error.message}` });
      }
    } finally {
      if (sendButton) sendButton.disabled = false;
      if (input) input.disabled = false;
      renderMessages();
    }
  }

  if (sendButton) {
    sendButton.addEventListener("click", sendMessage);
  }
  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  function quickAsk(text) {
    if (!input) return;
    input.value = text;
    sendMessage();
  }

  if (qa1) qa1.addEventListener("click", () => quickAsk("What is the status of the invoice MAINT-2024-556?"));
  if (qa2) qa2.addEventListener("click", () => quickAsk("List W9 updates mentioned in the last 7 days."));
  if (qa3) qa3.addEventListener("click", () => quickAsk("Summarize unresolved payment inquiries by vendor and invoice."));
}
