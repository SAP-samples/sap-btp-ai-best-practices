/* Chat UI5 components */
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents-icons/dist/navigation-right-arrow.js";
import { marked } from "marked";
import dayjs from "dayjs";
import calendar from "dayjs/plugin/calendar.js";

import { request } from "../../services/api.js";

// API for chat with history
async function postChatHistory(messages) {
  return await request(`/api/chat/history`, "POST", { messages });
}

export default function initChatPage() {
  dayjs.extend(calendar);

  const sendButton = document.getElementById("send-button");
  const chatTimeline = document.getElementById("chat-timeline");
  const chatSplash = document.getElementById("chat-splash");
  const input = document.getElementById("message-input");
  const qa1 = document.getElementById("qa-1");
  const qa2 = document.getElementById("qa-2");
  const qa3 = document.getElementById("qa-3");

  const conversation = [];

  function formatTimestamp(epochMs) {
    return dayjs(epochMs).calendar(null, {
      sameDay: "[Today] h:mm A",
      lastDay: "[Yesterday] h:mm A",
      lastWeek: "MMM D, YYYY h:mm A",
      sameElse: "MMM D, YYYY h:mm A"
    });
  }

  function renderMessages() {
    if (!chatTimeline) return;
    chatTimeline.innerHTML = "";
    // Toggle splash visibility when there are no messages (ignoring typing placeholder)
    const hasRealMessages = conversation.some((m) => m.content !== "…");
    if (chatSplash) {
      if (!hasRealMessages) {
        chatSplash.classList.add("visible");
      } else {
        chatSplash.classList.remove("visible");
      }
    }
    for (const msg of conversation) {
      const row = document.createElement("div");
      row.className = `message-row ${msg.role}`;

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
      row.appendChild(col);

      chatTimeline.appendChild(row);
    }

    setTimeout(() => {
      chatTimeline.scrollTop = chatTimeline.scrollHeight;
    }, 10);
  }

  async function sendMessage() {
    const text = (input?.value || "").trim();
    if (!text) return;

    // push user message with timestamp (not sent to API)
    conversation.push({ role: "user", content: text, ts: Date.now() });
    input.value = "";
    renderMessages();

    // show typing placeholder
    conversation.push({ role: "assistant", content: "…" });
    renderMessages();

    if (sendButton) sendButton.disabled = true;
    if (input) input.disabled = true;
    try {
      const data = await postChatHistory(conversation.filter((m) => m.content !== "…").map((m) => ({ role: m.role, content: m.content })));
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
      if (input) input.disabled = false;
      if (sendButton) {
        const hasText = (input?.value || "").trim().length > 0;
        sendButton.disabled = !hasText;
      }
      renderMessages();
      if (input) {
        setTimeout(() => {
          input.focus();
        }, 10);
      }
    }
  }

  if (sendButton) {
    sendButton.addEventListener("click", sendMessage);
    // initialize disabled state based on input
    sendButton.disabled = !(input && (input.value || "").trim().length > 0);
  }
  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        if (e.shiftKey) {
          // allow newline
          return;
        }
        // plain Enter: send
        if (!e.metaKey && !e.ctrlKey) {
          e.preventDefault();
          sendMessage();
        }
      }
    });
    input.addEventListener("input", () => {
      if (sendButton) {
        const hasText = (input.value || "").trim().length > 0;
        sendButton.disabled = !hasText;
      }
    });
  }

  function quickAsk(text) {
    if (!input) return;
    input.value = text;
    sendMessage();
    if (sendButton) sendButton.disabled = false;
    input.focus();
  }

  if (qa1) qa1.addEventListener("click", () => quickAsk("Summarize the following into 5 bullets and a one-line takeaway. I'll paste the text next."));
  if (qa2) qa2.addEventListener("click", () => quickAsk("From the text I paste, extract invoice_number, vendor, amount, due_date as JSON."));
  if (qa3) qa3.addEventListener("click", () => quickAsk("Turn the following notes into action items grouped by owner, with due dates if mentioned."));

  // Initial render to toggle splash visibility on load
  renderMessages();
}
