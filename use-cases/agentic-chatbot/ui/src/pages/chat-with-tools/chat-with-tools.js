/* Chat UI5 components */
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents-icons/dist/navigation-right-arrow.js";
import { marked } from "marked";
import dayjs from "dayjs";
import calendar from "dayjs/plugin/calendar.js";

import { request, streamNDJSON } from "../../services/api.js";

// API for chat with tools (LangGraph-enabled)
async function postChatWithTools(messages) {
  return await request(`/api/chat/tools`, "POST", { messages });
}

async function streamChatWithTools(messages, onChunk) {
  return await streamNDJSON(`/api/chat/tools/stream`, {
    method: "POST",
    body: { messages },
    onChunk
  });
}

// Toggle streaming vs direct API; defaults to streaming unless explicitly disabled
const USE_STREAM = true;

async function doStreamChat(baseMessages, conversation, onRender) {
  let accumulated = "";
  let toolHint = "";

  await streamChatWithTools(baseMessages, (evt) => {
    if (evt.type === "assistant") {
      accumulated = evt.content || "";
      // Assistant content means tool phase is over
      toolHint = "";
    }
    if (evt.type === "tool") {
      const name = evt.name || "tool";
      // For singular tool event, show only the latest tool
      toolHint = `Using ${name}…`;
      // mark the typing row as tool-active
      const lastMsg = conversation[conversation.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.toolActive = true;
      }
    }
    if (evt.type === "tools" && Array.isArray(evt.items)) {
      const names = evt.items.map((it) => it && it.name).filter(Boolean);
      toolHint = names.length > 1 ? `Using ${names.join(", ")}…` : names[0] ? `Using ${names[0]}…` : "";
      const lastMsg = conversation[conversation.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.toolActive = true;
      }
    }
    if (evt.type === "error") {
      accumulated = `An error occurred: ${evt.error}`;
      toolHint = "";
    }

    // update the typing bubble in place: keep spinner with hint until assistant text appears
    const last = conversation[conversation.length - 1];
    if (last && last.role === "assistant") {
      if (accumulated) {
        last.content = accumulated;
        last.hint = undefined;
        last.toolActive = undefined;
      } else {
        last.content = "…";
        last.hint = toolHint || undefined;
      }
    }
    onRender();
  });

  // finalize
  const last = conversation[conversation.length - 1];
  if (last && last.role === "assistant" && (!last.content || last.content === "…")) {
    last.content = accumulated || last.content;
    last.hint = undefined;
  }
  onRender();
}

async function doDirectChat(baseMessages, conversation, onRender) {
  const data = await postChatWithTools(baseMessages);
  const last = conversation[conversation.length - 1];
  if (last && last.role === "assistant") {
    last.content = data.text;
    last.hint = undefined;
  } else {
    conversation.push({ role: "assistant", content: data.text });
  }
  onRender();
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
  const qa4 = document.getElementById("qa-4");
  const qa5 = document.getElementById("qa-5");
  const qa6 = document.getElementById("qa-6");

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
      if (msg.content === "…") {
        // Standalone loader row
        const loadRow = document.createElement("div");
        loadRow.className = "loader-row";

        if (msg.toolActive) {
          const ring = document.createElement("div");
          ring.className = "loader-ring";
          loadRow.appendChild(ring);
        } else {
          const loader = document.createElement("ui5-busy-indicator");
          loader.setAttribute("active", "");
          loader.setAttribute("size", "M");
          loader.setAttribute("delay", "0");
          loadRow.appendChild(loader);
        }

        if (msg.hint) {
          const hint = document.createElement("ui5-text");
          hint.className = "loader-hint";
          hint.textContent = msg.hint;
          loadRow.appendChild(hint);
        }

        chatTimeline.appendChild(loadRow);
        continue;
      }

      const row = document.createElement("div");
      row.className = `message-row ${msg.role}`;

      const bubble = document.createElement("div");
      bubble.className = `bubble ${msg.role}`;
      bubble.innerHTML = marked.parse(msg.content || "");

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
      const baseMessages = conversation.filter((m) => m.content !== "…").map((m) => ({ role: m.role, content: m.content }));

      if (USE_STREAM) {
        try {
          await doStreamChat(baseMessages, conversation, renderMessages);
        } catch (streamErr) {
          console.warn("Stream failed, falling back to direct:", streamErr);
          await doDirectChat(baseMessages, conversation, renderMessages);
        }
      } else {
        await doDirectChat(baseMessages, conversation, renderMessages);
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

  if (qa1) qa1.addEventListener("click", () => quickAsk("What location reflects the highest/lowest yield for material M335449?"));
  if (qa2) qa2.addEventListener("click", () => quickAsk("What item, produced at P8103, produces the highest/lowest yield?"));
  if (qa3) qa3.addEventListener("click", () => quickAsk("What month reflected the highest/lowest yield in the last 32 months?"));
  if (qa4) qa4.addEventListener("click", () => quickAsk("What location reflects the highest/lowest efficiency for material M335449?"));
  if (qa5) qa5.addEventListener("click", () => quickAsk("What's the earliest and latest date you have access to?"));
  if (qa6) qa6.addEventListener("click", () => quickAsk("Show scrapping totals by month for 2024"));

  // Initial render to toggle splash visibility on load
  renderMessages();
}
