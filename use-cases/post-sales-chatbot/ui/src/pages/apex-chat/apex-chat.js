/* Apex Chat UI5 components */
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents-icons/dist/paper-plane.js";
import "@ui5/webcomponents-icons/dist/refresh.js";
import "@ui5/webcomponents-icons/dist/car-rental.js";
import { marked } from "marked";
import dayjs from "dayjs";
import calendar from "dayjs/plugin/calendar.js";

import { request, streamNDJSON, setSessionId, getSessionId } from "../../services/api.js";

// API calls for Apex chat
async function postApexChat(message) {
  return await request(`/api/apex/chat`, "POST", { message });
}

async function streamApexChat(message, onChunk) {
  return await streamNDJSON(`/api/apex/chat/stream`, {
    method: "POST",
    body: { message },
    onChunk
  });
}

async function resetApexChat() {
  return await request(`/api/apex/reset`, "POST");
}

// Toggle streaming vs direct API; defaults to streaming
const USE_STREAM = true;

async function doStreamChat(message, conversation, onRender) {
  let accumulated = "";
  let toolHint = "";

  await streamApexChat(message, (evt) => {
    if (evt.type === "assistant") {
      accumulated = evt.content || "";
      toolHint = "";
    }
    if (evt.type === "tool") {
      const name = evt.name || "tool";
      // Format tool name nicely
      const displayName = name.replace(/_/g, " ").replace(/tool$/, "").trim();
      toolHint = `Using ${displayName}...`;
      const lastMsg = conversation[conversation.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.toolActive = true;
      }
    }
    if (evt.type === "tools" && Array.isArray(evt.items)) {
      const names = evt.items.map((it) => it && it.name).filter(Boolean);
      const displayNames = names.map(n => n.replace(/_/g, " ").replace(/tool$/, "").trim());
      toolHint = displayNames.length > 1 ? `Using ${displayNames.join(", ")}...` : displayNames[0] ? `Using ${displayNames[0]}...` : "";
      const lastMsg = conversation[conversation.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.toolActive = true;
      }
    }
    if (evt.type === "error") {
      accumulated = `An error occurred: ${evt.error}`;
      toolHint = "";
    }

    // Update session ID if present
    if (evt.session_id) {
      setSessionId(evt.session_id);
    }

    // Update the typing bubble in place
    const last = conversation[conversation.length - 1];
    if (last && last.role === "assistant") {
      if (accumulated) {
        last.content = accumulated;
        last.hint = undefined;
        last.toolActive = undefined;
      } else {
        last.content = "...";
        last.hint = toolHint || undefined;
      }
    }
    onRender();
  });

  // Finalize
  const last = conversation[conversation.length - 1];
  if (last && last.role === "assistant" && (!last.content || last.content === "...")) {
    last.content = accumulated || last.content;
    last.hint = undefined;
  }
  onRender();
}

async function doDirectChat(message, conversation, onRender) {
  const data = await postApexChat(message);

  // Update session ID if present
  if (data.session_id) {
    setSessionId(data.session_id);
  }

  const last = conversation[conversation.length - 1];
  if (last && last.role === "assistant") {
    last.content = data.text;
    last.hint = undefined;
  } else {
    conversation.push({ role: "assistant", content: data.text });
  }
  onRender();
}

export default function initApexChatPage() {
  dayjs.extend(calendar);

  const sendButton = document.getElementById("send-button");
  const resetButton = document.getElementById("reset-button");
  const chatTimeline = document.getElementById("chat-timeline");
  const chatSplash = document.getElementById("chat-splash");
  const input = document.getElementById("message-input");
  const qa1 = document.getElementById("qa-1");
  const qa2 = document.getElementById("qa-2");
  const qa3 = document.getElementById("qa-3");
  const qa4 = document.getElementById("qa-4");
  const qa5 = document.getElementById("qa-5");

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
    const hasRealMessages = conversation.some((m) => m.content !== "...");
    if (chatSplash) {
      if (!hasRealMessages) {
        chatSplash.classList.add("visible");
      } else {
        chatSplash.classList.remove("visible");
      }
    }

    for (const msg of conversation) {
      if (msg.content === "...") {
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

    // Push user message with timestamp
    conversation.push({ role: "user", content: text, ts: Date.now() });
    input.value = "";
    renderMessages();

    // Show typing placeholder
    conversation.push({ role: "assistant", content: "..." });
    renderMessages();

    if (sendButton) sendButton.disabled = true;
    if (input) input.disabled = true;

    try {
      if (USE_STREAM) {
        try {
          await doStreamChat(text, conversation, renderMessages);
        } catch (streamErr) {
          console.warn("Stream failed, falling back to direct:", streamErr);
          await doDirectChat(text, conversation, renderMessages);
        }
      } else {
        await doDirectChat(text, conversation, renderMessages);
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

  async function resetConversation() {
    conversation.length = 0;

    // Show loading indicator immediately to skip splash screen
    conversation.push({ role: "assistant", content: "..." });
    renderMessages();

    try {
      const data = await resetApexChat();
      if (data.session_id) {
        setSessionId(data.session_id);
      }
      // Replace loading placeholder with actual message
      const last = conversation[conversation.length - 1];
      if (last && last.role === "assistant") {
        last.content = data.message;
        last.ts = Date.now();
      }
      renderMessages();
    } catch (error) {
      console.error("Reset error:", error);
      // Replace loading placeholder with fallback greeting
      const last = conversation[conversation.length - 1];
      if (last && last.role === "assistant") {
        last.content = "Hello! I'm Apex Assistant, your customer service helper for Apex Automotive Services. How can I assist you today? Please share your name, email, or phone number to get started.";
        last.ts = Date.now();
      }
      renderMessages();
    }
  }

  // Event listeners
  if (sendButton) {
    sendButton.addEventListener("click", sendMessage);
    sendButton.disabled = !(input && (input.value || "").trim().length > 0);
  }

  if (resetButton) {
    resetButton.addEventListener("click", resetConversation);
  }

  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        if (e.shiftKey) {
          // Allow newline
          return;
        }
        // Plain Enter: send
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

  // Quick action handlers
  function quickAsk(text) {
    if (!input) return;
    input.value = text;
    sendMessage();
    if (sendButton) sendButton.disabled = false;
    input.focus();
  }

  if (qa1) qa1.addEventListener("click", () => quickAsk("Show me my vehicles"));
  if (qa2) qa2.addEventListener("click", () => quickAsk("Show me my service history"));
  if (qa3) qa3.addEventListener("click", () => quickAsk("What service do you recommend for my car?"));
  if (qa4) qa4.addEventListener("click", () => quickAsk("Are there any promotions for my vehicle?"));
  if (qa5) qa5.addEventListener("click", () => quickAsk("I'd like to schedule an appointment"));

  // Initialize conversation on page load
  resetConversation();
}
