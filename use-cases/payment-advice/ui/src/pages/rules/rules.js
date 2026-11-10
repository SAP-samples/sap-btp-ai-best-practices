/* Screen 2 - Deduction Rules Assistant: a multi-turn chat over the UC-02
   rules-authoring agent. Attach PDF/Word/Excel rule documents (paperclip), ask
   about a customer's existing rules, and confirm to persist a playbook to HANA. */

import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/FileUploader.js";

import { renderSafeMarkdown } from "../../services/safe-markdown.js";

import { upload } from "../../services/api.js";

/** Stable per-tab session id so history survives in-app navigation. */
function getSessionId() {
  const KEY = "rulesChatSessionId";
  let id = sessionStorage.getItem(KEY);
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) || `s-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
    sessionStorage.setItem(KEY, id);
  }
  return id;
}

/** Append a chat bubble; returns the bubble element for later replacement. */
function addBubble(log, role, htmlOrText, { isHtml = false } = {}) {
  const bubble = document.createElement("div");
  bubble.className = `rules-bubble rules-${role}`;
  bubble.setAttribute("role", "article");
  bubble.setAttribute("aria-label", role === "user" ? "You" : "Assistant");
  if (isHtml) {
    bubble.innerHTML = htmlOrText;
  } else {
    bubble.textContent = htmlOrText;
  }
  log.appendChild(bubble);
  log.scrollTop = log.scrollHeight;
  return bubble;
}

export default function initRulesPage() {
  const log = document.getElementById("rules-log");
  const input = document.getElementById("rules-input");
  const sendBtn = document.getElementById("rules-send");
  const fileUploader = document.getElementById("rules-file");
  const chips = document.getElementById("rules-attachments");
  if (!log || !input || !sendBtn) return;

  const sessionId = getSessionId();
  let pending = []; // File objects staged for the next message

  if (!log.childElementCount) {
    addBubble(log, "assistant", "Hi! Ask me for a customer's deduction rules, or attach rule documents and I'll interpret them.");
  }

  // Attachment chips -------------------------------------------------------
  function renderChips() {
    chips.innerHTML = "";
    pending.forEach((file, index) => {
      const chip = document.createElement("span");
      chip.className = "rules-chip";
      const label = document.createElement("span");
      label.textContent = file.name;
      const remove = document.createElement("button");
      remove.className = "rules-chip-x";
      remove.textContent = "×"; // ×
      remove.title = "Remove";
      remove.addEventListener("click", () => {
        pending.splice(index, 1);
        renderChips();
      });
      chip.append(label, remove);
      chips.appendChild(chip);
    });
  }

  const onFiles = () => {
    const picked = Array.from(fileUploader.files || []);
    for (const f of picked) {
      if (!pending.some((p) => p.name === f.name)) pending.push(f);
    }
    renderChips();
  };
  fileUploader.addEventListener("change", onFiles);
  fileUploader.addEventListener("ui5-change", onFiles);

  // Send -------------------------------------------------------------------
  async function send() {
    const message = (input.value || "").trim();
    if (!message && pending.length === 0) return;

    // User bubble (message + any attachment names).
    const attachNote = pending.length ? `\n\nAttached: ${pending.map((f) => f.name).join(", ")}` : "";
    addBubble(log, "user", `${message}${attachNote}`);

    const form = new FormData();
    form.append("message", message);
    form.append("session_id", sessionId);
    for (const f of pending) form.append("files", f, f.name);

    // Reset the composer immediately; the request is in flight.
    input.value = "";
    const sentFilesCount = pending.length;
    pending = [];
    renderChips();
    fileUploader.value = "";

    sendBtn.loading = true;
    const typing = addBubble(log, "assistant", "…");

    try {
      const res = await upload("/api/payment-advice/rules-chat", form);
      renderSafeMarkdown(typing, res.reply || "(no reply)");

      const tools = (res.tools_called || []).map((t) => t.tool).filter(Boolean);
      if (tools.length) {
        const note = document.createElement("div");
        note.className = "rules-tools";
        note.textContent = `used: ${[...new Set(tools)].join(", ")}`;
        typing.appendChild(note);
      }
      log.scrollTop = log.scrollHeight;
    } catch (err) {
      typing.textContent = `Error: ${err.message}`;
      typing.classList.add("rules-error");
      // Restore the staged file count hint (files themselves are gone from the input).
      if (sentFilesCount) {
        addBubble(log, "assistant", "Your attachments were cleared; please re-attach and try again.");
      }
    } finally {
      sendBtn.loading = false;
    }
  }

  sendBtn.addEventListener("click", send);

  // Enter to send, Shift+Enter for newline.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
}
