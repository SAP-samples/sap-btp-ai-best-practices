import { marked, Renderer } from "marked";
import { streamNDJSON, request } from "../services/api.js";
import { el } from "./dom.js";
import { runProgress } from "./progress.js";

/** Escape text used when raw HTML or unsafe Markdown URLs are encountered. */
function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const markdownRenderer = new Renderer();
markdownRenderer.html = (html) => escapeHTML(html);
markdownRenderer.image = (_href, _title, text) => escapeHTML(text);
markdownRenderer.link = (href, title, text) => {
  if (!/^https?:\/\//i.test(href)) return text;
  const titleAttribute = title ? ` title="${escapeHTML(title)}"` : "";
  return `<a href="${escapeHTML(href)}"${titleAttribute} target="_blank" rel="noopener noreferrer">${text}</a>`;
};

/** Render GFM from the assistant while neutralizing model-authored HTML and URLs. */
export function formatChatMarkdown(text) {
  return marked.parse(String(text || ""), {
    gfm: true,
    breaks: true,
    renderer: markdownRenderer,
  });
}

/** Retain the current request and the preceding complete exchange after compaction. */
export function applyBrowserContextSummary(history, summary) {
  history.splice(0, Math.max(0, history.length - 3));
  return summary;
}

/** Clear all page-only conversation material; reloads naturally do the same. */
export function clearBrowserContext(history) {
  history.length = 0;
  return null;
}

/** Connect a collapsible assistant to the selected dataset, draft, run and point. */
export function mountChat(app) {
  const panel = document.querySelector("#chat-panel");
  const messages = document.querySelector("#chat-messages");
  const input = document.querySelector("#chat-input");
  const send = document.querySelector("#chat-send");
  let pending = false;
  const history = [];
  let contextSummary = null;
  let generation = 0;
  let controller = null;

  /** Stop this chat, clear its page history and create a fresh context without changing jobs. */
  async function reset() {
    const button = document.querySelector("#reset-chat");
    button.disabled = true;
    send.disabled = true;
    ++generation;
    controller?.abort();
    try {
      await request(`/api/chat/${encodeURIComponent(app.state.context_id)}/reset`, "POST", {});
      contextSummary = clearBrowserContext(history);
      messages.replaceChildren();
      document.querySelector("#notice").hidden = true;
      input.value = "";
      app.select({ context_id: crypto.randomUUID() });
      message("assistant", "New conversation started. Workspace selections remain available; submitted optimizer runs continue independently.");
    } catch (error) {
      app.fail(error);
    } finally {
      pending = false;
      send.disabled = false;
      button.disabled = false;
    }
  }
  document.querySelector("#reset-chat").addEventListener("click", reset);

  /** Update context chips after every manual or assistant selection change. */
  function context() {
    const chips = document.querySelector("#chat-context");
    chips.replaceChildren();
    for (const key of ["plant_profile_id", "dataset_id", "draft_id", "run_id", "point_index"])
      if (app.state[key] != null)
        chips.append(
          el(
            "span",
            `${key.replace("_id", "")}: ${app.state[key]}`,
            "context-chip",
          ),
        );
    if (!chips.children.length)
      chips.append(el("span", "No dataset selected", "muted"));
  }

  /** Render assistant Markdown; keep user/tool messages as literal text. */
  function message(role, text) {
    const item = el("div", null, `chat-message ${role}`);
    const body = el("div", null, "chat-message-body");
    if (role === "assistant") body.innerHTML = formatChatMarkdown(text);
    else body.textContent = text;
    item.append(
      el(
        "strong",
        role === "user"
          ? "You"
          : role === "tool"
            ? "Tool activity"
            : "Assistant",
      ),
      body,
    );
    messages.append(item);
    messages.scrollTop = messages.scrollHeight;
  }

  /** Toggle the assistant while keeping the full optimizer workspace accessible. */
  function toggle(force) {
    const closed = force ?? !panel.hidden;
    panel.hidden = closed;
    document
      .querySelector(".app-layout")
      .classList.toggle("chat-closed", closed);
    document
      .querySelector("#toggle-chat")
      .setAttribute("aria-expanded", String(!closed));
  }
  document
    .querySelector("#toggle-chat")
    .addEventListener("click", () => toggle());
  document
    .querySelector("#close-chat")
    .addEventListener("click", () => toggle(true));

  /** Send a turn with a context snapshot and apply server-authorized draft/run events. */
  async function submit(event) {
    event?.preventDefault();
    if (pending || !input.value.trim()) return;
    const turnGeneration = generation;
    controller = new AbortController();
    const text = input.value.trim();
    const selection = { ...app.state };
    const priorTurns = history.slice();
    pending = true;
    send.disabled = true;
    input.value = "";
    message("user", text);
    history.push({ role: "user", content: text });
    history.splice(0, Math.max(0, history.length - 40));
    const activity = el("p", "Thinking…", "muted");
    messages.append(activity);
    try {
      await streamNDJSON("/api/chat", {
        body: {
          message: text,
          history: priorTurns,
          ...selection,
          ...(contextSummary ? { context_summary: contextSummary } : {}),
        },
        signal: controller.signal,
        onChunk: async (chunk) => {
          if (generation !== turnGeneration) return;
          if (chunk.type === "run_progress") activity.textContent = `${runProgress(chunk).text} You can start a new conversation while the run continues.`;
          if (chunk.type === "assistant") {
            if (chunk.history_compacted && chunk.context_summary) {
              // The server summary covers older page turns. Keep only the latest
              // completed exchange plus the current user request verbatim.
              contextSummary = applyBrowserContextSummary(
                history,
                chunk.context_summary,
              );
            }
            message("assistant", chunk.text || chunk.message || "");
            history.push({
              role: "assistant",
              content: chunk.text || chunk.message || "",
            });
            history.splice(0, Math.max(0, history.length - 40));
          }
          if (chunk.type === "tool_call" || chunk.type === "tool_result") {
            activity.textContent = `${chunk.type === "tool_call" ? "Using" : "Finished"} ${chunk.name || chunk.tool || "optimizer tool"}…`;
          }
          // Do not attach a response from a previous dataset to the user's new selection.
          if (
            chunk.type === "draft_changed" &&
            selection.dataset_id === app.state.dataset_id
          ) {
            app.select({ draft_id: chunk.draft_id || app.state.draft_id });
            app.events.dispatchEvent(new Event("draft-changed"));
            message(
              "tool",
              "The draft was updated. The configuration form is reloading the latest revision; review it before launch.",
            );
          }
          if (
            chunk.type === "run_created" &&
            selection.dataset_id === app.state.dataset_id
          ) {
            app.select({ run_id: chunk.run_id, point_index: null });
            app.events.dispatchEvent(new Event("run-created"));
          }
          if (chunk.type === "error")
            throw new Error(
              chunk.text ||
                chunk.message ||
                chunk.error ||
                "Assistant request failed.",
            );
        },
      });
    } catch (error) {
      if (generation !== turnGeneration) return;
      app.fail(error);
      message("assistant", `Request could not finish: ${error.message}`);
    } finally {
      activity.remove();
      if (generation === turnGeneration) {
        pending = false;
        send.disabled = false;
      }
    }
  }
  send.addEventListener("click", submit);
  document.querySelector("#chat-form").addEventListener("submit", submit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) submit(event);
  });
  app.events.addEventListener("selection", context);
  context();
}
