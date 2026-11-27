import { marked, Renderer } from "marked";
import { streamNDJSON, request } from "../services/api.js";
import { el } from "./dom.js";
import { runProgress } from "./progress.js";
import { changeRequest, decisionNote, renderProposalCard } from "./proposals.js";
import { importDecisionNote, pendingImportContext, renderImportSettingsCard } from "./import_settings.js";

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
  // Synchronous: no async extension is registered, so parse() returns a string.
  return /** @type {string} */ (marked.parse(String(text || ""), {
    gfm: true,
    breaks: true,
    renderer: markdownRenderer,
  }));
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

/** Selection keys the assistant may see on each page; the Optimizer sees all of them. */
export const PAGE_CONTEXT_KEYS = {
  datasets: ["plant_profile_id", "dataset_id"],
  settings: ["plant_profile_id"],
  workspace: ["plant_profile_id", "dataset_id", "draft_id", "run_id", "point_index"],
};

/**
 * Build the selection the assistant receives for the visible page.
 * Input: app state with `page` and the persisted selection identifiers.
 * Output: every context key; keys outside the page (or unset) are explicit null,
 * because the server merges each turn's selection into its session context and an
 * omitted key would keep a stale value from another page. Unknown pages send none.
 */
export function pageContext(state) {
  const visible = PAGE_CONTEXT_KEYS[state.page] || [];
  return Object.fromEntries(
    PAGE_CONTEXT_KEYS.workspace.map((key) => [key, visible.includes(key) ? (state[key] ?? null) : null]),
  );
}

/** Chip captions for context keys; names replace IDs when they can be looked up. */
const CHIP_CAPTIONS = {
  plant_profile_id: "Plant profile",
  dataset_id: "Dataset",
  draft_id: "Draft",
  run_id: "Run",
  point_index: "Solution point",
};

/** Connect a collapsible assistant to the selection of the visible page. */
export function mountChat(app) {
  /** @type {HTMLElement} */
  const panel = document.querySelector("#chat-panel");
  /** @type {HTMLElement} */
  const messages = document.querySelector("#chat-messages");
  /** @type {HTMLTextAreaElement} */
  const input = document.querySelector("#chat-input");
  /** @type {UI5ButtonElement} */
  const send = document.querySelector("#chat-send");
  let pending = false;
  const history = [];
  let contextSummary = null;
  let generation = 0;
  let controller = null;

  /** Stop this chat, clear its page history and create a fresh context without changing jobs. */
  async function reset() {
    /** @type {UI5ButtonElement} */
    const button = document.querySelector("#reset-chat");
    button.disabled = true;
    send.disabled = true;
    ++generation;
    controller?.abort();
    try {
      await request(`/api/chat/${encodeURIComponent(app.state.context_id)}/reset`, "POST", {});
      contextSummary = clearBrowserContext(history);
      messages.replaceChildren();
      app.clearNotice();
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

  // Readable names per "key:id"; a failed lookup resolves to null and the chip keeps the ID.
  const names = new Map();
  let chipRender = 0;

  /** Resolve a profile ("name · rN") or dataset name; other keys have no name. */
  function lookupName(key, id) {
    const cacheKey = `${key}:${id}`;
    if (!names.has(cacheKey)) {
      const path =
        key === "plant_profile_id" ? `/api/plant-profiles/${encodeURIComponent(id)}`
        : key === "dataset_id" ? `/api/datasets/${encodeURIComponent(id)}`
        : null;
      names.set(cacheKey, path
        ? request(path).then((item) =>
            key === "plant_profile_id" ? `${item.name} · r${item.revision}` : item.name || null,
          ).catch(() => null)
        : Promise.resolve(null));
    }
    return names.get(cacheKey);
  }

  /** Forget profile labels after a save or an approved change bumps the revision. */
  function forgetProfileNames() {
    for (const cacheKey of names.keys())
      if (cacheKey.startsWith("plant_profile_id:")) names.delete(cacheKey);
    context();
  }

  /** Show the page-scoped selection the next turn will send, by name where possible. */
  function context() {
    const render = ++chipRender;
    const chips = document.querySelector("#chat-context");
    chips.replaceChildren();
    for (const [key, value] of Object.entries(pageContext(app.state))) {
      if (value == null) continue;
      const chip = el("span", `${CHIP_CAPTIONS[key]}: ${value}`, "context-chip");
      chips.append(chip);
      lookupName(key, value).then((name) => {
        // Ignore lookups that finish after a newer selection re-rendered the chips.
        if (name && render === chipRender) chip.textContent = `${CHIP_CAPTIONS[key]}: ${name}`;
      });
    }
    const pendingImport = pendingImportContext(app.state, app.pendingImport);
    if (pendingImport)
      chips.append(el("span", `Pending import: ${pendingImport.primary_filename}`, "context-chip"));
    if (!chips.children.length)
      chips.append(el("span", "No selection on this page", "muted"));
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

  /** Save an approved profile change, refresh Settings and tell the next turn about it. */
  // Extraction settings cards apply to the Datasets page's next upload; nothing is saved.
  const importHandlers = {
    async apply(proposal) {
      if (!app.importSettings)
        throw new Error("Open the Datasets page and choose the workbook again, then ask the Assistant.");
      app.importSettings.apply(proposal);
      history.push({ role: "user", content: importDecisionNote(proposal, true) });
    },
    async reject(proposal) {
      history.push({ role: "user", content: importDecisionNote(proposal, false) });
    },
  };

  const proposalHandlers = {
    async approve(proposal) {
      const { path, body } = changeRequest(proposal.profile_id, proposal.profile_revision, proposal.change);
      const saved = await request(path, "POST", body);
      app.events.dispatchEvent(new CustomEvent("profile-changed", { detail: { profile_id: proposal.profile_id } }));
      history.push({ role: "user", content: decisionNote(proposal, true, saved.revision) });
      return saved;
    },
    async reject(proposal) {
      history.push({ role: "user", content: decisionNote(proposal, false) });
    },
  };

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
          context_id: selection.context_id,
          ...pageContext(selection),
          pending_import: pendingImportContext(selection, app.pendingImport),
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
          // The assistant stored a decision or generated SAP settings for the run on screen.
          if ((chunk.type === "decision_selected" || chunk.type === "sap_settings_generated") && chunk.run_id === app.state.run_id) {
            app.events.dispatchEvent(new CustomEvent("sap-settings-changed", { detail: { run_id: chunk.run_id, type: chunk.type } }));
          }
          if (chunk.type === "extraction_settings_proposal") {
            messages.append(renderImportSettingsCard(chunk, importHandlers));
            messages.scrollTop = messages.scrollHeight;
          }
          if (chunk.type === "profile_proposal") {
            messages.append(renderProposalCard(chunk, proposalHandlers));
            messages.scrollTop = messages.scrollHeight;
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

  /**
   * Ask the assistant a prepared question from a page (e.g. a dataset finding).
   * Input: the question text. Opens the panel and sends it as the user's turn with
   * the visible page's selection. While another turn is running, the question is
   * only placed in the input box so the user can send it afterwards.
   */
  app.askAssistant = (question) => {
    toggle(false);
    input.value = question;
    if (pending) {
      input.focus();
      return;
    }
    submit();
  };
  /** Open the assistant panel and put the cursor in its input box. Used by page "Ask the Assistant" buttons. */
  app.openAssistant = () => {
    toggle(false);
    input.focus();
  };
  app.events.addEventListener("selection", context);
  app.events.addEventListener("pending-import", context);
  app.events.addEventListener("profile-saved", forgetProfileNames);
  app.events.addEventListener("profile-changed", forgetProfileNames);
  context();
}
