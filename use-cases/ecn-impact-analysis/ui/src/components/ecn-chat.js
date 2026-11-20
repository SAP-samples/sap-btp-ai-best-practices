import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents-icons/dist/discussion-2.js";
import "@ui5/webcomponents-icons/dist/paper-plane.js";
import { chat, globalChat } from "../services/workspace-api.js";
import { renderMarkdown } from "../modules/markdown.js";
import { button, element } from "./workspace-dom.js";

// Browser memory only: per-ECN messages survive route changes, never reload.
const conversations = new Map();

function ecnLinks(ecns) {
  const links = element("div", "", "chat-sources");
  links.append(element("small", "ECNs consulted:"));
  ecns.forEach(ecn => {
    const link = element("a", ecn.ecn_number);
    link.href = `/ecns/${encodeURIComponent(ecn.assessment_id)}`;
    links.append(link);
  });
  return links;
}

/** The existing scoped/global chat services in a minimal SAP-themed presentation. */
export function ecnChat(host, id, signal, inspectObject, graphReady = true) {
  const scoped = id !== null;
  const turns = conversations.get(id ?? "global") || [];
  conversations.set(id ?? "global", turns);
  let busy = false;
  let pendingText = "";

  const welcome = element("div", "", "chat-welcome");
  const illustration = document.createElement("ui5-icon");
  illustration.setAttribute("name", "discussion-2");
  illustration.setAttribute("aria-hidden", "true");
  welcome.append(illustration, element("h3", "How can I help?"));
  const scope = element("p", scoped
    ? "Ask what changes, which parts are affected, or where they are used."
    : "Ask about any stored ECN: materials, impacts, plants and reviews.", "assistant-scope");
  const messages = element("div", "", "chat-messages");
  messages.setAttribute("role", "log");
  messages.setAttribute("aria-label", "Conversation");
  messages.setAttribute("aria-live", "polite");
  const input = element("textarea");
  input.maxLength = 4000;
  input.rows = 1;
  input.placeholder = "Message ECN Assistant...";
  input.setAttribute("aria-label", scoped ? "Question about this ECN" : "Question about the stored ECNs");
  input.setAttribute("aria-describedby", "ecn-chat-scope");
  scope.id = "ecn-chat-scope";
  const status = element("p", "", "chat-status");
  status.setAttribute("role", "status");

  function appendMessage(turn) {
    const bubble = element("article", "", `chat-message chat-${turn.role}`);
    bubble.setAttribute("aria-label", turn.role === "user" ? "Your message" : "ECN Assistant response");
    bubble.append(renderMarkdown(turn.content));
    if (turn.sources?.length && inspectObject) {
      const sources = element("div", "", "chat-sources");
      sources.append(element("small", "Objects consulted:"));
      turn.sources.forEach(source => sources.append(button(source.number, () => inspectObject(source.number, source.technical))));
      bubble.append(sources);
    }
    if (turn.ecns?.length) bubble.append(ecnLinks(turn.ecns));
    messages.append(bubble);
  }

  function render() {
    const hasConversation = turns.length > 0 || busy;
    host.classList.toggle("chat-has-conversation", hasConversation);
    welcome.hidden = hasConversation;
    messages.hidden = !hasConversation;
    messages.replaceChildren();
    for (const turn of turns) appendMessage(turn);
    if (busy) {
      appendMessage({ role: "user", content: pendingText });
      messages.append(element("div", scoped ? "Checking this ECN…" : "Checking stored ECNs…", "chat-pending"));
    }
    messages.scrollTop = messages.scrollHeight;
  }

  const form = element("form", "", "chat-composer");
  const send = button("", () => form.requestSubmit());
  send.className = "chat-send";
  send.setAttribute("aria-label", "Send");
  send.title = "Send";
  const sendIcon = document.createElement("ui5-icon");
  sendIcon.setAttribute("name", "paper-plane");
  sendIcon.setAttribute("aria-hidden", "true");
  send.append(sendIcon);

  function syncInput() {
    input.disabled = busy || !graphReady;
    send.disabled = busy || !graphReady || !input.value.trim();
    input.style.height = "auto";
    input.style.height = `${Math.max(38, Math.min(input.scrollHeight, 120))}px`;
  }
  input.addEventListener("input", syncInput);
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || busy || !graphReady || signal.aborted) return;
    const restoreInputFocus = host.contains(document.activeElement);
    const prior = turns.slice(-12).map(t => ({ role: t.role, content: t.content.slice(0, 12000) }));
    while (prior.reduce((sum, t) => sum + t.content.length, 0) > 24000) prior.shift();
    busy = true;
    pendingText = text;
    form.setAttribute("aria-busy", "true");
    status.textContent = "";
    syncInput();
    render();
    try {
      const response = scoped
        ? await chat(id, { assessment_id: id, message: text, history: prior }, signal)
        : await globalChat({ message: text, history: prior }, signal);
      if (signal.aborted) return;
      turns.push({ role: "user", content: text }, {
        role: "assistant", content: response.answer,
        sources: scoped ? response.consulted_objects : [], ecns: response.consulted_ecns,
      });
      while (turns.length > 40) turns.splice(0, 2);
      input.value = "";
      if (response.review_saved) window.dispatchEvent(new CustomEvent("ecn-review-saved", { detail: { id } }));
    } catch (error) {
      if (!signal.aborted && error.name !== "AbortError") status.textContent = error.message;
    } finally {
      if (!signal.aborted) {
        busy = false;
        pendingText = "";
        form.setAttribute("aria-busy", "false");
        syncInput();
        render();
        // Closing during a request must not steal focus when the answer arrives.
        if (!host.closest("[hidden]") && restoreInputFocus
          && (host.contains(document.activeElement) || document.activeElement === document.body)) input.focus();
      }
    }
  });

  const actions = element("div", "", "chat-composer-actions");
  actions.append(send);
  form.append(input, actions);
  const footer = element("footer", "", "chat-footer");
  footer.append(form, status, element("p", "AI can make mistakes. Verify results.", "chat-disclaimer"));
  host.append(welcome, scope, messages, footer);
  if (!graphReady) status.textContent = "Chat is unavailable until the RDF projection is verified.";
  syncInput();
  render();
}
