import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents-icons/dist/discussion-2.js";
import "@ui5/webcomponents-icons/dist/decline.js";
import { ecnChat } from "../components/ecn-chat.js";
import { button, element } from "../components/workspace-dom.js";

/** One floating assistant across routes, with browser-memory conversations per ECN. */
let parts = null;
let binding = null;

function applyPresentation(open) {
  if (!parts) return;
  const { panel, shellAction, app } = parts;
  panel.hidden = !open;
  if (shellAction) {
    shellAction.hidden = false;
    // UI5 forwards these public attributes to the actual focusable button.
    shellAction.accessibilityAttributes = {
      ...shellAction.accessibilityAttributes,
      expanded: String(open),
      controls: panel.id,
    };
  }
  app.classList.toggle("assistant-open", open);
}

function setOpen(open, moveFocus = false) {
  if (!parts) return;
  applyPresentation(open);
  if (moveFocus) {
    const input = parts.panel.querySelector("textarea");
    const target = open
      ? (input && !input.disabled ? input : parts.close)
      : parts.shellAction;
    target?.focus();
  }
}

function handlePanelKeydown(event) {
  if (!parts || parts.panel.hidden || event.key !== "Escape") return;
  // Source-inspection dialogs own Escape while they are open.
  if (document.querySelector("dialog[open], ui5-dialog[open]")) return;
  event.preventDefault();
  setOpen(false, true);
}

/** Mount the assistant using the ShellBar's public assistant slot. */
export function mountShellChat(app) {
  if (!app || parts) return;
  const panel = element("aside", "", "ecn-assistant");
  panel.id = "ecn-assistant-panel";
  panel.setAttribute("role", "complementary");
  panel.setAttribute("aria-labelledby", "ecn-assistant-title");
  const title = element("h2", "", "assistant-title");
  title.id = "ecn-assistant-title";
  const icon = document.createElement("ui5-icon");
  icon.setAttribute("name", "discussion-2");
  icon.setAttribute("aria-hidden", "true");
  title.append(icon, document.createTextNode("ECN Assistant"));
  const close = button("", () => setOpen(false, true));
  close.className = "assistant-close";
  close.setAttribute("aria-label", "Close chat");
  close.title = "Close chat";
  const closeIcon = document.createElement("ui5-icon");
  closeIcon.setAttribute("name", "decline");
  closeIcon.setAttribute("aria-hidden", "true");
  close.append(closeIcon);
  const header = element("header", "", "assistant-header");
  header.append(title, close);
  const body = element("div", "", "assistant-body");
  panel.append(header, body);

  const shellAction = app.querySelector("#assistant-shell-action");
  const handleShellAction = () => setOpen(panel.hidden, true);
  shellAction?.addEventListener("click", handleShellAction);
  app.append(panel);
  document.addEventListener("keydown", handlePanelKeydown);
  parts = { panel, shellAction, handleShellAction, close, body, app };
  setChatContext(null);
  setOpen(false);
}

/** Bind to an open ECN, or to all stored ECNs when context is null. */
export function setChatContext(context) {
  if (!parts) return;
  binding?.abort();
  const own = new AbortController();
  binding = own;
  parts.body.replaceChildren();
  if (!context) {
    ecnChat(parts.body, null, own.signal);
    return;
  }
  context.signal?.addEventListener("abort", () => {
    if (binding === own) setChatContext(null);
  }, { once: true });
  ecnChat(parts.body, context.id, own.signal, context.inspect, context.graphReady);
}

/** Remove the shell chat and listeners so tests can mount a fresh one. */
export function resetShellChat() {
  binding?.abort();
  binding = null;
  document.removeEventListener("keydown", handlePanelKeydown);
  if (parts) {
    parts.shellAction?.removeEventListener("click", parts.handleShellAction);
    if (parts.shellAction) {
      parts.shellAction.hidden = true;
      parts.shellAction.accessibilityAttributes = {
        ...parts.shellAction.accessibilityAttributes, expanded: "false", controls: undefined,
      };
    }
    parts.panel.remove();
    parts.app.classList.remove("assistant-open");
  }
  parts = null;
}
