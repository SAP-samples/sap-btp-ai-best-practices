import { afterEach, beforeEach, expect, it, vi } from "vitest";
import shellHtml from "../index.html?raw";
import shellCss from "../src/style.css?raw";
import { ecnChat } from "../src/components/ecn-chat.js";
import { mountShellChat, resetShellChat, setChatContext } from "../src/modules/shell-chat.js";

vi.mock("../src/components/ecn-chat.js", () => ({ ecnChat: vi.fn() }));
let width;
beforeEach(() => {
  resetShellChat();
  document.body.replaceChildren();
  vi.clearAllMocks();
  width = 390;
  vi.stubGlobal("matchMedia", vi.fn(query => ({ matches: query === "(min-width: 1400px)" && width >= 1400 })));
  // No provider, workspace API, conversation writes, or customer data in this seam.
  ecnChat.mockImplementation((body, id, signal, inspect, graphReady = true) => {
    const guidance = document.createElement("p");
    guidance.className = "chat-empty";
    guidance.textContent = id === null ? "Synthetic global empty chat" : "Synthetic scoped empty chat";
    const input = document.createElement("textarea");
    input.setAttribute("aria-label", "Synthetic question");
    input.disabled = !graphReady;
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    const clear = document.createElement("button");
    clear.textContent = "Synthetic clear";
    body.append(guidance, input, status, clear);
  });
});
afterEach(() => { resetShellChat(); vi.unstubAllGlobals(); });

function mount(nativeAction = true) {
  // Copy only shell markup: the application's module script is never mounted/executed.
  const template = document.createElement("template");
  template.innerHTML = shellHtml;
  document.body.append(template.content.querySelector("#app"));
  const app = document.querySelector("#app");
  if (nativeAction) {
    // Native focus seam for deterministic shell logic; actual UI5 forwarding/slots are E2E-owned.
    const existing = app.querySelector("#assistant-shell-action");
    const action = document.createElement("button");
    for (const attribute of existing.attributes) action.setAttribute(attribute.name, attribute.value);
    action.textContent = "Open ECN chat";
    existing.replaceWith(action);
  }
  mountShellChat(app);
  return { app, action: app.querySelector("#assistant-shell-action"), panel: app.querySelector("#ecn-assistant-panel") };
}
function key(key, shiftKey = false) {
  const event = new KeyboardEvent("keydown", { key, shiftKey, bubbles: true, cancelable: true });
  document.dispatchEvent(event);
  return event;
}

it("uses the public assistant slot with a differentiated chat icon", () => {
  const { app, action } = mount(false);
  const shell = app.querySelector("ui5-shellbar");
  expect([...shell.querySelectorAll('[slot="startButton"]')].map(node => node.id)).toEqual(["startButton"]);
  expect(action.tagName).toBe("UI5-BUTTON");
  expect(action.getAttribute("icon")).toBe("discussion-2");
  expect(action.getAttribute("design")).toBe("Transparent");
  expect(action.getAttribute("accessible-name")).toBe("Open ECN chat");
  expect(action.getAttribute("tooltip")).toBe("Open ECN chat");
  expect(app.querySelector("#startButton").getAttribute("accessible-name")).toBe("Toggle navigation");
  expect(action.getAttribute("slot")).toBe("assistant");
  expect(shellCss).not.toMatch(/#assistant-shell-action\s*\{\s*display:\s*none/);
  expect(shellCss).not.toContain(".assistant-launcher");
});

it.each([320, 390, 480, 600, 601, 1024, 1399, 1400, 1512])("keeps one stable action available open or closed at %i px without rebinding", viewportWidth => {
  width = viewportWidth;
  const { app, action, panel } = mount();
  const context = { id: "SIM-ECN", signal: new AbortController().signal, inspect: vi.fn(), graphReady: true };
  setChatContext(context);
  expect(panel.hidden).toBe(true);
  expect(panel.getAttribute("role")).toBe("complementary");
  expect(action.hidden).toBe(false);
  expect(action.accessibilityAttributes).toMatchObject({ controls: panel.id, expanded: "false" });
  expect(action.hasAttribute("aria-expanded")).toBe(false); // not host-only ARIA
  expect(app.querySelectorAll("#assistant-shell-action")).toHaveLength(1);
  expect(app.querySelectorAll(".assistant-launcher")).toHaveLength(0);
  const input = panel.querySelector("textarea");
  input.value = "Synthetic unsent draft";
  panel.querySelector(".assistant-close").click();
  expect(panel.hidden).toBe(true);
  expect(action.hidden).toBe(false);
  expect(action.accessibilityAttributes.expanded).toBe("false");
  expect(document.activeElement).toBe(action);
  action.click();
  expect(panel.hidden).toBe(false);
  expect(action.hidden).toBe(false);
  expect(action.accessibilityAttributes.expanded).toBe("true");
  expect(panel.querySelector("textarea")).toBe(input);
  expect(input.value).toBe("Synthetic unsent draft");
  action.click(); // shell action toggles without remounting
  expect(panel.hidden).toBe(true);
  action.click();
  expect(panel.hidden).toBe(false);
  expect(ecnChat).toHaveBeenCalledTimes(2); // initial global + scoped only
  expect(ecnChat.mock.lastCall.slice(0, 2)).toEqual([panel.querySelector(".assistant-body"), "SIM-ECN"]);
  expect(ecnChat.mock.lastCall[3]).toBe(context.inspect);
  expect(key("Escape").defaultPrevented).toBe(true);
  expect(document.activeElement).toBe(action);
  expect(panel.hidden).toBe(true);
});

it("leaves the underlying page available and respects nested-dialog Escape", () => {
  const { action, panel } = mount();
  action.click();
  expect(document.activeElement).toBe(panel.querySelector("textarea"));
  expect(document.querySelector(".assistant-backdrop")).toBeNull();
  const last = panel.querySelector(".assistant-body button");
  last.focus();
  expect(key("Tab").defaultPrevented).toBe(false);
  const nested = document.createElement("dialog"); nested.open = true; document.body.append(nested);
  expect(key("Escape").defaultPrevented).toBe(false);
  expect(panel.hidden).toBe(false);
  nested.remove();
  panel.querySelector(".assistant-close").click();
  expect(panel.hidden).toBe(true);
  expect(document.activeElement).toBe(action);
});

it("preserves empty, loading, error, success and unavailable child states across visibility changes", () => {
  const { action, panel } = mount();
  expect(panel.textContent).toContain("Synthetic global empty chat");
  const input = panel.querySelector("textarea");
  const status = panel.querySelector('[role="status"]');
  for (const message of ["Synthetic loading", "Synthetic error", "Synthetic success"]) {
    status.textContent = message;
    input.disabled = message === "Synthetic loading";
    action.click();
    expect(document.activeElement).toBe(input.disabled ? panel.querySelector(".assistant-close") : input);
    panel.querySelector(".assistant-close").click();
    expect(panel.querySelector('[role="status"]').textContent).toBe(message);
  }
  setChatContext({ id: "SIM-UNAVAILABLE", graphReady: false });
  action.click();
  expect(panel.querySelector("textarea").disabled).toBe(true);
  expect(document.activeElement).toBe(panel.querySelector(".assistant-close"));
});

it("keeps nonmodal semantics and drafts when the viewport changes", () => {
  const { app, action, panel } = mount();
  action.click();
  const input = panel.querySelector("textarea"); input.value = "Synthetic retained draft";
  for (const viewport of [1512, 600]) {
    width = viewport; window.dispatchEvent(new Event("resize"));
    expect(panel.hidden).toBe(false);
    expect(panel.getAttribute("role")).toBe("complementary");
    expect(panel.hasAttribute("aria-modal")).toBe(false);
    expect(action.accessibilityAttributes.hasPopup).toBeUndefined();
    expect(app.classList.contains("assistant-open")).toBe(true);
    expect(input.value).toBe("Synthetic retained draft");
  }
  expect(ecnChat).toHaveBeenCalledTimes(1);
});

it("aborts replaced scope, returns to global, and cleans up action/listeners on remount", () => {
  const { app, action, panel } = mount();
  const globalSignal = ecnChat.mock.lastCall[2];
  const first = new AbortController();
  setChatContext({ id: "SIM-A", signal: first.signal, graphReady: true });
  const scopedSignal = ecnChat.mock.lastCall[2];
  expect(globalSignal.aborted).toBe(true);
  const second = new AbortController();
  setChatContext({ id: "SIM-B", signal: second.signal, graphReady: true });
  expect(scopedSignal.aborted).toBe(true);
  first.abort();
  expect(ecnChat.mock.lastCall[1]).toBe("SIM-B");
  second.abort();
  expect(ecnChat.mock.lastCall[1]).toBeNull();
  const finalSignal = ecnChat.mock.lastCall[2];
  mountShellChat(app); // idempotent
  expect(app.querySelectorAll("#ecn-assistant-panel")).toHaveLength(1);
  resetShellChat();
  expect(finalSignal.aborted).toBe(true);
  expect(panel.isConnected).toBe(false);
  expect(action.hidden).toBe(true);
  action.click(); window.dispatchEvent(new Event("resize"));
  expect(key("Escape").defaultPrevented).toBe(false);
  expect(app.querySelectorAll("#ecn-assistant-panel")).toHaveLength(0);
  mountShellChat(app);
  action.click();
  expect(app.querySelectorAll("#ecn-assistant-panel")).toHaveLength(1);
  expect(app.querySelector("#ecn-assistant-panel").hidden).toBe(false);
  expect(app.querySelectorAll(".assistant-launcher")).toHaveLength(0);
});
