import "@ui5/webcomponents/dist/Button.js";
import { restoreSelection, persistSelection } from "./workspace/state.js";
import { mountDatasets } from "./workspace/datasets.js";
import { mountOptimizer } from "./workspace/optimizer.js";
import { mountSettings } from "./workspace/settings.js";
import { mountChat } from "./workspace/chat.js";

const state = restoreSelection(localStorage);
// A page load owns one chat session. It is deliberately absent from localStorage,
// so reloads and new tabs cannot recover earlier conversation turns.
state.context_id = crypto.randomUUID();
const events = new EventTarget();
const app = {
  state,
  events,
  /** Apply selection identifiers and notify UI subscribers without persisting data. */
  select(values) {
    Object.assign(state, values);
    persistSelection(localStorage, state);
    events.dispatchEvent(new Event("selection"));
  },
  /** Display readable request errors in a dismissible global live region. */
  fail(error) {
    const notice = document.querySelector("#notice");
    notice.textContent = error.message || String(error);
    notice.hidden = false;
    notice.onclick = () => {
      notice.hidden = true;
    };
  },
};
let dispose = null;

/** Navigate between the workspace pages and dispose page-owned timers and charts. */
function route() {
  dispose?.();
  const page =
    ["workspace", "settings"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "datasets";
  for (const link of document.querySelectorAll("[data-nav]")) {
    link.classList.toggle("active", link.dataset.nav === page);
    if (link.dataset.nav === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  dispose = ({workspace: mountOptimizer, settings: mountSettings, datasets: mountDatasets}[page])(
    document.querySelector("#main"),
    app,
  );
  window.scrollTo(0, 0);
}
mountChat(app);
route();
window.addEventListener("hashchange", route);
