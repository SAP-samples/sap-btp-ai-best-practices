import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import { restoreSelection, persistSelection } from "./workspace/state.js";
import { mountPageLoading } from "./workspace/loading.js";
import { mountDatasets } from "./workspace/datasets.js";
import { mountOptimizer } from "./workspace/optimizer.js";
import { mountSettings } from "./workspace/settings.js";
import { mountChat } from "./workspace/chat.js";
import { mountNotice } from "./workspace/notice.js";
import { mountUserSession } from "./workspace/user_session.js";

const state = restoreSelection(localStorage);
// A page load owns one chat session. It is deliberately absent from localStorage,
// so reloads and new tabs cannot recover earlier conversation turns.
state.context_id = crypto.randomUUID();
const events = new EventTarget();
// The red banner; see notice.js for how API failures and page code bugs are told apart.
const notice = mountNotice(document.querySelector("#notice"));
/** Selections that move the user to another record; a banner about the previous one is stale then. */
const RECORD_KEYS = ["plant_profile_id", "dataset_id", "run_id"];
// Shows "Loading…" over the page while its initial API data is on the way (see loading.js).
const pageLoading = mountPageLoading({
  overlay: document.querySelector("#page-loading"),
  text: document.querySelector("#page-loading-text"),
  main: document.querySelector("#main"),
});
const app = {
  state,
  events,
  /**
   * Show the page loading indicator until `promise` settles; returns the same promise.
   * Pages wrap their initial data load in it and keep their own .catch(app.fail).
   */
  loading(promise, label) {
    return pageLoading.track(promise, label);
  },
  /** Apply selection identifiers and notify UI subscribers without persisting data. */
  select(values) {
    if (RECORD_KEYS.some((key) => key in values && values[key] !== state[key])) notice.clear();
    Object.assign(state, values);
    persistSelection(localStorage, state);
    events.dispatchEvent(new Event("selection"));
  },
  /** Show a failure in the dismissible global banner (API error or page code bug). */
  fail(error) {
    notice.show(error);
  },
  /** Hide the banner, for example when the user edits the form the error was about. */
  clearNotice() {
    notice.clear();
  },
};
let dispose = null;

/** Navigate between the workspace pages and dispose page-owned timers and charts. */
function route() {
  dispose?.();
  const page =
    ["workspace", "settings"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "datasets";
  // A banner from the previous page no longer applies.
  notice.clear();
  for (const link of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("[data-nav]"))) {
    link.classList.toggle("active", link.dataset.nav === page);
    if (link.dataset.nav === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  // The assistant only sees the selections that belong to the visible page.
  app.select({ page });
  // Forget the previous page's loads so a late response cannot hide this page's indicator.
  pageLoading.begin();
  dispose = ({workspace: mountOptimizer, settings: mountSettings, datasets: mountDatasets}[page])(
    document.querySelector("#main"),
    app,
  );
  window.scrollTo(0, 0);
}
// A promise rejection that no page handled would otherwise fail silently.
// Errors go to the banner (page code bugs are also logged there); anything else is only logged.
window.addEventListener("unhandledrejection", (event) => {
  if (event.reason instanceof Error) app.fail(event.reason);
  else console.error("Unhandled workspace rejection:", event.reason);
});
mountChat(app);
// Show who is signed in (the approuter has already signed the user in) and the sign-out link.
mountUserSession({
  name: document.querySelector("#user-name"),
  signOut: /** @type {HTMLAnchorElement} */ (document.querySelector("#sign-out")),
});
route();
window.addEventListener("hashchange", route);
