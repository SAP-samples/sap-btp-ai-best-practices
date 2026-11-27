/**
 * Page loading indicator.
 *
 * Every workspace page paints its HTML at once and then fetches its data from the API.
 * Until the data arrives the page looks ready but is empty. This module shows a visible
 * "Loading…" indicator during that wait and blocks the page so that nobody edits a form
 * which the arriving data will overwrite.
 *
 * Two parts:
 * - createLoadTracker: pure bookkeeping (pending loads, delay, page generations). No DOM,
 *   so node:test can cover it (ui/tests/loading.test.js).
 * - mountPageLoading: connects a tracker to the overlay element in index.html and to <main>.
 */

/** Wait this long before showing the indicator, so fast responses do not make it flash. */
export const DEFAULT_LOADING_DELAY_MS = 250;

/**
 * Count pending page loads and decide when the indicator is visible.
 *
 * @param {object} options
 * @param {(label: string) => void} options.show Called when the indicator must appear, or
 *   when its text changes while it is visible.
 * @param {() => void} options.hide Called when the indicator must disappear.
 * @param {number} [options.delay] Milliseconds a load must last before the indicator appears.
 * @param {Function} [options.setTimer] Timer function (setTimeout by default; tests inject a fake).
 * @param {Function} [options.clearTimer] Timer cancel function (clearTimeout by default).
 * @returns {{begin: () => void, track: (work: any, label?: string) => Promise<any>, readonly pending: number, readonly visible: boolean}}
 *   begin() starts a new page and forgets the loads of the previous one.
 *   track() registers a load and returns the same promise, so callers keep their own error handling.
 */
export function createLoadTracker({
  show,
  hide,
  delay = DEFAULT_LOADING_DELAY_MS,
  setTimer = setTimeout,
  clearTimer = clearTimeout,
}) {
  // Each navigation increments the generation. A load that settles after the user has
  // left its page belongs to an older generation and must not touch the new page's count.
  let generation = 0;
  let pending = 0;
  let timer = null;
  let visible = false;
  let label = "";

  /** Cancel the delayed show, zero the count and hide the indicator if it is visible. */
  function reset() {
    if (timer != null) clearTimer(timer);
    timer = null;
    pending = 0;
    if (visible) {
      visible = false;
      hide();
    }
  }

  /** Start a new page: loads still pending from the previous page are ignored from now on. */
  function begin() {
    generation += 1;
    reset();
  }

  /**
   * Register one load of the current page.
   * @param {Promise<any>|any} work The load; a non-promise value counts as an instant load.
   * @param {string} [text] Text shown next to the spinner; the latest load's text wins.
   * @returns {Promise<any>} The same outcome as `work`, unchanged.
   */
  function track(work, text = "Loading…") {
    const promise = Promise.resolve(work);
    const owner = generation;
    pending += 1;
    label = text;
    if (visible) show(label);
    else if (timer == null)
      timer = setTimer(() => {
        timer = null;
        if (pending > 0) {
          visible = true;
          show(label);
        }
      }, delay);
    // Success and failure both end the wait. The caller still receives the rejection
    // through the returned promise and reports it (normally with app.fail).
    const settle = () => {
      if (owner !== generation) return;
      pending -= 1;
      if (pending === 0) reset();
    };
    promise.then(settle, settle);
    return promise;
  }

  return {
    begin,
    track,
    get pending() {
      return pending;
    },
    get visible() {
      return visible;
    },
  };
}

/**
 * Connect a load tracker to the page overlay and the main content area.
 *
 * While visible: the overlay shows the spinner and text, <main> is dimmed, marked
 * aria-busy and made inert (no clicks, no keyboard focus). The assistant panel stays usable.
 *
 * @param {object} elements
 * @param {HTMLElement} elements.overlay The #page-loading container (role="status").
 * @param {HTMLElement} elements.text The element that receives the loading text.
 * @param {HTMLElement} elements.main The page content area that is blocked during the load.
 * @param {number} [delay] Optional override of DEFAULT_LOADING_DELAY_MS.
 * @returns {ReturnType<typeof createLoadTracker>} The tracker; call begin() on navigation.
 */
export function mountPageLoading({ overlay, text, main }, delay = DEFAULT_LOADING_DELAY_MS) {
  return createLoadTracker({
    delay,
    show(label) {
      text.textContent = label;
      overlay.hidden = false;
      main.inert = true;
      main.setAttribute("aria-busy", "true");
      main.classList.add("page-is-loading");
    },
    hide() {
      overlay.hidden = true;
      text.textContent = "";
      main.inert = false;
      main.removeAttribute("aria-busy");
      main.classList.remove("page-is-loading");
    },
  });
}
