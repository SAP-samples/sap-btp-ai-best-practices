/**
 * The red notice banner at the top of the workspace (#notice in index.html).
 *
 * Every page reports failures through app.fail(error), which lands here. Two kinds
 * of failure reach it and are shown differently:
 *
 * - API failures (ApiError from services/api.js) and errors the page code raises on
 *   purpose with a readable message (for example a draft field check): shown as is.
 * - Bugs in the page code itself (TypeError, ReferenceError, RangeError, SyntaxError,
 *   such as "Cannot read properties of undefined"): shown as a display error that
 *   says the data is not affected, with the technical text kept short. The full
 *   error and its stack trace go to the browser console for diagnosis.
 *
 * - An expired sign-in session (HTTP 401 from the approuter or the API): shown as a
 *   "session expired" banner; clicking it reloads the page, which sends the user
 *   through the SAP BTP sign-in again (see docs/user-authentication.md).
 *
 * Aborted requests (chat Stop / New conversation) are not failures and are ignored.
 */
import { ApiError, SESSION_EXPIRED_STATUS } from "../services/api.js";

/** Banner text for an expired sign-in session. */
export const SESSION_EXPIRED_TEXT =
  "Your sign-in session has expired. Click here to sign in again; unsaved form input on this page will be lost.";

/** Built-in error types that mean the page code failed, not a request or a check. */
const PAGE_CODE_ERRORS = [TypeError, ReferenceError, RangeError, SyntaxError];

/**
 * Decide how a thrown value is presented.
 *
 * @param {unknown} error Anything passed to app.fail or left as an unhandled rejection.
 * @returns {{text: string, pageError: boolean, sessionExpired?: boolean} | null} Banner
 *   text, whether it is a page code bug and whether the user must sign in again; null
 *   when nothing should be shown (aborted request).
 */
export function describeFailure(error) {
  if (error instanceof Error && error.name === "AbortError") return null;
  if (error instanceof ApiError && error.status === SESSION_EXPIRED_STATUS)
    return { text: SESSION_EXPIRED_TEXT, pageError: false, sessionExpired: true };
  if (error instanceof ApiError) return { text: error.message, pageError: false };
  if (PAGE_CODE_ERRORS.some((type) => error instanceof type)) {
    const detail = /** @type {Error} */ (error).message;
    return {
      text: `The page could not display part of this view because of an error in the page itself. Your data and runs are not affected; reload the page to try again. Technical detail: ${detail}`,
      pageError: true,
    };
  }
  if (error instanceof Error) return { text: error.message || error.name, pageError: false };
  return { text: String(error), pageError: false };
}

/**
 * Bind the banner element and return the functions app.fail and app.clearNotice use.
 *
 * @param {HTMLElement} element The #notice element (role="alert"); a click hides it, or
 *   reloads the page to sign in again when it reports an expired session.
 * @param {{error: (...values: unknown[]) => void}} [log] Console used for page code bugs.
 * @param {() => void} [reload] Page reload used to restart the sign-in (tests replace it).
 * @returns {{show: (error: unknown) => void, clear: () => void}}
 */
export function mountNotice(element, log = console, reload = () => window.location.reload()) {
  element.addEventListener("click", () => {
    // A full page load (not an AJAX call) lets the approuter redirect to the sign-in page.
    if (element.dataset.kind === "session") reload();
    else element.hidden = true;
  });
  return {
    show(error) {
      const failure = describeFailure(error);
      if (!failure) return;
      // Keep the stack trace for diagnosis; the banner only carries the message.
      if (failure.pageError) log.error("Workspace page error:", error);
      element.textContent = failure.text;
      element.dataset.kind = failure.sessionExpired ? "session" : failure.pageError ? "page" : "request";
      element.hidden = false;
    },
    clear() {
      element.hidden = true;
    },
  };
}
