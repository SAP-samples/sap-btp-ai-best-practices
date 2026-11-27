/**
 * Signed-in user indicator and "Sign out" link in the workspace header.
 *
 * Sign-in itself happens before this page loads: the application router
 * (approuter/) redirects the browser to the SAP BTP login and only then serves
 * the UI. This module reads who is signed in from GET /api/me and shows the name
 * next to a "Sign out" link that ends the approuter session (/do/logout).
 *
 * Local development (AUTH_MODE=local, UI served by Vite) has no approuter, so the
 * "Sign out" link stays hidden there; the name still shows "Local Developer".
 *
 * The public showcase deployment (./deploy.sh public) is served by an approuter
 * without sign-in. GET /api/me answers sign_in_required=false there, so the link
 * stays hidden and the name shows "Public access".
 *
 * See docs/user-authentication.md.
 */
import { behindApprouter, LOGOUT_PATH, request } from "../services/api.js";

/**
 * Fill the header user indicator.
 *
 * @param {{name: HTMLElement, signOut: HTMLAnchorElement}} elements Header elements:
 *   the user name span and the sign-out link (both start hidden).
 * @param {{loadUser?: () => Promise<{display_name: string, email?: string|null,
 *   sign_in_required?: boolean}>, approuter?: boolean}} [options] Test seams: the
 *   /api/me loader and whether the approuter serves the page.
 * @returns {Promise<void>} Resolves once the header is filled; a failed lookup only
 *   leaves the name hidden (the page's own calls report session problems).
 */
export async function mountUserSession(elements, options = {}) {
  const { loadUser = () => request("/api/me"), approuter = behindApprouter() } = options;
  // Sign out only makes sense behind an approuter that signed the user in. When
  // the lookup fails, keep the link available so a user with a broken session
  // can still end it (the public deployment answers /api/me without sign-in).
  let signInRequired = true;
  try {
    const user = await loadUser();
    signInRequired = user.sign_in_required !== false;
    elements.name.textContent = user.display_name;
    if (user.email) elements.name.title = user.email;
    elements.name.hidden = false;
  } catch (error) {
    console.warn("Could not load the signed-in user:", error);
  }
  if (approuter && signInRequired) {
    elements.signOut.href = LOGOUT_PATH;
    elements.signOut.hidden = false;
  }
}
