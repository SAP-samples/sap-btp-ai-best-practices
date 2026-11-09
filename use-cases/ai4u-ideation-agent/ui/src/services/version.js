/**
 * Version check service — queries backend for update availability.
 */
import { request } from "./api.js";

/**
 * Check if a newer version of the application is available upstream.
 * Non-blocking: returns null on any error (caller should not show a banner).
 *
 * @returns {Promise<{updateAvailable: boolean|null, currentVersion: string, latestVersion: string|null, checkEnabled: boolean}|null>}
 */
export async function checkForUpdate() {
  try {
    return await request("/api/version-check");
  } catch {
    return null;
  }
}
