/**
 * Shared API configuration and JSON request helper.
 *
 * VITE_API_BASE_URL and VITE_API_KEY are compiled into the browser bundle at build time.
 * The key is therefore visible to anyone who loads the UI; production must replace it
 * with XSUAA login through an approuter (see docs/deployment-runbook.md).
 */
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY;

/**
 * Send a JSON request to the API with the shared X-API-Key header.
 * @param {string} endpoint Path starting with "/" (e.g. "/api/a2a").
 * @param {string} method HTTP method.
 * @param {object|null} body JSON body, or null.
 * @param {object} headers Extra headers.
 * @returns {Promise<object>} Parsed JSON response; throws on non-2xx status.
 */
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...headers
    },
    body: body ? JSON.stringify(body) : null
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}
