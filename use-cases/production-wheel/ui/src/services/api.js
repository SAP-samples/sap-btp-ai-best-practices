/**
 * HTTP adapters for the workspace API.
 *
 * Authentication: in Cloud Foundry the page is served by the application router
 * (approuter/), which signs the user in through SAP BTP and keeps the session in
 * a cookie. API calls go to the same origin (`/api/...`) and the approuter adds
 * the user's token before forwarding them, so this module sends no credentials
 * of its own. See docs/user-authentication.md.
 *
 * VITE_API_BASE_URL: empty in the deployed build (same-origin calls through the
 * approuter); set to the API URL for local development. When the variable is not
 * defined at all, the local API on port 8000 is used.
 */
export const API_BASE_URL = (
  import.meta.env?.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

/**
 * Headers sent with every API call.
 * X-Requested-With marks the call as AJAX: when the sign-in session has expired the
 * approuter then answers 401 instead of redirecting the request to the login page,
 * which fetch could not follow. The 401 becomes a "session expired" notice.
 */
export const AJAX_HEADERS = Object.freeze({ "X-Requested-With": "XMLHttpRequest" });

/** HTTP status that means the user must sign in again. */
export const SESSION_EXPIRED_STATUS = 401;

/** Path of the approuter logout endpoint (approuter/xs-app.json "logout"). */
export const LOGOUT_PATH = "/do/logout";

/**
 * True when the page is served by the approuter: API calls are then same-origin
 * (empty API_BASE_URL). Local development calls the API directly.
 * @param {string} [baseUrl] API base URL, defaults to the build setting.
 * @returns {boolean}
 */
export function behindApprouter(baseUrl = API_BASE_URL) {
  return baseUrl === "";
}

/**
 * CSRF protection (approuter "csrfProtection", on by default).
 * The approuter rejects state-changing calls (POST, PUT, PATCH, DELETE) on the
 * signed-in session unless they carry a token it issued. The token is fetched
 * once with a GET carrying "X-CSRF-Token: Fetch", kept for the page's lifetime,
 * and refreshed once when the approuter answers 403 with "X-CSRF-Token: Required"
 * (for example after a new sign-in). Local development has no approuter, so no
 * token is used there.
 */
export const CSRF_HEADER = "X-CSRF-Token";
/** Any authenticated GET route works for the token fetch; /api/me is the lightest. */
export const CSRF_TOKEN_PATH = "/api/me";
const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/**
 * Decide whether a call must carry the approuter CSRF token.
 * @param {string | undefined} method HTTP method of the call (GET when absent).
 * @param {string} [baseUrl] API base URL, defaults to the build setting.
 * @returns {boolean}
 */
export function needsCsrfToken(method, baseUrl = API_BASE_URL) {
  return behindApprouter(baseUrl) && !SAFE_METHODS.has((method || "GET").toUpperCase());
}

/** CSRF token cached for this page; null until the first state-changing call. */
let csrfToken = null;

/**
 * Ask the approuter for a CSRF token.
 * @param {typeof fetch} fetchImpl Fetch implementation (tests replace it).
 * @param {string} baseUrl API base URL.
 * @returns {Promise<string | null>} The token, or null when none was issued.
 */
async function fetchCsrfToken(fetchImpl, baseUrl) {
  const response = await fetchImpl(`${baseUrl}${CSRF_TOKEN_PATH}`, {
    method: "GET",
    headers: { ...AJAX_HEADERS, [CSRF_HEADER]: "Fetch" },
  });
  return response.headers.get(CSRF_HEADER);
}

/**
 * Copy fetch options and add the CSRF token header.
 * @param {RequestInit} init Original fetch options.
 * @param {string | null} token CSRF token.
 * @returns {RequestInit}
 */
function withCsrfToken(init, token) {
  return token ? { ...init, headers: { ...(/** @type {Record<string, string>} */ (init.headers || {})), [CSRF_HEADER]: token } } : init;
}

/**
 * Fetch that adds the approuter CSRF token to state-changing calls.
 * @param {string} url Absolute or same-origin request URL.
 * @param {RequestInit} init Fetch options.
 * @param {{fetchImpl?: typeof fetch, baseUrl?: string}} [options] Test seams.
 * @returns {Promise<Response>} The response, whatever its status.
 */
export async function csrfFetch(url, init, options = {}) {
  const { fetchImpl = fetch, baseUrl = API_BASE_URL } = options;
  if (!needsCsrfToken(init.method, baseUrl)) return fetchImpl(url, init);
  csrfToken ??= await fetchCsrfToken(fetchImpl, baseUrl);
  const response = await fetchImpl(url, withCsrfToken(init, csrfToken));
  if (response.status !== 403 || response.headers.get(CSRF_HEADER)?.toLowerCase() !== "required") return response;
  // The cached token belonged to an earlier session: fetch a fresh one and retry once.
  csrfToken = await fetchCsrfToken(fetchImpl, baseUrl);
  return fetchImpl(url, withCsrfToken(init, csrfToken));
}

/** Forget the cached CSRF token (tests only). */
export function resetCsrfToken() {
  csrfToken = null;
}

/**
 * A failed call to the workspace API, as opposed to a bug in the page code.
 *
 * `status` is the HTTP status, or 0 when the API could not be reached at all.
 * The message keeps the "<status>: <detail>" form that callers already test
 * (for example `message.startsWith("409")`); unreachable API errors read as plain text.
 */
export class ApiError extends Error {
  /**
   * @param {number} status HTTP status code, or 0 for a network failure.
   * @param {string} detail Server detail text, or the network failure description.
   */
  constructor(status, detail) {
    super(status ? `${status}: ${detail}` : detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Call fetch and turn a network failure into an ApiError with a readable message.
 * Aborted requests (the chat's Stop and reset) are re-thrown unchanged.
 * @param {string} url Absolute request URL.
 * @param {RequestInit} init Fetch options.
 * @returns {Promise<Response>} The response, whatever its status.
 */
async function send(url, init) {
  try {
    return await csrfFetch(url, init);
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError(0, `Cannot reach the workspace API (${error?.message || error}). Check the connection, then reload the page.`);
  }
}

/** Read a response and surface structured API validation errors to the caller. */
async function checked(response) {
  if (!response.ok) {
    const content = await response.text();
    let detail = content;
    try {
      const value = JSON.parse(content);
      detail = value.detail || value.error || value;
    } catch {
      /* Retain non-JSON server errors. */
    }
    throw new ApiError(
      response.status,
      typeof detail === "string" ? detail : JSON.stringify(detail),
    );
  }
  return response;
}

/** Send JSON or multipart input and return the decoded response. */
export async function request(
  endpoint,
  method = "GET",
  body = null,
  headers = {},
) {
  const multipart = body instanceof FormData;
  const response = await checked(
    await send(`${API_BASE_URL}${endpoint}`, {
      method,
      headers: {
        ...AJAX_HEADERS,
        ...(multipart ? {} : { "Content-Type": "application/json" }),
        ...headers,
      },
      body: body == null ? undefined : multipart ? body : JSON.stringify(body),
    }),
  );
  return response.status === 204 ? null : response.json();
}

/**
 * Delete an entity, treating "already deleted" as success.
 * A retried delete whose first attempt already committed gets the API's
 * unknown-identifier 404; that is the desired end state, not an error.
 * Route-level 404s ("Not Found") still throw so misconfiguration stays visible.
 * Input: API path. Output: decoded response, or null when the entity was already gone.
 */
export async function deleteEntity(endpoint) {
  try {
    return await request(endpoint, "DELETE");
  } catch (error) {
    if (String(error.message).startsWith("404: Unknown")) return null;
    throw error;
  }
}

/** Consume newline-delimited JSON, including a final line without a newline. */
export async function decodeNDJSON(stream, onChunk) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done
        ? decoder.decode()
        : decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines)
        if (line.trim()) await onChunk(JSON.parse(line));
      if (done) break;
    }
    if (buffer.trim()) await onChunk(JSON.parse(buffer));
  } finally {
    reader.releaseLock();
  }
}

/** Stream chat events in order; callback failures propagate rather than disappear. */
export async function streamNDJSON(endpoint, { body, onChunk, signal }) {
  const response = await checked(
    await send(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { ...AJAX_HEADERS, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),
  );
  if (!response.body) throw new Error("Server returned no response stream.");
  await decodeNDJSON(response.body, onChunk);
}

/** Download a CSV or XLSX through fetch, so the export uses the signed-in session and keeps data out of URLs and history. */
export async function downloadExport(endpoint, filename = "optimizer-export.csv", body = null) {
  const response = await checked(
    await send(`${API_BASE_URL}${endpoint}`, {
      method: body === null ? "GET" : "POST",
      headers: {...AJAX_HEADERS, ...(body === null ? {} : {"Content-Type": "application/json"})},
      body: body === null ? undefined : JSON.stringify(body),
    }),
  );
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
