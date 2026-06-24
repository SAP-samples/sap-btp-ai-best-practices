const env = import.meta.env || {};

export const API_BASE_URL = (env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = env.VITE_API_KEY;

/**
 * Parse a fetch response body as JSON without throwing on empty/plain-text bodies.
 *
 * @param {Response} response Fetch response.
 * @returns {Promise<object>} Parsed JSON object or fallback payload.
 */
async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

/**
 * Build an Error that preserves structured API failure details.
 *
 * @param {Response} response Fetch response.
 * @param {object} payload Parsed error payload.
 * @returns {Error & { status?: number, payload?: object, detail?: unknown }} Request error.
 */
function buildRequestError(response, payload) {
  const detail = payload?.detail;
  const message =
    detail && typeof detail === "object" && detail.message
      ? detail.message
      : typeof detail === "string" && detail
        ? detail
        : `HTTP error! status: ${response.status}`;
  const error = new Error(message);
  error.status = response.status;
  error.payload = payload;
  error.detail = detail;
  return error;
}

// General Request function to the API
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

  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw buildRequestError(response, payload);
  }

  return payload;
}

/**
 * Send multipart form data to the API without forcing a JSON content type.
 *
 * @param {string} endpoint API endpoint path.
 * @param {string} method HTTP method.
 * @param {FormData} formData Multipart payload.
 * @param {object} headers Additional request headers.
 * @returns {Promise<object>} Parsed JSON response.
 */
export async function requestFormData(endpoint, method = "POST", formData, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: {
      "X-API-Key": API_KEY,
      ...headers
    },
    body: formData
  });

  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw buildRequestError(response, payload);
  }

  return payload;
}

export async function streamNDJSON(endpoint, { method = "POST", body = null, headers = {}, onChunk } = {}) {
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

  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const obj = JSON.parse(trimmed);
        if (onChunk) onChunk(obj);
      } catch (e) {
        console.error("Failed to parse NDJSON line:", line, e);
      }
    }
  }

  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer.trim());
      if (onChunk) onChunk(obj);
    } catch (e) {
      // ignore trailing partial
    }
  }
}
