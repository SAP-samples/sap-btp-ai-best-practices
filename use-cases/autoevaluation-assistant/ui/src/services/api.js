export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY;

/**
 * Build standard JSON headers for backend API requests.
 *
 * @param {Record<string, string>} headers - Additional request headers.
 * @returns {Record<string, string>} Headers with API key and JSON content type.
 */
function jsonHeaders(headers = {}) {
  return {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
    ...headers
  };
}

/**
 * Build standard non-JSON headers for backend API requests.
 *
 * @param {Record<string, string>} headers - Additional request headers.
 * @returns {Record<string, string>} Headers with API key.
 */
function apiHeaders(headers = {}) {
  return {
    "X-API-Key": API_KEY,
    ...headers
  };
}

/**
 * Build a useful error with structured backend details when available.
 *
 * @param {Response} response - Fetch response returned by the browser.
 * @returns {Promise<Error & {status?: number, detail?: unknown}>} Structured request error.
 */
async function responseError(response) {
  const detail = await response.text();
  if (!detail) {
    const error = new Error(`HTTP error! status: ${response.status}`);
    error.status = response.status;
    error.detail = null;
    return error;
  }
  try {
    const payload = JSON.parse(detail);
    const apiDetail = payload.detail ?? payload;
    const detailMessage =
      typeof apiDetail === "string"
        ? apiDetail
        : apiDetail?.message || JSON.stringify(apiDetail);
    const error = new Error(`HTTP error! status: ${response.status} - ${detailMessage}`);
    error.status = response.status;
    error.detail = apiDetail;
    return error;
  } catch {
    const error = new Error(`HTTP error! status: ${response.status} - ${detail}`);
    error.status = response.status;
    error.detail = detail;
    return error;
  }
}

/**
 * Send a JSON request to the backend API.
 *
 * @param {string} endpoint - API path beginning with "/api".
 * @param {string} method - HTTP method.
 * @param {object | null} body - JSON payload to send.
 * @param {Record<string, string>} headers - Additional request headers.
 * @returns {Promise<object>} Parsed JSON response.
 */
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: jsonHeaders(headers),
    body: body ? JSON.stringify(body) : null
  });

  if (!response.ok) {
    throw await responseError(response);
  }

  return response.json();
}

/**
 * Send multipart form data to the backend API.
 *
 * @param {string} endpoint - API path beginning with "/api".
 * @param {FormData} formData - Browser FormData payload.
 * @param {string} method - HTTP method.
 * @param {Record<string, string>} headers - Additional request headers.
 * @returns {Promise<object>} Parsed JSON response.
 */
export async function requestForm(endpoint, formData, method = "POST", headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: apiHeaders(headers),
    body: formData
  });

  if (!response.ok) {
    throw await responseError(response);
  }

  return response.json();
}

/**
 * Download a binary response from the backend API.
 *
 * @param {string} endpoint - API path beginning with "/api".
 * @param {Record<string, string>} headers - Additional request headers.
 * @returns {Promise<Blob>} Downloaded response body.
 */
export async function requestBlob(endpoint, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "GET",
    headers: apiHeaders(headers)
  });

  if (!response.ok) {
    throw await responseError(response);
  }

  return response.blob();
}

/**
 * Stream newline-delimited JSON from the backend API.
 *
 * @param {string} endpoint - API path beginning with "/api".
 * @param {object} options - Stream request options and chunk callback.
 * @returns {Promise<void>} Promise resolved after the stream is exhausted.
 */
export async function streamNDJSON(endpoint, { method = "POST", body = null, headers = {}, onChunk } = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: jsonHeaders(headers),
    body: body ? JSON.stringify(body) : null
  });

  if (!response.ok) {
    throw await responseError(response);
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
