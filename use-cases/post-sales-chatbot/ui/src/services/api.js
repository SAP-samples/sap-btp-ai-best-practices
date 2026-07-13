export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY;

// Session management
let currentSessionId = null;

/**
 * Set the current session ID
 * @param {string} id - Session ID
 */
export function setSessionId(id) {
  currentSessionId = id;
}

/**
 * Get the current session ID
 * @returns {string|null} Current session ID
 */
export function getSessionId() {
  return currentSessionId;
}

/**
 * General Request function to the API
 * @param {string} endpoint - API endpoint
 * @param {string} method - HTTP method
 * @param {object|null} body - Request body
 * @param {object} headers - Additional headers
 * @returns {Promise<object>} Response JSON
 */
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  const requestHeaders = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
    ...headers
  };

  // Add session ID if available
  if (currentSessionId) {
    requestHeaders["X-Session-Id"] = currentSessionId;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: requestHeaders,
    body: body ? JSON.stringify(body) : null
  });

  // Extract session ID from response headers
  const newSessionId = response.headers.get("X-Session-Id");
  if (newSessionId) {
    currentSessionId = newSessionId;
  }

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Stream NDJSON responses from the API
 * @param {string} endpoint - API endpoint
 * @param {object} options - Request options
 * @param {string} options.method - HTTP method
 * @param {object|null} options.body - Request body
 * @param {object} options.headers - Additional headers
 * @param {function} options.onChunk - Callback for each chunk
 */
export async function streamNDJSON(endpoint, { method = "POST", body = null, headers = {}, onChunk } = {}) {
  const requestHeaders = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
    ...headers
  };

  // Add session ID if available
  if (currentSessionId) {
    requestHeaders["X-Session-Id"] = currentSessionId;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: requestHeaders,
    body: body ? JSON.stringify(body) : null
  });

  // Extract session ID from response headers
  const newSessionId = response.headers.get("X-Session-Id");
  if (newSessionId) {
    currentSessionId = newSessionId;
  }

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

        // Update session ID if present in the chunk
        if (obj.session_id) {
          currentSessionId = obj.session_id;
        }

        if (onChunk) onChunk(obj);
      } catch (e) {
        console.error("Failed to parse NDJSON line:", line, e);
      }
    }
  }

  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer.trim());
      if (obj.session_id) {
        currentSessionId = obj.session_id;
      }
      if (onChunk) onChunk(obj);
    } catch (e) {
      // ignore trailing partial
    }
  }
}
