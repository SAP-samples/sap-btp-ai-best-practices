import { getAccessTokenAsync, login } from "./auth.js";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

/**
 * Get authorization headers with a fresh token.
 * For CDC users, this calls the Gigya SDK for a fresh JWT every time.
 */
export async function getAuthHeaders() {
  const token = await getAccessTokenAsync();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

async function _handleUnauthorized() {
  // Try to get a fresh token (covers CDC token expiry)
  const token = await getAccessTokenAsync();
  if (token) {
    return true; // Refresh succeeded — caller should retry
  }
  // Session truly expired — redirect to login
  await login(window.location.href);
  return false;
}

export async function request(endpoint, method = "GET", body = null, headers = {}, signal = undefined) {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders,
      ...headers
    },
    body: body ? JSON.stringify(body) : null,
    signal
  });

  if (response.status === 401) {
    const refreshed = await _handleUnauthorized();
    if (refreshed) {
      // Retry once with fresh token
      const retryHeaders = await getAuthHeaders();
      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        method,
        headers: { "Content-Type": "application/json", ...retryHeaders, ...headers },
        body: body ? JSON.stringify(body) : null,
        signal
      });
      if (retryResponse.ok) return retryResponse.json();
      // Still failing — give up
    }
    return;
  }

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function streamNDJSON(endpoint, { method = "POST", body = null, headers = {}, onChunk } = {}) {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders,
      ...headers
    },
    body: body ? JSON.stringify(body) : null
  });

  if (response.status === 401) {
    const refreshed = await _handleUnauthorized();
    if (refreshed) {
      // Retry once
      const retryHeaders = await getAuthHeaders();
      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        method,
        headers: { "Content-Type": "application/json", ...retryHeaders, ...headers },
        body: body ? JSON.stringify(body) : null
      });
      if (!retryResponse.ok) return;
      // Continue with retry response
      const reader = retryResponse.body?.getReader();
      if (!reader) return;
      await _readNDJSON(reader, onChunk);
      return;
    }
    return;
  }

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) return;
  await _readNDJSON(reader, onChunk);
}

async function _readNDJSON(reader, onChunk) {
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

export async function requestMultipart(endpoint, formData, headers = {}) {
  const authHeaders = await getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      ...authHeaders,
      ...headers
    },
    body: formData
  });

  if (response.status === 401) {
    const refreshed = await _handleUnauthorized();
    if (refreshed) {
      const retryHeaders = await getAuthHeaders();
      const retryResponse = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { ...retryHeaders, ...headers },
        body: formData
      });
      if (retryResponse.ok) return retryResponse.json();
    }
    return;
  }

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}
