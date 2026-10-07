export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY;

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

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
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

// Multipart upload (files + form fields). Unlike request()/streamNDJSON, this must
// NOT set Content-Type: the browser sets multipart/form-data with the correct
// boundary from the FormData body. Injects the API key and surfaces the FastAPI
// error "detail" when the response is not ok.
export async function upload(endpoint, formData, { headers = {} } = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "X-API-Key": API_KEY, ...headers },
    body: formData
  });

  if (!response.ok) {
    let detail = `HTTP error! status: ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (e) {
      // non-JSON error body; keep the status-based message
    }
    throw new Error(detail);
  }

  return response.json();
}
