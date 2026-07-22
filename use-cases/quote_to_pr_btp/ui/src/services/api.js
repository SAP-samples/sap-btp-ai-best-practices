async function errorMessageFromResponse(response) {
  let detail = "";
  try {
    const payload = await response.clone().json();
    const raw = payload?.detail ?? payload;
    if (typeof raw === "string") detail = raw;
    else if (raw?.message) detail = raw.details ? `${raw.message}: ${raw.details}` : raw.message;
    else detail = JSON.stringify(raw);
  } catch (_) {
    try {
      detail = await response.text();
    } catch (__) {
      detail = "";
    }
  }
  return detail ? `HTTP ${response.status}: ${detail}` : `HTTP ${response.status}`;
}

const isLocalHost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8056";
const configuredLocalApiBaseUrl = import.meta.env.VITE_LOCAL_API_BASE_URL || "http://127.0.0.1:8056";
export const API_BASE_URL = (isLocalHost ? configuredLocalApiBaseUrl : configuredApiBaseUrl).replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY;

// General Request function to the API
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        ...headers
      },
      body: body ? JSON.stringify(body) : null
    });
  } catch (error) {
    throw new Error(`Backend is not reachable at ${API_BASE_URL}. ${error?.message || "Network request failed."}`);
  }

  if (!response.ok) {
    throw new Error(await errorMessageFromResponse(response));
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
    throw new Error(await errorMessageFromResponse(response));
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



export async function uploadFiles(endpoint, files) {
  const formData = new FormData();
  Array.from(files || []).forEach((file) => formData.append("files", file));
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY
    },
    body: formData
  });

  if (!response.ok) {
    throw new Error(await errorMessageFromResponse(response));
  }

  return response.json();
}
