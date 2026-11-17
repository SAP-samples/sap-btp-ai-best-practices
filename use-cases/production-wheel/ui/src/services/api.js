/** HTTP adapters; keep the existing Vite URL and API-key conventions. */
export const API_BASE_URL = (
  import.meta.env?.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");
export const API_KEY = import.meta.env?.VITE_API_KEY;

/** Build authentication headers without transmitting an undefined key. */
export function authHeaders() {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
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
    throw new Error(
      `${response.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`,
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
    await fetch(`${API_BASE_URL}${endpoint}`, {
      method,
      headers: {
        ...authHeaders(),
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
    await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),
  );
  if (!response.body) throw new Error("Server returned no response stream.");
  await decodeNDJSON(response.body, onChunk);
}

/** Download an authenticated CSV or XLSX without leaking API keys into URLs or history. */
export async function downloadExport(endpoint, filename = "optimizer-export.csv", body = null) {
  const response = await checked(
    await fetch(`${API_BASE_URL}${endpoint}`, {
      method: body === null ? "GET" : "POST",
      headers: {...authHeaders(), ...(body === null ? {} : {"Content-Type": "application/json"})},
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
