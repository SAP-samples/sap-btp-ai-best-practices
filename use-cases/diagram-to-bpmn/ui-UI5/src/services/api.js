export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(
  /\/$/,
  ""
);
export const API_KEY = import.meta.env.VITE_API_KEY;

function buildHeaders(extra = {}) {
  const headers = { ...extra };
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }
  return headers;
}

function extractErrorMessage(data, response) {
  if (data && typeof data === "object") {
    if (data.error) return data.error;
    if (data.detail) {
      if (typeof data.detail === "string") return data.detail;
      try {
        return JSON.stringify(data.detail);
      } catch (error) {
        return "Request failed.";
      }
    }
    try {
      return JSON.stringify(data);
    } catch (error) {
      return "Request failed.";
    }
  }

  if (typeof data === "string" && data.trim()) {
    return data.trim();
  }

  return `Request failed with status ${response.status}`;
}

async function parseResponse(response, { allowErrorResponse = false } = {}) {
  const contentType = response.headers.get("content-type") || "";
  let data = null;

  if (contentType.includes("application/json")) {
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
  } else {
    try {
      data = await response.text();
    } catch (error) {
      data = null;
    }
  }

  if (!response.ok && !allowErrorResponse) {
    const message = extractErrorMessage(data, response);
    const httpError = new Error(message);
    httpError.status = response.status;
    httpError.data = data;
    throw httpError;
  }

  return data;
}

// General request function to the API (JSON-based)
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;

  const requestHeaders = buildHeaders(headers);
  if (!isFormData && body !== null && body !== undefined && !requestHeaders["Content-Type"]) {
    requestHeaders["Content-Type"] = "application/json";
  }

  const options = {
    method,
    headers: requestHeaders
  };

  if (body !== null && body !== undefined) {
    options.body = isFormData ? body : JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
  return parseResponse(response);
}

export async function checkHealth() {
  return request("/api/health");
}

export async function generateBpmn({ file, filename, provider, model }) {
  if (!file) {
    throw new Error("A diagram file is required to generate BPMN.");
  }

  const formData = new FormData();
  formData.append("file", file, filename || file.name || "diagram.png");

  if (provider) {
    formData.append("provider", provider);
  }

  if (model) {
    formData.append("model", model);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/bpmn/generate`, {
      method: "POST",
      headers: buildHeaders(),
      body: formData
    });

    const data = await parseResponse(response, { allowErrorResponse: true });

    if (!response.ok) {
      return {
        success: false,
        error: extractErrorMessage(data, response),
        provider,
        model,
        usage: data && typeof data === "object" ? data.usage : null
      };
    }

    if (data && typeof data === "object") {
      return {
        success: Boolean(data.success),
        bpmn_xml: data.bpmn_xml || "",
        provider: data.provider || provider || "",
        model: data.model || model || "",
        usage: data.usage || null,
        error: data.error
      };
    }

    return {
      success: false,
      error: "Unexpected response format from BPMN API.",
      provider,
      model
    };
  } catch (error) {
    return {
      success: false,
      error: error.message || "Failed to reach BPMN generation API.",
      provider,
      model
    };
  }
}

export async function streamNDJSON(
  endpoint,
  { method = "POST", body = null, headers = {}, onChunk } = {}
) {
  const requestHeaders = buildHeaders({
    "Content-Type": "application/json",
    ...headers
  });

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: requestHeaders,
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
      } catch (error) {
        console.error("Failed to parse NDJSON line:", line, error);
      }
    }
  }

  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer.trim());
      if (onChunk) onChunk(obj);
    } catch (error) {
      // Ignore trailing partial
    }
  }
}
