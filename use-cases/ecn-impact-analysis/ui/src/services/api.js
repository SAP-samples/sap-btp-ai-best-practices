const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const API_KEY = (import.meta.env.VITE_API_KEY || "").trim();

/** A safe HTTP error whose fields mirror the API ErrorResponse contract. */
export class ApiError extends Error {
  /** Create an API error from the response status and safe server envelope. */
  constructor(status, code, message, details = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/** Call one API endpoint with JSON/FormData-aware headers and abort support. */
export async function request(
  endpoint,
  { method = "GET", body = null, headers = {}, signal = undefined, responseType = "json" } = {},
) {
  const requestHeaders = new Headers(headers);
  if (API_KEY) requestHeaders.set("X-API-Key", API_KEY);

  let requestBody = body;
  if (body !== null && !(body instanceof FormData)) {
    requestHeaders.set("Content-Type", "application/json");
    requestBody = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: requestHeaders,
    body: requestBody,
    signal,
  });
  if (response.ok && responseType === "blob") return response.blob();
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new ApiError(
      response.status,
      payload?.code || "HTTP_ERROR",
      payload?.message || `Request failed with status ${response.status}`,
      payload?.details ?? null,
    );
  }
  return payload;
}
