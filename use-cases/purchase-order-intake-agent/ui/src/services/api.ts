/**
 * Thin HTTP client for the FastAPI backend.
 *
 * Every request carries the shared `X-API-Key` header expected by `api/app/security.py`.
 * Base URL and key come from Vite env vars (ui/.env locally, manifest.yaml on CF).
 *
 * - `request`      JSON in / JSON out
 * - `requestForm`  multipart upload (browser sets the multipart boundary)
 * - `requestBlob`  binary download (attachments; an <iframe> cannot send the API key header)
 * Non-2xx responses throw `ApiError` carrying the status and the parsed `detail` of the body.
 */
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY ?? "";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

/** Error thrown for non-2xx responses; `detail` is FastAPI's `detail` (string or object). */
export class ApiError extends Error {
  status: number;
  detail: unknown;

  /**
   * @param status - HTTP status code.
   * @param detail - Parsed `detail` field of the error body (or the raw text).
   */
  constructor(status: number, detail: unknown) {
    const text = typeof detail === "string" ? detail : (detail as { message?: string })?.message;
    super(text || `HTTP error ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Throws an `ApiError` for a failed response.
 *
 * @param response - Fetch response with a non-2xx status.
 */
async function raiseFor(response: Response): Promise<never> {
  const body = await response.text();
  let detail: unknown = body;
  try {
    detail = (JSON.parse(body) as { detail?: unknown }).detail ?? body;
  } catch {
    // not JSON: keep the text
  }
  throw new ApiError(response.status, detail);
}

/**
 * Sends a JSON request and parses the JSON response.
 *
 * @param endpoint - Path starting with "/", e.g. "/api/health".
 * @param method - HTTP method, defaults to GET.
 * @param body - Optional payload, serialized as JSON.
 * @returns Parsed JSON body typed as `T`.
 * @throws ApiError when the response status is not 2xx.
 */
export async function request<T = unknown>(endpoint: string, method: HttpMethod = "GET", body: unknown = null): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
    body: body !== null ? JSON.stringify(body) : null
  });
  if (!response.ok) await raiseFor(response);
  return (await response.json()) as T;
}

/**
 * Posts multipart form data (file uploads) and parses the JSON response.
 *
 * @param endpoint - Path starting with "/".
 * @param form - FormData with files and fields.
 * @returns Parsed JSON body typed as `T`.
 * @throws ApiError when the response status is not 2xx.
 */
export async function requestForm<T = unknown>(endpoint: string, form: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: "POST", headers: { "X-API-Key": API_KEY }, body: form });
  if (!response.ok) await raiseFor(response);
  return (await response.json()) as T;
}

/**
 * Downloads a binary resource.
 *
 * @param endpoint - Path starting with "/".
 * @returns The response body as Blob (with the server's content type).
 * @throws ApiError when the response status is not 2xx.
 */
export async function requestBlob(endpoint: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, { headers: { "X-API-Key": API_KEY } });
  if (!response.ok) await raiseFor(response);
  return response.blob();
}
