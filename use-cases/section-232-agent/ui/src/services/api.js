export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_KEY = import.meta.env.VITE_API_KEY || "your-super-secret-api-key";
const TERMINAL_CLASSIFICATION_JOB_STATUSES = new Set(["completed", "failed", "partial_failed"]);

export function buildQuery(params = {}) {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") return;
    searchParams.set(key, String(value));
  });
  const query = searchParams.toString();
  return query ? `?${query}` : "";
}

// General Request function to the API
export async function request(endpoint, method = "GET", body = null, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: buildHeaders(headers, body),
    body: body == null ? null : body instanceof FormData ? body : JSON.stringify(body)
  });

  await throwIfNotOk(response);

  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function buildHeaders(headers = {}, body = null) {
  const finalHeaders = {
    "X-API-Key": API_KEY,
    ...headers
  };
  if (!(body instanceof FormData)) {
    finalHeaders["Content-Type"] = "application/json";
  }
  return finalHeaders;
}

async function throwIfNotOk(response) {
  if (response.ok) {
    return;
  }

  let message = `HTTP error! status: ${response.status}`;
  let errorBody = null;
  try {
    errorBody = await response.json();
    const detail = errorBody?.detail;
    if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail.message === "string") {
      message = detail.message;
    } else if (typeof errorBody?.error === "string") {
      message = errorBody.error;
    } else {
      message = JSON.stringify(errorBody);
    }
  } catch (_error) {
    // Keep the generic message when the response is not JSON.
  }
  const error = new Error(message);
  error.status = response.status;
  error.payload = errorBody;
  throw error;
}

function parseDownloadFilename(headerValue) {
  const match = /filename="([^"]+)"|filename=([^;]+)/i.exec(headerValue || "");
  const rawFilename = match?.[1] || match?.[2] || "";
  return rawFilename.trim();
}

export async function requestBinary(endpoint, method = "GET", body = null, headers = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: buildHeaders(headers, body),
    body: body == null ? null : body instanceof FormData ? body : JSON.stringify(body)
  });

  await throwIfNotOk(response);

  return {
    blob: await response.blob(),
    filename: parseDownloadFilename(response.headers.get("Content-Disposition"))
  };
}

export function isTerminalClassificationJobStatus(status) {
  return TERMINAL_CLASSIFICATION_JOB_STATUSES.has(String(status || ""));
}

function isTransientJobPollingError(error) {
  const status = Number(error?.status);
  if ([502, 503, 504].includes(status)) {
    return true;
  }
  return error instanceof TypeError || error?.message === "Failed to fetch";
}

export async function pollClassificationJob(
  jobId,
  {
    intervalMs = 2000,
    maxConsecutiveFailures = 5,
    onUpdate
  } = {}
) {
  let latest = null;
  let consecutiveFailures = 0;
  while (true) {
    try {
      latest = await request(`/api/metal-composition/classification-jobs/${encodeURIComponent(jobId)}`);
      consecutiveFailures = 0;
      if (onUpdate) onUpdate(latest);
      if (isTerminalClassificationJobStatus(latest?.status)) {
        return latest;
      }
      await sleep(intervalMs);
    } catch (error) {
      consecutiveFailures += 1;
      if (!isTransientJobPollingError(error) || consecutiveFailures >= maxConsecutiveFailures) {
        throw error;
      }
      await sleep(intervalMs * consecutiveFailures);
    }
  }
}
