/**
 * docai-new-client.ts
 * --------------------
 * API client for the DOC AI NEW pipeline endpoints.
 */

import type {
  DocAiNewProcessResponse,
  DocAiNewTemplatesResponse,
  TrainingResult,
} from "./docai-new-types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || "";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/**
 * Process one or multiple PDF invoices through the DOC AI NEW pipeline.
 */
export async function processInvoicesNew(
  files: File[],
  clientId = "default",
  autoCreateTemplate = true
): Promise<DocAiNewProcessResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }

  const params = new URLSearchParams({
    client_id: clientId,
    auto_create_template: String(autoCreateTemplate),
  });

  const res = await fetch(`${API_BASE}/api/v2/docai-new/process?${params}`, {
    method: "POST",
    body: form,
  });

  return handleResponse<DocAiNewProcessResponse>(res);
}

/**
 * List all available templates for the Train Template screen.
 */
export async function listTemplatesNew(
  clientId = "default"
): Promise<DocAiNewTemplatesResponse> {
  const params = new URLSearchParams({ client_id: clientId });
  const res = await fetch(`${API_BASE}/api/v2/docai-new/templates?${params}`);
  return handleResponse<DocAiNewTemplatesResponse>(res);
}

/**
 * Train a template with one or more PDFs.
 */
export async function trainTemplate(
  templateId: string,
  files: File[],
  clientId = "default"
): Promise<TrainingResult> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }

  const params = new URLSearchParams({
    template_id: templateId,
    client_id: clientId,
  });

  const res = await fetch(`${API_BASE}/api/v2/docai-new/train?${params}`, {
    method: "POST",
    body: form,
  });

  return handleResponse<TrainingResult>(res);
}