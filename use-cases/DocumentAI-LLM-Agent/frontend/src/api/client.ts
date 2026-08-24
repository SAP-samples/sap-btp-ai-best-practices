import axios, { type AxiosInstance, type AxiosError } from "axios";
import type {
  HealthResponse,
  SchemasResponse,
  TemplatesResponse,
  InvoiceProcessResponse,
  GenAIPipelineResponse,
  EvaluationResponse,
  OutputFilesResponse,
  PostInvoiceRequest,
  PostInvoiceResponse,
  PostPOInvoiceRequest,
  PostPOInvoiceResponse,
  PurchaseOrdersResponse,
  CustomerSearchResponse,
  ChatRequest,
  ExtractedPurchaseOrder,
  SOValidationResult,
  CreateSORequest,
  CreateSOResponse,
  PostPaymentAdviceRequest,
  PostPaymentAdviceResponse,
} from "./types.js";

// ─── API Client ───────────────────────────────────────────────────────────────

const http: AxiosInstance = axios.create({
  baseURL: "",
  timeout: 300_000,
  headers: { Accept: "application/json" },
});

// ─── Error handling ───────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
    message?: string
  ) {
    super(message ?? detail);
    this.name = "ApiError";
  }
}

function handleError(err: unknown): never {
  if (axios.isAxiosError(err)) {
    const axiosErr = err as AxiosError<{ detail?: string }>;
    const status = axiosErr.response?.status ?? 0;
    const detail =
      axiosErr.response?.data?.detail ??
      axiosErr.message ??
      "Unknown error";
    throw new ApiError(status, detail);
  }
  throw new ApiError(0, String(err));
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

export const api = {
  /** GET /health */
  async health(): Promise<HealthResponse> {
    try {
      const { data } = await http.get<HealthResponse>("/health");
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/v1/schemas */
  async getSchemas(clientId = "default"): Promise<SchemasResponse> {
    try {
      const { data } = await http.get<SchemasResponse>("/api/v1/schemas", {
        params: { client_id: clientId },
      });
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/v1/templates */
  async getTemplates(clientId = "default"): Promise<TemplatesResponse> {
    try {
      const { data } = await http.get<TemplatesResponse>("/api/v1/templates", {
        params: { client_id: clientId },
      });
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/invoice/process */
  async processInvoice(
    file: File,
    schemaName = "SAP_invoice_schema",
    clientId = "default",
    documentType = "invoice"
  ): Promise<InvoiceProcessResponse> {
    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await http.post<InvoiceProcessResponse>(
        "/api/v1/invoice/process",
        formData,
        {
          params: { schema_name: schemaName, client_id: clientId, document_type: documentType },
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/genai/pipeline */
  async runGenAIPipeline(
    file: File,
    schemaName = "SAP_invoice_schema",
    clientId = "default",
    documentType = "invoice"
  ): Promise<GenAIPipelineResponse> {
    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await http.post<GenAIPipelineResponse>(
        "/api/v1/genai/pipeline",
        formData,
        {
          params: { schema_name: schemaName, client_id: clientId, document_type: documentType },
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/evaluation/run */
  async runEvaluation(): Promise<EvaluationResponse> {
    try {
      const { data } = await http.post<EvaluationResponse>("/api/v1/evaluation/run");
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/v1/output/genai */
  async listGenAIOutputs(): Promise<OutputFilesResponse> {
    try {
      const { data } = await http.get<OutputFilesResponse>("/api/v1/output/genai");
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/v1/output/genai/{filename} */
  async getGenAIOutput(filename: string): Promise<Record<string, unknown>> {
    try {
      const { data } = await http.get<Record<string, unknown>>(
        `/api/v1/output/genai/${filename}`
      );
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/v1/output/evaluation */
  async listEvaluationOutputs(): Promise<OutputFilesResponse> {
    try {
      const { data } = await http.get<OutputFilesResponse>("/api/v1/output/evaluation");
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/v1/fi/config — returns FI posting defaults (default GL account, etc.) */
  async getFIConfig(): Promise<{ gl_account: string }> {
    try {
      const { data } = await http.get<{ gl_account: string }>("/api/v1/fi/config");
      return data;
    } catch {
      return { gl_account: "" };
    }
  },

  /** POST /api/v1/fi/post-invoice */
  async postSupplierInvoice(payload: PostInvoiceRequest): Promise<PostInvoiceResponse> {
    try {
      const { data } = await http.post<PostInvoiceResponse>(
        "/api/v1/fi/post-invoice",
        payload,
      );
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/fi/post-po-invoice */
  async postPOInvoice(payload: PostPOInvoiceRequest): Promise<PostPOInvoiceResponse> {
    try {
      const { data } = await http.post<PostPOInvoiceResponse>(
        "/api/v1/fi/post-po-invoice",
        payload,
      );
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/purchase-orders?supplier=<bp>&top=<n> */
  async searchPurchaseOrders(supplier: string, top = 20): Promise<PurchaseOrdersResponse> {
    try {
      const { data } = await http.get<PurchaseOrdersResponse>("/api/purchase-orders", {
        params: { supplier, top },
      });
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** GET /api/customers/search?q=<name>&top=<n> */
  async searchCustomers(q: string, top = 10): Promise<CustomerSearchResponse> {
    try {
      const { data } = await http.get<CustomerSearchResponse>("/api/customers/search", {
        params: { q, top },
      });
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/chat/message — returns raw Response for streaming */
  async streamChat(payload: ChatRequest): Promise<Response> {
    const response = await fetch('/api/v1/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`Chat failed: ${response.status}`);
    return response;
  },

  /** POST /api/v1/so/extract — extract customer PO from document */
  async extractSalesOrder(file: File, clientId = 'default'): Promise<ExtractedPurchaseOrder> {
    try {
      const form = new FormData();
      form.append('file', file);
      const { data } = await http.post<ExtractedPurchaseOrder>(
        `/api/v1/so/extract?client_id=${clientId}`, form,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/so/validate — validate extracted PO against S4 */
  async validateSalesOrder(po: ExtractedPurchaseOrder): Promise<SOValidationResult> {
    try {
      const { data } = await http.post<SOValidationResult>('/api/v1/so/validate', po);
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/so/create — create Sales Order in S4 */
  async createSalesOrder(payload: CreateSORequest): Promise<CreateSOResponse> {
    try {
      const { data } = await http.post<CreateSOResponse>('/api/v1/so/create', payload);
      return data;
    } catch (err) {
      handleError(err);
    }
  },

  /** POST /api/v1/pa/post — post Payment Advice to S/4HANA FI */
  async postPaymentAdvice(payload: PostPaymentAdviceRequest): Promise<PostPaymentAdviceResponse> {
    try {
      const { data } = await http.post<PostPaymentAdviceResponse>('/api/v1/pa/post', payload);
      return data;
    } catch (err) {
      handleError(err);
    }
  },
};