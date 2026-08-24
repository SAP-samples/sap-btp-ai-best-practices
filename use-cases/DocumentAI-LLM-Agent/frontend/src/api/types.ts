// ─── Health ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

// ─── Auth ────────────────────────────────────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// ─── Schemas ─────────────────────────────────────────────────────────────────

export interface SchemaItem {
  id?: string;
  name?: string;
  documentType?: string;
  state?: string;
  status?: string;
  [key: string]: unknown;
}

export interface SchemasResponse {
  schemas?: SchemaItem[];
  value?: SchemaItem[];
  [key: string]: unknown;
}

// ─── Templates ───────────────────────────────────────────────────────────────

export interface TemplateItem {
  id?: string;
  name?: string;
  documentType?: string;
  state?: string;
  status?: string;
  description?: string;
  [key: string]: unknown;
}

export interface TemplatesResponse {
  results?: TemplateItem[];
  templates?: TemplateItem[];
  value?: TemplateItem[];
  [key: string]: unknown;
}

// ─── Invoice / SAP DocAI ─────────────────────────────────────────────────────

export interface HeaderField {
  name?: string;
  value?: string | number | null;
  rawValue?: string;
  confidence?: number;
  [key: string]: unknown;
}

export interface LineItem {
  description?: string;
  quantity?: number | string;
  unitPrice?: number | string;
  netAmount?: number | string;
  [key: string]: unknown;
}

export interface SapExtractionResult {
  id?: string;
  status?: string;
  extraction?: {
    headerFields?: HeaderField[];
    lineItems?: LineItem[];
    [key: string]: unknown;
  };
  document?: {
    headerFields?: HeaderField[];
    lineItems?: LineItem[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface InvoiceProcessResponse {
  job_id: string;
  output_file: string;
  result: SapExtractionResult;
}

// ─── GenAI ───────────────────────────────────────────────────────────────────

export interface FieldConfidence {
  [fieldName: string]: number;
}

export interface LlmExtractionResult {
  documentNumber?: string | null;
  documentDate?: string | null;
  grossAmount?: number | string | null;
  netAmount?: number | string | null;
  taxAmount?: number | string | null;
  taxRate?: number | string | null;
  currencyCode?: string | null;
  senderName?: string | null;
  receiverName?: string | null;
  purchaseOrderNumber?: string | null;
  deliveryDate?: string | null;
  senderAddress?: string | null;
  receiverAddress?: string | null;
  senderBankAccount?: string | null;
  taxId?: string | null;
  receiverContact?: string | null;
  senderCity?: string | null;
  senderStreet?: string | null;
  senderPostalCode?: string | null;
  senderCountry?: string | null;
  receiverCity?: string | null;
  receiverStreet?: string | null;
  receiverPostalCode?: string | null;
  receiverCountry?: string | null;
  lineItems?: LineItem[];
  fieldConfidence?: FieldConfidence;
  [key: string]: unknown;
}

export interface ComparisonSummary {
  sap_fields_found: number;
  sap_confidence_avg: number;
  llm_prompting_fields_found: number;
  llm_prompting_confidence_avg: number;
  llm_structured_fields_found: number;
  llm_structured_confidence_avg: number;
  total_unique_fields: number;
  agreements: number;
  conflicts: number;
  only_in_sap: number;
  only_in_llm: number;
}

export interface ComparisonConflict {
  field: string;
  sap: unknown;
  llm_prompting: unknown;
  llm_structured: unknown;
}

export interface ComparisonResult {
  summary: ComparisonSummary;
  conflicts?: ComparisonConflict[];
  only_in_sap?: string[];
  only_in_llm?: string[];
  agreements?: string[];
  sap_normalized?: Record<string, unknown>;
  llm_prompting_normalized?: Record<string, unknown>;
  llm_structured_normalized?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface GenAIPipelineResponse {
  job_id: string;
  output_dir: string;
  route?: "genai" | "template" | "purchase_order" | "payment_advice" | null;
  document_type?: "invoice" | "purchase_order" | "payment_advice" | null;
  routing_decision?: {
    route?: string;
    decision_reason?: string;
    supplier_name?: string | null;
    template_name?: string | null;
    template_id?: string | null;
    confidence_pct?: number | null;
  };
  sap_result: SapExtractionResult;
  template_result?: SapExtractionResult | null;
  llm_prompting: LlmExtractionResult;
  llm_structured: LlmExtractionResult;
  comparison: ComparisonResult;
  po_number?: string | null;
  // Sales Order fields — populated when document_type === "purchase_order"
  extracted_po?: ExtractedPurchaseOrder | null;
  so_validation?: SOValidationResult | null;
  // Payment Advice fields — populated when document_type === "payment_advice"
  extracted_pa?: ExtractedPaymentAdvice | null;
}

// ─── Payment Advice ───────────────────────────────────────────────────────────

export interface PaymentAdviceLine {
  invoice_number: string;
  invoice_date: string;
  gross_amount: number;
  discount_amount: number;
  net_payment_amount: number;
  currency: string;
  payment_reference: string;
}

export interface ExtractedPaymentAdvice {
  payer_name: string;
  payer_bp?: string;
  payment_date: string;
  total_amount: number;
  currency: string;
  bank_reference: string;
  our_reference: string;
  payment_advice_note: string;
  line_items: PaymentAdviceLine[];
  raw_sap_fields?: Record<string, unknown>;
}

export interface PostPaymentAdviceRequest {
  payer_name: string;
  payer_bp?: string;
  payment_date: string;
  total_amount: number;
  currency: string;
  bank_reference?: string;
  payment_advice_note?: string;
  line_items: PaymentAdviceLine[];
}

export interface PostPaymentAdviceResponse {
  success: boolean;
  payment_advice: string;
  company_code: string;
  business_partner_used: string;
  payer_name_matched: string;
  error: string;
}

// ─── Evaluation ──────────────────────────────────────────────────────────────

export interface MethodScore {
  overall_score?: number;
  completeness?: number;
  confidence_avg?: number;
  fields_found?: number;
  missing_fields?: number;
  consistency?: number;
  field_coverage?: number;
  [key: string]: unknown;
}

export interface ScoresResult {
  sap?: MethodScore;
  llm_prompting?: MethodScore;
  llm_structured?: MethodScore;
  [key: string]: unknown;
}

export interface FieldAnalysis {
  total_fields?: number;
  sap_fields?: number;
  llm_prompting_fields?: number;
  llm_structured_fields?: number;
  missing_in_sap?: string[];
  missing_in_llm_prompting?: string[];
  missing_in_llm_structured?: string[];
  conflicts?: unknown[];
  [key: string]: unknown;
}

export interface LlmEvaluation {
  overall_assessment?: string;
  best_method?: string;
  recommendations?: string[];
  quality_score?: number;
  [key: string]: unknown;
}

export interface EvaluationResponse {
  analysis: FieldAnalysis;
  scores: ScoresResult;
  llm_evaluation: LlmEvaluation;
  summary: string;
  output_paths: Record<string, string>;
}

// ─── Output files ────────────────────────────────────────────────────────────

export interface OutputFilesResponse {
  directory?: string;
  files: string[];
}

// ─── FI Posting ──────────────────────────────────────────────────────────────

export interface PostInvoiceRequest {
  supplier_name: string;
  invoice_number: string;
  invoice_date: string;
  total_amount: number;
  currency: string;
  business_partner?: string;
  gl_account?: string;
}

export interface PostInvoiceResponse {
  success: boolean;
  fi_document: string;
  company_code: string;
  fiscal_year: string;
  business_partner_used: string;
  supplier_name_matched: string;
  error: string;
}

export interface PostPOInvoiceRequest {
  supplier_name: string;
  invoice_number: string;
  invoice_date: string;
  total_amount: number;
  currency: string;
  purchase_order: string;
  purchase_order_item?: string;
  tax_code?: string;
  business_partner?: string;
}

export interface PostPOInvoiceResponse {
  success: boolean;
  fi_document: string;
  company_code: string;
  fiscal_year: string;
  business_partner_used: string;
  supplier_name_matched: string;
  purchase_order: string;
  purchase_order_item: string;
  error: string;
}

export interface PurchaseOrderResult {
  purchase_order: string;
  supplier: string;
  company_code: string;
  purchasing_organization: string;
  purchasing_group: string;
  document_date: string | null;
  currency: string;
  status: string;
  supplier_name: string;
}

export interface PurchaseOrdersResponse {
  success: boolean;
  supplier: string;
  count: number;
  purchase_orders: PurchaseOrderResult[];
}

export interface CustomerSearchResult {
  business_partner: string;
  customer_name: string;
  score: number;
  confidence: string;
}

export interface CustomerSearchResponse {
  query: string;
  count: number;
  results: CustomerSearchResult[];
  source: string;
}

// ─── Sales Order ─────────────────────────────────────────────────────────────

export interface SOLineItem {
  material_code: string;
  sap_material?: string;
  description?: string;
  quantity: number;
  uom?: string;
  unit_price?: number;
  currency?: string;
}

export interface ExtractedPurchaseOrder {
  customer_name?: string;
  customer_bp?: string;
  purchase_order_number?: string;
  order_date?: string;
  requested_delivery_date?: string;
  currency?: string;
  total_amount?: number;
  line_items?: SOLineItem[];
  special_instructions?: string;
  raw_sap_fields?: Record<string, unknown>;
}

export interface SOValidationResult {
  customer_resolved: boolean;
  customer_bp: string;
  customer_name_matched: string;
  customer_score: number;
  items_validation: Array<{
    material_code_extracted: string;
    sap_material: string;
    description: string;
    matched: boolean;
    score: number;
  }>;
  ready_to_create: boolean;
  issues: string[];
}

export interface CreateSORequest {
  customer_bp: string;
  purchase_order_number?: string;
  sales_organization?: string;
  distribution_channel?: string;
  division?: string;
  currency?: string;
  items: SOLineItem[];
  special_instructions?: string;
}

export interface CreateSOResponse {
  success: boolean;
  sales_order: string;
  customer: string;
  items_created: number;
  message: string;
  error: string;
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatMessage[];
  context?: Record<string, unknown>;
}

// ─── App State ───────────────────────────────────────────────────────────────

export type Scenario =
  | "genai"
  | "evaluation"
  | "doc-ai-new"
  | "train-template"
  | "sales-order";

export type ProcessingStatus =
  | "idle"
  | "uploading"
  | "processing"
  | "completed"
  | "error";

export interface ProcessingStep {
  id: string;
  label: string;
  description: string;
  status: "pending" | "active" | "completed" | "error";
}

export interface AppState {
  scenario: Scenario;
  selectedFile: File | null;
  selectedFiles: File[];
  processingStatus: ProcessingStatus;
  processingSteps: ProcessingStep[];
  pipelineResult: GenAIPipelineResponse | null;
  invoiceResult: InvoiceProcessResponse | null;
  schemasResult: SchemasResponse | null;
  templatesResult: TemplatesResponse | null;
  evaluationResult: EvaluationResponse | null;
  fiPostResult: PostInvoiceResponse | null;
  soResult: CreateSOResponse | null;
  paResult: PostPaymentAdviceResponse | null;
  errorMessage: string | null;
  apiHealthy: boolean;
  chatMessages: ChatMessage[];
  streamingText: string;
}