/**
 * docai-new-types.ts
 * ------------------
 * TypeScript types for the DOC AI NEW pipeline.
 */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Annotation {
  fieldName: string;
  value: string;
  page: number;
  boundingBox: BoundingBox;
}

export interface LineItem {
  description: string | null;
  quantity: number | null;
  unit_price: number | null;
  line_total: number | null;
}

export interface FieldCoordinates {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ExtractionResult {
  customer_name: string | null;
  customer_address: string | null;
  customer_tax_id: string | null;
  invoice_number: string | null;
  invoice_date: string | null;
  due_date: string | null;
  subtotal: number | null;
  tax_amount: number | null;
  total_amount: number | null;
  currency: string | null;
  line_items: LineItem[];
  field_coordinates: Record<string, FieldCoordinates>;
  confidence: Record<string, number>;
}

export interface DocAiNewResult {
  filename: string;
  pdf_type: "searchable" | "scanned" | null;
  extraction: ExtractionResult | null;
  customer_name: string | null;
  template: Record<string, unknown> | null;
  template_id: string | null;
  template_name: string | null;
  template_created: boolean;
  sap_result: Record<string, unknown> | null;
  annotations: Annotation[];
  route: "existing_template" | "template_created" | "free_prompt" | "free_prompt_only" | "error" | null;
  errors: string[];
}

export interface DocAiNewProcessResponse {
  results: DocAiNewResult[];
  total: number;
}

export interface DocAiNewTemplate {
  id: string;
  name: string;
  schemaName?: string;
  documentType?: string;
  status?: string;
  createdAt?: string;
}

export interface DocAiNewTemplatesResponse {
  templates: DocAiNewTemplate[];
  total: number;
}

export interface TrainingResult {
  template_id: string;
  documents_processed: number;
  fields_annotated: number;
  training_status: "triggered" | "skipped" | "failed";
  training_result: Record<string, unknown>;
  extraction_results: Array<{
    filename: string;
    extraction: ExtractionResult;
  }>;
  errors: string[];
  success: boolean;
}