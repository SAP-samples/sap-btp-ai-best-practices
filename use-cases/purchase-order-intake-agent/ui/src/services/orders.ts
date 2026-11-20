/**
 * Typed calls to the purchase-order API (`api/app/routers/orders.py`, `master_data.py`).
 * Types mirror `api/app/models/orders.py`; numbers are in display format (no leading zeros).
 */
import { request, requestBlob, requestForm } from "./api";

export type OrderStatus = "QUEUED" | "PROCESSING" | "NEEDS_REVIEW" | "POSTED" | "FAILED" | "NOT_AN_ORDER";
export type Severity = "ERROR" | "REVIEW" | "WARNING" | "INFO";
export type LineType = "MATERIAL" | "CHARGE" | "TEXT";

/** Row of the incoming purchase orders table. */
export interface OrderSummary {
  order_id: string;
  status: OrderStatus;
  classification: string | null;
  status_message: string | null;
  po_number: string | null;
  po_date: string | null;
  sold_to: string | null;
  sold_to_name: string | null;
  ship_to: string | null;
  ship_to_name: string | null;
  sales_org: string | null;
  currency: string | null;
  net_amount: number | null;
  sales_doc: string | null;
  sender: string | null;
  subject: string | null;
  received_at: string | null;
  source_type: string | null;
  item_count: number;
  issue_count: number;
  min_confidence: number | null;
  updated_at: string | null;
}

/** Header of one order (editable and system fields). */
export interface OrderHeader {
  order_id: string;
  message_id: string;
  status: OrderStatus;
  classification: string | null;
  classification_reason: string | null;
  status_message: string | null;
  attempts: number;
  reviewer_note: string | null;
  po_number: string | null;
  po_date: string | null;
  req_delivery_date: string | null;
  sold_to: string | null;
  sold_to_name: string | null;
  sold_to_address: string | null;
  ship_to: string | null;
  ship_to_name: string | null;
  ship_to_address: string | null;
  sales_org: string | null;
  distr_channel: string | null;
  division: string | null;
  order_type: string | null;
  po_type: string | null;
  currency: string | null;
  net_amount: number | null;
  gross_amount: number | null;
  payment_terms_text: string | null;
  payment_terms: string | null;
  incoterms: string | null;
  incoterms_place: string | null;
  header_note: string | null;
  agent_summary: string | null;
  sales_doc: string | null;
  posted_at: string | null;
  created_at: string | null;
}

/** One order line. */
export interface OrderItem {
  item_no: number;
  line_type: LineType;
  customer_item_no: string | null;
  material: string | null;
  customer_material: string | null;
  description: string | null;
  quantity: number | null;
  uom: string | null;
  unit_price: number | null;
  line_amount: number | null;
  colour_code: string | null;
  plant: string | null;
}

/** Extraction / validation metadata of one field. */
export interface FieldState {
  raw_value: string | null;
  source_ref: string | null;
  extract_confidence: number | null;
  match_confidence: number | null;
  confidence: number | null;
  changed_by: "AGENT" | "SYSTEM" | "USER";
  fix_note: string | null;
  issue_severity: Severity | null;
  issue_code: string | null;
  issue_text: string | null;
  /** Material match evidence from the agent plus the system's price check. */
  evidence?: MatchEvidence | null;
}

/** Why the agent chose a material (stored per MATERIAL field). */
export interface MatchEvidence {
  matched?: string[];
  conflicts?: string[];
  alternatives?: string[];
  price_check?: string | null;
}

/** Validation finding (item_no 0 = header; fields starting with "_" are order-level). */
export interface Issue {
  severity: Severity;
  item_no: number;
  field: string;
  code: string | null;
  message: string;
}

/** Stored file of the inbound message. */
export interface AttachmentInfo {
  attachment_id: string;
  parent_id: string | null;
  file_name: string;
  mime_type: string | null;
  size_bytes: number | null;
  role: "SOURCE_EMAIL" | "DOCUMENT" | "INLINE_IMAGE" | "FORWARDED_EMAIL";
  model_input: "NATIVE" | "TEXT" | "NONE";
  model_input_reason: string | null;
  viewer: "pdf" | "image" | "text" | "download";
  extracted_text: string | null;
}

/** Everything the review page needs. */
export interface OrderDetail {
  header: OrderHeader;
  items: OrderItem[];
  fields: Record<string, Record<string, FieldState>>;
  issues: Issue[];
  message: {
    source_type: string;
    sender: string | null;
    recipients: string | null;
    subject: string | null;
    sent_at: string | null;
    received_at: string | null;
    body_text: string | null;
    sales_org_hint: string | null;
  };
  attachments: AttachmentInfo[];
  run: {
    outcome: string | null;
    model: string | null;
    llm_calls: number | null;
    input_tokens: number | null;
    cached_input_tokens: number | null;
    output_tokens: number | null;
    error_text: string | null;
  } | null;
}

/** Posted sales document (order history). */
export interface SalesOrderSummary {
  sales_doc: string;
  order_type: string | null;
  sales_org: string | null;
  sold_to: string | null;
  sold_to_name: string | null;
  ship_to: string | null;
  ship_to_name: string | null;
  po_number: string | null;
  po_date: string | null;
  currency: string | null;
  net_value: number | null;
  item_count: number;
  created_on: string | null;
  order_id: string | null;
}

/** Line in a save request; omitted keys keep their value. */
export type ItemUpdate = Partial<Omit<OrderItem, "item_no">> & { item_no?: number };

/** Body of save / confirm. */
export interface OrderUpdate {
  header?: Record<string, string | number | null>;
  items?: ItemUpdate[];
  classification?: string;
}

/** Ship-to partner offered on the review page. */
export interface ShipToPartner {
  customer_number: string;
  name: string;
  address: string;
  default: boolean;
  blocked: boolean;
  sales_org: string;
}

/** Customer search hit. */
export interface CustomerHit {
  customer_number: string;
  name: string;
  address: string;
  score: number;
}

/** Material search hit. */
export interface MaterialHit {
  material_number: string;
  description: string | null;
  sales_code: string | null;
  base_unit: string | null;
  allowed_units: string[];
  score: number;
}

export const listOrders = () => request<{ items: OrderSummary[] }>("/api/orders").then((r) => r.items);
export const listSalesOrders = () => request<{ items: SalesOrderSummary[] }>("/api/sales-orders").then((r) => r.items);
export const listSalesOrgs = () => request<{ items: string[] }>("/api/master-data/sales-orgs").then((r) => r.items);
export const getOrder = (orderId: string) => request<OrderDetail>(`/api/orders/${orderId}`);
export const saveOrder = (orderId: string, update: OrderUpdate) => request<OrderDetail>(`/api/orders/${orderId}`, "PUT", update);
export const confirmOrder = (orderId: string, update?: OrderUpdate) =>
  request<{ order_id: string; status: OrderStatus; sales_doc: string }>(`/api/orders/${orderId}/confirm`, "POST", update ?? null);
export const reprocessOrder = (orderId: string, note: string) =>
  request<{ order_id: string; status: OrderStatus }>(`/api/orders/${orderId}/reprocess`, "POST", { note: note || null });
export const fetchAttachment = (orderId: string, attachmentId: string) =>
  requestBlob(`/api/orders/${orderId}/attachments/${attachmentId}`);
export const getShipTos = (customer: string, salesOrg?: string | null) =>
  request<{ ship_tos: ShipToPartner[] }>(
    `/api/master-data/customers/${encodeURIComponent(customer)}/partners${salesOrg ? `?sales_org=${salesOrg}` : ""}`
  ).then((r) => r.ship_tos);
export const searchCustomers = (q: string) =>
  request<{ items: CustomerHit[] }>(`/api/master-data/customers?q=${encodeURIComponent(q)}`).then((r) => r.items);
export const searchMaterials = (q: string, salesOrg?: string | null, soldTo?: string | null) => {
  const params = new URLSearchParams({ q });
  if (salesOrg) params.set("sales_org", salesOrg);
  if (soldTo) params.set("sold_to", soldTo);
  return request<{ items: MaterialHit[] }>(`/api/master-data/materials?${params}`).then((r) => r.items);
};

/**
 * PoC only: deletes an order, its lines, agent runs and (if posted) its mock sales document, so the
 * same document can be uploaded again without a duplicate warning.
 *
 * @param orderId - Order id.
 * @returns Deletion confirmation.
 */
export const deleteOrder = (orderId: string) =>
  request<{ order_id: string; deleted: boolean }>(`/api/orders/${orderId}`, "DELETE");

/**
 * Uploads e-mails / documents; every .eml becomes its own order, other files share one.
 *
 * @param files - Selected files.
 * @param emailText - Optional pasted e-mail content.
 * @param salesOrg - Optional sales organisation (simulated inbox mapping).
 * @returns Created orders.
 */
export function uploadOrders(files: File[], emailText: string, salesOrg: string) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  if (emailText.trim()) form.append("email_text", emailText);
  if (salesOrg) form.append("sales_org", salesOrg);
  return requestForm<{ orders: { order_id: string; status: OrderStatus; source: string }[] }>("/api/orders/upload", form);
}
