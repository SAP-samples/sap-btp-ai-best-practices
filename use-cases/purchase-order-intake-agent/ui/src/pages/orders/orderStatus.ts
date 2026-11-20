/**
 * Display mapping of order statuses to UI5 Tag designs (pill colours of the mock-up):
 * Processing = blue, Needs Review = orange, Posted = green, Failed = red, Not an Order = grey.
 */
import type { OrderStatus } from "../../services/orders";

type TagDesign = "Information" | "Critical" | "Positive" | "Negative" | "Neutral";

/** Label and Tag design per status (QUEUED is shown as Processing to the user). */
export const STATUS_DISPLAY: Record<OrderStatus, { label: string; design: TagDesign }> = {
  QUEUED: { label: "Processing", design: "Information" },
  PROCESSING: { label: "Processing", design: "Information" },
  NEEDS_REVIEW: { label: "Needs Review", design: "Critical" },
  POSTED: { label: "Posted", design: "Positive" },
  FAILED: { label: "Failed", design: "Negative" },
  NOT_AN_ORDER: { label: "Not an Order", design: "Neutral" }
};

/**
 * Tells whether the agent is still working on an order.
 *
 * @param status - Order status.
 * @returns True for QUEUED and PROCESSING.
 */
export const isProcessing = (status: OrderStatus) => status === "QUEUED" || status === "PROCESSING";

/**
 * Formats an ISO timestamp as "YYYY-MM-DD HH:MM" (local time).
 *
 * @param value - ISO date-time string.
 * @returns Formatted text or "-".
 */
export function formatDateTime(value: string | null): string {
  if (!value) return "-";
  const date = new Date(value.endsWith("Z") || value.includes("+") ? value : `${value}Z`);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/**
 * Formats an amount with thousands separators.
 *
 * @param value - Amount.
 * @param currency - ISO currency code.
 * @returns Formatted text or "-".
 */
export function formatAmount(value: number | null, currency: string | null): string {
  if (value === null || value === undefined) return "-";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${currency ?? ""}`.trim();
}
