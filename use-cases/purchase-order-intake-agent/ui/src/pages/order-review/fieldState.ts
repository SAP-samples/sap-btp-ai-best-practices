/**
 * Helpers mapping per-field validation metadata to UI5 value states and texts.
 */
import type { FieldState, OrderDetail } from "../../services/orders";

export type ValueState = "None" | "Positive" | "Critical" | "Negative" | "Information";

/**
 * Returns the metadata of one field.
 *
 * @param detail - Order detail.
 * @param itemNo - 0 for header fields, otherwise the item number.
 * @param column - Upper-case field name, e.g. "PO_NUMBER".
 * @returns The field state or undefined.
 */
export function fieldState(detail: OrderDetail, itemNo: number, column: string): FieldState | undefined {
  return detail.fields[String(itemNo)]?.[column];
}

/**
 * UI5 value state of a field: issues first (ERROR red, REVIEW/WARNING orange), then system
 * fills from master data (blue), otherwise none.
 *
 * @param state - Field state.
 * @returns Value state.
 */
export function valueStateOf(state: FieldState | undefined): ValueState {
  switch (state?.issue_severity) {
    case "ERROR":
      return "Negative";
    case "REVIEW":
    case "WARNING":
      return "Critical";
    default:
      // Blue only for values the system filled from master data; agent notes stay in the tooltip.
      return state?.changed_by === "SYSTEM" ? "Information" : "None";
  }
}

/**
 * Text shown under the field (issue text, otherwise the fix note).
 *
 * @param state - Field state.
 * @returns Message or null.
 */
export function messageOf(state: FieldState | undefined): string | null {
  if (state?.issue_text && state.issue_severity !== "INFO") return state.issue_text;
  return state?.fix_note ?? state?.issue_text ?? null;
}

/**
 * Who set the value: "Extracted by: agent" when the agent only read it, "Changed by: ..." when the
 * agent applied a fix (it has a note), the system filled it or a reviewer edited it.
 *
 * @param state - Field state.
 * @returns One tooltip line.
 */
function authorLine(state: FieldState): string {
  if (state.changed_by === "AGENT" && !state.fix_note) return "Extracted by: agent";
  return `Changed by: ${state.changed_by.toLowerCase()}`;
}

/**
 * Tooltip with the printed value, its source, the agent's note, the material match evidence
 * and the confidences.
 *
 * @param state - Field state.
 * @returns Multi-line tooltip text.
 */
export function tooltipOf(state: FieldState | undefined): string | undefined {
  if (!state) return undefined;
  const evidence = state.evidence;
  const list = (label: string, values?: string[]) => (values?.length ? `${label}: ${values.join("; ")}` : null);
  return [
    state.raw_value ? `Printed: ${state.raw_value}` : null,
    state.source_ref ? `Source: ${state.source_ref}` : null,
    state.fix_note ? `${state.changed_by === "SYSTEM" ? "System fill" : "Agent note"}: ${state.fix_note}` : null,
    list("Matched", evidence?.matched),
    list("Conflicts", evidence?.conflicts),
    list("Alternatives", evidence?.alternatives),
    evidence?.price_check ? `Price: ${evidence.price_check}` : null,
    state.extract_confidence !== null ? `Agent confidence: ${Math.round(state.extract_confidence * 100)}%` : null,
    state.match_confidence !== null ? `Master-data text match: ${Math.round(state.match_confidence * 100)}%` : null,
    authorLine(state)
  ].filter(Boolean).join("\n");
}
