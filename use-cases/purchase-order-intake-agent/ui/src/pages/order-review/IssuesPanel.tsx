/**
 * Summary strip and list of what needs attention, plus the agent's note for the reviewer.
 */
import { MessageStrip, Text, Title } from "@ui5/webcomponents-react";

import type { Issue, OrderDetail } from "../../services/orders";

const FIELD_LABELS: Record<string, string> = {
  _CLASSIFICATION: "Classification", _DUPLICATE: "Duplicate", _MULTIPLE_ORDERS: "Message",
  _NO_MATERIAL_LINES: "Lines", LINE_TYPE: "Line type"
};

/** Human-readable location of an issue ("Header · SHIP_TO", "Item 2 · MATERIAL"). */
function where(issue: Issue): string {
  const field = issue.field.startsWith("_AGENT_FLAG_") ? "Agent" : FIELD_LABELS[issue.field] ?? issue.field.toLowerCase().replace(/_/g, " ");
  return issue.item_no ? `Item ${issue.item_no} · ${field}` : field;
}

/**
 * Issues panel.
 *
 * @param props.detail - Order detail.
 * @param props.confirmIssues - ERROR issues returned by a rejected confirm (shown on top).
 * @returns Strips and lists.
 */
export default function IssuesPanel({ detail, confirmIssues }: { detail: OrderDetail; confirmIssues: Issue[] }) {
  const blocking = detail.issues.filter((i) => i.severity === "ERROR" || i.severity === "REVIEW");
  const warnings = detail.issues.filter((i) => i.severity === "WARNING");
  const fixes = detail.issues.filter((i) => i.severity === "INFO");
  const { header } = detail;
  return (
    <div className="review-issues">
      {confirmIssues.length > 0 && (
        <MessageStrip design="Negative" hideCloseButton>
          Confirmation blocked: {confirmIssues.map((i) => `${where(i)}: ${i.message}`).join(" | ")}
        </MessageStrip>
      )}
      {header.status === "QUEUED" || header.status === "PROCESSING" ? (
        <MessageStrip design="Information" hideCloseButton>
          {header.status_message ?? "The agent is processing this order."} The page updates automatically.
        </MessageStrip>
      ) : header.status === "POSTED" ? (
        <MessageStrip design="Positive" hideCloseButton>{`Posted to ECC as sales order ${header.sales_doc}.`}</MessageStrip>
      ) : header.status === "FAILED" ? (
        <MessageStrip design="Negative" hideCloseButton>{header.status_message ?? "Processing failed."}</MessageStrip>
      ) : header.status === "NOT_AN_ORDER" ? (
        <MessageStrip design="Information" hideCloseButton>{`Not an order: ${header.classification_reason ?? ""}`}</MessageStrip>
      ) : blocking.length > 0 ? (
        <MessageStrip design="Critical" hideCloseButton>
          {`${blocking.length} ${blocking.length === 1 ? "value needs" : "values need"} attention. Verify the highlighted values against the original document.`}
        </MessageStrip>
      ) : (
        <MessageStrip design="Positive" hideCloseButton>All checks passed. Confirm to post the order to ECC.</MessageStrip>
      )}
      {(blocking.length > 0 || warnings.length > 0) && (
        <ul className="review-issue-list">
          {[...blocking, ...warnings].map((issue, index) => (
            <li key={index} className={`review-issue review-issue--${issue.severity.toLowerCase()}`}>
              <strong>{where(issue)}</strong> {issue.message}
            </li>
          ))}
        </ul>
      )}
      {header.agent_summary && (
        <div className="review-agent">
          <Title level="H6">Agent summary</Title>
          <Text>{header.agent_summary}</Text>
        </div>
      )}
      {fixes.length > 0 && (
        <details className="review-fixes">
          <summary>{`${fixes.length} value${fixes.length === 1 ? "" : "s"} filled automatically from master data`}</summary>
          <ul>{fixes.map((issue, index) => <li key={index}><strong>{where(issue)}</strong> {issue.message}</li>)}</ul>
        </details>
      )}
    </div>
  );
}
