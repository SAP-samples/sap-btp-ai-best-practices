/**
 * "Incoming Purchase Orders" table: one row per inbound message with its processing status.
 * Clicking a row (or Review) opens the review page; the bin icon deletes the order (PoC re-testing).
 */
import { useNavigate } from "react-router";
import {
  BusyIndicator,
  Button,
  Icon,
  Table,
  TableCell,
  TableHeaderCell,
  TableHeaderRow,
  TableRow,
  Tag,
  Text
} from "@ui5/webcomponents-react";

import type { OrderSummary } from "../../services/orders";
import { STATUS_DISPLAY, formatDateTime, isProcessing } from "./orderStatus";

/**
 * Renders the incoming orders.
 *
 * @param props.orders - Orders to show (already filtered).
 * @param props.loading - Initial loading state.
 * @param props.onDelete - Called with an order after the user confirmed its deletion.
 * @returns Table element.
 */
export default function IncomingOrdersTable({ orders, loading, onDelete }:
  { orders: OrderSummary[]; loading: boolean; onDelete: (order: OrderSummary) => void }) {
  const navigate = useNavigate();

  return (
    <Table
      loading={loading}
      noDataText="No purchase orders yet. Use Upload to add one."
      onRowClick={(event) => navigate(`/orders/${event.detail.row.rowKey}`)}
      headerRow={
        <TableHeaderRow sticky>
          <TableHeaderCell minWidth="160px">Purchase Order</TableHeaderCell>
          <TableHeaderCell minWidth="200px">Customer</TableHeaderCell>
          <TableHeaderCell minWidth="140px">Received</TableHeaderCell>
          <TableHeaderCell minWidth="60px">Lines</TableHeaderCell>
          <TableHeaderCell minWidth="150px">Processing Status</TableHeaderCell>
          <TableHeaderCell minWidth="110px">Sales Order</TableHeaderCell>
          <TableHeaderCell minWidth="130px">Action</TableHeaderCell>
        </TableHeaderRow>
      }
    >
      {orders.map((order) => {
        const display = STATUS_DISPLAY[order.status];
        const icon = order.source_type === "EML" ? "email" : order.source_type === "TEXT" ? "notes" : "attachment";
        return (
          <TableRow key={order.order_id} rowKey={order.order_id} interactive>
            <TableCell>
              <span className="orders-po">
                <Icon name={icon} />
                <Text>{order.po_number || order.subject || "(not yet extracted)"}</Text>
              </span>
            </TableCell>
            <TableCell>
              <Text>{order.sold_to_name || order.sender || "-"}</Text>
            </TableCell>
            <TableCell>
              <Text>{formatDateTime(order.received_at)}</Text>
            </TableCell>
            <TableCell>
              <Text>{order.item_count || "-"}</Text>
            </TableCell>
            <TableCell>
              <span className="orders-status" title={order.status_message ?? undefined}>
                {isProcessing(order.status) && <BusyIndicator active size="S" />}
                <Tag design={display.design} hideStateIcon>{display.label}</Tag>
                {order.status === "NEEDS_REVIEW" && order.issue_count > 0 && <Text>{order.issue_count}</Text>}
              </span>
            </TableCell>
            <TableCell>
              <Text>{order.sales_doc ?? "-"}</Text>
            </TableCell>
            <TableCell>
              <Button design="Transparent" disabled={isProcessing(order.status)}
                onClick={(event) => { event.stopPropagation(); navigate(`/orders/${order.order_id}`); }}>
                {order.status === "POSTED" || order.status === "NOT_AN_ORDER" ? "View" : "Review"}
              </Button>
              <Button design="Transparent" icon="delete" tooltip="Delete order (test data)"
                disabled={order.status === "PROCESSING"}
                onClick={(event) => {
                  event.stopPropagation();
                  const name = order.po_number || order.subject || "this order";
                  const posted = order.sales_doc ? ` and its sales order ${order.sales_doc}` : "";
                  if (window.confirm(`Delete ${name}${posted}? This cannot be undone.`)) onDelete(order);
                }} />
            </TableCell>
          </TableRow>
        );
      })}
    </Table>
  );
}
