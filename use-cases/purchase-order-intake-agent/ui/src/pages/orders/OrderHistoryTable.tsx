/**
 * "Order History": sales documents posted to the (mock) ECC system, newest first.
 */
import { useNavigate } from "react-router";
import { Table, TableCell, TableHeaderCell, TableHeaderRow, TableRow, Text } from "@ui5/webcomponents-react";

import type { SalesOrderSummary } from "../../services/orders";
import { formatAmount } from "./orderStatus";

/**
 * Renders posted sales orders; a row opens the originating purchase order.
 *
 * @param props.orders - Posted sales documents.
 * @param props.loading - Initial loading state.
 * @returns Table element.
 */
export default function OrderHistoryTable({ orders, loading }: { orders: SalesOrderSummary[]; loading: boolean }) {
  const navigate = useNavigate();
  return (
    <Table
      loading={loading}
      noDataText="No sales orders posted yet."
      onRowClick={(event) => event.detail.row.rowKey && navigate(`/orders/${event.detail.row.rowKey}`)}
      headerRow={
        <TableHeaderRow sticky>
          <TableHeaderCell minWidth="120px">Sales Order</TableHeaderCell>
          <TableHeaderCell minWidth="140px">Customer PO</TableHeaderCell>
          <TableHeaderCell minWidth="200px">Sold-To</TableHeaderCell>
          <TableHeaderCell minWidth="200px">Ship-To</TableHeaderCell>
          <TableHeaderCell minWidth="90px">Sales Org</TableHeaderCell>
          <TableHeaderCell minWidth="60px">Items</TableHeaderCell>
          <TableHeaderCell minWidth="130px">Value</TableHeaderCell>
          <TableHeaderCell minWidth="110px">Created</TableHeaderCell>
        </TableHeaderRow>
      }
    >
      {orders.map((order) => (
        <TableRow key={order.sales_doc} rowKey={order.order_id ?? ""} interactive={Boolean(order.order_id)}>
          <TableCell><Text>{order.sales_doc}</Text></TableCell>
          <TableCell><Text>{order.po_number ?? "-"}</Text></TableCell>
          <TableCell><Text>{`${order.sold_to_name ?? ""} (${order.sold_to ?? "-"})`}</Text></TableCell>
          <TableCell><Text>{`${order.ship_to_name ?? ""} (${order.ship_to ?? "-"})`}</Text></TableCell>
          <TableCell><Text>{order.sales_org ?? "-"}</Text></TableCell>
          <TableCell><Text>{order.item_count}</Text></TableCell>
          <TableCell><Text>{formatAmount(order.net_value, order.currency)}</Text></TableCell>
          <TableCell><Text>{order.created_on ?? "-"}</Text></TableCell>
        </TableRow>
      ))}
    </Table>
  );
}
