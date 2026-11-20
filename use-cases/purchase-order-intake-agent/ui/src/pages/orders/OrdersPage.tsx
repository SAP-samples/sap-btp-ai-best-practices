/**
 * Purchase Order Intake page: filter bar, "Incoming Purchase Orders" (with Upload) and
 * "Order History" (posted sales orders). Polls every 3 s while any order is being processed.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Card, CardHeader, Icon, Input, Option, Select, Title, Toast } from "@ui5/webcomponents-react";

import { ApiError } from "../../services/api";
import { deleteOrder, listOrders, listSalesOrders, type OrderStatus, type OrderSummary } from "../../services/orders";
import IncomingOrdersTable from "./IncomingOrdersTable";
import OrderHistoryTable from "./OrderHistoryTable";
import UploadOrderDialog from "./UploadOrderDialog";
import { STATUS_DISPLAY, isProcessing } from "./orderStatus";
import { usePolling } from "./usePolling";
import "./orders.css";

const STATUS_FILTERS: { key: string; label: string; match: (status: OrderStatus) => boolean }[] = [
  { key: "open", label: "Open Documents", match: (s) => s !== "POSTED" && s !== "NOT_AN_ORDER" },
  { key: "all", label: "All Documents", match: () => true },
  { key: "processing", label: "Processing", match: isProcessing },
  { key: "review", label: "Needs Review", match: (s) => s === "NEEDS_REVIEW" },
  { key: "failed", label: "Failed", match: (s) => s === "FAILED" },
  { key: "posted", label: "Posted", match: (s) => s === "POSTED" },
  { key: "not-order", label: "Not an Order", match: (s) => s === "NOT_AN_ORDER" }
];

/**
 * Orders overview page.
 *
 * @returns The page element.
 */
export default function OrdersPage() {
  const [uploadOpen, setUploadOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("open");
  const [toast, setToast] = useState<string | null>(null);

  const orders = usePolling(listOrders, (items) => items.some((o) => isProcessing(o.status)));
  const history = usePolling(listSalesOrders, () => false);

  const visible = useMemo(() => {
    const match = STATUS_FILTERS.find((f) => f.key === filter)?.match ?? (() => true);
    const query = search.trim().toLowerCase();
    return (orders.data ?? []).filter((order) =>
      match(order.status) &&
      (!query || [order.po_number, order.sold_to_name, order.sender, order.subject, order.sales_doc]
        .some((value) => value?.toLowerCase().includes(query))));
  }, [orders.data, filter, search]);

  // Refresh the history whenever an order leaves processing (it may have been posted).
  const processingCount = (orders.data ?? []).filter((o) => isProcessing(o.status)).length;
  const lastProcessing = useRef(processingCount);
  const reloadHistory = history.reload;
  useEffect(() => {
    if (processingCount < lastProcessing.current) void reloadHistory();
    lastProcessing.current = processingCount;
  }, [processingCount, reloadHistory]);

  const handleDelete = (order: OrderSummary) => {
    deleteOrder(order.order_id)
      .then(() => {
        setToast(`${order.po_number || "Order"} deleted`);
        void orders.reload();
        void history.reload();
      })
      .catch((error: unknown) => setToast(error instanceof ApiError ? error.message : "Delete failed"));
  };

  const counts = (orders.data ?? []).reduce<Record<string, number>>((acc, order) => {
    const label = STATUS_DISPLAY[order.status].label;
    acc[label] = (acc[label] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="orders-container">
      <div className="orders-header">
        <Title level="H3">Track Purchase Orders ({orders.data?.length ?? 0})</Title>
        <div className="orders-filters">
          <Input placeholder="Search PO, customer, sender..." icon={<Icon name="search" />} value={search}
            onInput={(event) => setSearch(event.target.value)} showClearIcon />
          <Select onChange={(event) => setFilter(event.detail.selectedOption.dataset.key ?? "open")}>
            {STATUS_FILTERS.map((f) => (
              <Option key={f.key} data-key={f.key} selected={f.key === filter}>{f.label}</Option>
            ))}
          </Select>
          <span className="orders-counts">
            {Object.entries(counts).map(([label, count]) => `${label}: ${count}`).join(" · ")}
          </span>
        </div>
      </div>

      <Card
        className="orders-card"
        header={
          <CardHeader
            titleText={`Incoming Purchase Orders (${visible.length})`}
            action={<Button icon="upload" design="Transparent" onClick={() => setUploadOpen(true)}>Upload</Button>}
          />
        }
      >
        <IncomingOrdersTable orders={visible} loading={orders.loading} onDelete={handleDelete} />
      </Card>

      <Card className="orders-card" header={<CardHeader titleText={`Order History (${history.data?.length ?? 0})`} />}>
        <OrderHistoryTable orders={history.data ?? []} loading={history.loading} />
      </Card>

      <UploadOrderDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUploaded={(count) => {
          setToast(`${count} order${count === 1 ? "" : "s"} queued for processing`);
          void orders.reload();
        }}
      />
      <Toast open={toast !== null} onClose={() => setToast(null)}>{toast}</Toast>
    </div>
  );
}
