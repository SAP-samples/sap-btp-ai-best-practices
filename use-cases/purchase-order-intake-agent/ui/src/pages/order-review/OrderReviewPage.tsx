/**
 * Review page of one purchase order (mock-up "Review Purchase Order").
 *
 * Left: e-mail and documents. Right: issues, header form and line items, all editable while the
 * order is in Needs Review / Failed / Not an Order. Footer: Save (persist edits, re-validate),
 * Confirm (save + post; blocked with the reasons when required data is missing), Reprocess (new
 * agent run with an instruction), Cancel. While the agent is working the page polls.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";
import {
  Bar,
  BusyIndicator,
  Button,
  Dialog,
  MessageStrip,
  Tag,
  Text,
  TextArea,
  Title,
  Toast
} from "@ui5/webcomponents-react";

import { ApiError } from "../../services/api";
import {
  confirmOrder,
  getOrder,
  reprocessOrder,
  saveOrder,
  type Issue,
  type OrderDetail,
  type OrderItem,
  type OrderUpdate
} from "../../services/orders";
import { STATUS_DISPLAY, formatDateTime, isProcessing } from "../orders/orderStatus";
import DocumentViewer from "./DocumentViewer";
import HeaderForm, { type HeaderDraft } from "./HeaderForm";
import IssuesPanel from "./IssuesPanel";
import LineItemsTable from "./LineItemsTable";
import "./order-review.css";

const HEADER_KEYS = ["po_number", "po_date", "req_delivery_date", "currency", "sales_org", "sold_to", "sold_to_name",
  "sold_to_address", "ship_to", "ship_to_name", "ship_to_address", "net_amount", "gross_amount", "payment_terms_text",
  "incoterms", "incoterms_place", "header_note"] as const;
const ITEM_KEYS: (keyof OrderItem)[] = ["line_type", "material", "customer_material", "description", "quantity", "uom",
  "unit_price", "line_amount", "colour_code"];

/** Copy of the editable header fields. */
function headerDraftOf(detail: OrderDetail): HeaderDraft {
  return Object.fromEntries(HEADER_KEYS.map((key) => [key, detail.header[key] ?? null]));
}

/**
 * Review page component.
 *
 * @returns The page element.
 */
export default function OrderReviewPage() {
  const { orderId = "" } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<OrderDetail | null>(null);
  const [draft, setDraft] = useState<HeaderDraft>({});
  const [items, setItems] = useState<OrderItem[]>([]);
  const [classification, setClassification] = useState("STANDARD");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmIssues, setConfirmIssues] = useState<Issue[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const [reprocessOpen, setReprocessOpen] = useState(false);
  const [note, setNote] = useState("");

  const apply = useCallback((next: OrderDetail) => {
    setDetail(next);
    setDraft(headerDraftOf(next));
    setItems(next.items.map((item) => ({ ...item })));
    setClassification(next.header.classification ?? "STANDARD");
  }, []);

  const load = useCallback(async () => {
    try {
      apply(await getOrder(orderId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [orderId, apply]);

  useEffect(() => {
    void load();
  }, [load]);

  // Poll while the agent is working on this order.
  useEffect(() => {
    if (!detail || !isProcessing(detail.header.status)) return;
    const timer = setTimeout(() => void load(), 3000);
    return () => clearTimeout(timer);
  }, [detail, load]);

  /** Builds the save payload: changed header fields + the full line list. */
  const update = useMemo<OrderUpdate | null>(() => {
    if (!detail) return null;
    const original = headerDraftOf(detail);
    const header = Object.fromEntries(HEADER_KEYS.filter((k) => String(draft[k] ?? "") !== String(original[k] ?? ""))
      .map((k) => [k, draft[k]]));
    const itemsChanged = JSON.stringify(items) !== JSON.stringify(detail.items);
    const result: OrderUpdate = {};
    if (Object.keys(header).length) result.header = header;
    if (itemsChanged) {
      result.items = items.map((item) => ({
        ...(item.item_no > 0 ? { item_no: item.item_no } : {}),
        ...Object.fromEntries(ITEM_KEYS.map((k) => [k, item[k]]))
      }));
    }
    if (classification !== (detail.header.classification ?? "STANDARD")) result.classification = classification;
    return Object.keys(result).length ? result : null;
  }, [detail, draft, items, classification]);

  const run = async (label: string, action: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await action();
    } catch (err) {
      if (err instanceof ApiError && err.status === 422 && typeof err.detail === "object") {
        setConfirmIssues(((err.detail as { issues?: Issue[] }).issues) ?? []);
        await load();
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setBusy(null);
    }
  };

  const save = () => run("Saving", async () => {
    if (update) apply(await saveOrder(orderId, update));
    setConfirmIssues([]);
    setToast("Saved");
  });

  const confirm = () => run("Posting", async () => {
    const result = await confirmOrder(orderId, update ?? undefined);
    setConfirmIssues([]);
    setToast(`Posted to ECC as sales order ${result.sales_doc}`);
    await load();
  });

  const reprocess = () => run("Queuing", async () => {
    await reprocessOrder(orderId, note);
    setReprocessOpen(false);
    setNote("");
    setToast("Queued for reprocessing");
    await load();
  });

  if (!detail) {
    return error ? <MessageStrip design="Negative" hideCloseButton className="review-page-error">{error}</MessageStrip>
      : <BusyIndicator active className="content-loading" />;
  }

  const { header } = detail;
  const status = STATUS_DISPLAY[header.status];
  const editable = ["NEEDS_REVIEW", "FAILED", "NOT_AN_ORDER"].includes(header.status);
  const canConfirm = (header.status === "NEEDS_REVIEW" || header.status === "FAILED") && classification !== "NOT_AN_ORDER";

  return (
    <div className="review-page">
      <div className="review-title">
        <Button icon="nav-back" design="Transparent" tooltip="Back" onClick={() => navigate("/orders")} />
        <div>
          <div className="review-title-line">
            <Title level="H3">{`Purchase Order ${header.po_number ?? ""}`}</Title>
            <Tag design={status.design} hideStateIcon>{status.label}</Tag>
            {isProcessing(header.status) && <BusyIndicator active size="S" />}
          </div>
          <Text className="review-subtitle">
            {`${header.sold_to_name ?? detail.message.sender ?? ""} · Received ${formatDateTime(detail.message.received_at)} via `
              + `${detail.message.source_type === "EML" ? "Email" : detail.message.source_type === "TEXT" ? "pasted text" : "upload"}`
              + (detail.run ? ` · ${detail.run.llm_calls ?? 0} model calls, ${detail.run.input_tokens ?? 0} input tokens` : "")}
          </Text>
        </div>
      </div>

      {error && <MessageStrip design="Negative" onClose={() => setError(null)}>{error}</MessageStrip>}

      <div className="review-body">
        <div className="review-left"><DocumentViewer detail={detail} /></div>
        <div className="review-right">
          <IssuesPanel detail={detail} confirmIssues={confirmIssues} />
          <HeaderForm detail={detail} draft={draft} readOnly={!editable} classification={classification}
            setClassification={setClassification} setField={(key, value) => setDraft((d) => ({ ...d, [key]: value }))} />
          <LineItemsTable detail={detail} items={items} setItems={setItems} readOnly={!editable}
            soldTo={draft.sold_to ? String(draft.sold_to) : null} salesOrg={draft.sales_org ? String(draft.sales_org) : null} />
        </div>
      </div>

      <Bar
        design="FloatingFooter"
        className="review-footer"
        startContent={<Text>{busy ? `${busy}...` : "Confirming posts the validated order to ECC."}</Text>}
        endContent={
          <>
            <Button design="Emphasized" disabled={!canConfirm || Boolean(busy)} onClick={confirm}>Confirm</Button>
            <Button disabled={!editable || !update || Boolean(busy)} onClick={save}>Save</Button>
            <Button disabled={!editable || Boolean(busy)} onClick={() => setReprocessOpen(true)}>Reprocess</Button>
            <Button design="Transparent" onClick={() => navigate("/orders")}>Cancel</Button>
          </>
        }
      />

      <Dialog open={reprocessOpen} headerText="Reprocess with the agent" onClose={() => setReprocessOpen(false)}
        footer={<Bar design="Footer" endContent={<>
          <Button design="Emphasized" onClick={reprocess}>Reprocess</Button>
          <Button design="Transparent" onClick={() => setReprocessOpen(false)}>Cancel</Button>
        </>} />}>
        <Text>Optional instruction for the agent (e.g. "deliver to the second ship-to", "line 3 is a free item"):</Text>
        <TextArea rows={4} value={note} onInput={(e) => setNote(e.target.value)} className="review-note" />
      </Dialog>
      <Toast open={toast !== null} onClose={() => setToast(null)}>{toast}</Toast>
    </div>
  );
}
