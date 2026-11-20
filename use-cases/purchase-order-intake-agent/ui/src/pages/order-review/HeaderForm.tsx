/**
 * Right pane, upper part: header fields in the sections of the mock-up
 * (Document, Sold-To, Ship-To, Amounts and Terms), each with confidence and issue highlighting.
 */
import { useEffect, useState } from "react";
import { Label, Option, Select, Text, Title } from "@ui5/webcomponents-react";

import { getShipTos, searchCustomers, type OrderDetail, type ShipToPartner } from "../../services/orders";
import FieldInput from "./FieldInput";
import { fieldState, messageOf, valueStateOf } from "./fieldState";

export type HeaderDraft = Record<string, string | number | null>;

interface Props {
  detail: OrderDetail;
  draft: HeaderDraft;
  setField: (key: string, value: string | null) => void;
  classification: string;
  setClassification: (value: string) => void;
  readOnly: boolean;
}

const CLASSIFICATIONS = [
  { key: "STANDARD", text: "Standard order" },
  { key: "CONTRACT", text: "Contract-based order" },
  { key: "NOT_AN_ORDER", text: "Not an order" }
];

/**
 * Header form component.
 *
 * @param props - Order detail, editable draft and setters.
 * @returns Form sections.
 */
export default function HeaderForm({ detail, draft, setField, classification, setClassification, readOnly }: Props) {
  const [shipTos, setShipTos] = useState<ShipToPartner[]>([]);
  const [customerHits, setCustomerHits] = useState<{ text: string; additionalText?: string }[]>([]);
  const soldTo = draft.sold_to ? String(draft.sold_to) : null;
  const salesOrg = draft.sales_org ? String(draft.sales_org) : null;

  useEffect(() => {
    if (!soldTo) return setShipTos([]);
    getShipTos(soldTo, salesOrg).then(setShipTos).catch(() => setShipTos([]));
  }, [soldTo, salesOrg]);

  const field = (key: string, label: string, extra: Partial<Parameters<typeof FieldInput>[0]> = {}) => (
    <FieldInput
      key={key}
      label={label}
      value={draft[key]}
      state={fieldState(detail, 0, key.toUpperCase())}
      readOnly={readOnly}
      onChange={(value) => setField(key, value === "" ? null : value)}
      {...extra}
    />
  );

  const shipToState = fieldState(detail, 0, "SHIP_TO");
  const h = detail.header;

  return (
    <div className="review-form">
      <section>
        <Title level="H5">Document</Title>
        <div className="review-grid">
          {field("po_number", "Customer PO Number")}
          {field("po_date", "PO Date")}
          {field("req_delivery_date", "Requested Delivery")}
          {field("currency", "Currency")}
          {field("sales_org", "Sales Organisation")}
          <div className="review-field">
            <div className="review-field-label"><Label showColon>Classification</Label></div>
            <Select disabled={readOnly} onChange={(e) => setClassification(e.detail.selectedOption.dataset.key ?? "STANDARD")}>
              {CLASSIFICATIONS.map((c) => (
                <Option key={c.key} data-key={c.key} selected={classification === c.key}>{c.text}</Option>
              ))}
            </Select>
          </div>
        </div>
      </section>

      <section>
        <Title level="H5">Sold-To</Title>
        <div className="review-grid">
          {field("sold_to", "Customer Number", {
            suggestions: customerHits,
            onTyping: (q) => (q.length >= 2 ? searchCustomers(q).then((hits) =>
              setCustomerHits(hits.map((c) => ({ text: c.customer_number, additionalText: c.name })))) : undefined)
          })}
          {field("sold_to_name", "Name (as printed)")}
          {field("sold_to_address", "Address (as printed)", { wide: true })}
        </div>
      </section>

      <section>
        <Title level="H5">Ship-To</Title>
        <div className="review-grid">
          <div className="review-field review-field--wide" title={shipToState?.raw_value ?? undefined}>
            <div className="review-field-label">
              <Label showColon>Ship-To Party (partner function WE)</Label>
              {shipToState?.confidence != null && (
                <span className={`review-confidence${shipToState.confidence < 0.8 ? " review-confidence--low" : ""}`}>
                  {Math.round(shipToState.confidence * 100)}%
                </span>
              )}
            </div>
            <Select disabled={readOnly || !shipTos.length} valueState={valueStateOf(shipToState)}
              valueStateMessage={messageOf(shipToState) ? <div>{messageOf(shipToState)}</div> : undefined}
              onChange={(e) => setField("ship_to", e.detail.selectedOption.dataset.key || null)}>
              <Option data-key="" selected={!draft.ship_to}>{shipTos.length ? "Select ship-to..." : "No ship-to partners"}</Option>
              {shipTos.map((p) => (
                <Option key={p.customer_number} data-key={p.customer_number} selected={String(draft.ship_to) === p.customer_number}
                  additionalText={p.sales_org}>
                  {`${p.customer_number} - ${p.name}, ${p.address}${p.default ? " (default)" : ""}`}
                </Option>
              ))}
            </Select>
          </div>
          {field("ship_to_name", "Delivery Name (as printed)")}
          {field("ship_to_address", "Delivery Address (as printed)")}
        </div>
      </section>

      <section>
        <Title level="H5">Amounts and Terms</Title>
        <div className="review-grid">
          {field("net_amount", "Net Amount", { type: "Number" })}
          {field("gross_amount", "Gross Amount", { type: "Number" })}
          {field("payment_terms_text", "Payment Terms (as printed)")}
          {field("incoterms", "Incoterms")}
          {field("incoterms_place", "Incoterms Place")}
          {field("header_note", "Notes", { wide: true })}
        </div>
        <Text className="review-system">
          {`ECC: order type ${h.order_type ?? "-"} · channel ${h.distr_channel ?? "-"} · division ${h.division ?? "-"} · `
            + `PO type ${h.po_type ?? "-"} · payment terms ${h.payment_terms ?? "-"}`}
        </Text>
      </section>
    </div>
  );
}
