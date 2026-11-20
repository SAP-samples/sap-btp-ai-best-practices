/**
 * Right pane, lower part: editable order lines with per-cell confidence and issue highlighting,
 * material value help, add / remove line.
 */
import { useState } from "react";
import {
  Button,
  Input,
  Option,
  Select,
  SuggestionItem,
  Table,
  TableCell,
  TableHeaderCell,
  TableHeaderRow,
  TableRow,
  Title
} from "@ui5/webcomponents-react";

import { searchMaterials, type LineType, type OrderDetail, type OrderItem } from "../../services/orders";
import { fieldState, messageOf, tooltipOf, valueStateOf } from "./fieldState";

interface Props {
  detail: OrderDetail;
  items: OrderItem[];
  setItems: (items: OrderItem[]) => void;
  soldTo: string | null;
  salesOrg: string | null;
  readOnly: boolean;
}

const EDITABLE: { key: keyof OrderItem; label: string; width: string; numeric?: boolean }[] = [
  { key: "material", label: "Material", width: "140px" },
  { key: "description", label: "Description", width: "220px" },
  { key: "quantity", label: "Quantity", width: "90px", numeric: true },
  { key: "uom", label: "UoM", width: "70px" },
  { key: "unit_price", label: "Unit Price", width: "100px", numeric: true },
  { key: "line_amount", label: "Net Amount", width: "110px", numeric: true },
  { key: "colour_code", label: "RAL / tint code", width: "110px" },
  { key: "customer_material", label: "Customer Material", width: "130px" }
];

/**
 * Line items component.
 *
 * @param props - Detail (for field states), editable items and context for value help.
 * @returns Title + table.
 */
export default function LineItemsTable({ detail, items, setItems, soldTo, salesOrg, readOnly }: Props) {
  const [materialHits, setMaterialHits] = useState<{ text: string; additionalText?: string }[]>([]);

  const lookUpMaterials = (query: string) => {
    void searchMaterials(query, salesOrg, soldTo).then((hits) =>
      setMaterialHits(hits.map((m) => ({ text: m.material_number, additionalText: `${m.sales_code ?? ""} ${m.description ?? ""}` }))));
  };

  const update = (index: number, key: keyof OrderItem, raw: string) => {
    const column = EDITABLE.find((c) => c.key === key);
    const value = raw === "" ? null : column?.numeric ? Number(raw.replace(",", ".")) : raw;
    setItems(items.map((item, i) => (i === index ? { ...item, [key]: value } : item)));
  };

  const addLine = () => {
    const next = Math.max(0, ...items.map((i) => i.item_no)) + 10;
    setItems([...items, { item_no: -next, line_type: "MATERIAL", customer_item_no: null, material: null,
      customer_material: null, description: null, quantity: null, uom: "PC", unit_price: null, line_amount: null,
      colour_code: null, plant: null }]);
  };

  return (
    <div className="review-lines">
      <div className="review-lines-header">
        <Title level="H5">Line Items ({items.length})</Title>
        {!readOnly && <Button design="Transparent" icon="add" onClick={addLine}>Add</Button>}
      </div>
      <Table
        headerRow={
          <TableHeaderRow>
            <TableHeaderCell width="60px">Item</TableHeaderCell>
            <TableHeaderCell width="120px">Type</TableHeaderCell>
            {EDITABLE.map((c) => <TableHeaderCell key={c.key} width={c.width}>{c.label}</TableHeaderCell>)}
            <TableHeaderCell width="70px">Plant</TableHeaderCell>
            <TableHeaderCell width="50px" />
          </TableHeaderRow>
        }
      >
        {items.map((item, index) => (
          <TableRow key={item.item_no} rowKey={String(item.item_no)}>
            <TableCell>{item.item_no > 0 ? item.item_no : "new"}</TableCell>
            <TableCell>
              <Select disabled={readOnly} onChange={(e) => {
                const value = (e.detail.selectedOption.dataset.key ?? "MATERIAL") as LineType;
                setItems(items.map((it, i) => (i === index ? { ...it, line_type: value } : it)));
              }}>
                {(["MATERIAL", "CHARGE", "TEXT"] as LineType[]).map((t) => (
                  <Option key={t} data-key={t} selected={item.line_type === t}>{t.toLowerCase()}</Option>
                ))}
              </Select>
            </TableCell>
            {EDITABLE.map((column) => {
              const state = fieldState(detail, item.item_no, column.key.toUpperCase());
              const message = messageOf(state);
              const isMaterial = column.key === "material";
              const value = item[column.key];
              const isEmpty = value === null || value === undefined || value === "";
              // An unresolved material shows what was printed, so the reviewer sees the starting point.
              const printedHint = isMaterial && isEmpty && state?.raw_value ? `printed: ${state.raw_value}` : undefined;
              return (
                <TableCell key={column.key}>
                  <div className="review-cell" title={tooltipOf(state)}>
                    <Input
                      value={isEmpty ? "" : String(value)}
                      placeholder={printedHint}
                      readonly={readOnly}
                      valueState={valueStateOf(state)}
                      valueStateMessage={message ? <div>{message}</div> : undefined}
                      showSuggestions={isMaterial}
                      onInput={(e) => {
                        update(index, column.key, e.target.value);
                        if (isMaterial && e.target.value.length >= 2) lookUpMaterials(e.target.value);
                      }}
                      onFocus={() => {
                        // Pre-load the value help with the printed description when no material was found.
                        if (printedHint && !readOnly) lookUpMaterials(item.description || state?.raw_value || "");
                      }}
                      onChange={(e) => update(index, column.key, e.target.value)}
                    >
                      {isMaterial && materialHits.map((m) => <SuggestionItem key={m.text} text={m.text} additionalText={m.additionalText} />)}
                    </Input>
                    {state?.confidence != null && (
                      <span className={`review-confidence${state.confidence < 0.8 ? " review-confidence--low" : ""}`}>
                        {Math.round(state.confidence * 100)}%
                      </span>
                    )}
                  </div>
                </TableCell>
              );
            })}
            <TableCell>{item.plant ?? "-"}</TableCell>
            <TableCell>
              {!readOnly && (
                <Button design="Transparent" icon="delete" tooltip="Remove line"
                  onClick={() => setItems(items.filter((_, i) => i !== index))} />
              )}
            </TableCell>
          </TableRow>
        ))}
      </Table>
    </div>
  );
}
