---
name: header-extraction
description: Extract purchase-order header fields correctly (PO number and date, buyer, ship-to, amounts, terms) and avoid the common traps.
---

# Header extraction

Read `references/field-definitions.md` for the definition of every header field. Report each field
as a FieldValue: `value` per the language rule of your instructions (free text as printed,
Vietnamese translated to English; dates, numbers, currency and Incoterms normalised), `raw`
exactly as printed, `confidence`, `source`, and `note` for any interpretation you applied. Omit
fields that are not on the document.

## Traps seen on real orders
- **PO number**: the buyer's own order identifier. Not the supplier's order/confirmation/invoice
  number, not a contract, project, job, quote, form-template or document-control code. On a
  supplier pro forma or confirmation take the "customer order reference". If the only identifier is
  the buyer system's order number (e.g. "Bestellung 123456") use it.
- **PO date**: the date the order was issued. Not the delivery date, a print date, the e-mail sent
  date or a form-template issue date. If there is none, omit it (the system uses the received date).
- **Buyer (sold-to)**: the company placing and paying for the order. Not the supplier the order is
  addressed to, not a contact person. The buyer may appear only in a logo or letterhead image -
  look at the native attachments. Put VAT/GST/registration numbers in `buyer_tax_id` and the phone
  number of the buyer block in `buyer_phone`; they are never the customer number.
- **Ship-to**: only when a delivery party or address is explicitly stated ("ship to", "deliver
  to", "nơi giao hàng", "địa chỉ giao hàng", "Lieferadresse"). Never the supplier. A bare place name
  ("BÌNH DƯƠNG") is a location, not a company: put it in `ship_to_address`, not `ship_to_name`.
- **E-mail body vs attachment**: when the body gives a delivery address or instruction that
  differs from the attachment, use the body (it is the latest instruction), keep both in `note`,
  and add a review flag "ship_to: e-mail body and order document differ".
- **Totals**: `net_amount` is the final total before tax of the whole order - on multi-page orders
  the grand total, not a page subtotal; after line discounts. If a discount is applied to the
  order, say so in `header_note` and add a review flag.
- **Numbers**: documents use local formats - "1.234,56" (comma decimals), "1,234.56", "3.093.784.540"
  (dot thousands, no decimals). Convert to plain numbers.
- **Currency**: from a code, symbol or currency word next to the amounts (EUR for €, VND for đ /
  "VNĐ", SGD for "S$"). Omit it if no amount shows a currency.
- **Language**: translate only Vietnamese free text to English in `value` (original in `raw`).
  German and other languages stay as printed. Company names and addresses are never translated -
  normalise only spacing.
