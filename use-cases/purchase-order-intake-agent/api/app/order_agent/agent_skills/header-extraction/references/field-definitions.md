# Header field definitions

| Field | Definition |
|---|---|
| po_number | The buyer's own order identifier, in any language ("Order No.", "P.O.", "Số đơn hàng", "Bestellung"). On a supplier pro forma / confirmation use the customer's PO reference. |
| po_date | Date the order was issued (YYYY-MM-DD). Not delivery, print, e-mail or template dates. |
| requested_delivery_date | Requested / required delivery date for the whole order, if printed. |
| sold_to | `value` = customer number found with search_customers; `raw` = buyer name as printed. |
| sold_to_name | Buyer company name as printed (from text or letterhead image). |
| sold_to_address | Buyer postal address in one line. Not the supplier's address, not the delivery address unless it is the same block. |
| buyer_tax_id | VAT / GST / TIN / company registration number of the buyer, as printed. |
| buyer_phone | Phone number printed in the buyer block (on some order forms it is the last line of the buyer address). |
| ship_to | `value` = customer number of the chosen ship-to partner (function ship-to of the sold-to). |
| ship_to_name | Company receiving the goods, only when explicitly stated. |
| ship_to_address | Delivery address, only when explicitly stated (document or e-mail body). |
| sales_org | Only if the inbox mapping gives one or the customer profile makes it unambiguous; otherwise omit (the system derives it). |
| currency | ISO 4217 code of the order amounts. |
| net_amount | Order total before tax, final total of all pages, after line discounts. |
| gross_amount | Final amount payable including tax and charges. |
| payment_terms_text | Payment terms as written ("30 days", "Advance", "100% after 45 days"); Vietnamese translated to English, other languages as printed. |
| incoterms | Incoterms code (EXW, FCA, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP) when stated. "Ocean" or "by air" alone is a shipping mode, not an Incoterm. |
| incoterms_place | Named place of the Incoterm. |
| header_note | Delivery instructions, receiver name/phone, project or vessel names, discount remarks worth keeping for the reviewer - quoted as printed (Vietnamese translated to English). |
