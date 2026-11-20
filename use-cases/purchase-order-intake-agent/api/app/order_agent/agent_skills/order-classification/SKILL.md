---
name: order-classification
description: Decide whether an inbound message is a standard purchase order, a contract-based order, or not an order at all, and which file holds the order.
---

# Order classification

Classify every inbound message before extracting anything. Put the evidence in `classification_reason`.

## STANDARD
The buyer issues (or confirms) a concrete order to the supplier: an order number or reference,
goods with quantities, usually a date. All of these count:
- a purchase order PDF / scan / photo, an Excel or Word order form;
- an order typed in the e-mail body or pasted as an image (screenshot of a table) into the body;
- a **supplier document that wraps the customer's order** (order confirmation, pro forma
  invoice): it is still an order; the customer's PO number is the "customer order reference",
  not the supplier's own order or invoice number.

## CONTRACT
A call-off or release against a framework agreement. Signals: a contract / agreement / framework
number printed next to the order number, words like "contract", "agreement", "hợp đồng",
"Rahmenvertrag", "call-off", "release", promotional programmes agreed for a period
("buy 10 get 1", free-of-charge quantities granted over a period), quantities "to be picked up in
batches" or "schedule to follow". Contract orders are never posted automatically.

## NOT_AN_ORDER
Nothing should be ordered from this message. Typical cases:
- internal approval threads (discount, free-of-charge or price approvals), even if they mention
  an order or carry a forwarded e-mail;
- price lists, quotations, catalogues, product data sheets alone;
- delivery, invoice or payment queries, complaints, newsletters, auto-replies.
Submit with `classification = NOT_AN_ORDER`, empty `lines`, a clear reason, and stop.

## Several orders
If one message contains more than one distinct customer PO number with its own lines, set
`multiple_orders_detected = true`, extract the first (or main) order, and add a review flag naming
the others. Attachments that only support the order (price lists, terms, previous correspondence)
are not separate orders.

## Which file is the order
Use the attachment manifest labels ([A1], [A2], ...). Prefer, in this order: the customer's own
order document, a supplier confirmation that quotes the customer's order, the e-mail body or a
pasted image. Cite the label in each field's `source` (for example "A2 p.1").
