# UI: purchase-order intake and review

## What it is

The React + UI5 Web Components screens of the purchase-order intake app, modelled on the
"Track Sales Orders" mock-up:

1. **Purchase Order Intake** (`/orders`): filter bar, *Incoming Purchase Orders* table with the
   processing status of every inbound message, *Upload* dialog, and *Order History* (sales
   documents posted to the mock ECC; replaces the mock-up's "Sales Orders" section).
2. **Review page** (`/orders/:orderId`): documents on the left, extracted / validated values on the
   right, Save / Confirm / Reprocess (see part 2 below).

## What it is for

Users drop customer e-mails or documents, the agent processes them in the background, orders that
pass the validation gate are posted automatically (green *Posted*), the rest wait for a human
(orange *Needs Review*).

## Part 1 - orders list, upload, history

### Inputs and outputs

| Element | Behaviour |
|---|---|
| Upload dialog | Drop zone + *Browse* (`.eml`, PDF, images, `.docx`, `.xlsx`, `.txt`, `.csv`), optional pasted e-mail text, optional sales organisation (simulates the inbox mapping); *Extract* calls `POST /api/orders/upload`. Every `.eml` becomes its own order; other files are processed together (first file = purchase order, others = supporting). |
| Status pills | `Processing` (blue, busy indicator; QUEUED/PROCESSING), `Needs Review` (orange, with the number of blocking issues), `Posted` (green), `Failed` (red), `Not an Order` (grey). The status message is the tooltip. |
| Filters | Search (PO number, customer, sender, subject, sales order) and status selection (*Open Documents* = everything not posted and not "not an order"); counters per status. |
| Polling | Every 3 s while any order is processing and the browser tab is visible; data-driven, so a manual refresh after an upload restarts it. Order History reloads when an order leaves processing. |
| Row click / Review | Opens the review page. |
| Delete (bin icon) | **PoC only**: after a browser confirmation, deletes the order (and its mock sales order, if posted) via `DELETE /api/orders/{id}` so the same document can be uploaded again without a duplicate warning. Disabled while the agent processes the order. |

### Related files

| File | Role |
|---|---|
| `ui/src/pages/orders/OrdersPage.tsx` | Page, filters, polling, toast |
| `ui/src/pages/orders/IncomingOrdersTable.tsx` | Incoming purchase orders (UI5 `Table`) |
| `ui/src/pages/orders/OrderHistoryTable.tsx` | Posted sales documents |
| `ui/src/pages/orders/UploadOrderDialog.tsx` | Upload dialog |
| `ui/src/pages/orders/orderStatus.ts` | Status labels / Tag designs, formatters |
| `ui/src/pages/orders/usePolling.ts` | Data-driven polling hook |
| `ui/src/services/orders.ts` | Typed API calls (mirror `api/app/models/orders.py`) |
| `ui/src/services/api.ts` | `request`, `requestForm`, `requestBlob`, `ApiError` |
| `ui/src/routes.tsx` | `/orders` in the side navigation, `/orders/:orderId` in `detailRoutes` |

### Key code

```tsx
// Polling is data-driven: each completed load schedules the next one while orders are processing.
const orders = usePolling(listOrders, (items) => items.some((o) => isProcessing(o.status)));
```

```ts
export const STATUS_DISPLAY = {
  QUEUED: { label: "Processing", design: "Information" },
  NEEDS_REVIEW: { label: "Needs Review", design: "Critical" },
  POSTED: { label: "Posted", design: "Positive" }, ...
};
```

### How to test

Start the `api` and `ui` launch configurations (`.claude/launch.json`) or:

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir api --port 8000
npm --prefix ui run dev -- --port 5173
```

Open `http://localhost:5173/orders`, click *Upload*, paste an order e-mail (or drop a PDF / `.eml`),
click *Extract*: the row appears as *Processing* and turns *Posted* / *Needs Review* /
*Not an Order* without reloading; posted orders appear in *Order History*.

Verified: a pasted order e-mail was posted automatically within 30 s (sales order 9100000006) and a
price-list request was classified *Not an Order*, both without a manual refresh.

## Part 2 - review page (`/orders/:orderId`)

### Layout

| Area | Content |
|---|---|
| Title | PO number, status pill, customer, received date and channel, model calls / input tokens of the last agent run |
| Left: documents | `Email` tab (headers, plain-text body - HTML is never rendered - and a gallery of the e-mail's inline images tagged *sent to agent* / *not sent to agent* with the reason), then one tab per document: PDF in an `<iframe>` (fit to width), images, Word / Excel / forwarded e-mails as text |
| Right: issues | Strip "N values need attention" (or Processing / Posted / Failed / Not an order / all checks passed), list of ERROR / REVIEW / WARNING points (field, item, agent flags), the agent's summary, expandable list of values filled from master data, and - after a rejected Confirm - a red strip with the blocking reasons |
| Right: header form | Document (PO number, PO date, requested delivery, currency, sales org, classification), Sold-To (customer number with value help, name and address as printed), Ship-To (select limited to the customer's `WE` partners, delivery name/address as printed), Amounts and Terms (+ read-only ECC values: order type, channel, division, PO type, payment terms) |
| Right: line items | Editable table: type, material (value help: material number, sales code, customer code, description), description, quantity, unit, price, amount, RAL / tint code, customer material; plant (system); add / remove lines. Lines are numbered 1, 2, 3. An unresolved material shows the printed code as a grey hint (`printed: ...`) and focusing it pre-loads the value help with the printed description. |
| Footer | **Confirm** (save edits + post; blocked with reasons on ERROR), **Save**, **Reprocess** (dialog with an instruction for the agent), **Cancel** |

Every field shows its confidence (orange below 80 %) and a value state: red = ERROR, orange =
REVIEW / WARNING, blue = value filled by the system from master data. The tooltip shows the printed
value, its source (e.g. `A2 p.1`), the agent's note (or the system fill), for materials the agent's
evidence (*Matched*, *Conflicts*, *Alternatives*) and the customer price check, the agent and
master-data text-match confidences, and who set it: *Extracted by: agent* when the agent only read
the value, *Changed by: agent / system / user* otherwise. An empty material shows 0 %. Fields a reviewer edits become *user* fields (confidence 100 %, no longer flagged).

### Related files

| File | Role |
|---|---|
| `ui/src/pages/order-review/OrderReviewPage.tsx` | Page, draft state, Save / Confirm / Reprocess, polling while processing |
| `ui/src/pages/order-review/DocumentViewer.tsx` | E-mail and document tabs, object URLs for files |
| `ui/src/pages/order-review/IssuesPanel.tsx` | Status strip, issues, agent summary, system fills |
| `ui/src/pages/order-review/HeaderForm.tsx` | Header sections, ship-to partner select, customer value help |
| `ui/src/pages/order-review/LineItemsTable.tsx` | Editable lines with material value help |
| `ui/src/pages/order-review/FieldInput.tsx`, `fieldState.ts` | Field rendering, value state / tooltip mapping |

### Key code

```ts
// Only changed header fields and (if lines changed) the full line list are sent.
const header = Object.fromEntries(HEADER_KEYS.filter((k) => String(draft[k] ?? "") !== String(original[k] ?? ""))
  .map((k) => [k, draft[k]]));
```

```ts
// A 422 from confirm carries the ERROR issues that block posting.
if (err instanceof ApiError && err.status === 422) setConfirmIssues(err.detail.issues);
```

### How to test

Open an order in *Needs Review*: the documents render on the left; highlighted fields match the
issue list. Click *Confirm* while an ERROR exists: a red strip lists the blocking reasons. Choose a
ship-to / material, *Save*: the related error disappears and the field shows 100 %. When no ERROR is
left, *Confirm* posts and the page switches to *Posted* with the sales order number.

Verified in the browser: an order with a ship-to conflict between e-mail body and PDF and two
products missing from master data showed 11 points; Confirm was blocked with the reasons (422);
selecting the ship-to partner and saving removed the ship-to error (10 points left).

Performance note: Save / Confirm re-validate against current master data (about 20 HANA queries);
measured 5-6 s from a laptop to HANA Cloud after batching the row inserts, much less when the app
runs in the same region.
