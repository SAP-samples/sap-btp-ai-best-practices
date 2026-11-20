# Order processing pipeline, validation gate, posting and review actions

## What it is

Everything between "an order was uploaded" and "a sales order exists in ECC":

1. **Workers** (`orders/pipeline.py`) pick queued orders from HANA and run the order agent,
   up to `ORDER_WORKERS` in parallel per app instance.
2. **Final gate** (`orders/service.finalize_after_agent`) re-validates the stored order against
   current master data with the deterministic rules (`orders/validation.py`).
3. **Posting** (`orders/posting.py`, `ecc/sales_documents.py`) creates the sales document in the
   mock ECC tables when the gate passes and auto-posting is enabled.
4. **Review actions** (Save, Confirm, Reprocess) let a human fix and post held orders.

## What it is for

The requirement is a mostly automatic process: orders that are extracted, (possibly) corrected
and complete are posted without a human; anything the agent cannot settle is held in
`NEEDS_REVIEW` with precise, field-level reasons.

## Statuses

| Status | Meaning | Set by |
|---|---|---|
| `QUEUED` | waiting for a worker | upload, reprocess, retry, restart recovery |
| `PROCESSING` | agent running (or posting) | worker claim |
| `POSTED` | sales document created (`SALES_DOC`) | gate (auto) or Confirm |
| `NEEDS_REVIEW` | blocking issues, or ready but auto-post disabled | gate, Save |
| `NOT_AN_ORDER` | message is not a purchase order | gate |
| `FAILED` | agent error/timeout after retries, or ECC rejected the order | worker, posting |

## The gate (strict, as agreed)

An order is posted automatically only when **all** hold:

* classification `STANDARD` (contract-based orders always go to review);
* no `ERROR`: PO number, sold-to (exists, not blocked, sold-to account), sales area, ship-to (a
  `WE` partner; single or default partner used automatically, several without a matching address
  need a human), at least one material line, each material exists / is not deleted / is sold in
  the sales org and not sales-blocked, valid unit, positive whole quantity for pieces, plant
  derivable (customer-material record, ship-to delivering plant, material sales data);
* no `REVIEW`: confidence below 0.80 on a posting-relevant field (PO number/date, sold-to, ship-to,
  sales org, currency, material, quantity, unit, colour), ambiguous master-data match, currency or
  incoterms differing from the customer master, line amounts not adding up to the net amount (1 %),
  duplicate customer PO, charge lines, several orders in one message, any agent review flag.

`WARNING` (future PO date, PO number longer than 20 characters, low confidence on non-posting
fields) and `INFO` (system fills: PO date from received date, currency / terms / incoterms from the
customer master, ship-to, plant, division, order type `TA`, channel `10`, PO type `WWW`) never block.

## Posting

`build_posting_payload` produces the requirements' writeback payload; only `MATERIAL` lines are
transferred (charges must become pricing conditions). `post_order` runs in one transaction:
`SELECT ... FOR UPDATE` on the order, status guard (a second call returns 409), `create_sales_order`
(mock: `VBAK`/`VBKD`/`VBPA`(AG, WE)/`VBAP` rows numbered from `VBELN_SEQ_POAGENT`), then
`STATUS = POSTED`, `SALES_DOC`, `POSTING_PAYLOAD`. ECC answers in BAPIRET2 form; `E`/`A`
messages roll the document back and the order becomes `FAILED` with the message.

```json
{"orderReference": "2802", "orderDate": "20251218", "orderType": "TA", "soldToId": "762527",
 "shipToId": "762431", "salesOrgId": "SG10", "distributionChannel": "10", "division": "Z3",
 "customerPurchaseOrderType": "WWW",
 "items": [{"lineItemNumber": 1, "sapMaterialNumber": "5620197", "quantity": 40, "unitOfMeasure": "PC", "plant": "SG01"}]}
```

Item numbers: order lines are numbered 1, 2, 3 in document order (`ITEM_NO`, sent to ECC as
`lineItemNumber` / `ITM_NUMBER` `000001`, ...), as in the writeback API specification. Orders
created before this change keep 10, 20, ... .

## Review actions (API)

| Endpoint | Behaviour |
|---|---|
| `PUT /api/orders/{id}` | Body `{"header": {"ship_to": "762431"}, "items": [{"item_no": 1, "material": "5620197"}, ...], "classification": "STANDARD"}`. Changed fields become `CHANGED_BY = USER` (confidence 1.0, never flagged for confidence), the order is re-validated and saved; returns the order detail. Allowed in `NEEDS_REVIEW`, `FAILED`, `NOT_AN_ORDER`. |
| `POST /api/orders/{id}/confirm` | Optional body = Save first. Re-validates; any `ERROR` returns **422** with `{"message", "issues": [...]}`; `REVIEW` issues are accepted (a human looked at them); posts and returns `{"status": "POSTED", "sales_doc"}`. Second confirm: **409**. |
| `POST /api/orders/{id}/reprocess` | Body `{"note": "..."}`. Requeues the order (202); the note is given to the agent. |
| `DELETE /api/orders/{id}` | **PoC only** (`ORDER_DELETE_ENABLED`, default `true`; set `false` in production). Deletes the order, its lines, field states, agent runs, reviewer corrections made on it, the mock ECC sales document if it was posted, and the inbound message/attachments when no other order uses them, so the same document can be uploaded again without a duplicate warning. **403** when disabled, **404** unknown order, **409** while the agent is processing it. Returns `{"order_id", "deleted": true}`. |
| `GET /api/sales-orders` | Order history from the mock ECC tables. |
| `GET /api/master-data/...` | Value help: `sales-orgs`, `customers?q=`, `customers/{no}/partners`, `materials?q=&sales_org=&sold_to=`. |

## Concurrency and recovery

* **Queue = table**: a worker claims with `UPDATE ... SET STATUS = 'PROCESSING' WHERE ORDER_ID = ? AND
  STATUS = 'QUEUED'`; only the worker whose update affects one row owns the order (safe across
  workers and app instances).
* The upload endpoint wakes idle workers (`app.state.order_queue_event`); idle workers poll every 5 s.
* Restart: an instance requeues the orders it left in `PROCESSING` (`LOCKED_BY` = CF instance
  index or host name). Another instance's orphans are requeued by the sweeper after
  `ORDER_AGENT_TIMEOUT_SECONDS + 120 s`, and marked `FAILED` after `ORDER_MAX_ATTEMPTS`.
* Agent errors and timeouts are retried up to `ORDER_MAX_ATTEMPTS`.

## Settings (environment)

| Variable | Default | |
|---|---|---|
| `ORDER_PIPELINE_ENABLED` | `true` | `false` disables workers (e.g. a read-only instance) |
| `ORDER_WORKERS` | 4 | parallel agent runs per instance |
| `ORDER_AGENT_TIMEOUT_SECONDS` | 900 | high reasoning effort with many lines can take several minutes |
| `ORDER_MAX_ATTEMPTS` | 2 | |
| `AUTO_POST_ENABLED` | `true` | `false` keeps every order for human confirmation |
| `CONFIDENCE_THRESHOLD` | 0.8 | requirement R4 |
| `DEFAULT_ORDER_TYPE` / `DEFAULT_DISTR_CHANNEL` / `DEFAULT_PO_TYPE` | TA / 10 / WWW | requirements, to be confirmed by the business |

## Related files

`api/app/orders/{pipeline,service,validation,posting,repository}.py`,
`api/app/ecc/sales_documents.py`, `api/app/routers/{orders,master_data}.py`,
`api/app/models/{orders,master_data}.py`, `api/app/main.py` (lifespan).

## How to test

```bash
cd api
../.venv/bin/python -m unittest discover -s ../tests/unit -t ..
# test_validation, test_posting_payload, test_pipeline, test_orders_router
../.venv/bin/python -m app.cli.run_samples ../data/orders --post   # end to end incl. posting
```

Verified live: 6 uploads at once -> at most 4 orders `PROCESSING`; 4 orders auto-posted
(9100000001-4) with correct `VBAK`/`VBKD`/`VBPA`/`VBAP` rows; freight-line order held for review,
then Confirm returned 422 while a material was cleared, posted after restoring it, and 409 on a second
confirm; API killed while an order was processing: on restart the order was requeued and finished.
