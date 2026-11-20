# ECC integration: from the mock to a real SAP ECC

## What it is

This app runs against a **mock** of SAP ECC: HANA tables with ECC table and field names plus the
suffix `_POAGENT` (see [hana-data-model.md](hana-data-model.md)). The business logic (matching,
validation, posting payload, review) is real. This document describes how each mocked piece is
replaced in a customer landscape, which BAPIs / IDocs and connectivity are needed, and how every
payload field maps to ECC.

## Target landscape

```
 Microsoft 365                     SAP BTP (Cloud Foundry)                              On-premise
 ─────────────                     ─────────────────────────────────────────            ──────────────────
 Shared inbox ──► Email trigger ──► PO intake app (this repo)                            SAP ECC
                 (Power Automate,   - intake, gpt-6-luna agent, validation, review UI
                  MS Graph webhook  - POST /api/orders/upload  (same endpoint as the UI)
                  or CPI Mail       │
                  adapter)          ├─ master data ──► SAP Integration Suite (CPI) ──┐
                                    └─ post order  ──► SAP Integration Suite (CPI) ──┼─► Cloud Connector ─► RFC / IDoc
                                                                                      │   BAPI_SALESORDER_CREATEFROMDAT2
                                                                                      │   + BAPI_TRANSACTION_COMMIT
                                                                                      │   or IDoc ORDERS05 -> VBAK/VBAP
```

SAP Document AI is not used: the gpt-6-luna agent replaces the "Document AI + human review" step
and additionally resolves master data. The app cannot call ECC directly (no RFC from Python on
Cloud Foundry); SAP Integration Suite is the middle layer, as confirmed for the landscape.

## Swap points

| Landscape box | Mock in this repo | Production implementation | Code that changes |
|---|---|---|---|
| Inbox + e-mail trigger | Upload dialog | Power Automate flow "When a new email arrives" -> HTTP POST of the `.eml` (multipart field `files`) to `/api/orders/upload`; or a CPI iFlow with the Mail adapter doing the same | none |
| Master-data lookup | `api/app/ecc/master_data.py` reads `*_POAGENT` | Replicated master data in HANA (recommended for fuzzy search) or CPI iFlows calling BAPIs / `RFC_READ_TABLE` | function bodies of `ecc/master_data.py` |
| Post order | `api/app/ecc/sales_documents.create_sales_order` inserts `VBAK/VBKD/VBPA/VBAP_POAGENT` | HTTP POST of the same payload to a CPI iFlow (BTP Destination, OAuth2 client credentials) that calls the BAPI or sends ORDERS05 and returns `{vbeln, messages}` | body of `create_sales_order` |
| Order history | `list_sales_orders` reads `VBAK_POAGENT` | Keep a local record of posted documents (already in `POAGENT_ORDER_HEADER.SALES_DOC`) or read `VBAK` through CPI | `list_sales_orders` |
| "Corrections train model" | `POAGENT_CORRECTION_LOG` | unchanged | none |

`create_sales_order` already returns BAPIRET2-shaped messages (`type`, `id`, `number`, `message`);
`E`/`A` messages roll back and mark the order FAILED, so the production iFlow only has to forward
the BAPI `RETURN` table.

## Posting: payload -> `BAPI_SALESORDER_CREATEFROMDAT2`

The app builds this payload (`orders/posting.build_posting_payload`), following the requirements'
"Writeback API - required fields":

```json
{"orderId": "...", "orderDate": "20251218", "orderReference": "2802", "orderType": "TA",
 "soldToId": "762527", "shipToId": "762431", "salesOrgId": "SG10", "distributionChannel": "10",
 "division": "Z3", "customerPurchaseOrderType": "WWW", "requestedDeliveryDate": "20251222",
 "currency": "SGD", "paymentTerms": "N030", "incoterms": "DAP", "incotermsPlace": "SINGAPORE",
 "items": [{"lineItemNumber": 1, "sapMaterialNumber": "5620197", "quantity": 40, "unitOfMeasure": "PC",
            "plant": "SG01", "customerMaterialNumber": null, "tintedMaterialColourCode": null,
            "description": "CORESHIELD 200 GREEN 2.5L KIT"}]}
```

| Payload field | BAPI parameter / field | Notes |
|---|---|---|
| `orderType` | `ORDER_HEADER_IN-DOC_TYPE` | `TA` (requirements); confirm customer-specific order types |
| `salesOrgId` | `ORDER_HEADER_IN-SALES_ORG` | |
| `distributionChannel` | `ORDER_HEADER_IN-DISTR_CHAN` | `10` |
| `division` | `ORDER_HEADER_IN-DIVISION` | from the customer sales area |
| `orderReference` | `ORDER_HEADER_IN-PURCH_NO_C` | 35 characters, stored in `VBKD-BSTKD` (`VBAK-BSTNK` keeps 20) |
| `orderDate` | `ORDER_HEADER_IN-PURCH_DATE` | `YYYYMMDD` |
| `customerPurchaseOrderType` | `ORDER_HEADER_IN-PO_METHOD` | `WWW` |
| `requestedDeliveryDate` | `ORDER_HEADER_IN-REQ_DATE_H` and `ORDER_SCHEDULES_IN-REQ_DATE` | |
| `paymentTerms` | `ORDER_HEADER_IN-PMNTTRMS` | usually defaulted by ECC from `KNVV`; may be omitted |
| `incoterms`, `incotermsPlace` | `ORDER_HEADER_IN-INCOTERMS1`, `INCOTERMS2` | |
| `currency` | `ORDER_HEADER_IN-CURRENCY` | |
| `soldToId` | `ORDER_PARTNERS` row `PARTN_ROLE = 'AG'`, `PARTN_NUMB` | ALPHA format (`0000762527`) |
| `shipToId` | `ORDER_PARTNERS` row `PARTN_ROLE = 'WE'`, `PARTN_NUMB` | |
| `lineItemNumber` | `ORDER_ITEMS_IN-ITM_NUMBER`, `ORDER_SCHEDULES_IN-ITM_NUMBER` | `000001` (lines numbered 1, 2, 3 in document order, as in the writeback specification; ECC's own default interval would be 10) |
| `sapMaterialNumber` | `ORDER_ITEMS_IN-MATERIAL` (`MATERIAL_LONG` from EhP7 when 40-character numbers are active) | ALPHA format (18) |
| `quantity` | `ORDER_SCHEDULES_IN-REQ_QTY` (and `ORDER_ITEMS_IN-TARGET_QTY`) | in the sales unit after conversion |
| `unitOfMeasure` | `ORDER_ITEMS_IN-TARGET_QU` / `SALES_UNIT` | SAP internal unit |
| `plant` | `ORDER_ITEMS_IN-PLANT` | required for composite sets (requirements) |
| `customerMaterialNumber` | `ORDER_ITEMS_IN-CUST_MAT35` | |
| `description` | `ORDER_ITEMS_IN-SHORT_TEXT` | normally determined by ECC |
| `tintedMaterialColourCode` | `EXTENSIONIN` (`BAPE_VBAP` / `BAPE_VBAPX`) -> append field `VBAP-ZZCOLOR` | open point of the requirements (RAL mapping) |

Sequence in the iFlow: optionally `BAPI_SALESORDER_SIMULATE` (pricing, availability, incompletion)
-> `BAPI_SALESORDER_CREATEFROMDAT2` -> if no `E`/`A` in `RETURN`: `BAPI_TRANSACTION_COMMIT`
(`WAIT = 'X'`), else `BAPI_TRANSACTION_ROLLBACK`. Return `SALESDOCUMENT` as `vbeln` and the
`RETURN` table as `messages`. Charge lines (freight) are not transferred as items; map them to
pricing conditions (`ORDER_CONDITIONS_IN`, e.g. a freight condition type) if required.

## Posting alternative: IDoc `ORDERS05`

Message type `ORDERS`, basic type `ORDERS05`, inbound process code `ORDE`, partner profile of type
`KU` per customer (or a generic logical system with CPI's IDoc adapter):

| Segment | Qualifier / field | Value |
|---|---|---|
| `E1EDK01` | `CURCY`, `BSART` | currency; order type via `E1EDK14` |
| `E1EDK14` | `QUALF 008` / `006` / `007` / `012` | sales org / division / distribution channel / order type |
| `E1EDK03` | `IDDAT 022` / `002` | PO date / requested delivery date |
| `E1EDK02` | `QUALF 001`, `BELNR` | customer PO number |
| `E1EDKA1` | `PARVW AG` / `WE`, `PARTN` | sold-to / ship-to |
| `E1EDP01` | `POSEX`, `MENGE`, `MENEE`, `WERKS` | item number, quantity, unit, plant |
| `E1EDP19` | `QUALF 002` / `001`, `IDTNR` | material number / customer material number |
| `E1EDP20` | `WMENG`, `EDATU` | schedule line |

Use the IDoc route when the customer already runs EDI order intake; the BAPI route gives a
synchronous answer (document number or errors) for the UI.

## Master data: mock table -> ECC source

| Mock table | ECC source | Read option |
|---|---|---|
| `KNA1_POAGENT` | `KNA1` (+ `ADRC`) | `BAPI_CUSTOMER_GETDETAIL2`, `RFC_READ_TABLE`, or replication |
| `KNVV_POAGENT` | `KNVV` | `BAPI_CUSTOMER_GETSALESAREAS`, replication |
| `KNVP_POAGENT` | `KNVP` | `RFC_READ_TABLE`, replication |
| `KNVK_POAGENT` | `KNVK` + `ADR6` (e-mail) | replication |
| `MARA_POAGENT`, `MAKT_POAGENT`, `MARM_POAGENT`, `MVKE_POAGENT` | `MARA`, `MAKT`, `MARM`, `MVKE` | `BAPI_MATERIAL_GET_ALL`, replication |
| `KNMT_POAGENT` | `KNMT` (customer-material info) | replication |
| `ZUOM_ALIAS_POAGENT` | custom Z-table (keep in HANA) | - |

Recommendation: replicate these tables into HANA (SAP Landscape Transformation / CPI scheduled
iFlow / Datasphere replication flow) and keep `ecc/master_data.py` on SQL. Fuzzy customer and
material search over live RFC calls would be slow; with replicated data the candidate retrieval
can move into HANA (`CONTAINS(..., FUZZY(0.8))`) and the Python scoring stays as is.
`RFC_READ_TABLE` is limited to 512 bytes per row and should only be used for small lookups.

## Connectivity

| Hop | Technology |
|---|---|
| App -> CPI | BTP Destination of type HTTP, authentication `OAuth2ClientCredentials` (CPI process integration runtime service key); the app reads it through the Destination service |
| CPI -> ECC | RFC receiver adapter (BAPI) or IDoc receiver adapter, Cloud Connector with a location ID, technical user with authorisations for `V_VBAK_AAT`/`V_VBAK_VKO`, `S_RFC` on the BAPI function groups |
| Inbox -> app | Power Automate HTTP action / MS Graph change notification / CPI Mail adapter; authenticate with the app's API key today, XSUAA client credentials in production |

Alternative without CPI: a Gateway OData service in ECC (SEGW) wrapping the BAPI, reached through
Cloud Connector and the Connectivity proxy from the app. `API_SALES_ORDER_SRV` exists only in
S/4HANA, not in ECC.

## Production checklist

* Replace the shared API key by XSUAA (user identity also lets the correction log name reviewers).
* Idempotency at intake: reject a second upload of the same e-mail `Message-ID`.
* Confirm order type / PO type mapping per sales org and the RAL / tint colour mapping.
* Map BAPI `RETURN` messages to `NEEDS_REVIEW` (data issues the reviewer can fix) vs `FAILED`.
* Run `BAPI_SALESORDER_SIMULATE` in the gate to catch pricing, credit and availability issues.
* Retention for stored e-mails, attachments and agent traces.

## How to test

The mock path is covered end to end (see [order-processing-pipeline.md](order-processing-pipeline.md)).
For the production path, test the CPI iFlow with the payload above (Postman / Bruno against the
iFlow endpoint), then point `create_sales_order` at it and run
`python -m app.cli.run_samples <files> --post`.
