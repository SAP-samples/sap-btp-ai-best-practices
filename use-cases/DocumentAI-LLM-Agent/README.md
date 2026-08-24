# AI4U — Document AI Agent

> **SAP Document AI + SAP GenAI Hub — Intelligent Document Processing & S/4HANA Integration**

A full-stack application that combines **SAP Document Information Extraction (Document AI)** with **SAP GenAI Hub** (multimodal LLMs) to extract, validate, and post structured data from business documents — invoices, customer Purchase Orders, and Payment Advices — directly into **SAP S/4HANA** (FI, MM, and SD modules).

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture Overview](#architecture-overview)
- [Document Processing Pipeline](#document-processing-pipeline)
- [S/4HANA Integration](#s4hana-integration)
- [Pipeline V1 — Intelligent Invoice Routing](#pipeline-v1--intelligent-invoice-routing)
- [Pipeline V2 — DocAI NEW (Auto-Template)](#pipeline-v2--docai-new-auto-template)
- [Sales Order Process Pipeline](#sales-order-process-pipeline)
- [Payment Advice Pipeline](#payment-advice-pipeline)
- [3-Panel UX & Chat Assistant](#3-panel-ux--chat-assistant)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [API Reference](#api-reference)
- [Demo Files](#demo-files)
- [Deployment](#deployment)

---

## What It Does

The AI4U Document AI Agent automates the full document-to-posting workflow:

1. **Upload any business document** (invoice, customer PO, or payment advice) — the system automatically detects the document type
2. **Extract structured data** via SAP Document AI (with the correct schema per document type) + LLM enrichment where needed
3. **Validate against S/4HANA master data** (customers, materials, vendors)
4. **Post to the right SAP module** with a single click:
   - **Supplier Invoice (no PO)** → FI direct GL account posting (`API_SUPPLIERINVOICE_PROCESS_SRV`)
   - **Supplier Invoice (with PO reference)** → MM MIRO-equivalent posting via PO line item reference
   - **Customer Purchase Order** → SD Sales Order creation (`API_SALES_ORDER_SRV`)
   - **Payment Advice** → FI Payment Advice creation with line items (`API_PAYMENT_ADVICE_SRV`)
5. **Chat with the DocAI Assistant** to ask questions about extraction results, schemas, or posting status

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AI4U Document AI Agent                          │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │  V1 Pipeline     │  │  V2 DocAI NEW    │  │  Sales Order Process   │ │
│  │  /api/v1/...     │  │  /api/v2/...     │  │  /api/v1/so/...        │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘ │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    S/4HANA Integration Layer                      │   │
│  │  FI (GL posting) │ MM (MIRO/PO) │ SD (Sales Order) │ BP Search  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Streaming Chat Assistant                        │   │
│  │           /api/v1/chat/message (NDJSON streaming)                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Document Processing Pipeline

The main pipeline (`POST /api/v1/genai/pipeline`) is **fully transparent** — the user uploads any document and the system routes automatically:

```
Upload any document (PDF / JPG / PNG / TIFF)
              │
              ▼
┌─────────────────────────────────┐
│  Step 0: Document Classification │  ← LLM multimodal + PDF keyword scan
│  "invoice" or "purchase_order"? │     (fast, no SAP call needed)
└──────────────────┬──────────────┘
                   │
       ┌───────────┴──────────┐
       ▼                      ▼
  INVOICE                PURCHASE ORDER
       │                      │
       ▼                      ▼
SAP DocAI                SAP DocAI
SAP_invoice_schema       SAP_purchaseOrder_schema
+ LLM Technique 1        + LLM enrichment
+ LLM Technique 2        + S/4HANA validation
+ Comparison             (customer BP, materials)
       │                      │
       ▼                      ▼
  POST S4 button          POST S4 button
  (smart routing)         → Create Sales Order
       │
  ┌────┴────┐
  ▼         ▼
Has PO?   No PO
  │         │
  ▼         ▼
MM (MIRO)  FI (GL) or
           manual PO lookup
```

### Document Type Detection

Detection runs **before** SAP Document AI to ensure the correct schema is used:

1. **Fast keyword scan** of the PDF text layer (deterministic, zero cost):
   - "Payment Advice" / "Zahlungsavis" / "Remittance" → `payment_advice` *(checked first — most specific)*
   - "Purchase Order" / "Bestellung" / "Bestellnummer" → `purchase_order`
   - "Invoice Number" / "Rechnung" / "Factura" → `invoice`
2. **LLM fallback** for scanned/image-only PDFs where text extraction is not possible

---

## S/4HANA Integration

### POST S4 — Smart Routing

The **POST S4** button automatically routes to the correct posting based on document type:

| Document Type | Action | SAP API |
|---|---|---|
| Invoice — no PO | Opens PO search modal → GL posting or manual PO selection | `A_SupplierInvoice` (GL node) |
| Invoice — PO detected | Direct MIRO-equivalent posting | `A_SupplierInvoice` (PO ref node) |
| Customer PO | Validation modal → Create Sales Order | `A_SalesOrder` |
| Payment Advice | Auto-matches payer BP → Creates Payment Advice with line items | `A_PaymentAdvice` |

### Supplier Invoice — GL Account Posting (FI)

```
POST /api/v1/fi/post-invoice
```
Posts to FI using `to_SupplierInvoiceItemGLAcct` node. Requires:
- Supplier name (auto-matched to SAP Business Partner)
- Invoice number, date, gross amount, currency
- GL Account: configured via `FI_EXPENSE_GL_ACCOUNT` (default `11001000`)
- Company Code: configured via `FI_COMPANY_CODE` (default `1010`)

### Supplier Invoice — PO-Based Posting (MM / MIRO)

```
POST /api/v1/fi/post-po-invoice
```
Replicates MIRO using `to_SuplrInvcItemPurOrdRef` node deep-create. Requires:
- Purchase Order number (auto-detected from document)
- PO item number (default `00010`)
- Tax code: configured via `FI_PO_TAX_CODE` (default `V0`)

> **Note:** Requires a Goods Receipt (GR) to be posted first in MIGO for GR-Based IV purchase orders.

### Customer PO — Sales Order Creation (SD)

```
POST /api/v1/so/create
```
Creates a Sales Order with multi-item support, header text, and full organizational data. Requires:
- Customer Business Partner (auto-matched from PO buyer name)
- SAP material codes (auto-matched from item descriptions)
- Sales Organization, Distribution Channel, Division

### No-PO Modal — Manual Vendor Search

When no PO is detected, the system opens a modal that:
1. Pre-fills the vendor name from the extracted document
2. Searches S/4HANA Business Partners using `substringof()` OData filter
3. Fetches open Purchase Orders for the selected vendor
4. Lets the user select a PO and item, or proceed with GL-only posting

---

## Pipeline V1 — Intelligent Invoice Routing

The V1 pipeline processes a single invoice and decides between SAP templates and LLM extraction:

```
Invoice → SAP DocAI (generic) → Routing Engine
                                      │
                           ┌──────────┴──────────┐
                           ▼                     ▼
                     Template found         No template
                           │                     │
                           ▼                     ▼
                     SAP Template          GenAI Hub LLM
                     (specialized)    ├── Technique 1: Free Prompting
                                      └── Technique 2: Structured JSON
                                               │
                                               ▼
                                      Field-by-field Comparison
                                               │
                                               ▼
                                      Evaluation & Scoring
```

**Routing Engine:**
- Detects supplier name from SAP extraction result
- Fetches all SAP Document AI templates
- Fuzzy-matches supplier name vs template names (rapidfuzz)
- Routes to template (≥75% confidence) or GenAI fallback

**Evaluation scores each method on:**
- Completeness (35%), Confidence avg (30%), Consistency (20%), Field coverage (15%)

---

## Pipeline V2 — DocAI NEW (Auto-Template)

LLM-first pipeline that automatically manages SAP templates:

```
PDF → LLM Free Extraction → Customer Detection → Template Discovery
                                                         │
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                        Template exists       Not found
                                              │                     │
                                              ▼                     ▼
                                        Return result         Auto-create
                                                              + Annotate
                                                              + Activate
                                                              + Associate
```

Supports batch processing of multiple PDFs in a single request.

---

## Sales Order Process Pipeline

Handles **Customer Purchase Orders** (buyer ordering from AI4U):

```
Customer PO (PDF)
      │
      ▼
Document Classifier → "purchase_order"
      │
      ▼
SAP DocAI (SAP_purchaseOrder_schema)
      │                    ↓
      │            + LLM enrichment (fills gaps)
      │
      ▼
Field Extraction:
  - Customer name  → receiverName (buyer, not AI4U)
  - PO number      → purchaseOrderNumber
  - Delivery date  → requestedDeliveryDate
  - Line items     → material codes / descriptions
      │
      ▼
S/4HANA Validation:
  - Customer name → Business Partner lookup (substringof)
  - Material descriptions → SAP material code matching
    (exact code → substringof → description search → fallback)
      │
      ▼
Confirmation Modal (user reviews):
  SoldTo / ShipTo │ Material codes │ Quantities │ Issues
      │
      ▼
POST /api/v1/so/create → A_SalesOrder (deep-insert + per-item + header text)
      │
      ▼
Sales Order number returned
```

---

## Payment Advice Pipeline

Handles **Payment Advice documents** (remittance slips sent by a payer to notify which invoices are being settled):

```
Payment Advice (PDF)
      │
      ▼
Document Classifier → "payment_advice"
(keyword: "payment advice", "zahlungsavis", "remittance")
      │
      ▼
SAP DocAI (SAP_paymentAdvice_schema)
  - payer name, payment date, currency, total amount
  - bank reference, our reference
  - line items: invoice number, date, gross, discount, net payment
      │
      ▼
Payer BP resolution → search_customer_odata(payer_name)
      │
      ▼
POST /api/v1/pa/post → API_PAYMENT_ADVICE_SRV/A_PaymentAdvice
  - deep-create: header + to_PaymentAdviceItem
  - PaymentAdviceAccountType: "K" (vendor/supplier)
  - PaymentAdviceType: "10" (standard)
      │
      ▼
Payment Advice document number returned
```

**Key API fields:**

| Field | Description |
|---|---|
| `PaymentAdviceAccountType` | `K` = Vendor/Supplier, `D` = Customer |
| `PaymentAdviceAccount` | Business Partner number (payer) |
| `PaymentAdviceType` | `10` = standard payment advice |
| `PaymentCurrency` | Currency of the payment |
| `to_PaymentAdviceItem.AssignmentReference` | Invoice reference being settled |
| `to_PaymentAdviceItem.NetPaymentAmountInPaytCurrency` | Amount paid per invoice |

**Note:** Requires Communication Scenario `SAP_COM_0331` (Finance - Payment Advice Integration) in S/4HANA Cloud. The actual FI clearing of open items is a separate step.

---

## 3-Panel UX & Chat Assistant

The frontend uses a **3-panel workspace** layout:

```
┌─────────────────┬──────────────────────────┬─────────────────────┐
│  1 · UPLOAD     │  2 · RESULTS             │  3 · ASSISTANT      │
│  Document &     │  Extraction Review       │  DocAI Assistant    │
│  Pipeline       │                          │                     │
│                 │  ┌ SAP DocAI             │  ┌ Chat messages    │
│  • Pipeline     │  ├ LLM Technique 1       │  │                  │
│    selector     │  ├ LLM Technique 2       │  │ Streaming NDJSON │
│  • File upload  │  ├ Comparison            │  │ response from    │
│  • Execute btn  │  └ Summary               │  │ SAP GenAI Hub    │
│                 │                          │  │                  │
│                 │  [POST S4]               │  └ Textarea + Send  │
└─────────────────┴──────────────────────────┴─────────────────────┘
```

**Streaming Chat Assistant** (`POST /api/v1/chat/message`):
- Knows the full DocAI system context (pipelines, schemas, FI/MM/SD posting)
- Streams responses token-by-token via NDJSON (`{"type":"delta","content":"..."}`)
- Receives extraction context automatically (invoice fields, route, amounts)
- Powered by SAP GenAI Hub (gpt-4o)

---

## Project Structure

```
docai/
├── backend/
│   ├── api.py                          # FastAPI app — all endpoints + transparent routing
│   ├── config.py                       # Settings (S4, FI, SAP GenAI Hub)
│   ├── docai.json                      # SAP DocAI credentials (not committed)
│   ├── requirements.txt
│   ├── modules/
│   │   ├── auth/get_token.py           # OAuth2 token manager (SAP BTP UAA)
│   │   ├── invoice/process_invoice.py  # SAP Document AI job submission + polling
│   │   ├── genai/                      # LLM client + multimodal extraction
│   │   ├── routing/                    # Supplier detection + template matching
│   │   ├── docai_new/                  # V2 auto-template pipeline
│   │   ├── evaluation/                 # Quality evaluation + LLM judge
│   │   ├── schemas/                    # SAP DocAI schema fetcher
│   │   └── templates/                  # SAP DocAI template fetcher
│   ├── S4/
│   │   ├── sap_credentials.py          # Dynamic credentials (headers or .env)
│   │   ├── s4_client.py                # S/4HANA HTTP session builder
│   │   ├── sap_session_routes.py       # POST /api/sap/session/login
│   │   ├── search_routes.py            # GET /api/customers/search, /api/materials/search
│   │   ├── debug_routes.py             # GET /api/debug/ping-sap
│   │   ├── business_partners/          # GET /api/business-partners
│   │   ├── supplier_invoice/           # POST /api/v1/fi/post-invoice (GL)
│   │   ├── po_invoice/                 # POST /api/v1/fi/post-po-invoice (MIRO)
│   │   │   ├── po_detector.py          # Detect PO number in extraction results
│   │   │   └── document_type_detector.py
│   │   └── purchase_orders/            # GET /api/purchase-orders (by vendor)
│   ├── SalesOrderProcess/
│   │   ├── document_classifier.py      # Keyword + LLM document type detection (invoice/PO/PA)
│   │   ├── so_extractor.py             # SAP DocAI with SAP_purchaseOrder_schema
│   │   ├── so_validator.py             # Customer BP + material S4 validation
│   │   ├── so_creator.py               # CSRF + deep-insert → A_SalesOrder
│   │   └── so_routes.py                # POST /api/v1/so/extract|validate|create
│   ├── PaymentAdviceProcess/
│   │   ├── pa_models.py                # ExtractedPaymentAdvice, PostPaymentAdviceRequest/Response
│   │   ├── pa_extractor.py             # SAP DocAI with SAP_paymentAdvice_schema
│   │   ├── pa_poster.py                # CSRF + POST → API_PAYMENT_ADVICE_SRV/A_PaymentAdvice
│   │   └── pa_routes.py                # POST /api/v1/pa/extract|post
│   ├── matching/
│   │   ├── customer_api_matcher.py     # BP search (substringof V2 OData + rapidfuzz)
│   │   └── product_api_matcher.py      # Material search (exact + description)
│   └── modules/chat/
│       └── chat_service.py             # Streaming NDJSON chat via GenAI Hub
├── frontend/
│   ├── index.html                      # SAP UI5 Horizon theme shell
│   ├── vite.config.ts
│   └── src/
│       ├── main.ts                     # UI5 web components bootstrap
│       ├── api/
│       │   ├── client.ts               # Axios + fetch API client
│       │   ├── types.ts                # All TypeScript interfaces
│       │   ├── docai-new-client.ts     # V2 pipeline client
│       │   └── docai-new-types.ts      # V2 TypeScript types
│       └── components/
│           ├── app.ts                  # 3-panel shell + all routing logic
│           ├── app-render.ts           # Legacy result renderers
│           ├── docai-new-app.ts        # V2 pipeline UI
│           └── train-template-app.ts   # Template training UI
└── DemoFiles/
    ├── invoice_GL_no_PO.pdf            # Supplier invoice — GL posting demo
    ├── invoice_PO_4500001270.pdf       # Supplier invoice with PO reference
    ├── po_TG11_only.pdf                # Customer PO — single material (TG11)
    ├── po_TG12_only.pdf                # Customer PO — single material (TG12)
    └── po_TG11_TG12_multi.pdf         # Customer PO — two materials
```

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- Access to **SAP Document Information Extraction** (BTP service instance with `SAP_invoice_schema` and `SAP_purchaseOrder_schema`)
- Access to **SAP GenAI Hub** (AI Core deployment with `gpt-4o` or compatible multimodal model)
- Access to **SAP S/4HANA** system (on-premise or Cloud) with:
  - `API_BUSINESS_PARTNER` activated
  - `API_SUPPLIERINVOICE_PROCESS_SRV` activated
  - `API_SALES_ORDER_SRV` activated
  - `API_PRODUCT_SRV` activated

---

## Backend Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure SAP Document AI credentials

Place your SAP BTP service key in `backend/docai.json`.

### 3. Configure environment variables

Create `backend/.env`:

```env
# SAP GenAI Hub
AICORE_AUTH_URL=https://...
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2
AICORE_RESOURCE_GROUP=default

# S/4HANA Connection
S4_BASE_URL=https://<host>:<port>
S4_CLIENT=100
S4_VERIFY=false
S4_USERNAME=...
S4_PASSWORD=...

# FI Invoice Posting
FI_COMPANY_CODE=1010
FI_EXPENSE_GL_ACCOUNT=11001000
FI_PO_TAX_CODE=V0
```

### 4. Run the API server

```bash
cd backend
uvicorn api:app --reload --port 8001
```

Interactive docs: `http://localhost:8001/docs`

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dev server runs at `http://localhost:3000` (proxies `/api/*` to `http://localhost:8001`).

---

## API Reference

### System
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |

### SAP Session
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/sap/session/login` | Validate S/4HANA credentials |
| `GET` | `/api/sap/session/status` | Check current credential source |

### Master Data Search
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/customers/search?q=` | Search Business Partners by name |
| `GET` | `/api/materials/search?q=` | Search materials by code or description |
| `GET` | `/api/business-partners` | List all Business Partners |
| `GET` | `/api/purchase-orders?supplier=` | List open POs by vendor BP |
| `GET` | `/api/debug/ping-sap` | Connectivity diagnostics |

### Document AI — V1 Pipeline
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/genai/pipeline` | **Main pipeline** — auto-detects doc type, routes to invoice or PO flow |
| `POST` | `/api/v1/invoice/process` | SAP DocAI extraction only |
| `POST` | `/api/v1/evaluation/run` | Evaluate last extraction results |
| `GET` | `/api/v1/schemas` | List SAP DocAI schemas |
| `GET` | `/api/v1/templates` | List SAP DocAI templates |

### Document AI — V2 Pipeline
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/docai-new/process` | LLM-first + auto-template pipeline |
| `POST` | `/api/v2/docai-new/train` | Train existing template with PDFs |
| `GET` | `/api/v2/docai-new/templates` | List all templates |

### S/4HANA Posting
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/fi/post-invoice` | Post supplier invoice to FI (GL account) |
| `POST` | `/api/v1/fi/post-po-invoice` | Post supplier invoice via PO reference (MIRO) |
| `POST` | `/api/v1/so/extract` | Extract customer PO via SAP DocAI |
| `POST` | `/api/v1/so/validate` | Validate PO against S/4HANA master data |
| `POST` | `/api/v1/so/create` | Create Sales Order in S/4HANA |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat/message` | Streaming chat (NDJSON) with DocAI Assistant |

---

## Demo Files

The `DemoFiles/` directory contains ready-to-use test documents:

| File | Type | Description |
|---|---|---|
| `invoice_GL_no_PO.pdf` | Supplier Invoice | EUR 9,994.96 — Vendor 10300003 — no PO reference |
| `invoice_PO_4500001270.pdf` | Supplier Invoice (PO) | EUR 10,000 — PO 4500001270 — tests MIRO posting |
| `po_TG11_only.pdf` | Customer PO | Single item: TG11 (MXA920W-S-60CM) — EUR |
| `po_TG12_only.pdf` | Customer PO | Single item: TG12 (920 Ceiling Array) — USD |
| `po_TG11_TG12_multi.pdf` | Customer PO | Two items: TG11 + TG12 — EUR — tests multi-item SO |

All customer POs are addressed **TO: AI4U GmbH** (vendor) FROM real S/4HANA customers (BP `10100002`, `10100004`, `1000046`). Materials TG11 and TG12 exist in the S/4HANA demo system.

---

## Deployment (SAP BTP Cloud Foundry)

```bash
# Backend
cd backend
cf push

# Frontend
cd frontend
npm run build
cf push
```

---

## License

Internal SAP BTP AI Services CoE project — AI4U initiative.
