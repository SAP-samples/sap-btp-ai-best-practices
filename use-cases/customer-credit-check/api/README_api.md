# Document Extraction, Verification & Credit Policy API

Comprehensive FastAPI service for:
- Document information extraction from PDFs
- Attribute verification between two documents (KYC ↔ CSF/INE, etc.)
- Credit policy evaluation (new/update/exception) and payment score metrics

Base URL: `/api`

---

## Health

- GET `/api/health`
  - Returns service status.

Example:
```bash
curl -s http://127.0.0.1:8000/api/health
```

---

## Extraction API

Router prefix: `/api/extraction`

### 1) Extract from single PDF
- POST `/api/extraction/single`
- Content-Type: `multipart/form-data`
- Fields:
  - `file`: PDF file
  - `document_type`: one of `conoce_cliente`, `comentarios_vendedor`, `constancia_fiscal`, `ine`, `custom`
  - `questions` (optional): JSON array string of custom questions
  - `fields` (optional): JSON array string of display field names (same length as questions)
  - `temperature` (optional, default 0.1)
  - `language` (optional, default `es`)
  - `use_simple_extractor` (optional, default `true`)

Response (model `ExtractionResponse`):
```json
{
  "success": true,
  "document_type": "conoce_cliente",
  "results": [
    {"question": "¿Cuál es el RFC?", "answer": "XAXX010101000", "field": "RFC"}
  ],
  "processing_time_ms": 1234.56,
  "error": null,
  "metadata": null
}
```

Curl example:
```bash
curl -X POST http://127.0.0.1:8000/api/extraction/single \
  -F "file=@/path/to/file.pdf" \
  -F "document_type=constancia_fiscal" \
  -F "use_simple_extractor=true"
```

With custom questions/fields:
```bash
curl -X POST http://127.0.0.1:8000/api/extraction/single \
  -F "file=@/path/to/file.pdf" \
  -F 'questions=["¿Cuál es el RFC?","¿Cuál es la razón social?"]' \
  -F 'fields=["RFC","Razón Social"]' \
  -F "document_type=custom"
```

### 2) Batch extraction
- POST `/api/extraction/batch`
- Content-Type: `multipart/form-data`
- Fields:
  - `files`: multiple PDF files
  - `document_types`: JSON array string (one type per uploaded file)
  - `max_concurrent` (optional, default 10)
  - `use_simple_extractor` (optional, default `true`)

Response (model `BatchExtractionResponse`):
```json
{
  "task_id": "<uuid>",
  "total_documents": 2,
  "status": "processing",
  "created_at": 1730000000.0
}
```

Curl example:
```bash
curl -X POST http://127.0.0.1:8000/api/extraction/batch \
  -F "files=@/path/a.pdf" -F "files=@/path/b.pdf" \
  -F 'document_types=["constancia_fiscal","ine"]'
```

### 3) Batch status
- GET `/api/extraction/status/{task_id}`

Response (model `TaskStatusResponse`):
```json
{
  "task_id": "...",
  "status": "completed",
  "progress": 100,
  "message": "Processing 2 documents",
  "result": [ /* array of ExtractionResponse when completed */ ]
}
```

### 4) Schemas
- GET `/api/extraction/schemas` → returns configured questions/fields for each document type.
- POST `/api/extraction/schemas` → update schema for a document type.
  - Body:
```json
{
  "document_type": "constancia_fiscal",
  "questions": ["¿Cuál es el RFC?"],
  "fields": ["RFC"]
}
```

---

## Verification API

Router prefix: `/api/verification`

Compare attributes between two documents with flexible comparators.

### Comparators
- `name`: token-set equality (case/accents-insensitive, order-insensitive)
- `id`: alphanumeric-only uppercase equality (tolerates spaces/dashes/case)
- `text`: normalized text equality (lowercase, accents removed, whitespace collapsed)
- `fuzzy`: token Jaccard similarity in [0,1]; set `threshold` (default 0.82)
- `currency`: normalizes to MXN/USD/EUR synonyms before equality
- `contains`: substring check after normalization
- `address`: specialized address comparison
  - strips trailing "entre calle ..." details
  - treats `avenida`/`av.` as `av`
  - ignores stopwords like `cp`, `calle`, `col`, `mexico`, etc.
  - enforces equal 5-digit C.P. if present in both
  - then uses token Jaccard with configurable `threshold` (e.g., 0.50)
- `number`: compares the first integer found in each string (e.g., "30", "30 días", "30 (días)" → 30)

### Request models
- `VerifyRequest`:
```json
{
  "doc_a": {"RFC":"ABC010203-XYZ","Customer Legal Name":"Comercializadora Demo SA de CV"},
  "doc_b": {"RFC":"abc010203xyz","Razón social":"comercializadora demo s.a. de c.v."},
  "key_map": {
    "RFC": {"to":"RFC","comparator":"id"},
    "Customer Legal Name": {"to":"Razón social","comparator":"name"},
    "Domicilio Fiscal (Información Fiscal)": {"to":"Datos del Domicilio registrado","comparator":"address","threshold":0.50}
  }
}
```
- `key_map` may also use the short form string value, which implies comparator `auto`:
```json
{"RFC":"RFC","Customer Legal Name":"Razón social"}
```

### 1) Verify attributes
- POST `/api/verification/verify`

Response (model `VerifyResponse`):
```json
{
  "verified": [ {"key_a":"RFC","key_b":"RFC","value_a":"ABC...","value_b":"abc...","comparator":"id","match":true} ],
  "failed": [ {"key_a":"Fiscal Address","key_b":"Domicilio Fiscal","match":false,"reason":"values differ after normalization"} ],
  "summary": {"verified": 1, "failed": 0, "total": 1}
}
```

Curl example:
```bash
curl -X POST http://127.0.0.1:8000/api/verification/verify \
  -H "Content-Type: application/json" \
  -d '{
    "doc_a": {"RFC":"ABC010203-XYZ","Customer Legal Name":"Comercializadora Demo SA de CV"},
    "doc_b": {"RFC":"abc010203xyz","Razón social":"comercializadora demo s.a. de c.v."},
    "key_map": {
      "RFC": {"to":"RFC","comparator":"id"},
      "Customer Legal Name": {"to":"Razón social","comparator":"name"}
    }
  }'
```

### 2) Verification health
- GET `/api/verification/health`
```json
{"status":"ok","component":"verification","version":"0.2.0"}
```

---

## Credit Policy API

Router prefix: `/api/credit`

Provides programmatic access to the credit policy engine defined in `credit_policy_engine.py`. You can evaluate a credit request end‑to‑end or just compute payment history scores.

### Models overview
- `Invoice`: `{invoice_id, issue_date, due_date, paid_date?, amount, currency}`
- `CustomerInput`: `{customer_id, legal_name, persona(PF|PM), country, entity_name?, customer_group?(A|B), cgv_signed_date?, pagare_signed, guarantors, insurance_full_credit}`
- `DocsInput`: `{kyc_date?, seller_comments_present, address_proof_date?, tax_cert_date?}`
- `CreditRequest`: `{use_case(new|update|exception), requested_amount, requested_currency, requested_terms_days, last_update_date?, current_credit_line, current_credit_currency}`
- `Investigation`: `{mmr_amount?, mmr_currency, legal_risk?, external_investigation_date?, onsite_visit_done}`
- `BehaviorInput`: `{invoices:[Invoice], has_overdue_invoices, advance_purchases_count, has_active_credit, exceptions_in_semester}`
- `RoleContext`: `{role(analyst|coordinator)}`

### 1) Evaluate credit request
- POST `/api/credit/evaluate`
- Body (model `EvaluateRequest`):
```json
{
  "customer": {"customer_id":"C1","legal_name":"Cliente Demo","persona":"PF","country":"MX","entity_name":"Sample Foods S.A. de C.V.","cgv_signed_date":"2025-06-01T00:00:00","pagare_signed":true,"guarantors":1,"insurance_full_credit":false},
  "docs": {"kyc_date":"2025-02-01T00:00:00","seller_comments_present":true,"address_proof_date":"2025-07-01T00:00:00","tax_cert_date":"2025-07-10T00:00:00"},
  "request": {"use_case":"new","requested_amount":800000,"requested_currency":"MXN","requested_terms_days":30},
  "investigation": {"mmr_amount":1000000,"mmr_currency":"MXN","legal_risk":"low","external_investigation_date":"2025-08-20T00:00:00"},
  "behavior": {"invoices":[],"has_overdue_invoices":false,"advance_purchases_count":3,"has_active_credit":false,"exceptions_in_semester":0},
  "role": {"role":"analyst"},
  "as_of": "2025-09-05T00:00:00"
}
```

Response: JSON including `scores` (CAL/C3M/CA_by_year), `checks` per use case, and `decision_hint.needs_director`.

Curl example:
```bash
curl -X POST http://127.0.0.1:8000/api/credit/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {"customer_id":"C1","legal_name":"Cliente Demo","persona":"PF","country":"MX"},
    "docs": {"seller_comments_present": true},
    "request": {"use_case":"new","requested_amount":800000,"requested_currency":"MXN","requested_terms_days":30},
    "investigation": {},
    "behavior": {"invoices":[],"has_overdue_invoices":false,"advance_purchases_count":0,"has_active_credit":false,"exceptions_in_semester":0},
    "role": {"role":"analyst"}
  }'
```

### 2) Compute payment scores
- POST `/api/credit/scores`
- Body (model `ScoresRequest`):
```json
{
  "invoices": [
    {"invoice_id":"INV001","issue_date":"2025-05-01T00:00:00","due_date":"2025-05-31T00:00:00","paid_date":"2025-06-07T00:00:00","amount":50000,"currency":"MXN"}
  ],
  "group": "A"
}
```
- Alternatively omit `group` and provide `entity_name` and optional `explicit_group` to derive it.

Response:
```json
{"cp_by_invoice":[{"invoice_id":"INV001","days_late":7,"label":"Good","score":8}],"CA_by_year_pct":{"2025":80.0},"C3M_pct":80.0,"CH_pct":80.0,"CAL":"Good"}
```

### 3) Credit policy health
- GET `/api/credit/health`
```json
{"status":"ok","component":"credit","version":"0.1.0"}
```

---

## Running locally

Python 3.9+ recommended.

Install dependencies:
```bash
pip install -r api/requirements.txt
```

Run the server:
```bash
uvicorn api.api_server:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

---

## Notes
- CORS is enabled for common localhost ports in development; configure `ALLOWED_ORIGIN` and `APP_ENV` for production.
- Extraction uses a simplified image-only path by default; you can switch to the LangGraph path by setting `use_simple_extractor=false` in requests where supported.
