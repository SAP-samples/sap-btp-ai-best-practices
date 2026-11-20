# Order intake (upload, e-mail parsing, storage)

## What it is

The deterministic first step of the pipeline. It receives what a customer sent (an `.eml`
e-mail, loose documents, pasted e-mail text), splits e-mails into body, attachments, inline
images and forwarded e-mails, decides which files the AI agent receives and how, and stores
everything in HANA together with one order per message in status `QUEUED`. No LLM is involved.

## What it is for

* It is the seam of the mocked inbox: the UI upload dialog calls it today; in production the
  e-mail trigger (Power Automate / MS Graph webhook / SAP Integration Suite Mail adapter) posts
  the raw `.eml` to the same endpoint.
* It prepares the agent's input: PDFs and images are passed natively to the multimodal model
  (so logos, stamps and scanned pages are visible), Word/Excel files and forwarded e-mails as
  text, and every file the model does not receive is listed with the reason.
* It gives the reviewer every original file in the UI document viewer.

## Inputs and outputs

`POST /api/orders/upload` (multipart, header `X-API-Key`):

| Field | Type | Meaning |
|---|---|---|
| `files` | 0..n files | `.eml`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.docx`, `.xlsx`, `.txt`, `.csv` |
| `email_text` | string, optional | Pasted e-mail content |
| `sales_org` | string, optional | Sales organisation (simulates the inbox-to-sales-org mapping of the requirements) |

Response `202`: `{"orders": [{"order_id", "message_id", "status": "QUEUED", "source"}]}`.
Errors: `400` empty upload / unsupported type / Outlook `.msg` (out of scope), `413` above `MAX_UPLOAD_MB`.

Grouping: every `.eml` is its own message (one order each); all other files plus the pasted text
form one message; pasted text alone is a `TEXT` message; pasted text with exactly one `.eml`
and nothing else is appended to that e-mail's body.

Other endpoints of this phase:

| Endpoint | Purpose |
|---|---|
| `GET /api/orders?status=&q=&limit=` | Order list (status filter is comma separated) |
| `GET /api/orders/{order_id}` | Header, items, field states, issues, message, attachments (with `viewer`), latest run |
| `GET /api/orders/{order_id}/attachments/{attachment_id}` | Original bytes, `Content-Disposition: inline`, `X-Content-Type-Options: nosniff` |

Stored rows: `POAGENT_INBOUND_MESSAGE` (headers, body, received date, sales-org hint),
`POAGENT_ATTACHMENT` (bytes, role, model input decision and reason, extracted text, nesting) and
`POAGENT_ORDER_HEADER` (status `QUEUED`). See [hana-data-model.md](hana-data-model.md).

### How files reach the model

| Role | Example | `MODEL_INPUT` |
|---|---|---|
| `SOURCE_EMAIL` | the uploaded `.eml` itself | `NONE` (its body and parts are given separately) |
| `DOCUMENT` | PDF / image attachment or loose file | `NATIVE` |
| `DOCUMENT` | `.docx`, `.xlsx`, `.txt`, `.csv` | `TEXT` (converted) |
| `DOCUMENT` | `.msg`, `.zip`, other types inside an e-mail | `NONE` + reason |
| `INLINE_IMAGE` | image referenced from the body (`Content-ID` in `multipart/related`) | `NATIVE`, unless filtered |
| `FORWARDED_EMAIL` | `message/rfc822` part | `TEXT` (headers + body); its own files are nested children |

**Inline images are kept by default**, because a customer may paste a screenshot of the order
into the e-mail body. Only obvious noise is filtered, deterministically: images whose shorter
side is below `INLINE_IMAGE_MIN_PX` (default 150 px: logos, social icons, tracking pixels) and
exact duplicates of an image already kept in the same message (a signature repeated in a
thread). Filtered images keep `MODEL_INPUT = NONE` with a reason such as
`likely logo/icon (179x62 px)` and remain visible to the reviewer.

## Related files

| File | Role |
|---|---|
| `api/app/routers/orders.py` | Endpoints, row-to-API conversion, attachment streaming |
| `api/app/models/orders.py` | API models (`OrderSummary`, `OrderDetail`, `AttachmentInfo`, ...) |
| `api/app/intake/service.py` | Validation, grouping, single-transaction persistence |
| `api/app/intake/email_parsing.py` | MIME walk, HTML-to-text, forwarded e-mails, inline-image filter |
| `api/app/intake/documents.py` | Per-file `MODEL_INPUT` decision and roles |
| `api/app/intake/office_text.py` | `.docx` (standard library) and `.xlsx` (openpyxl) to text |
| `api/app/intake/image_info.py` | PNG/GIF/JPEG pixel size from headers |
| `api/app/orders/repository.py` | Order list / detail / attachment queries |
| `api/app/db/sql.py` | Row helpers (upper-case keys, LOBs read, Decimal to float) |
| `api/app/orders/settings.py` | `MAX_UPLOAD_MB`, `INLINE_IMAGE_MIN_PX` and other settings |

## Key code

```python
# Inline = referenced from the body: Content-ID inside multipart/related (clients differ on the
# disposition they set), or an explicit "inline" disposition.
if ctype.startswith("image/") and ((in_related and part["Content-ID"]) or disposition == "inline"):
    files.append(StoredFile(name, ctype, content, ROLE_INLINE_IMAGE, "NATIVE"))
```

```python
elif size and min(size) < min_px:
    stored.model_input, stored.model_input_reason = "NONE", f"likely logo/icon ({size[0]}x{size[1]} px)"
```

## How to test

```bash
cd api
../.venv/bin/python -m unittest discover -s ../tests/unit -t ..   # tests/unit/test_email_parsing.py
```

The tests build e-mails in memory: attachment, pasted screenshot kept, 88x31 logo and duplicate
filtered, forwarded e-mail with its own inline image, HTML-only body, generated `.docx`/`.xlsx`,
image header parsing, grouping and rejection rules.

Manual check (API running, `X-API-Key` set):

```bash
curl -s -H "X-API-Key: $API_KEY" -F "files=@order.eml" http://127.0.0.1:8000/api/orders/upload
curl -s -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/api/orders
```
