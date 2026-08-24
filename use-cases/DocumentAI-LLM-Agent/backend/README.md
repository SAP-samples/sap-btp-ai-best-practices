# SAP Document AI + Gen AI Hub — Invoice Extraction Pipeline

A modular Python 3.11+ project for extracting structured data from invoices using **SAP Document AI** and **SAP Gen AI Hub** (multimodal LLM).

---

## Project Structure

```
docai/
├── modules/
│   ├── auth/
│   │   └── get_token.py              # OAuth2 token management (TokenManager)
│   ├── schemas/
│   │   └── get_schema.py             # Fetch available schemas
│   ├── templates/
│   │   └── get_templates.py          # Fetch available templates
│   ├── invoice/
│   │   └── process_invoice.py        # Submit invoice + poll job
│   ├── genai/
│   │   ├── llm_client.py             # SAP Gen AI Hub LLM client
│   │   ├── multimodal_prompting.py   # Technique 1: free prompting
│   │   ├── multimodal_structured.py  # Technique 2: strict JSON schema
│   │   ├── compare_results.py        # Compare SAP vs LLM results
│   │   └── process_with_genai.py     # Full pipeline orchestrator
│   └── evaluation/
│       ├── evaluator.py              # Evaluation pipeline orchestrator
│       ├── field_analyzer.py         # Field completeness & conflict analysis
│       ├── score_calculator.py       # Scoring per extraction method
│       ├── llm_evaluator.py          # LLM-powered intelligent evaluation
│       └── generate_report.py        # Report generation & file output
├── utils/
│   └── config_loader.py              # Load & normalize docai.json credentials
├── output/
│   ├── genai/                        # GenAI pipeline outputs
│   └── evaluation/                   # Evaluation report outputs
├── invoice/                          # Place PDF invoices here
├── docai.json                        # SAP Document AI credentials
├── .env                              # Gen AI Hub environment variables
├── main.py                           # Entry point
└── requirements.txt                  # Python dependencies
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd docai

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

### docai.json

Place your SAP Document AI service key in `docai.json` at the project root.

Supported formats:

**SAP BTP Service Key (with `uaa` block):**
```json
{
  "uaa": {
    "clientid": "sb-...",
    "clientsecret": "...",
    "url": "https://....authentication.eu10.hana.ondemand.com"
  },
  "url": "https://...",
  "tenantuiurl": "https://ar-demo-xxx.eu10.doc.cloud.sap"
}
```

**Flat structure:**
```json
{
  "clientid": "sb-...",
  "clientsecret": "...",
  "token_url": "https://....authentication.eu10.hana.ondemand.com/oauth/token",
  "service_url": "https://ar-demo-xxx.eu10.doc.cloud.sap"
}
```

### .env

Configure SAP Gen AI Hub credentials:

```env
AICORE_AUTH_URL=https://...
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_RESOURCE_GROUP=default
AICORE_BASE_URL=https://...
GENAI_MODEL_NAME=gpt-4o
GENAI_MAX_RETRIES=3
```

---

## Usage

### Interactive menu

```bash
python main.py
```

```
======================================================
  SAP Document AI + Gen AI Hub — Main Menu
======================================================

  [1] Get Schemas
  [2] Get Templates
  [3] Process Invoice (SAP DocAI)
  [4] Process Invoice with GenAI Multimodal
  [5] Evaluate Extraction Quality
  [0] Exit
```

### Command-line flags

```bash
python main.py --schemas      # Fetch all available schemas
python main.py --templates    # Fetch all available templates
python main.py --invoice      # Process invoice with SAP Document AI
python main.py --genai        # Full pipeline: SAP + LLM Technique 1 + LLM Technique 2
python main.py --evaluate     # Evaluate and compare extraction results
```

---

## Pipeline Overview

### Option 3 — Process Invoice (SAP DocAI)

1. Lists PDF files in `invoice/`
2. User selects a file
3. Submits to SAP Document AI
4. Polls until job is `DONE`
5. Saves result to `output/{JOB_ID}.json`

### Option 4 — Process Invoice with GenAI

Runs 4 steps in sequence:

| Step | Description |
|------|-------------|
| 1/4  | SAP Document AI extraction |
| 2/4  | LLM Technique 1 — Free Prompting |
| 3/4  | LLM Technique 2 — Structured JSON Schema |
| 4/4  | Comparison of all three results |

**Output files** (`output/genai/`):
- `sap_result.json`
- `llm_multimodal_prompting.json`
- `llm_multimodal_structured.json`
- `comparison.json`
- `final_summary.txt`

### Option 5 — Evaluate Extraction Quality

Reads the outputs from Option 4 and runs:

1. **Field analysis** — completeness, missing fields, conflicts
2. **Score calculation** — weighted score per method
3. **LLM evaluation** — intelligent quality assessment
4. **Report generation** — executive summary

**Output files** (`output/evaluation/`):
- `evaluation.json`
- `missing_fields.json`
- `scores.json`
- `executive_summary.txt`

---

## Extracted Fields

The pipeline extracts 25 invoice fields:

| Field | Description |
|-------|-------------|
| `documentNumber` | Invoice number |
| `documentDate` | Invoice date (YYYY-MM-DD) |
| `grossAmount` | Total gross amount |
| `netAmount` | Net amount before tax |
| `taxAmount` | Tax amount |
| `taxRate` | Tax rate (%) |
| `currencyCode` | ISO 4217 currency code |
| `senderName` | Sender company name |
| `receiverName` | Receiver company name |
| `purchaseOrderNumber` | PO number |
| `deliveryDate` | Delivery date |
| `senderAddress` | Full sender address |
| `receiverAddress` | Full receiver address |
| `senderBankAccount` | Sender bank account |
| `taxId` | Tax identification number |
| `receiverContact` | Receiver contact person |
| `senderCity` / `senderStreet` / etc. | Address components |
| `lineItems` | Array of line items with description, quantity, unit price, net amount |
| `fieldConfidence` | LLM confidence score per field (0.0–1.0) |

---

## Confidence Levels

Each extracted field includes a confidence score from the LLM:

| Score | Meaning |
|-------|---------|
| `1.0` | Found and certain |
| `0.8` | Very likely correct |
| `0.6` | Likely correct |
| `0.4` | Uncertain |
| `0.2` | Guessed |
| `0.0` | Not found in document |

**Console output example:**
```
  - receiverName: Ben Dover  (100%)
  - grossAmount: 335000.0  (80%)
  - purchaseOrderNumber: PO-2024-001  (90%)
  - deliveryDate: 2024-03-15  (60%)
```

---

## Evaluation Scores

The evaluation module scores each method on:

| Metric | Weight |
|--------|--------|
| Completeness | 35% |
| Confidence avg | 30% |
| Consistency | 20% |
| Field coverage | 15% |

**Example output:**
```
  SAP Document AI:
    Overall score  : 84/100
    Completeness   : 80%
    Confidence avg : 87%
    Fields found   : 20/25
    Missing fields : 5

  LLM Structured:
    Overall score  : 94/100
    Completeness   : 97%
    Confidence avg : 92%
    Fields found   : 24/25
    Missing fields : 1
```

---

## Requirements

- Python 3.11+
- SAP Document AI service instance
- SAP AI Core / Gen AI Hub access
- PDF invoices in `invoice/` folder

See `requirements.txt` for all Python dependencies.