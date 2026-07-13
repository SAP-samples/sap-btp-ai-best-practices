# Matching Process

This document explains the two-stage invoice-to-payment matching process in detail.

---

## Overview

The system matches bank payments to customer invoices in two stages:

1. **Rule-based matching** — runs entirely in the browser (Web Worker), fast and deterministic
2. **AI-powered matching** — runs on the server using an LLM, handles the remaining unmatched invoices

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Upload CSVs    │────▶│  Rule-Based      │────▶│  AI Matching    │
│  (Invoice +     │     │  (Web Worker)    │     │  (Server + LLM) │
│   Payment)      │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                         │
                              ▼                         ▼
                        Matched invoices          AI-matched invoices
                        (high certainty)          (with confidence levels)
```

---

## Stage 1: Rule-Based Matching

Runs in the browser as a Web Worker for non-blocking performance.

### Input

- **Invoice CSV**: Contains invoice numbers and amounts (columns: `Invoice#`, `Invoice Amt`)
- **Payment CSV**: Contains payment amounts and text fields (columns: `TRANSACTION_AMT`, `BANK_REF`, `BY_ORD_OF_NAME`, `BY_ORD_OF_ADDR`, `REMIT_NAME`)

### Algorithm

1. **Parse both CSVs** — handles quoted fields, BOM, CRLF/LF, and different encodings (UTF-8 for invoices, Windows-1252 for payments)

2. **Build a payment index** keyed by integer-cent amount:
   - For each payment, divide `TRANSACTION_AMT` by 100 (bank amounts are in sen/cents)
   - Round to integer cents to avoid floating-point key issues
   - Concatenate all text columns into a single searchable string
   - Store in a hash map: `amount_cents → [{payAmt, text, bankRef}]`

3. **For each invoice**, attempt to match:
   - Look up payments within ±tolerance of the invoice amount (hash lookup, O(1))
   - Among amount-matched candidates, check if the invoice number appears as a substring in the payment's concatenated text fields
   - **Match criteria**: Exactly 1 payment passes both the amount check AND the text check
   - If 0 matches: unmatched (no candidate)
   - If 2+ matches: unmatched (ambiguous — cannot determine which payment is correct)

### Output

Each invoice gets a status:
- `matched` — exactly one payment found by amount + invoice number
- `unmatched` — zero or ambiguous matches; passed to Stage 2

### Performance

- O(n) for indexing payments, O(m) for matching invoices
- Handles 30,000+ invoices × 50,000+ payments in under 2 seconds
- Runs in a Web Worker so the UI remains responsive

---

## Stage 2: AI-Powered Matching

Runs on the FastAPI server. Only processes invoices that Stage 1 could not match.

### Why AI Is Needed

Bank payments in Japan have limited payer information:
- Names are romanized from katakana (e.g., カブシキガイシャ → KABUSIKIKAISHA or KA))
- Fields have fixed-width limits, causing names to split across multiple fields
- Abbreviations are common (KA = 株式会社)
- Romanization varies (OU/O, SHI/SI, CHI/TI, TSU/TU, FU/HU)
- The same company may appear under different billing entities

Invoice records have formal customer names, addresses, and city information that don't directly match the abbreviated bank data.

### Process

The AI matching has three sub-phases:

#### Phase 0: Local Fuzzy Matching

Before calling the LLM, the system attempts local fuzzy matching:

1. **Group payments by payer name** and invoices by customer name
2. **Pre-filter by amount overlap**: Only consider payer-customer pairs where at least one payment amount matches at least one invoice amount (within tolerance)
3. **Score each candidate pair** using:
   - **Name normalization**: Strip whitespace, punctuation, company suffixes; apply romanization normalization (OU→O, SHI→SI, etc.)
   - **Exact match after normalization**: score = 1.0
   - **Containment**: one name contains the other → score = 0.8
   - **Substring overlap**: 60%+ of the shorter name found in the longer → score = 0.5
   - **City/bank match**: bank branch name contains customer city → score = 0.4
4. **Match if score ≥ 0.4** AND amount is within tolerance

#### Phase 1: LLM Name Matching (Payer → Customer)

Identifies which payer identities correspond to which customer identities.

**Prompt structure:**
```
=== PAYERS (from bank payments) ===
0. Payer: SHIMOHARA KA) | Bank: TOKYO CHUO | Payments: 5 | Sample amounts: 15000, 23400
1. Payer: MATSUDA CORP | Bank: OSAKA NISHI | Payments: 3 | Sample amounts: 8900

=== CUSTOMERS (from invoices) ===
0. Customer: Shimohara Kabushiki Kaisha / 下原株式会社 | City: Tokyo | Invoices: 4
1. Customer: Matsuda Corporation | City: Osaka | Invoices: 2
```

**LLM response:**
```json
{"matches": [
  {"payerIdx": 0, "customerIdx": 0, "confidence": "HIGH", "reason": "SHIMOHARA KA = Shimohara Kabushiki Kaisha"},
  {"payerIdx": 1, "customerIdx": 1, "confidence": "HIGH", "reason": "MATSUDA CORP = Matsuda Corporation"}
]}
```

After receiving name matches, the system deterministically links individual payments to invoices within each matched payer-customer pair by amount (within tolerance).

#### Phase 2: LLM Direct Matching (Payment → Invoice)

For remaining unmatched payments, directly asks the LLM to match individual payments to specific invoices.

**Prompt structure:**
```
=== PAYMENTS ===
0. 15000.00 JPY | Payer: TANAKA | Bank: NAGOYA | Ref: BR-12345

=== UNMATCHED INVOICES ===
0. 15000.00 JPY | INV-001 | Customer: Tanaka Industries | City: Nagoya
1. 15000.00 JPY | INV-002 | Customer: Suzuki Ltd | City: Tokyo
```

**Rules enforced:**
- Amount MUST be within tolerance (hard filter — even if the LLM says they match, the system rejects if amounts differ beyond tolerance)
- Supports aggregated payments (one payment covering multiple invoices whose amounts sum to the payment amount)

### Parallel Execution

Phase 1 and Phase 2 LLM calls are fired simultaneously to minimize latency. Both prompts are constructed before any LLM call is made, then sent in parallel.

### Confidence Levels

- **HIGH**: Amount within tolerance + clear name match or city/bank location match
- **MEDIUM**: Amount within tolerance + partial name overlap or location similarity
- **LOW**: Amount within tolerance + weak signal (same region, similar structure)

### Deduplication

The system maintains sets of already-matched invoice numbers and bank references throughout all phases. Once an invoice or payment is matched, it cannot be matched again — preventing double-counting.

---

## Result Structure

Each matched invoice produces a result object:

| Field | Description |
|-------|-------------|
| `invoiceNumber` | The invoice identifier |
| `invoiceAmount` | Invoice amount in JPY |
| `paymentAmount` | Matched payment amount in JPY |
| `bankRef` | Bank reference number of the matched payment |
| `matchStatus` | `matched` (rule-based) or `ai_matched` (AI) or `unmatched` |
| `confidence` | `HIGH`, `MEDIUM`, or `LOW` (AI matches only) |
| `matchReason` | Explanation of why the match was made (AI matches only) |
| `branchProximity` | Whether the bank branch location matches the customer city |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Tolerance | 0.01 JPY | Maximum allowed difference between invoice and payment amounts |
| Demo mode | Off | When enabled, randomly samples 100 invoices for faster testing |
| Invoice number column | `Invoice#` | Column name in invoice CSV |
| Invoice amount column | `Invoice Amt` | Column name in invoice CSV |
| Payment amount column | `TRANSACTION_AMT` | Column name in payment CSV (values in sen, divided by 100) |
| Payment text columns | `BY_ORD_OF_NAME`, `BY_ORD_OF_ADDR`, `REMIT_NAME` | Columns searched for invoice number in rule-based matching |

---

## Data Flow Diagram

```
┌──────────────┐
│ Invoice CSV  │──┐
└──────────────┘  │    ┌─────────────────────────────┐
                  ├───▶│ Web Worker (Rule-Based)      │
┌──────────────┐  │    │                             │
│ Payment CSV  │──┘    │ 1. Parse CSVs              │
└──────────────┘       │ 2. Index payments by amount │
                       │ 3. Match: amount + text     │
                       └──────────────┬──────────────┘
                                      │
                            ┌─────────┴─────────┐
                            │                   │
                      ┌─────▼─────┐      ┌──────▼──────┐
                      │  Matched  │      │  Unmatched  │
                      │ (done)    │      │ (to server) │
                      └───────────┘      └──────┬──────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ FastAPI Server         │
                                    │                       │
                                    │ Phase 0: Fuzzy local  │
                                    │ Phase 1: LLM names  ──┼──┐
                                    │ Phase 2: LLM direct ──┼──┤ (parallel)
                                    │                       │  │
                                    │ Deduplicate & merge   │◀─┘
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ Final Results         │
                                    │ matched + ai_matched  │
                                    │ + unmatched           │
                                    └───────────────────────┘
```
