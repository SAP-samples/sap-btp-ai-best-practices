---
name: deduction-interpretation
description: Reusable methodology for interpreting payment-advice deductions — document-nature taxonomy, client-playbook workflow, 323 escalation defaults, evidence-based linkage guardrails, and human-confirmed playbook correction loop.
---

# Deduction-Interpretation Methodology

This skill defines the **client-agnostic methodology** for interpreting payment-advice
deductions.  It tells you *how* to reason about any customer's remittance; the actual
per-customer signals live in the client's playbook, not here.

---

## 1. Document-Nature Taxonomy

Every line item on a payment advice belongs to one of the following natures:

| Nature | What it represents |
|---|---|
| `invoice` | A supplier invoice the customer is paying (positive amounts are the norm; a negative can be a supplier credit memo — see below). |
| `credit_memo` | A supplier-issued credit memo; the customer is applying it as an offset.  Distinguishable from a chargeback because the *supplier* originated it. |
| `chargeback` | A customer-created deduction: a claim against the supplier for pricing errors, shortages, allowances not taken, etc.  The customer originated it. |
| `allowance_reversal` | A previously granted allowance the customer is reversing.  Often appears as a negative line adjacent to positive invoice rows. |
| `unknown` | The line cannot be confidently classified; use this value when no playbook rule or structural cue applies.  Lines with `unknown` nature are treated as unresolved and escalated via reason code 323. |

**Critical: these natures are determined by the client playbook, not by hard-coded prefix
rules.**  The signals that map a line to a nature — reference prefixes, marker column values,
sign conventions, gross-vs-net equality — are customer-specific and documented in
`ANCHORS_JSON`.  Never assume that a prefix like `PRG`, `SDRBG`, or any other string has
universal meaning.  Always read the client's playbook before classifying.

---

## 2. General Workflow

You receive **ONLY** the pre-filtered anomalous lines (negatives / chargebacks /
lines with a `deduction_reason`), each tagged with a `line_index` that identifies
the line's position in the original full `line_items` list.  You do **NOT** see
ordinary invoice lines — never invent or relabel them.

For each line you are given:
- Classify `document_nature` (see §1 taxonomy).
- Assign `reason_code` (`"323"` + escalation flags when unmapped).
- Set `flags` as appropriate (e.g. `"linkage-outside-document"`,
  `"reason-code-defaulted"`, `"needs-customer-confirmation"`).
- **Always write a `rationale`** naming your reasoning and any tools you called
  (e.g. "playbook prefix rule ABC matched → chargeback; source explicitly references
  INV-001, retrieved with fetch_invoices").

Return exactly one entry per provided line, echoing its `line_index`.
Do NOT add entries for lines that were not provided.

Follow this sequence for every interpretation request:

### 2a. Resolve the client

Call `find_customers(query)` with the client name or alias provided by the caller.
It performs server-side fuzzy search and returns the top ranked `(client_key,
display_name, score)` matches.  Pick the best match; if ambiguous, ask the user.
Do not download the full customer list.

### 2b. Load the client playbook

Call `get_deduction_rules(client_key)`.

- **Playbook exists** → proceed to §2c.
- **Returns None (no playbook yet)** → proceed to §2e (bootstrap path).

### 2c. Understand the document

Before classifying any line, gather document-level context using the server-side
full-document tools.  These tools read the currently-bound document automatically —
**never pass the payload dict as a tool argument**.

1. Call `document_statistics()` to see the total line count, normal line count, and
   anomalous line count.  This tells you the overall document scale and how unusual
   the anomalous subset is relative to the whole.
2. Call `list_invoice_references(offset=0, limit=20)` to page through invoice
   references from the normal (positive) lines, then call `fetch_invoices(references)`
   with a sample of those references to inspect what ordinary invoice lines look like
   for this client — their structure, reference formats, sign conventions, and
   gross-vs-net relationships.

   You only receive the **anomalous** lines in the prompt; you must **pull** normal
   context via these tool calls to distinguish normal from anomalous.  Never assume
   what a normal invoice looks like without checking.

   Iterate `list_invoice_references` with a larger `offset` if you need more examples
   to establish reliable structural patterns.

### 2d. Apply the playbook

Use the `PLAYBOOK_TEXT` narrative plus the structured `ANCHORS_JSON` (column meanings,
marker semantics, code-prefix semantics, net equation, confirmed `code -> reason_code`
map) to classify each line item into one of the four document natures, assign a reason
code, and populate `residual_items` and `flags`.

For each line:
1. Apply the structural rules from the playbook.
2. If the playbook defines a `code -> reason_code` mapping and the line carries a
   matching code, use that reason code directly.
3. If a line does not match any known pattern, apply the §3 defaults (reason code 323).
4. Describe invoice links only when supported by explicit source references or saved
   evidence. Never search invoice combinations or infer links from equal amounts,
   even if an older customer playbook asks for amount matching. If linkage is not
   evidenced, leave it unresolved and request customer confirmation.
5. Always write a `rationale` documenting which rule matched and which tools were called.

### 2e. Bootstrap path — no playbook yet

When a client has no playbook:

1. Call `document_statistics()` to see the total line count and the normal/anomalous
   split.  Use `list_invoice_references(offset=0, limit=20)` and
   `fetch_invoices(references)` to sample both normal and anomalous lines to understand
   the structural patterns — reference formats, marker column values, sign conventions,
   gross-vs-net relationships.  Do not pass the payload as a tool argument; these tools
   read the server-side bound document automatically.
2. Reason over the interesting rows to hypothesize the structural logic: what markers
   appear, what prefixes recur, how sign and gross-vs-net relate.
3. Draft a proposed playbook (natural-language `PLAYBOOK_TEXT` + structured
   `ANCHORS_JSON`) based on your hypothesis.
4. Call `save_deduction_rules(client_key, playbook, user_confirmed=False)` to **stage
   the proposal** — this returns a preview/diff without writing to HANA.
5. Present the preview to the user and ask for confirmation.
6. On human approval, call `save_deduction_rules(client_key, playbook,
   user_confirmed=True)` to **persist the first revision**.

---

## 3. Default — Reason Code 323 and Escalation

When a line cannot be mapped to a known deduction type:

- Set `reason_code: "323"`.
- Add `"reason-code-defaulted"` and `"needs-customer-confirmation"` to the line's
  `flags`.
- Include a `residual_items` entry of the appropriate deduction type (use `"unknown"`
  if the nature itself is undetermined).

**Note on 323:** this placeholder has no authoritative SAP business meaning today.  It
means "deduction could not be mapped; escalate for manual review."  The real reason-code
catalog and true default must come from the client's SAP configuration.  Until then,
323 is the safe escalation sentinel.

Separate residual items per deduction type — do not consolidate a chargeback and a
credit memo into one residual line.

---

## 4. Guardrails — Document Boundaries

**Never auto-link a chargeback to invoices in a different document.**

Combinatorial invoice matching is not available. Use document lookup tools to inspect
explicit references, not to reconstruct an amount-based search. When explaining an
existing link, distinguish saved historical rationale from a verified source reference;
an old algorithm's output is not proof of a business relationship.

When a chargeback cannot be reconciled within the current document, do NOT reach outside:

- Set `reason_code: "323"`.
- Add `"linkage-outside-document"` only when the evidence actually points outside this document.
- Add `"needs-customer-confirmation"` to `flags`.
- Leave `residual_items` with the full unresolved amount.

Cross-document linkage (e.g. resolving a chargeback against invoices in other files or
against S/4 historical data) requires dedicated tooling that is out of scope until a
future iteration.

---

## 5. Playbook Correction Loop

After a human reviews the interpretation results, they may find the logic incomplete or
wrong.  Handle corrections as follows:

### Editing a playbook

To update an existing client's playbook:

1. Draft the amended `PLAYBOOK_TEXT` and `ANCHORS_JSON` incorporating the correction.
2. Call `save_deduction_rules(client_key, amended_playbook, user_confirmed=False)` to
   **stage the amendment** — returns a diff preview without writing.
3. Present the diff to the user.
4. On human approval, call `save_deduction_rules(client_key, amended_playbook,
   user_confirmed=True)` to persist the **next revision** (`REVISION + 1`).

Do not rewrite or overwrite a playbook on your own initiative.  Always stage first.

### Removing a playbook

To discard a client's entire playbook (e.g. it was hypothesized incorrectly and must
be rebuilt from scratch):

1. Call `delete_deduction_rules(client_key, user_confirmed=False)` to **preview the
   deletion** — no data is removed at this step.
2. Present the preview to the user and ask for explicit confirmation.
3. On human approval, call `delete_deduction_rules(client_key, user_confirmed=True)` to
   **execute the deletion**.

Edits and deletes are always human-confirmed.  The agent never rewrites or drops a
playbook autonomously.
