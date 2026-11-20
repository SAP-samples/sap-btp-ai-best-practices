# Order agent (LangGraph ReAct, gpt-6-luna)

## What it is

The AI part of the pipeline: a skill-aware LangGraph ReAct agent that reads one inbound message
(e-mail body, forwarded e-mails, PDFs, images, Word/Excel text), classifies it, extracts the
purchase order, resolves customer, ship-to and materials against (mock) ECC master data,
converts quantities, and hands the result over with `submit_order`. The agent never posts:
a deterministic gate decides afterwards whether the order is posted or held for review.

It is built on the `create-langgraph-react-agent` template runtime (v0.5.0), vendored in
`api/app/agent_core/` and embedded in the FastAPI process.

```
START -> load_skill_node -> compact_context -> agent -> tools -> compact_context -> agent -> END
```

## What it is for

* Replace manual keying of customer purchase orders into SAP ECC (VA01 / `BAPI_SALESORDER_CREATEFROMDAT2`).
* Fix what can be fixed automatically (normalisation, Vietnamese translation, unit conversion,
  ship-to and master-data completion), and flag precisely what a human must look at.

## Inputs and outputs

**Input** (built by `order_agent/prompt.py` from HANA):

```text
## Inbound message      order id, received date, inbox sales org, sender, recipients, subject
## E-mail body          plain text
## Attachment manifest  [A1] file (kind, type, size; pixels) -> attached natively | converted text below | not attached: reason
## Forwarded e-mails and converted documents   text of TEXT files, labelled [A#]
## Reviewer note        only when reprocessing was requested
```

PDFs and images are attached natively (multimodal), capped by `MAX_NATIVE_FILES` (5) and
`MAX_NATIVE_MB` (15): order documents first, then inline images by size.

The e-mail body, the converted document texts and every tool result are passed to the model in
full, never truncated, so no order content is dropped silently. If a run exceeds the model's
context window, the model call fails and the run is recorded as `ERROR`.

**Output** of `submit_order` (persisted, see [hana-data-model.md](hana-data-model.md)):
classification (`STANDARD`, `CONTRACT`, `NOT_AN_ORDER`) with reason, header and lines as
`{field: FieldValue}` maps, review flags, multiple-orders signal. Each `FieldValue` carries
`value` (normalised), `raw` (as printed), `confidence`, `source`, `note` (fix applied).
The agent's final text is stored as `AGENT_SUMMARY` for the reviewer; every run is recorded in
`POAGENT_AGENT_RUN` (outcome, model, token usage incl. cached tokens, trace without file bytes).

## Model configuration (`api/app/order_agent/config/agent.yaml`)

| Setting | Value | Why |
|---|---|---|
| model | `gpt-6-luna`, Responses API | requirement |
| reasoning_effort | `high` | requirement |
| temperature / max_tokens | not sent | requirement |
| summarization | disabled | the template's token estimate counts base64 PDF blocks as text and would summarise the documents away mid-run |
| memory (HANA conversation) | disabled | each order is one single-turn run; the audit trace is stored in `POAGENT_AGENT_RUN` |
| recursion_limit | 60 | about 19 tool rounds (skills, customer, profile, materials, quantities, up to three submissions) |
| a2a | disabled | the runtime is embedded in FastAPI |
| store | `false` (set in `api/app/agent_core/providers.py`) | stateless calls; see "Reasoning on SAP AI Core" |

### Reasoning on SAP AI Core

`gpt-6-luna` returns a `reasoning` item (encrypted, `summary: []`) with every answer. LangChain keeps it
in the AI message and sends it back with the next call so the model can continue its chain of thought.
SAP AI Core does not accept that (tested 2026-11-06 with plain HTTP against the deployment, no LangChain):

| Next call after a tool call | Result |
|---|---|
| reasoning item sent back (`store` true or false, any `api-version`) | `404 Resource not found` from the AI Core gateway, or a hang until the gateway gives up |
| previous response referenced with `previous_response_id` | 404 or hang (0 of 5 attempts succeeded) |
| no reasoning item, `store=false` | 200 |

The symptom was an order ending `FAILED` with `NotFoundError: Error code: 404` after one to several
minutes; short runs sometimes got through, longer ones almost never.

`ResponsesCompatibleChatOpenAI` therefore drops `type == "reasoning"` content blocks from every answer
(`_without_reasoning`, applied in `_generate` / `_agenerate`) and the model is created with `store=False`.
The blocks never reach the agent state, so they are neither sent back nor written to
`POAGENT_AGENT_RUN.TRACE` (about 40 % of a trace used to be encrypted reasoning nobody could read). The
model still sees every earlier tool call and result and reasons again in each step; its decisions stay
readable in the tool arguments (`material_evidence`, `note`, `review_flags`).

When AI Core accepts replayed reasoning again, remove the two overrides and `store=False`.

## Tools (`api/app/order_agent/tools.py`)

| Tool | Purpose |
|---|---|
| `load_skill` | Template tool: loads the skills below (all in one call) |
| `search_customers` | Rank sold-to accounts by tax id, sender e-mail domain, phone, name + address |
| `get_customer_profile` | Sales areas (currency, terms, incoterms, plant), partners (ship-tos), contacts, past reviewer corrections |
| `search_materials` | Candidate materials for all lines at once (material no., sales code variants, customer codes, manufacturer part number, description/colour/pack) with customer price and price check |
| `convert_quantities` | Deterministic unit conversion to the sales unit (litres -> pails via fill volume) |
| `submit_order` | Validate + persist; returns issues, system fills, ignored field names; never posts |

`submit_order` takes the order id from the graph state:

```python
def submit_order(classification, classification_reason, header, lines,
                 context_id: Annotated[str, InjectedState("context_id")], review_flags=None, ...):
```

so the model can neither see nor change which order it writes. The header and lines are
`{field_name: FieldValue}` maps (allowed names listed in the schema description); this keeps the
tool schema at about 4 KB instead of 36 KB with one nested model per field.

## Skills (`api/app/order_agent/agent_skills/`)

| Skill | Content |
|---|---|
| `order-classification` | STANDARD vs CONTRACT (agreements, promotions, call-offs) vs NOT_AN_ORDER (approval threads, price lists, queries); supplier pro forma wrapping a customer order; several orders in one message |
| `header-extraction` | Field definitions (`references/field-definitions.md`) and traps: PO number vs supplier numbers, PO date vs other dates, buyer only in the logo, tax id / phone are never the customer number, explicit ship-to only, e-mail body vs attachment conflicts, grand total vs page subtotal, local number formats, language rule |
| `line-item-extraction` | Line type and field definitions (`references/field-definitions.md`) and procedure: what is a line, charges vs materials, descriptions as printed, code forms (codes of other systems are hints only), RAL / NCS / tint colour codes, pack-size columns, unit words in several languages, volume-to-pieces conversion, pack conflicts |
| `master-data-matching` | Customer signal priority, ship-to only from partners, material order of the requirements (description, material number, colour, `/XA`), English retry search, when to flag instead of choosing |
| `confidence-and-fixes` | Confidence rubric, allowed and forbidden automatic fixes, what is (not) a review reason, reacting to `submit_order` issues |

Both extraction skills follow the same layout: `SKILL.md` holds the procedure and traps, and
`references/field-definitions.md` holds one table row per field (what the field means, what goes
into `value` and `raw`). The `load_skill` tool (`app/agent_core/skills.py`) returns `SKILL.md`
followed by every nested text file of the skill folder, so the reference tables reach the agent in
the same tool result without a separate file read.

## Language and printed text

Rule in the base prompt (`config/agent.yaml`), applied by every skill:

| Value | Stored `value` | Stored `raw` |
|---|---|---|
| Free text (description, names, addresses, payment terms text, notes) | exactly as printed (line breaks joined) | as printed |
| Vietnamese free text (requirement R3, slide "Data Extraction Requirements (1/2)") | English translation | Vietnamese original |
| Other languages (German, ...) | as printed, not translated | as printed |
| Structured values (dates, numbers, currency, units, Incoterms) | normalised | as printed |

Company names and addresses are never translated. Material searches may use an English
translation or expanded abbreviations when the printed text finds no candidate; that query text is
never stored. Whether languages other than Vietnamese must be translated is an open point with the
customer.

Example: printed description `43038 BRAND 325U PKIT DISC 150MM P600 100PC 43038` is stored
unchanged; the agent may search with `pelkit disc 325U 150mm P600` on a retry.

## Confidence and the gate

**Sold-to and ship-to:** field confidence = min(agent confidence, deterministic match
confidence). Match confidence is recomputed by `orders/service.py` from the *printed* values
(buyer name/address/tax id/phone, delivery address) against master data, so the agent cannot
vouch for its own choice.

**Materials** (hard text rules are brittle when variants differ by one number, so the agent
judges and the code applies the policy):

| Step | Who | What |
|---|---|---|
| Candidates | `search_materials` | Up to 8 per line from material number, sales code, customer code, manufacturer part number, description, colour, pack; each with the customer price and a `price_check` |
| Choice | agent | Picks the candidate describing the same product and submits `material_evidence`: `matched` attributes, `conflicts`, `alternatives` |
| Confidence | `service.material_confidence` | Starts from the agent's confidence; caps it at 0.60 for conflicts, 0.75 when the printed price differs from the customer price by more than 10 %, 0.70 when the printed line hardly resembles the material text (score < 0.35); alternatives make the field ambiguous (review) |

The evidence is stored in `POAGENT_FIELD_STATE.EVIDENCE` (JSON) and shown in the review UI. An
empty material has confidence 0 and only the "could not be determined" error (no extra
low-confidence issue). Orders submitted without evidence keep the previous rule (deterministic
score, rival-based ambiguity).

```python
material_confidence(meta, text_score=0.95, price_text="Printed price 49.00 matches ...", price_differs=False)
# meta["CONFIDENCE"] == agent confidence, meta["AMBIGUOUS"] == bool(evidence["alternatives"])
```

Below 0.80 a posting-relevant field (PO number/date, sold-to, ship-to, sales org, currency,
material, quantity, unit, colour) needs review, with the reason in the issue text; other fields
only get a warning. The full rule set is in `orders/validation.py` (see
[order-processing-pipeline.md](order-processing-pipeline.md)).

## Related files

| File | Role |
|---|---|
| `api/app/agent_core/` | Vendored template runtime (changed: `__init__` without A2A import; `providers.py` drops reasoning blocks, `store=False`) |
| `api/app/order_agent/config/agent.yaml` | Model, prompt, runtime settings |
| `api/app/order_agent/agent_skills/*/SKILL.md` | Skills |
| `api/app/order_agent/schemas.py` | `OrderSubmission`, `FieldValue`, field-name to column maps |
| `api/app/order_agent/tools.py` | Tools |
| `api/app/order_agent/prompt.py` | First message and native attachments |
| `api/app/order_agent/runner.py` | Shared runtime, run with timeout, token usage events, run record |
| `api/app/orders/service.py` | Submission -> rows, match confidence, evaluation, final gate |
| `api/app/orders/validation.py` | Deterministic rules and system fills |
| `api/app/orders/quantity.py` | Unit conversion |
| `api/app/cli/run_samples.py` | End-to-end runner for sample files |
| `tests/unit/test_provider_reasoning.py` | Reasoning blocks dropped, other blocks kept |

## How to test

```bash
cd api
../.venv/bin/python -m unittest discover -s ../tests/unit -t ..
```

* `test_order_agent.py`: a scripted chat model drives the real runtime; asserts the tool set and
  that `submit_order` receives the order id from the graph state.
* `test_validation.py`, `test_quantity.py`: the deterministic rules.

Live (Gen AI Hub + HANA credentials in `api/.env`, seed loaded):

```bash
../.venv/bin/python -m app.cli.run_samples "../data/orders/PO to MRY on the 20-12-2024.pdf" --verbose
../.venv/bin/python -m app.cli.run_samples ../data/orders --concurrency 4
```

The CLI prints per order: status, classification, PO number, resolved sold-to/ship-to, number of
lines, duration, LLM calls and input/cached/output tokens, and the blocking issues.
