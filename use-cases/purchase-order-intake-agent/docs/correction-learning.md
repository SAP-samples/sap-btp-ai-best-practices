# Correction learning (long-term agent memory)

## What it is

When a reviewer corrects a value on the review page and saves, the change is stored in
`POAGENT_CORRECTION_LOG` together with what was printed on the document and what the agent had
chosen. Later runs use these corrections as precedent:

* **Materials** - for the same sold-to, a printed code the reviewer mapped to a material is
  matched to that material with score 0.98 (`matched_on: learned`), in the agent's
  `search_materials` tool and in the deterministic match confidence of the validator.
* **Customers** - a printed buyer name the reviewer assigned to a customer number is matched to
  that customer with 0.98 in `search_customers` and in the match confidence.
* **All fields** - `get_customer_profile` returns the last 20 corrections for the customer, so the
  agent sees how reviewers handled this customer before (the `master-data-matching` skill tells it
  to follow them for the same printed values).

This is the "corrections train model" box of the target architecture, implemented as retrieval
(precedent) instead of model training: it takes effect immediately, is explainable and can be
reverted by deleting log rows.

## What it is for

Recurring customers print the same codes and names every time (their own article codes, codes
without a prefix, abbreviations). After one correction the next order from that customer resolves
without review.

## Inputs and outputs

* Input: `PUT /api/orders/{id}` (Save) or `POST /api/orders/{id}/confirm` with edits. Every changed
  header or line field produces one row: `ORDER_ID`, `ITEM_NO`, `FIELD_NAME`, `SOLD_TO` (after the
  edit), `RAW_VALUE` (printed), `AGENT_VALUE`, `USER_VALUE`, `CREATED_AT`. Unchanged values are
  not logged.
* Output: learned mappings (latest correction wins):
  * `repository.learned_materials(sold_to)` -> `{compact printed code: MATNR}`
  * `repository.learned_customers()` -> `{normalised printed name: KUNNR}`
  * `repository.recent_corrections(sold_to)` -> last 20 corrections for the agent.

Precedence is unchanged otherwise: an exact material number or sales code (1.00) and the
customer-material info record (0.99) still outrank a learned mapping (0.98). ECC master data
(`KNMT`) is never written by the app.

## Related files

| File | Role |
|---|---|
| `api/app/orders/service.py` | `save_user_edits` collects corrections and writes them in the same transaction; match confidence uses learned maps |
| `api/app/orders/repository.py` | `insert_corrections`, `learned_materials`, `learned_customers`, `recent_corrections` |
| `api/app/orders/matching.py` | `learned` parameter of `rank_customers` / `score_material`, `name_key` |
| `api/app/order_agent/tools.py` | Tools pass the learned maps; profile returns recent corrections |
| `tests/unit/test_matching.py` (`LearnedMappingTests`), `tests/unit/test_corrections.py` | Tests |

## Key code

```python
if learned and learned.get(compact_id(variant)) == material["MATNR"] and 0.98 > best:
    best, matched_on, notes = 0.98, ["learned"], []
```

## How to test

```bash
cd api && ../.venv/bin/python -m unittest discover -s ../tests/unit -t ..
```

Live: open an order whose line could not be resolved, set the material, Save, then Reprocess. The
new agent run resolves the line to the same material and its trace shows `"matched_on": ["learned"]`.

Verified: a line printed `YEW F236` (no material in master data) was mapped by the reviewer; the log
held `YEWF236 -> 5574113` for that sold-to; after Reprocess the agent resolved the line to
`5574113` via the learned match, and the reviewer note reached the agent's prompt.
