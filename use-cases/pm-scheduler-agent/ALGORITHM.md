# Scheduling Algorithm

The FMI scheduler is a **deterministic greedy algorithm** — no machine learning or AI is involved in generating the schedule itself. AI (Claude) is used separately, after scheduling, to explain and analyze results in natural language.

---

## Step 1 — Filter Candidates

From all work orders in the dataset, only those that pass **every** filter below are considered for scheduling:

| Filter | Value |
|---|---|
| Plant | Selected by the planner |
| `WO_PHASE` | `Released` (when "Released Only" is enabled) |
| `PRIORITY` | `Medium` (P3) or `Low` (P4) |
| Work Center | Work centers selected by the planner |
| Date window | See below |

### Date window logic

Depends on the order type (`ORDER_TYPE_CODE`):

- **MN03** (urgent corrective): only included if `BASIC_START_DATE` falls **within** the selected week.
- **All other types**: included if `BASIC_START_DATE ≤ week_end` **AND** `BASIC_FINISH_DATE ≥ week_start` — i.e., any order whose planned execution window overlaps the selected week.

---

## Step 2 — Priority Scoring

Each candidate order receives a numeric score. **Lower score = higher scheduling priority.**

```
score = (order_type_rank × 0.4) + (criticality_rank × 0.3) + (urgency_rank × 0.3)
```

| Component | Logic | Values |
|---|---|---|
| Order type rank | MN03 = 1, all others = 2 | 1–2 |
| Criticality rank | Equipment criticality A=1, B=2, C=3 | 1–3 |
| Urgency rank | Days to `LATEST_EXECTN_FINISH_DATE`: ≤7 days=1, ≤30 days=2, >30 days=3 | 1–3 |

**Example:** MN03 order, equipment criticality A, deadline in 3 days:
```
score = (1 × 0.4) + (1 × 0.3) + (1 × 0.3) = 1.0  ← highest possible priority
```

**Example:** MN01 order, equipment criticality B, deadline in 60 days:
```
score = (2 × 0.4) + (2 × 0.3) + (3 × 0.3) = 0.8 + 0.6 + 0.9 = 2.3
```

Candidates are sorted by score (ascending) before scheduling.

---

## Step 3 — Greedy Capacity Allocation

For each work center, the **total available capacity** for the selected week is:

```
capacity_available = SUM(AVAILABLETIME × MANPOWER) over all days in the week
```

Where `AVAILABLETIME` = shift hours per day and `MANPOWER` = number of workers assigned. Both come from the Work Centers capacity data.

**Unit**: man-hours (h). Example: 10h shift × 6 workers = 60 man-hours/day × 7 days = 420 h/week.

The algorithm then iterates through sorted candidates (best score first):

```
for each order (sorted by score, ascending):
    work_hours = ACTIVITY_WORK_INVOLVE
                 (fallback to ACTIVITY_NORMAL_DURATION if zero)

    if work_hours ≤ remaining_capacity:
        → SCHEDULED   (deduct work_hours from remaining_capacity)
    else:
        → UNSCHEDULED / Deferred
```

Orders are never split — an order is either fully scheduled or fully deferred.

---

## Capacity vs. Work Units

Both sides use the same unit — **standard man-hours (STD)**:

- Capacity: `AVAILABLETIME × MANPOWER` → total man-hours the crew can deliver
- Orders: `ACTIVITY_WORK_INVOLVE` → total man-hours the operation requires

---

## Opportunity Orders

When a piece of equipment is taken offline for planned maintenance, the planner can pull forward P3/P4 orders on the **same equipment** that are currently planned for **future weeks**. These are found by:

1. Matching `EQUIPMENT_NO`
2. Filtering for `Priority = Medium or Low`, status `Ready to Schedule`
3. `BASIC_START_DATE > end of current week`

The planner reviews the list and manually selects which ones to add to the current schedule.

---

## Where AI Fits

The scheduling algorithm itself contains **no AI**. Claude (LLM) is invoked separately after a schedule is generated for:

| Feature | What it does |
|---|---|
| **Explain Schedule** | Summarizes why orders were scheduled or deferred in plain English |
| **Conflict Analysis** | Identifies overloaded work centers, critical orders deferred, scheduling risks |
| **Ask a Question** | Free-form Q&A about the generated schedule |
