# Purpose

This document describes the rules, calculations, and decision checks used to evaluate customer credit for three use cases: new credit, update of terms and conditions, and credit exceptions. It is intended to be provided as system context to an LLM together with the JSON output produced by the `credit_policy_engine.py` script. The goal is that, when a check fails, the LLM understands precisely why and can explain next steps.

---

## Inputs and Core Concepts

### Inputs expected by the engine

* **CustomerInput**: customer\_id, legal\_name, persona (PF, PM), country, entity\_name or explicit group (A or B), cgv\_signed\_date, pagare\_signed, guarantors, insurance\_full\_credit.
* **DocsInput**: kyc\_date, seller\_comments\_present, address\_proof\_date, tax\_cert\_date.
* **CreditRequest**: use\_case (new, update, exception), requested\_amount, requested\_currency, requested\_terms\_days, last\_update\_date, current\_credit\_line, current\_credit\_currency.
* **Investigation**: mmr\_amount, mmr\_currency, legal\_risk (low, medium, high), external\_investigation\_date, onsite\_visit\_done.
* **BehaviorInput**: invoices list, has\_overdue\_invoices, advance\_purchases\_count, has\_active\_credit, exceptions\_in\_semester.
* **RoleContext**: role (analyst, coordinator).

### Customer groups

* **Group A** entities: Sample Foods S.A. de C.V., Sample Foods LLC, Sample Ingredients S.A. de C.V., Sample Gluten Free Ingredients S.A. de C.V., Sample Gluten Free Jalisco S. de R.L. de C.V.
* **Group B**: Sample Pastas de Occidente S.A. de C.V.
* If group not provided and entity not matched, default to Group A.

### Payment classification and scores

* **Per-invoice CP score** based on days late (days from due\_date to paid\_date; negative or 0 means paid before or on due date). Each invoice is labeled and scored:

  * Excellent: before due date or same day → 10
  * Good: Group A 1–5 days late, Group B 1–3 → 8
  * Regular: Group A 6–10, Group B 4–6 → 6
  * Poor: Group A 11–15, Group B 7–9 → 4
  * Critical: Group A ≥16, Group B ≥10 → 0
* **CA (Annual rating)**: for each year, average of that year’s invoice CP scores, scaled to a 0–100 percentage.
* **C3M (Last 3 months)**: average score of invoices with due\_date in the last 92 days, scaled to percentage. May be absent if no recent activity.
* **CH (Historical rating)**: weighted average of up to the last 4 years of CA with weights \[3, 6, 8, 10] from oldest to newest. Result is a percentage.
* **CAL (Customer class)** from CH percentage:

  * Excellent: ≥95
  * Good: \[80, 95)
  * Regular: \[65, 80)
  * Poor: \[50, 65)
  * Critical: <50 or not enough data.

---

## Universal prerequisites – Table D requirements

These apply to all use cases unless noted.

1. **Commercial investigation (MMR)**: Must exist and typically must be ≥ the credit requested (CS). If MMR < CS, approval must not exceed MMR unless escalated to the Director of Finance.
2. **Advance purchases or active credit**: At least 3 advance purchases, or the customer must already have an active credit line.
3. **Legal investigation**: Result must be low or medium risk.
4. **Pagaré and CGV** for Mexican customers:

   * **CGV** (Condiciones Generales de Venta) must be signed.
   * **Pagaré** may be omitted only if the credit is 100% insured.
  * For **Persona física** (PF) without insurance: pagaré required and guarantors by amount - < 620k MXN: 0 guarantors; 620k–<1.25M: 1; ≥1.25M: 2.
   * For **Persona moral** (PM) without insurance: pagaré required; aval(es) at credit department’s discretion; may omit if fully insured.

### Recency of documentation

* **KYC (FO-FIN-006)**: valid if issued within the last 24 months.
* **Seller comments (FO-FIN-007)**: present.
* **Address proof** and **Tax certificate**: each valid if issued within the last 3 months.

---

## Use case A – New credit

### Eligibility and checks

* **Table D** prerequisites must pass: commercial investigation OK (MMR present and usually ≥ CS), advance purchases or active credit, legal investigation low/medium, pagaré/CGV rules obeyed.
* **Role monetary caps** (by persona and currency):

  * Analyst: PF up to 620k MXN / 31k USD / 34k EUR; PM up to 1.05M MXN / 52k USD / 58k EUR.
  * Coordinator: PF up to 1.25M MXN / 62k USD / 69k EUR; PM up to 2.1M MXN / 105k USD / 138k EUR.
* **Terms authority**: Analyst up to 32 days; Coordinator up to 47 days. Anything beyond needs Director of Finance.

### Decision guidance

* If any of the following is true, escalate to Director of Finance: amount exceeds role cap, tenor exceeds role limit, or MMR < requested credit.

---

## Use case B – Update of terms and conditions

### Baseline eligibility

* **Performance**: CAL must be Regular or better from CH. If there were sales in the last 3 months, C3M must also be Regular or better (≥68%).
* **No overdue invoices**.
* **Cooldown**: at least 3 months since last update.
* **CGV** must be signed and updated when terms change.

### Increase limits

* **LA% (max increase percentage over current line)** depends on CAL, role, and whether the current line is above a minimum threshold VL:

  * VL thresholds: 320k MXN, 16k USD, 17k EUR.
  * Analyst LA%: Excellent 20%, Good 20%, Regular 15%, Poor 0%, Critical 0%.
  * Coordinator LA%: if current line ≤ VL then 100% for Excellent/Good/Regular; if current line > VL then 50% for Excellent/Good, 30% for Regular; 0% for Poor/Critical in either case.
* **Role monetary caps** by CAL for the requested line:

  * Analyst: Exc/Good up to 1.25M MXN (62k USD, 69k EUR); Regular up to 620k MXN (31k USD, 34k EUR); Poor/Critical 0.
  * Coordinator: Exc/Good up to 2.6M MXN (130k USD, 144k EUR); Regular up to 1.55M MXN (78k USD, 85k EUR); Poor/Critical 0.
* **Terms authority**: Analyst up to 32 days; Coordinator up to 47 days.

### Decision guidance

* Escalate to Director of Finance if requested line breaches LA% cap, exceeds role monetary cap, or requested tenor exceeds the role’s limit.

---

## Use case C – Credit exception

### Eligibility

* CAL must be Regular or better.
* No overdue invoices.
* At most 3 exceptions authorized per customer per semester.

### Exception amount limits

* Requested amount must be ≤ 100% above the current line (i.e., requested ≤ 2 × current line).
* **Absolute caps by group**:

  * Group A: up to 1.25M MXN, 62k USD, 69k EUR.
  * Group B: up to 720k MXN (no USD/EUR cap defined).
* **Role caps** for exceptions (requested amount):

  * Analyst: Exc/Good up to 620k MXN (31k USD, 34k EUR); Regular up to 310k MXN (16k USD, 17k EUR); Poor/Critical 0.
  * Coordinator: Exc/Good up to 1.25M MXN (62k USD, 69k EUR); Regular up to 620k MXN (31k USD, 34k EUR); Poor up to 310k MXN (16k USD, 17k EUR); Critical 0.
* **Terms authority** still applies by role for requested tenor (32-day limit for analyst, 47-day limit for coordinator).

### Decision guidance

* Escalate to Director of Finance when exception exceeds the absolute cap, breaches the role cap, or exceeds +100% over current line.

---

## Reinstatement after delinquency – Appendix C logic

Based on the maximum observed days late across invoices:

* **15–<30 days late**: new credit request, update investigation, optional on-site visit, include interest moratorium in CGV if missing.
* **30–<60 days late**: wait 6 months from last settlement while staying active with advance payments, new credit request, share current financial statements, optional on-site visit, include interest moratorium if missing.
* **60–<90 days late**: wait 12 months from last settlement while staying active with advance payments, new credit request, share current financial statements, mandatory on-site visit, include interest moratorium if missing.
* **≥90 days late**: credit reactivation inadmissible.

The engine returns a `late_payment_reinstatement` object summarizing the band, max days late, months since last settlement, requirements, and auto-validated checks (e.g., whether the waiting period appears satisfied).

---

## JSON output schema and interpretation

The engine returns a structured JSON. Key fields:

* `use_case`: "new" | "update" | "exception".
* `group`: "A" | "B".
* `as_of`: ISO timestamp used for calculations.
* `scores`:

  * `cp_by_invoice`: per-invoice items with `invoice_id`, `days_late`, `label` (Excellent, Good, Regular, Poor, Critical), and numeric `score`.
  * `CA_by_year_pct`: map from year to that year’s CA percentage.
  * `C3M_pct`: last 3 months percentage, if applicable.
  * `CH_pct`: historical percentage across up to 4 years.
  * `CAL`: overall class inferred from CH (Excellent, Good, Regular, Poor, Critical).
* `checks`:

  * `table_d`: universal prerequisites

    * `commercial_investigation`: ok, reason
    * `advance_purchases_or_active`: ok, reason
    * `legal_investigation`: ok, reason
    * `pagare`: ok, reason
    * `cgv_signed`: ok, reason
  * `docs`: document recency

    * `kyc`, `address_proof`, `tax_cert`, `seller_comments`: ok, reason
  * `new_credit` (when `use_case` = new):

    * `within_role_max`: ok, cap, reason
    * `terms_authority`: ok, cap\_days, reason
  * `update_terms` (when `use_case` = update):

    * `eligibility`: map of checks: `cal_regular_or_better`, `c3m_regular_or_better`, `no_overdue`, `last_update_ge_3m`
    * `la_caps`: ok, pct\_cap, max\_allowed, reason
    * `within_role_max`: ok, cap, reason
    * `terms_authority`: ok, cap\_days, reason
  * `credit_exception` (when `use_case` = exception):

    * `eligibility`: `cal_regular_or_better`, `no_overdue`
    * `exception_caps`: `no_overdue`, `overage_le_100pct`, `absolute_cap`, `max_3_per_semester`, `role_cap`
* `decision_hint`: heuristic flags like `needs_director` and guidance notes.
* `late_payment_reinstatement`: see above.

### Failure interpretation pattern for the LLM

When a check returns `ok: false`, build messages like:

* **What failed**: e.g., "MMR 500,000 < requested 800,000"
* **Why it matters**: violates Table D prerequisite or exceeds role limit
* **What to do**: propose alternatives such as reduce requested amount to cap, escalate to Director, wait required months, update documents, or include missing clauses in CGV.

---

## Manual policy controls the engine does not fully enforce

Provide these as narrative context when relevant:

* **Investigation cadence**: investigations are valid for 1 year and limited to 2 per client per year. The engine does not currently reject stale investigations; analyst must verify recency and count.
* **On-site visit triggers**: if external investigation not favorable or the legal entity is younger than 2 years; the engine exposes `onsite_visit_done` but does not auto-trigger decisions.
* **CGV update for updates**: the engine checks CGV presence but does not explicitly verify that CGV was re-signed upon term changes.

---

## Known alignment gaps and recommended fixes

* **PF guarantor rule bug**: logic for PF < 620k should allow 0 guarantors without error. Ensure conditions are coded as: `<620k: ≥0`, `620k–<1.25M: ≥1`, `≥1.25M: ≥2`.
* **CGV recency on updates**: add a check that CGV was updated and signed when terms change.
* **Investigation recency**: enforce that both external and legal investigations are not older than 12 months and track the per-year maximum of 2 investigations.
* **Entity name normalization**: consider normalizing punctuation in entity names to ensure correct group classification.

---

## Example narratives the LLM can produce

* **New credit approved by analyst**: "Approved 820k MXN for PF at 32 days. MMR ≥ request, PF pagaré with 1 guarantor satisfied, documents valid."
* **Update denied due to LA%**: "Requested increase from 400k to 720k MXN exceeds LA% 50% cap for coordinator with CAL=Good and current line > VL; max allowed is 620k. Consider 620k or escalate."
* **Exception partially approved**: "Requested 1.5M MXN exceeds Group A absolute cap 1.25M and analyst cap 620k; coordinator may authorize up to 1.25M subject to no overdue and not exceeding +100%.
  "
