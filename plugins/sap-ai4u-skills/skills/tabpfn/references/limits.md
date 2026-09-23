# TabPFN 3.5-plus — limits, sizing, and cost

Two sources, kept deliberately separate: what the SAP "Tabular AI on SAP" FAQ
states about the model, and what has been empirically validated against the live
`/predict` deployment. Don't conflate them — the FAQ describes model capability;
the deployment may impose its own per-call caps.

## FAQ-stated model limits (Tabular AI on SAP FAQ, Sept 2026, v1.0)

- **Scale:** up to **1,000,000 rows** and **200 features**.
- **Speed claim:** classifies a **10,000-row dataset in under 3 seconds**.
- **Checkpoint deployed:** **TabPFN Plus** ("balanced production model, strong
  accuracy across a broad range of dataset types", available today). TabPFN Fast
  (~4× faster, alpha) and TabPFN Thinking (highest accuracy) are **not yet
  available** in SAP AI Core.
- **Positioning:** #1 on TabArena (51 datasets) and BeyondArena; strong on
  text-rich, high-cardinality, high-dimensional, grouped/temporal data. General
  model — no SAP-native column semantics (that's SAP-RPT's edge) and no built-in
  row/column explainability (also SAP-RPT).
- **No training:** in-context learning; predictions returned with no fit step.
- **Also available** via Prior Labs API (free tier), AWS SageMaker, Azure ML.

## Empirically validated (this deployment, direct `/predict`)

- **`n_estimators` capped at 8** — the deployment rejects `>8`. This is *not* in
  the FAQ; it is a real per-call ceiling. See `settings.md`.
- **Task_config is strict** (unknown keys → 400); hyper-parameters must nest in
  `tabpfn_config`.
- **Confirmed working sizes:** up to ~4,000 context rows × ~12 features with 100
  query rows returned in ~1 s. Titanic/Boston/German (400–900 context) return in
  a few seconds. The 1M-row / 200-feature ceiling has **not** been exercised
  here, and no explicit per-call row/column cap has been observed in testing —
  but absence of a hit is not proof one doesn't exist. For large contexts,
  measure latency and watch for 400/413 before committing.
- **Missing values** are accepted as JSON `null` (no imputation needed).

## Cost model (FAQ)

Charged per **cell** = the unit of data processed per API call:

```
input cells   = context rows × context columns
predict cells  = query rows × predict columns
cost per call  = input cells + predict cells
```

The response's `usage.num_cells` is exactly this total. Cost scales with context
size × features, so trimming columns and context rows is the primary cost lever.

**FAQ worked example:** 8,192 context rows × 15 features = 122,880 input cells;
128 query rows × 2 predict columns = 256 predict cells; at ~47 calls/month
≈ **4.21 EUR/month**. Use the SAP AI Core Cost Calculator for current rates.

## Practical sizing guidance

- Keep context representative rather than maximal — TabPFN is in-context, so more
  context = more cells (cost) and more latency, with diminishing accuracy return.
- Batch query rows into one call where possible (fewer round trips), but note the
  billed cells are the same either way.
- If you need SAP-native column semantics, row/column-level explainability, or
  context beyond what fits comfortably, compare against SAP-RPT-1.6 (see the
  project's `docs/tabpfn-vs-rpt16-comparison.md`).
