# Manufacturing-feasibility token and cost estimate per drawing

## Scope

This estimate covers only the manufacturing-feasibility drawing extraction performed by `POST /api/drawing-feasibility/assessments`. It excludes ECN chat, summary generation, deterministic PS4 calculations, report rendering, retries, and orchestration services.

The current workflow makes one billable SAP Generative AI Hub Responses call per successfully assessed drawing. PS4 evaluation and report generation after extraction are deterministic and consume no LLM tokens.

## Result at a glance

For steady-state planning with the measured `gpt-5.4` deployment:

- **Average inclusive input:** 19,477 tokens per drawing
  - **Uncached input:** 1,493
  - **Cache-read input:** 17,984
- **Average output:** 3,246 tokens
- **Average grand total:** 22,723 tokens
- **Average calculator estimate:** **0.0669 CU per drawing**
- **Average assuming no input tokens are cached:** **0.1158 CU per drawing**
- **Recommended planning value:** **0.08 CU per drawing** for normal warm-cache operation
- **Cold-cache observation:** **0.1284 CU**, rounded to **0.13 CU**, for the measured first request

All four grand token totals are within **±6%** of the 22,723-token average. Cost does not have the same narrow range because output tokens have the highest rate and varied with drawing complexity.

## Measurement method

Measurements were taken on 2026-10-27 through the unchanged production extraction provider using:

- SAP Generative AI Hub model: `gpt-5.4`
- Endpoint: Responses API
- Pricing mode: cached
- `store=False`
- One warm-up request followed by one measured request for each fixture
- No retries; all calls succeeded and returned provider usage telemetry

The four measured source files were:

1. `QKX5632660_2_AFS.pdf`
2. `QKX5525860_2_AFS.pdf`
3. `QKX3896510_2_IW_old.pdf`
4. `QKX0971490_2_IW_old.pdf`

`Input total` is inclusive input: uncached input plus cache-read input. No cache-write count was reported by the provider.

## Per-drawing results

| Drawing | Uncached input | Cache read | Input total | Output | Grand total | Total vs average | Estimated CU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `QKX5632660_2_AFS.pdf` | 275 | 19,328 | 19,603 | 3,791 | 23,394 | +3.0% | 0.0731 |
| `QKX5525860_2_AFS.pdf` | 1,759 | 17,536 | 19,295 | 2,176 | 21,471 | -5.5% | 0.0488 |
| `QKX3896510_2_IW_old.pdf` | 1,969 | 17,536 | 19,505 | 3,629 | 23,134 | +1.8% | 0.0749 |
| `QKX0971490_2_IW_old.pdf` | 1,968 | 17,536 | 19,504 | 3,389 | 22,893 | +0.7% | 0.0707 |
| **Average** | **1,492.75** | **17,984** | **19,476.75** | **3,246.25** | **22,723** | — | **0.0669** |

The four averages reconcile exactly:

```text
1,492.75 uncached input
+ 17,984.00 cache-read input
+  3,246.25 output
= 22,723.00 total tokens per drawing
```

### Variation

| Bucket | Observed range | Deviation from average |
|---|---:|---:|
| Inclusive input | 19,295–19,603 | -0.9% to +0.6% |
| Output | 2,176–3,791 | -33.0% to +16.8% |
| Grand total | 21,471–23,394 | -5.5% to +3.0% |
| Estimated cost | 0.0488–0.0749 CU | -27.0% to +12.0% |

Inclusive input is very stable across these drawings. Output is the main uncertainty and is model- and drawing-dependent. For that reason, use **±6% only as the observed token-total margin**, not as a guaranteed cost margin.

## Cost calculation

The estimate uses the external mode of the [SAP AI Core Cost Calculator](https://ai-core-calculator.cfapps.eu10.hana.ondemand.com/) for `gpt-5.4`, version `2026-03-05`, Pricing Mode `Cached`. The rates and formula below were captured on 2026-10-27 from calculator build `1793099807312`.

Calculator values captured at measurement time:

| Token bucket | Calculator rate per 1,000 tokens |
|---|---:|
| Standard/uncached input | 0.00159 |
| Read-cached input | 0.00016 |
| Output | 0.00920 |
| Capacity-unit multiplier | 1.90385 |
| Cache write | 0 for this calculator model/version |

Formula:

```text
CU per drawing = 1.90385 × (
    uncached_input / 1,000 × 0.00159
  + cache_read_input / 1,000 × 0.00016
  + output / 1,000 × 0.00920
)
```

Average-cost calculation:

```text
Uncached input: 1,492.75 / 1,000 × 0.00159 × 1.90385 = 0.00452 CU
Cache read:    17,984.00 / 1,000 × 0.00016 × 1.90385 = 0.00548 CU
Output:         3,246.25 / 1,000 × 0.00920 × 1.90385 = 0.05686 CU
                                                                    --------
Average                                                            0.06686 CU
```

Output represents about 85% of the mean cost even though it is only about 14% of total tokens. Raw grand-token counts therefore must not be multiplied by one blended rate.

### Cost assuming no input tokens are cached

For a conservative counterfactual, the measured `input_total_tokens` can be priced entirely at the standard/uncached input rate. This assumes **zero cache reads** for every drawing while retaining each drawing's measured total input and output. It is a mathematical recalculation of the existing measurements; no additional model calls were made.

```text
No-cache CU per drawing = 1.90385 × (
    input_total / 1,000 × 0.00159
  + output / 1,000 × 0.00920
)
```

| Drawing | Input total priced as uncached | Assumed cache read | Output | No-cache estimated CU |
|---|---:|---:|---:|---:|
| `QKX5632660_2_AFS.pdf` | 19,603 | 0 | 3,791 | 0.1257 |
| `QKX5525860_2_AFS.pdf` | 19,295 | 0 | 2,176 | 0.0965 |
| `QKX3896510_2_IW_old.pdf` | 19,505 | 0 | 3,629 | 0.1226 |
| `QKX0971490_2_IW_old.pdf` | 19,504 | 0 | 3,389 | 0.1184 |
| **Average** | **19,476.75** | **0** | **3,246.25** | **0.1158** |

Average no-cache calculation:

```text
Standard input: 19,476.75 / 1,000 × 0.00159 × 1.90385 = 0.05896 CU
Output:          3,246.25 / 1,000 × 0.00920 × 1.90385 = 0.05686 CU
                                                                       --------
Average                                                               0.11582 CU
```

The no-cache average is about **73.2% higher** than the measured warm-cache average of 0.0669 CU. Across the four drawings, the no-cache estimate ranges from **0.0965 to 0.1257 CU per drawing**. This scenario is separate from the measured cold request, whose output was 3,940 tokens rather than the four-drawing mean.

The calculator labels results as estimates that do not constitute an SAP quote. Rates and model availability can change; refresh the calculator result before a commercial commitment.

## Cache interpretation

The measured average represents steady-state reuse of the large fixed policy/request prefix:

- Average cache-read rate: **92.3%** of inclusive input.
- The first cold request reported 19,603 uncached input, zero cache-read input, 3,940 output, and 23,543 total tokens.
- Applying the same calculator rates gives **0.1284 CU** for that cold observation, approximately 1.9 times the measured warm average.
- The first drawing was also the warm-up drawing. Its measured request received 19,328 cache-read tokens, versus 17,536 for the other three drawings. A validation call using the same bytes with a changed filename produced the same 19,328-token cache read, confirming strong content/prefix reuse rather than a telemetry error.

Do not assume the 92.3% cache-read rate for every isolated request. Cache reuse is provider-, deployment-, tenant-, request-shape-, and time-dependent. Use returned usage telemetry as the source of truth.

## Planning recommendation

Use two values rather than one false-precision estimate:

| Scenario | Planning value | Basis |
|---|---:|---|
| Normal sequential operation with observed prefix-cache reuse | **0.08 CU/drawing** | Above all four measured warm costs; about 20% over the 0.0669-CU mean |
| Cold-cache or isolated first request | **0.13 CU/drawing** | Rounded measured cold observation of 0.1284 CU |

Example warm-cache volume planning:

| Drawings/month | Measured-average estimate | Recommended budget at 0.08 CU/drawing |
|---:|---:|---:|
| 100 | 6.69 CU | 8 CU |
| 1,000 | 66.86 CU | 80 CU |
| 10,000 | 668.56 CU | 800 CU |

These figures cover successful single-call extraction only. Production budgeting should add an explicit retry/error allowance based on observed operational data. A 5% retry rate, for example, should be applied to the chosen warm/cold mix rather than hidden in the per-call token measurements.

## Limitations and refresh triggers

This is the best current estimate from four representative drawings, not a statistical production benchmark.

Refresh it when any of the following changes:

- model name or deployed model version;
- extraction prompt, policy bundle, or strict output schema;
- Responses request shape or PDF/image processing behavior;
- provider caching behavior;
- calculator rates or CU multiplier;
- representative drawing complexity or page count.

For a production baseline, capture at least 30 representative successful assessments and report percentiles for output tokens, CU cost, latency, cache-read rate, and retries.

## Reproduction inputs

For each call, aggregate one `btp.llm_usage.v1` event where:

- `route="/api/drawing-feasibility/assessments"`
- `stage="EXTRACTION"`
- `model="gpt-5.4"`
- `outcome="success"`

Use:

- `input_tokens` as standard/uncached calculator input;
- `cache_read_input_tokens` as read-cached input;
- `cache_write_input_tokens` as write-cached input when the calculator offers and prices it;
- `output_tokens` as output;
- `input_total_tokens` only as a reconciliation value, not as an additional billed bucket.
