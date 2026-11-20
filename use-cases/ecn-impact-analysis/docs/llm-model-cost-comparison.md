# LLM model cost comparison: `gpt-5.4` vs `gpt-5.6-luna`

## What this document is

A measured comparison of the LLM token consumption and estimated cost of one full ECN impact analysis run on two SAP AI Core text-generation deployments:

- `gpt-5.4` (the default when these measurements were taken; `DEFAULT_LLM_MODEL` in `api/app/ecn/settings.py` is now `gpt-5.6-luna`)
- `gpt-5.6-luna`

The same ECN and BOM source files were analysed once per model through the normal UI flow. Token counts come from the backend's structured `llm_usage` log lines (`api/app/observability/llm_usage_logging.py`). Cost figures come from the SAP AI Core pricing calculator ("Foundation Models" section, Pricing Mode "Cached"), with the measured token counts typed in as one request per month.

This is a single-run comparison, not a benchmark. See "Limitations" before using the numbers for planning.

## Result at a glance

| Metric (one ECN analysis) | `gpt-5.4` | `gpt-5.6-luna` |
|---|---|---|
| LLM calls | 9 | 14 |
| Uncached input tokens (includes cache writes) | 197,366 | 256,740 |
| Cache write tokens (subset of uncached input) | 0 | 44,238 |
| Cache read tokens | 4,352 | 154,194 |
| Input total (uncached + cache read) | 201,718 | 410,934 |
| Output tokens | 3,900 | 11,003 |
| Grand total tokens | 205,618 | 421,937 |
| Cache hit rate (read / input total) | 2.2% | 37.5% |
| Sum of call latencies | 58.7 s | 116.3 s |
| Calculator cost at 1 request/month | 0.67 CU/mo | 0.10 CU/mo |

The calculator's "Foundation Models (2)" group total for both models together was 0.76 CU/mo (0.67 + 0.10 = 0.77 before rounding of the two line items).

Luna processed about twice as many tokens but is estimated at roughly one seventh of the cost per ECN (0.10 vs 0.67 CU), because most of its extra input was cache reads, which the calculator prices far below standard input.

## How the measurements were taken

1. The backend was started locally from `api/` with the same command Cloud Foundry uses (`uvicorn app.main:app`), with stdout teed to a log file.
2. For the `gpt-5.6-luna` run the model was selected with the environment variable `LLM_MODEL=gpt-5.6-luna` on the start command. `app/main.py` loads `.env` with `load_dotenv` without `override`, so a shell value takes precedence and `.env` was not edited.
3. The ECN was uploaded and processed through the UI. Each LLM call emits one JSON line (`schema_version: btp.llm_usage.v1`) with `stage`, `model`, `input_tokens`, `cache_read_input_tokens`, `cache_write_input_tokens`, `input_total_tokens`, `output_tokens`, `total_tokens` and `latency_ms`.
4. Per-call rows were aggregated per `job_id`.

### The model is pinned per job at upload time

The analysis model is written onto the ECN job in HANA when the upload is published (`model_name=self._settings.llm_model_name` in `api/app/ecn/services/ingestion.py`), and the analysis repository reads the model only from that stored job (`api/app/persistence/analysis_repository.py`). Restarting the backend with a different `LLM_MODEL` and re-running an existing ECN therefore keeps using the old model. The first luna attempt silently reused the old `gpt-5.4` job (same `job_id`, every log row said `gpt-5.4`) and had to be discarded. To compare models, delete the existing ECN and re-upload it after the backend restart, then check that every row of the new `job_id` shows the expected `model`.

`LLM_TIMEOUT_FALLBACK_MODEL` in `api/app/ecn/settings.py` is hard-coded to `gpt-5.4`. A luna call that times out is retried on `gpt-5.4`; the `model` field of the log line would show it. No fallback occurred in the luna run reported here (all 14 rows are `gpt-5.6-luna`).

## Token measurements per call

Columns: `Uncached in` includes cache-write tokens; `Input total` = `Uncached in` + `Cache read`.

### `gpt-5.4` (job `3714ad3f-19e2-404d-953f-2eb19e7d3a7b`, first run)

| # | Stage | Uncached in | Cache write | Cache read | Input total | Output | Total | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | DISCOVERY_1 | 1,761 | 0 | 0 | 1,761 | 181 | 1,942 | 2,988 |
| 2 | DISCOVERY_2 | 681 | 0 | 1,792 | 2,473 | 202 | 2,675 | 3,139 |
| 3 | DISCOVERY_3 | 6,855 | 0 | 2,560 | 9,415 | 87 | 9,502 | 2,271 |
| 4 | DISCOVERY_4 | 9,546 | 0 | 0 | 9,546 | 4 | 9,550 | 1,041 |
| 5 | PROCUREMENT | 41,995 | 0 | 0 | 41,995 | 358 | 42,353 | 6,749 |
| 6 | MANUFACTURING | 41,996 | 0 | 0 | 41,996 | 446 | 42,442 | 8,893 |
| 7 | CONFIGURATION | 41,994 | 0 | 0 | 41,994 | 387 | 42,381 | 6,370 |
| 8 | DATA_READINESS | 3,479 | 0 | 0 | 3,479 | 316 | 3,795 | 5,245 |
| 9 | SUMMARY | 49,059 | 0 | 0 | 49,059 | 1,919 | 50,978 | 21,993 |
| | **Total** | **197,366** | **0** | **4,352** | **201,718** | **3,900** | **205,618** | **58,689** |

### `gpt-5.6-luna` (job `759a797a-e07d-4173-988d-443709f3d2ae`)

| # | Stage | Uncached in | Cache write | Cache read | Input total | Output | Total | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | DISCOVERY_1 | 1,758 | 1,755 | 0 | 1,758 | 228 | 1,986 | 2,920 |
| 2 | DISCOVERY_2 | 1,024 | 1,021 | 1,755 | 2,779 | 343 | 3,122 | 3,171 |
| 3 | DISCOVERY_3 | 11,578 | 11,575 | 2,776 | 14,354 | 168 | 14,522 | 2,628 |
| 4 | DISCOVERY_4 | 863 | 860 | 14,351 | 15,214 | 240 | 15,454 | 2,811 |
| 5 | DISCOVERY_5 | 6,400 | 6,397 | 15,211 | 21,611 | 303 | 21,914 | 4,064 |
| 6 | DISCOVERY_6 | 1,193 | 1,190 | 21,608 | 22,801 | 308 | 23,109 | 3,165 |
| 7 | DISCOVERY_7 | 10,134 | 10,131 | 22,798 | 32,932 | 322 | 33,254 | 6,442 |
| 8 | DISCOVERY_8 | 9,840 | 9,837 | 32,929 | 42,769 | 307 | 43,076 | 2,991 |
| 9 | DISCOVERY_9 | 1,475 | 1,472 | 42,766 | 44,241 | 89 | 44,330 | 2,399 |
| 10 | PROCUREMENT | 48,929 | 0 | 0 | 48,929 | 1,849 | 50,778 | 18,553 |
| 11 | MANUFACTURING | 48,930 | 0 | 0 | 48,930 | 1,822 | 50,752 | 19,312 |
| 12 | CONFIGURATION | 48,928 | 0 | 0 | 48,928 | 1,173 | 50,101 | 12,529 |
| 13 | DATA_READINESS | 3,479 | 0 | 0 | 3,479 | 1,546 | 5,025 | 15,597 |
| 14 | SUMMARY | 62,209 | 0 | 0 | 62,209 | 2,305 | 64,514 | 19,743 |
| | **Total** | **256,740** | **44,238** | **154,194** | **410,934** | **11,003** | **421,937** | **116,325** |

## Cost calculation

Tool: SAP AI Core pricing calculator, section "Foundation Models", Pricing Mode "Cached" (cached reads cost significantly less, cache writes cost more). One calculator "request" represents one full ECN analysis, so the per-request token fields hold the summed totals of all LLM calls of that analysis, and "Requests per month" is 1.

### Calculator inputs and outputs

| Calculator field | GPT-5.4 | GPT 5.6 Luna |
|---|---|---|
| Input Tokens per request | 197,000 | 256,740 |
| Output Tokens per request | 3,900 | 11,003 |
| Read Cached Input Tokens per request | 4,352 | 154,194 |
| Write Cached Input Tokens per request | not offered for this model | 0 |
| Requests per month | 1 | 1 |
| **Estimated cost** | **0.67 CU/mo** | **0.10 CU/mo** |

The group total shown by the calculator for both models was 0.76 CU/mo.

Notes on how the inputs map to the log fields:

- "Input Tokens per request" was filled with the log's `input_tokens` sum (uncached input). For `gpt-5.4` it was rounded from 197,366 to 197,000.
- "Read Cached Input Tokens" is the sum of `cache_read_input_tokens`.
- "Write Cached Input Tokens" was left at 0 for luna even though the log reports 44,238 written tokens (see Limitations).

### Comparison

| | `gpt-5.4` | `gpt-5.6-luna` | Luna vs `gpt-5.4` |
|---|---|---|---|
| Cost per ECN analysis | 0.67 CU | 0.10 CU | about 0.15x (roughly 6.7x cheaper) |
| Total tokens per ECN | 205,618 | 421,937 | 2.1x |
| Cost per 1M total tokens (derived) | about 3.3 CU | about 0.24 CU | about 0.07x |
| Latency (sum of calls) | 58.7 s | 116.3 s | 2.0x |

Cost per 1M total tokens is derived as cost divided by total tokens and is only a rough indicator, because input, cached-read and output tokens are priced differently.

If the calculator scales linearly with "Requests per month" (not verified), 100 ECN analyses per month would be about 67 CU for `gpt-5.4` and about 10 CU for `gpt-5.6-luna`.

## Why the token profiles differ

- Discovery: luna made 9 sequential discovery calls versus 4 on `gpt-5.4` (a second `gpt-5.4` run made 3). Each call continues the previous prompt, so luna's cache reads grow from 0 to 42,766 tokens by call 9. All 44,238 cache-write tokens come from these calls.
- Specialists (PROCUREMENT, MANUFACTURING, CONFIGURATION): about 48.9k input tokens each on luna versus about 42.0k on `gpt-5.4`, and 1.2k to 1.8k output tokens versus 0.36k to 0.45k. None got cache reads on either model; they start within seconds of each other, so the first finished call cannot prime the cache for the others.
- SUMMARY: 62.2k input tokens on luna versus 49.1k on `gpt-5.4`. It was not checked whether the extra input comes from tokenizer differences or from the larger discovery context.
- Output: luna produced 2.8x more output tokens overall, which explains most of its latency increase.

## Limitations

- One run per model. A second `gpt-5.4` run on the same job varied by about 3% in total tokens (199,057 vs 205,618) but had 8 calls instead of 9 and 1,792 instead of 4,352 cache-read tokens. Discovery step counts are not stable between runs.
- Luna cache writes were entered as 0. The calculator states that write operations cost more than standard input, and 44,238 of luna's 256,740 uncached input tokens were cache writes. The 0.10 CU figure is therefore likely a lower bound. Re-run the calculator with 44,238 in "Write Cached Input Tokens" and, depending on how the calculator splits input, possibly 212,502 in "Input Tokens per request" (uncached minus writes) to refine it.
- The calculator's unit rates were not captured, only the resulting CU per model, so the per-token split of the cost is not derivable from this data.
- `gpt-5.4` input was rounded to 197,000 in the calculator (0.2% below the measured 197,366).
- The comparison covers cost and token volume only. Output quality of the two models (accuracy of the impact assessment, schema-conformance retries) was not evaluated.
- Results are for one ECN. Larger ECNs change the input size; the discovery step count and cache behaviour may not scale the same way for both models.

## Related files

- `api/app/observability/llm_usage_logging.py`: emits the `llm_usage` JSON lines used as the source of the token counts.
- `api/app/ecn/settings.py`: `DEFAULT_LLM_MODEL`, `LLM_TIMEOUT_FALLBACK_MODEL`, `LLM_MODEL` environment mapping.
- `api/app/ecn/services/ingestion.py`: pins `model_name` on the job at upload time.
- `api/app/persistence/analysis_repository.py`: derives the attempt's model from the stored job.
- `docs/ecn-model-timeout-fallback.md`: behaviour of the timeout fallback model.
- `docs/ecn-analysis-generation.md` and `docs/ecn-staged-impact-analysis.md`: the stages (DISCOVERY, PROCUREMENT, MANUFACTURING, CONFIGURATION, DATA_READINESS, SUMMARY) whose calls are counted here.

## How to reproduce

```bash
cd api
LLM_MODEL=gpt-5.6-luna ../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/ecn_backend_luna.log
```

Delete the existing ECN in the UI, re-upload it, wait for the analysis to finish, then aggregate the usage lines of the new `job_id`:

```bash
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open('/tmp/ecn_backend_luna.log') if l.startswith('{')]
for r in rows:
    print(r['job_id'][:8], r['stage'], r['model'], r['input_tokens'],
          r['cache_read_input_tokens'], r['cache_write_input_tokens'],
          r['output_tokens'], r['latency_ms'])
EOF
```

Confirm every row of the new job shows the intended `model` before trusting the totals.
