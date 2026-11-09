# Ideation Agent — Test Suite

## Prerequisites

```bash
cd api
source .venv/bin/activate
# .env must contain: CSE_URL, CSE_KEY, AICORE credentials
```

## Quality Scorecard (`test_quality.py`)

Proves our technical decisions are correct via head-to-head preference judging.
Two conversations are generated for the same scenario (one with the technology, one without),
then a judge LLM picks which produced a more useful result. Position is randomized to
eliminate bias.

```bash
# Full scorecard (~30s)
python -m ideation_agent.tests.test_quality

# Statistical (3 runs, ~90s)
python -m ideation_agent.tests.test_quality --runs 10

# Include model comparison (~2 min)
python -m ideation_agent.tests.test_quality --all --runs 10

# Test a specific section in isolation
python -m ideation_agent.tests.test_quality --only stack   # Full Stack vs Naive
python -m ideation_agent.tests.test_quality --only models  # model comparison
python -m ideation_agent.tests.test_quality --only cse     # CSE enrichment
python -m ideation_agent.tests.test_quality --only perplexity  # Perplexity market research
python -m ideation_agent.tests.test_quality --only linkedin    # LinkedIn hiring intelligence
```

**What it tests:**

| Section | What it proves |
|---|---|
| Full Stack vs Naive | Combined system value (target: >70% win rate) |
| Model comparison (with --all) | GPT-4.1 vs GPT-5.4 vs Gemini 2.5 Flash |
| Per-Technology | Individual value of data-enrichment decisions |

**Technologies tested individually:**

| Flag | Technology | What it means |
|---|---|---|
| `cse` | CSE enrichment | Background company research injected into agent context |
| `perplexity` | Perplexity market research | Real-time web research (market trends, strategic signals) via Sonar models |
| `linkedin` | LinkedIn hiring intelligence | Hiring pattern analysis (tech stack, AI maturity, bottlenecks) from job postings |

**Output**: Results saved to `tests/results/quality/` as timestamped JSON + `latest_batch.json`.
The JSON includes a `report` field with the human-readable terminal output.

## What We Don't Test with Simulators

Conversation-structure decisions **cannot** be reliably A/B tested with a user simulator:

| Technology | Why not testable | How it's still validated |
|---|---|---|
| Two-tier strategy | Simulator gives perfect answers regardless of question structure | Combined value captured in Stack vs Naive |
| 5 questions (vs 3) | Simulator doesn't forget details — more questions can't help it | Combined value captured in Stack vs Naive |

These technologies are designed to guide **real humans** who are vague, forget context, or need
structured progression. A simulator doesn't replicate that behavior. Their combined value is
proven by the Stack vs Naive comparison (FULL_STACK has them enabled, NAIVE doesn't).

## Configurations Compared

**Full Stack** (production config — all technologies enabled):

| Setting | Value |
|---|---|
| Model | GPT-4.1 |
| CSE enrichment | Enabled |
| Perplexity market research | Enabled |
| LinkedIn hiring intelligence | Enabled |
| Two-tier strategy | Enabled |
| Max questions | 5 |
| Repair loops | Enabled |
| JSON mode | Enabled |

**Naive** (stripped baseline — same model, no engineering):

| Setting | Value |
|---|---|
| Model | GPT-4.1 |
| CSE enrichment | Disabled |
| Perplexity market research | Disabled |
| LinkedIn hiring intelligence | Disabled |
| Two-tier strategy | Disabled |
| Max questions | 3 |
| Repair loops | Disabled |
| JSON mode | Enabled |

## Interpreting Results

| Win Rate | Verdict | Meaning |
|---|---|---|
| ≥ 70% | PASS | System clearly outperforms baseline |
| 50-69% | MARGINAL | Improvement exists but not strong |
| < 50% | FAIL | Baseline wins more often |

For per-technology tests (CSE), >60% indicates the technology provides measurable value.

## How Per-Technology Testing Works

Each scenario has a `stresses` field indicating which technologies it exercises.
Per-technology tests only run against scenarios where that technology can show value:

| Scenario | Stresses | Why |
|---|---|---|
| Volvo - PDF procurement | `perplexity_problem` | Info-rich input, problem research adds context |
| Volvo - incoming quotations | `perplexity_problem` | Same — detailed input |
| Illumina - safety stock MEIO | `perplexity_problem` | Same — detailed input |
| Vague - invoice automation (Bosch) | `cse`, `perplexity`, `perplexity_problem`, `linkedin` | Company name + minimal detail — all enrichments add context |
| Vague - demand forecasting (Nestlé) | `cse`, `perplexity`, `perplexity_problem`, `linkedin` | Company name + minimal detail — all enrichments add context |
| Vague - talent onboarding (Siemens) | `cse`, `perplexity`, `perplexity_problem`, `linkedin` | Company name + minimal detail — all enrichments add context |

## Adding a New Technology

When you add a new **data-enrichment** capability (e.g., LinkedIn enrichment, Perplexity search):

1. **Add a field to `RunnerConfig`** in `eval/config.py`:
   ```python
   linkedin_enabled: bool = True
   ```

2. **Wire it in `runner.py`** — fetch/inject the data when enabled, skip when disabled.

3. **Add to `TECHNOLOGIES` list** in `test_quality.py`:
   ```python
   ("linkedin", "LinkedIn enrichment", RunnerConfig(linkedin_enabled=False)),
   ```

4. **Add scenarios that stress it** in `eval/scenarios.py` — include the tech key in `stresses`:
   ```python
   {"name": "Person context needed", "company": "Siemens",
    "message": "...", "stresses": ["linkedin", "cse"]},
   ```

5. **Run**: `python -m ...test_quality --only linkedin --runs 3`

The technology automatically becomes available as an `--only` target.

**Note:** Only add technologies whose value shows up in LLM output (data-enrichment).
Conversation-structure decisions (question flow, phasing) should be validated with
real users, not the simulator.

## Adding a New Scenario

Add to `SCENARIOS` or `SCENARIOS_STRESS` in `eval/scenarios.py`:

```python
{
    "name": "Short descriptive name",
    "company": "CompanyName",   # or "" if testing without company context
    "message": "The user's input message",
    "stresses": ["cse"],        # which data-enrichment technologies this exercises
}
```

Tagging rules for `stresses`:
- `cse` → add when company name is provided but message has little detail
- `perplexity` → add when company name is provided but message has little detail (market context adds value)
- `linkedin` → add when company name is provided but message has little detail (hiring patterns reveal tech stack and bottlenecks)
- Future enrichment tech → add when the scenario requires that technology's data

## LinkedIn Rate Limiting

LinkedIn's guest API enforces per-IP rate limits. After multiple test runs in quick succession, you may get 429 responses and see 0 or very few positions returned. This is temporary — wait some time (typically minutes to hours) for the limit to reset. The production server is unaffected since it only makes one request per conversation.

If you see `rows_analyzed: 0` or very low counts during local testing, this is the most likely cause.

## Check Model Deployments (`check_models.py`)

Verifies which LLM models are deployed and reachable in the AI Core proxy.

```bash
python -m ideation_agent.tests.check_models
```

## Unit Tests

```bash
python -m pytest ideation_agent/tests/test_response_contract.py -v
python -m pytest ideation_agent/tests/test_agent.py -v
python -m pytest ideation_agent/tests/test_router.py -v
```

## Integration Tests

```bash
# CSE company search (requires CSE_URL + CSE_KEY)
python -m pytest ideation_agent/tests/test_company_search_integration.py -v
python -m pytest ideation_agent/tests/test_company_search.py -v

# Refinement flow (~60s, requires AICORE credentials)
python -m ideation_agent.tests.test_refinement
python -m ideation_agent.tests.test_refinement --verbose
```

### Refinement Tests (`test_refinement.py`)

Verifies the post-final refinement flow with real LLM calls:

| Test | What it proves |
|---|---|
| Go deeper triggers clarification | Broad/ambiguous feedback → agent asks which aspect to focus on |
| Specific feedback triggers refinement | Actionable feedback → agent acknowledges conversationally |
| Update report produces final | `[ACTION:UPDATE_REPORT]` → updated report incorporating feedback |

## Environment Overrides

| Variable | Default | Purpose |
|---|---|---|
| `AB_EVAL_PROVIDER` | `google` | Judge/simulator LLM provider |
| `AB_EVAL_MODEL` | `gemini-2.0-flash` | Judge/simulator model |

## Architecture

```
tests/
├── test_quality.py          # Quality scorecard (main entry point)
├── test_refinement.py       # Refinement flow integration tests
├── check_models.py          # Model availability checker
├── eval/                    # Shared evaluation infrastructure
│   ├── config.py            #   RunnerConfig dataclass
│   ├── scenarios.py         #   Test scenario definitions (with stresses tags)
│   ├── runner.py            #   Conversation runner + user simulator
│   └── judge.py             #   Head-to-head preference judge
├── eval_output.py           # Result persistence (JSON)
├── results/
│   └── quality/             #   Quality scorecard results
├── test_response_contract.py  # Unit: response parsing
├── test_agent.py              # Unit: agent turn logic
├── test_router.py             # Unit: API router
├── test_company_search.py     # Integration: CSE
└── test_company_search_integration.py
```
