# Production Wheel Optimization Workspace

A production wheel is the recurring cycle in which a plant fills its finished
products. Finished products (FINIs) that share a semi-finished bulk (SEFI) are
grouped into subgroups that are produced together. Large groups lower inventory
coverage per product but add changeovers inside the group; small groups do the
opposite.

This workspace redesigns those subgroups. For every plant/SEFI block it builds
candidate groups, solves a mixed-integer model with Pyomo and HiGHS, and returns
a Pareto frontier that trades demand-weighted inventory coverage against an
expected within-group changeover proxy (`J_CH`, shown as **Changeover** in the
UI). Every result is independently re-validated before it is shown.

It is **not** a detailed production scheduler. It does not choose production
dates, orders, physical lines, tanks, labor, cleaning sequences or capacity, and
it performs no write-back to SAP.

Main capabilities:

- **Datasets.** Upload a production workbook and an optional enrichment workbook.
  Extraction is deterministic, records source rows and hashes, and produces an
  immutable dataset in SAP HANA that you review and publish.
- **Runs.** Configure a run (scope, group-size cap, compatibility mode, typed
  constraints), launch it, and follow it while an independent worker solves it.
  Results, retained options and lineage are stored in HANA.
- **Analytics.** Compare frontier points, inspect groups and members, review
  matrix exceptions, and export the proposed wheel.
- **Assistant.** A LangGraph ReAct agent on SAP Generative AI Hub prepares and
  validates run drafts from plain-language requests, launches them on explicit
  instruction, and explains results. It cannot run SQL, Python or arbitrary
  solver flags.

## Architecture

| Component | Path | Technology |
|---|---|---|
| UI | `ui/` | Vite, UI5 Web Components |
| API | `api/app/` | FastAPI; workspace services in `api/app/workspace/`, agent in `api/app/agent/` |
| Optimizer worker | `api/app/workspace/worker.py` | Runs queued jobs; embedded in the API locally, a separate app on Cloud Foundry |
| Optimizer | `api/production_wheel/` | Pyomo + HiGHS (no commercial solver license), also usable offline through a CLI |
| Storage | SAP HANA Cloud | `PRODUCTION_WHEEL_*` tables, created and validated on startup |
| LLM | SAP Generative AI Hub | Model configured in `api/app/agent/config/agent.yaml` |
| Logging | SAP Cloud Logging | Token-usage logging in `api/app/observability/` |

The table and column contract lives in `api/app/workspace/columns.json`. The
runtime creates missing tables in the HANA user's current schema and refuses to
start when an existing table does not match the contract.

## Demo data

`data/anonymized/` holds two anonymized workbooks with fake material numbers,
plant codes, descriptions and scaled volumes:

| File | Role |
|---|---|
| `THDP - product groups for wheel.xlsx` | Production (primary) workbook for plant `THDP` |
| `Enrichment - production wheel assessment - INPUTS.xlsx` | Optional enrichment workbook (pallets, forecasts, network) |

Extraction reads Excel tables by name, not sheet position. Plant-specific table
names use a `{plant}` placeholder (for example
`PCK_per_filling_line___{plant}_specific`), so workbooks for other plants work
without code changes.

## Prerequisites

- Python 3.12
- Node.js 20.19 or later (required by Vite 7)
- An SAP HANA Cloud instance and a database user that can create tables
- An SAP AI Core instance with a Generative AI Hub deployment of the model named
  in `api/app/agent/config/agent.yaml` (only needed for the assistant)
- For deployment: the Cloud Foundry CLI and a `Cloud Logging` service instance in
  the target space

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r api/requirements.txt
cp api/.env.example api/.env
cp ui/.env.example ui/.env
npm --prefix ui install
```

Fill in `api/.env`:

| Variable | Meaning |
|---|---|
| `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP` | SAP AI Core service key values |
| `HANA_ADDRESS`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD`, `HANA_ENCRYPT` | SAP HANA Cloud connection |
| `HANA_SSL_VALIDATE_CERTIFICATE` | Optional, default `true` |
| `API_KEY` | Shared key the UI sends in the `X-API-Key` header |
| `LOG_USER_HASH_SALT` | Optional salt for the hashed user id in token-usage logs |

In `ui/.env`, set `VITE_API_KEY` to the same value as `API_KEY`, and
`VITE_API_BASE_URL` to the API address (default `http://127.0.0.1:8000/`).

Check the HANA connection and create the tables:

```bash
cd api
../.venv/bin/python -m app.workspace check-hana
```

## Run locally

API (with the embedded optimizer worker):

```bash
cd api
../.venv/bin/python -m app.main
```

UI, in a second terminal:

```bash
npm --prefix ui run dev
```

Open `http://localhost:5173`. The interactive API documentation is at
`http://127.0.0.1:8000/docs`.

Outside `APP_ENV=production` the API starts the worker in-process. Set
`WORKSPACE_EMBEDDED_WORKER=false` to run it separately:

```bash
cd api
../.venv/bin/python -m app.workspace worker
```

## Using the workspace

1. **Datasets page.** Give the snapshot a name, upload
   `data/anonymized/THDP - product groups for wheel.xlsx` as the production
   workbook and the enrichment workbook as the optional one, then select
   **Upload and extract**. Review the extraction evidence and quality issues, then
   select **Publish dataset**.
2. **Workspace page.** Open the published dataset, choose a plant profile and
   scope, adjust settings, and launch a run. The worker keeps running when the
   browser disconnects.
3. **Results.** Compare frontier points, inspect groups, and export the wheel.
4. **Assistant.** Ask in plain language, for example:
   "Prepare and validate a draft scoped to plant THDP with a group cap of 3 and
   three frontier points. Do not launch it."

The assistant is also available from the command line. Each `ask` call is
independent, so include the dataset ID:

```bash
cd api
../.venv/bin/python -m app.agent skills-list
../.venv/bin/python -m app.agent ask \
  "List the published datasets, then prepare and validate a draft for dataset <dataset-id> scoped to plant THDP with a group cap of 3 and three frontier points. Do not launch it." \
  --context-id demo
```

Browser chat turns exist only in the current page session and are not written to
HANA.

### Constraints the assistant can apply

The assistant maps supported rules to a typed, validated `SolveRequest`. Requests
outside this vocabulary are declined instead of submitted.

| Rule | Meaning | Scope |
|---|---|---|
| solve scope | Restrict the run to specific plant/SEFI blocks | plant/SEFI |
| `fini_disposition` | Include or exclude a specific FINI | one FINI |
| `max_group_size` | Cap the maximum group size | per plant/SEFI |
| `group_size_relaxation` | Cap = `base_limit + max_excess` | per plant/SEFI |
| `must_link` | Force FINIs into the same group | 2+ FINIs |
| `cannot_link` | Forbid two FINIs from sharing a group | FINI pair |
| `fixed_pv` | Pin a block to one production version | per plant/SEFI |
| `allowed_pvs` | Restrict a block to a PV allowlist | per plant/SEFI |
| `required_lines` | Restrict a block to specific filling lines | per plant/SEFI |

`coverage_bound`, `freeze_assignment` and `volume_compatibility_override` are
defined in the schema but not compiled yet, so the assistant refuses them.

Some properties vary along the frontier rather than being constraints, for
example the singleton count, which falls as `J_CH` rises. Run the frontier and
filter the points ("report only the points with 100 to 150 singletons") instead
of adding them as constraints.

### Resetting the HANA workspace tables

After a change to `api/app/workspace/columns.json`, rebuild the tables. The script
lists every `PRODUCTION_WHEEL_*` table in the current schema and drops them only
with `--confirm`. Other tables are never touched. Dropping deletes all datasets,
runs, jobs and agent memory.

```bash
.venv/bin/python api/scripts/reset_workspace_tables.py            # dry run
.venv/bin/python api/scripts/reset_workspace_tables.py --confirm  # drop
```

Restart the API afterwards; it recreates the tables, and datasets are uploaded and
published again through the UI.

## Offline optimizer (CLI, no HANA, no LLM)

The optimizer can run directly on the workbooks. Run every command from `api/`.
Output paths below are relative to the use-case root; `prototype/output/` is
git-ignored.

### 1. Extract and validate the workbooks

```bash
cd api
../.venv/bin/python -m production_wheel.cli extract \
  --primary '../data/anonymized/THDP - product groups for wheel.xlsx' \
  --enrichment '../data/anonymized/Enrichment - production wheel assessment - INPUTS.xlsx' \
  --output-root ../prototype/output \
  --run-id demo
WHEEL_RUN_DIR=../prototype/output/demo
```

Extraction writes canonical tables under `$WHEEL_RUN_DIR/extracted/` and a
`run_manifest.json` with source and output hashes, table ranges, row counts,
warnings and versions. Do not solve a run whose manifest status is not `ready`.

### 2. Run the greenfield Pareto frontier

A quick pass:

```bash
../.venv/bin/python -m production_wheel.cli solve \
  --run-directory "$WHEEL_RUN_DIR" \
  --frontier customer_families_greenfield_pareto \
  --frontier-points 5 \
  --per-block-total-seconds 30 \
  --output-directory "$WHEEL_RUN_DIR/solutions/greenfield-quick"
```

`--frontier-points`, `--per-block-total-seconds` and `--block-options-per-block`
drive quality. For a deeper run use, for example, 33 points, 120 seconds per
block, 10 options per block and `--block-workers 4`.

A scoped and constrained solve from a request file (the path the agent uses):

```bash
cat > /tmp/req.json <<'JSON'
{"config": {"scenario_id": "probe", "coverage_mode": "PARETO", "matrix_mode": "FLEXIBLE",
            "versions": {"matrix_version": "CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"}},
 "scope": [{"plant": "THDP"}],
 "constraints": [{"kind": "max_group_size", "constraint_id": "c1",
                  "scope": {"plant": "THDP"}, "maximum": 3,
                  "approval_status": "approved"}]}
JSON

../.venv/bin/python -m production_wheel.cli solve \
  --run-directory "$WHEEL_RUN_DIR" \
  --request /tmp/req.json \
  --frontier-points 4 --per-block-total-seconds 30 \
  --output-directory /tmp/probe-out
```

### 3. Validate the frontier bundle

```bash
../.venv/bin/python -m production_wheel.cli validate \
  --frontier-directory "$WHEEL_RUN_DIR/solutions/greenfield-quick"
```

Validation reports separately:

- `valid` / `integrity_valid`: files, hashes, exact cover, rules and KPIs are
  internally consistent;
- `complete`: the requested artifact bundle exists;
- `business_acceptance_assessed` and `business_acceptable`: `false` for
  greenfield points until an approved business policy exists.

A point can be structurally valid and still be `runtime-limited` or
`restricted-library`. Structural validation is neither business acceptance nor a
proof of optimality.

## CLI reference

Print every command with `../.venv/bin/python -m production_wheel.cli --help`.

### `extract`

```text
--primary PATH                 required production workbook
--enrichment PATH              optional enrichment workbook
--snapshot-expectations PATH   optional JSON of reviewed figures to enforce
                               (row counts, demand, excluded materials, pallet
                               fallbacks); absent keys are not checked
--output-root PATH             default: prototype/output
--run-id ID                    optional explicit run ID; otherwise generated
```

### `build-matrix`

Exports a versioned package-volume compatibility matrix as CSV for review.
`solve` uses the built-in versions directly.

```text
--matrix-version VERSION   BASELINE_EMPIRICAL_MATRIX_V1,
                           SYNTHETIC_VOLUME_MATRIX_V1 or
                           CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
--run-directory PATH       required for the empirical version
--output PATH              default: prototype/output/package_volume_matrix.csv
```

### `solve`

```text
--run-directory PATH       required completed extraction run
--suite demo               run the complete scenario suite
--scenario ID              run one named scenario instead
--frontier customer_families_greenfield_pareto
                           run the recommended coverage/J_CH frontier
--request PATH             typed SolveRequest JSON (PARETO frontier)
--frontier-points N        low-epsilon-biased global samples (default 17)
--global-epsilon-exponent N
                           density power; 1 uniform, >1 denser low (default 2)
--block-options-per-block N
                           maximum adaptive local requests (default 5)
--per-block-stage-seconds N
                           limit for every block objective tier
--per-block-total-seconds N
                           total time budget per block
--block-workers N          isolated block processes (default 2)
--large-block-candidate-threshold N
                           candidate count scheduled alone (default 250000)
--exhaustive-block PLANT/SEFI
                           repeatable block to enumerate exhaustively
--output-directory PATH    explicit new result directory
--per-stage-seconds N      scenario-wide budget per lexicographic tier
```

`--suite`, `--scenario` and `--frontier` are mutually exclusive. `--frontier` is
the recommended workflow; the suite is a long-running regression workflow.

### `validate`

Exactly one of `--scenario-directory`, `--suite-directory` or
`--frontier-directory`.

### `capabilities`

Prints the supported modes, matrix versions, defaults, guardrails and declared
future constraint kinds as JSON.

## Named scenarios

All scenarios share the fixed-PV, cap-seven, empirical-diagnostic model unless
the row says otherwise.

| Scenario ID | Meaning |
|---|---|
| `core_max` | Minimize the single worst group coverage |
| `core_demand_weighted_mean` | Minimize coverage weighted by demand litres |
| `core_group_mean` | Minimize the equal-weight mean across groups |
| `core_target_band` | Minimize violation count, worst excess, total excess, then weighted mean |
| `core_operations_first` | Minimize violation count, worst excess, total excess, then `J_CH` |
| `customer_families_hard_target` | Target band with operational-family `AVOID` pairs rejected |
| `customer_families_flexible_target` | Target band with `AVOID` pairs admitted and reported |
| `customer_families_hard_operations` | Operations-first with `AVOID` pairs rejected |
| `customer_families_flexible_operations` | Operations-first with `AVOID` pairs admitted and reported |
| `customer_families_greenfield_coverage` | Minimize demand-weighted coverage, then `J_CH`, without baseline constraints |
| `customer_families_greenfield_operations` | Minimize `J_CH`, then demand-weighted coverage, without baseline constraints |
| `customer_families_baseline_constrained_coverage` | One-block baseline neighbourhood search for lower coverage within the baseline guardrails |
| `customer_families_baseline_constrained_operations` | One-block baseline neighbourhood search for lower `J_CH` within the baseline guardrails |
| `target_base_group_whole_pallet_rounding` | Target band with whole-pallet rounding |
| `target_adjusted_group_minimum_only` | Target band on pallet-adjusted group coverage |
| `target_adjusted_group_whole_pallet_rounding` | Adjusted group plus whole-pallet rounding |
| `target_worst_fini_minimum_only` | Target band on worst member coverage |
| `target_worst_fini_whole_pallet_rounding` | Worst member plus whole-pallet rounding |
| `target_cap_8` | Permit groups up to eight |
| `target_cap_9` | Permit groups up to nine |
| `sensitivity_synthetic_hard_matrix` | Enforce the synthetic family matrix as hard |
| `sensitivity_matrix_off` | Ignore package-volume evidence |
| `sensitivity_optimized_pv` | Let the optimizer select the production version; sensitivity only |

## Python API

The CLI exposes reviewed configurations only. For a custom combination, use the
typed API from `api/`:

```python
import json
from pathlib import Path

from production_wheel.reporting import write_scenario_artifacts
from production_wheel.scenarios import (
    assess_baseline_guardrails,
    load_canonical_inputs,
    run_single_scenario,
)
from production_wheel.schemas import (
    BaselineGuardrailPolicy,
    CoverageBasis,
    CoverageMode,
    GroupSizeMode,
    GroupSizePolicy,
    MatrixMode,
    PalletFormula,
    PoolLimits,
    PVMode,
    RunConfig,
    SolveRequest,
    SolverLimits,
    TargetBand,
)
from production_wheel.solution_validation import validate_solution
from production_wheel.suite_reporting import validate_scenario_artifacts
from production_wheel.validation import ValidationStatus, validate_request

run_directory = Path("../prototype/output/demo")
output_directory = run_directory / "solutions" / "custom-review"
inputs = load_canonical_inputs(run_directory)
if output_directory.exists():
    raise FileExistsError(output_directory)

config = RunConfig(
    scenario_id="custom-review",
    coverage_mode=CoverageMode.DEMAND_WEIGHTED_MEAN,
    coverage_basis=CoverageBasis.BASE_GROUP,
    pallet_formula=PalletFormula.MINIMUM_ONLY,
    group_size=GroupSizePolicy(mode=GroupSizeMode.BOUNDED_RELAXATION, max_excess=1),
    pv_mode=PVMode.FIXED,
    matrix_mode=MatrixMode.DIAGNOSTIC,
    target_band=TargetBand(lower_days=5, upper_days=365),
    baseline_guardrails=BaselineGuardrailPolicy(j_ch_relative_tolerance=0.10),
    pool_limits=PoolLimits(),
    solver_limits=SolverLimits(
        time_limit_seconds=60,
        suite_time_limit_seconds=3600,
        mip_gap=0.01,
        threads=1,
        random_seed=0,
    ),
)

request_check = validate_request(SolveRequest(config=config))
if request_check.status is not ValidationStatus.VALID:
    raise ValueError(request_check.messages)
config = request_check.request.config

outcome = run_single_scenario(inputs, config)
if outcome.solve_result is None or not outcome.solve_result.has_incumbent:
    raise RuntimeError(outcome.error_message or "No feasible incumbent")

validation = validate_solution(
    outcome.solve_result,
    inputs.members,
    outcome.pools,
    config,
    inputs.production_versions,
    inputs.baseline_evidence_members or inputs.members,
)
acceptance = assess_baseline_guardrails(inputs, outcome.pools, config, outcome.solve_result)
source_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
write_scenario_artifacts(
    output_directory,
    inputs.fini_rows,
    inputs.members,
    outcome.pools,
    outcome.solve_result,
    config,
    validation,
    manifest_metadata={"source_extraction": source_manifest},
    production_versions=inputs.production_versions,
    acceptance_summary=acceptance.as_dict(),
    baseline_evidence_members=inputs.baseline_evidence_members,
)
artifact_check = validate_scenario_artifacts(output_directory)
if not artifact_check["valid"]:
    raise ValueError(artifact_check["errors"])
```

### Main `RunConfig` options

| Setting | Supported values / meaning |
|---|---|
| `scenario_id` | Run label used in artifacts; letters, digits, `_`, `.` and `-` |
| `coverage_mode` | `MAX`, `DEMAND_WEIGHTED_MEAN`, `GROUP_MEAN`, `TARGET_BAND`, `OPERATIONS_FIRST`, `PARETO` |
| `coverage_basis` | `BASE_GROUP`, `ADJUSTED_GROUP`, `WORST_FINI` |
| `pallet_formula` | `MINIMUM_ONLY`, `WHOLE_PALLET_ROUNDING` |
| `group_size` | Hard seven or bounded relaxation to eight/nine; above nine needs manual review |
| `pv_mode` | `FIXED` (default) or `OPTIMIZED` sensitivity |
| `matrix_mode` | `HARD`, `DIAGNOSTIC`, `FLEXIBLE`, `OFF` |
| `target_band` | Lower/upper coverage days; default 5 to 365 |
| `baseline_guardrails` | No worse violation count, excess, weighted mean, P90, group and singleton counts, `J_CH` and matrix exceptions; optional `J_CH` tolerances |
| `pool_limits` | Exact subset/configuration limits, restricted ceiling, beam width |
| `solver_limits` | Per-stage seconds, suite seconds, MIP gap, threads, seed |
| `versions` | Matrix, ruleset and output-schema identifiers |
| `demand_days` | Fixed to 250 |
| `productive_weeks` | Fixed to 50 |
| `canonical_factor` | Fixed to 0.90 |

## Formulas

For group `g` and FINI `i`:

```text
D_g = sum(D_i)
B_g = 0.90 × nominal PV lot
q_i = B_g × D_i / D_g
F_g = D_g / (50 × B_g)
J_CH,g = F_g × (n_g - 1)
```

Pallet formulas:

```text
MINIMUM_ONLY:          Q_i = max(q_i, pallet_i)
WHOLE_PALLET_ROUNDING: Q_i = pallet_i × ceil(q_i / pallet_i)
```

Coverage views:

```text
BASE_GROUP:     250 × B_g / D_g
ADJUSTED_GROUP: 250 × sum(Q_i) / D_g
WORST_FINI:     max_i(250 × Q_i / D_i)
```

The optimizer uses the canonical 0.90 lot factor for every production version.
A workbook that uses a different legacy factor for one version shows different
worst-coverage values, so compare results with the same factor.

## Package-volume compatibility

The `CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1` rule groups can volumes into four
inclusive, overlapping families: `<=1 L`, `2-5 L`, `4-10 L` and `10-15 L`. A
volume pair is `Y` when both volumes share at least one family and `AVOID`
otherwise. `N` is reserved for a site-certified hard prohibition. The empirical
matrix (`BASELINE_EMPIRICAL_MATRIX_V1`) instead derives evidence from the
historical wheel and shared filling lines.

Modes:

- `HARD` rejects every `AVOID` or `N` pair.
- `FLEXIBLE` admits `AVOID` pairs and reports every selected exception, without
  ranking exception minimization ahead of the business objective.
- `DIAGNOSTIC` admits exceptions, minimizes them before the business objective,
  and reports them.
- `OFF` ignores the matrix, for sensitivity analysis.

## How to interpret a solution

Read `solution_report.md` first, then `scenario_summary.csv`.

1. **Integrity before business value.** `validation_status=valid` means the
   selected groups cover every modeled FINI exactly once and the independent
   rule and KPI recomputation passed. It does not mean the result is optimal,
   approved or ready for SAP.
2. **Judge the majority, not only the worst group.**

   | Metric | Interpretation |
   |---|---|
   | `demand_weighted_mean_coverage_days` | Portfolio inventory proxy; high-demand groups weigh more |
   | `median_coverage_days` | Typical selected group |
   | `p90_coverage_days` | 90% of selected groups are at or below this value |
   | `group_mean_coverage_days` | Equal weight per group |
   | `maximum_coverage_days` | The single most extreme group |
   | `target_violation_count` | Groups outside the active band |
   | `target_total_excess_days` | Combined severity outside the band |

3. **Check manufacturing and fragmentation together.** `J_CH` estimates recurring
   transitions inside each group only; singletons contribute zero and changes
   between groups are excluded. Compare `group_count` and
   `singleton_group_count` so fragmentation does not look artificially attractive.
4. **Separate numeric acceptance from proof.**

   | Status | Meaning |
   |---|---|
   | `ACCEPTED_BASELINE_GUARDRAILS` | No configured guardrail limit is exceeded |
   | `PARETO_REVIEW_REQUIRED` | At least one metric improves while at least one guardrail is exceeded |
   | `BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES` | The baseline is not comparable under the active rules |
   | `NOT_ACCEPTABLE` | Guardrails fail and no governed metric improves |

   `feasible_limit`, `heuristic`, `partially_reached`, restricted pools or missing
   bounds mean a usable incumbent, not a proof of optimality.

## Output files

| File | Purpose |
|---|---|
| `proposed_subgroups.csv` | Every source FINI in source order; modeled rows receive proposals |
| `solution_groups.csv` | One row per proposed group with members and recomputed evidence |
| `scenario_summary.csv` | Portfolio metrics, baseline deltas, acceptance and solver status |
| `pareto_frontier.csv` | Suite only: trade-off points |
| `constraint_audit.csv` | Every active configuration and typed rule |
| `matrix_exception_audit.csv` | One row per selected `AVOID` or `N` FINI pair |
| `candidate_pool_summary.csv` | Pool evidence and per-block objective traces |
| `validation_issues.csv` | Independent errors and proof warnings |
| `pool_cache.csv` | Suite only: structural pool reuse evidence |
| `solution_report.md` | Reader-facing interpretation and caveats |
| `run_manifest.json` | Configuration, hashes, counts, solver evidence and provenance |

## Tests

No credentials or LLM needed:

```bash
cd api
../.venv/bin/python -m pytest tests -q
```

```bash
npm --prefix ui test
```

## Deployment (Cloud Foundry)

`manifest.yaml` defines three apps:

| App | Path | Purpose |
|---|---|---|
| `production-wheel-api` | `api` | FastAPI and agent (2 GB) |
| `production-wheel-worker` | `api` | Optimizer worker, no route (8 GB; size to your data) |
| `production-wheel-ui` | `ui` | UI served by `vite preview` |

Before the first deploy, replace the routes and the `ALLOWED_ORIGIN`,
`API_BASE_URL`, `VITE_API_BASE_URL` and `VITE_APP_HOST` values in `manifest.yaml`
with your own app names and Cloud Foundry apps domain. Then:

```bash
cf login
./deploy.sh
```

`deploy.sh` reads `api/.env`, generates a fresh API key shared by the API and UI,
pushes the three apps, binds the API to the `Cloud Logging` service instance and
restarts it. The worker runs with `WORKSPACE_EMBEDDED_WORKER=false` on the API.
One active job runs per worker instance.

## Project structure

```text
api/
  app/
    agent/             LangGraph agent, skills, config, CLI
    routers/           FastAPI routes (/api/datasets, /api/runs, /api/chat, ...)
    workspace/         HANA repository, datasets, runs, worker, analytics
    observability/     Token-usage logging
  production_wheel/    Extraction, candidate generation, MILP, validation, CLI
  scripts/             reset_workspace_tables.py, compare_benchmarks.py
  tests/
ui/
  src/                 Pages, workspace views, API client
  tests/
data/anonymized/       Demo workbooks
manifest.yaml
deploy.sh
```
