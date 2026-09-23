# TabPFN 3.5-plus — settings per task

All settings are validated live against the deployment. The value the model
actually used is echoed back in `metadata.tabpfn_config` on every response.

## Where settings go

```json
"task_config": {
  "task": "classification",          // or "regression" — the discriminator
  "tabpfn_config": {                 // ALL hyper-parameters nest here
    "n_estimators": 4,
    "random_state": 42,
    "categorical_features_indices": [0, 2, 6]
  }
}
```

`task_config` is **strict**: putting `n_estimators` (or any hyper-parameter, or a
typo) at the `task_config` top level returns `400 extra_forbidden`. They must be
inside `task_config.tabpfn_config`. `tabpfn_config` is also strict — only the
keys below are accepted.

## `tabpfn_config` knobs (validated)

| Key | Type | Notes |
| --- | --- | --- |
| `n_estimators` | int **1–8** | Ensemble members. **Hard-capped at 8** (the default); `>8` → 400 `Input should be less than or equal to 8`. Fewer = faster, slightly different predictions. |
| `random_state` | int | Reproducibility. Same seed → **bit-identical** predictions; different seed → different ensemble (small prediction shift). Default 0. |
| `categorical_features_indices` | list[int] | Positional indices of categorical feature columns. Defaults to `[]` (TabPFN guesses). Set it for integer-coded categoricals (e.g. Pclass, season) — a free accuracy lever. |
| `inference_precision` | str | Default `"auto"`. |
| `inference_config` | object/null | Advanced; default null. |
| `fit_mode` | str | Default `"fit_preprocessors"`. |
| `memory_saving_mode` | bool | Default false. |
| `n_preprocessing_jobs` | int | Default 4. |
| `ignore_pretraining_limits` | bool | Default true. |
| `average_before_softmax` | bool | Classification-side; default false. |
| `softmax_temperature` | float/null | Classification calibration; default null. |
| `balance_probabilities` | bool | **Classification only** — rebalances class probabilities. |

Task differences: the classification config (`ClassifierConfig`) additionally
exposes `balance_probabilities`; the regression config (`RegressorConfig`) does
not. `softmax_temperature` / `average_before_softmax` are meaningful only for
classification. Otherwise both tasks share the same knobs.

## Not available on this deployment

- **No thinking mode.** `thinking_effort`, `thinking_metric`, and `thinking` are
  all rejected (`400 extra_forbidden`). The Prior Labs SDK exposes these
  (`create_default_for_version(..., thinking_effort=..., thinking_metric=...)`),
  but they are **client-side** and target the separate **TabPFN Thinking**
  checkpoint. The SAP AI Core deployment serves the **TabPFN Plus** checkpoint
  only; Fast and Thinking checkpoints are not yet available (per the SAP FAQ).
- **No version switching.** The SDK's `ModelVersion.V3 / V3_5` selection does not
  apply — the deployment is a fixed `tabpfn-3.5-plus` model.

## Unexplored

The 400 message hints at an alternate `model_params` / `predict_params` path
alongside `tabpfn_config`. Only the `tabpfn_config` path has been validated; the
effect of `inference_precision` / `inference_config` / `fit_mode` on results has
not been measured. Probe the endpoint (read the 400/422 bodies) before relying
on anything not in the table above.
