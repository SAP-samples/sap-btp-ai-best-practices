---
name: autoresearch
description: "Autonomous ML research agent. Given a dataset and a prediction goal, explores data, builds models, and iterates autonomously — modifying code, running experiments, keeping improvements, discarding regressions. Use when: (1) User provides a dataset and wants to predict a target column, (2) User wants to iterate on model performance without manual intervention, (3) User asks to 'find the best model', 'optimize predictions', 'run experiments', or 'autoresearch'. Triggers: 'train a model on this data', 'predict X from this dataset', 'find the best approach for this', 'run ML experiments', 'autoresearch', 'autonomous ML', 'iterate on this model', 'improve this prediction', 'optimize this classifier/regressor'. Always use this skill for any autonomous ML experimentation loop, even if the user does not explicitly say 'research'."
---

# Autonomous ML Research

Autonomous experiment loop for tabular ML: propose → implement → evaluate → keep/discard → repeat, indefinitely.

Read `references/strategy.md` before starting the experiment loop — it contains the technique catalog and decision framework.
Read `references/RULES.md` before starting — it contains the mandatory post-experiment validation checklist.

## Setup

### 1. Understand the goal

Parse `$ARGUMENTS` to identify:
- Path to dataset (CSV, Parquet, Excel, TSV, etc.)
- Target column(s)
- Columns to ignore (IDs, timestamps, leak-prone columns)
- Domain constraints (e.g. "predictions must be non-negative")
- Whether the user cares about secondary objectives (inference time, interpretability, model size)

### 2. Explore the data

Before writing any code, produce a concise data profile:
- Shape, dtypes, head/tail, describe()
- **Target**: distribution, class balance (classification) or range/skew (regression)
- **Features**: missing-value rates, cardinality, pairwise correlations with target, potential leakage flags
- **Data quality**: duplicates, constant columns, near-zero-variance features, outlier counts
- For classification: if minority class < 15% of data, flag imbalance — this affects metric choice and strategy

Print the summary for the user. Do NOT proceed until the user confirms the data looks right.

### 3. Determine task type and metric

Based on data and user input:
- **Task**: binary classification, multiclass, regression, multi-output
- **Primary metric** (the single number for keep/discard). Propose one and confirm:
  - Regression: RMSE (default), MAE, R², MAPE, WAPE
  - Binary classification: AUC-ROC (default), F1, log-loss. If imbalanced: use AUC-PR or F1
  - Multiclass: weighted F1 (default), accuracy, macro F1
- Record `LOWER_IS_BETTER` or `HIGHER_IS_BETTER`

### 4. Configure the run

- **Run tag**: Based on today's date (e.g. `mar21`). Branch `research/<tag>` must not already exist.
- **Time budget**: Default 60 seconds per experiment. For large datasets or deep learning, propose higher. Confirm with user.
- **Validation strategy**: Single holdout split (default). If dataset < 5000 rows, propose 5-fold stratified CV instead and adjust the harness accordingly.
- **Git**: If no git repo exists, run `git init` and create an initial commit.

### 5. Create the branch

```bash
git checkout -b research/<tag>
```

### 6. Install dependencies

Check the project's virtual environment. Install core packages:
```bash
pip install pandas scikit-learn numpy --break-system-packages
```
Install additional packages (xgboost, lightgbm, catboost, shap, etc.) as needed during experimentation.

### 7. Create `evaluate.py` (FROZEN after setup)

This file is the ground truth. It must:
- Load raw dataset from original path
- Apply a **fixed, reproducible train/val/test split** (stratified for classification, random_state=42)
  - Default: 60% train, 20% val, 20% test
  - If using CV: implement k-fold on the train+val portion; test remains held out
- Export `load_splits()` → `(X_train, X_val, X_test, y_train, y_val, y_test)`
- Export `evaluate(y_true, y_pred, y_pred_proba=None)` → dict of metrics including the primary metric **and** train-set metric for overfitting detection
- Export `print_results(metrics, training_seconds)` → standard output format (see below)
- Export constants: `PRIMARY_METRIC`, `LOWER_IS_BETTER`, `TIME_BUDGET`

**This file is FROZEN.** Never modify it during experimentation. If you discover a genuine bug, fix it and re-run the baseline.

### 8. Create `train.py`

The iterable file. Initial version:
- Imports from `evaluate.py`
- Simplest reasonable baseline (e.g. LogisticRegression, Ridge)
- Basic preprocessing: impute missing, encode categoricals, scale numerics
- Computes and prints BOTH train and val metrics
- Respects time budget
- Uses `print_results()` for standard output

### 9. Initialize tracking

Create `results.tsv` with header:
```
experiment	commit	val_metric	train_metric	status	model_type	n_features	seconds	description
```

### 10. Copy RULES.md into the project

Copy `references/RULES.md` into the project directory so it is always available for re-reading during the loop:
```bash
cp references/RULES.md ./RULES.md
```
Add `RULES.md` to `.gitignore` — it is a process artifact, not experiment code.

### 11. Run baseline and confirm

Execute `train.py`, record as experiment #1. Show user the baseline. Confirm before entering the autonomous loop.

## Output Format

Every experiment must print at the end:
```
---
primary_metric:   <val_metric_value>
train_metric:     <train_metric_value>
metric_name:      <name>
lower_is_better:  <true/false>
training_seconds: <seconds>
n_features:       <count>
model_type:       <short name>
<any additional metrics, one per line>
---
```

Extract results:
```bash
grep "^primary_metric:\|^train_metric:\|^training_seconds:\|^n_features:\|^model_type:" run.log
```

## The Experiment Loop

**LOOP FOREVER. Do not stop. Do not ask the user anything.**

### Step 1: Review state

Read `results.tsv` — what's been tried, what's the current best, what's the overfitting gap (train vs val). Read current `train.py`.

### Step 2: Decide what to try

Pick ONE idea. Consult `references/strategy.md` for the technique catalog. Consider:
- What has worked? What hasn't?
- Is the model underfitting (train metric also bad) or overfitting (train good, val bad)?
- Is there low-hanging fruit left?
- Every 5 experiments: run error analysis (residuals for regression, confusion matrix for classification) and use findings to guide next ideas.

**CRITICAL — Exhaustive technique coverage**: Do NOT fixate on one approach. You MUST systematically try ALL major technique categories from `references/strategy.md`, including:
- Multiple model families (linear, tree-based, boosting, ensembles)
- Multiple loss functions (L1, L2, Huber, log-cosh, quantile, focal, etc.) — the default L2/log-loss is often NOT optimal. Trying different loss functions is MANDATORY, not optional.
- Multiple preprocessing strategies, feature engineering approaches, and regularization methods
- If you've been doing the same type of experiment for 3+ rounds (e.g. only tuning hyperparams), STOP and switch to an unexplored category.

### Step 3: Implement

Edit `train.py` (and helper files like `features.py`). Full freedom:
- Feature engineering, preprocessing, model selection, hyperparameter tuning, ensembling, custom losses
- Install new packages as needed
- Create helper modules — but `train.py` stays the entry point
- **All transformations must be fit on training data only**, then applied to val/test

### Step 4: Commit

```bash
git add -A && git commit -m "<short description>"
```

### Step 5: Run with timeout

```bash
timeout $((TIME_BUDGET * 2)) python train.py > run.log 2>&1
```
If timeout fires, treat as crash.

### Step 6: Read results

```bash
grep "^primary_metric:\|^train_metric:\|^training_seconds:" run.log
```
If empty → crash. Run `tail -n 50 run.log` to see error.

### Step 7: Evaluate

Compare primary_metric to current best:
- **Improved** → keep commit. This is the new baseline.
- **Equal or worse** → `git reset --hard HEAD~1`. Branch stays at previous best.
- **Crashed** → attempt fix if trivial (typo, missing import, OOM). If still broken after 2 attempts, discard and move on.

**Overfitting gate**: If train_metric is dramatically better than val_metric (e.g. train R²=0.99, val R²=0.60), the experiment is overfitting regardless of whether val improved slightly. Flag this in the log and consider regularization or feature reduction next.

### Step 8: Log

Append to `results.tsv`. Do NOT commit results.tsv — leave it untracked.
You MUST always update `results.tsv` after every experiment to keep track of it properly

### Step 9: Post-experiment validation

Re-read `RULES.md` and run the "After EVERY Experiment" checklist. Verify all 6 items passed before continuing. If any check fails, fix it now. Additionally, check the periodic rules (every 5 and every 10 experiments) and execute them when due.

**Do NOT skip this step.** This is what keeps the process honest.

### Step 10: Progress summary

Every 10 experiments, print a progress block:
```
═══ PROGRESS (after N experiments) ═══
Best val metric:   <value> (experiment #X)
Current overfit gap: <train - val>
Experiments: <kept>/<total> kept
Top 3 improvements: <descriptions>
Current bottleneck: <underfitting|overfitting|plateau>
Next direction: <what you plan to explore>
═══════════════════════════════════════
```

### Step 11: Repeat

Go to Step 1. Never stop. If you run out of ideas, re-read the raw data, examine error distributions, try combining past near-misses, try radical departures.

## Constraints

- **Time budget**: Each experiment within configured budget. This is a soft contraint: if the new proposed technique requires more time to compute than the original estimate baseline, while still being reasonable, continue training.
- **Evaluation harness**: FROZEN. Never modify `evaluate.py` during experimentation. This evaluation must be the same for all experiments to be fair.
- **Reproducibility**: Fixed random seeds everywhere. Same code → same result.
- **Simplicity criterion**: Tiny improvement + massive complexity = not worth it. Removing code for equal performance is a win.
- **No data leakage**: Never use test/val information during training or feature engineering. Fit all transformers on training data only.
- **Test set**: NEVER use test set for keep/discard decisions. Test is for final reporting ONLY, when the user asks for it.
- **Crashes**: Max 2 fix attempts per idea, then move on.

## Recovering from a fresh session

If continuing from a previous run:
1. Read `RULES.md` for the process checklist
2. Read `results.tsv` for history
3. Read `evaluate.py` for task/metric definition
4. Read current `train.py` and any helper files
5. Run `git log --oneline -20` on `research/<tag>` for context
6. Resume the loop
