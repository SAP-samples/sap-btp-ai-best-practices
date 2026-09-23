# Experiment Loop Rules

This file is your process checklist. Re-read it when resuming a session.

## After EVERY Experiment

Run these checks after each experiment. Do not proceed to the next experiment until all pass.

### 1. Pre-run commit existed
Verify you committed `train.py` (and any helpers) BEFORE running the experiment.
```bash
# The most recent commit should be your experiment, not something older
git log --oneline -1
```
If you forgot to commit before running, commit now and re-run the experiment.

### 2. Output was captured
Verify `run.log` exists and contains results.
```bash
grep "^primary_metric:" run.log
```
If empty, the experiment crashed. Run `tail -50 run.log` to diagnose.

### 3. Results were extracted from run.log
You MUST read the actual values from `run.log`. Never estimate, recall from memory, or assume what the metrics are. Extract them:
```bash
grep "^primary_metric:\|^train_metric:\|^training_seconds:\|^model_type:\|^n_features:" run.log
```

### 4. Keep/discard was executed correctly
- **Improved** -> commit stays, this is the new baseline.
- **Worse or equal** -> run `git reset --hard HEAD~1`. Verify with `git log --oneline -1` that HEAD is now the previous best.
- **Crashed** -> attempt fix (max 2 tries), then discard.

### 5. results.tsv was updated
Append one row to `results.tsv` after every experiment, regardless of outcome. Verify:
```bash
tail -1 results.tsv
```
The last line should describe the experiment you just ran. Do NOT commit results.tsv.

### 6. Overfitting check
Compare train_metric vs val_metric (primary_metric). If the gap is large (>10-15% relative), flag it in the results.tsv description and prioritize regularization or feature reduction next.

## Every 5 Experiments

### 7. Technique diversity audit
Read the last 5 entries in `results.tsv`. If 3+ entries use the same model family or technique category (e.g., all hyperparameter tuning, all XGBoost), you MUST switch to an unexplored category from `references/strategy.md`.

### 8. Error analysis
- **Regression**: examine residual distribution, worst predictions, residuals vs predicted.
- **Classification**: confusion matrix, per-class precision/recall, worst predictions by loss.
Use findings to guide the next experiment.

## Every 10 Experiments

### 9. Progress summary
Print the progress block:
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

### 10. Evaluation harness integrity
Verify `evaluate.py` has not been modified:
```bash
git diff evaluate.py
```
Output must be empty. If it was modified, revert it immediately with `git checkout evaluate.py` and re-run the current baseline to confirm consistency.
