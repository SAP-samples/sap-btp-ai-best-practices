# Strategy Reference

Technique catalog and decision framework for the experiment loop. Read this before starting.

## Phase Map

### Phase 1: Baselines (experiments 1–5)

Goal: understand what model families work for this data.

1. Simplest possible model (LogisticRegression / Ridge)
2. Random Forest with defaults
3. Gradient boosting with defaults (XGBoost or LightGBM)
4. If classification is imbalanced: re-run best model with `class_weight='balanced'` or `scale_pos_weight`
5. Basic feature cleanup: proper missing-value imputation, ordinal vs one-hot encoding

**Decision point**: Which model family has the best val metric? Focus there for Phase 2.

### Phase 2: Loss Functions + Feature Engineering + Tuning (experiments 6–25)

Goal: squeeze performance from the best model family.

**Loss function exploration** (try EARLY — these can unlock large gains):
- **Regression**: Don't default to L2 (MSE) and stop there. Systematically try:
  - MAE / L1 loss — robust to outliers
  - Huber loss — combines L1 and L2, good when some outliers exist
  - Log-cosh loss — smooth approximation of MAE
  - Quantile loss — useful for skewed targets
  - Tweedie loss — for zero-inflated or strictly positive targets
  - Pseudo-Huber loss — differentiable version of Huber
  - For XGBoost/LightGBM: set `objective` parameter (e.g. `reg:pseudohuber_loss`, `huber`, `mae`, `quantile`, `tweedie`)
  - For sklearn: use `HuberRegressor`, `QuantileRegressor`, or write custom losses via `make_scorer` + custom objective functions
- **Binary classification**: Don't just use log-loss. Try:
  - Focal loss — down-weights easy examples, great for imbalanced data (LightGBM: `objective='binary'` + custom, XGBoost: custom objective)
  - Weighted cross-entropy — adjust class weights beyond `balanced`
  - Hinge loss (SVM-style) — for margin-based models
- **Multiclass**: Try focal loss variants, label-smoothed cross-entropy
- **General rule**: The loss function shapes what the model optimizes. If your metric is MAE, training with L2 loss is suboptimal — train with L1. If outliers exist, Huber beats both. ALWAYS try at least 3 different loss functions for any task before moving on to other techniques.

**Feature engineering ideas** (try 1 per experiment):
- Interaction terms between top correlated features
- Polynomial features (degree 2) on top numeric predictors
- Binning continuous features (quantile-based)
- Date decomposition: year, month, day, day-of-week, is-weekend, days-since-reference
- Text vectorization: TF-IDF, character n-grams, or sentence embeddings if text columns exist
- Target encoding for high-cardinality categoricals (fit on train only, with regularization)
- Log/sqrt transforms on skewed numerics
- Ratio features (A/B for pairs that make domain sense)
- Aggregation features: group-by statistics (mean, std, count) on categorical groupings
- Missing-value indicators as binary features
- Frequency encoding for categoricals

**Hyperparameter tuning** (after features stabilize):
- Use RandomizedSearchCV or Optuna within time budget
- For XGBoost/LightGBM: focus on learning_rate, max_depth, n_estimators, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda
- For RF: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

**Feature selection** (if n_features > 50):
- Drop features with zero importance from tree models
- Recursive feature elimination (start coarse: remove bottom 20%)
- Correlation-based removal: if two features correlate > 0.95, drop the one with lower target correlation

**Preprocessing alternatives**:
- Try RobustScaler vs StandardScaler vs no scaling (tree models don't need it)
- Compare mean vs median vs KNN imputation
- Outlier handling: clip at 1st/99th percentile or Winsorize

### Phase 3: Ensembling + Advanced (experiments 25+)

Goal: combine multiple good approaches.

- **Voting/Averaging**: Weighted average of top 2–3 models (calibrate weights on val set)
- **Stacking**: Use predictions of base models as features for a meta-learner (logistic regression or ridge)
- **Blending**: Train base models on 80% of train, predict on remaining 20% + val, train meta-learner on those
- **Probability calibration**: CalibratedClassifierCV for classification
- **Threshold tuning**: Optimize classification threshold on val set for F1 or business metric
- **Custom loss functions**: By Phase 3 you should have already explored loss functions in Phase 2. If not, do it now — this is mandatory, not optional.
- **Advanced models**: CatBoost (handles categoricals natively), neural networks (if time budget allows and data is >10K rows)

### Phase 4: Radical departures (when plateau)

- Remove 50% of features and retrain — simpler models sometimes generalize better
- Try a completely different model family you haven't explored
- Reverse an assumption: what if that column you're ignoring is actually useful?
- PCA/UMAP dimensionality reduction as preprocessing
- Create a hand-crafted "business rule" feature based on domain knowledge inferred from column names
- Try quantile regression or conformal prediction for uncertainty

## Diagnostic Framework

Run diagnostics when progress stalls or every 10 experiments.

### For regression
```python
residuals = y_val - y_pred
# Distribution of residuals — should be roughly normal, centered at 0
# Residuals vs predicted — look for heteroscedasticity or patterns
# Worst predictions — examine the rows where |residual| is largest
# Feature importance — are the top features sensible?
```

### For classification
```python
from sklearn.metrics import confusion_matrix, classification_report
# Confusion matrix — which classes are confused?
# Per-class precision/recall — where is the model weak?
# Predicted probability distribution — is the model confident or hedging?
# Worst predictions — examine rows with highest loss (cross-entropy)
```

### Overfitting signals
- Train metric >> val metric (gap > 10-15% relative) → add regularization, reduce model complexity, reduce features, add dropout
- Val metric gets worse while train stays the same → model is memorizing noise
- Very high feature count relative to sample count → dimensionality reduction

### Underfitting signals
- Both train and val metrics are poor → model is too simple, needs more features, more capacity, or a different model family
- Learning curves show both train and val improving with more data → get more data if possible, or augment

## Imbalanced Data Techniques

When minority class < 15%:
1. First try: `class_weight='balanced'` parameter (most models support this)
2. Threshold tuning on val set — don't default to 0.5
3. SMOTE or ADASYN on training set only (never on val/test)
4. Focal loss (supported in LightGBM, or custom in XGBoost)
5. Switch metric to AUC-PR or F1-macro instead of accuracy or AUC-ROC
6. Ensemble over multiple resampled training sets (EasyEnsemble pattern)

## Package Quick Reference

| Package | Install | Use case |
|---------|---------|----------|
| scikit-learn | pre-installed | Baselines, preprocessing, metrics |
| xgboost | `pip install xgboost` | Gradient boosting |
| lightgbm | `pip install lightgbm` | Fast gradient boosting |
| catboost | `pip install catboost` | Categorical-native boosting |
| optuna | `pip install optuna` | Bayesian hyperparameter search |
| shap | `pip install shap` | Feature importance / interpretability |
| category_encoders | `pip install category_encoders` | Target encoding, WoE, etc. |
| imbalanced-learn | `pip install imbalanced-learn` | SMOTE, ADASYN, EasyEnsemble |

Add `--break-system-packages` flag to pip if not using a venv.
