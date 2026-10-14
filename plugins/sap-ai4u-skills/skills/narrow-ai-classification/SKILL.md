---
name: narrow-ai-classification
description: Implement hana-ml classification pipelines on SAP HANA Cloud, especially churn/flight-risk style binary or multiclass prediction. Use when creating scripts for data upload to HANA, train/test splits, UnifiedClassification training, scoring, and prediction explanations.
---

# Narrow AI Classification (hana-ml)

Use this skill for supervised classification tasks running in SAP HANA via hana-ml PAL.

## Use Required Environment Variables

```bash
hana_address=""
hana_port="443"
hana_user=""
hana_password=""
hana_encrypt="true"
HANA_SCHEMA=""
```

## Core Implementation Pattern

```python
import os
import pandas as pd
from dotenv import load_dotenv
import hana_ml.dataframe as dataframe
from hana_ml.algorithms.pal.partition import train_test_val_split
from hana_ml.algorithms.pal.unified_classification import UnifiedClassification

load_dotenv()

conn = dataframe.ConnectionContext(
    address=os.getenv("hana_address"),
    port=int(os.getenv("hana_port", 443)),
    user=os.getenv("hana_user"),
    password=os.getenv("hana_password"),
    encrypt=True,
    sslValidateCertificate=False,
    current_schema=os.getenv("HANA_SCHEMA", ""),
)

df_local = pd.read_csv("./Emp_Churn_Train.csv")
hdf = dataframe.create_dataframe_from_pandas(
    connection_context=conn,
    pandas_df=df_local,
    table_name="EMPLOYEE_CHURN_DATA",
    force=True,
    replace=False,
)

df_train, df_test, _ = train_test_val_split(
    data=hdf,
    id_column="EMPLOYEE_ID",
    partition_method="stratified",
    stratified_column="FLIGHT_RISK",
    training_percentage=0.85,
    testing_percentage=0.15,
    validation_percentage=0.0,
    random_seed=1234,
)

clf = UnifiedClassification(
    func="HybridGradientBoostingTree",
    n_estimators=101,
    learning_rate=0.1,
    max_depth=6,
    evaluation_metric="error_rate",
)
clf.fit(data=df_train.drop("EMPLOYEE_ID"), label="FLIGHT_RISK")
scorepredictions, scorestats, scorecm, scoremetrics = clf.score(
    data=df_test, key="EMPLOYEE_ID", label="FLIGHT_RISK"
)
```

## Include Explainability Output

For decision support outputs, call:

```python
pred = clf.predict(
    df_test.drop("FLIGHT_RISK").head(1000),
    key="EMPLOYEE_ID",
    attribution_method="tree-shap",
    missing_replacement="feature_marginalized",
)
```

Then inspect `"REASON_CODE"` with JSON functions (`json_query`) to show top drivers.

## Best Practices

- Use stratified splits on the target for imbalanced labels.
- Keep ID/key column explicit (`EMPLOYEE_ID`) across split, score, and predict.
- Upload source data once to HANA and operate on HANA DataFrames for scale.
- Track AUC and confusion matrix from score artifacts, not only raw accuracy.
- Close connection and clean temporary tables after runs.

## Expected Output Patterns

- HANA connection check returns `True`.
- Row counts around `19115` for sample churn dataset.
- Class distribution similar to `FLIGHT_RISK: No/Yes`.
- Score outputs include confusion matrix and model metrics.
- Prediction output includes `SCORE`, `CONFIDENCE`, and `REASON_CODE`.
