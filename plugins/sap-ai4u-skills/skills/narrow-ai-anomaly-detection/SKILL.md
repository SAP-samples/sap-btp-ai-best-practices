---
name: narrow-ai-anomaly-detection
description: Implement hana-ml anomaly and outlier detection pipelines on SAP HANA Cloud for tabular, time-series, and regression data. Use when tasks require Isolation Forest, DBSCAN, One-Class SVM, KMeans-based outlier detection, OutlierDetectionTS, or OutlierDetectionRegression with evaluation.
---

# Narrow AI Anomaly Detection (hana-ml)

Use this skill for anomaly detection workflows using PAL algorithms in HANA.

## Use Required Environment Variables

```bash
hana_address=""
hana_port="443"
hana_user=""
hana_password=""
hana_encrypt="true"
HANA_SCHEMA=""
```

## Shared Data Upload Pattern

```python
import os
import pandas as pd
from dotenv import load_dotenv
from hana_ml import ConnectionContext
from hana_ml.dataframe import create_dataframe_from_pandas

load_dotenv()

cc = ConnectionContext(
    address=os.getenv("hana_address"),
    port=int(os.getenv("hana_port", 443)),
    user=os.getenv("hana_user"),
    password=os.getenv("hana_password"),
    encrypt=os.getenv("hana_encrypt", "true").lower() == "true",
    current_schema=os.getenv("HANA_SCHEMA", ""),
)

df_local = pd.read_csv("synthetic_anomaly_data.csv")
if "ID" not in df_local.columns:
    df_local.insert(0, "ID", range(len(df_local)))
feature_cols = [c for c in df_local.columns if c.startswith("feature_")]

hdf_input = create_dataframe_from_pandas(
    connection_context=cc,
    pandas_df=df_local,
    table_name="SYNTHETIC_ANOMALY_DATA_IFOREST",
    force=True,
    replace=True,
    primary_key="ID",
)
```

## Algorithm Patterns

Isolation Forest:

```python
from hana_ml.algorithms.pal.preprocessing import IsolationForest

iforest = IsolationForest(n_estimators=100, max_samples=256, random_state=42, thread_ratio=-1)
iforest.fit(data=hdf_input, key="ID", features=feature_cols)
results_hdf = iforest.predict(data=hdf_input, key="ID", contamination=float(df_local["is_anomaly"].mean()))
```

DBSCAN:

```python
from hana_ml.algorithms.pal.clustering import DBSCAN

dbscan = DBSCAN(minpts=None, eps=None, thread_ratio=1, metric="euclidean")
dbscan.fit(data=hdf_input, key="ID", features=feature_cols)
results_hdf = dbscan.predict(data=hdf_input, key="ID", features=feature_cols)
```

One-Class SVM:

```python
from hana_ml.algorithms.pal.svm import OneClassSVM

ocsvm = OneClassSVM(nu=0.05, kernel="rbf", scale_info="standardization", thread_ratio=0.8)
ocsvm.fit(data=hdf_input, key="ID", features=feature_cols)
results_hdf = ocsvm.predict(data=hdf_input, key="ID", features=feature_cols)
```

KMeans outlier detection:

```python
from hana_ml.algorithms.pal.clustering import outlier_detection_kmeans

outliers_hdf, stats_hdf, centers_hdf = outlier_detection_kmeans(
    data=hdf_input,
    key="ID",
    features=feature_cols,
    n_clusters=None,
    contamination=0.05,
)
```

Time-series anomalies:

```python
from hana_ml.algorithms.pal.tsa.outlier_detection import OutlierDetectionTS

tsod = OutlierDetectionTS(auto=True)
results_hdf = tsod.fit_predict(data=hdf_input, key="timestamp", endog="value")
```

Regression anomalies:

```python
from hana_ml.algorithms.pal.preprocessing import OutlierDetectionRegression

odr = OutlierDetectionRegression(regression_model="linear", threshold=3.0)
results_hdf = odr.fit_predict(data=hdf_input, key="ID", features=FEATURE_COLS, label="target")
```

## Best Practices

- Use one table per algorithm run to avoid column-shape conflicts.
- Convert prediction labels to a common binary format before evaluation.
- Compare against ground truth with confusion matrix and F1 when available.
- For time series, ensure timestamp typing and sort order before fit.
- Treat contamination/nu/threshold as business-sensitive knobs; tune per domain.

## Expected Output Patterns

- HANA connection banner and schema print.
- Table upload logs with row counts (for example 1050 or 1095 rows in synthetic sets).
- Model training completion messages.
- Prediction outputs with algorithm-specific columns:
  - IsolationForest: `SCORE`, `LABEL`
  - DBSCAN: `CLUSTER_ID`, `DISTANCE`
  - OneClassSVM: `SCORE`
  - OutlierDetectionTS: `IS_OUTLIER`, `OUTLIER_SCORE`, `RESIDUAL`
  - OutlierDetectionRegression: `IS_OUTLIER`, `OUTLIER_SCORE`, `RESIDUAL`
