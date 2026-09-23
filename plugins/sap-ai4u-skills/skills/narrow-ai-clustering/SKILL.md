---
name: narrow-ai-clustering
description: Implement hana-ml clustering workflows on SAP HANA Cloud, including KMeans training, cluster prediction, and exploratory profiling for unlabeled datasets. Use when tasks involve segmentation, grouping, or unsupervised structure discovery in tabular data.
---

# Narrow AI Clustering (hana-ml)

Use this skill for unsupervised clustering with SAP HANA PAL.

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
from hana_ml.algorithms.pal.utility import train_test_val_split
from hana_ml.algorithms.pal import clustering

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

iris_pd = pd.read_csv("./Iris.csv")
iris_hdf = dataframe.create_dataframe_from_pandas(
    connection_context=conn,
    pandas_df=iris_pd,
    table_name="IRISDATASET",
    force=True,
    replace=False,
)

df_train, df_test, _ = train_test_val_split(
    data=iris_hdf,
    id_column="Id",
    partition_method="stratified",
    stratified_column="Species",
    training_percentage=0.8,
    testing_percentage=0.2,
    validation_percentage=0.0,
    random_seed=2,
)

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
kmeans = clustering.KMeans(
    n_clusters=3,
    distance_level="euclidean",
    normalization="min_max",
    max_iter=100,
)
model = kmeans.fit(data=df_train, key="Id", features=features)
pred = model.predict(data=df_test, key="Id", features=features)
```

## Best Practices

- Normalize numeric features before clustering (`normalization="min_max"`).
- Keep a stable key column (`Id`) for merges and evaluation joins.
- Use stratified split by known label only for offline validation, not as a model input.
- Inspect cluster-to-class cross-tab for interpretability.
- Drop staging tables when complete.

## Expected Output Patterns

- HANA connection status `True`.
- Iris shape around `[150, 6]`.
- KMeans labels output with `CLUSTER_ID`.
- Training/test sample counts displayed.
- Final prediction DataFrame includes IDs and assigned clusters.
