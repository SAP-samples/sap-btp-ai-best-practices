---
name: narrow-ai-linear-regression
description: Implement hana-ml regression workflows on SAP HANA Cloud for continuous target prediction, including baseline model training, scoring, and hyperparameter search. Use when tasks involve house-price style prediction, feature importance, and model tuning with PAL regressors.
---

# Narrow AI Linear Regression (hana-ml)

Use this skill for continuous-value prediction workflows built with hana-ml PAL regressors.

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
from hana_ml.algorithms.pal.trees import HybridGradientBoostingRegressor

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

df_local = pd.read_csv("./california_housing_dataset.csv")
hdf = dataframe.create_dataframe_from_pandas(
    connection_context=conn,
    pandas_df=df_local,
    table_name="CALIFORNIA_HOUSING",
    force=True,
    replace=False,
).add_id(id_col="ID")

train_hdf, test_hdf, _ = train_test_val_split(
    data=hdf,
    id_column="ID",
    partition_method="random",
    training_percentage=0.7,
    testing_percentage=0.3,
    validation_percentage=0.0,
    random_seed=2,
)

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
reg = HybridGradientBoostingRegressor(
    n_estimators=20,
    learning_rate=0.3,
    max_depth=2,
    evaluation_metric="rmse",
    ref_metric=["mae"],
)
reg.fit(train_hdf, features=["ID"] + features, label="Target")
r2 = reg.score(test_hdf, key="ID", features=features, label="Target")
```

## Add Hyperparameter Search

```python
from hana_ml.algorithms.pal.model_selection import ParamSearchCV

base = HybridGradientBoostingRegressor(n_estimators=50, subsample=0.8, col_subsample_tree=0.7)
search = ParamSearchCV(
    estimator=base,
    search_strategy="grid",
    param_grid={"learning_rate": [0.05, 0.1], "max_depth": [4, 6], "split_threshold": [0.1, 0.4]},
    train_control={"fold_num": 10, "evaluation_metric": "rmse"},
    scoring="mae",
)
search.fit(data=train_hdf, features=features, label="Target", key="ID")
```

## Best Practices

- Keep feature list centralized and reused for fit/score/search.
- Evaluate with at least RMSE and MAE; track R2 separately.
- Use train/test split plus CV search to avoid overfitting.
- Inspect feature importances before deployment.
- Remove temporary HANA tables after training.

## Expected Output Patterns

- hana-ml version prints (for example `2.24.x`).
- Connection check is `True`.
- Table query output for `CALIFORNIA_HOUSING`.
- Feature importance table sorted by `IMPORTANCE`.
- R2 score returned from `.score(...)`.
