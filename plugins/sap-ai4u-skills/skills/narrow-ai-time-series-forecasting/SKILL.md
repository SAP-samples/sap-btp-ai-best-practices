---
name: narrow-ai-time-series-forecasting
description: Implement hana-ml time-series forecasting on SAP HANA Cloud using PAL additive forecasting models, including aggregation, future horizon generation, prediction intervals, and persistence of forecast outputs.
---

# Narrow AI Time-Series Forecasting (hana-ml)

Use this skill for monthly/periodic forecasting workflows in SAP HANA.

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
from hana_ml.algorithms.pal.tsa.additive_model_forecast import AdditiveModelForecast
from hana_ml.algorithms.pal.random import binomial

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

df_local = pd.read_csv("./OVERNIGHTSTAYS.csv")
df_local["MONTH"] = pd.to_datetime(df_local["MONTH"], format="%d/%m/%Y")
hdf = dataframe.create_dataframe_from_pandas(
    connection_context=conn,
    pandas_df=df_local,
    table_name="OVERNIGHTSTAYS",
    force=True,
    replace=False,
)

df_agg = hdf.agg([("sum", "OVERNIGHTSTAYS", "OVERNIGHTSTAYS_SUM")], group_by="MONTH").sort("MONTH")
amf = AdditiveModelForecast()
amf.fit(data=df_agg)

last_date = str(df_agg.tail(1, ref_col="MONTH").collect().iloc[0, 0])[:10]
future = binomial(conn, n=1, p=1, num_random=12)
future = future.select("*", (f"ADD_MONTHS(TO_DATE ('{last_date}', 'YYYY-MM-DD'), ID+1)", "MONTH")).select("MONTH", ("0", "TARGET"))
pred = amf.predict(data=future)
```

## Merge Historical + Forecast Output

```python
pred_fmt = pred.select("MONTH", ("NULL", "OVERNIGHTSTAYS_SUM"), ("YHAT", "FORECAST"), ("YHAT_LOWER", "FORECAST_LOWER"), ("YHAT_UPPER", "FORECAST_UPPER"))
hist_fmt = df_agg.select("*", ("NULL", "FORECAST"), ("NULL", "FORECAST_LOWER"), ("NULL", "FORECAST_UPPER"))
full = pred_fmt.union(hist_fmt).sort("MONTH")
full.save("OVERNIGHTSTAYS_FORECAST_TOTAL", force=True)
```

## Best Practices

- Convert date column before upload and keep it typed as timestamp/date in HANA.
- Aggregate per business grain (monthly here) before model fitting.
- Always persist lower/upper intervals with the point forecast.
- Keep horizon generation deterministic (for example `12` months).
- Save combined historical + forecast table for dashboards.

## Expected Output Patterns

- hana-ml version output.
- Connection status `True`.
- Aggregated series with `MONTH` and `OVERNIGHTSTAYS_SUM`.
- Prediction columns `YHAT`, `YHAT_LOWER`, `YHAT_UPPER`.
- Persisted output table like `OVERNIGHTSTAYS_FORECAST_TOTAL`.
