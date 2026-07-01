import numpy as np
import pandas as pd

def classify_driver(row: pd.Series) -> str:
    if row["Amount_Q3_24"] != 0 and row["Amount_Q3_25"] == 0:
        return "Reversal"
    if row["Amount_Q3_24"] == 0 and row["Amount_Q3_25"] != 0:
        return "New Posting"
    if np.sign(row["Amount_Q3_24"]) != np.sign(row["Amount_Q3_25"]):
        return "Accrual / Correction"
    if row["Amount_Q3_24"] != row["Amount_Q3_25"]:
        return "Timing"
    return "Other"

def apply_driver_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Driver"] = df.apply(classify_driver, axis=1)
    return df
