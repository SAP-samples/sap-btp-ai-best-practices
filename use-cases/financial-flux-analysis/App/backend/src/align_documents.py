
import pandas as pd

KEY_COLS = [
    "Ledger",
    "Fiscal Year",
    "Document Number",
    "Posting Item",
    "Consolidation Unit",
    "Account Number",
    "Cost Center"
]

AMOUNT_COL = "Amount in Group Crcy"

def align_periods(df_24: pd.DataFrame, df_25: pd.DataFrame) -> pd.DataFrame:

    missing_24 = [c for c in KEY_COLS if c not in df_24.columns]
    missing_25 = [c for c in KEY_COLS if c not in df_25.columns]

    if missing_24 or missing_25:
        raise ValueError(
            f"Missing columns - Q3_24: {missing_24}, Q3_25: {missing_25}"
        )

    df_24 = df_24.rename(columns={AMOUNT_COL: "Amount_Q3_24"})
    df_25 = df_25.rename(columns={AMOUNT_COL: "Amount_Q3_25"})

    merged = pd.merge(
        df_24,
        df_25,
        on=KEY_COLS,
        how="outer"
    )

    merged["Amount_Q3_24"] = merged["Amount_Q3_24"].fillna(0)
    merged["Amount_Q3_25"] = merged["Amount_Q3_25"].fillna(0)

    merged["Delta"] = merged["Amount_Q3_25"] - merged["Amount_Q3_24"]

    return merged
