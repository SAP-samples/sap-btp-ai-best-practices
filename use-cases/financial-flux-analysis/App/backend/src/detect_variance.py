import pandas as pd

def detect_variances(
    df: pd.DataFrame,
    materiality: float = 0
) -> pd.DataFrame:
    """
    Filters document-level variances based on materiality threshold.
    """

    df = df.copy()

    # Keep only real variances
    df = df[df["Delta"] != 0]

    df["Abs_Delta"] = df["Delta"].abs()

    if materiality > 0:
        df = df[df["Abs_Delta"] >= materiality]

    return df.sort_values("Abs_Delta", ascending=False)
