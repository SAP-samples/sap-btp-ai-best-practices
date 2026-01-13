from __future__ import annotations

import pandas as pd
from typing import Tuple

from app.regressor.io_utils import load_demographics as _load_demographics


def load_demographics_with_typing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load demographics (static real-estate features) and CRM customer mix (time-varying).

    Returns:
        static_df: Per-store static features.
        crm_df: Per-store/channel/week customer mix percentages.
    """
    return _load_demographics()


# Backward-compatible alias expected by feature pipeline
def load_crm_mix() -> pd.DataFrame:
    _, crm_df = _load_demographics()
    return crm_df
