"""Temporal and conservative identity boundaries for historical RPT-1 context."""
import hashlib
import json
import numpy as np
import pandas as pd


def invoice_key(row):
    """Hash facility/customer/reference; ambiguous repeated matches are excluded together."""
    values = [row.get('Company Code'),row.get('Customer'),row.get('Invoice Reference')]
    if any(value is None or pd.isna(value) or not str(value).strip() for value in values):
        return None
    return hashlib.sha256(json.dumps([str(value).strip() for value in values]).encode()).hexdigest()


def admissible_history(rows, prediction_at, excluded_invoice_keys):
    """Return finite positive outcomes known strictly before the saved prediction instant."""
    if rows.empty:
        return rows.copy()
    known = pd.to_datetime(rows['outcome_known_at'],utc=True,errors='coerce')
    duration = pd.to_numeric(rows['credit_duration_days'],errors='coerce')
    instant = pd.Timestamp(prediction_at)
    instant = instant.tz_localize('UTC') if instant.tzinfo is None else instant.tz_convert('UTC')
    mask = known.lt(instant) & duration.gt(0) & np.isfinite(duration)
    mask &= rows['invoice_key'].notna() & ~rows['invoice_key'].isin(excluded_invoice_keys)
    return rows.loc[mask].copy()
