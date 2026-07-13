from __future__ import annotations

"""Utilities for selecting sales orders for analysis."""

import random
from dataclasses import dataclass
from numbers import Number
from typing import Optional

import pandas as pd


@dataclass
class OrderKey:
    document_number: str
    document_item: str


def normalize_identifier(value: object) -> str:
    """Return a stable string identifier for document numbers/items."""
    if pd.isna(value):
        return ""

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, Number):
        # Preserve integer-style formatting for floats that represent whole numbers
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return str(value).rstrip("0").rstrip(".")
        return str(value)

    return str(value).strip()


def create_order_key(row: pd.Series) -> Optional[OrderKey]:
    """Build an ``OrderKey`` from a dataframe row, handling missing identifiers."""
    doc_number = normalize_identifier(row.get("Sales Document Number"))
    doc_item = normalize_identifier(row.get("Sales Document Item"))

    if not doc_number or not doc_item:
        return None

    return OrderKey(document_number=doc_number, document_item=doc_item)


def select_random_anomalous_order(results_df: pd.DataFrame) -> Optional[OrderKey]:
    """Select a random anomalous order prioritizing AI results."""
    if "ai_anomaly_result" in results_df.columns:
        ai_anomalies = results_df[results_df["ai_anomaly_result"] == True]
        if not ai_anomalies.empty:
            row = ai_anomalies.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0]
            return create_order_key(row)

    if "predicted_anomaly" in results_df.columns and not results_df.empty:
        anomalies = results_df[results_df["predicted_anomaly"] == 1]
        if anomalies.empty and "anomaly_score" in results_df.columns:
            anomalies = results_df[results_df["anomaly_score"] < 0]
        if not anomalies.empty:
            row = anomalies.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0]
            return create_order_key(row)

    return None


def find_order(results_df: pd.DataFrame, document_number: str, document_item: str) -> Optional[pd.Series]:
    if "Sales Document Number" not in results_df.columns or "Sales Document Item" not in results_df.columns:
        return None

    doc_number = normalize_identifier(document_number)
    doc_item = normalize_identifier(document_item)

    normalized_docs = results_df["Sales Document Number"].map(normalize_identifier)
    normalized_items = results_df["Sales Document Item"].map(normalize_identifier)

    mask = (normalized_docs == doc_number) & (normalized_items == doc_item)
    matches = results_df[mask]
    if matches.empty:
        return None
    return matches.iloc[0]
