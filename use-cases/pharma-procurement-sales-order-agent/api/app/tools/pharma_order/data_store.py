"""Small JSON-backed data access layer for the Pharma Procurement Sales Order Agent prototype.

The prototype intentionally uses local synthetic data instead of a live
S/4HANA integration. The functions are tolerant of schema changes: they
search records by business terms and return source payloads that an agent
can cite in its answer.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATASETS = {
    "pricing": "API_SALES_ORDER_SIMULATION_SRV__pricing_simulations.json",
    "stock": "API_MATERIAL_STOCK_SRV__material_stock_availability.json",
    "sales_orders": "API_SALES_ORDER_SRV__sales_orders_header_item_partner_status.json",
    "batches": "API_BATCH_SRV__batch_expiry_lot_status.json",
    "invoices": "API_BILLING_DOCUMENT_SRV__invoice_pdf_metadata.json",
    "customer_compliance": "ZSD_EXTERNAL_INFO__customers_dea_gts_lookup.json",
    "materials": "ZAPI_DEL_LIST_PRICE_V4__materials_ndc_catalog.json",
    "scenarios": "PHARMA_ORDER_SCENARIOS__sample_questions_tool_mapping.json",
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "mg", "of", "on", "or", "please", "show", "the", "to",
    "what", "with",
}

ALIASES = {
    "glycemor": "glycemor",
    "glycemor": "glycemor",
    "northstar": "northstar pharmacy distribution",
    "metromed": "metromed wholesale",
}


def dataset_path(dataset_name: str) -> Path:
    file_name = DATASETS.get(dataset_name, dataset_name)
    return DATA_DIR / file_name


@lru_cache(maxsize=32)
def load_dataset(dataset_name: str) -> Any:
    path = dataset_path(dataset_name)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(_scalar_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_scalar_text(item) for item in value)
    return str(value)


def _looks_like_record(value: dict[str, Any]) -> bool:
    scalar_count = sum(
        1
        for item in value.values()
        if isinstance(item, (str, int, float, bool)) or item is None
    )
    return scalar_count >= 2


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, parent_key)
            return
        if isinstance(value, dict):
            if _looks_like_record(value):
                record = dict(value)
                if parent_key and "_source_collection" not in record:
                    record["_source_collection"] = parent_key
                records.append(record)
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    visit(item, str(key))

    visit(payload)
    if isinstance(payload, dict) and not records:
        records.append(payload)
    return records


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(*parts: Any) -> list[str]:
    raw = " ".join(_scalar_text(part) for part in parts if part not in (None, ""))
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", raw.lower())
    tokens = [word for word in words if len(word) > 1 and word not in STOP_WORDS]
    expanded = list(tokens)
    for token in tokens:
        alias = ALIASES.get(token)
        if alias:
            expanded.extend(alias.split())
    seen: set[str] = set()
    return [token for token in expanded if not (token in seen or seen.add(token))]


def score_record(record: dict[str, Any], tokens: Iterable[str], phrase: str = "") -> int:
    haystack = _normalize(_scalar_text(record))
    score = 0
    normalized_phrase = _normalize(phrase)
    if normalized_phrase and normalized_phrase in haystack:
        score += 8
    for token in tokens:
        if token in haystack:
            score += 2 if len(token) > 3 else 1
    return score


def search_records(dataset_name: str, *query_parts: Any, limit: int = 5) -> dict[str, Any]:
    payload = load_dataset(dataset_name)
    records = records_from_payload(payload)
    tokens = tokenize(*query_parts)
    phrase = " ".join(_scalar_text(part) for part in query_parts if part not in (None, ""))
    ranked = []
    for record in records:
        score = score_record(record, tokens, phrase=phrase)
        if score > 0:
            ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    matches = [{"score": score, "record": record} for score, record in ranked[:limit]]
    return {
        "dataset": dataset_name,
        "file": DATASETS.get(dataset_name, dataset_name),
        "query_terms": tokens,
        "match_count": len(matches),
        "matches": matches,
    }


def search_many(
    dataset_names: Iterable[str], *query_parts: Any, limit_per_dataset: int = 3
) -> dict[str, Any]:
    results = [
        search_records(name, *query_parts, limit=limit_per_dataset)
        for name in dataset_names
    ]
    return {
        "query": " ".join(
            _scalar_text(part) for part in query_parts if part not in (None, "")
        ),
        "results": results,
    }
