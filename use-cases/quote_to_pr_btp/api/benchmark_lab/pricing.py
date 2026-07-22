"""Token cost estimation for benchmark comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPrice:
    """Price per 1K tokens for a model."""

    input_per_1k: float
    output_per_1k: float
    currency: str = "USD"
    source: str = "static_fallback"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentAIPriceTier:
    """SAP Document AI subscription tier expressed as blocks of 100 documents."""

    label: str
    up_to_blocks: int
    eur_per_100_documents: float
    source: str = "SAP Discovery Center service plan"

    @property
    def eur_per_document(self) -> float:
        return self.eur_per_100_documents / 100

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eur_per_document"] = round(self.eur_per_document, 4)
        return data


# Conservative placeholders for UI comparison. Replace with extracted SAP pricing
# from 3437766_E_20260625.pdf when that commercial parsing block is implemented.
STATIC_MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5": ModelPrice(input_per_1k=0.00125, output_per_1k=0.01000),
    "gpt-5-mini": ModelPrice(input_per_1k=0.00025, output_per_1k=0.00200),
    "gpt-4.1": ModelPrice(input_per_1k=0.00200, output_per_1k=0.00800),
    "gpt-4.1-nano": ModelPrice(input_per_1k=0.00010, output_per_1k=0.00040),
    "gpt-4o": ModelPrice(input_per_1k=0.00250, output_per_1k=0.01000),
    "gpt-4o-mini": ModelPrice(input_per_1k=0.00015, output_per_1k=0.00060),
    "gemini-2.5-flash": ModelPrice(input_per_1k=0.00030, output_per_1k=0.00250),
    "gemini-2.5-pro": ModelPrice(input_per_1k=0.00125, output_per_1k=0.01000),
    "gemini-2.0-flash": ModelPrice(input_per_1k=0.00010, output_per_1k=0.00040),
    "anthropic--claude-4.5-opus": ModelPrice(input_per_1k=0.01500, output_per_1k=0.07500),
    "anthropic--claude-4.6-opus": ModelPrice(input_per_1k=0.01500, output_per_1k=0.07500),
    "anthropic--claude-4.7-opus": ModelPrice(input_per_1k=0.01500, output_per_1k=0.07500),
    "anthropic--claude-4.8-opus": ModelPrice(input_per_1k=0.00367, output_per_1k=0.01806, source="SAP Note 3437766, aws-bedrock"),
}


DOCUMENT_AI_PRICE_TIERS: list[DocumentAIPriceTier] = [
    DocumentAIPriceTier(label="Up to 5 blocks of 100 documents", up_to_blocks=5, eur_per_100_documents=300.00),
    DocumentAIPriceTier(label="Up to 500 blocks of 100 documents", up_to_blocks=500, eur_per_100_documents=60.00),
    DocumentAIPriceTier(label="Up to 1,000 blocks of 100 documents", up_to_blocks=1000, eur_per_100_documents=45.00),
    DocumentAIPriceTier(label="Up to 3,000 blocks of 100 documents", up_to_blocks=3000, eur_per_100_documents=35.00),
    DocumentAIPriceTier(label="Up to 8,000 blocks of 100 documents", up_to_blocks=8000, eur_per_100_documents=25.00),
]


def _usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def estimate_model_cost(model: str | None, usage: dict[str, Any] | None) -> dict[str, Any]:
    """Estimate token cost with a safe not-configured fallback."""

    usage = usage or {}
    model = model or ""
    price = STATIC_MODEL_PRICES.get(model)
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    if not price:
        return {
            "status": "not_configured",
            "model": model,
            "estimated_cost": None,
            "cost_display": "not configured",
            "currency": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "price": None,
        }

    cost = (input_tokens / 1000) * price.input_per_1k + (output_tokens / 1000) * price.output_per_1k
    return {
        "status": "estimated",
        "model": model,
        "estimated_cost": round(cost, 6),
        "cost_display": f"{price.currency} {cost:.4f}",
        "currency": price.currency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "price": price.to_json_dict(),
    }


def document_ai_price_table() -> list[dict[str, Any]]:
    """Return user-facing Document AI price tiers."""

    return [
        {
            "Volume tier": tier.label,
            "EUR per 100 documents": tier.eur_per_100_documents,
            "EUR per document": round(tier.eur_per_document, 4),
            "Unit": "commercial document",
            "Source": tier.source,
        }
        for tier in DOCUMENT_AI_PRICE_TIERS
    ]


def estimate_document_ai_cost(document_count: int) -> dict[str, Any]:
    """Estimate Document AI range for a selected document count."""

    count = max(0, int(document_count or 0))
    if not count:
        return {
            "document_count": 0,
            "currency": "EUR",
            "min_cost": 0.0,
            "max_cost": 0.0,
            "range_display": "EUR 0.00",
            "unit": "selected documents",
        }
    per_doc_values = [tier.eur_per_document for tier in DOCUMENT_AI_PRICE_TIERS]
    min_cost = min(per_doc_values) * count
    max_cost = max(per_doc_values) * count
    return {
        "document_count": count,
        "currency": "EUR",
        "min_cost": round(min_cost, 4),
        "max_cost": round(max_cost, 4),
        "min_per_document": round(min(per_doc_values), 4),
        "max_per_document": round(max(per_doc_values), 4),
        "range_display": f"EUR {min_cost:.2f} - EUR {max_cost:.2f}",
        "unit": "selected documents",
        "source": "SAP Discovery Center service plan, blocks of 100 documents",
    }


def estimate_document_ai_method_cost(document_count: int = 1) -> dict[str, Any]:
    """Return a comparison-row cost estimate for SAP Document AI.

    The graph uses the lowest available per-document tier as the comparable
    cost point, while the table keeps the full min-max range visible.
    """

    estimate = estimate_document_ai_cost(document_count)
    min_cost = estimate.get("min_cost", 0.0)
    max_cost = estimate.get("max_cost", 0.0)
    return {
        "status": "estimated",
        "estimated_cost": min_cost,
        "cost_display": f"EUR {min_cost:.4f} - EUR {max_cost:.4f}",
        "currency": "EUR",
        "cost_min": min_cost,
        "cost_max": max_cost,
        "cost_basis": "SAP Document AI commercial document, cheapest-to-highest tier",
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "price": {
            "source": estimate.get("source"),
            "unit": "commercial document",
            "range": estimate.get("range_display"),
        },
    }
