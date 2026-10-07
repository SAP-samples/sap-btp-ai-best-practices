"""Shared rule-engine evaluation for legacy uploads and persisted workspace analyses."""

from .parser import parse_offer_file
from .engine import EligibilityEngine


def evaluate_offer(content, filename, purchase_date, settings, *, strict=False):
    """Parse one upload and return input rows, rule results, and legacy categories."""
    invoices = parse_offer_file(content, filename, strict=strict)
    results, funded, non_funded = EligibilityEngine(settings).analyze_batch(invoices, purchase_date)
    return invoices, results, funded, non_funded
