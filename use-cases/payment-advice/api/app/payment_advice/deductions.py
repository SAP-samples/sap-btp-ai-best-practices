"""Deterministic deduction helpers for UC-02 (no LLM, no network).

This module provides pure-function utilities to surface anomalous (negative)
line items from a payment advice payload for downstream classification.
Additional helpers (classify_deductions, format_deduction_report) will be
added here in Tasks 5 and 6.
"""
from __future__ import annotations
import copy
from typing import Any


def _is_negative(line: dict[str, Any]) -> bool:
    """Return True if the line item has a negative gross_amount or net_amount.

    Args:
        line: A single line-item dict from a payment advice payload.

    Returns:
        True when gross_amount < 0 or net_amount < 0; False otherwise.
        Non-numeric or absent values are treated as non-negative.
    """
    for key in ("gross_amount", "net_amount"):
        value = line.get(key)
        if isinstance(value, (int, float)) and value < 0:
            return True
    return False


def inspect_deductions(payload: dict[str, Any], positive_sample: int = 5) -> dict[str, Any]:
    """Return negative/anomalous lines plus a small positive sample, by sign only.

    A line is classified as negative when gross_amount < 0 or net_amount < 0.
    No invoice-prefix assumptions (PRG, SDRBG, etc.) are made.

    Args:
        payload: Payment advice dict containing a "line_items" list.
        positive_sample: Maximum number of non-negative lines to include in
            the returned sample. Defaults to 5.

    Returns:
        A dict with three keys:
            - "negative_lines": list of all line-item dicts where the sign
              check triggers.
            - "positive_sample": up to `positive_sample` non-negative lines.
            - "counts": {"negative": int, "positive": int, "total": int}.
    """
    lines = payload.get("line_items", [])
    negatives = [l for l in lines if _is_negative(l)]
    positives = [l for l in lines if not _is_negative(l)]
    return {
        "negative_lines": negatives,
        "positive_sample": positives[:positive_sample],
        "counts": {"negative": len(negatives), "positive": len(positives), "total": len(lines)},
    }


def enrich_payload(payload: dict[str, Any], interpretations: list[dict[str, Any]],
                   client_key: str = "", playbook_revision: int = 0) -> dict[str, Any]:
    """Return a deep copy of the payload with per-line interpretation fields applied.

    Merges interpretation data (document_nature, reason_code, residual_items, flags)
    into each line_item using INDEX-ALIGNED lookup against the ``interpretations``
    list.  ``interpretations[i]`` is applied to ``payload["line_items"][i]``.
    Lines whose index has no entry in the list, or whose entry is not a dict,
    receive safe defaults.  A header.interpretation summary block is written
    with client provenance and an unresolved-count metric.

    Using positional alignment (not invoice_reference keys) avoids silent
    data loss when the same invoice_reference appears on multiple sibling lines
    with different deduction codes or sign conventions.

    Semantics:
        - Input payload is NEVER mutated (deep copy is returned).
        - document_nature  defaults to "unknown" when no interpretation exists.
        - reason_code      defaults to "323" when the interpretation dict has no
          "reason_code" key; "323" lines also receive the flag
          "reason-code-defaulted" (appended exactly once).
        - reason_code=None (explicitly supplied) means the model resolved the line;
          no default flag is added.
        - residual_items   defaults to [].
        - flags            starts from interp["flags"] (if any) plus any defaults.
        - unresolved_count = number of lines where reason_code=="323" OR flags is
          non-empty.

    Args:
        payload:            Payment advice dict containing a "header" dict and a
                            "line_items" list.
        interpretations:    Index-aligned list of interpretation dicts.  Element i
                            corresponds to line_items[i].  Each dict may contain
                            keys: document_nature, reason_code, residual_items, flags.
                            A missing or non-dict element triggers per-field defaults.
        client_key:         Identifier for the customer/client (stored in summary).
        playbook_revision:  Revision number of the playbook used (stored in summary).

    Returns:
        A new dict (deep copy of payload) with enriched line_items and a
        header["interpretation"] summary block.
    """
    out = copy.deepcopy(payload)
    unresolved = 0

    for i, line in enumerate(out.get("line_items", [])):
        # Index-aligned lookup: interpretations[i] for line i.
        # Fall back to empty dict when index is out of range or entry is not a dict.
        interp = (
            interpretations[i]
            if isinstance(interpretations, list)
            and i < len(interpretations)
            and isinstance(interpretations[i], dict)
            else {}
        )

        # Document nature: use interpretation value or default to "unknown"
        nature = interp.get("document_nature", "unknown")

        # Flags: start from any flags already in the interpretation
        flags = list(interp.get("flags", []))

        # Reason code: if the key exists in interp, trust it (even None);
        # otherwise default to "323" and mark with the defaulted flag
        if "reason_code" in interp:
            reason_code = interp["reason_code"]
        else:
            reason_code = "323"
            flags.append("reason-code-defaulted")

        # Also flag "323" that was explicitly provided without the defaulted marker
        if reason_code == "323" and "reason-code-defaulted" not in flags:
            flags.append("reason-code-defaulted")

        line["document_nature"] = nature
        line["reason_code"] = reason_code
        line["residual_items"] = list(interp.get("residual_items", []))
        line["flags"] = flags
        # Carry the agent's free-text rationale forward when present
        if "rationale" in interp:
            line["rationale"] = interp["rationale"]

        # Count as unresolved if reason_code is the catch-all or any flags present
        if reason_code == "323" or flags:
            unresolved += 1

    # Write provenance summary into header
    header = out.setdefault("header", {})
    header["interpretation"] = {
        "client_key": client_key,
        "playbook_revision": playbook_revision,
        "unresolved_count": unresolved,
    }
    return out


# ---------------------------------------------------------------------------
# UC-02 selective-analysis helpers
# ---------------------------------------------------------------------------

def needs_analysis(line: dict[str, Any]) -> bool:
    """Return True when a line is a deduction or anomaly requiring agent analysis.

    A line needs analysis when ANY of the following are true:
        - gross_amount is a negative number
        - net_amount is a negative number
        - deduction_reason is a non-empty, non-whitespace string

    Plain invoice lines (positive amounts, no deduction_reason) return False.
    Missing or non-numeric amounts are treated as non-negative.

    Args:
        line: A single line-item dict from a payment advice payload.

    Returns:
        True if the line requires agent analysis; False for plain invoice lines.
    """
    for key in ("gross_amount", "net_amount"):
        v = line.get(key)
        if isinstance(v, (int, float)) and v < 0:
            return True
    reason = line.get("deduction_reason")
    return isinstance(reason, str) and reason.strip() != ""


def select_analysis_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return analysis lines tagged with their original position, preserving order.

    Iterates ``payload["line_items"]`` and collects every line where
    ``needs_analysis`` returns True.  Each returned dict is a shallow copy of
    the line augmented with a ``line_index`` key holding its position in the
    original list so the agent can refer back to the source line without
    re-scanning the full payload.

    Args:
        payload: Payment advice dict containing a ``line_items`` list.

    Returns:
        A list of dicts, one per analysis line, each with ``{"line_index": i, **line}``.
        Order matches the original ``line_items`` order.  Empty list when no
        analysis lines are present.
    """
    out: list[dict[str, Any]] = []
    for i, line in enumerate(payload.get("line_items", [])):
        if needs_analysis(line):
            out.append({"line_index": i, **line})
    return out


def assemble_interpretations(
    payload: dict[str, Any],
    agent_entries: list[dict[str, Any]],
    engine_entries: dict[int, dict[str, Any]] | None = None,
    analysis_indexes: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Build an index-aligned interpretation list from the agent's subset entries.

    Given the full payload and the agent's response (covering only analysis lines),
    return one interpretation dict per line in ``payload["line_items"]``:

        - Plain lines  → a brand-new dict with fresh ``residual_items`` and ``flags``
          lists each time, so callers can mutate individual entries without affecting
          siblings (no shared list objects).
        - Analysis lines with a matching ``agent_entries`` item (by ``line_index``)
          → that item minus the ``line_index`` key (all other fields, including
          ``rationale``, are preserved).
        - Analysis lines with NO matching entry → an "unknown / 323" escalation
          dict with ``needs-customer-confirmation`` flag and a fixed rationale.

    Args:
        payload:       Payment advice dict containing a ``line_items`` list.
        agent_entries: List of interpretation dicts produced by the agent.  Each
                       must contain a numeric ``line_index`` key that maps it back
                       to its position in ``payload["line_items"]``.  Non-dict
                       entries and entries without an integer ``line_index`` are
                       silently ignored.
        engine_entries: Interpretations already settled by documented customer
                       rules (``rule_engine.route_lines``), by line index; they win.
        analysis_indexes: Lines that were sent to the agent.  Defaults to every
                       line where ``needs_analysis`` is true; the rule engine also
                       routes positive lines that match a deduction rule.

    Returns:
        A list aligned to ``payload["line_items"]`` with one interpretation dict
        per line.
    """
    # Build a lookup from line_index → entry (skip any malformed entries)
    by_index: dict[int, dict[str, Any]] = {
        e["line_index"]: e
        for e in agent_entries
        if isinstance(e, dict) and isinstance(e.get("line_index"), int)
    }

    result: list[dict[str, Any]] = []
    for i, line in enumerate(payload.get("line_items", [])):
        if engine_entries and i in engine_entries:
            entry = engine_entries[i]
            result.append({**entry, "residual_items": list(entry.get("residual_items", [])),
                           "flags": list(entry.get("flags", []))})
            continue
        analysed = i in analysis_indexes if analysis_indexes is not None else needs_analysis(line)
        if not analysed:
            # Build a fresh dict with fresh lists for each plain line so that
            # mutating one entry's residual_items or flags never aliases another.
            result.append({
                "document_nature": "invoice",
                "reason_code": None,
                "residual_items": [],
                "flags": [],
            })
            continue

        entry = by_index.get(i)
        if entry is None:
            # The agent produced no interpretation for this analysis line; escalate.
            result.append({
                "document_nature": "unknown",
                "reason_code": "323",
                "residual_items": [],
                "flags": ["needs-customer-confirmation"],
                "rationale": "agent returned no interpretation for this analysis line",
            })
        else:
            # Strip line_index before returning; keep all other agent-supplied fields.
            result.append({k: v for k, v in entry.items() if k != "line_index"})

    return result


# ---------------------------------------------------------------------------
# UC-02 full-document information helpers (deterministic, no LLM/network)
# ---------------------------------------------------------------------------

def document_statistics(payload: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate statistics for all line items in a payment advice payload.

    Counts are computed in a single pass over ``payload["line_items"]``.
    Anomalous lines are those where ``needs_analysis`` returns True (negative
    amounts OR non-empty deduction_reason).  Plain lines are the complement.
    Negative lines are those with gross_amount < 0 OR net_amount < 0.

    Numeric totals (gross_total, net_total) sum only lines where the
    respective value is an int or float; non-numeric or absent values are
    skipped.

    Args:
        payload: Payment advice dict.  Must contain a ``line_items`` list.
            A ``header`` dict with a ``payment_currency`` key is optional.

    Returns:
        A dict with keys:
            - total_lines (int): total number of line items.
            - anomalous_count (int): lines where needs_analysis is True.
            - plain_count (int): total_lines - anomalous_count.
            - negative_count (int): lines with gross_amount < 0 or net_amount < 0.
            - with_deduction_reason_count (int): lines with a non-empty, non-whitespace
              deduction_reason string.
            - gross_total (float): sum of numeric gross_amount values.
            - net_total (float): sum of numeric net_amount values.
            - currency (str | None): payload["header"]["payment_currency"] or None.
            - distinct_customer_account_references (list[str]): sorted unique
              non-empty customer_account_reference strings.
    """
    lines: list[dict[str, Any]] = payload.get("line_items", [])
    anomalous_count = 0
    negative_count = 0
    with_deduction_reason_count = 0
    gross_total = 0.0
    net_total = 0.0
    account_refs: set[str] = set()

    for line in lines:
        # anomalous: uses the existing needs_analysis function (neg amounts OR deduction_reason)
        if needs_analysis(line):
            anomalous_count += 1

        # negative: only sign-based (gross < 0 or net < 0)
        if _is_negative(line):
            negative_count += 1

        # deduction_reason present and non-empty
        reason = line.get("deduction_reason")
        if isinstance(reason, str) and reason.strip():
            with_deduction_reason_count += 1

        # numeric-only totals
        gross = line.get("gross_amount")
        if isinstance(gross, (int, float)):
            gross_total += gross

        net = line.get("net_amount")
        if isinstance(net, (int, float)):
            net_total += net

        # customer account references
        ref = line.get("customer_account_reference")
        if isinstance(ref, str) and ref.strip():
            account_refs.add(ref.strip())

    total_lines = len(lines)
    header = payload.get("header") or {}
    currency = header.get("payment_currency")

    return {
        "total_lines": total_lines,
        "anomalous_count": anomalous_count,
        "plain_count": total_lines - anomalous_count,
        "negative_count": negative_count,
        "with_deduction_reason_count": with_deduction_reason_count,
        "gross_total": gross_total,
        "net_total": net_total,
        "currency": currency,
        "distinct_customer_account_references": sorted(account_refs),
    }


def list_invoice_references(
    payload: dict[str, Any],
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Return a paginated flat list of invoice_reference values in line order.

    Only lines with a real, non-empty (non-whitespace) string invoice_reference
    are included.  Lines whose invoice_reference is absent, None, or blank are
    skipped so that the list is consistent with what ``fetch_invoices`` can
    actually retrieve (a blank ref would never match a fetch request).
    Duplicates across lines are preserved.

    Args:
        payload: Payment advice dict containing a ``line_items`` list.
        offset: Zero-based start index into the flat reference list.
        limit: Maximum number of references to return per page.

    Returns:
        A dict with keys:
            - total (int): count of lines with a real invoice_reference.
            - offset (int): the requested offset.
            - limit (int): the requested limit.
            - references (list[str]): the slice ``refs[offset:offset+limit]``.
            - has_more (bool): True when ``offset + limit < total``.
    """
    lines: list[dict[str, Any]] = payload.get("line_items", [])
    # Only include lines that carry a real, non-empty invoice_reference string.
    # This keeps list and fetch consistent: a line skipped here is also unmatchable
    # in fetch_invoices (which gets None for the absent key and cannot match it).
    refs = [
        r for line in lines
        if isinstance(r := line.get("invoice_reference"), str) and r.strip()
    ]
    total = len(refs)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "references": refs[offset: offset + limit],
        "has_more": offset + limit < total,
    }


def fetch_invoices(
    payload: dict[str, Any],
    references: "str | list[str]",
) -> dict[str, Any]:
    """Return all line items whose invoice_reference matches the requested set.

    Accepts a single reference string or a list of strings.  The returned
    invoices are in the original ``line_items`` order.  Each matched line is
    augmented with a ``line_index`` key holding its position in the original
    list.  References that match no line are collected in ``not_found``
    (preserving request order, deduplicated, first occurrence wins).

    Args:
        payload: Payment advice dict containing a ``line_items`` list.
        references: A single invoice_reference string or a list of such strings
            to look up.

    Returns:
        A dict with keys:
            - invoices (list[dict]): matched lines in line order, each with
              ``{"line_index": i, **line}``.
            - not_found (list[str]): requested references that matched no line,
              in the order they appeared in the request (deduplicated).
    """
    # Normalise str → list[str]
    if isinstance(references, str):
        references = [references]

    # Build ordered-deduped request list (preserve first occurrence)
    seen_request: set[str] = set()
    ordered_refs: list[str] = []
    for r in references:
        if r not in seen_request:
            seen_request.add(r)
            ordered_refs.append(r)

    request_set: set[str] = set(ordered_refs)

    # Track which requested refs actually matched at least one line
    matched_refs: set[str] = set()
    invoices: list[dict[str, Any]] = []

    for i, line in enumerate(payload.get("line_items", [])):
        inv_ref = line.get("invoice_reference")
        if inv_ref in request_set:
            matched_refs.add(inv_ref)
            invoices.append({"line_index": i, **line})

    not_found = [r for r in ordered_refs if r not in matched_refs]

    return {
        "invoices": invoices,
        "not_found": not_found,
    }
