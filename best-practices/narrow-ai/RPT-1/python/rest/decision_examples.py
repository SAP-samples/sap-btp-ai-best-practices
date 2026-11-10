"""Synthetic support examples for teaching and testing priority decisions.

The labels follow an illustrative policy, not a universal support SLA. Examples
cover each tone at each priority so anger alone does not define urgency.
"""

from __future__ import annotations

import copy


def expanded_triage_request(request: dict) -> dict:
    """Return a copy of a triage request with 18 balanced synthetic examples.

    Input is the notebook's typed three-question request. Output keeps its state
    and question types, clarifies the priority policy, and replaces examples.
    No customer data is read or stored.
    """
    expanded = copy.deepcopy(request)
    expanded["questions"]["priority"]["instructions"] = (
        "Apply this support policy: can wait for feedback or future ideas with "
        "no action needed; this week for routine information or a single invoice "
        "correction without blocked work; today for blocked access, an outage, "
        "repeated incorrect charges, or an explicit deadline today. Tone alone "
        "does not determine priority."
    )
    cases = [
        ("Love the new feature, thank you! No action needed.", False, "calm", "0"),
        ("Thanks for considering a dashboard improvement for next year.", False, "calm", "0"),
        ("Feedback only: the menu could be shorter in a future release.", False, "neutral", "0"),
        ("For your roadmap: a yearly invoice summary would be useful someday.", True, "neutral", "0"),
        ("I hate the new colors. Just feedback, no support action needed.", False, "angry", "0"),
        ("Your invoice layout is awful. Change it in a future release, no rush.", True, "angry", "0"),
        ("Could you kindly send last month's invoice when you have time this week?", True, "calm", "1"),
        ("Thank you for checking a single incorrect charge on my last bill this week; my account still works.", True, "calm", "1"),
        ("Where can I download last year's invoices?", True, "neutral", "1"),
        ("When is the next release? Please reply sometime this week.", False, "neutral", "1"),
        ("This one bill is wrong, $99 instead of $49. Fix it this week; nothing is blocked.", True, "angry", "1"),
        ("I am frustrated by your documentation. Send the guide this week, no deadline today.", False, "angry", "1"),
        ("Please help today: all users are blocked from signing in. Thank you.", False, "calm", "2"),
        ("Could you please reverse repeated duplicate charges today? Thank you.", True, "calm", "2"),
        ("Production is unavailable for every user; work is blocked now.", False, "neutral", "2"),
        ("The invoice must be corrected before the payment deadline at 3pm today.", True, "neutral", "2"),
        ("My account is locked AGAIN and I cannot work. Fix access today!", False, "angry", "2"),
        ("You billed me three times AGAIN. Reverse these repeated charges today!", True, "angry", "2"),
    ]
    expanded["examples"] = [
        {"state": state, "answers": {"billing": billing, "tone": tone, "priority": priority}}
        for state, billing, tone, priority in cases
    ]
    return expanded


def priority_checks() -> list[tuple[str, str]]:
    """Return nine fresh synthetic messages and policy-based expected levels.

    These inputs are excluded from the 18 examples. Output pairs contain text
    and a level key; this tiny authored check is not a production benchmark.
    """
    return [
        ("Nice update! I just wanted to share positive feedback.", "0"),
        ("Suggestion for next year: add more chart colors. No action needed now.", "0"),
        ("The new font is terrible. This is feedback for a future redesign only.", "0"),
        ("Please email a copy of my February bill sometime this week.", "1"),
        ("A single fee on my statement is incorrect; access works and there is no urgent deadline.", "1"),
        ("Where is the manual for exporting reports? A reply this week is fine.", "1"),
        ("Our whole department cannot sign in and all work has stopped. Help now.", "2"),
        ("Duplicate charges happened for the fourth time. Reverse them today.", "2"),
        ("Please correct this invoice before today's payment cutoff in two hours.", "2"),
    ]
