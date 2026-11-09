"""Shared scenario definitions for all evaluation tests."""
from __future__ import annotations

from typing import Dict, List

SCENARIOS: List[Dict[str, str]] = [
    {
        "name": "Volvo - PDF procurement classification",
        "company": "Volvo",
        "message": (
            "The suppliers send quotations in PDF format but without material groups, "
            "so procurement clerks have to manually check the PDF, check their internal "
            "excel with hundreds of material groups + short + long description and manually "
            "assign those in S4, which takes time and is very error prone."
        ),
        "stresses": ["perplexity_problem"],
    },
    {
        "name": "Volvo - incoming quotations material group",
        "company": "Volvo",
        "message": (
            "we get incoming quotations from customers that do not have any material group, "
            "and we have to manually assign them in our S4 system following our excel which "
            "has material group short and long description."
        ),
        "stresses": ["perplexity_problem"],
    },
    {
        "name": "Illumina - inventory safety stock MEIO",
        "company": "Illumina",
        "message": (
            "Customer name is Illumina who is a Pharmaceutical Device manufacturer with "
            "concerns around inventory safety stock calculations. They are using a robust "
            "planning process using SAP's IBP tools and the output is then managed by the "
            "buyers where there is significant overrides to accommodate for the business "
            "realities and the conservativeness of the business operations.\n"
            "MEIO - Multilevel Echelon Inventory Optimization\n"
            "Inputs: Services Levels, Customer services levels, min/max internal service "
            "levels, Demand, Forward demand, past demand, sales history, forecast error, "
            "Supply, Production Procurement Transportation Lead Time, Lead Time Variability, "
            "Lot Size\n"
            "Parameters: Non Stocking Node, Safety Stock Policy, Subnetwork ID, PBR, "
            "Inventory Holding Cost Rate\n"
            "Outputs: Inventory Norms - Safety Stock, Inventory Components, Pipeline Stock, "
            "cycle stock, safety stock, SS Root Cause, Demand Variability SS, Supply "
            "variability SS, Service Variability SS"
        ),
        "stresses": ["perplexity_problem"],
    },
    {
        "name": "ENI - conflicting document evidence",
        "company": "ENI",
        "message": (
            "Users submit security and risk evaluation forms and attach supporting evidence "
            "documents. Sometimes a form answer claims a security control, certification, "
            "or policy is in place, but the attached PDFs, Word files, images, or Excel "
            "files contain outdated, partial, ambiguous, or contradictory evidence. "
            "Reviewers need help deciding what can be trusted, what should be flagged, "
            "and what needs escalation without manually reading hundreds of pages."
        ),
        "stresses": ["perplexity_problem"],
    },
]

SCENARIOS_STRESS: List[Dict[str, str]] = [
    {
        "name": "Vague - invoice automation",
        "company": "Bosch",
        "message": "We want to automate invoice processing",
        "stresses": ["cse", "perplexity", "perplexity_problem", "linkedin"],
    },
    {
        "name": "Vague - demand forecasting",
        "company": "Nestlé",
        "message": "Help us with demand forecasting",
        "stresses": ["cse", "perplexity", "perplexity_problem", "linkedin"],
    },
    {
        "name": "Vague - talent onboarding",
        "company": "Siemens",
        "message": "We need AI to help with employee onboarding",
        "stresses": ["cse", "perplexity", "perplexity_problem", "linkedin"],
    },
]

ALL_SCENARIOS = SCENARIOS + SCENARIOS_STRESS
