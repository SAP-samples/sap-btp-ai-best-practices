"""Versioned factual reference for optimizer methodology and metric questions."""

import json
import re
from pathlib import Path

CONFIGURATION_HELP = json.loads(
    Path(__file__).with_name("configuration_help.json").read_text()
)

INTERPRETATION_RULES = {
    "objective_directions": {"coverage_days": "minimize", "J_CH": "minimize"},
    "structural_validity_is_business_acceptance": False,
    "subproblem_optimality_proves_full_run": False,
    "runtime_limited_run_can_be_called_optimal": False,
    "historical_comparison_requires_common_population": True,
    "required_report_axes": [
        "structural_validation",
        "run_proof_scope",
        "business_acceptance",
    ],
    "proof_precedence": "Report the run-level limitation first; qualify optimal solver statuses by their exact block, candidate library or master subproblem scope.",
    "comparison_evidence": [
        "population identities and demand",
        "coverage basis and calendar",
        "pallet formula and PV factor",
        "constraint feasibility",
    ],
}

SECTIONS = {
    "settings": " ".join(item["text"] for item in CONFIGURATION_HELP.values()),
    "views": CONFIGURATION_HELP["views"]["text"],
    "workflow": "Uploads become immutable HANA snapshots. Each run assigns modeled FINIs to exactly one group independently per plant/SEFI block. PARETO samples trade-offs between inventory coverage and within-group changeover effort. Historical assignments are optional post-solve comparisons only.",
    "coverage": "Coverage and J_CH are both minimized; lower is better under comparable assumptions. Example: 46.36 days versus 21.00 days is 25.36 days HIGHER, never a coverage improvement. Historical numbers are incomparable until common-population and formula evidence is established. BASE_GROUP coverage days = demand_days * effective_batch_litres / group_demand_litres. ADJUSTED_GROUP substitutes the sum of pallet-adjusted FINI allocations for effective batch. WORST_FINI takes the largest demand_days * adjusted_fini_allocation / fini_annual_demand. Demand-weighted mean uses group annual demand as weight, never group size. P90 uses nearest rank ceil(0.9*n); median averages the central pair for even n.",
    "changeovers": "J_CH = sum over groups of frequency_per_week * (group_size - 1). frequency_per_week = group annual demand / (productive_weeks * effective batch). Singletons contribute zero. This measure excludes initial/between-group setups, sequence, cleaning time, monetary cost and scheduled line assignment.",
    "pallets": "Effective batch = nominal PV lot litres * canonical_factor. Allocate effective batch proportional to FINI demand. MINIMUM_ONLY uses max(allocation,pallet_litres); WHOLE_PALLET_ROUNDING uses ceil(allocation/pallet_litres)*pallet_litres. Recurrence still uses effective batch independently of pallet coverage.",
    "proof": "Structural validity, completeness of candidate pools, optimality evidence and business acceptance are independent. VALID_PARETO_POINT is a structural validation status, not business acceptance. A completed run may still be runtime-limited. Report the overall run proof limitation first: do not label a runtime-limited point optimal, even when a block, restricted library or global master solver has status OPTIMAL. Those statuses prove only their explicitly identified subproblem, not candidate completeness, full-problem optimality or the complete frontier. Restricted-library optimality may be stated only with explicit evidence and its library scope; it cannot replace a runtime-limited overall conclusion. Business acceptance must have a separately recorded decision; absent that decision report not assessed. More time cannot recover omitted candidates. Internal nondominance is not a recommendation.",
    "comparison": "A lower coverage value or lower J_CH is favorable only on a comparable basis. Before claiming historical improvement, establish the same material population and demand weights, calendar, coverage basis, pallet formula and PV factor, and report constraint-feasibility differences. Different source populations are not directly comparable: either recalculate both on an evidenced common population or state that no like-for-like improvement conclusion is available. A historical marker or greenfield VALID_PARETO_POINT alone supplies neither comparability nor business acceptance.",
    "compatibility": "Common eligible lines are alternatives supported by all group members, not a scheduled selected line. Operational volume families define compatibility evidence; FLEXIBLE may retain AVOID pairs with disclosed exceptions. Unknown volumes remain explicit. Constraints and matrix versions must be read from the specific run.",
    "constraints": "Compiled constraints: max_group_size, group_size_relaxation, must_link, cannot_link, fini_disposition, fixed_pv, allowed_pvs, required_lines. coverage_bound, freeze_assignment and volume_compatibility_override are deferred and rejected. Counterfactuals launch a new linked run; saved results are never rewritten.",
    "fields": "Inputs: FINI is finished material; SEFI is the intermediate-product block key. PV is production version with nominal batch lot. Metrics use litres, days or recurrence per week. Material, plant, SEFI, PV and line values are identifiers, not quantities. Query capabilities list registered views and their typed query contract.",
}


def reference(topic: str) -> dict:
    """Return matching sections, interpretation guardrails and source provenance.

    Explicit topic names take priority over prose keyword matches, so requesting
    proof/comparison cannot be displaced by sections mentioning those words.
    All lookups carry the same factual direction and evidence-scope guardrails.
    """
    words = re.findall(r"[a-z_]+", topic.lower())
    exact = [key for key in SECTIONS if key in words]
    related = [
        key
        for key, value in SECTIONS.items()
        if key not in exact
        and any(word in (key + " " + value).lower() for word in words)
    ]
    matches = (exact + related)[:4] or ["workflow", "fields"]
    return {
        "version": "workspace-reference-v2",
        "configuration_help": CONFIGURATION_HELP
        if any(
            w in words
            for w in (
                "settings",
                "configuration",
                "config",
                "base_group",
                "members",
                "groups",
                "views",
                "budget",
            )
        )
        else {},
        "sections": {key: SECTIONS[key] for key in matches},
        "interpretation_rules": {
            **INTERPRETATION_RULES,
            "objective_directions": dict(INTERPRETATION_RULES["objective_directions"]),
            "required_report_axes": list(INTERPRETATION_RULES["required_report_axes"]),
            "comparison_evidence": list(INTERPRETATION_RULES["comparison_evidence"]),
        },
        "available_topics": list(SECTIONS),
        "sources": [
            "production_wheel/metrics.py",
            "production_wheel/constraints/compiler.py",
            "production_wheel/greenfield_reporting.py",
        ],
    }
