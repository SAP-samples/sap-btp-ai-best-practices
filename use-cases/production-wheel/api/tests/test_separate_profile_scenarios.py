"""Meeting strategy examples remain isolated requests rather than cumulative defaults."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from production_wheel.business_rules import evaluate

ROOT = Path(__file__).resolve().parents[2]


def test_separate_scenario_boundaries_and_fresh_defaults():
    """Each case uses only its own predicates, and prior cases cannot alter the next."""
    if not (ROOT / 'scripts/validate_full_profile_portfolio.py').is_file():
        pytest.skip('acceptance harness scripts and docs are not part of this checkout')
    harness = runpy.run_path(str(ROOT / 'scripts/validate_full_profile_portfolio.py'))
    cases = runpy.run_path(str(ROOT / 'scripts/plant_profile_acceptance_cases.py'))['cases']()
    catalog = json.loads((ROOT / 'docs/plant-profile-direct-solve-requests.json').read_text())
    defaults = WorkspaceService(MemoryRepository()).plant_profile_defaults()
    args = SimpleNamespace(plant='PL01', horizon_days=80, beam_width=50, exact_subsets=5000, candidate_ceiling=100000)
    builder = harness['scenario_request']
    for case in cases:
        if case.get('clarification'):
            continue
        request = builder(case['id'], catalog, defaults, args)
        assert request.config.business_rules == ()
        for data, expected in case['checks']:
            actual = all(not evaluate(rule.when, data) or evaluate(rule.assertion, data) for rule in request.constraints)
            assert actual == expected, case['id']
    preferred = builder('line6_preference_common_cap', catalog, defaults, args)
    assert preferred.config.preferred_line == '6'
    assert 'member.runner' not in json.dumps(preferred.model_dump(mode='json'))
    for name in ('common_line_cap', 'assigned_line_cap'):
        request = builder(name, catalog, defaults, args)
        assert request.config.preferred_line is None
        assert 'member.runner' not in json.dumps(request.model_dump(mode='json'))
    baseline = builder('baseline', catalog, defaults, args)
    assert baseline.constraints == () and baseline.config.business_rules == ()
    assert baseline.config.preferred_line is None
    assert defaults['rules'] == []
    assert all(row['title'] != 'assigned line cap high' for row in catalog['scenarios'])
    assert catalog['prior_combined_diagnostics'][0]['default'] is False
