"""Keep large BOM-driven assessments within the model budget without losing totals."""

import copy
import json

from app.ecn.evidence.bom_changes import build_bom_changes
from app.ecn.evidence.plant_impact import build_plant_impact
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.llm.two_pass import FINAL_MAX_CHARACTERS, build_final_input
from tests.fixtures.builders import make_evidence_bundle
from tests.unit.test_bom_changes import loc


def test_large_bom_context_preserves_totals_and_reports_omitted_examples():
    """Hundreds of ordinary additions and plants remain analyzable and traceable."""
    rows = [loc(f'PART{i}', f'ASSEMBLY{i}', start='2026-01-02', change='SIM-ECN-001',
                plants=({'plant_code': f'P{i}', 'plant_name': f'Plant {i}'},)) for i in range(150)]
    changes = build_bom_changes('SIM-ECN-001', rows)
    snapshot = {'objects': [], 'questions': [], 'changes': [], 'bom_changes': changes,
                'plant_impact': build_plant_impact(changes, [], {})}
    before = copy.deepcopy(snapshot)
    projection = build_synthesis_projection(make_evidence_bundle(), snapshot)
    assert len(json.dumps(build_final_input(projection), ensure_ascii=False)) <= FINAL_MAX_CHARACTERS
    context = projection.snapshot_context
    for name, key in [('bom_changes', 'lines'), ('bom_changes', 'materials'), ('plant_impact', 'plants')]:
        section = context[name]
        assert section[f'{key}_total'] == 150
        assert section[f'{key}_omitted'] == 150 - len(section[key]) > 0
    assert context['bom_changes']['counts']['ADDED'] == 150
    assert context['plant_impact']['status_counts'] == {'CONFIRMED': 150}
    assert snapshot == before
