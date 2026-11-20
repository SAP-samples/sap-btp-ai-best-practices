"""Keep chat replacement dates aligned with the same BOM facts shown by the UI."""

from types import SimpleNamespace

import pytest

from app.ecn.assistant.tools import create_tools
from app.ecn.evidence.bom_changes import build_bom_changes
from tests.unit.test_bom_changes import replacement_rows


@pytest.mark.asyncio
async def test_chat_can_page_exact_bom_replacement_dates_and_sources():
    """Chat must retrieve manufacturing cut-over from BOM rows, not ECN JSON dates."""
    snapshot = {'assessment_id': 'selected', 'assessment_date': '2026-09-24',
                'bom_changes': build_bom_changes('ECN345274', replacement_rows(), coverage_complete=True)}
    snapshots = SimpleNamespace(get=lambda identity: snapshot if identity == 'selected' else None)
    tools = {t.name: t for t in create_tools(snapshots, None, None, 'selected', set(), [], set())}
    assert 'list_bom_changes' in tools
    result = (await tools['list_bom_changes'].ainvoke({'number': 'QKX4638810'}))['result']
    assert result['total'] == 1 and not result['has_next']
    [line] = result['items']
    assert line['outgoing']['date'] == '2026-06-29'
    assert line['manufacturing_date'] == '2026-06-30'
    assert line['engineering_date'] == '2026-01-16'
    assert [p['plant_code'] for p in line['plants']] == ['PL10']
    assert len(line['sources']) == 3
    assert {m['status'] for m in result['materials']} == {'FULLY_PHASED_OUT', 'NEW_PART'}


@pytest.mark.asyncio
async def test_overview_excludes_full_reference_arrays_and_keeps_bom_totals():
    """Large document/part arrays must not discard the whole chat overview."""
    changes = build_bom_changes('ECN345274', replacement_rows(), coverage_complete=True)
    snapshot = {'assessment_id': 'selected', 'counts': {'technical_objects': 200}, 'questions': [],
                'bom_changes': changes, 'plant_impact': {'plants': []},
                'parts': [{'number': 'part', 'text': 'x' * 500} for _ in range(200)],
                'reference_documents': [{'text': 'y' * 500} for _ in range(200)], 'programs': []}
    snapshots = SimpleNamespace(overview=lambda identity: snapshot)
    tools = {t.name: t for t in create_tools(snapshots, None, None, 'selected', set(), [], set())}
    response = await tools['get_ecn_overview'].ainvoke({})
    assert not response.get('detail_omitted')
    result = response['result']
    assert 'parts' not in result and 'reference_documents' not in result
    assert result['bom_changes']['counts']['REPLACED'] == 2
