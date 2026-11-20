"""Shared overview lists must match the records shown on the dedicated pages."""

from app.ecn.evidence.snapshot_views import complete_snapshot_views


def test_readiness_includes_source_warnings_once_and_lists_exact_source_objects():
    """Source issues and deterministic findings share one count and one list."""
    snapshot = {'questions': [{'code': 'BOM_MISSING'}], 'changes': [], 'counts': {}, 'objects': [
        {'number': 'P1', 'technical': True, 'origins': ['ecn'], 'types': ['Part'],
         'source_references': [{'role': 'RESULTING', 'revision': 'A', 'state': 'Released'}]},
        {'number': 'DOC1', 'technical': False, 'origins': ['ecn'], 'types': ['Drawing'],
         'source_references': [{'role': 'RESULTING', 'revision': '2', 'state': 'Released'}]},
    ]}
    warnings = [{'code': 'BAD_DATE', 'summary': 'The start date is missing.', 'count': 2}]
    complete_snapshot_views(snapshot, warnings)
    assert snapshot['counts']['attention_items'] == 2 == len(snapshot['questions'])
    assert snapshot['questions'][1]['affected_count'] == 2
    assert len(snapshot['parts']) == 1
    assert snapshot['parts'][0]['types'] == ['Part']
    assert {r['number'] for r in snapshot['reference_documents']} == {'P1', 'DOC1'}
    assert snapshot['reference_documents'][1]['state'] == 'Released'
