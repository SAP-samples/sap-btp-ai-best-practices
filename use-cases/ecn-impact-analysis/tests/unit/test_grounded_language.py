"""Guard explicit replacement direction against the deterministic BOM pairs."""

from app.ecn.llm.grounded_language import normalize_replacement_prose


def test_reversed_passive_and_active_wording_follow_recorded_bom_direction():
    """Correct exact reversed assertions without changing unrelated claims or IDs."""
    changes = {'lines': [{'change': 'REPLACED', 'outgoing': {'number': 'OLD0'}, 'incoming': {'number': 'NEWA'}}]}
    payload = {'factor': {'assessment': '- New part NEWA is replaced by OLD0. OLD0 replaces NEWA.',
                          'evidence_ids': ['E-0123456789abcdef']}}
    result = normalize_replacement_prose(payload, changes)
    assert result['factor']['assessment'] == '- OLD0 is replaced by NEWA. NEWA replaces OLD0.'
    assert result['factor']['evidence_ids'] == payload['factor']['evidence_ids']
    assert payload['factor']['assessment'].startswith('- New part NEWA')


def test_correct_unlinked_or_bidirectional_pairs_are_not_rewritten():
    """Never guess a pair or override pairs that legitimately run in both directions."""
    pair = {'change': 'REPLACED', 'outgoing': {'number': 'OLD0'}, 'incoming': {'number': 'NEWA'}}
    payload = {'assessment_markdown': 'OLD0 is replaced by NEWA. NEWA replaces OLD0. OTHER is replaced by OLD0.'}
    assert normalize_replacement_prose(payload, {'lines': [pair]}) == payload
    reverse = {'change': 'REPLACED', 'outgoing': pair['incoming'], 'incoming': pair['outgoing']}
    reversed_text = {'assessment': 'NEWA is replaced by OLD0.'}
    assert normalize_replacement_prose(reversed_text, {'lines': [pair, reverse]}) == reversed_text
    assert normalize_replacement_prose(reversed_text, {'lines': [{**pair, 'change': 'ADDED'}]}) == reversed_text
    conditional = {'assessment': 'Confirm whether NEWA is replaced by OLD0. If OLD0 replaces NEWA, check timing.'}
    assert normalize_replacement_prose(conditional, {'lines': [pair]}) == conditional


def test_normalization_preserves_strict_model_enums_when_pairs_exist():
    """Real replacements must not turn validated severity enums into plain strings."""
    from app.ecn.analysis_models import ImpactSeverity
    from app.ecn.llm.staged import FactorStageResult
    changes = {'lines': [{'change': 'REPLACED', 'outgoing': {'number': 'OLD0'}, 'incoming': {'number': 'NEWA'}}]}
    payload = {'factor': {'severity': ImpactSeverity.MEDIUM, 'assessment': '- NEWA is replaced by OLD0.',
                          'evidence_ids': ['E-0123456789abcdef'], 'missing_evidence': []}, 'additional_next_steps': []}
    result = normalize_replacement_prose(payload, changes)
    assert result['factor']['severity'] is ImpactSeverity.MEDIUM
    assert FactorStageResult.model_validate(result).factor.assessment == '- OLD0 is replaced by NEWA.'
