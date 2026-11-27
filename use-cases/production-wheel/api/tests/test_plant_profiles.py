"""Profile persistence, immutable configuration and source compatibility contracts."""
import pytest
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from test_workspace import dataset


def fixture():
    """Return a service, published source and reviewed profile."""
    service = WorkspaceService(MemoryRepository())
    source = dataset(service)
    service.publish_dataset(source['dataset_id'])
    profile = service.create_plant_profile({'name': 'Plant plan', 'plant': 'P', **service.plant_profile_defaults()})
    return service, source, profile


def test_profile_revision_snapshot_and_confirmed_removal():
    """Edits and removal preserve frozen drafts, while stale edits are rejected."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'], 'Scenario')
    updated = service.update_plant_profile(profile['profile_id'], {'revision': 1, 'name': 'Changed'})
    assert service.get_draft(draft['draft_id'])['plant_profile']['name'] == 'Plant plan'
    with pytest.raises(ValueError, match='revision'):
        service.update_plant_profile(profile['profile_id'], {'revision': 1, 'name': 'Stale'})
    with pytest.raises(ValueError, match='confirm'):
        service.remove_plant_profile(profile['profile_id'], updated['revision'], False)
    service.remove_plant_profile(profile['profile_id'], updated['revision'], True)
    assert not service.list_plant_profiles()
    with pytest.raises(ValueError, match='removed'):
        service.submit({'draft_id': draft['draft_id'], 'revision': 1, 'idempotency_key': 'launch'})


def test_profile_settings_cannot_be_overridden_and_horizon_is_checked():
    """Profile values remain fixed and observed source evidence blocks mismatches."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    with pytest.raises(ValueError, match='profile'):
        service.update_draft(draft['draft_id'], 1, {'request': {'config': {'demand_days': 100}}})
    rows = service.repo.rows(source['dataset_id'], 'fini_master')
    rows[0]['source_horizon_days'] = 100
    service.repo.replace_tables(source['dataset_id'], {'fini_master': rows})
    result = service.validate_draft(draft['draft_id'])
    assert not result['valid'] and 'horizon' in result['errors'][0]


def test_scenario_may_change_compatibility_mode_but_not_matrix_pairs():
    """The profile seeds the mode; a scenario override is explicit and reaches the solver."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    assert draft['request']['config']['matrix_mode'] == profile['settings']['matrix_mode'] == 'FLEXIBLE'
    assert draft['parameter_sources']['request.config.matrix_mode']['origin'] == 'plant_profile'
    updated = service.update_draft(draft['draft_id'], 1, {'request': {'config': {'matrix_mode': 'DIAGNOSTIC'}}})
    assert updated['request']['config']['matrix_mode'] == 'DIAGNOSTIC'
    assert updated['parameter_sources']['request.config.matrix_mode'] == {'origin': 'explicit'}
    assert service.compiled_draft(updated).config.matrix_mode.value == 'DIAGNOSTIC'
    # The frozen profile snapshot keeps its own default; only the scenario changed.
    assert updated['plant_profile']['settings']['matrix_mode'] == 'FLEXIBLE'
    with pytest.raises(ValueError, match='profile'):
        service.update_draft(draft['draft_id'], updated['revision'], {'request': {'config': {'matrix_pairs': []}}})


def test_profile_api_and_frozen_submission():
    """HTTP profile CRUD preserves historical execution after settings edits."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers.workspace import router
    from app.workspace.dependencies import get_service
    service, source, profile = fixture()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)
    assert client.get('/api/plant-profile-defaults').status_code == 200
    assert len(client.get('/api/plant-profiles').json()['profiles']) == 1
    draft = client.post('/api/run-drafts', json={'dataset_id': source['dataset_id'], 'plant_profile_id': profile['profile_id'], 'title': 'Base'}).json()
    validation = service.validate_draft(draft['draft_id'])
    assert validation['valid'], validation
    run = service.submit({'draft_id': draft['draft_id'], 'revision': 1, 'idempotency_key': 'freeze'})
    latest = service.get_plant_profile(profile['profile_id'])
    service.update_plant_profile(profile['profile_id'], {'revision': latest['revision'], 'settings': {**profile['settings'], 'runner_thresholds': {**profile['settings']['runner_thresholds'], 'high_max_days': 5}}})
    assert service.compiled_draft(run).config.runner_thresholds.high_max_days == 10
    assert run['plant_profile_revision'] == 1
    assert client.patch('/api/plant-profiles/'+profile['profile_id'], json={'revision': 1, 'name': 'stale'}).status_code == 409


def line_cap_rule(constraint_id='line-six-cap'):
    """Build an assigned-line cap as the assistant would propose it."""
    return {'kind': 'group_rule', 'constraint_id': constraint_id, 'source_text': 'Line L1 at most 2 FINIs',
        'description': 'Groups assigned to line L1 contain at most two FINIs.',
        'when': {'op': 'eq', 'args': [{'op': 'field', 'field': 'group.selected_line'}, {'op': 'literal', 'value': 'L1'}]},
        'assertion': {'op': 'lte', 'args': [{'op': 'field', 'field': 'group.size'}, {'op': 'literal', 'value': 2}]}}


def test_profile_changes_add_replace_remove_and_settings_with_revision_checks():
    """Approved changes are validated, scoped to the profile plant and revision-protected."""
    service, _, profile = fixture()
    pid = profile['profile_id']
    added = service.apply_plant_profile_change(pid, profile['revision'], {'action': 'add_constraint', 'constraint': line_cap_rule()})
    rule = added['rules'][-1]
    assert rule['scope']['plant'] == profile['plant'] and rule['approval_status'] == 'approved'
    with pytest.raises(ValueError, match='stale'):
        service.apply_plant_profile_change(pid, profile['revision'], {'action': 'remove_constraint', 'constraint_id': 'line-six-cap'})
    with pytest.raises(ValueError, match='already exists'):
        service.apply_plant_profile_change(pid, added['revision'], {'action': 'add_constraint', 'constraint': line_cap_rule()})
    with pytest.raises(ValueError, match='does not exist'):
        service.apply_plant_profile_change(pid, added['revision'], {'action': 'remove_constraint', 'constraint_id': 'missing'})
    replaced_rule = {**line_cap_rule(), 'description': 'changed'}
    replaced = service.apply_plant_profile_change(pid, added['revision'], {'action': 'replace_constraint', 'constraint': replaced_rule})
    assert replaced['rules'][-1]['description'] == 'changed'
    patched = service.apply_plant_profile_change(pid, replaced['revision'], {'action': 'update_settings', 'settings': {'maximum_group_size': 3}})
    assert patched['settings']['maximum_group_size'] == 3 and patched['rules'] == replaced['rules']
    with pytest.raises(ValueError):
        service.apply_plant_profile_change(pid, patched['revision'], {'action': 'update_settings', 'settings': {'maximum_group_size': 0}})
    removed = service.apply_plant_profile_change(pid, patched['revision'], {'action': 'remove_constraint', 'constraint_id': 'line-six-cap'})
    assert all(r['constraint_id'] != 'line-six-cap' for r in removed['rules'])
    kept = service.update_plant_profile(pid, {'revision': removed['revision'], 'name': 'renamed'})
    assert kept['rules'] == removed['rules']


def test_scenario_can_enable_line_assignment_without_changing_frozen_profile():
    """An additive assigned-line rule may enable assignment for this scenario only."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    updated = service.update_draft(draft['draft_id'], 1, {'request': {'constraints': [{
        'kind': 'group_rule', 'constraint_id': 'assigned-line', 'approval_status': 'approved',
        'assertion': {'op': 'eq', 'args': [{'op': 'field', 'field': 'group.selected_line'}, {'op': 'literal', 'value': 'L1'}]}
    }]}})
    compiled = service.compiled_draft(updated)
    assert compiled.config.assign_filling_lines
    assert updated['plant_profile']['rules'] == []


def test_scenario_may_tighten_but_not_relax_frozen_group_cap():
    """Legacy per-block caps intersect with the frozen maximum without replacing it."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    updated = service.update_draft(draft['draft_id'], 1, {'request': {'constraints': [{
        'kind': 'max_group_size', 'constraint_id': 'smaller', 'maximum': 2, 'approval_status': 'approved',
    }]}})
    compiled = service.compiled_draft(updated)
    assert compiled.config.group_size_overrides[0].maximum == 2
    assert updated['plant_profile']['settings']['maximum_group_size'] == 7
    relaxed = service.update_draft(draft['draft_id'], updated['revision'], {'request': {'constraints': [{
        'kind': 'max_group_size', 'constraint_id': 'larger', 'maximum': 8, 'approval_status': 'approved',
    }]}})
    assert service.compiled_draft(relaxed).config.group_size_overrides[0].maximum == 7
    with pytest.raises(ValueError, match='profile'):
        service.update_draft(draft['draft_id'], relaxed['revision'], {'request': {'config': {'group_size_overrides': [{'plant': 'P', 'sefi': 'S', 'maximum': 8}]}}})


@pytest.mark.parametrize('use_profile', [False, True])
def test_submitted_business_rules_replay_once(use_profile):
    """Source requests prevent compiled scenario rules from duplicating on worker replay."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'] if use_profile else None)
    updated = service.update_draft(draft['draft_id'], 1, {'request': {'constraints': [
        {'kind': 'group_rule', 'constraint_id': 'group-check', 'approval_status': 'approved',
         'assertion': {'op': 'gte', 'args': [{'op': 'field', 'field': 'group.size'}, {'op': 'literal', 'value': 1}]}},
        {'kind': 'selection_bound', 'constraint_id': 'total-check', 'approval_status': 'approved',
         'measure': {'op': 'field', 'field': 'group.size'}, 'lower': 1, 'upper': 2},
    ]}})
    run = service.submit({'draft_id': draft['draft_id'], 'revision': updated['revision'], 'idempotency_key': 'replay'})
    assert len(run['request']['config']['business_rules']) == 2
    assert run['source_request']['config']['business_rules'] == []
    compiled = service.compiled_draft(run)
    assert [rule.constraint_id for rule in compiled.config.business_rules] == ['group-check', 'total-check']


def test_uncovered_diagnostic_distinguishes_source_impossibility():
    """Missing only-line6 coverage with no block high runner is visible without data copies."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[2] / 'scripts/validate_full_profile_portfolio.py'
    if not path.is_file():
        pytest.skip('acceptance harness scripts are not part of this checkout')
    spec = importlib.util.spec_from_file_location('portfolio_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from production_wheel.schemas import RunConfig
    # B (500 days) is low by days and ratio; A (100 days) is the SEFI's fastest, so it is high.
    members = [SimpleNamespace(block_key=('P', 'S'), fini_id=f, eligible_lines=frozenset({'6'}), lot_size_considered_litres=lot,
               reference_coverage_source_days=None, demand_litres=80) for f, lot in (('A', 100), ('B', 500))]
    pool = SimpleNamespace(block_key=('P', 'S'), candidates=(), completeness='complete')
    evidence = module.uncovered_summary([pool], members, RunConfig(demand_days=80))
    assert evidence == {'uncovered_count': 2, 'blocks': [{'block': ('P', 'S'), 'uncovered_count': 2, 'uncovered_only_line6_count': 2, 'source_high_runner_count': 1, 'pool_completeness': 'complete'}]}


def test_runner_threshold_migration_seeds_profiles_from_former_policy():
    """Legacy profiles get their plant's former SAP cut-offs; the policy rows are removed."""
    from app.workspace.schema_migrations import migrate_runner_thresholds
    service, _, profile = fixture()
    legacy = {k: v for k, v in profile['settings'].items() if k != 'runner_thresholds'}
    rule = {'kind': 'group_rule', 'constraint_id': 'mix', 'approval_status': 'approved',
            'assertion': {'op': 'any', 'args': [{'op': 'eq', 'args': [{'op': 'field', 'field': 'member.runner'},
                                                                     {'op': 'literal', 'value': 'high'}]}]}}
    service.repo.cas('profiles', profile['profile_id'], 1, {
        'settings': {**legacy, 'high_runner_threshold_days': 15}, 'rules': [rule]})
    service.repo.replace_tables('sap-policy', {'sap_policy_rules': [
        {'plant': '*', 'setting': 'runner_high_days', 'abc_class': '*', 'value': None, 'number': 12},
        {'plant': 'P', 'setting': 'runner_ratio_low', 'abc_class': '*', 'value': None, 'number': 8},
        {'plant': 'P', 'setting': 'max_cycle_days', 'abc_class': '*', 'value': None, 'number': 150}]})

    # Before migration, reads still work with default cut-offs.
    assert service.get_plant_profile(profile['profile_id'])['settings']['runner_thresholds']['high_max_days'] == 10
    report = migrate_runner_thresholds(service.repo)
    assert report == {'profiles': [profile['profile_id']], 'runner_rule_profiles': [profile['profile_id']],
                      'policy_rows_removed': 2}
    stored = service.repo.get('profiles', profile['profile_id'])['settings']
    assert 'high_runner_threshold_days' not in stored
    assert stored['runner_thresholds'] == {'high_max_days': 12, 'low_min_days': 80, 'high_max_ratio': 3, 'low_min_ratio': 8}
    assert [r['setting'] for r in service.repo.rows('sap-policy', 'sap_policy_rules')] == ['max_cycle_days']
    assert migrate_runner_thresholds(service.repo) == {'profiles': [], 'runner_rule_profiles': [], 'policy_rows_removed': 0}


def test_cycle_stock_policy_migration_seeds_profiles_from_former_policy():
    """Profiles without a cycle stock policy get their plant's former SAP values in single mode."""
    from app.workspace.schema_migrations import migrate_cycle_stock_policy
    service, _, profile = fixture()
    legacy = {k: v for k, v in profile['settings'].items() if k != 'cycle_stock_policy'}
    service.repo.cas('profiles', profile['profile_id'], 1, {'settings': legacy})
    service.repo.replace_tables('sap-policy', {'sap_policy_rules': [
        {'plant': '*', 'setting': 'runner_low_min_days', 'abc_class': '*', 'value': None, 'number': 45},
        {'plant': 'P', 'setting': 'max_cycle_days', 'abc_class': '*', 'value': None, 'number': 150},
        {'plant': 'P', 'setting': 'calendar_lot_size', 'abc_class': '*', 'value': 'PK', 'number': None}]})

    # Before migration, reads fall back to the default policy.
    assert service.get_plant_profile(profile['profile_id'])['settings']['cycle_stock_policy']['mode'] == 'single'
    report = migrate_cycle_stock_policy(service.repo)
    assert report == {'profiles': [profile['profile_id']], 'policy_rows_removed': 2}
    stored = service.repo.get('profiles', profile['profile_id'])['settings']['cycle_stock_policy']
    assert stored == {'mode': 'single', 'high_min_days': 0, 'medium_min_days': 20, 'low_min_days': 45,
                      'medium_batch_factor': 0.5, 'max_days': 150, 'mixed_groups_only': True}
    assert [r['setting'] for r in service.repo.rows('sap-policy', 'sap_policy_rules')] == ['calendar_lot_size']
    assert migrate_cycle_stock_policy(service.repo) == {'profiles': [], 'policy_rows_removed': 0}


def test_legacy_frozen_profile_copy_still_compiles():
    """A draft frozen before the three-tier classes runs with default cut-offs."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    frozen = dict(draft['plant_profile'])
    frozen['settings'] = {**{k: v for k, v in frozen['settings'].items() if k != 'runner_thresholds'},
                          'high_runner_threshold_days': 15}
    service.repo.cas('drafts', draft['draft_id'], draft['revision'], {'plant_profile': frozen})
    config = service.compiled_draft(service.get_draft(draft['draft_id'])).config
    assert config.runner_thresholds.model_dump() == {'high_max_days': 10, 'low_min_days': 80, 'high_max_ratio': 3, 'low_min_ratio': 10}


def test_validation_blocks_volumes_missing_from_profile_matrix():
    """A profile matrix that lacks modeled volumes blocks the launch with the gap listed."""
    from app.workspace.matrix_description import expand_matrix_description
    service = WorkspaceService(MemoryRepository())
    source = dataset(service)
    service.publish_dataset(source['dataset_id'])
    defaults = service.plant_profile_defaults()
    narrow = expand_matrix_description({'volumes': ['999'], 'default_status': 'N'})
    profile = service.create_plant_profile({'name': 'Default matrix', 'plant': 'P',
        'settings': {**defaults['settings'], 'volume_matching': 'exact'}, 'rules': [], 'matrix_rows': narrow})
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    result = service.validate_draft(draft['draft_id'])
    assert not result['valid'] and 'missing from it' in result['errors'][0]
    coverage = result['matrix_coverage']
    assert coverage['matrix_volume_count'] == 1 and coverage['missing_volumes']
    assert coverage['affected_member_count'] == coverage['modeled_member_count']
    with pytest.raises(ValueError, match='missing from it'):
        service.submit({'draft_id': draft['draft_id'], 'revision': 1, 'idempotency_key': 'gap'})
    report = service.dataset_volume_coverage(source['dataset_id'], profile['profile_id'])
    assert report['missing_volumes'] and report['profile_volumes_unused'] == ['999']


def test_covered_profile_reports_complete_coverage_and_run_recheck():
    """A covering matrix validates, and an existing run's inputs can be re-checked read-only."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    validation = service.validate_draft(draft['draft_id'])
    assert validation['valid'] and validation['matrix_coverage']['complete']
    run = service.submit({'draft_id': draft['draft_id'], 'revision': 1, 'idempotency_key': 'ok'})
    recheck = service.recheck_run_inputs(run['run_id'])
    assert recheck['valid'] and recheck['matrix_coverage']['complete'] and recheck['counts']['modeled_fini_count']


def test_outdated_draft_is_flagged_and_recreated_with_current_profile():
    """A profile edit after draft creation is reported, and recreate keeps scenario choices only."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'], 'Scenario A')
    assert service.draft_profile_status(draft)['outdated'] is False
    # Save draft in the UI marks profile-owned values explicit as well; they must not be copied.
    config = draft['request']['config']
    draft = service.update_draft(draft['draft_id'], draft['revision'], {
        'request': {'config': {'matrix_mode': 'DIAGNOSTIC', 'runner_thresholds': config['runner_thresholds']}},
        'budget': {'frontier_points': 9}})
    settings = {**profile['settings'], 'runner_thresholds': {**profile['settings']['runner_thresholds'], 'high_max_days': 5}}
    service.update_plant_profile(profile['profile_id'], {'revision': 1, 'settings': settings})
    status = service.draft_profile_status(draft['draft_id'])
    assert status['outdated'] and status['draft_profile_revision'] == 1 and status['current_profile_revision'] == 2
    validation = service.validate_draft(draft['draft_id'])
    assert validation['profile_status']['outdated'] and 'revision 1' in validation['warnings'][0]
    result = service.recreate_draft_with_current_profile(draft['draft_id'])
    new = result['draft']
    assert new['draft_id'] != draft['draft_id'] and new['title'] == 'Scenario A'
    assert new['plant_profile']['revision'] == 2 and not result['profile_status']['outdated']
    assert new['request']['config']['matrix_mode'] == 'DIAGNOSTIC' and new['budget']['frontier_points'] == 9
    assert new['request']['config']['runner_thresholds']['high_max_days'] == 5
    assert result['carried_over'] == ['budget.frontier_points', 'request.config.matrix_mode'] and result['skipped'] == []
    assert service.get_draft(draft['draft_id'])['plant_profile']['revision'] == 1
    assert service.validate_draft(new['draft_id'])['valid']


def test_recreate_refuses_removed_profile():
    """A removed profile has nothing to refresh to; the status explains why."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    service.remove_plant_profile(profile['profile_id'], 1, True)
    status = service.draft_profile_status(draft)
    assert status['removed'] and not status['outdated'] and 'removed' in status['message']
    with pytest.raises(ValueError, match='removed'):
        service.recreate_draft_with_current_profile(draft['draft_id'])


def test_partial_sap_settings_patch_keeps_every_other_value():
    """A one-field SAP patch is merged into the section; None clears a class override."""
    service, _, profile = fixture()
    first = service.apply_plant_profile_change(profile['profile_id'], 1, {'action': 'update_settings', 'settings': {
        'sap_settings': {'calendar_odd_profile': 'ODD-W1', 'calendar_even_profile': 'EVEN-W2',
                         'dc_min_lot': {'C': {'rule': 'litres', 'number': 300}}}}})
    second = service.apply_plant_profile_change(profile['profile_id'], first['revision'], {
        'action': 'update_settings', 'settings': {'sap_settings': {'calendar_even_profile': 'EVEN-W4', 'dc_min_lot': {'C': None}}}})
    sap = second['settings']['sap_settings']
    assert sap['calendar_odd_profile'] == 'ODD-W1' and sap['calendar_even_profile'] == 'EVEN-W4'
    assert sap['dc_min_lot']['C'] is None and sap['dc_min_lot']['all_classes']['rule'] == 'pallet'
    assert sap['calendar_lot_size'] == 'PK'
    with pytest.raises(ValueError, match='profile'):
        service.apply_plant_profile_change(profile['profile_id'], second['revision'], {
            'action': 'update_settings', 'settings': {'sap_settings': {'calendar_odd_profile': 'NODASH'}}})


def test_partial_runner_threshold_patch_keeps_the_other_cut_offs():
    """Regression: a nested patch used to reset the sibling fields to defaults."""
    service, _, profile = fixture()
    service.apply_plant_profile_change(profile['profile_id'], 1, {'action': 'update_settings', 'settings': {
        'runner_thresholds': {'high_max_days': 6, 'low_min_days': 90}}})
    updated = service.apply_plant_profile_change(profile['profile_id'], 2, {'action': 'update_settings', 'settings': {
        'runner_thresholds': {'high_max_days': 7}}})
    assert updated['settings']['runner_thresholds']['high_max_days'] == 7
    assert updated['settings']['runner_thresholds']['low_min_days'] == 90


def test_full_settings_save_without_sap_section_keeps_stored_sap_rules():
    """Older clients that send settings without sap_settings do not wipe them."""
    service, _, profile = fixture()
    saved = service.apply_plant_profile_change(profile['profile_id'], 1, {'action': 'update_settings', 'settings': {
        'sap_settings': {'calendar_odd_profile': 'ODD-W1', 'calendar_even_profile': 'EVEN-W2'}}})
    settings = {k: v for k, v in saved['settings'].items() if k != 'sap_settings'}
    updated = service.update_plant_profile(profile['profile_id'], {'revision': saved['revision'], 'settings': {**settings, 'horizon_days': 250}})
    assert updated['settings']['sap_settings']['calendar_odd_profile'] == 'ODD-W1'


def test_sap_only_profile_edit_does_not_outdate_drafts():
    """SAP rules are read live at generation, so editing them leaves drafts current."""
    service, source, profile = fixture()
    draft = service.create_draft(source['dataset_id'], profile['profile_id'])
    service.apply_plant_profile_change(profile['profile_id'], 1, {'action': 'update_settings', 'settings': {
        'sap_settings': {'calendar_odd_profile': 'ODD-W1', 'calendar_even_profile': 'EVEN-W2'}}})
    status = service.draft_profile_status(draft['draft_id'])
    assert status['outdated'] is False and status['current_profile_revision'] == 2
    assert 'SAP settings' in status['note'] and 'message' not in status
    service.apply_plant_profile_change(profile['profile_id'], 2, {'action': 'update_settings', 'settings': {'maximum_group_size': 5}})
    assert service.draft_profile_status(draft['draft_id'])['outdated'] is True


def _without_sap_section(service, profile):
    """Store a profile as it looked before SAP settings moved into it."""
    current = service.repo.get('profiles', profile['profile_id'])
    legacy = {k: v for k, v in current['settings'].items() if k != 'sap_settings'}
    service.repo.cas('profiles', profile['profile_id'], current['revision'], {'settings': legacy})


def _row(plant, setting, value=None, number=None, abc_class='*'):
    """One stored SAP policy row."""
    return {'plant': plant, 'setting': setting, 'abc_class': abc_class, 'value': value, 'number': number}


def test_sap_settings_migration_moves_rows_into_every_profile_of_the_plant():
    """'*' rows reach every profile, plant rows only their plant; orphans and unknown rows stay."""
    from app.workspace.schema_migrations import migrate_sap_settings
    service, _, first = fixture()
    defaults = service.plant_profile_defaults()
    second = service.create_plant_profile({'name': 'Plant plan B', 'plant': 'P', **defaults})
    other = service.create_plant_profile({'name': 'Other plant', 'plant': 'Q', **defaults})
    for profile in (first, second, other):
        _without_sap_section(service, profile)
    service.repo.replace_tables('sap-policy', {'sap_policy_rules': [
        _row('*', 'dc_rounding', 'layer'),
        _row('P', 'calendar_odd_profile', 'ODD-W1'),
        _row('P', 'calendar_even_profile', 'EVEN-W2'),
        _row('P', 'dc_min_lot', 'litres', 300, 'C'),
        _row('P', 'calendar_frequency_code', ''),  # invalid: blank text, skipped on its own
        _row('Z', 'calendar_odd_profile', 'ODD-W9'),  # orphan plant without a profile
        _row('P', 'retired_setting', 'x')]})

    report = migrate_sap_settings(service.repo)
    assert sorted(report['profiles']) == sorted(p['profile_id'] for p in (first, second, other))
    for profile in (first, second):
        sap = service.repo.get('profiles', profile['profile_id'])['settings']['sap_settings']
        assert sap['calendar_odd_profile'] == 'ODD-W1' and sap['calendar_even_profile'] == 'EVEN-W2'
        assert sap['dc_rounding']['all_classes']['rule'] == 'layer'
        assert sap['dc_min_lot']['C'] == {'rule': 'litres', 'number': 300}
        assert sap['calendar_frequency_code'] == '02W'
    other_sap = service.repo.get('profiles', other['profile_id'])['settings']['sap_settings']
    assert other_sap['dc_rounding']['all_classes']['rule'] == 'layer' and other_sap['calendar_odd_profile'] is None
    remaining = {(r['plant'], r['setting']) for r in service.repo.rows('sap-policy', 'sap_policy_rules')}
    assert remaining == {('Z', 'calendar_odd_profile'), ('P', 'retired_setting')}
    assert report['policy_rows_removed'] == 5 and report['rows_kept'] == 2

    revisions = [service.repo.get('profiles', p['profile_id'])['revision'] for p in (first, second, other)]
    assert migrate_sap_settings(service.repo)['profiles'] == []
    assert [service.repo.get('profiles', p['profile_id'])['revision'] for p in (first, second, other)] == revisions


def test_sap_settings_migration_reaches_removed_profiles_and_keeps_rows_on_conflict():
    """Removed profiles are upgraded too; a plant whose profile could not be saved keeps its rows."""
    from app.workspace.schema_migrations import migrate_sap_settings
    service, _, profile = fixture()
    _without_sap_section(service, profile)
    current = service.repo.get('profiles', profile['profile_id'])
    service.repo.cas('profiles', profile['profile_id'], current['revision'], {'status': 'removed'})
    service.repo.replace_tables('sap-policy', {'sap_policy_rules': [_row('P', 'calendar_odd_profile', 'ODD-W1'),
                                                                      _row('P', 'calendar_even_profile', 'EVEN-W2')]})
    original_cas = service.repo.cas

    def conflicting_cas(kind, key, revision, patch):
        """Simulate another process saving the profile first."""
        raise ValueError('stale revision')

    service.repo.cas = conflicting_cas
    assert migrate_sap_settings(service.repo)['profiles'] == []
    assert len(service.repo.rows('sap-policy', 'sap_policy_rules')) == 2
    service.repo.cas = original_cas
    report = migrate_sap_settings(service.repo)
    assert report['profiles'] == [profile['profile_id']] and report['policy_rows_removed'] == 2
    stored = service.repo.get('profiles', profile['profile_id'])
    assert stored['status'] == 'removed' and stored['settings']['sap_settings']['calendar_odd_profile'] == 'ODD-W1'
