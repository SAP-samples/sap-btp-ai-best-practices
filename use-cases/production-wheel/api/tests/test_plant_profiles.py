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
    service.update_plant_profile(profile['profile_id'], {'revision': latest['revision'], 'settings': {**profile['settings'], 'high_runner_threshold_days': 99}})
    assert service.compiled_draft(run).config.high_runner_threshold_days == 15
    assert run['plant_profile_revision'] == 1
    assert client.patch('/api/plant-profiles/'+profile['profile_id'], json={'revision': 1, 'name': 'stale'}).status_code == 409


def test_interpretation_uses_structured_readonly_runtime(monkeypatch):
    """The preview runtime has no mutation tools and returns validated structured rules."""
    import asyncio
    from types import SimpleNamespace
    from app.agent import runtime as runtime_module, providers
    from app.workspace.profile_interpretation import interpret_profile
    service, source, _ = fixture()
    service.save_ai_model_settings('gpt-5.6-terra', 0)
    captured = {}

    class Runtime:
        """Capture the constrained runtime without making provider calls."""
        def __init__(self, config, model, skills, mcp, memory, tools):
            """Record effective integrations and read-only capabilities."""
            captured['tools'] = {tool.name for tool in tools}
            captured['model'] = config.model.name
            assert not config.mcp.servers and not config.memory.enabled

        async def ainvoke(self, text, context_id, response_model, session_history):
            """Return a realistic provider structured-output payload."""
            return SimpleNamespace(output_parsed={'rules': [], 'interpretation': 'Clarify the desired maximum.', 'warnings': [], 'clarification_required': True})

        async def aclose(self):
            """Release the test runtime without external resources."""

    monkeypatch.setattr(runtime_module, 'AgentRuntime', Runtime)
    monkeypatch.setattr(providers, 'create_chat_model', lambda _: object())
    result = asyncio.run(interpret_profile(service, {'text': 'Keep groups small', 'plant': 'P', 'dataset_id': source['dataset_id']}))
    assert result['clarification_required']
    assert captured['tools'] == {'get_optimizer_capabilities', 'list_selected_plant_datasets', 'query_selected_plant_data'}
    assert captured['model'] == 'gpt-5.6-terra'


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
    member = SimpleNamespace(block_key=('P', 'S'), fini_id='A', eligible_lines=frozenset({'6'}), lot_size_considered_litres=100, demand_litres=80)
    pool = SimpleNamespace(block_key=('P', 'S'), candidates=(), completeness='complete')
    evidence = module.uncovered_summary([pool], [member], 80, 15)
    assert evidence == {'uncovered_count': 1, 'blocks': [{'block': ('P', 'S'), 'uncovered_count': 1, 'uncovered_only_line6_count': 1, 'source_high_runner_count': 0, 'pool_completeness': 'complete'}]}
