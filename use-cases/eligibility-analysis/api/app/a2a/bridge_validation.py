"""Local Joule bridge contracts independent from deployment or live tenant access."""
import json
from pathlib import Path
import yaml


def validate_bridge(root):
    """Validate filenames, aliases and every explicit conversation-context round-trip edge."""
    root=Path(root);errors=[]
    try:
        manifest=yaml.safe_load((root/'da.sapdas.yaml').read_text())
        capability=yaml.safe_load((root/'joule/a2a/capability.sapdas.yaml').read_text())
        context=yaml.safe_load((root/'joule/a2a/capability_context.yaml').read_text())
        function=yaml.safe_load((root/'joule/a2a/functions/query_receivables_agent.yaml').read_text())
        scenario=yaml.safe_load((root/'joule/a2a/scenarios/agent/receivables_assistance.yaml').read_text())
    except (OSError,yaml.YAMLError) as error:return [str(error)]
    if manifest['capabilities'][0]['folder']!='./joule/a2a':errors.append('Manifest must point to the bridge capability')
    if len(scenario['target']['name'])>30:errors.append('Function name exceeds 30 characters')
    if context.get('variables')!=[{'name':'agent_context_id'}]:errors.append('Missing capability context variable')
    if function.get('parameters')!=[{'name':'agent_context_id','optional':True}]:errors.append('Context parameter must be optional')
    action=function['action_groups'][0]['actions'][0]
    if action.get('type')!='agent-request' or action.get('agent_type')!='remote':errors.append('Use a remote A2A agent-request')
    if action.get('system_alias')!='RECEIVABLES_AGENT' or capability['system_aliases'].get('RECEIVABLES_AGENT',{}).get('destination')!='RECEIVABLES_AGENT':errors.append('Destination alias mismatch')
    try:
        if json.loads(action['body']).get('contextId')!='<? agent_context_id ?>':errors.append('Outbound contextId is not forwarded')
    except (ValueError,KeyError):errors.append('Invalid agent-request body')
    if function.get('result',{}).get('agent_context_id')!='<? _agent_response.body.contextId ?>':errors.append('Flatten remote contextId into the function result')
    if scenario.get('target',{}).get('parameters')!=[{'name':'agent_context_id','value':'$capability_context.agent_context_id'}]:errors.append('Scenario must inject capability context')
    if scenario.get('capability_context')!=[{'name':'agent_context_id','value':'$target_result.agent_context_id'}]:errors.append('Scenario must save the root result context ID')
    if 'response_context' in scenario:errors.append('Bridge-only mode must not enable response_context')
    if scenario.get('slots'):errors.append('Context may not also be supplied through slots')
    return errors
