"""A2A completed retries return saved answers before mutable source validation."""
import unittest
from unittest.mock import Mock,patch,AsyncMock
from app.a2a.a2a_server import _handle_message_send


class RequestReplayTests(unittest.IsolatedAsyncioTestCase):
    """Use the actual message boundary with a completed request and now-stale workspace."""

    async def test_cached_response_precedes_stale_scope_validation(self):
        """An uncertain-response retry must not invoke the model or reject a newer run revision."""
        store=Mock();store.claim.return_value={'cached':{'id':'task','contextId':'context','kind':'task'}}
        params={'message':{'role':'user','messageId':'same','contextId':'context','parts':[{'kind':'text','text':'Explain this run'}]},
                'metadata':{'workspace_context':{'analysis_id':'offer','run_id':'run','revision':1}}}
        with patch('app.a2a.persistence.get_conversation_store',return_value=store),patch('app.a2a.persistence.access_domain',return_value='owner'),\
             patch('app.a2a.workspace_context.resolve_workspace_context',side_effect=ValueError('stale')) as resolver,\
             patch('app.a2a.a2a_server.run_agent',new_callable=AsyncMock) as agent:
            response=await _handle_message_send('rpc',params)
        self.assertEqual(response['result']['id'],'task');resolver.assert_not_called();agent.assert_not_called()
