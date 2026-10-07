"""Complete turns and request identity persist across store replacement."""
import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,messages_to_dict,messages_from_dict
from app.services.database.backend import DatabaseBackend,BackendType
from app.a2a.conversation_store import ConversationStore


class ConversationTests(unittest.TestCase):
    """Use real isolated SQL claims and supported LangChain message serialization."""

    def setUp(self):
        """Create a fresh explicit test database and authenticated domain."""
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'turns.db';self.backend=DatabaseBackend(BackendType.SQLITE)
        self.store=ConversationStore(self.backend,self.path)

    def test_complete_replay_and_tool_history_survive_restart(self):
        """Reopening preserves ToolMessage associations and returns cached completed responses."""
        claim=self.store.claim('context','request','owner','hash')
        messages=messages_to_dict([HumanMessage('Question'),AIMessage('',tool_calls=[{'name':'read','args':{},'id':'call'}]),ToolMessage('Evidence',tool_call_id='call'),AIMessage('Answer')])
        self.store.complete('context','request','owner',claim['revision'],messages,{'id':'task','text':'Answer'},[],{})
        restarted=ConversationStore(self.backend,self.path)
        self.assertEqual(restarted.claim('context','request','owner','hash')['cached']['text'],'Answer')
        restored=messages_from_dict(restarted.load('context','owner')['messages'])
        self.assertEqual(restored[2].tool_call_id,'call')
        self.assertEqual(restarted.claim('context','next','owner','next-hash')['revision'],1)

    def test_concurrent_turn_and_wrong_domain_rejected(self):
        """A context ID does not bypass the existing access domain or active-turn claim."""
        self.store.claim('context','first','owner','hash')
        with self.assertRaises(ValueError):self.store.claim('context','second','owner','other')
        with self.assertRaises(LookupError):self.store.load('context','different-owner')
        self.store.fail('context','first','owner')
        self.assertEqual(self.store.load('context','owner')['messages'],[])
        self.assertEqual(self.store.claim('context','second','owner','other')['revision'],0)
