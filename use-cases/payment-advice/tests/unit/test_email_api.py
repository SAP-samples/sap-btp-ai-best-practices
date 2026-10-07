"""Offline HTTP security and scoped tool acceptance checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.email_ingestion import router, advices, IntakeSizeLimit
from app.security import get_api_key
from app.email_ingestion.assistant import scoped_reads
from app.email_ingestion.store import Store
from tests.unit.workspace_support import MemoryStore


class EmailAPITests(unittest.TestCase):
    """Exercise authorization before any workspace/attachment lookup."""
    def test_oversized_request_rejected_before_multipart_parsing(self):
        """A claimed oversized body is rejected without spooling or calling intake."""
        app = FastAPI()
        app.include_router(router, prefix='/api/email-ingestion')
        app.add_middleware(IntakeSizeLimit)
        with TestClient(app) as client:
            response = client.post('/api/email-ingestion/manual', content=b'', headers={'Content-Length':str(60 * 1024 * 1024)})
            self.assertEqual(response.status_code, 413)

    def test_attachment_and_advice_routes_require_auth(self):
        """Without the shared API key, downloads and advice operations are rejected."""
        app = FastAPI()
        app.include_router(router, prefix='/api/email-ingestion')
        app.include_router(advices, prefix='/api/payment-advice/advices')
        with patch('app.security.API_KEY', 'test-secret'), TestClient(app) as client:
            for path in ('/api/email-ingestion/attachments/x', '/api/email-ingestion/inbox', '/api/payment-advice/advices/x'):
                self.assertIn(client.get(path).status_code, (401, 403))

    def test_download_headers_and_wrong_record_type(self):
        """Authorized attachment download is inert; other record kinds cannot be downloaded."""
        app = FastAPI()
        app.include_router(router, prefix='/api/email-ingestion')
        app.dependency_overrides[get_api_key] = lambda: 'test'
        app.state.workspace = store = MemoryStore()
        attachment = store.insert('attachment', {'filename': 'test.txt', 'mime_type': 'text/plain'}, content=b'original')
        email = store.insert('email', {})
        with TestClient(app) as client:
            response = client.get('/api/email-ingestion/attachments/' + attachment)
            self.assertEqual(response.content, b'original')
            self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
            self.assertIn('attachment', response.headers['content-disposition'])
            self.assertEqual(client.get('/api/email-ingestion/attachments/' + email).status_code, 404)

    def test_delete_email_removes_its_complete_workspace_tree(self):
        """Deleting an inbox entry removes its stored email, files, advices and audit children."""
        app = FastAPI()
        app.include_router(router, prefix='/api/email-ingestion')
        app.dependency_overrides[get_api_key] = lambda: 'test'
        app.state.workspace = store = MemoryStore()
        email = store.insert('email', {'subject': 'Delete me'}, status='needs_review')
        store.insert('attachment', {'filename': 'source.txt'}, parent=email, content=b'original')
        advice = store.insert('advice', {'filename': 'source.txt'}, parent=email, status='needs_review')
        for kind in ('job', 'edit', 'proposal'):
            store.insert(kind, {}, parent=advice)

        with TestClient(app) as client:
            response = client.request('DELETE', '/api/email-ingestion/inbox/' + email, json={'revision': 0})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'deleted': email})
        self.assertEqual(store.records, {})
        self.assertEqual(store.bytes, {})

    def test_delete_email_refuses_active_processing(self):
        """An in-flight worker keeps its complete workspace tree until processing finishes."""
        app = FastAPI()
        app.include_router(router, prefix='/api/email-ingestion')
        app.dependency_overrides[get_api_key] = lambda: 'test'
        app.state.workspace = store = MemoryStore()
        email = store.insert('email', {}, status='processing')
        store.insert('advice', {}, parent=email, status='processing')

        with TestClient(app) as client:
            response = client.request('DELETE', '/api/email-ingestion/inbox/' + email, json={'revision': 0})

        self.assertEqual(response.status_code, 400)
        self.assertIn(email, store.records)

    def test_hana_delete_removes_advice_descendants_before_email(self):
        """The HANA transaction deletes nested audit/job rows before their owning inbox entry."""
        engine, connection = MagicMock(), MagicMock()
        engine.begin.return_value.__enter__.return_value = connection
        connection.execute.return_value = SimpleNamespace(rowcount=1)
        repository = Store(engine)
        records = {
            'email-1': {'id':'email-1','kind':'email','revision':3,'status':'needs_review'},
            'advice-1': {'id':'advice-1','kind':'advice','revision':2,'status':'needs_review'},
        }
        with patch.object(repository, 'get', side_effect=lambda record_id, conn=None: records[record_id]), \
             patch.object(repository, 'list', return_value=[records['advice-1']]), \
             patch.object(repository, 'lock_record'):
            deleted = repository.delete_email('email-1', 3)

        statements = [str(call.args[0]) for call in connection.execute.call_args_list]
        self.assertEqual(deleted, 'email-1')
        self.assertEqual(len(statements), 2)
        self.assertIn('"PARENT_ID"=:parent', statements[0])
        self.assertIn('"ID"=:email', statements[1])

    def test_scoped_tools_cannot_read_other_customers_or_mutate(self):
        """The advice tool catalog excludes global writes and rejects foreign rule reads."""
        tools = {t.name: t for t in scoped_reads(object(), 'allowed')}
        self.assertFalse(set(tools) & {'save_deduction_rules', 'delete_deduction_rules', 'set_customer_priority', 'create_customer'})
        with self.assertRaises(ValueError), patch('app.email_ingestion.assistant.get_playbook') as lookup:
            tools['get_deduction_rules'].invoke({'client_key': 'other'})
        lookup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
