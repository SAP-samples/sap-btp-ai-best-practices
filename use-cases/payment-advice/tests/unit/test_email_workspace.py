"""Offline acceptance checks for email matching, intake and advice corrections."""
import unittest
import copy
from types import SimpleNamespace
from unittest.mock import patch, Mock
from app.email_ingestion.domain import match_customer, email_status, apply_edit, prepare_attachment
from app.email_ingestion.intake import intake
from app.email_ingestion.corrections import edit, edit_many, revert_all, undo, propose, confirm
from app.email_ingestion.worker import synchronize, process_advice
from app.email_ingestion.gmail import Gmail, CursorExpired
from app.email_ingestion.store import Conflict
from tests.unit.workspace_support import MemoryStore


class WorkspaceTests(unittest.TestCase):
    """Exercise real normalization and mutation boundaries without cloud credentials."""

    def test_match_and_ambiguous_fallback(self):
        """Sender-domain identity matches; close fuzzy scores stay unresolved."""
        customers = [{"client_key": "globex", "display_name": "Globex", "status": "active"},
                     {"client_key": "other", "display_name": "Other", "status": "active"}]
        self.assertEqual(match_customer("noreply@globex.com", "", customers, lambda q: [])['client_key'], 'globex')
        result = match_customer("sender@example.com", "Example", customers,
                                lambda q: [("globex", "Globex", .89), ("other", "Other", .85)])
        self.assertIsNone(result['client_key'])

    def test_match_subject_and_body_tiers(self):
        """Subject/body names resolve forwarded mail; sender wins; ambiguity stays open."""
        customers = [{"client_key": "globex", "display_name": "Globex", "status": "active"},
                     {"client_key": "other", "display_name": "Other", "status": "active"},
                     {"client_key": "fabrikam", "display_name": "Fabrikam Inc.", "status": "active"}]
        none = lambda q: []
        result = match_customer("ar@ourco.com", "", customers, none, subject="Fwd: Remittance from Globex")
        self.assertEqual((result['client_key'], result['method']), ('globex', 'exact_subject'))
        result = match_customer("ar@ourco.com", "", customers, none, subject="Payment advice",
                                body="Payment from Globex EU S.a.r.l. attached.")
        self.assertEqual((result['client_key'], result['method']), ('globex', 'exact_body'))
        result = match_customer("noreply@globex.com", "", customers, none, body="Also cc Other team")
        self.assertEqual((result['client_key'], result['method']), ('globex', 'exact_sender'))
        result = match_customer("ar@ourco.com", "", customers, lambda q: [("globex", "Globex", .95)],
                                body="Globex and Other paid")
        self.assertIsNone(result['client_key'])
        self.assertEqual(result['hits']['body'], ['globex', 'other'])
        self.assertIsNone(match_customer("ar@ourco.com", "", customers, none,
                                         body="otherwise unrelated")['client_key'])
        result = match_customer("ar@ourco.com", "", customers, none, body="Remitted by Fabrikam today")
        self.assertEqual(result['client_key'], 'fabrikam')
        result = match_customer("ar@ourco.com", "", customers,
                                lambda q: [("globex", "Globex", .9)] if q == 'globx' else [],
                                subject="Remittance Globx 4711")
        self.assertEqual((result['client_key'], result['method']), ('globex', 'fuzzy'))
        self.assertIn('globx', result['terms'])
        self.assertNotIn('remittance', result['terms'])

    def test_edit_requires_unique_reference_and_keeps_original(self):
        """Duplicate references cannot mutate rows; row IDs provide disambiguation."""
        original = {"header": {}, "line_items": [
            {"row_id": "a", "invoice_reference": "123", "reason_code": "323"},
            {"row_id": "b", "invoice_reference": "123", "reason_code": "323"}]}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            apply_edit(original, "123", "reason_code", "321")
        changed, ids = apply_edit(original, "a", "reason_code", "321")
        self.assertEqual(ids, ["a"])
        self.assertEqual(changed['line_items'][0]['reason_code'], '321')
        self.assertEqual(original['line_items'][0]['reason_code'], '323')
        with self.assertRaises(ValueError):
            apply_edit(original, "a", "row_id", "other")

    def test_attachment_and_rollup(self):
        """Untrusted file names are sanitized and statuses reflect all advices."""
        self.assertEqual(prepare_attachment('../../a.txt', b'hello')['filename'], 'a.txt')
        with self.assertRaises(ValueError):
            prepare_attachment('a.html', b'<script>bad()</script>')
        self.assertEqual(email_status(['reviewed', 'ready']), 'ready')
        self.assertEqual(email_status(['reviewed', 'reviewed']), 'reviewed')
        self.assertEqual(email_status(['failed', 'ready']), 'needs_review')

    def test_types_and_bad_files(self):
        """Reject script-capable extensions, nonfinite amounts and invalid date types."""
        payload = {'header': {}, 'line_items': [{'row_id': 'x'}]}
        for invalid in (float('nan'), float('inf'), True, '1.2'):
            with self.assertRaises(ValueError):
                apply_edit(payload, 'x', 'net_amount', invalid)
        with self.assertRaises(ValueError):
            apply_edit(payload, 'x', 'invoice_date', 123)
        for filename, data in [('x.svg', b'<svg/>'), ('x.pdf', b'not a pdf'), ('x.xlsx', b'bad')]:
            with self.assertRaises(ValueError):
                prepare_attachment(filename, data)

    def test_row_ids_survive_reordered_reprocessing(self):
        """Invoice identity, not extraction position, determines stable row identity."""
        from app.email_ingestion.domain import stable_rows
        old = [{'row_id': 'a', 'invoice_reference': 'I1'}, {'row_id': 'b', 'invoice_reference': 'I2'}]
        new = [{'invoice_reference': 'I2'}, {'invoice_reference': 'I1'}]
        stable_rows(new, old, 'advice')
        self.assertEqual([r['row_id'] for r in new], ['b', 'a'])

    def test_new_duplicate_reference_does_not_inherit_wrong_row(self):
        """New duplicates must not steal an old row identity before exact content matching."""
        from app.email_ingestion.domain import stable_rows
        old = [{'row_id':'old-id','invoice_reference':'I1','net_amount':100}]
        new = [{'invoice_reference':'I1','net_amount':200}, {'invoice_reference':'I1','net_amount':100}]
        stable_rows(new, old, 'advice')
        self.assertNotEqual(new[0]['row_id'], 'old-id')
        self.assertEqual(new[1]['row_id'], 'old-id')


class IntakeTests(unittest.IsolatedAsyncioTestCase):
    """Shared pipeline acceptance with fake cloud calls and real business functions."""
    def setUp(self):
        """Create only test-memory persistence and deterministic customer lookups."""
        self.store = MemoryStore()
        self.patches = [patch('app.email_ingestion.intake.list_customers', return_value=[]),
                        patch('app.email_ingestion.intake.find_customers', return_value=[]),
                        patch('app.email_ingestion.worker.ServiceKey.from_json', return_value=object()),
                        patch('app.email_ingestion.worker.SapDoxClient', return_value=object())]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.message = {'sender': 'noreply@example.test', 'subject': 'Acceptance advice', 'body': 'Payment 123, amount 100', 'attachments': []}

    def test_multiple_body_fallback_and_duplicates(self):
        """Two attachments produce two jobs; body fallback preserves unsupported warnings."""
        email_id = intake(self.store, {**self.message, 'attachments': [('a.txt', b'one'), ('b.txt', b'two')]})
        self.assertEqual(len(self.store.list('advice', parent=email_id)), 2)
        email_id = intake(self.store, {**self.message, 'attachments': [('unsafe.html', b'<script/>')], 'gmail_id': 'g1'}, 'gmail', 'mailbox')
        self.assertEqual(len(self.store.list('advice', parent=email_id)), 1)
        self.assertEqual(len(self.store.get(email_id)['warnings']), 1)
        again = intake(self.store, {**self.message, 'gmail_id': 'g1'}, 'gmail', 'mailbox')
        self.assertEqual(email_id, again)
        self.assertEqual(len(self.store.list('advice', parent=email_id)), 1)

    async def test_manual_and_gmail_same_result_unknown_no_rules(self):
        """Equivalent sources use canonical extraction and never consult a guessed playbook."""
        payload = {'header': {'payment_reference': 'P1'}, 'line_items': [{'invoice_reference': 'I1', 'net_amount': -10}]}
        state = SimpleNamespace(dox=object(), settings=SimpleNamespace(service_key_data={}), runtime=None)
        results = []
        with patch('app.email_ingestion.worker.extract_to_canonical', return_value={'canonical': payload}) as extract, patch('app.email_ingestion.worker.get_playbook') as rules:
            for source in ('manual', 'gmail'):
                email_id = intake(self.store, {**self.message, 'gmail_id': 'g1'}, source, 'mailbox')
                advice = self.store.list('advice', parent=email_id)[0]
                job = self.store.list('job', parent=advice['id'])[0]
                await process_advice(state, self.store, job, 'worker')
                saved = self.store.get(advice['id'])
                results.append(saved['result']['line_items'][0]['reason_code'])
                self.assertIsNone(saved['client_key'])
                self.assertEqual(saved['status'], 'needs_review')
            self.assertEqual(results, [None, None])
            rules.assert_not_called()
            self.assertIsNone(extract.call_args.args[1])
            self.assertIn('out_dir', extract.call_args.kwargs)

    def test_edits_undo_revision_and_review_rollup(self):
        """Edits preserve original values, reopen review, reject stale writes and undo."""
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        payload = {'header': {}, 'line_items': [{'row_id': 'r1', 'invoice_reference': 'I1', 'reason_code': '323'}]}
        advice.update(result=payload, original_result=copy.deepcopy(payload), status='reviewed')
        advice = self.store.update(advice)
        saved = edit(self.store, advice['id'], advice['revision'], 'I1', 'reason_code', '321', 'Confirmed allowance')
        self.assertEqual(saved['result']['line_items'][0]['reason_code'], '321')
        self.assertEqual(saved['original_result']['line_items'][0]['reason_code'], '323')
        self.assertEqual(self.store.get(email_id)['status'], 'needs_review')
        with self.assertRaises(Conflict):
            edit(self.store, advice['id'], advice['revision'], 'I1', 'reason_code', '999', 'stale')
        reverted = undo(self.store, advice['id'], saved['revision'])
        self.assertEqual(reverted['result']['line_items'][0]['reason_code'], '323')
        self.assertEqual(len(self.store.list('edit', parent=advice['id'])), 2)

    def test_table_batch_edit_is_atomic_and_revert_restores_originals(self):
        """A batch saves as one revision, an invalid cell saves nothing, revert restores originals."""
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        payload = {'header': {'payment_amount': 10}, 'line_items': [
            {'row_id': 'r1', 'invoice_reference': 'I1', 'net_amount': 10, 'reason_code': '323', 'flags': ['x']}]}
        advice.update(result=payload, original_result=copy.deepcopy(payload), status='reviewed')
        advice = self.store.update(advice)
        with self.assertRaisesRegex(ValueError, 'r1 / net_amount'):
            edit_many(self.store, advice['id'], advice['revision'], [
                {'target': 'r1', 'field': 'reason_code', 'value': '321'},
                {'target': 'r1', 'field': 'net_amount', 'value': 'abc'}], 'table')
        self.assertEqual(self.store.get(advice['id'])['result'], payload)
        saved = edit_many(self.store, advice['id'], advice['revision'], [
            {'target': 'r1', 'field': 'reason_code', 'value': '321'},
            {'target': 'r1', 'field': 'net_amount', 'value': 12.5},
            {'target': 'r1', 'field': 'flags', 'value': []}], 'table')
        self.assertEqual(saved['revision'], advice['revision'] + 1)
        self.assertEqual(saved['result']['line_items'][0], {**payload['line_items'][0], 'reason_code': '321', 'net_amount': 12.5, 'flags': []})
        self.assertEqual(len(saved['corrections']), 3)
        self.assertEqual(len(self.store.list('edit', parent=advice['id'])), 3)
        with self.assertRaises(Conflict):
            revert_all(self.store, advice['id'], advice['revision'])
        reverted = revert_all(self.store, advice['id'], saved['revision'])
        self.assertEqual(reverted['result'], payload)
        self.assertEqual(reverted['corrections'], [])
        self.assertEqual(reverted['changed_row_ids'], ['r1'])
        with self.assertRaises(ValueError):
            revert_all(self.store, advice['id'], reverted['revision'])

    def test_proposals_scope_and_staleness(self):
        """A proposal cannot be confirmed for another advice or after its source changes."""
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        advice.update(client_key='customer', status='ready')
        advice = self.store.update(advice)
        with patch('app.email_ingestion.corrections.get_playbook', return_value=None):
            proposal = propose(self.store, advice['id'], advice['revision'], 'New rule', 'Evidence')
        self.assertEqual(proposal['status'], 'pending')
        other = self.store.insert('advice', {'client_key': 'customer'})
        with self.assertRaises(KeyError):
            confirm(self.store, other, proposal['id'])
        self.store.update(advice)
        with self.assertRaises(Conflict):
            confirm(self.store, advice['id'], proposal['id'])

    def test_gmail_initial_cursor_pagination_and_history_recovery(self):
        """Initial cursor precedes enumeration; expired history resumes without duplicate records."""
        events = []
        fake = Mock()
        fake.get.side_effect = lambda path: {'emailAddress': 'mailbox', 'historyId': str(100 + len(events))}
        def pages(path, **params):
            """Simulate paginated full sync and an expired incremental cursor."""
            events.append(path)
            if path == 'history':
                raise CursorExpired('expired')
            yield {'messages': [{'id': 'a'}], 'nextPageToken': 'next'}
            yield {'messages': [{'id': 'b'}]}
        fake.pages.side_effect = pages
        fake.message.side_effect = lambda mid: {**self.message, 'gmail_id': mid, 'inbox': True}
        with patch('app.email_ingestion.worker.Gmail', return_value=fake):
            for _ in range(2):
                job_id = self.store.insert('job', {'operation': 'fetch'})
                synchronize(self.store, self.store.get(job_id), 'worker')
        self.assertEqual(len(self.store.list('email')), 2)
        self.assertEqual(len(self.store.list('advice')), 2)
        self.assertEqual(events, ['messages', 'history', 'messages'])

    def test_lost_lease_cannot_commit_intake(self):
        """A stale worker does not save records or advance its checkpoint."""
        with self.assertRaises(Conflict):
            intake(self.store, self.message, owner='lost')
        self.assertFalse(self.store.records)

    def test_stale_rule_revision_rolls_back_confirmation(self):
        """A concurrent playbook update rejects confirmation and rolls back advice changes."""
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        advice.update(client_key='customer', status='ready')
        advice = self.store.update(advice)
        playbook = SimpleNamespace(revision=7, playbook_text='Old rules', anchors={})
        with patch('app.email_ingestion.corrections.get_playbook', return_value=playbook):
            proposal = propose(self.store, advice['id'], advice['revision'], 'New rules', 'Evidence')
        with patch.object(self.store, 'execute', create=True, return_value=SimpleNamespace(rowcount=0)), self.assertRaises(Conflict):
            confirm(self.store, advice['id'], proposal['id'])
        self.assertEqual(self.store.get(advice['id'])['revision'], advice['revision'])
        self.assertEqual(self.store.get(proposal['id'])['status'], 'pending')

    async def test_chat_cannot_choose_row_id_to_bypass_ambiguous_user_reference(self):
        """Only a user-specified row ID may disambiguate duplicate invoice references."""
        from app.email_ingestion.assistant import chat
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        advice.update(status='ready', result={'header':{}, 'line_items':[
            {'row_id':'row-a','invoice_reference':'I1'}, {'row_id':'row-b','invoice_reference':'I1'}]})
        advice = self.store.update(advice)
        def runtime_factory(base, tools, prompt):
            """Simulate a model attempting to choose a row the user did not identify."""
            async def invoke(**kwargs):
                """Invoke the real bound edit tool with an ambiguous model-chosen ID."""
                edit_tool = next(t for t in tools if t.name == 'edit_field')
                edit_tool.invoke({'target':'row-a','field':'reason_code','value':'"321"','reason':'User correction'})
                return SimpleNamespace(output_text='Changed')
            return SimpleNamespace(ainvoke=invoke)
        with patch('app.email_ingestion.assistant.scoped_runtime', side_effect=runtime_factory), self.assertRaises(ValueError):
            await chat(object(), self.store, advice['id'], advice['revision'], 'Change invoice I1 reason code to 321', [])
        self.assertEqual(self.store.get(advice['id'])['revision'], advice['revision'])

    def test_rollup_reloads_children_after_concurrent_conflict(self):
        """A rollup loser must recompute from newer children, not retain its stale status."""
        from app.email_ingestion.intake import refresh_email
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        advice['status'] = 'ready'
        self.store.update(advice)
        update = self.store.update
        first = True
        def racing_update(record, **kwargs):
            """Commit another review/rollup just before the first stale email CAS."""
            nonlocal first
            if first and record['id'] == email_id:
                first = False
                child = self.store.get(advice['id'])
                child['status'] = 'reviewed'
                update(child)
                update(self.store.get(email_id))
            return update(record, **kwargs)
        with patch.object(self.store, 'update', side_effect=racing_update):
            refresh_email(self.store, email_id)
        self.assertEqual(self.store.get(email_id)['status'], 'reviewed')

    async def test_processing_binds_one_rule_snapshot(self):
        """Concurrent rule saves cannot change tools, provenance or persisted run evidence."""
        from app.payment_advice.deduction_rules import Playbook
        old = Playbook('customer', 'Original rules', {}, 7, None)
        new = Playbook('customer', 'Later rules', {}, 8, None)
        email_id = intake(self.store, self.message)
        advice = self.store.list('advice', parent=email_id)[0]
        advice['client_key'] = 'customer'
        advice = self.store.update(advice)
        job = self.store.list('job', parent=advice['id'])[0]
        payload = {'header': {}, 'line_items':[{'invoice_reference':'I1','net_amount':-10}]}
        seen = []
        def runtime_factory(base, tools, prompt):
            """Read the rules after an intervening simulated save."""
            async def invoke(**kwargs):
                """Return a minimal interpretation after capturing actual bound rules."""
                seen.append(next(t for t in tools if t.name == 'get_deduction_rules').invoke({}))
                return SimpleNamespace(output_parsed={'interpretations':[]}, messages=[])
            return SimpleNamespace(ainvoke=invoke)
        with patch('app.email_ingestion.worker.extract_to_canonical', return_value={'canonical':payload}), \
             patch('app.email_ingestion.worker.get_playbook', side_effect=[old,new]) as lookup, \
             patch('app.email_ingestion.assistant.get_playbook', return_value=new), \
             patch('app.payment_advice.deduction_rules.get_playbook', return_value=new), \
             patch('app.email_ingestion.assistant.scoped_runtime', side_effect=runtime_factory):
            await process_advice(SimpleNamespace(dox=object(), settings=SimpleNamespace(service_key_data={}), runtime=object()), self.store, job, 'worker')
        saved = self.store.get(advice['id'])
        self.assertEqual(seen[0]['revision'], 7)
        self.assertEqual(saved['rule_snapshot']['revision'], 7)
        self.assertEqual(saved['original_result']['header']['interpretation']['playbook_revision'], 7)
        self.assertEqual(lookup.call_count, 1)


if __name__ == '__main__':
    unittest.main()
