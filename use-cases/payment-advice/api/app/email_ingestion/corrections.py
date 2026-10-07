"""Revision-checked analyst edits, undo and explicitly confirmed rule proposals."""
import copy
import json
from dataclasses import asdict
from sqlalchemy import text
from .domain import apply_edit
from .intake import refresh_email
from .store import Conflict
from ..payment_advice.verify import verify_canonical
from ..payment_advice.hana_schema import DEDUCTION_RULES
from ..payment_advice.deduction_rules import get_playbook


POSTED = 'This advice was posted to S/4HANA and can no longer be changed'


def advice_record(store, advice_id, revision=None, mutable=False):
    """Read an advice and reject wrong kinds, stale revisions and (when mutable) posted advices."""
    advice = store.get(advice_id)
    if advice['kind'] != 'advice':
        raise KeyError(advice_id)
    if revision is not None and advice['revision'] != revision:
        raise Conflict('Advice changed; reload before editing')
    if mutable and advice['status'] == 'posted':
        raise ValueError(POSTED)
    return advice


def edit(store, advice_id, revision, target, field, value, reason):
    """Apply one explicit correction with an atomic audit record and before/after values."""
    return edit_many(store, advice_id, revision, [{'target': target, 'field': field, 'value': value}], reason)


def edit_many(store, advice_id, revision, edits, reason):
    """Apply several field corrections as one revision; any invalid edit aborts the whole batch.

    Inputs: ``edits`` is a list of ``{'target', 'field', 'value'}`` dicts where ``target`` is a row_id,
    a unique invoice/document reference, or ``'header'``. Output: the saved advice plus
    ``changed_row_ids``. One audit ``edit`` record is written per corrected field.
    """
    advice = advice_record(store, advice_id, revision, mutable=True)
    if advice['status'] in {'processing', 'queued'} or not advice.get('result'):
        raise Conflict('Wait for processing before editing')
    if not reason or len(reason) > 10000:
        raise ValueError('Supply a correction reason of at most 10000 characters')
    if not edits:
        raise ValueError('Supply at least one correction')
    before = copy.deepcopy(advice['result'])
    result, changed, new_corrections = before, [], []
    for item in edits:
        prior = result
        try:
            result, ids = apply_edit(prior, item['target'], item['field'], item['value'])
        except ValueError as exc:
            # Name the offending cell so the reviewer can find it in the table.
            raise ValueError(f"{item['target']} / {item['field']}: {exc}") from None
        row = next((r for r in prior['line_items'] if r['row_id'] in ids), prior['header'])
        new_corrections.append({'target': ids[0] if ids else 'header', 'reference': row.get('invoice_reference'),
            'field': item['field'], 'before': row.get(item['field']), 'value': item['value'], 'reason': reason})
        changed += [i for i in ids if i not in changed]
    advice['corrections'] = [*advice.get('corrections', []), *new_corrections]
    advice.update(result=result, status='needs_review', verification=asdict(verify_canonical(result['header'], result['line_items'])))
    with store.engine.begin() as conn:
        saved = store.update(advice, revision, conn)
        for correction in new_corrections:
            store.insert('edit', {**correction, 'before_result': before, 'after_result': result,
                'advice_revision': saved['revision'], 'actor': 'shared-poc-credential'}, parent=advice_id, conn=conn)
    refresh_email(store, advice['parent_id'])
    return {**saved, 'changed_row_ids': changed}


def revert_all(store, advice_id, revision):
    """Drop every active correction and restore the interpreted result; the audit trail is kept.

    Clearing ``corrections`` also means a later Retry has nothing to replay, so the restore survives
    reprocessing. Returns the saved advice plus ``changed_row_ids`` of the reverted rows.
    """
    advice = advice_record(store, advice_id, revision, mutable=True)
    if advice['status'] in {'queued', 'processing'}:
        raise Conflict('Wait for processing before undo')
    if not advice.get('corrections'):
        raise ValueError('No correction to undo')
    if not advice.get('original_result'):
        raise ValueError('Original values are unavailable for this advice')
    previous = advice['corrections']
    result = copy.deepcopy(advice['original_result'])
    advice.update(result=result, corrections=[], status='needs_review',
        verification=asdict(verify_canonical(result['header'], result['line_items'])))
    with store.engine.begin() as conn:
        saved = store.update(advice, revision, conn)
        store.insert('edit', {'operation': 'revert_all', 'corrections': previous, 'advice_revision': saved['revision'],
            'actor': 'shared-poc-credential'}, parent=advice_id, conn=conn)
    refresh_email(store, advice['parent_id'])
    return {**saved, 'changed_row_ids': sorted({c['target'] for c in previous if c['target'] != 'header'})}


def undo(store, advice_id, revision):
    """Undo the latest active correction, preserving original extraction and audit."""
    advice = advice_record(store, advice_id, revision, mutable=True)
    if advice['status'] in {'queued', 'processing'}:
        raise Conflict('Wait for processing before undo')
    if not advice.get('corrections'):
        raise ValueError('No correction to undo')
    previous = advice['corrections'][-1]
    result, ids = apply_edit(advice['result'], previous['target'], previous['field'], previous['before'])
    advice.update(result=result, corrections=advice['corrections'][:-1], status='needs_review',
        verification=asdict(verify_canonical(result['header'], result['line_items'])))
    with store.engine.begin() as conn:
        saved = store.update(advice, revision, conn)
        store.insert('edit', {'operation': 'undo', 'correction': previous, 'advice_revision': saved['revision'],
            'actor': 'shared-poc-credential'}, parent=advice_id, conn=conn)
    refresh_email(store, advice['parent_id'])
    return {**saved, 'changed_row_ids': ids}


def propose(store, advice_id, revision, playbook_text, reason):
    """Stage a full scoped playbook revision, never save rules from the advice agent."""
    advice = advice_record(store, advice_id, revision)
    if not advice.get('client_key'):
        raise ValueError('Confirm the customer before proposing rules')
    if not playbook_text.strip() or len(playbook_text) > 200000 or not reason.strip():
        raise ValueError('Provide a bounded playbook and supporting reason')
    pb = get_playbook(store.engine, advice['client_key'])
    proposal = {'client_key': advice['client_key'], 'rule_revision': pb.revision if pb else 0,
        'playbook_text': playbook_text, 'before_text': pb.playbook_text if pb else '',
        'anchors': pb.anchors if pb else {}, 'reason': reason, 'advice_revision': revision,
        'scope': 'Future interpretation for this customer only; existing advices remain unchanged'}
    proposal_id = store.insert('proposal', proposal, parent=advice_id, status='pending')
    return store.get(proposal_id)


def confirm(store, advice_id, proposal_id):
    """Commit the exact pending proposal with atomic customer/rule revision checks."""
    proposal = store.get(proposal_id)
    advice = advice_record(store, advice_id)
    if proposal['kind'] != 'proposal' or proposal['parent_id'] != advice_id:
        raise KeyError(proposal_id)
    if proposal['status'] != 'pending' or proposal['client_key'] != advice.get('client_key') or proposal['advice_revision'] != advice['revision']:
        raise Conflict('Proposal is stale; request a new proposal')
    with store.engine.begin() as conn:
        # CAS the advice too: customer correction cannot race rule confirmation.
        store.update(advice, conn=conn)
        params = {'ck': proposal['client_key'], 'pt': proposal['playbook_text'],
            'aj': json.dumps(proposal['anchors']), 'rev': proposal['rule_revision'], 'next': proposal['rule_revision'] + 1}
        if proposal['rule_revision']:
            changed = conn.execute(text(f'''UPDATE "{DEDUCTION_RULES}" SET "PLAYBOOK_TEXT"=:pt,
                "ANCHORS_JSON"=:aj,"REVISION"=:next,"UPDATED_BY"='shared-poc-credential',"UPDATED_AT"=CURRENT_UTCTIMESTAMP
                WHERE "CLIENT_KEY"=:ck AND "REVISION"=:rev'''), params)
            if changed.rowcount != 1:
                raise Conflict('Customer rules changed; request a new proposal')
        else:
            conn.execute(text(f'''INSERT INTO "{DEDUCTION_RULES}" ("CLIENT_KEY","PLAYBOOK_TEXT","ANCHORS_JSON","REVISION","UPDATED_BY")
                VALUES (:ck,:pt,:aj,:next,'shared-poc-credential')'''), params)
        proposal['status'] = 'confirmed'
        return store.update(proposal, conn=conn)
