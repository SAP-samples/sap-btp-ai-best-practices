"""Compact, complete lists shared by overview, Readiness and document references."""


def complete_snapshot_views(snapshot, source_warnings=()):
    """Add source warnings, part actions, released versions and their exact counts.

    Mutates a completed snapshot using grouped warnings from its pinned bundle.
    Each warning is one collapsible Readiness item; the count is never taken
    from the upload's unrelated warning total. Returns nothing.
    """
    known = {question['code'] for question in snapshot['questions']}
    for warning in source_warnings:
        code = 'SOURCE_' + warning['code']
        if code in known:
            continue
        snapshot['questions'].append({
            'code': code, 'question': warning['code'].replace('_', ' ').capitalize(),
            'why': warning['summary'], 'affected_count': warning['count'],
            'resolution': 'Check the listed source records. Correct the file in Settings and reassess this ECN.',
            'subjects': [], 'scopes': warning.get('sources', []),
            'factors': ['manufacturing', 'procurement', 'configuration'],
        })
        known.add(code)
    actions = {change['number']: change for change in snapshot.get('changes', [])}
    snapshot['parts'] = [dict(actions.get(obj['number'], {'number': obj['number'], 'action': 'REFERENCED'}),
                               description=obj.get('description'), types=obj['types'],
                               revisions=sorted({ref['revision'] for ref in obj['source_references'] if ref.get('revision')}))
                          for obj in snapshot['objects'] if obj['technical'] and 'ecn' in obj['origins']]
    snapshot['reference_documents'] = [
        {'number': obj['number'], 'types': obj['types'], 'revision': ref.get('revision'),
         'state': ref.get('state'), 'path': ref.get('path')}
        for obj in snapshot['objects'] for ref in obj['source_references'] if ref['role'] == 'RESULTING'
    ]
    snapshot['counts']['attention_items'] = len(snapshot['questions'])
