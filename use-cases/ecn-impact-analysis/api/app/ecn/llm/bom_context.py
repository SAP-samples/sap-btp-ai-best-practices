"""Bound BOM and plant examples while preserving complete deterministic totals."""

from collections import Counter, defaultdict
from itertools import zip_longest

EXAMPLE_LIMIT = 25


def _examples(rows, group_key):
    """Return up to 25 rows, alternating groups so small categories stay visible."""
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return [row for group in zip_longest(*groups.values()) for row in group if row is not None][:EXAMPLE_LIMIT]


def _sample(section, key, rows, group_key):
    """Attach sampled rows and exact total/omitted counts to a copied section."""
    section[key] = _examples(rows, group_key)
    section[f'{key}_total'] = len(rows)
    section[f'{key}_omitted'] = len(rows) - len(section[key])


def bounded_bom_context(snapshot):
    """Return independent, compact BOM/plant sections with explicit truncation counts.

    The input snapshot remains complete for UI and API use. Model examples omit
    duplicate source payloads; evidence records supply bounded source citations.
    Counts cover the complete assessment, including rows omitted from examples.
    """
    changes, impact = snapshot.get('bom_changes') or {}, snapshot.get('plant_impact') or {}
    bom = {k: v for k, v in changes.items() if k not in {'lines', 'materials'}}
    lines = [{k: v for k, v in row.items() if k != 'sources'} for row in changes.get('lines', [])]
    materials = [{**row, 'still_used_in': row.get('still_used_in', [])[:5],
                  'still_used_in_total': len(row.get('still_used_in', []))} for row in changes.get('materials', [])]
    _sample(bom, 'lines', lines, lambda r: (tuple(p['plant_code'] for p in r.get('plants', [])), r.get('change')))
    _sample(bom, 'materials', materials, lambda r: (r.get('role'), r.get('status')))
    plants = []
    for row in impact.get('plants', []):
        example = dict(row)
        for key in ('old_part_stock', 'new_parts_with_purchasing', 'new_parts_without_purchasing', 'materials'):
            example[key] = row.get(key, [])[:5]
            example[f'{key}_total'] = len(row.get(key, []))
            example[f'{key}_omitted'] = max(0, len(row.get(key, [])) - 5)
        plants.append(example)
    plant_context = {k: v for k, v in impact.items() if k != 'plants'}
    plant_context['status_counts'] = dict(Counter(row.get('status') for row in plants))
    _sample(plant_context, 'plants', plants, lambda r: r.get('status'))
    return {'bom_changes': bom, 'plant_impact': plant_context}


def trim_bom_example(context):
    """Remove one large-section example, update omissions and report whether reduced."""
    candidates = [(context.get(section, {}), key) for section, key in
                  (('bom_changes', 'lines'), ('bom_changes', 'materials'), ('plant_impact', 'plants'))]
    section, key = max(candidates, key=lambda pair: len(pair[0].get(pair[1], [])))
    if not section.get(key):
        return False
    section[key].pop()
    section[f'{key}_omitted'] = section[f'{key}_total'] - len(section[key])
    return True
