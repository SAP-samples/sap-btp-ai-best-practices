"""Conservative pairing of dated BOM rows within one parent and alternative."""

from app.ecn.evidence.material_changes import _number_pattern


def pair_rows(outgoing, incoming, ecn_pairs):
    """Return proven row pairs and leftovers without losing repeated part rows.

    Inputs are lists in one BOM kind, parent, alternative, and sequence. Output
    is (pairs of old/new/basis, remaining old rows, remaining new rows). A rule
    must match uniquely in both directions. Duplicate rows stay unpaired.
    """
    olds, news, pairs = list(outgoing), list(incoming), []
    rules = (
        ('SAME_PART', lambda old, new: old == new),
        ('ECN', lambda old, new: new in ecn_pairs.get(old, ())),
        ('NUMBER', _number_pattern),
    )
    for basis, matches in rules:
        selected = []
        for old_index, old in enumerate(olds):
            candidates = [i for i, new in enumerate(news) if matches(old['number'], new['number'])]
            if len(candidates) == 1:
                new_index = candidates[0]
                if sum(matches(other['number'], news[new_index]['number']) for other in olds) == 1:
                    selected.append((old_index, new_index))
        pairs.extend((olds[o], news[n], basis) for o, n in selected)
        olds = [row for i, row in enumerate(olds) if i not in {o for o, _ in selected}]
        news = [row for i, row in enumerate(news) if i not in {n for _, n in selected}]
    if len(olds) == len(news) == 1:
        pairs.append((olds.pop(), news.pop(), 'ONLY_CHANGE'))
    return pairs, olds, news
