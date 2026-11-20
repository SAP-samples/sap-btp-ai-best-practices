"""Keep explicit replacement wording aligned with proven, directed BOM pairs."""

import re


def normalize_replacement_prose(payload, bom_changes):
    """Return a copied stage payload with exact reversed replacement clauses fixed.

    Only pairs classified REPLACED by the BOM engine qualify. If both directions
    exist in the assessed changes, leave the prose unchanged rather than guessing
    its plant/date context. Other wording, uncertainty, IDs and fields are kept.
    The original provider response remains unchanged for diagnostics.
    """
    pairs = {(line['outgoing']['number'], line['incoming']['number'])
             for line in bom_changes.get('lines', []) if line.get('change') == 'REPLACED'
             and line.get('outgoing') and line.get('incoming')}
    corrections = []
    for old, new in sorted(pairs):
        if (new, old) in pairs:
            continue
        # Fix the observed passive inversion and its active equivalent. Do not
        # match negative/conditional clauses or infer a pair from mere co-usage.
        corrections.extend([
            (re.compile(r'(?<!\w)(?:new part\s+)?' + re.escape(new) +
                        r'\s+is replaced by\s+' + re.escape(old) + r'(?!\w)', re.I), f'{old} is replaced by {new}'),
            (re.compile(r'(?<!\w)' + re.escape(old) + r'\s+replaces\s+(?:new part\s+)?' +
                        re.escape(new) + r'(?!\w)', re.I), f'{new} replaces {old}'),
        ])

    def visit(value):
        """Copy JSON-compatible values, applying only the proven text corrections."""
        # Pydantic's strict Python dump retains StrEnum fields; do not coerce
        # validated severity/status enums while adjusting ordinary prose.
        if type(value) is str:
            for pattern, replacement in corrections:
                def correct(match):
                    """Keep questions and hypothetical clauses; correct explicit assertions."""
                    prefix = re.split(r'[.!?\n]', value[:match.start()])[-1]
                    suffix = re.split(r'[.!\n]', value[match.end():])[0]
                    if re.search(r'\b(if|whether|might|may|could|would|not|never)\b', prefix, re.I) or '?' in suffix:
                        return match.group()
                    return replacement
                value = pattern.sub(correct, value)
            return value
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(payload)
