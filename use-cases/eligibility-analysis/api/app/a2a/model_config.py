"""Assistant-specific model configuration, independent from reports and RPT-1."""


def resolve_assistant_model(environ):
    """Return the explicit assistant model or the approved Luna default."""
    return environ.get('A2A_MODEL','gpt-5.6-luna').strip() or 'gpt-5.6-luna'
