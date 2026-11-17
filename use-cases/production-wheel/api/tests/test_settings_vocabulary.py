"""Keep the settings-vocabulary agent skill in sync with the draft schemas.

Run with:
    cd api && python -m pytest tests/test_settings_vocabulary.py -q
"""

import re
import typing
from pathlib import Path

from pydantic import BaseModel

from app.workspace.models import ExecutionBudget
from production_wheel.schemas import ConstraintBase, ConstraintSpec, SolveRequest

SKILL = Path(__file__).parents[1] / 'app/agent/agent_skills/settings-vocabulary/SKILL.md'


def _field_names(model: type[BaseModel], seen: set) -> set[str]:
    """Return every field name of a model and of its nested models.

    Args:
        model: Pydantic model class to walk.
        seen: Models already visited, preventing repeated traversal.

    Returns:
        Field names found in the model tree. Discriminated unions are skipped;
        their members are documented by kind name instead.
    """
    if model in seen:
        return set()
    seen.add(model)
    names = set()
    for name, field in model.model_fields.items():
        names.add(name)
        annotation = field.annotation
        # Unwrap tuple[Model, ...] so repeated nested objects are also covered.
        candidates = typing.get_args(annotation) if typing.get_origin(annotation) is tuple else (annotation,)
        for candidate in candidates:
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                names |= _field_names(candidate, seen)
    return names


def _documented_tokens() -> set[str]:
    """Return every identifier written inside backticks in the skill text."""
    spans = re.findall(r'`([^`]+)`', SKILL.read_text())
    return {token for span in spans for token in re.findall(r'\w+', span)}


def test_every_draft_setting_and_constraint_kind_is_documented():
    """A new schema field or constraint kind must get a plain-language entry."""
    union = typing.get_args(ConstraintSpec)[0]
    kinds = {typing.get_args(member.model_fields['kind'].annotation)[0]
             for member in typing.get_args(union)}
    seen: set = set()
    required = (_field_names(ExecutionBudget, seen) | _field_names(SolveRequest, seen)
                | _field_names(ConstraintBase, seen) | kinds)
    missing = sorted(required - _documented_tokens())
    assert not missing, f'settings-vocabulary lacks entries for: {missing}'
