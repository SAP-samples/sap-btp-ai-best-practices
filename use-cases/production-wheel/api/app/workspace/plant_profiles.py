"""Versioned plant configuration, relational matrices and frozen run inheritance."""
from __future__ import annotations

import json
import uuid
from typing import Literal

from pydantic import Field, model_validator
from production_wheel.rule_models import BusinessConstraint, MatrixPair
from .models import StrictModel
from .datasets_service import now
from .profile_matrix import validate_matrix_rows


class PlantSettings(StrictModel):
    """Fixed demand calendar, runner classification and grouping limits."""

    horizon_days: int = Field(default=250, gt=0)
    demand_days_per_week: float = Field(default=5, gt=0, le=7, allow_inf_nan=False)
    high_runner_threshold_days: float = Field(default=15, gt=0, allow_inf_nan=False)
    runner_basis: Literal['reference', 'candidate_pv'] = 'reference'
    matrix_mode: Literal['FLEXIBLE', 'HARD', 'DIAGNOSTIC', 'OFF'] = 'FLEXIBLE'
    maximum_group_size: int = Field(default=7, ge=1)


class ProfileInput(StrictModel):
    """Reviewed profile inputs; saving approves the finite typed rule trees."""

    name: str = Field(min_length=1, max_length=255)
    plant: str = Field(min_length=1, max_length=128)
    settings: PlantSettings = Field(default_factory=PlantSettings)
    rules: list[BusinessConstraint] = Field(default_factory=list, max_length=100)
    rules_text: str = Field(default='', max_length=30000)
    matrix_rules_text: str = Field(default='', max_length=30000)
    matrix_rows: list[MatrixPair] = Field(min_length=1, max_length=10000)

    @model_validator(mode='after')
    def check_review(self):
        """Reject blank names, incomplete/asymmetric matrices and missing interpretation."""
        if not self.name.strip() or not self.plant.strip():
            raise ValueError('profile name and plant cannot be blank')
        if self.rules_text.strip() and not self.rules and self.matrix_rules_text != self.rules_text:
            raise ValueError('preview and review rules before saving')
        ids = [r.constraint_id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError('profile rule identifiers must be unique')
        validate_matrix_rows(self.matrix_rows)
        if any(r.scope.plant and r.scope.plant != self.plant for r in self.rules):
            raise ValueError('profile rules must match profile plant')
        return self


def profile_config(profile, rows):
    """Translate a frozen profile and source block inventory into authoritative config."""
    settings = profile['settings']
    return {
        'demand_days': settings['horizon_days'],
        'productive_weeks': settings['horizon_days'] / settings['demand_days_per_week'],
        'high_runner_threshold_days': settings['high_runner_threshold_days'],
        'runner_basis': settings['runner_basis'],
        'matrix_mode': settings['matrix_mode'],
        'matrix_pairs': profile['matrix_rows'],
        'group_size': {'mode': 'HARD', 'base_limit': 7, 'max_excess': 0},
        'group_size_overrides': [{'plant': p, 'sefi': s, 'maximum': settings['maximum_group_size']} for p,s in sorted({(str(r['plant']),str(r['sefi'])) for r in rows if r.get('plant') == profile['plant'] and r.get('sefi')})],
        'assign_filling_lines': 'group.selected_line' in json.dumps(profile['rules']),
        'business_rules': profile['rules'],
    }


def check_profile_source(profile, rows):
    """Require matching plant and observed demand horizon; never rescale silently."""
    plants = {str(row['plant']) for row in rows if row.get('plant')}
    if profile['plant'] not in plants:
        raise ValueError('profile plant does not match dataset; select a compatible dataset')
    horizons = {float(row['source_horizon_days']) for row in rows if row.get('plant') == profile['plant'] and row.get('source_horizon_days') not in (None, '')}
    if any(abs(value - profile['settings']['horizon_days']) > 1e-8 for value in horizons):
        raise ValueError(f"source horizon {sorted(horizons)} differs from profile horizon {profile['settings']['horizon_days']} demand days; select matching settings or source forecast")


class PlantProfileService:
    """CRUD profiles with revision checks and automatic HANA relational storage."""

    def plant_profile_defaults(self):
        """Return current explicit operational matrix and editable settings defaults."""
        from production_wheel.matrix import build_customer_operational_volume_matrix
        return {'settings': PlantSettings().model_dump(), 'rules': [], 'rules_text': '', 'matrix_rows': [{k:r[k] for k in ('volume_a','volume_b','status')} for r in build_customer_operational_volume_matrix()]}

    def list_plant_profiles(self):
        """List active profiles with their relational matrix for Settings editing."""
        return [self.get_plant_profile(p['profile_id']) for p in self.repo.list('profiles', {'status': 'active'})]

    def get_plant_profile(self, profile_id, include_removed=False):
        """Read one profile, rejecting removed selections unless explicitly inspecting history."""
        profile = self.repo.get('profiles', profile_id)
        if profile['status'] != 'active' and not include_removed:
            raise ValueError('plant profile was removed; select an active profile')
        profile['matrix_rows'] = self.repo.rows(profile_id, 'plant_profile_matrix')
        return profile

    def create_plant_profile(self, body):
        """Validate reviewed inputs and insert the first profile revision atomically."""
        value = ProfileInput.model_validate(body).model_dump(mode='json')
        value['rules'] = [{**r, 'approval_status': 'approved'} for r in value['rules']]
        rows = value.pop('matrix_rows')
        value.update(profile_id=uuid.uuid4().hex, revision=1, status='active', created_at=now())
        with self.repo.transaction():
            self.repo.insert('profiles', value['profile_id'], value)
            self.repo.replace_tables(value['profile_id'], {'plant_profile_matrix': rows})
        return {**value, 'matrix_rows': rows}

    def update_plant_profile(self, profile_id, body):
        """Patch a complete reviewed profile with optimistic revision protection."""
        patch = dict(body)
        revision = patch.pop('revision', None)
        with self.repo.transaction():
            current = self.get_plant_profile(profile_id)
            if revision != current['revision']:
                raise ValueError('stale profile revision; reload profile')
            value = ProfileInput.model_validate({**{k:current[k] for k in ProfileInput.model_fields if k in current}, **patch}).model_dump(mode='json')
            value['rules'] = [{**r, 'approval_status': 'approved'} for r in value['rules']]
            rows = value.pop('matrix_rows')
            updated = self.repo.cas('profiles', profile_id, revision, value)
            self.repo.replace_tables(profile_id, {'plant_profile_matrix': rows})
        return {**updated, 'matrix_rows': rows}

    def remove_plant_profile(self, profile_id, revision, confirmed):
        """Soft-delete an explicitly confirmed revision, leaving frozen runs intact."""
        if confirmed is not True:
            raise ValueError('confirm profile removal explicitly')
        with self.repo.transaction():
            self.get_plant_profile(profile_id)
            return self.repo.cas('profiles', profile_id, revision, {'status': 'removed', 'removed_at': now()})
