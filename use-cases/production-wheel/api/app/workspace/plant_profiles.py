"""Versioned plant configuration, relational matrices and frozen run inheritance."""
from __future__ import annotations

import json
import uuid
from typing import Literal

from pydantic import Field, model_validator
from production_wheel.rule_models import BusinessConstraint
from production_wheel.runners import CycleStockPolicy, RunnerThresholds
from production_wheel.sap_settings.profile_policy import SapSettingsPolicy
from .models import StrictModel
from .datasets_service import now
from .profile_matrix import ProfileMatrixPair, validate_matrix_rows


class PlantSettings(StrictModel):
    """Fixed demand calendar, runner classification and grouping limits."""

    horizon_days: int = Field(default=250, gt=0)
    demand_days_per_week: float = Field(default=5, gt=0, le=7, allow_inf_nan=False)
    # High / medium / low runner cut-offs; also used by SAP settings of runs from this profile.
    runner_thresholds: RunnerThresholds = Field(default_factory=RunnerThresholds)
    runner_basis: Literal['reference', 'candidate_pv'] = 'reference'
    # Single RoC or runner multiples per subgroup; also read by SAP settings of runs from this profile.
    cycle_stock_policy: CycleStockPolicy = Field(default_factory=CycleStockPolicy)
    matrix_mode: Literal['FLEXIBLE', 'HARD', 'DIAGNOSTIC', 'OFF'] = 'FLEXIBLE'
    # How FINI package volumes find their matrix row (production_wheel.volume_classes):
    # 'exact' = the matrix lists every package volume; 'nearest_class' = the matrix
    # lists reference volumes (stops) plus optional '<n' / '>n' classes and each
    # volume takes the row of its nearest stop. New profiles get 'nearest_class'
    # from plant_profile_defaults(); a payload without the key stays 'exact'.
    volume_matching: Literal['exact', 'nearest_class'] = 'exact'
    maximum_group_size: int = Field(default=7, ge=1)
    # Rules of the SAP settings generated after a run (DC lot sizes, planning calendar
    # profiles, recipe characters, ...). The optimizer never reads them, and SAP settings
    # generation reads the profile's current revision, so editing them needs no re-run.
    sap_settings: SapSettingsPolicy = Field(default_factory=SapSettingsPolicy)


def upgrade_settings(settings):
    """Return profile settings in the current shape (three-tier runners, cycle stock policy).

    Profiles and frozen draft copies saved before the three-tier classes carry a
    two-tier ``high_runner_threshold_days``. That value has no three-tier
    equivalent, so it is dropped and default ``runner_thresholds`` are added.
    Copies saved before the cycle stock strategy get the default
    ``cycle_stock_policy`` (``single``, today's behaviour), and copies saved
    before SAP settings moved into the profile get default ``sap_settings``.
    Stored profiles are normally upgraded once at startup with their plant's SAP
    policy values (see ``schema_migrations.migrate_runner_thresholds`` and
    ``migrate_cycle_stock_policy``, ``migrate_sap_settings``); this is the
    fallback for copies that migration does not reach.

    Args:
        settings: Stored settings mapping.

    Returns:
        A new settings dict that ``PlantSettings`` accepts.
    """
    upgraded = {k: v for k, v in settings.items() if k != 'high_runner_threshold_days'}
    upgraded.setdefault('runner_thresholds', RunnerThresholds().model_dump())
    upgraded.setdefault('cycle_stock_policy', CycleStockPolicy().model_dump())
    upgraded.setdefault('sap_settings', SapSettingsPolicy().model_dump())
    # Copies saved before volume classes list every package volume exactly.
    upgraded.setdefault('volume_matching', 'exact')
    return upgraded


def merge_settings(current, patch):
    """Apply a partial settings patch, merging nested sections key by key.

    A patch such as ``{"sap_settings": {"calendar_odd_profile": "ODD-W1"}}`` must
    change that one value and keep every other SAP setting; a plain dict update
    would replace the whole section with defaults.

    Args:
        current: Current settings mapping (already upgraded).
        patch: Partial settings. Nested dicts are merged recursively; any other
            value, including ``None`` (clear an optional value), replaces the
            current one.

    Returns:
        A new merged settings dict; the inputs are not modified.
    """
    merged = dict(current)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def optimizer_view(profile):
    """Return the parts of a profile that change optimization results.

    SAP settings rules are excluded: they only shape the SAP upload generated
    after a run, so editing them must not mark drafts as outdated.

    Args:
        profile: Profile dict with ``plant``, ``settings``, ``rules`` and ``matrix_rows``.

    Returns:
        A comparable dict of plant, settings without ``sap_settings``, rules and
        matrix rows (sorted, so storage order does not matter).
    """
    settings = {k: v for k, v in upgrade_settings(profile.get('settings') or {}).items() if k != 'sap_settings'}
    matrix = sorted(json.dumps(row, sort_keys=True, default=str) for row in profile.get('matrix_rows') or [])
    return {'plant': profile.get('plant'), 'settings': settings,
            'rules': profile.get('rules') or [], 'matrix_rows': matrix}


class ProfileInput(StrictModel):
    """Reviewed profile inputs; saving approves the finite typed rule trees."""

    name: str = Field(min_length=1, max_length=255)
    plant: str = Field(min_length=1, max_length=128)
    settings: PlantSettings = Field(default_factory=PlantSettings)
    rules: list[BusinessConstraint] = Field(default_factory=list, max_length=100)
    matrix_rows: list[ProfileMatrixPair] = Field(min_length=1, max_length=10000)

    @model_validator(mode='after')
    def check_review(self):
        """Reject blank names, incomplete/asymmetric matrices and missing interpretation."""
        if not self.name.strip() or not self.plant.strip():
            raise ValueError('profile name and plant cannot be blank')
        ids = [r.constraint_id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError('profile rule identifiers must be unique')
        validate_matrix_rows(self.matrix_rows, self.settings.volume_matching)
        if any(r.scope.plant and r.scope.plant != self.plant for r in self.rules):
            raise ValueError('profile rules must match profile plant')
        return self


class ProfileChange(StrictModel):
    """One reviewable edit of a saved profile, proposed by the assistant or the Settings list.

    Actions:
        add_constraint / replace_constraint: `constraint` is a typed business rule;
            replace keeps the position of the rule with the same constraint_id.
        remove_constraint: `constraint_id` names the rule to drop.
        update_settings: `settings` is a partial PlantSettings patch (nested
            sections such as `sap_settings` are merged key by key) and/or
            `matrix_rows` is a complete replacement volume matrix.
    """

    action: Literal['add_constraint', 'replace_constraint', 'remove_constraint', 'update_settings']
    constraint: BusinessConstraint | None = None
    constraint_id: str | None = Field(default=None, min_length=1, max_length=128)
    settings: dict | None = None
    matrix_rows: list[ProfileMatrixPair] | None = Field(default=None, max_length=10000)

    @model_validator(mode='after')
    def check_payload(self):
        """Require exactly the payload each action needs."""
        constraint_action = self.action in ('add_constraint', 'replace_constraint')
        if constraint_action != (self.constraint is not None):
            raise ValueError(f'{self.action} requires constraint only for add/replace')
        if (self.action == 'remove_constraint') != (self.constraint_id is not None):
            raise ValueError('remove_constraint requires constraint_id')
        if (self.action == 'update_settings') != (self.settings is not None or self.matrix_rows is not None):
            raise ValueError('update_settings requires settings and/or matrix_rows')
        return self


def apply_profile_change(profile, change):
    """Return the validated full ProfileInput payload after applying one change.

    Args:
        profile: Stored profile dict including rules, settings and matrix_rows.
        change: ProfileChange or its dict form.

    Returns:
        JSON-ready dict with name, plant, settings, rules and matrix_rows.

    Raises:
        ValueError: unknown/duplicate constraint IDs or any ProfileInput violation
            (grammar, plant scope, incomplete matrix, invalid settings).
    """
    change = ProfileChange.model_validate(change)
    rules = list(profile.get('rules') or [])
    ids = [r['constraint_id'] for r in rules]
    settings, matrix_rows = upgrade_settings(profile['settings']), profile['matrix_rows']
    if change.constraint is not None:
        # Profile rules always belong to the profile plant; approval happens on save.
        rule = change.constraint.model_dump(mode='json')
        rule['scope'] = {**rule['scope'], 'plant': profile['plant']}
        rule['approval_status'] = 'approved'
        if change.action == 'add_constraint':
            if rule['constraint_id'] in ids:
                raise ValueError(f"constraint {rule['constraint_id']} already exists; use replace_constraint")
            rules.append(rule)
        elif rule['constraint_id'] not in ids:
            raise ValueError(f"constraint {rule['constraint_id']} does not exist; use add_constraint")
        else:
            rules[ids.index(rule['constraint_id'])] = rule
    elif change.action == 'remove_constraint':
        if change.constraint_id not in ids:
            raise ValueError(f'constraint {change.constraint_id} does not exist')
        rules.pop(ids.index(change.constraint_id))
    else:
        settings = PlantSettings.model_validate(merge_settings(settings, change.settings)).model_dump()
        if change.matrix_rows is not None:
            matrix_rows = [row.model_dump() for row in change.matrix_rows]
    return ProfileInput.model_validate({'name': profile['name'], 'plant': profile['plant'],
        'settings': settings, 'rules': rules, 'matrix_rows': matrix_rows}).model_dump(mode='json')


# Profile values that only seed a new draft. A scenario may change them; the change is
# recorded as explicit in parameter_sources and frozen with the run. The compatibility
# mode decides how much the solver avoids discouraged pairs; the pairs themselves
# (matrix_pairs) stay fixed by the profile.
PROFILE_DEFAULT_KEYS = frozenset({'matrix_mode'})


def profile_matrix_pairs(profile, rows):
    """Return the explicit volume pairs and volume classes a run freezes from a profile.

    Args:
        profile: Frozen profile (``settings``, ``matrix_rows``, ``plant``).
        rows: ``fini_master`` rows of the run's dataset.

    Returns:
        ``(matrix_pairs, matrix_volume_classes)``. In ``exact`` mode the profile
        rows unchanged and no classes. In ``nearest_class`` mode the class matrix
        expanded to every package volume of the profile plant
        (``production_wheel.volume_classes.expand_class_matrix``) and the sorted
        ``[[volume, class], ...]`` map. Volumes in a tail the matrix lacks stay
        out of the pairs, so the run's coverage check names them.
    """
    from production_wheel.volume_classes import NEAREST_CLASS, expand_class_matrix, plant_volumes

    settings = upgrade_settings(profile['settings'])
    if settings['volume_matching'] != NEAREST_CLASS:
        return profile['matrix_rows'], []
    pairs, classes, _ = expand_class_matrix(profile['matrix_rows'], plant_volumes(rows, profile.get('plant')))
    return pairs, classes


def profile_config(profile, rows):
    """Translate a frozen profile and source block inventory into authoritative config."""
    settings = upgrade_settings(profile['settings'])
    matrix_pairs, volume_classes = profile_matrix_pairs(profile, rows)
    return {
        'demand_days': settings['horizon_days'],
        'productive_weeks': settings['horizon_days'] / settings['demand_days_per_week'],
        'runner_thresholds': settings['runner_thresholds'],
        'runner_basis': settings['runner_basis'],
        'cycle_stock_policy': settings['cycle_stock_policy'],
        'matrix_mode': settings['matrix_mode'],
        'matrix_pairs': matrix_pairs,
        'volume_matching': settings['volume_matching'],
        'matrix_volume_classes': volume_classes,
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
        """Return the defaults of a new profile: editable settings and the default class matrix.

        New profiles use ``nearest_class`` volume matching with the customer's
        class matrix (``production_wheel.volume_classes.default_class_matrix``).
        """
        from production_wheel.volume_classes import NEAREST_CLASS, default_class_matrix
        return {'settings': PlantSettings(volume_matching=NEAREST_CLASS).model_dump(), 'rules': [],
                'matrix_rows': default_class_matrix()}

    def dataset_volume_coverage(self, dataset_id, profile_id=None, proposed_stops=None):
        """Compare a dataset's package volumes with a plant profile's volume matrix.

        Args:
            dataset_id: Dataset whose ``fini_master`` rows are counted. Products
                modeled in either production version mode are included.
            profile_id: Optional profile; when given, rows are limited to its plant
                and each volume is marked as covered or missing by its matrix.
            proposed_stops: Optional class axis (for example ``["<1", "1", "2",
                "2.5", "5", "10", "15", ">15"]``) to preview how the volumes would
                be classified; nothing is saved.

        Returns:
            ``dataset_volumes`` (``{volume, product_count, in_profile_matrix}``,
            plus ``matrix_class`` in ``nearest_class`` mode and ``proposed_class``
            with ``proposed_stops``), ``missing_volumes`` and
            ``missing_product_count`` (volumes the profile matrix cannot place; a
            run stops on them), ``profile_volumes_unused`` (matrix rows no product
            uses), ``volume_matching``, ``class_product_counts`` (class mode) and
            ``reclassified_volumes`` (with ``proposed_stops``: volumes whose class
            changes, ``{volume, product_count, from, to}``).
        """
        from production_wheel.volume_classes import (
            NEAREST_CLASS, check_labels, classify_volume, label_sort_key, sorted_labels)
        from .matrix_description import dataset_volume_usage

        profile = self.get_plant_profile(profile_id) if profile_id else None
        usage = dataset_volume_usage(self.repo.rows(dataset_id, 'fini_master'), profile['plant'] if profile else None)
        matching = upgrade_settings(profile['settings'])['volume_matching'] if profile else None
        labels = sorted_labels([row['volume_a'] for row in profile['matrix_rows']]) if profile else []
        classed = matching == NEAREST_CLASS
        proposed = check_labels(proposed_stops) if proposed_stops else None

        def placed(volume):
            """Return the matrix row a volume uses, or None when the matrix cannot place it."""
            if not profile:
                return None
            if classed:
                return classify_volume(volume, labels)
            return volume if volume in labels else None

        items, missing, used = [], [], set()
        for volume, count in usage.items():
            row = placed(volume)
            item = {'volume': volume, 'product_count': count,
                    'in_profile_matrix': (row is not None) if profile else None}
            if classed:
                item['matrix_class'] = row
            if proposed:
                item['proposed_class'] = classify_volume(volume, proposed)
            if profile and row is None:
                missing.append({'volume': volume, 'product_count': count})
            used.add(row)
            items.append(item)
        result = {
            'dataset_id': dataset_id,
            'profile_id': profile_id,
            'plant': profile['plant'] if profile else None,
            'volume_matching': matching,
            'dataset_volume_count': len(usage),
            'dataset_product_count': sum(usage.values()),
            'profile_matrix_volume_count': len(labels),
            'dataset_volumes': items,
            'missing_volumes': missing,
            'missing_product_count': sum(item['product_count'] for item in missing),
            'profile_volumes_unused': [v for v in labels if v not in used],
        }
        if classed:
            counts = {}
            for item in items:
                if item['matrix_class']:
                    counts[item['matrix_class']] = counts.get(item['matrix_class'], 0) + item['product_count']
            result['class_product_counts'] = dict(sorted(counts.items(), key=lambda kv: label_sort_key(kv[0])))
        if proposed:
            result['proposed_stops'] = proposed
            result['reclassified_volumes'] = [
                {'volume': item['volume'], 'product_count': item['product_count'],
                 'from': item.get('matrix_class', item['volume'] if item['in_profile_matrix'] else None),
                 'to': item['proposed_class']}
                for item in items
                if item.get('matrix_class', item['volume'] if item['in_profile_matrix'] else None) != item['proposed_class']]
        return result

    def list_plant_profiles(self):
        """List active profiles with their relational matrix for Settings editing."""
        return [self.get_plant_profile(p['profile_id']) for p in self.repo.list('profiles', {'status': 'active'})]

    def get_plant_profile(self, profile_id, include_removed=False):
        """Read one profile, rejecting removed selections unless explicitly inspecting history."""
        profile = self.repo.get('profiles', profile_id)
        if profile['status'] != 'active' and not include_removed:
            raise ValueError('plant profile was removed; select an active profile')
        profile['settings'] = upgrade_settings(profile['settings'])
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
            if isinstance(patch.get('settings'), dict) and 'sap_settings' not in patch['settings']:
                # Clients that predate the SAP settings section send settings without it;
                # keep the stored rules instead of resetting them to defaults.
                patch['settings'] = {**patch['settings'], 'sap_settings': current['settings']['sap_settings']}
            value = ProfileInput.model_validate({**{k:current[k] for k in ProfileInput.model_fields if k in current}, **patch}).model_dump(mode='json')
            value['rules'] = [{**r, 'approval_status': 'approved'} for r in value['rules']]
            rows = value.pop('matrix_rows')
            updated = self.repo.cas('profiles', profile_id, revision, value)
            self.repo.replace_tables(profile_id, {'plant_profile_matrix': rows})
        return {**updated, 'matrix_rows': rows}

    def apply_plant_profile_change(self, profile_id, revision, change):
        """Apply one user-approved ProfileChange against the exact current revision."""
        with self.repo.transaction():
            current = self.get_plant_profile(profile_id)
            if revision != current['revision']:
                raise ValueError('stale profile revision; the profile changed since this proposal')
            value = apply_profile_change(current, change)
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
