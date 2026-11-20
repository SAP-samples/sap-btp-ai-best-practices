"""HANA persistence for the Settings "Plant Code Mapping" and rule-state filter.

Tables (created automatically by ``ensure_schema``):

* ``ECN_IMPACT_PLANT_MAPPING``: one row per brand suffix (``JB`` -> PL10 Plant A).
* ``ECN_IMPACT_APP_SETTING``: key ``excluded_rule_states`` holds a JSON list of
  rule lifecycle states ignored by plant assignment. A missing key means the
  default (``DEFAULT_EXCLUDED_RULE_STATES``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ecn.plant_settings import (
    DEFAULT_EXCLUDED_RULE_STATES,
    PlantMapping,
    PlantSettings,
)
from app.persistence.codec import normalize_row_keys, read_lob_text
from app.persistence.hana import to_hana_timestamp
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier

EXCLUDED_RULE_STATES_KEY = "excluded_rule_states"


class PlantSettingsRepository:
    """Read and atomically replace the plant-assignment settings.

    Args:
        engine: SQLAlchemy engine connected to HANA.
        tables: Physical table names; defaults to production names.
    """

    def __init__(self, engine: Engine, tables: PersistenceTableNames = PersistenceTableNames()) -> None:
        """Store the engine and table names used by every query."""
        self._engine = engine
        self._tables = tables

    def get(self) -> PlantSettings:
        """Return the saved mapping, excluded states and states seen in rule uploads.

        Returns:
            ``PlantSettings`` with mappings sorted by suffix. The available
            states are every distinct ``LIFECYCLE_STATE`` of uploaded rule rows
            plus the saved exclusions, so a saved state never disappears from
            the Settings checkboxes.
        """
        mapping = quote_identifier(self._tables.plant_mapping)
        setting = quote_identifier(self._tables.app_setting)
        rules = quote_identifier(self._tables.conditional_rule_row)
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(f'SELECT "BRAND_SUFFIX", "PLANT_CODE", "PLANT_NAME" FROM {mapping} ORDER BY "BRAND_SUFFIX"')
            ).mappings().all()
            saved = connection.execute(
                text(f'SELECT "VALUE_JSON" FROM {setting} WHERE "SETTING_KEY" = :key'),
                {"key": EXCLUDED_RULE_STATES_KEY},
            ).mappings().all()
            states = connection.execute(
                text(f'SELECT DISTINCT "LIFECYCLE_STATE" FROM {rules} WHERE "LIFECYCLE_STATE" IS NOT NULL')
            ).mappings().all()
        mappings = tuple(
            PlantMapping(row["BRAND_SUFFIX"], row["PLANT_CODE"], row["PLANT_NAME"])
            for row in map(normalize_row_keys, rows)
        )
        excluded = (
            tuple(json.loads(read_lob_text(normalize_row_keys(saved[0])["VALUE_JSON"])))
            if saved else DEFAULT_EXCLUDED_RULE_STATES
        )
        seen = {normalize_row_keys(row)["LIFECYCLE_STATE"] for row in states}
        return PlantSettings(mappings, excluded, tuple(sorted(seen | set(excluded))))

    def replace(self, settings: PlantSettings) -> PlantSettings:
        """Replace all mapping rows and the excluded states in one transaction.

        Args:
            settings: Already validated settings (``normalize_plant_settings``).

        Returns:
            The settings as read back after commit.
        """
        mapping = quote_identifier(self._tables.plant_mapping)
        setting = quote_identifier(self._tables.app_setting)
        now = to_hana_timestamp(datetime.now(timezone.utc))
        with self._engine.begin() as connection:
            connection.execute(text(f"DELETE FROM {mapping}"))
            for row in settings.mappings:
                connection.execute(
                    text(
                        f'INSERT INTO {mapping} ("BRAND_SUFFIX", "PLANT_CODE", "PLANT_NAME", "UPDATED_AT") '
                        "VALUES (:suffix, :code, :name, :now)"
                    ),
                    {"suffix": row.suffix, "code": row.plant_code, "name": row.plant_name, "now": now},
                )
            connection.execute(
                text(f'DELETE FROM {setting} WHERE "SETTING_KEY" = :key'), {"key": EXCLUDED_RULE_STATES_KEY}
            )
            connection.execute(
                text(f'INSERT INTO {setting} ("SETTING_KEY", "VALUE_JSON", "UPDATED_AT") VALUES (:key, :value, :now)'),
                {"key": EXCLUDED_RULE_STATES_KEY, "value": json.dumps(list(settings.excluded_rule_states)), "now": now},
            )
        return self.get()
