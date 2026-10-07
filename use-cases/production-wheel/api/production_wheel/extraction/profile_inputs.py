"""Preserve optional planning evidence and verify workbook-declared PV defaults."""

from __future__ import annotations

import math
import re
from statistics import median

from production_wheel.extraction.common import TableData, as_number, as_text

# Match the actual template lookup, including its plant/SEFI key and fallback.
# A mere mention of PV - 1 elsewhere in a formula is not default evidence.
_DEFAULT_FORMULA = '''=IFNA(INDEX(Production_versions___List[Lot size (Litres)],MATCH(CONCAT(FINI_grouping_for_wheel[[#This Row],[SEFI/Plant]],"/",FINI_grouping_for_wheel[[#This Row],[Selected PV]]),Production_versions___List[PV lot],0)),INDEX(Production_versions___List[Lot size (Litres)],MATCH(CONCAT(FINI_grouping_for_wheel[[#This Row],[SEFI/Plant]],"/","PV - 1"),Production_versions___List[PV lot],0)))'''


def _formula_key(value: object) -> str:
    """Normalize Excel compatibility prefixes and spacing for template comparison."""
    return re.sub(r'\s+', '', str(value).replace('_xlfn.', '')).casefold()


def parse_xyz(value: object) -> tuple[tuple[int | None, ...], str]:
    """Parse X-Y-Z nonnegative integer slots; missing cells stay unknown."""
    text = as_text(value)
    if not text:
        return (None, None, None), 'missing'
    parts = text.split('-')
    if len(parts) != 3 or any(part.strip() and not re.fullmatch(r'[0-9]+', part.strip()) for part in parts):
        return (None, None, None), 'invalid'
    return tuple(int(part.strip() or '0') for part in parts), 'valid'


def enrich_profile_inputs(rows: list[dict], main: TableData, formulas: TableData, versions: list[dict]) -> list[dict]:
    """Enrich newly imported rows from source tables/catalog and return issues.

    Historical baseline flags are already derived from explicit source selections
    and are never changed by the operational default. Conflicting source demand
    horizons remain unknown, with row evidence in validation issues.
    """
    issues = []
    source_rows = {record['_source_row']: record for record in main.records}
    formula_rows = {record['_source_row']: record for record in formulas.records}
    lots: dict[tuple[str, str], set[float]] = {}
    for version in versions:
        size = as_number(version.get('lot_size_litres'))
        if version.get('production_version') == 'PV - 1' and size is not None and size > 0:
            lots.setdefault((str(version.get('plant')), str(version.get('sefi'))), set()).add(size)
    ratios = []
    for row in rows:
        source = source_rows[row['source_row']]
        total = as_number(source.get('Total FINI forecast (litres)'))
        daily = as_number(source.get('Avg daily demand'))
        ratio = total / daily if total is not None and total > 0 and daily is not None and daily > 0 else None
        if ratio is not None and math.isfinite(ratio):
            ratios.append(ratio)
        else:
            ratio = None
        row.update(forecast_litres_period=total, source_horizon_days=ratio)
    horizon = median(ratios) if ratios else None
    consistent = horizon is not None and all(math.isclose(ratio, horizon, rel_tol=1e-6, abs_tol=1e-6) for ratio in ratios)
    for row in rows:
        source = source_rows[row['source_row']]
        formula = formula_rows.get(row['source_row'], {}).get('SEFI litres lot size of selected PV')
        declared_default = _formula_key(formula) == _formula_key(_DEFAULT_FORMULA)
        selected = as_text(source.get('Selected PV')) or None
        row.update(selected_pv_source=selected, fixed_pv_source='explicit' if selected else 'missing', fixed_pv_default_formula=str(formula) if declared_default else None)
        if not selected and declared_default and len(lots.get((str(row['plant']), str(row['sefi'])), set())) == 1:
            row.update(fixed_pv='PV - 1', fixed_pv_source='workbook_default')
        raw = source.get('XYZ - # DCs')
        counts, status = parse_xyz(raw)
        row.update(xyz_raw=raw, xyz_status=status, **dict(zip(('xyz_x_dc_count', 'xyz_y_dc_count', 'xyz_z_dc_count'), counts)))
        if status == 'invalid':
            issues.append(_input_issue(row, 'INVALID_XYZ_COUNTS', raw, 'Use three nonnegative integer slots X-Y-Z'))
        for column, field in (('Sales scenario', 'sales_scenario'), ('Sales network scenario', 'sales_network_scenario')):
            row[field] = as_text(source.get(column))
        for column, field in (('Primary DC count', 'primary_dc_count'), ('Secondary DC count', 'secondary_dc_count')):
            raw = source.get(column)
            count = as_number(raw)
            valid = count is not None and count >= 0 and count.is_integer()
            row[field + '_raw'] = raw
            row[field] = int(count) if valid else None
            if as_text(raw) and not valid:
                issues.append(_input_issue(row, 'INVALID_DC_COUNT', {column: raw}, 'Use a nonnegative integer DC count'))
        row['reference_coverage_source_days'] = as_number(source.get('RoC 1 full SEFI batch'))
        if not consistent and row['source_horizon_days'] is not None:
            issues.append(_input_issue(row, 'INCONSISTENT_SOURCE_HORIZON', row['source_horizon_days'], 'Resolve inconsistent total forecast / source daily demand ratios'))
            row['source_horizon_days'] = None
    return issues


def _input_issue(row: dict, rule: str, observed: object, action: str) -> dict:
    """Return a validation warning with exact source lineage and rejected input."""
    return dict(rule_id=rule, severity='warning', entity='fini', entity_key=row['fini_plant_key'], observed=observed, action=action, source_sheet=row['source_sheet'], source_row=row['source_row'])
