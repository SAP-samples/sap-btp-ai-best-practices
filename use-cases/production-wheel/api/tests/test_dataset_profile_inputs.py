"""Check optional source inputs and formula-backed PV defaults at import boundary."""

import os
from pathlib import Path

import pytest
from production_wheel.extraction.common import TableData
from production_wheel.extraction.profile_inputs import enrich_profile_inputs, parse_xyz


@pytest.mark.parametrize('raw,counts,status', [
    ('1 -  - ', (1, 0, 0), 'valid'), (' - 2 - 3', (0, 2, 3), 'valid'),
    (None, (None, None, None), 'missing'), ('', (None, None, None), 'missing'),
    ('1-2', (None, None, None), 'invalid'), ('1-2.5-3', (None, None, None), 'invalid'),
    ('-1-2-3', (None, None, None), 'invalid'), ('#REF!', (None, None, None), 'invalid'),
])
def test_positional_xyz(raw, counts, status):
    """Preserve three positions, distinguishing blank source from blank slots."""
    assert parse_xyz(raw) == (counts, status)


def test_q4_formula_defaults_and_source_horizon():
    """Import the supplied Q4 source without manufacturing historical assignments.

    Needs the private Q4 template workbook; set ``Q4_TEMPLATE_WORKBOOK`` to its path.
    """
    from production_wheel.extraction.datasets import extract_workbooks
    source = os.environ.get('Q4_TEMPLATE_WORKBOOK')
    if not source:
        pytest.skip('Q4_TEMPLATE_WORKBOOK is not set')
    path = Path(source)
    result = extract_workbooks(path)
    rows = result['tables']['fini_master']
    assert result['metadata']['settings']['demand_days'] == 80
    assert result['metadata']['settings']['productive_weeks'] == 16
    assert len(rows) == 967
    assert sum(row['model_status'] == 'modeled' for row in rows) == 770
    assert sum(row['optimized_pv_model_status'] == 'modeled' for row in rows) == 770
    assert sum(row['model_status'] == 'excluded' for row in rows) == 29
    assert sum(row['model_status'] == 'out_of_scope' for row in rows) == 168
    assert all(row['source_horizon_days'] == pytest.approx(80) for row in rows)
    assert all(row['forecast_litres_period'] == row['forecast_litres_12m'] for row in rows)
    assert all(row['selected_pv_source'] is None and not row['baseline_assigned'] for row in rows)
    assert any(row['fixed_pv_source'] == 'workbook_default' for row in rows)


@pytest.mark.parametrize('selection,formula,lot,expected', [
    (None, True, 1000, 'workbook_default'),
    ('PV - 99', True, 1000, 'explicit'),
    (None, False, 1000, 'missing'),
    (None, True, 0, 'missing'),
])
def test_defaults_require_source_formula_and_valid_catalog(selection, formula, lot, expected):
    """Never default explicit invalid selections or rows without positive PV evidence."""
    from production_wheel.extraction.profile_inputs import _DEFAULT_FORMULA
    record = {'_source_row': 2, 'Selected PV': selection, 'XYZ - # DCs': '1-two-3',
              'Primary DC count': -1, 'Secondary DC count': 2,
              'Sales scenario': 'Regional', 'Sales network scenario': 'Direct',
              'RoC 1 full SEFI batch': 12}
    row = {'source_row': 2, 'source_sheet': 'Inputs', 'plant': 'TEST', 'sefi': '200',
           'fini_plant_key': '100/TEST', 'fixed_pv': selection, 'baseline_assigned': 0}
    main = TableData('main', 'Inputs', 'A1:A2', (record,))
    formula_table = TableData('main', 'Inputs', 'A1:A2', ({'_source_row': 2,
        'SEFI litres lot size of selected PV': _DEFAULT_FORMULA if formula else '=IFNA(1,"PV - 1")'},))
    issues = enrich_profile_inputs([row], main, formula_table, [
        {'plant': 'TEST', 'sefi': '200', 'production_version': 'PV - 1', 'lot_size_litres': lot}])
    assert row['fixed_pv_source'] == expected
    assert row['selected_pv_source'] == selection
    assert row['baseline_assigned'] == 0
    assert row['xyz_raw'] == '1-two-3' and row['xyz_status'] == 'invalid'
    assert row['primary_dc_count_raw'] == -1 and row['primary_dc_count'] is None
    assert row['secondary_dc_count'] == 2
    assert row['sales_scenario'] == 'Regional'
    assert row['sales_network_scenario'] == 'Direct'
    assert row['reference_coverage_source_days'] == 12
    assert {issue['rule_id'] for issue in issues} == {'INVALID_XYZ_COUNTS', 'INVALID_DC_COUNT'}
    assert all(issue['source_row'] == 2 for issue in issues)


def test_inconsistent_horizons_remain_unknown():
    """Reject conflicting implied source calendars rather than silently choosing one."""
    records = tuple({'_source_row': number, 'Total FINI forecast (litres)': total,
                     'Avg daily demand': 10} for number, total in ((2, 800), (3, 2500)))
    main = TableData('main', 'Inputs', 'A1:A3', records)
    rows = [dict(source_row=record['_source_row'], source_sheet='Inputs', plant='TEST',
                 sefi='200', fini_plant_key=str(record['_source_row'])) for record in records]
    issues = enrich_profile_inputs(rows, main, main, [])
    assert all(row['source_horizon_days'] is None for row in rows)
    assert len(issues) == 2
    assert all(issue['rule_id'] == 'INCONSISTENT_SOURCE_HORIZON' for issue in issues)


@pytest.mark.parametrize('catalog', [
    [{'plant': 'OTHER', 'sefi': '200', 'production_version': 'PV - 1', 'lot_size_litres': 1000}],
    [{'plant': 'TEST', 'sefi': '200', 'production_version': 'PV - 1', 'lot_size_litres': size} for size in (1000, 2000)],
])
def test_default_rejects_cross_plant_or_ambiguous_lot(catalog):
    """A matching formula cannot supply a catalog entry from another plant or conflict."""
    from production_wheel.extraction.profile_inputs import _DEFAULT_FORMULA
    row = dict(source_row=2, source_sheet='Inputs', plant='TEST', sefi='200', fini_plant_key='100/TEST')
    main = TableData('main', 'Inputs', 'A1:A2', ({'_source_row': 2,
        'SEFI litres lot size of selected PV': _DEFAULT_FORMULA},))
    enrich_profile_inputs([row], main, main, catalog)
    assert row['fixed_pv_source'] == 'missing'
    assert row.get('fixed_pv') is None
