"""Excel and natural-language compatibility use the same strict matrix contract."""

from io import BytesIO
import pytest
from openpyxl import load_workbook
from app.workspace.profile_matrix import matrix_workbook, read_matrix_workbook
from app.workspace.matrix_description import expand_matrix_description


def test_excel_matrix_roundtrip_changed_volumes_and_invalid_cells():
    """Accept fewer/new volume axes and reject incomplete, asymmetric or formula cells."""
    rows = expand_matrix_description({'volumes': ['1', '2', '9'],
        'compatible_groups': [['1', '2']], 'default_status': 'N'})
    workbook = load_workbook(BytesIO(matrix_workbook(rows)))
    sheet = workbook['Volume compatibility']
    sheet.delete_cols(4); sheet.delete_rows(4)
    sheet.cell(1, 3, 3); sheet.cell(3, 1, 3)
    output = BytesIO(); workbook.save(output)
    parsed = read_matrix_workbook(output.getvalue())
    assert {r['volume_a'] for r in parsed} == {'1', '3'}
    assert len(parsed) == 4
    for invalid in (None, 'MAYBE', '=1+1', 'N'):
        sheet.cell(2, 3).value = invalid
        output = BytesIO(); workbook.save(output)
        with pytest.raises(ValueError):
            read_matrix_workbook(output.getvalue())


def test_matrix_http_preview_and_explicit_profile_save():
    """Downloads/uploads preview only; an explicit save persists the accepted matrix."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers.workspace import router
    from app.workspace.dependencies import get_service
    from app.workspace.repository import MemoryRepository
    from app.workspace.service import WorkspaceService
    service = WorkspaceService(MemoryRepository())
    app = FastAPI(); app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)
    rows = expand_matrix_description({'volumes': ['1', '2', '99'],
        'compatible_groups': [['1', '2']], 'default_status': 'N'})
    download = client.post('/api/plant-profiles/matrix-template', json={'matrix_rows': rows})
    assert download.status_code == 200
    preview = client.post('/api/plant-profiles/matrix-preview', files={'file': ('matrix.xlsx', download.content)})
    assert preview.status_code == 200 and preview.json()['volume_count'] == 3
    assert service.list_plant_profiles() == []
    body = {'name': 'Preview test', 'plant': 'P', 'matrix_rows': preview.json()['matrix_rows']}
    saved = client.post('/api/plant-profiles', json=body)
    assert saved.status_code == 200 and saved.json()['rules'] == []
    assert service.get_plant_profile(saved.json()['profile_id'])['matrix_rows'] == rows
    assert client.post('/api/plant-profiles/matrix-preview', files={'file': ('bad.xlsx', b'not an xlsx')}).status_code == 422
    help_text = client.get('/api/plant-profile-help').json()
    assert 'horizon_days' in help_text and 'plant-constraints' in help_text


def test_class_matrix_roundtrips_through_the_excel_template():
    """'<1' and '>15' headers survive download and upload in axis order."""
    from production_wheel.volume_classes import default_class_matrix
    rows = default_class_matrix()
    workbook = load_workbook(BytesIO(matrix_workbook(rows)))
    header = [cell.value for cell in workbook['Volume compatibility'][1]][1:]
    assert header == ['<1', 1, 2.5, 5, 10, 15, '>15']
    assert read_matrix_workbook(matrix_workbook(rows)) == rows
