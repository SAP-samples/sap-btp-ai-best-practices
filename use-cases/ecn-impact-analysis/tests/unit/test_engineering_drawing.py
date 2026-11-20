"""Exact packaged drawing availability, ECN membership and authenticated delivery."""

from app.ecn.evidence.snapshot import build_snapshot
from app.ecn.services.automated_manufacturing_feasibility import FixtureDrawingResolver
from app.ecn.services.snapshots import SnapshotService
from tests.fixtures.builders import make_normalized_ecn, make_source_object
from tests.unit.test_ecn_api import ECN_ID, _client
from tests.fixtures.persistence import RecordingEngine
import pytest


def drawing_service(numbers, resolver=None):
    """Use the real snapshot builder and material reader without a database."""
    source = make_normalized_ecn(
        affected_objects=tuple(make_source_object(number=number) for number in numbers),
        resulting_objects=(),
    )
    service = object.__new__(SnapshotService)
    service.get = lambda _: build_snapshot(source, (), ())
    service.drawings = resolver or FixtureDrawingResolver()
    service.manufacturing_feasibility = None
    service.engine = RecordingEngine()
    service._active_price_overlay_rows = lambda *_, **kwargs: []
    return service


@pytest.mark.parametrize("number", ["QKX5632660", "QKX5525860", "QKX3896510", "QKX0971490"])
def test_exact_sample_is_advertised_and_served_with_authentication(monkeypatch, number):
    """The material's advertised sample and authenticated bytes are the same file."""
    client, _ = _client(monkeypatch)
    client.services.snapshot_service = drawing_service([number])
    fixture, content = FixtureDrawingResolver().resolve(number)
    with client:
        detail = client.get(f"/api/ecns/{ECN_ID}/object", params={"number": number}, headers={"X-API-Key": "test-api-key"})
        assert detail.status_code == 200
        assert detail.json()["engineering_drawing"] == {"file_name": fixture.file_name, "media_type": "application/pdf"}
        assert client.get(f"/api/ecns/{ECN_ID}/drawing", params={"number": number}).status_code == 403
        drawing = client.get(f"/api/ecns/{ECN_ID}/drawing", params={"number": number}, headers={"X-API-Key": "test-api-key"})
    assert drawing.status_code == 200
    assert drawing.content == content
    assert drawing.content.startswith(b"%PDF")
    assert drawing.headers["content-type"] == "application/pdf"
    assert fixture.file_name in drawing.headers["content-disposition"]
    assert drawing.headers["cache-control"] == "private, no-store"


def test_missing_unmatched_and_nonmember_drawings_stay_unavailable(monkeypatch, tmp_path):
    """No guessing, arbitrary filename access, or access outside the ECN membership."""
    client, _ = _client(monkeypatch)
    client.services.snapshot_service = drawing_service(["QKX5525860", "OTHER"], FixtureDrawingResolver(tmp_path))
    with client:
        for number in ("QKX5525860", "OTHER"):
            detail = client.get(f"/api/ecns/{ECN_ID}/object", params={"number": number}, headers={"X-API-Key": "test-api-key"})
            assert detail.json()["engineering_drawing"] is None
        for number, code in (("QKX5525860", "DRAWING_NOT_FOUND"), ("OTHER", "DRAWING_NOT_FOUND"),
                             ("QKX5632660", "OBJECT_NOT_FOUND"), ("../drawings/QKX5632660_2_AFS.pdf", "OBJECT_NOT_FOUND")):
            response = client.get(f"/api/ecns/{ECN_ID}/drawing", params={"number": number}, headers={"X-API-Key": "test-api-key"})
            assert response.status_code == 404
            assert response.json()["code"] == code
