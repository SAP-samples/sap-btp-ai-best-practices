"""Material reads keep operational data fresh while sharing one connection."""

import json

import pytest

from app.persistence.manufacturing_feasibility_repository import ManufacturingFeasibilityRepository
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository
from app.persistence.table_contracts import PersistenceTableNames
from app.ecn.errors import PersistenceNotFoundError
from tests.fixtures.persistence import _MappingResult
from tests.fixtures.persistence import RecordingEngine
from tests.unit.test_engineering_drawing import drawing_service
from tests.unit.test_reference_dataset_repository import _row


@pytest.fixture(autouse=True)
def scalar_result_support(monkeypatch):
    """Expose SQLAlchemy's scalar accessor on the existing recording result seam."""
    monkeypatch.setattr(_MappingResult, "scalar", _MappingResult.scalar_one, raising=False)


def service_with_repositories():
    """Use the real material reader and repositories over queued DB results."""
    service = drawing_service(["QKX5525860"])
    del service._active_price_overlay_rows
    service.tables = PersistenceTableNames()
    service.references = ReferenceDatasetRepository(service.engine)
    service.manufacturing_feasibility = ManufacturingFeasibilityRepository(service.engine)
    snapshot = service.get("id")
    snapshot["objects"][0]["reference_records"] = [{
        "dataset_type": "MATERIAL_MASTER", "plant_code": "PK01",
        "procurement_type_label": "External procurement",
    }]
    snapshot["datasets"] = [{
        "dataset_type": "PURCHASE_INFO_RECORDS", "comparison": False,
        "model_range": "Q900V", "data_group": "customer", "environment": "QA",
    }]
    service.get = lambda _: snapshot
    return service


def publication(identifier, order, *, environment="QA"):
    """Return a price publication with a complete activation boundary."""
    row = _row(identifier, "PURCHASE_INFO_PRICES", order, None)
    row["CONTEXT_JSON"] = json.dumps({
        "model_range": "Q900V", "data_group": "customer", "environment": environment,
    })
    return row


def test_prices_and_report_share_one_connection_and_each_open_reads_current_data():
    """Changing a publication/report is reflected without altering the snapshot."""
    service = service_with_repositories()
    for version in (1, 2):
        identifier = f"00000000-0000-4000-8000-00000000030{version}"
        service.engine.queue_mappings([
            publication("00000000-0000-4000-8000-000000000399", version + 10, environment="PRODUCTION"),
            publication(identifier, version),
        ])
        service.engine.queue_mappings([{
            "MATERIAL_NUMBER": "QKX5525860", "PLANT_CODE": "PK01", "NET_PRICE": str(version),
            "CURRENCY": "EUR", "PRICE_PER": "1", "ORDER_PRICE_UNIT": "EA",
        }])
        service.engine.queue_scalar(json.dumps({"version": version}))
        result = service.object_details("id", "QKX5525860")
        assert result["manufacturing_feasibility"] == {"version": version}
        assert result["material_brief"]["sourcing"][0]["source_purchase_price"]["net_price"] == str(version)
        assert service.engine.connect_count == version
        price_sql, price_params = service.engine.executions[-2]
        assert price_params == {"dataset_id": identifier, "material_number": "QKX5525860"}
        assert '"MATERIAL_NUMBER" = :material_number' in price_sql
        assert service.engine.executions[-1][1]["state"] == "COMPLETED"
    assert len(service.engine.executions) == 6


def test_missing_overlay_and_report_do_not_invent_operational_data():
    """The latest report is still read when no same-scope price is available."""
    service = service_with_repositories()
    service.engine.queue_mappings([])
    service.engine.queue_scalar(None)
    result = service.object_details("id", "QKX5525860")
    assert result["manufacturing_feasibility"] is None
    assert service.engine.connect_count == 1
    assert len(service.engine.executions) == 2


def test_nonmember_lookup_does_not_open_an_operational_connection():
    """Membership remains the boundary before reading prices or drawing reports."""
    service = service_with_repositories()
    with pytest.raises(PersistenceNotFoundError) as error:
        service.object_details("id", "OTHER")
    assert error.value.code == "OBJECT_NOT_FOUND"
    assert service.engine.connect_count == 0


def test_owned_connection_remains_open_and_standalone_report_read_still_works():
    """Repositories can participate in a short read session or own their connection."""
    engine = RecordingEngine()
    references = ReferenceDatasetRepository(engine)
    reports = ManufacturingFeasibilityRepository(engine)
    engine.queue_mappings([])
    engine.queue_scalar('{"version": 1}')
    with engine.connect() as connection:
        assert references.list(connection=connection) == ()
        assert reports.get_completed("id", "part", connection=connection) == {"version": 1}
        # A further read on this connection remains usable.
        engine.queue_scalar(1)
        assert connection.execute("SELECT 1").scalar() == 1
    assert engine.connect_count == 1
    engine.queue_scalar('{"version": 2}')
    assert reports.get_completed("id", "part") == {"version": 2}
    assert engine.connect_count == 2


def test_report_read_failure_does_not_return_a_partial_material():
    """A failed report read propagates rather than presenting incomplete success."""
    service = service_with_repositories()
    service.engine.queue_mappings([])
    service.engine.queue_exception(RuntimeError("synthetic read failure"))
    with pytest.raises(RuntimeError, match="synthetic read failure"):
        service.object_details("id", "QKX5525860")
    assert service.engine.connect_count == 1
