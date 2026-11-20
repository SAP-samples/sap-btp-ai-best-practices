"""Opt-in isolated HANA verification for the rebuildable ECN RDF projection.

Run from the repository root without sourcing credentials manually::

    RUN_HANA_INTEGRATION_TESTS=1 PYTHONPATH=api \\
      .venv/bin/python -m pytest -q tests/integration/test_hana_rdf_projection.py -m hana_live

The normal application import loads local dotenv configuration. This test creates
only UUID-suffixed relational tables and a UUID-suffixed RDF graph, then removes
those exact resources in ``finally``. It never deploys the application.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import text

# Importing the application factory uses the same dotenv loading path as runtime.
from app.main import create_app
from app.ecn.errors import PersistenceStateError
from app.ecn.rdf import ProjectionState, project_ecn_json, root_iri
from app.ecn.rdf.service import RdfProjectionService
from app.persistence.hana import create_hana_engine
from app.persistence.rdf_repository import RdfRepository
from app.persistence.schema import current_schema, ensure_schema, table_exists
from app.persistence.table_contracts import (
    PersistenceTableNames,
    build_table_contracts,
    quote_identifier,
)
from app.persistence.upload_repository import UploadRepository


pytestmark = pytest.mark.hana_live


def _drop_isolated_graph(engine: object, graph_iri: str) -> None:
    """Drop exactly the test graph while suppressing a best-effort cleanup race."""
    connection = engine.raw_connection()  # type: ignore[attr-defined]
    cursor = connection.cursor()
    try:
        connection.setautocommit(True)
        cursor.callproc("SYS.SPARQL_EXECUTE", (f"DROP GRAPH <{graph_iri}>", "", "?", "?"))
        connection.commit()
    except Exception:  # noqa: BLE001 - cleanup must not mask the test outcome.
        try:
            connection.rollback()
        except Exception:  # noqa: BLE001 - no cleanup path remains.
            pass
    finally:
        cursor.close()
        try:
            connection.setautocommit(False)
        except Exception:  # noqa: BLE001 - closing is still required.
            pass
        connection.close()


def _drop_isolated_tables(engine: object, names: PersistenceTableNames) -> None:
    """Drop only this test's UUID-suffixed tables in foreign-key-safe order."""
    with engine.connect() as connection:  # type: ignore[attr-defined]
        schema = current_schema(connection)
        for contract in reversed(build_table_contracts(names)):
            if table_exists(connection, schema, contract.name):
                connection.execute(text(f"DROP TABLE {quote_identifier(contract.name)}"))
                connection.commit()


def _publication_with_graph_sources() -> object:
    """Build two committed ECNs with requester, arrays, and explicit null fields."""
    from tests.integration.test_hana_persistence import _make_upload_publication

    publication = _make_upload_publication(ecn_count=2)
    ecns = []
    for index, item in enumerate(publication.ecns):
        source_json = {
            "REQUESTER_NAME": "Ada" if index == 0 else "Grace",
            "CNAffectLinks": [
                {
                    "AffectedObjects": [
                        {
                            "Number": f"A-{index + 1}",
                            "Name": "Old hydraulic pump",
                            "Description": "Hydraulic pump assembly",
                            "ObjectType": "WTPart",
                        }
                    ]
                }
            ],
            "ImplementationPlan": [
                {
                    "ResultingLinks": [
                        {
                            "OemReplacesAffected": None,
                            "ResultingObjects": [
                                {
                                    "Number": f"R-{index + 1}",
                                    "Name": "New hydraulic pump",
                                    "Description": "Hydraulic pump replacement",
                                    "ObjectType": "WTPart",
                                }
                            ],
                        }
                    ]
                }
            ],
            "nested": [None, {"preserved": True}],
        }
        ecns.append(replace(item, ecn=replace(item.ecn, source_json=source_json)))
    return replace(publication, ecns=tuple(ecns))


@pytest.mark.skipif(
    os.getenv("RUN_HANA_INTEGRATION_TESTS") != "1",
    reason="set RUN_HANA_INTEGRATION_TESTS=1 for isolated RDF HANA verification",
)
def test_hana_rdf_projection_rebuilds_without_cross_ecn_leakage() -> None:
    """Exercise schema bootstrap, graph reads, verification failure, and scoped rebuilds."""
    assert callable(create_app)
    suffix = "_RDFT" + uuid4().hex[:10].upper()
    names = PersistenceTableNames().with_suffix(suffix)
    graph_iri = f"urn:ecn-impact:graph:rdf-test-{uuid4().hex[:16]}"
    engine = create_hana_engine()
    repository = RdfRepository(engine, names, graph_iri)
    service = RdfProjectionService(repository, clock=lambda: datetime.now(timezone.utc))
    try:
        ensure_schema(engine, names)
        ensure_schema(engine, names)
        repository.ensure_capabilities()

        publication = _publication_with_graph_sources()
        result = UploadRepository(engine, names).publish_upload(publication)
        assert result.created is True
        first_ecn_id, second_ecn_id = tuple(
            item.ecn.ecn_row_id for item in publication.ecns
        )
        assert repository.get_projection(first_ecn_id).status is ProjectionState.PENDING  # type: ignore[union-attr]

        first_current = service.project(first_ecn_id)
        second_current = service.project(second_ecn_id)
        assert first_current.status is ProjectionState.CURRENT
        assert second_current.status is ProjectionState.CURRENT

        fields = service.list_fields(first_ecn_id, "request", page=1)
        assert [(item.name, item.example_json_pointer) for item in fields.items] == [
            ("REQUESTER_NAME", "/REQUESTER_NAME")
        ]

        requester = service.find_facts(
            first_ecn_id, "Ada", "value", None, None, page=1
        )
        assert [(item.value, item.json_pointer) for item in requester.items] == [
            ("Ada", "/REQUESTER_NAME")
        ]

        with pytest.raises(ValueError, match="selected ECN"):
            service.expand_nodes(
                first_ecn_id,
                (str(root_iri(second_ecn_id)),),
                "outgoing",
                (),
                1,
                1,
            )
        second_source = repository.load_source(second_ecn_id)
        assert second_source is not None
        second_projection = project_ecn_json(
            second_ecn_id,
            second_source.artifact_id,
            second_source.source_json,
        )
        second_counts_before = repository.verify_graph_projection(second_projection).counts

        with patch.object(
            repository,
            "verify_graph_projection",
            side_effect=PersistenceStateError(
                "RDF_PROJECTION_VERIFICATION_FAILED",
                "The RDF projection could not be verified after rebuild.",
                status_code=500,
            ),
        ):
            failed = service.project(first_ecn_id)
        assert failed.status is ProjectionState.FAILED
        assert repository.load_source(first_ecn_id) is not None

        rebuilt = service.project(first_ecn_id)
        second_counts_after = repository.verify_graph_projection(second_projection).counts
        assert rebuilt.status is ProjectionState.CURRENT
        assert rebuilt.source_sha256 == first_current.source_sha256
        assert rebuilt.projection_version == first_current.projection_version
        assert rebuilt.verified_node_count == first_current.verified_node_count
        assert rebuilt.verified_triple_count == first_current.verified_triple_count
        assert second_counts_after == second_counts_before
    finally:
        _drop_isolated_graph(engine, graph_iri)
        _drop_isolated_tables(engine, names)
        engine.dispose()
