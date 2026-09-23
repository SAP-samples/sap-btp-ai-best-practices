import json
import time
from pathlib import Path

import pytest

from dox_client import FieldDefinition, SapDoxClient, ServiceKey
from dox_client.sap_dox_client import DoxApiError, _join_url


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None, content=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")
        self.content = content if content is not None else self.text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


class FakeSession:
    def __init__(self):
        self.responses = []
        self.requests = []

    def queue(self, method, path_suffix, response):
        self.responses.append((method.upper(), path_suffix, response))

    def request(self, method, url, **kwargs):
        method = method.upper()
        self.requests.append({"method": method, "url": url, **kwargs})
        for index, (expected_method, suffix, response) in enumerate(self.responses):
            if method == expected_method and url.endswith(suffix):
                self.responses.pop(index)
                return response
        raise AssertionError(f"Unexpected request: {method} {url}")

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


def service_key():
    return ServiceKey(
        token_base_url="https://auth.example",
        client_id="client-id",
        client_secret="client-secret",
        dox_base_url="https://dox.example",
        swagger_path="/document-information-extraction/v1/",
    )


def make_client(session):
    client = SapDoxClient(service_key(), session=session)
    client._access_token = "cached-token"
    client._token_expires_at_epoch = time.time() + 3600
    return client


def test_service_key_parses_uaa_url_and_joins_paths():
    parsed = ServiceKey.from_json(
        {
            "uaa": {
                "url": "https://auth.example/",
                "clientid": "id",
                "clientsecret": "secret",
            },
            "url": "https://dox.example/",
            "swagger": "/document-information-extraction/v1/",
        }
    )

    assert parsed.token_base_url == "https://auth.example/"
    assert _join_url(parsed.dox_base_url, parsed.swagger_path) == (
        "https://dox.example/document-information-extraction/v1/"
    )


def test_get_token_caches_access_token():
    session = FakeSession()
    session.queue(
        "POST",
        "/oauth/token",
        FakeResponse(200, {"access_token": "new-token", "expires_in": 3600}),
    )
    client = SapDoxClient(service_key(), session=session)

    assert client.get_token() == "new-token"
    assert client.get_token() == "new-token"
    assert len(session.requests) == 1
    assert session.requests[0]["data"] == {"grant_type": "client_credentials"}


def test_dox_api_error_exposes_sap_error_details():
    session = FakeSession()
    session.queue(
        "GET",
        "/document-information-extraction/v1/capabilities",
        FakeResponse(
            400,
            {"code": "E93", "message": "Required parameters not provided.", "details": "clientId"},
        ),
    )
    client = make_client(session)

    with pytest.raises(DoxApiError) as exc_info:
        client.get_capabilities()

    error = exc_info.value
    assert error.status_code == 400
    assert error.sap_code == "E93"
    assert error.sap_message == "Required parameters not provided."
    assert error.details == "clientId"
    assert "GET /capabilities failed with 400" in str(error)


def test_list_methods_normalize_official_response_wrappers():
    session = FakeSession()
    session.queue(
        "GET",
        "/document-information-extraction/v1/clients",
        FakeResponse(200, {"id": "tenant", "payload": [{"clientId": "c_00"}]}),
    )
    session.queue(
        "GET",
        "/document-information-extraction/v1/schemas",
        FakeResponse(200, {"schemas": [[{"id": "schema-1", "name": "Invoice"}]]}),
    )
    session.queue(
        "GET",
        "/document-information-extraction/v1/schemas/schema-1/versions",
        FakeResponse(200, {"schemas": [[{"id": "schema-1", "version": "1"}]]}),
    )
    client = make_client(session)

    assert client.list_clients(limit=10, offset=0) == [{"clientId": "c_00"}]
    assert client.list_schemas(client_id="c_00") == [{"id": "schema-1", "name": "Invoice"}]
    assert client.list_schema_versions("schema-1", client_id="c_00") == [
        {"id": "schema-1", "version": "1"}
    ]


def test_delete_schema_uses_official_bulk_endpoint():
    session = FakeSession()
    session.queue(
        "DELETE",
        "/document-information-extraction/v1/schemas",
        FakeResponse(200, {"message": "Schemas deleted successfully."}),
    )
    client = make_client(session)

    result = client.delete_schema(["schema-1", "schema-2"], client_id="c_00")

    assert result == {"message": "Schemas deleted successfully."}
    request = session.requests[-1]
    assert request["params"] == {"clientId": "c_00"}
    assert request["json"] == {"value": ["schema-1", "schema-2"]}


def test_schema_helpers_cover_lookup_version_update_and_configure():
    session = FakeSession()
    session.queue(
        "GET",
        "/document-information-extraction/v1/schemas",
        FakeResponse(200, {"schemas": [[{"id": "schema-1", "name": "Invoice"}]]}),
    )
    session.queue(
        "POST",
        "/document-information-extraction/v1/schemas/schema-1",
        FakeResponse(201, {"id": "schema-1", "version": "2"}),
    )
    session.queue(
        "PUT",
        "/document-information-extraction/v1/schemas/schema-1",
        FakeResponse(201, {"message": "Schema has been updated successfully."}),
    )
    session.queue(
        "PUT",
        "/document-information-extraction/v1/schemas/schema-1/versions/2",
        FakeResponse(201, {"message": "Schema has been updated successfully."}),
    )
    session.queue(
        "GET",
        "/document-information-extraction/v1/schemas/schema-1/versions/2",
        FakeResponse(200, {"id": "schema-1", "version": "2", "state": "active"}),
    )
    session.queue(
        "POST",
        "/document-information-extraction/v1/schemas/schema-1/versions/2/deactivate",
        FakeResponse(201, {"message": "Schema version deactivated successfully."}),
    )
    session.queue(
        "POST",
        "/document-information-extraction/v1/schemas/schema-1/versions/2/fields",
        FakeResponse(201, {"message": "Schema fields have been uploaded successfully."}),
    )
    session.queue(
        "POST",
        "/document-information-extraction/v1/schemas/schema-1/versions/2/activate",
        FakeResponse(201, {"message": "Schema version activated successfully."}),
    )
    client = make_client(session)

    assert client.get_schema_by_name("Invoice", client_id="c_00") == {
        "id": "schema-1",
        "name": "Invoice",
    }
    assert client.create_schema_version("schema-1", client_id="c_00") == {
        "id": "schema-1",
        "version": "2",
    }
    assert client.update_schema("schema-1", client_id="c_00", name="New Name") == {
        "message": "Schema has been updated successfully."
    }
    assert client.update_schema_version(
        "schema-1", 2, client_id="c_00", schema_description="Better prompt"
    ) == {"message": "Schema has been updated successfully."}
    assert client.configure_schema_version(
        "schema-1",
        2,
        client_id="c_00",
        header_fields=[FieldDefinition(name="documentNumber")],
        line_item_fields=["netAmount"],
    ) == {"message": "Schema version activated successfully."}

    field_request = session.requests[-2]
    assert field_request["json"]["headerFields"][0]["name"] == "documentNumber"
    assert field_request["json"]["lineItemFields"][0]["name"] == "netAmount"
    assert field_request["json"]["lineItemFields"][0]["setup"]["type"] == "auto"


def test_upload_document_builds_schema_options(tmp_path):
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    session = FakeSession()
    session.queue(
        "POST",
        "/document-information-extraction/v1/document/jobs",
        FakeResponse(201, {"id": "job-1", "status": "PENDING"}),
    )
    client = make_client(session)

    result = client.upload_document(
        str(pdf),
        client_id="c_00",
        schema_id="schema-1",
        schema_version=2,
        document_type="invoice",
        received_date="2026-04-28",
        custom_label="April invoice",
        template_id="detect",
        candidate_template_ids=["template-1", "template-2"],
        enrichment={"sender": {"top": 3}},
    )

    assert result == {"id": "job-1", "status": "PENDING"}
    options_part = session.requests[-1]["files"]["options"]
    options = json.loads(options_part[1])
    assert options == {
        "clientId": "c_00",
        "schemaId": "schema-1",
        "schemaVersion": "2",
        "documentType": "invoice",
        "receivedDate": "2026-04-28",
        "customLabel": "April invoice",
        "templateId": "detect",
        "candidateTemplateIds": ["template-1", "template-2"],
        "enrichment": {"sender": {"top": 3}},
    }


def test_upload_document_builds_ad_hoc_extraction_options(tmp_path):
    pdf = tmp_path / "custom.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    session = FakeSession()
    session.queue(
        "POST",
        "/document-information-extraction/v1/document/jobs",
        FakeResponse(201, {"id": "job-2", "status": "PENDING"}),
    )
    client = make_client(session)

    client.upload_document(
        str(pdf),
        client_id="c_00",
        document_type="custom",
        header_fields=[FieldDefinition(name="contractNumber"), {"name": "documentDate"}],
        line_item_fields=["netAmount"],
    )

    options = json.loads(session.requests[-1]["files"]["options"][1])
    assert options == {
        "clientId": "c_00",
        "documentType": "custom",
        "extraction": {
            "headerFields": ["contractNumber", "documentDate"],
            "lineItemFields": ["netAmount"],
        },
    }


def test_upload_document_rejects_conflicting_schema_and_ad_hoc_fields(tmp_path):
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = make_client(FakeSession())

    with pytest.raises(ValueError, match="schema_id/schema_name cannot be combined"):
        client.upload_document(
            str(pdf),
            client_id="c_00",
            schema_id="schema-1",
            header_fields=["documentNumber"],
        )

    with pytest.raises(ValueError, match="template_id requires schema_id or schema_name"):
        client.upload_document(str(pdf), client_id="c_00", template_id="detect", document_type="invoice")


def test_document_helpers_get_list_search_delete_and_wait():
    session = FakeSession()
    session.queue(
        "GET",
        "/document-information-extraction/v1/document/jobs/job-1",
        FakeResponse(200, {"id": "job-1", "status": "DONE"}),
    )
    session.queue(
        "GET",
        "/document-information-extraction/v1/document/jobs",
        FakeResponse(200, {"results": [[{"id": "job-1"}]]}),
    )
    session.queue(
        "POST",
        "/document-information-extraction/v1/document/catalog",
        FakeResponse(200, {"results": [[{"id": "job-2"}]], "totalDocumentCount": 1}),
    )
    session.queue(
        "DELETE",
        "/document-information-extraction/v1/document/jobs",
        FakeResponse(200, {"status": "DONE", "message": "Deleted"}),
    )
    session.queue(
        "GET",
        "/document-information-extraction/v1/document/jobs/job-3",
        FakeResponse(200, {"id": "job-3", "status": "PENDING"}),
    )
    session.queue(
        "GET",
        "/document-information-extraction/v1/document/jobs/job-3",
        FakeResponse(200, {"id": "job-3", "status": "DONE"}),
    )
    client = make_client(session)

    assert client.get_job("job-1", extracted_values=True, return_null_values=True)["status"] == "DONE"
    assert client.list_documents(client_id="c_00") == [{"id": "job-1"}]
    assert client.search_document_catalog(
        client_id="c_00",
        filter_query="status eq done",
        like_filter='fileName like "invoice"',
        limit=10,
        offset=2,
        order="created desc",
    )["results"] == [[{"id": "job-2"}]]
    assert client.delete_jobs(["job-1", "job-2"]) == {"status": "DONE", "message": "Deleted"}
    assert client.wait_for_result("job-3", timeout_seconds=5, poll_interval_seconds=0)["status"] == "DONE"

    get_job_request = session.requests[0]
    assert get_job_request["params"] == {"extractedValues": "true", "returnNullValues": "true"}
    catalog_options = json.loads(session.requests[2]["files"]["options"][1])
    assert catalog_options == {
        "clientId": "c_00",
        "filter": "status eq done",
        "likeFilter": 'fileName like "invoice"',
        "limit": 10,
        "offset": 2,
        "order": "created desc",
    }
    assert session.requests[3]["json"] == {"value": ["job-1", "job-2"]}
