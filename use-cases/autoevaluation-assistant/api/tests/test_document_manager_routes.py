"""Tests for Document Manager API routes."""

from collections.abc import Iterator

import pytest

from app.main import app
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


@pytest.fixture
def document_repository() -> InMemoryAiReviewRepository:
    """Create an isolated repository for Document Manager route tests.

    Inputs:
        None.

    Outputs:
        InMemoryAiReviewRepository: Repository used by dependency overrides.
    """
    return InMemoryAiReviewRepository()


@pytest.fixture(autouse=True)
def override_repository(
    document_repository: InMemoryAiReviewRepository,
) -> Iterator[None]:
    """Use the in-memory repository for API route tests.

    Inputs:
        document_repository: Fresh repository fixture.

    Outputs:
        Iterator[None]: Yields while FastAPI dependency overrides are active.
    """
    app.dependency_overrides[get_ai_review_repository] = lambda: document_repository
    yield
    app.dependency_overrides.pop(get_ai_review_repository, None)


def test_upload_documents_creates_ingestion_job(api_client) -> None:
    """Verify multipart uploads create a corpus ingestion job."""
    response = api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={"files": ("strategy.pdf", b"%PDF strategy evidence", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["document_count"] == 1


def test_upload_documents_accepts_eml(api_client) -> None:
    """Verify EML files can enter the Document Manager corpus."""
    response = api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={
            "files": (
                "governance.eml",
                b"Subject: Governance\r\n\r\nBoard review evidence.",
                "message/rfc822",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["document_count"] == 1


def test_list_documents_returns_uploaded_metadata(api_client) -> None:
    """Verify list route returns document metadata without blob content."""
    api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={"files": ("strategy.pdf", b"%PDF strategy evidence", "application/pdf")},
    )

    response = api_client.get(
        "/api/documents?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_document_count"] == 1
    assert payload["documents"][0]["file_name"] == "strategy.pdf"
    assert "content" not in payload["documents"][0]


def test_get_ingestion_job_status_returns_progress(
    api_client,
    document_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify upload job polling returns the repository progress payload."""
    upload = api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={"files": ("strategy.pdf", b"%PDF strategy evidence", "application/pdf")},
    )
    job_id = upload.json()["job_id"]

    response = api_client.get(
        f"/api/documents/ingestion-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert response.json()["status"] == "pending"
    assert document_repository.document_ingestion_jobs[job_id]["document_count"] == 1


def test_download_document_returns_original_blob(
    api_client,
    document_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify downloads stream the original document bytes."""
    upload = api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={"files": ("strategy.pdf", b"%PDF strategy evidence", "application/pdf")},
    )
    job_id = upload.json()["job_id"]
    document_id = document_repository.document_ingestion_job_documents[job_id][0]

    response = api_client.get(
        f"/api/documents/{document_id}/download?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.content == b"%PDF strategy evidence"
    assert response.headers["content-type"].startswith("application/pdf")


def test_delete_document_removes_it_from_list(
    api_client,
    document_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify delete route clears document metadata and chunks."""
    upload = api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={"files": ("strategy.pdf", b"%PDF strategy evidence", "application/pdf")},
    )
    document_id = document_repository.document_ingestion_job_documents[
        upload.json()["job_id"]
    ][0]

    response = api_client.delete(
        f"/api/documents/{document_id}?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )
    list_response = api_client.get(
        "/api/documents?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == {"document_id": document_id, "deleted": True}
    assert list_response.json()["documents"] == []


def test_upload_documents_rejects_images(api_client) -> None:
    """Verify only text-extractable files enter the corpus."""
    response = api_client.post(
        "/api/documents",
        headers={"X-API-Key": "test-api-key"},
        data={"assessment_id": "assessment-1"},
        files={"files": ("photo.png", b"png", "image/png")},
    )

    assert response.status_code == 400
    assert "PDF, DOCX, XLSX, XLSM, and EML" in response.json()["detail"]


def test_upload_admin_documents_creates_global_ingestion_job(api_client) -> None:
    """Verify admin uploads create a global Joule RAG ingestion job."""
    response = api_client.post(
        "/api/admin-documents",
        headers={"X-API-Key": "test-api-key"},
        files={"files": ("policy.pdf", b"%PDF admin policy", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assessment_id"] == "admin"
    assert payload["status"] == "pending"
    assert payload["document_count"] == 1


def test_list_admin_documents_returns_uploaded_metadata(api_client) -> None:
    """Verify admin document listing is global and omits blob content."""
    api_client.post(
        "/api/admin-documents",
        headers={"X-API-Key": "test-api-key"},
        files={"files": ("policy.pdf", b"%PDF admin policy", "application/pdf")},
    )

    response = api_client.get(
        "/api/admin-documents",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assessment_id"] == "admin"
    assert payload["total_document_count"] == 1
    assert payload["documents"][0]["file_name"] == "policy.pdf"
    assert "content" not in payload["documents"][0]


def test_admin_document_job_status_download_and_delete(
    api_client,
    document_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify admin document polling, download, and deletion routes."""
    upload = api_client.post(
        "/api/admin-documents",
        headers={"X-API-Key": "test-api-key"},
        files={"files": ("policy.pdf", b"%PDF admin policy", "application/pdf")},
    )
    job_id = upload.json()["job_id"]
    document_id = document_repository.admin_document_ingestion_job_documents[job_id][0]

    status_response = api_client.get(
        f"/api/admin-documents/ingestion-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )
    download_response = api_client.get(
        f"/api/admin-documents/{document_id}/download",
        headers={"X-API-Key": "test-api-key"},
    )
    delete_response = api_client.delete(
        f"/api/admin-documents/{document_id}",
        headers={"X-API-Key": "test-api-key"},
    )
    list_response = api_client.get(
        "/api/admin-documents",
        headers={"X-API-Key": "test-api-key"},
    )

    assert status_response.status_code == 200
    assert status_response.json()["job_id"] == job_id
    assert download_response.status_code == 200
    assert download_response.content == b"%PDF admin policy"
    assert download_response.headers["content-type"].startswith("application/pdf")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"document_id": document_id, "deleted": True}
    assert list_response.json()["documents"] == []


def test_upload_admin_documents_rejects_images(api_client) -> None:
    """Verify admin uploads accept only text-extractable documents."""
    response = api_client.post(
        "/api/admin-documents",
        headers={"X-API-Key": "test-api-key"},
        files={"files": ("photo.png", b"png", "image/png")},
    )

    assert response.status_code == 400
    assert "PDF, DOCX, XLSX, XLSM, and EML" in response.json()["detail"]
