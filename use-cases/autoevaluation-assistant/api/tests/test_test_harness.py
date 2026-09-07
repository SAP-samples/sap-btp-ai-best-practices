"""Smoke tests for the backend pytest harness."""

from fastapi.testclient import TestClient


def test_health_endpoint_reports_healthy(api_client: TestClient) -> None:
    """Verify the public health endpoint responds with a healthy status.

    Inputs:
        api_client: FastAPI test client fixture configured for the backend app.

    Outputs:
        None. The test asserts that ``GET /api/health`` returns HTTP 200 and a
        JSON body whose ``status`` value is ``healthy``.
    """
    response = api_client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
