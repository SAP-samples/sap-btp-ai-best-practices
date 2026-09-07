"""Tests for import API routes."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from app.routers import imports as import_routes
from app.services.framework_importer import ITALIAN_DIMENSION_FILES, load_framework_seed
from scripts import import_assessment_framework


def _multipart_files_from_paths(
    files: list[tuple[Path, str]],
) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Build typed multipart tuples for FastAPI test client calls.

    Inputs:
        files: Sequence of file path and field-name pairs.

    Outputs:
        list[tuple[str, tuple[str, bytes, str]]]: Multipart body entries.
    """

    return [
        (field, (path.name, path.read_bytes(), "application/octet-stream"))
        for path, field in files
    ]


def test_import_assessment_framework_dry_run_returns_counts_and_skips_hana_write(
    api_client: TestClient,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify assessment dry-run returns counts and does not call the HANA writer."""

    writer_calls: list[object] = []

    def fake_write_framework_to_hana(
        *args: object,
        **kwargs: object,
    ) -> None:
        """Record API-level write attempts to validate dry-run behavior."""

        writer_calls.append((args, kwargs))

    monkeypatch.setattr(
        import_routes.import_assessment_framework,
        "write_framework_to_hana",
        fake_write_framework_to_hana,
    )

    workbook = repo_root / "data" / "sanitized" / "assessment_framework.xlsx"
    explanations = repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"
    seed = load_framework_seed(workbook_path=workbook, explanations_path=explanations)

    response = api_client.post(
        "/api/import/assessment-framework",
        headers={"X-API-Key": "test-api-key"},
        files=_multipart_files_from_paths(
            [
                (workbook, "workbook"),
                (explanations, "explanations"),
            ],
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dimensions"] == len(import_assessment_framework.ordered_dimensions(seed))
    assert payload["questions"] == len(seed.questions)
    assert payload["write_completed"] is False
    assert writer_calls == []


def test_import_assessment_framework_write_calls_writer_with_file_metadata_and_translations(
    api_client: TestClient,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify assessment write mode invokes HANA writer with normalized inputs."""

    writer_calls: list[tuple[object, Path, Path, object]] = []

    def fake_write_framework_to_hana(
        seed: object,
        workbook_path: Path,
        explanations_path: Path,
        translations: object,
    ) -> None:
        writer_calls.append((seed, workbook_path, explanations_path, translations))

    monkeypatch.setattr(
        import_routes.import_assessment_framework,
        "write_framework_to_hana",
        fake_write_framework_to_hana,
    )

    workbook = repo_root / "data" / "sanitized" / "assessment_framework.xlsx"
    explanations = repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"
    italian_dir = repo_root / "data" / "sanitized" / "IT"
    italian_filenames = [
        file_name for file_name, _display_name in ITALIAN_DIMENSION_FILES.values()
    ]

    response = api_client.post(
        "/api/import/assessment-framework",
        headers={"X-API-Key": "test-api-key"},
        data={"write": "true"},
        files=_multipart_files_from_paths(
            [
                (workbook, "workbook"),
                (explanations, "explanations"),
            ]
            + [
                (italian_dir / file_name, "italian_csvs")
                for file_name in sorted(italian_filenames)
            ],
        ),
    )

    assert response.status_code == 200
    assert writer_calls
    _, written_workbook, written_explanations, written_translations = writer_calls[0]
    assert Path(written_workbook).name == workbook.name
    assert Path(written_explanations).name == explanations.name
    assert written_translations is not None
    response_payload = response.json()
    assert response_payload["write_completed"] is True
    assert response_payload["italian_questions"] is not None
    assert response_payload["italian_answer_items"] is not None


def test_import_assessment_framework_rejects_incomplete_italian_upload_set(
    api_client: TestClient,
    repo_root: Path,
) -> None:
    """Verify incomplete Italian CSV uploads return a validation error."""

    workbook = repo_root / "data" / "sanitized" / "assessment_framework.xlsx"
    explanations = repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"
    italian_dir = repo_root / "data" / "sanitized" / "IT"
    italian_files = sorted(
        file_name for file_name, _ in ITALIAN_DIMENSION_FILES.values()
    )[:6]

    response = api_client.post(
        "/api/import/assessment-framework",
        headers={"X-API-Key": "test-api-key"},
        files=_multipart_files_from_paths(
            [
                (workbook, "workbook"),
                (explanations, "explanations"),
            ]
            + [(italian_dir / file_name, "italian_csvs") for file_name in italian_files]
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == (
        "Italian CSV uploads must exactly match the expected files."
    )


def test_import_assessment_framework_invalid_workbook_returns_400(
    api_client: TestClient,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Validate malformed workbook files are surfaced as import validation errors."""

    workbook = tmp_path / "bad_workbook.xlsx"
    wb = Workbook()
    wb.save(workbook)

    explanations = repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"
    response = api_client.post(
        "/api/import/assessment-framework",
        headers={"X-API-Key": "test-api-key"},
        files=_multipart_files_from_paths(
            [
                (workbook, "workbook"),
                (explanations, "explanations"),
            ],
        ),
    )

    assert response.status_code == 400
    assert "Invalid assessment workbook content" in str(response.json()["detail"])


def test_import_joule_knowledge_dry_run_returns_counts_only(
    api_client: TestClient,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Joule dry-run path returns summary without HANA writes."""

    writer_calls: list[object] = []

    def fake_write_joule(
        *args: object,
        **kwargs: object,
    ) -> None:
        """Record API-level write attempts to validate dry-run behavior."""

        writer_calls.append((args, kwargs))

    monkeypatch.setattr(
        import_routes.joule_import_script,
        "write_joule_knowledge_to_hana",
        fake_write_joule,
    )

    glossary = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_glossary.xlsx"
    )
    explanations = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_explanations.xlsx"
    )

    response = api_client.post(
        "/api/import/joule-knowledge",
        headers={"X-API-Key": "test-api-key"},
        files=_multipart_files_from_paths(
            [
                (glossary, "glossary_workbook"),
                (explanations, "explanations_workbook"),
            ],
        ),
    )

    assert response.status_code == 200
    assert response.json()["write_completed"] is False
    assert writer_calls == []


def test_import_joule_knowledge_write_uses_show_progress_false(
    api_client: TestClient,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify write mode passes controlled embedding parameters to writer helpers."""

    build_calls: list[tuple[int, bool, str]] = []
    write_calls: list[dict[str, object]] = []

    def fake_build(
        seed: object,
        embedding_client: object,
        batch_size: int,
        show_progress: bool,
    ) -> dict[str, tuple[list[float], list[float]]]:
        build_calls.append((batch_size, show_progress, embedding_client.__class__.__name__))
        return {}

    def fake_write(
        glossary_workbook: Path,
        explanations_workbook: Path,
        embedding_model: str,
        batch_size: int,
        question_embeddings: dict[str, tuple[list[float], list[float]]],
        seed: object,
        show_progress: bool,
    ) -> None:
        write_calls.append(
            {
                "glossary_workbook": glossary_workbook,
                "explanations_workbook": explanations_workbook,
                "embedding_model": embedding_model,
                "batch_size": batch_size,
                "show_progress": show_progress,
                "embeddings_empty": len(question_embeddings) == 0,
            }
        )

    monkeypatch.setattr(import_routes, "build_question_embedding_rows", fake_build)
    monkeypatch.setattr(
        import_routes.joule_import_script,
        "write_joule_knowledge_to_hana",
        fake_write,
    )

    glossary = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_glossary.xlsx"
    )
    explanations = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_explanations.xlsx"
    )

    response = api_client.post(
        "/api/import/joule-knowledge",
        headers={"X-API-Key": "test-api-key"},
        data={
            "write": "true",
            "batch_size": "8",
            "embedding_model": "text-embedding-4-large",
        },
        files=_multipart_files_from_paths(
            [
                (glossary, "glossary_workbook"),
                (explanations, "explanations_workbook"),
            ],
        ),
    )

    assert response.status_code == 200
    assert build_calls == [(8, False, "GenAiHubEmbeddingClient")]
    assert write_calls and write_calls[0]["show_progress"] is False
    assert write_calls[0]["embedding_model"] == "text-embedding-4-large"
    assert write_calls[0]["batch_size"] == 8
    assert response.json()["write_completed"] is True


def test_import_joule_knowledge_invalid_workbook_returns_400(
    api_client: TestClient,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Validate missing required sheets in uploaded Joule workbooks returns 400."""

    glossary = tmp_path / "invalid_workbook.xlsx"
    wb = Workbook()
    wb.save(glossary)

    explanations = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_explanations.xlsx"
    )
    response = api_client.post(
        "/api/import/joule-knowledge",
        headers={"X-API-Key": "test-api-key"},
        files=_multipart_files_from_paths(
            [
                (glossary, "glossary_workbook"),
                (explanations, "explanations_workbook"),
            ],
        ),
    )

    assert response.status_code == 400
    assert "Missing required glossary sheet" in str(response.json()["detail"])


def test_import_joule_knowledge_write_error_returns_500(
    api_client: TestClient,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify writer exceptions are mapped to HTTP 500 while validation remains decoupled."""

    def fake_build(
        seed: object,
        embedding_client: object,
        batch_size: int,
        show_progress: bool,
    ) -> dict[str, tuple[list[float], list[float]]]:
        return {}

    def fake_write(
        glossary_workbook: Path,
        explanations_workbook: Path,
        embedding_model: str,
        batch_size: int,
        question_embeddings: dict[str, object],
        seed: object,
        show_progress: bool,
    ) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(import_routes, "build_question_embedding_rows", fake_build)
    monkeypatch.setattr(
        import_routes.joule_import_script,
        "write_joule_knowledge_to_hana",
        fake_write,
    )

    glossary = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_glossary.xlsx"
    )
    explanations = (
        repo_root
        / "data"
        / "sanitized"
        / "assessment_explanations.xlsx"
    )

    response = api_client.post(
        "/api/import/joule-knowledge",
        headers={"X-API-Key": "test-api-key"},
        data={"write": "true"},
        files=_multipart_files_from_paths(
            [
                (glossary, "glossary_workbook"),
                (explanations, "explanations_workbook"),
            ],
        ),
    )

    assert response.status_code == 500
    assert "database unavailable" in str(response.json()["detail"])
