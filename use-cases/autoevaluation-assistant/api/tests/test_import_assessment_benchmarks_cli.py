"""Tests for the dry-run-first assessment benchmark import CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from app.services.benchmark_import.models import (
    BenchmarkDataset,
    BenchmarkValidationSummary,
    BenchmarkWriteResult,
)
from scripts import import_assessment_benchmarks


def _dataset() -> BenchmarkDataset:
    """Return a minimal successful parser result for CLI isolation tests.

    Inputs:
        None.

    Outputs:
        BenchmarkDataset: Empty entity lists plus a representative summary.
    """

    return BenchmarkDataset(
        workbook_bytes=b"workbook",
        companies=[],
        submissions=[],
        responses=[],
        scores=[],
        summary=BenchmarkValidationSummary(
            import_id="new-import",
            source_filename="benchmark.xlsx",
            source_sha256="abc123",
            scoring_version="assessment-v1",
            row_count=3306,
            company_count=6,
            questionnaire_count=6,
            question_count=38,
            accepted_count=3306,
            rejected_count=0,
            success=True,
            status="validated",
        ),
    )


def test_cli_defaults_to_dry_run_and_prints_useful_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the default path validates without creating a HANA engine.

    Inputs:
        monkeypatch: Pytest helper replacing parser, framework, and arguments.
        tmp_path: Temporary source workbook location.
        capsys: Captured terminal output fixture.

    Outputs:
        None. Text counts are printed and no database connection is attempted.
    """

    workbook = tmp_path / "benchmark.xlsx"
    workbook.write_bytes(b"workbook")
    parse_calls: list[tuple[bytes, str, list[Any]]] = []

    def fake_parse(content: bytes, filename: str, questions: list[Any]) -> BenchmarkDataset:
        """Record parser inputs and return a successful dataset.

        Inputs:
            content: Workbook bytes read by the CLI.
            filename: Original workbook filename.
            questions: Framework questions supplied by the CLI loader.

        Outputs:
            BenchmarkDataset: Stable successful test payload.
        """

        parse_calls.append((content, filename, questions))
        return _dataset()

    monkeypatch.setattr(import_assessment_benchmarks, "parse_benchmark_workbook", fake_parse)
    monkeypatch.setattr(
        import_assessment_benchmarks,
        "load_localized_framework_questions",
        lambda *_args, **_kwargs: ["framework-question"],
    )
    monkeypatch.setattr(
        import_assessment_benchmarks,
        "create_hana_engine",
        lambda: (_ for _ in ()).throw(AssertionError("dry run created HANA engine")),
    )
    monkeypatch.setattr(
        import_assessment_benchmarks.sys,
        "argv",
        ["import_assessment_benchmarks.py", "--workbook", str(workbook)],
    )

    import_assessment_benchmarks.main()

    output = capsys.readouterr().out
    assert "Status: validated" in output
    assert "Rows: 3,306 accepted / 0 rejected" in output
    assert "Companies: 6" in output
    assert "Dry run complete; no HANA writes performed." in output
    assert parse_calls == [(b"workbook", "benchmark.xlsx", ["framework-question"])]


def test_cli_json_output_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify ``--json`` emits only the JSON-safe validation summary.

    Inputs:
        monkeypatch: Pytest helper replacing pure parsing and CLI arguments.
        tmp_path: Temporary source workbook location.
        capsys: Captured terminal output fixture.

    Outputs:
        None. Parsed JSON retains counts, SHA, and dry-run status.
    """

    workbook = tmp_path / "benchmark.xlsx"
    workbook.write_bytes(b"workbook")
    monkeypatch.setattr(
        import_assessment_benchmarks,
        "parse_benchmark_workbook",
        lambda *_args, **_kwargs: _dataset(),
    )
    monkeypatch.setattr(
        import_assessment_benchmarks,
        "load_localized_framework_questions",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        import_assessment_benchmarks.sys,
        "argv",
        [
            "import_assessment_benchmarks.py",
            "--workbook",
            str(workbook),
            "--json",
        ],
    )

    import_assessment_benchmarks.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_sha256"] == "abc123"
    assert payload["accepted_count"] == 3306
    assert payload["write_completed"] is False


def test_cli_write_uses_transactional_repository_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify explicit writes use existing engine/session conventions safely.

    Inputs:
        monkeypatch: Pytest helper replacing parser, repository, and HANA factory.
        tmp_path: Temporary source workbook location.
        capsys: Captured terminal output fixture.

    Outputs:
        None. Writer receives the parsed dataset and all resources are closed.
    """

    class _Session:
        """Record whether the CLI closes its SQLAlchemy session."""

        def __init__(self) -> None:
            """Initialize an open fake session.

            Inputs:
                None.

            Outputs:
                None. ``closed`` starts false.
            """

            self.closed = False

        def close(self) -> None:
            """Mark the fake session closed.

            Inputs:
                None.

            Outputs:
                None. ``closed`` becomes true.
            """

            self.closed = True

    class _Engine:
        """Record whether the CLI disposes its SQLAlchemy engine."""

        def __init__(self) -> None:
            """Initialize an undisposed fake engine.

            Inputs:
                None.

            Outputs:
                None. ``disposed`` starts false.
            """

            self.disposed = False

        def dispose(self) -> None:
            """Mark the fake engine disposed.

            Inputs:
                None.

            Outputs:
                None. ``disposed`` becomes true.
            """

            self.disposed = True

    workbook = tmp_path / "benchmark.xlsx"
    workbook.write_bytes(b"workbook")
    session = _Session()
    engine = _Engine()
    write_calls: list[tuple[Any, BenchmarkDataset, int, bool]] = []

    def fake_write(
        supplied_session: Any,
        dataset: BenchmarkDataset,
        *,
        batch_size: int,
        replace_history: bool = False,
    ) -> BenchmarkWriteResult:
        """Record write inputs and return an active import result.

        Inputs:
            supplied_session: Session created by the CLI.
            dataset: Parsed benchmark dataset.
            batch_size: Requested bounded HANA batch size.
            replace_history: Whether inactive versions must be deleted.

        Outputs:
            BenchmarkWriteResult: Successful active-version result.
        """

        write_calls.append((supplied_session, dataset, batch_size, replace_history))
        return BenchmarkWriteResult(
            import_id="new-import",
            status="active",
            no_op=False,
        )

    monkeypatch.setattr(
        import_assessment_benchmarks,
        "parse_benchmark_workbook",
        lambda *_args, **_kwargs: _dataset(),
    )
    monkeypatch.setattr(
        import_assessment_benchmarks,
        "load_localized_framework_questions",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(import_assessment_benchmarks, "create_hana_engine", lambda: engine)
    monkeypatch.setattr(
        import_assessment_benchmarks,
        "sessionmaker",
        lambda *, bind: lambda: session,
    )
    monkeypatch.setattr(import_assessment_benchmarks, "write_benchmark_dataset", fake_write)
    monkeypatch.setattr(
        import_assessment_benchmarks.sys,
        "argv",
        [
            "import_assessment_benchmarks.py",
            "--workbook",
            str(workbook),
            "--write",
            "--replace-history",
            "--backup-confirmed",
            "--batch-size",
            "100",
        ],
    )

    import_assessment_benchmarks.main()

    assert write_calls == [(session, _dataset(), 100, True)]
    assert session.closed is True
    assert engine.disposed is True
    assert "Status: active" in capsys.readouterr().out


def test_history_replacement_requires_write_and_confirmed_backup(tmp_path: Path) -> None:
    """Verify destructive cleanup cannot run in dry-run or without backup consent.

    Inputs:
        tmp_path: Temporary directory containing a placeholder workbook.

    Outputs:
        None. Both missing safety gates are rejected before parsing or HANA access.
    """

    workbook = tmp_path / "benchmark.xlsx"
    workbook.write_bytes(b"workbook")
    parser = import_assessment_benchmarks.build_parser()

    dry_run = parser.parse_args(["--workbook", str(workbook), "--replace-history"])
    with pytest.raises(ValueError, match="write-only"):
        import_assessment_benchmarks.run_import(dry_run)

    unconfirmed = parser.parse_args(
        ["--workbook", str(workbook), "--write", "--replace-history"]
    )
    with pytest.raises(ValueError, match="backup-confirmed"):
        import_assessment_benchmarks.run_import(unconfirmed)


def test_import_script_runs_directly_from_api_directory(repo_root: Path) -> None:
    """Verify the documented direct Python invocation resolves package imports.

    Inputs:
        repo_root: Repository root containing the backend script and package.

    Outputs:
        None. ``--help`` exits successfully without parsing a workbook.
    """

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "api" / "scripts" / "import_assessment_benchmarks.py"),
            "--help",
        ],
        cwd=repo_root / "api",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--workbook" in result.stdout
    assert "--write" in result.stdout
