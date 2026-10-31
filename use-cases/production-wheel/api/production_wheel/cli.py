"""Command-line entry point for the production-wheel prototype.

Examples::

    .venv/bin/python -m production_wheel.cli extract \
      --primary "data/anonymized/<primary-workbook>.xlsx" \
      --enrichment "data/anonymized/<enrichment-workbook>.xlsx" \
      --output-root prototype/output

    .venv/bin/python -m production_wheel.cli extract \
      --primary "data/anonymized/<primary-workbook>.xlsx" \
      --snapshot-expectations data/snapshot_expectations.json

    .venv/bin/python -m production_wheel.cli build-matrix \
      --matrix-version BASELINE_EMPIRICAL_MATRIX_V1 \
      --run-directory prototype/output/<extraction-run> \
      --output prototype/output/baseline_empirical_matrix_v1.csv

    .venv/bin/python -m production_wheel.cli build-matrix \
      --matrix-version SYNTHETIC_VOLUME_MATRIX_V1 \
      --output prototype/output/synthetic_volume_matrix_v1.csv

    .venv/bin/python -m production_wheel.cli build-matrix \
      --matrix-version CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1 \
      --output prototype/output/customer_operational_volume_families_v1.csv

    .venv/bin/python -m production_wheel.cli solve \
      --run-directory prototype/output/<extraction-run> --suite demo

    .venv/bin/python -m production_wheel.cli solve \
      --run-directory prototype/output/<extraction-run> \
      --frontier customer_families_greenfield_pareto \
      --frontier-points 17 --global-epsilon-exponent 2 \
      --block-options-per-block 17 \
      --per-block-total-seconds 3600 \
      --exhaustive-block PL01/4000001 \
      --exhaustive-block PL01/4000002

    .venv/bin/python -m production_wheel.cli validate \
      --suite-directory prototype/output/<extraction-run>/solutions/<suite-run>

    .venv/bin/python -m production_wheel.cli validate \
      --frontier-directory prototype/output/<extraction-run>/solutions/<frontier-run>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from production_wheel.extraction.common import sha256_file, utc_now, write_csv
from production_wheel.extraction.pipeline import extract_snapshot, make_run_id
from production_wheel.matrix import (
    BASELINE_EMPIRICAL_MATRIX_VERSION,
    CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1,
    SYNTHETIC_MATRIX_VERSION,
    build_baseline_empirical_matrix,
    build_matrix_for_version,
)
from production_wheel.progress import ProgressBar
from production_wheel.schemas import GreenfieldValidationStatus
from production_wheel.validation import get_capabilities


def _extract_command(arguments: argparse.Namespace) -> int:
    """Run the workbook extraction command and print its manifest summary."""

    primary_path = Path(arguments.primary)
    snapshot_expectations = (
        json.loads(Path(arguments.snapshot_expectations).read_text(encoding="utf-8"))
        if arguments.snapshot_expectations
        else None
    )
    output_root = Path(arguments.output_root)
    run_id = arguments.run_id or make_run_id(sha256_file(primary_path))
    run_directory = output_root / run_id
    result = extract_snapshot(
        primary_path=primary_path,
        enrichment_path=Path(arguments.enrichment) if arguments.enrichment else None,
        snapshot_expectations=snapshot_expectations,
        run_directory=run_directory,
    )
    print(
        json.dumps(
            {"run_directory": str(result.run_directory), "status": result.manifest["status"]}
        )
    )
    return 0 if result.manifest["status"] == "ready" else 2


def _build_matrix_command(arguments: argparse.Namespace) -> int:
    """Write the selected explicit package-volume matrix."""

    output = Path(arguments.output)
    if arguments.matrix_version == BASELINE_EMPIRICAL_MATRIX_VERSION:
        if not arguments.run_directory:
            raise ValueError(
                "--run-directory is required for BASELINE_EMPIRICAL_MATRIX_V1"
            )
        from production_wheel.scenarios import load_canonical_inputs

        inputs = load_canonical_inputs(Path(arguments.run_directory))
        rows = build_baseline_empirical_matrix(
            inputs.members,
            inputs.baseline_evidence_members or inputs.members,
        )
    else:
        rows = build_matrix_for_version(arguments.matrix_version, ())
    write_csv(output, rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "matrix_version": arguments.matrix_version,
            }
        )
    )
    return 0


def _progress_callback() -> Callable[[str, int, int], None]:
    """Return a compact multi-phase terminal progress callback."""

    bars: dict[str, ProgressBar] = {}
    visible = {"blocks", "block_frontier", "scenarios", "pareto"}

    def update(phase: str, completed: int, total: int) -> None:
        """Update high-level phases while suppressing noisy inner-block bars."""

        if phase not in visible:
            return
        if phase not in bars:
            bars[phase] = ProgressBar(total, phase)
        bars[phase].update(completed)
        if completed >= total:
            bars[phase].finish()
            bars.pop(phase, None)

    return update


def _default_suite_directory(run_directory: Path) -> Path:
    """Return a timestamped non-colliding suite output directory."""

    timestamp = utc_now().replace("-", "").replace(":", "")
    return run_directory / "solutions" / f"demo-{timestamp}"


def _default_scenario_directory(run_directory: Path, scenario_id: str) -> Path:
    """Return a timestamped output directory for one named scenario."""

    timestamp = utc_now().replace("-", "").replace(":", "")
    return run_directory / "solutions" / f"{scenario_id}-{timestamp}"


def _parse_block_keys(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Parse repeatable ``PLANT/SEFI`` CLI values into stable block keys."""

    result = []
    for value in values:
        parts = tuple(part.strip() for part in value.split("/"))
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid exhaustive block {value!r}; expected PLANT/SEFI")
        result.append((parts[0], parts[1]))
    return tuple(sorted(set(result)))


def _prepare_frontier_config(
    arguments: argparse.Namespace, config: "RunConfig"
) -> "RunConfig":
    """Validate frontier-only CLI flags and fold them into the configuration.

    Shared by the ``--frontier`` and ``--request`` (PARETO) solve paths so both
    honour the same deep-run time budgets, exhaustive-block forcing, and per-stage
    limit. Returns the possibly-updated configuration.
    """

    if arguments.per_stage_seconds is not None and arguments.per_stage_seconds <= 0:
        raise ValueError("--per-stage-seconds must be positive")
    if (
        arguments.per_block_stage_seconds is not None
        and arguments.per_block_stage_seconds <= 0
    ):
        raise ValueError("--per-block-stage-seconds must be positive")
    if (
        arguments.per_block_total_seconds is not None
        and arguments.per_block_total_seconds <= 0
    ):
        raise ValueError("--per-block-total-seconds must be positive")
    if (
        arguments.per_block_stage_seconds is not None
        and arguments.per_block_total_seconds is not None
    ):
        raise ValueError(
            "--per-block-stage-seconds and --per-block-total-seconds are mutually exclusive"
        )
    if arguments.block_workers <= 0:
        raise ValueError("--block-workers must be positive")
    if arguments.large_block_candidate_threshold <= 0:
        raise ValueError("--large-block-candidate-threshold must be positive")
    if (
        not math.isfinite(arguments.global_epsilon_exponent)
        or arguments.global_epsilon_exponent < 1
    ):
        raise ValueError("--global-epsilon-exponent must be finite and at least one")
    exhaustive_blocks = _parse_block_keys(arguments.exhaustive_block)
    if exhaustive_blocks:
        config = config.model_copy(update={"exhaustive_blocks": exhaustive_blocks})
    if arguments.per_stage_seconds is not None:
        config = config.model_copy(
            update={
                "solver_limits": config.solver_limits.model_copy(
                    update={"time_limit_seconds": arguments.per_stage_seconds}
                )
            }
        )
    return config


def _run_request(arguments: argparse.Namespace) -> int:
    """Validate, compile, and solve a typed ``SolveRequest`` JSON file.

    The request carries the full ``RunConfig`` plus typed constraints and an
    optional solve scope. It is validated deterministically, compiled into scoped
    inputs / per-block caps / a candidate filter, then solved. This phase wires
    the PARETO greenfield frontier (the agent's primary path); non-PARETO configs
    return a clear unsupported-mode result.
    """

    from production_wheel.scenarios import (
        greenfield_frontier_configuration,
        load_canonical_inputs,
        run_greenfield_frontier,
    )
    from production_wheel.schemas import CoverageMode, SolveRequest
    from production_wheel.validation import ValidationContext, validate_request
    from production_wheel.constraints import ConstraintCompileError, compile_request
    from production_wheel.greenfield_reporting import (
        validate_greenfield_frontier_artifacts,
        write_greenfield_frontier_artifacts,
    )

    run_directory = Path(arguments.run_directory).resolve()
    inputs = load_canonical_inputs(run_directory)
    request = SolveRequest.model_validate_json(
        Path(arguments.request).read_text(encoding="utf-8")
    )

    versions_by_block: dict[tuple[str, str], set[str]] = {}
    for version in inputs.production_versions:
        versions_by_block.setdefault((version.plant, version.sefi), set()).add(
            version.pv_id
        )
    context = ValidationContext(
        materials=frozenset(member.fini_id for member in inputs.members),
        production_versions_by_block={
            block: frozenset(pv_ids) for block, pv_ids in versions_by_block.items()
        },
    )
    validation = validate_request(request, context)
    if not validation.is_valid:
        print(
            json.dumps(
                {
                    "status": validation.status.value,
                    "messages": [
                        {
                            "code": message.code,
                            "severity": message.severity,
                            "message": message.message,
                            "constraint_id": message.constraint_id,
                        }
                        for message in validation.messages
                    ],
                }
            )
        )
        return 2

    try:
        compiled = compile_request(request, inputs)
    except ConstraintCompileError as error:
        print(json.dumps({"status": "constraint_compile_error", "error": str(error)}))
        return 2

    config = compiled.config
    if config.coverage_mode is not CoverageMode.PARETO:
        print(
            json.dumps(
                {
                    "status": "unsupported_request_mode",
                    "error": (
                        "solve --request currently supports PARETO frontier "
                        f"configurations only; got {config.coverage_mode.value}. Use "
                        "--scenario for unconstrained named scenarios."
                    ),
                }
            )
        )
        return 2

    config = _prepare_frontier_config(arguments, config)
    source_manifest_path = run_directory / "run_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    manifest_metadata = {
        "source_extraction": source_manifest,
        "solve_request": {
            "request_id": request.request_id(),
            "scope": [
                {"plant": entry.plant, "sefi": entry.sefi} for entry in request.scope
            ],
            "applied_constraints": list(compiled.applied),
        },
    }
    output_directory = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory
        else _default_scenario_directory(run_directory, config.scenario_id)
    )
    frontier = run_greenfield_frontier(
        compiled.inputs,
        config,
        point_count=arguments.frontier_points,
        block_option_count=arguments.block_options_per_block,
        per_block_stage_seconds=arguments.per_block_stage_seconds,
        per_block_total_seconds=arguments.per_block_total_seconds,
        block_worker_count=arguments.block_workers,
        large_block_candidate_threshold=arguments.large_block_candidate_threshold,
        global_epsilon_exponent=arguments.global_epsilon_exponent,
        pool_transform=compiled.filter_pools,
        progress=_progress_callback(),
    )
    paths = write_greenfield_frontier_artifacts(
        output_directory,
        compiled.inputs.fini_rows,
        compiled.inputs.production_versions,
        frontier,
        baseline_evidence_members=compiled.inputs.baseline_evidence_members,
        manifest_metadata=manifest_metadata,
    )
    verification = validate_greenfield_frontier_artifacts(output_directory)
    print(
        json.dumps(
            {
                "frontier_directory": str(paths.output_directory),
                "pareto_points": len(frontier.points),
                "applied_constraints": list(compiled.applied),
                "validation": verification,
            }
        )
    )
    return 0 if verification["valid"] else 2


def _solve_command(arguments: argparse.Namespace) -> int:
    """Run the demonstrator suite and write every evidence artifact."""

    if getattr(arguments, "request", None):
        return _run_request(arguments)

    from production_wheel.scenarios import (
        demo_configurations,
        greenfield_frontier_configuration,
        load_canonical_inputs,
        named_configurations,
        run_greenfield_frontier,
        run_single_scenario,
        run_scenario_suite,
    )
    from production_wheel.suite_reporting import (
        validate_scenario_artifacts,
        validate_suite_artifacts,
        write_no_incumbent_artifacts,
        write_suite_artifacts,
    )
    from production_wheel.greenfield_reporting import (
        validate_greenfield_frontier_artifacts,
        write_greenfield_frontier_artifacts,
    )
    from production_wheel.reporting import write_scenario_artifacts
    from production_wheel.solution_validation import validate_solution

    run_directory = Path(arguments.run_directory).resolve()
    inputs = load_canonical_inputs(run_directory)
    available = named_configurations()
    configs = demo_configurations()
    if arguments.per_stage_seconds is not None:
        if arguments.per_stage_seconds <= 0:
            raise ValueError("--per-stage-seconds must be positive")
        configs = tuple(
            config.model_copy(
                update={
                    "solver_limits": config.solver_limits.model_copy(
                        update={"time_limit_seconds": arguments.per_stage_seconds}
                    )
                }
            )
            for config in configs
        )
    source_manifest_path = run_directory / "run_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    if arguments.frontier:
        config = _prepare_frontier_config(
            arguments, greenfield_frontier_configuration()
        )
        output_directory = (
            Path(arguments.output_directory).resolve()
            if arguments.output_directory
            else _default_scenario_directory(run_directory, config.scenario_id)
        )
        frontier = run_greenfield_frontier(
            inputs,
            config,
            point_count=arguments.frontier_points,
            block_option_count=arguments.block_options_per_block,
            per_block_stage_seconds=arguments.per_block_stage_seconds,
            per_block_total_seconds=arguments.per_block_total_seconds,
            block_worker_count=arguments.block_workers,
            large_block_candidate_threshold=(
                arguments.large_block_candidate_threshold
            ),
            global_epsilon_exponent=arguments.global_epsilon_exponent,
            progress=_progress_callback(),
        )
        paths = write_greenfield_frontier_artifacts(
            output_directory,
            inputs.fini_rows,
            inputs.production_versions,
            frontier,
            baseline_evidence_members=inputs.baseline_evidence_members,
            manifest_metadata={"source_extraction": source_manifest},
        )
        verification = validate_greenfield_frontier_artifacts(output_directory)
        print(
            json.dumps(
                {
                    "frontier_directory": str(paths.output_directory),
                    "pareto_points": len(frontier.points),
                    "validation": verification,
                }
            )
        )
        return 0 if verification["valid"] else 2
    if arguments.scenario:
        config = available[arguments.scenario]
        if arguments.per_stage_seconds is not None:
            config = config.model_copy(
                update={
                    "solver_limits": config.solver_limits.model_copy(
                        update={"time_limit_seconds": arguments.per_stage_seconds}
                    )
                }
            )
        output_directory = (
            Path(arguments.output_directory).resolve()
            if arguments.output_directory
            else _default_scenario_directory(run_directory, config.scenario_id)
        )
        outcome = run_single_scenario(inputs, config, progress=_progress_callback())
        if outcome.solve_result is None or not outcome.solve_result.has_incumbent:
            if output_directory.exists():
                raise FileExistsError(output_directory)
            write_no_incumbent_artifacts(
                output_directory,
                outcome,
                source_manifest,
            )
            verification = validate_scenario_artifacts(output_directory)
            print(
                json.dumps(
                    {
                        "scenario_directory": str(output_directory),
                        "status": outcome.status,
                        "result_class": outcome.result_class,
                        "validation": verification,
                    }
                )
            )
            return 2
        validation = validate_solution(
            outcome.solve_result,
            outcome.members or inputs.members,
            outcome.pools,
            config,
            inputs.production_versions,
            inputs.baseline_evidence_members or inputs.members,
        )
        active_members = outcome.members or inputs.members
        acceptance_summary = (
            outcome.acceptance.as_dict()
            if outcome.acceptance is not None
            else {
                "acceptance_status": (
                    GreenfieldValidationStatus.VALID_PARETO_POINT.value
                    if validation.is_valid
                    else GreenfieldValidationStatus.INVALID_PARETO_POINT.value
                ),
                "acceptance_evidence_source": "independent_greenfield_validation",
                "failed_baseline_guardrails": "",
            }
        )
        paths = write_scenario_artifacts(
            output_directory,
            inputs.fini_rows,
            outcome.members or inputs.members,
            outcome.pools,
            outcome.solve_result,
            config,
            validation,
            manifest_metadata={"source_extraction": source_manifest},
            production_versions=inputs.production_versions,
            acceptance_summary=acceptance_summary,
            baseline_evidence_members=(
                inputs.baseline_evidence_members or inputs.members
            ),
            include_baseline_comparison=(
                not config.is_greenfield
                or (
                    bool(active_members)
                    and all(member.baseline_group for member in active_members)
                )
            ),
        )
        verification = validate_scenario_artifacts(output_directory)
        print(
            json.dumps(
                {
                    "scenario_directory": str(output_directory),
                    "status": outcome.status,
                    "result_class": outcome.result_class,
                    "proposed_subgroups": str(paths.proposed_subgroups),
                    "validation": verification,
                }
            )
        )
        return 0 if verification["valid"] else 2

    if (
        arguments.exhaustive_block
        or arguments.per_block_stage_seconds is not None
        or arguments.per_block_total_seconds is not None
    ):
        raise ValueError(
            "deep-run time and exhaustive-block options require --frontier"
        )

    output_directory = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory
        else _default_suite_directory(run_directory)
    )
    suite = run_scenario_suite(inputs, configs=configs, progress=_progress_callback())
    paths = write_suite_artifacts(output_directory, suite, source_manifest)
    verification = validate_suite_artifacts(output_directory)
    print(
        json.dumps(
            {
                "suite_directory": str(paths.output_directory),
                "partial": suite.partial,
                "elapsed_seconds": round(suite.elapsed_seconds, 3),
                "summary_results": len(suite.outcomes),
                "validation": verification,
            }
        )
    )
    return 0 if verification["valid"] and not suite.partial else 2


def _validate_command(arguments: argparse.Namespace) -> int:
    """Verify written suite or single-scenario artifacts."""

    from production_wheel.suite_reporting import (
        validate_scenario_artifacts,
        validate_suite_artifacts,
    )

    if arguments.frontier_directory:
        from production_wheel.greenfield_reporting import (
            validate_greenfield_frontier_artifacts,
        )

        result = validate_greenfield_frontier_artifacts(
            Path(arguments.frontier_directory).resolve()
        )
    elif arguments.scenario_directory:
        result = validate_scenario_artifacts(
            Path(arguments.scenario_directory).resolve()
        )
    else:
        result = validate_suite_artifacts(Path(arguments.suite_directory).resolve())
    print(json.dumps(result))
    return 0 if result["valid"] else 2


def _capabilities_command(_arguments: argparse.Namespace) -> int:
    """Print the finite machine-readable optimizer capability contract."""

    print(json.dumps(get_capabilities(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""

    from production_wheel.scenarios import named_configurations

    parser = argparse.ArgumentParser(description="Auditable production-wheel optimizer prototype")
    subcommands = parser.add_subparsers(dest="command", required=True)
    extract = subcommands.add_parser(
        "extract", help="extract the primary workbook and optional enrichment data"
    )
    extract.add_argument("--primary", required=True, help="authoritative primary production workbook")
    extract.add_argument("--enrichment", help="optional enrichment input workbook")
    extract.add_argument(
        "--snapshot-expectations",
        help="optional JSON file with reviewed frozen-snapshot figures to enforce",
    )
    extract.add_argument("--output-root", default="prototype/output", help="run output root")
    extract.add_argument("--run-id", help="explicit run identifier")
    extract.set_defaults(handler=_extract_command)

    matrix = subcommands.add_parser(
        "build-matrix", help="write an explicit package-volume evidence matrix"
    )
    matrix.add_argument(
        "--matrix-version",
        choices=(
            BASELINE_EMPIRICAL_MATRIX_VERSION,
            SYNTHETIC_MATRIX_VERSION,
            CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1,
        ),
        required=True,
        help="empirical evidence, synthetic sensitivity, or customer family rule",
    )
    matrix.add_argument(
        "--run-directory",
        help="completed extraction run required for the empirical matrix",
    )
    matrix.add_argument(
        "--output",
        default="prototype/output/package_volume_matrix.csv",
        help="matrix CSV path",
    )
    matrix.set_defaults(handler=_build_matrix_command)

    solve = subcommands.add_parser("solve", help="run the optimizer scenario suite")
    solve.add_argument("--run-directory", required=True, help="completed extraction run")
    solve_mode = solve.add_mutually_exclusive_group()
    solve_mode.add_argument("--suite", choices=("demo",))
    solve_mode.add_argument(
        "--scenario",
        choices=tuple(named_configurations()),
        help="run one named typed scenario instead of the full suite",
    )
    solve_mode.add_argument(
        "--frontier",
        choices=("customer_families_greenfield_pareto",),
        help="run the reviewed greenfield coverage/J_CH Pareto workflow",
    )
    solve.add_argument("--output-directory", help="new explicit suite output directory")
    solve.add_argument(
        "--request",
        help=(
            "path to a typed SolveRequest JSON; it is validated, compiled into "
            "scoped inputs, per-block caps, and candidate filters, then solved "
            "(PARETO frontier configurations)"
        ),
    )
    solve.add_argument(
        "--per-stage-seconds",
        type=float,
        help=(
            "optional scenario-wide time budget for each HiGHS lexicographic "
            "stage, apportioned across decomposed blocks"
        ),
    )
    solve.add_argument(
        "--frontier-points",
        type=int,
        default=17,
        help="anchor-inclusive low-epsilon-biased global samples (default 17)",
    )
    solve.add_argument(
        "--global-epsilon-exponent",
        type=float,
        default=2.0,
        help="power-law density toward low J_CH; 1 is uniform (default 2)",
    )
    solve.add_argument(
        "--block-options-per-block",
        type=int,
        default=5,
        help="maximum adaptive anchor/epsilon requests per plant/SEFI block",
    )
    solve.add_argument(
        "--per-block-stage-seconds",
        type=float,
        help="optional independent time limit for every frontier block objective tier",
    )
    solve.add_argument(
        "--per-block-total-seconds",
        type=float,
        help="optional conservative total time budget for each frontier block",
    )
    solve.add_argument(
        "--block-workers",
        type=int,
        default=2,
        help="maximum process-isolated block frontier workers (default 2)",
    )
    solve.add_argument(
        "--large-block-candidate-threshold",
        type=int,
        default=250_000,
        help="candidate count at which a block is scheduled alone",
    )
    solve.add_argument(
        "--exhaustive-block",
        action="append",
        default=[],
        metavar="PLANT/SEFI",
        help="repeatable block to enumerate exhaustively in the greenfield pool",
    )
    solve.set_defaults(handler=_solve_command)

    validate = subcommands.add_parser("validate", help="verify written result artifacts")
    validate_mode = validate.add_mutually_exclusive_group(required=True)
    validate_mode.add_argument("--suite-directory")
    validate_mode.add_argument("--scenario-directory")
    validate_mode.add_argument("--frontier-directory")
    validate.set_defaults(handler=_validate_command)

    capabilities = subcommands.add_parser(
        "capabilities", help="print the agent-facing typed capability contract"
    )
    capabilities.set_defaults(handler=_capabilities_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and return a process exit code."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":  # pragma: no cover - exercised through console smoke tests
    raise SystemExit(main())
