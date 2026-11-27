"""Process-isolated, fork-shared orchestration for greenfield block frontiers."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from production_wheel.candidates import CandidateMember, CandidatePool
from production_wheel.candidates.models import BlockKey
from production_wheel.pareto import (
    GreenfieldBlockFailure,
    GreenfieldBlockOption,
    GreenfieldBlockSessionAudit,
    GreenfieldBlockSolveAudit,
    _build_one_greenfield_block_options,
)
from production_wheel.schemas import RunConfig

_POOLS: Mapping[BlockKey, CandidatePool] = {}
_MEMBERS: Mapping[BlockKey, tuple[CandidateMember, ...]] = {}
_CONFIG: RunConfig | None = None


@dataclass(frozen=True, slots=True)
class ParallelBlockExecution:
    """Stable successful results, failures, and actual execution policy."""

    completed: tuple[
        tuple[
            BlockKey,
            tuple[GreenfieldBlockOption, ...],
            tuple[GreenfieldBlockSolveAudit, ...],
            GreenfieldBlockSessionAudit,
        ],
        ...,
    ]
    failures: tuple[GreenfieldBlockFailure, ...]
    execution_mode: str
    worker_count: int


def _solve_registered_block(
    block_key: BlockKey,
    option_count: int,
    stage_seconds: float,
) -> tuple[
    tuple[GreenfieldBlockOption, ...],
    tuple[GreenfieldBlockSolveAudit, ...],
    GreenfieldBlockSessionAudit,
]:
    """Solve one fork-inherited block without serializing its candidate pool."""

    if _CONFIG is None:
        raise RuntimeError("parallel block worker context is not initialized")
    return _build_one_greenfield_block_options(
        _POOLS[block_key],
        _MEMBERS[block_key],
        _CONFIG,
        option_count,
        stage_seconds,
    )


def _failure(pool: CandidatePool, error: Exception) -> GreenfieldBlockFailure:
    """Convert one worker exception into stable partial-run evidence."""

    return GreenfieldBlockFailure(
        block_key=pool.block_key,
        candidate_count=len(pool.candidates),
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _run_batch(
    pools: Sequence[CandidatePool],
    option_count: int,
    time_limits: Mapping[BlockKey, float],
    worker_count: int,
    on_complete: Callable[[], None] | None = None,
) -> tuple[
    list[
        tuple[
            BlockKey,
            tuple[GreenfieldBlockOption, ...],
            tuple[GreenfieldBlockSolveAudit, ...],
            GreenfieldBlockSessionAudit,
        ]
    ],
    list[GreenfieldBlockFailure],
]:
    """Run one size-compatible batch and retain every completed result."""

    completed = []
    failures = []
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(worker_count, len(pools)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _solve_registered_block,
                pool.block_key,
                option_count,
                float(time_limits[pool.block_key]),
            ): pool
            for pool in pools
        }
        for future in as_completed(futures):
            pool = futures[future]
            try:
                options, audits, session_audit = future.result()
                completed.append(
                    (pool.block_key, options, audits, session_audit)
                )
            except Exception as error:  # Preserve successful sibling blocks.
                failures.append(_failure(pool, error))
            if on_complete:
                on_complete()
    return completed, failures


def solve_block_frontiers_parallel(
    pools: Sequence[CandidatePool],
    members: Sequence[CandidateMember],
    config: RunConfig,
    option_count: int,
    time_limits: Mapping[BlockKey, float],
    worker_count: int,
    large_block_candidate_threshold: int,
    progress: Callable[[str, int, int], None] | None = None,
) -> ParallelBlockExecution:
    """Solve small blocks concurrently and candidate-heavy blocks alone.

    Candidate pools and member records are inherited through POSIX ``fork`` so
    workers receive only a block key and scalar limits. Platforms without fork
    fall back to explicit sequential isolation avoidance instead of pickling
    million-candidate pools.
    """

    if worker_count < 1:
        raise ValueError("block worker count must be positive")
    if large_block_candidate_threshold < 1:
        raise ValueError("large block candidate threshold must be positive")
    ordered = tuple(sorted(pools, key=lambda item: item.block_key))
    progress_count = 0

    def completed_one() -> None:
        """Advance the parent-owned terminal progress bar once."""

        nonlocal progress_count
        progress_count += 1
        if progress:
            progress("block_frontier", progress_count, len(ordered))

    by_block_members = {
        pool.block_key: tuple(
            member for member in members if member.block_key == pool.block_key
        )
        for pool in ordered
    }
    if worker_count == 1 or "fork" not in multiprocessing.get_all_start_methods():
        completed = []
        failures = []
        for pool in ordered:
            try:
                options, audits, session_audit = _build_one_greenfield_block_options(
                    pool,
                    by_block_members[pool.block_key],
                    config,
                    option_count,
                    float(time_limits[pool.block_key]),
                )
                completed.append((pool.block_key, options, audits, session_audit))
            except Exception as error:
                failures.append(_failure(pool, error))
            completed_one()
        return ParallelBlockExecution(
            completed=tuple(completed),
            failures=tuple(failures),
            execution_mode=(
                "sequential_requested"
                if worker_count == 1
                else "sequential_no_fork"
            ),
            worker_count=1,
        )

    global _POOLS, _MEMBERS, _CONFIG
    _POOLS = {pool.block_key: pool for pool in ordered}
    _MEMBERS = by_block_members
    _CONFIG = config.model_copy(
        update={
            "solver_limits": config.solver_limits.model_copy(
                update={"threads": 1}
            )
        }
    )
    small = tuple(
        pool
        for pool in ordered
        if len(pool.candidates) < large_block_candidate_threshold
    )
    small_keys = {pool.block_key for pool in small}
    large = tuple(pool for pool in ordered if pool.block_key not in small_keys)
    completed = []
    failures = []
    try:
        if small:
            batch_completed, batch_failures = _run_batch(
                small, option_count, time_limits, worker_count, completed_one
            )
            completed.extend(batch_completed)
            failures.extend(batch_failures)
        for pool in large:
            batch_completed, batch_failures = _run_batch(
                (pool,), option_count, time_limits, 1, completed_one
            )
            completed.extend(batch_completed)
            failures.extend(batch_failures)
    finally:
        _POOLS = {}
        _MEMBERS = {}
        _CONFIG = None
    return ParallelBlockExecution(
        completed=tuple(sorted(completed, key=lambda item: item[0])),
        failures=tuple(sorted(failures, key=lambda item: item.block_key)),
        execution_mode="process_fork_shared_candidate_pools",
        worker_count=min(worker_count, max(len(small), 1)),
    )
