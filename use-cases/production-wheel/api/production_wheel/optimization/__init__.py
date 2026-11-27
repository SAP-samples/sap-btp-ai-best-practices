"""Public Pyomo/HiGHS optimization API for candidate set partitioning."""

from .block_session import BlockSolverSession
from .solver import (
    BlockSessionMetrics,
    BlockSolveEvidence,
    BaselineConstraintContext,
    BaselineConstraintLimits,
    CandidateCoefficients,
    HeuristicEvidence,
    ObjectiveLevel,
    SelectedCandidate,
    SolveResult,
    precompute_coefficients,
    strategy_candidates,
    solve_block_pool,
    solve_candidate_pools,
    solve_decomposed_pools,
)
from .solver_integrity import SolverResultInconsistentError, verify_solver_result

__all__ = [
    "BlockSessionMetrics",
    "BlockSolverSession",
    "BlockSolveEvidence",
    "BaselineConstraintContext",
    "BaselineConstraintLimits",
    "CandidateCoefficients",
    "HeuristicEvidence",
    "ObjectiveLevel",
    "SelectedCandidate",
    "SolveResult",
    "SolverResultInconsistentError",
    "precompute_coefficients",
    "strategy_candidates",
    "solve_block_pool",
    "solve_candidate_pools",
    "solve_decomposed_pools",
    "verify_solver_result",
]
