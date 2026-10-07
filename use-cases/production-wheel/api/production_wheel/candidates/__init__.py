"""Public deterministic candidate-generation API."""

from .generation import (
    generate_block_pool,
    generate_candidate_pools,
    line_feasible_subset_count,
    partition_blocks,
    project_block,
    rebuild_pool,
)
from .models import (
    BlockProjection,
    Candidate,
    CandidateMember,
    CandidatePool,
    PoolBudgetExceeded,
    PoolSizeTrace,
    ProductionVersion,
)

__all__ = [
    "BlockProjection",
    "Candidate",
    "CandidateMember",
    "CandidatePool",
    "PoolBudgetExceeded",
    "PoolSizeTrace",
    "ProductionVersion",
    "generate_block_pool",
    "generate_candidate_pools",
    "line_feasible_subset_count",
    "partition_blocks",
    "project_block",
    "rebuild_pool",
]
