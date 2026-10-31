"""Per-block group-size cap overrides: scoped, additive, membership-affecting.

These tests cover the feature that lets the agent tune the maximum group size
per (plant, SEFI) block. The historical cap of seven is plant-specific; other
plants may differ. An override-free configuration must behave exactly as before,
and an override must shrink only the block it names.
"""

from __future__ import annotations

from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.schemas import CoverageMode, GroupSizeOverride, RunConfig


def _member(fini_id: str, *, plant: str = "P1", sefi: str = "S1") -> CandidateMember:
    """One synthetic single-line member; identical volume keeps every subset feasible."""

    return CandidateMember(
        fini_id=fini_id,
        plant=plant,
        sefi=sefi,
        eligible_lines=frozenset({"1"}),
        demand_litres=1_000.0,
        pallet_litres=25.0,
        package_volume=0.25,
        fixed_pv="PV1",
        fixed_lot_litres=100.0,
        baseline_group=None,
        pck_code="PCK-A",
    )


def _versions(*blocks: tuple[str, str]) -> tuple[ProductionVersion, ...]:
    """Return one fixed production version per (plant, SEFI) block."""

    return tuple(
        ProductionVersion(plant=plant, sefi=sefi, pv_id="PV1", nominal_lot_litres=100.0)
        for plant, sefi in blocks
    )


def _max_size_by_block(pools) -> dict[tuple[str, str], int]:
    """Return the largest candidate group size generated for each block."""

    sizes: dict[tuple[str, str], int] = {}
    for pool in pools:
        for candidate in pool.candidates:
            key = candidate.block_key
            sizes[key] = max(sizes.get(key, 0), len(candidate.member_ids))
    return sizes


def test_cap_for_block_defaults_to_global_cap() -> None:
    """With no overrides the resolver returns the frozen global cap of seven."""

    config = RunConfig()
    assert config.cap_for_block("P1", "S1") == config.group_size.effective_cap == 7


def test_cap_for_block_returns_scoped_override_only_for_its_block() -> None:
    """An override applies to its named block; other blocks keep the global cap."""

    config = RunConfig(
        group_size_overrides=(GroupSizeOverride(plant="P1", sefi="S1", maximum=2),)
    )
    assert config.cap_for_block("P1", "S1") == 2
    assert config.cap_for_block("P2", "S2") == 7


def test_override_shrinks_only_the_targeted_block() -> None:
    """Generation caps the targeted block's candidates and leaves others untouched."""

    members = [
        _member("101"),
        _member("102"),
        _member("103"),
        _member("104"),
        _member("201", plant="P2", sefi="S2"),
        _member("202", plant="P2", sefi="S2"),
        _member("203", plant="P2", sefi="S2"),
        _member("204", plant="P2", sefi="S2"),
    ]
    versions = _versions(("P1", "S1"), ("P2", "S2"))

    base = RunConfig(coverage_mode=CoverageMode.GREENFIELD_COVERAGE)
    scoped = RunConfig(
        coverage_mode=CoverageMode.GREENFIELD_COVERAGE,
        group_size_overrides=(GroupSizeOverride(plant="P1", sefi="S1", maximum=2),),
    )

    base_sizes = _max_size_by_block(generate_candidate_pools(members, versions, base))
    scoped_sizes = _max_size_by_block(
        generate_candidate_pools(members, versions, scoped)
    )

    # Four line-compatible FINIs per block -> groups up to min(7, 4) = 4 by default.
    assert base_sizes[("P1", "S1")] == 4
    assert base_sizes[("P2", "S2")] == 4
    # The override caps only P1/S1 at 2; the untargeted P2/S2 block is unchanged.
    assert scoped_sizes[("P1", "S1")] == 2
    assert scoped_sizes[("P2", "S2")] == 4


def test_override_changes_structural_fingerprint() -> None:
    """A cap override is membership-affecting, so it changes the structural digest."""

    base = RunConfig()
    scoped = RunConfig(
        group_size_overrides=(GroupSizeOverride(plant="P1", sefi="S1", maximum=3),)
    )
    assert (
        base.structural_ruleset_fingerprint()
        != scoped.structural_ruleset_fingerprint()
    )
    # An override-free config must keep the exact prior fingerprint (byte-identical).
    assert (
        base.structural_ruleset_fingerprint()
        == RunConfig().structural_ruleset_fingerprint()
    )
