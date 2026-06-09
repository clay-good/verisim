"""Invariants for the product layer: manifest, leaderboard, conformance (SPEC-18 §6).

Structural, not magnitude (the macOS-first principle): the manifest hashes stably and versions;
Kendall τ is correct on known orderings; the leaderboard orders the fidelity ladder and the
rank-stability verdict is discriminative; the conformance suite is green; the metadata emitters
produce well-formed Croissant / datasheet / model-card.
"""

from __future__ import annotations

import json

from verisim.bench import (
    BatteryManifest,
    Proposer,
    all_passed,
    build_leaderboard,
    croissant_metadata,
    datasheet,
    kendall_tau,
    model_card,
    run_conformance,
)


def _small() -> BatteryManifest:
    return BatteryManifest(seeds=tuple(range(8)), n_steps=50)


def test_kendall_tau_known_orderings() -> None:
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0  # identical
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0  # reversed
    # A single swap of one adjacent pair is mostly concordant, strictly between -1 and 1.
    tau = kendall_tau([1, 2, 3, 4], [2, 1, 3, 4])
    assert 0.0 < tau < 1.0


def test_manifest_hash_stable_and_version_changes() -> None:
    a = BatteryManifest().manifest_hash()
    b = BatteryManifest().manifest_hash()
    assert a == b  # deterministic
    # A different battery (fewer seeds) is a different MAJOR identity.
    assert BatteryManifest(seeds=(0, 1)).manifest_hash() != a
    assert "verisim-bench@" in BatteryManifest().version_tag()


def test_leaderboard_orders_the_fidelity_ladder() -> None:
    rows, _ = build_leaderboard(_small())
    for world in {r.world for r in rows}:
        cells = {r.proposer: r.mean_faithful for r in rows if r.world == world}
        # The ladder is monotone in fidelity: floor ≤ lo ≤ mid ≤ hi ≤ ceiling.
        assert cells["null"] <= cells["learned-lo"] <= cells["learned-mid"] <= cells["learned-hi"]
        assert cells["learned-hi"] <= cells["oracle-ceiling"]
        assert cells["oracle-ceiling"] >= 0.99  # the ceiling is (near) perfectly faithful


def test_rank_stability_is_discriminative() -> None:
    _, stability = build_leaderboard(BatteryManifest(seeds=tuple(range(16)), n_steps=80))
    for s in stability:
        assert s.tau_mean >= 0.8  # the ranking is stable across seed splits
        assert s.tau_lo > 0.0  # the CI excludes 0
        assert s.discriminative  # adjacent tiers resolved above paired seed noise


def test_conformance_suite_is_green() -> None:
    results = run_conformance()
    assert results  # not empty
    assert all_passed(results)
    # Every world's RL env is covered by both contracts.
    surfaces = {r.surface for r in results}
    assert {"rl:filesystem", "rl:host", "rl:distributed"} <= surfaces


def test_metadata_emitters_well_formed() -> None:
    m = BatteryManifest()
    cr = croissant_metadata(m)
    assert cr["@type"] == "Dataset"
    assert cr["manifestHash"] == m.manifest_hash()
    json.dumps(cr)  # serializable
    ds = datasheet(m)
    assert "Datasheet" in ds and m.manifest_hash() in ds
    mc = model_card(m)
    assert "Model card" in mc and "oracle-ceiling" in mc


def test_custom_proposer_set() -> None:
    m = BatteryManifest(
        seeds=(0, 1, 2, 3),
        n_steps=40,
        proposers=(Proposer("a", "floor", 0.0), Proposer("b", "ceiling", 1.0)),
    )
    rows, _ = build_leaderboard(m)
    assert {r.proposer for r in rows} == {"a", "b"}
