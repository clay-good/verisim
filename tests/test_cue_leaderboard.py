"""SPEC-21 CL1 / H91 -- the verisim-cue leaderboard + its discriminative validity.

The contract: the controlled fidelity ladder is faithful on structure and drifts on content (so the
leaderboard ranks on the structure->content gradient), the stand-in is deterministic, the scorecard
*stably ranks* the ladder by fidelity (Kendall tau high, adjacent tiers resolved above seed noise --
the H65 parallel for the computer-use vertical), and the CSV round-trips. Torch-free.
"""

from __future__ import annotations

from verisim.acd.host_integrity import make_workload, oracle_step
from verisim.cue.leaderboard import (
    REFERENCE_CUE_PROPOSERS,
    CueLeaderboardConfig,
    HostFidelityProposer,
    build_cue_leaderboard,
    write_csv,
)
from verisim.cue.tasks import file_contents
from verisim.hostoracle.reference import ReferenceHostOracle


def test_fidelity_ladder_is_monotone_and_structure_clean():
    rows, _ = build_cue_leaderboard(CueLeaderboardConfig())
    by_alpha = {r.alpha: r for r in rows}
    # mean catch is monotone increasing in the fidelity proxy alpha
    alphas = sorted(by_alpha)
    catches = [by_alpha[a].mean_catch for a in alphas]
    assert catches == sorted(catches)
    assert by_alpha[0.0].mean_catch < by_alpha[1.0].mean_catch
    # structure tasks are never load-bearing: the stand-in keeps the process tree / fd table exact,
    # so every tier recalls them perfectly (the gradient lives in content)
    for r in rows:
        assert r.per_task["process-control"] == 1.0
        assert r.per_task["fd-control"] == 1.0
    # the ceiling (alpha=1) is content-faithful too; the floor (alpha=0) drifts on content
    assert by_alpha[1.0].per_task["content-value"] == 1.0
    assert by_alpha[0.0].per_task["content-value"] < 0.5
    # the structure->content gradient: content recall <= which-file recall at every tier
    for r in rows:
        assert r.per_task["content-value"] <= r.per_task["file-integrity"] + 1e-9


def test_content_recall_climbs_with_capacity():
    rows, _ = build_cue_leaderboard(CueLeaderboardConfig())
    by_alpha = {r.alpha: r.per_task["content-value"] for r in rows}
    series = [by_alpha[a] for a in sorted(by_alpha)]
    assert series == sorted(series)  # content recall is monotone in alpha
    assert series[-1] - series[0] > 0.5  # a wide, resolvable spread


def test_scorecard_is_discriminative_h91():
    _, stability = build_cue_leaderboard(CueLeaderboardConfig())
    # the strict SPEC-18 H65 test: stable ranking AND adjacent tiers resolved above paired noise
    assert stability.tau_mean >= 0.8
    assert stability.tau_lo > 0.0
    assert stability.min_adjacent_gap > 2 * stability.max_seed_noise
    assert stability.discriminative is True


def test_smoke_config_is_also_discriminative():
    _, stability = build_cue_leaderboard(CueLeaderboardConfig.smoke())
    assert stability.discriminative is True


def test_stand_in_is_deterministic_and_drifts_only_on_content():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(701, 16, oracle=oracle)
    true_step = oracle_step(oracle)
    # deterministic given (alpha, seed)
    a = HostFidelityProposer(0.5, 701)
    b = HostFidelityProposer(0.5, 701)
    s_a = s_b = start
    for act in actions:
        s_a, s_b = a.step(s_a, act), b.step(s_b, act)
    assert file_contents(s_a) == file_contents(s_b)
    # a perfectly-faithful stand-in (alpha=1) reproduces the oracle's content exactly
    perfect = HostFidelityProposer(1.0, 701)
    truth = start
    sp = start
    for act in actions:
        truth = true_step(truth, act)
        sp = perfect.step(sp, act)
    assert file_contents(sp) == file_contents(truth)


def test_csv_round_trips(tmp_path):
    rows, _ = build_cue_leaderboard(CueLeaderboardConfig.smoke())
    out = write_csv(rows, tmp_path / "cl1.csv", "verisim-cue@test")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("manifest,proposer,tier,alpha,mean_catch")
    assert len(lines) == 1 + len(REFERENCE_CUE_PROPOSERS)
    assert all("verisim-cue@test" in row for row in lines[1:])
