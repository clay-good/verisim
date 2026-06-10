"""ED24 — voluntary step-down: the graceful handoff + relinquish-needs-no-quorum (DS0 increment 17).

The smoke instance of the DS0-increment-17 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — `step_down` leaves the cluster leaderless at the same term,
no leaderless commit window exists and a successor's election lands at a strictly higher term (the
voluntary-handoff lifecycle, Panel A), and only the current leader may step down while a minority-
stranded leader can still relinquish where its `propose` is `no_quorum` (authority + partition-
independence, Panel B) — with Tier-B reproducing every transition bit-for-bit. The committed figure
comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed24 import ED24Config, ED24Result, run_ed24, write_csv


def test_panel_a_handoff_lifecycle_leaves_no_leaderless_commit_window():
    result = run_ed24(ED24Config())
    assert result.handoff_leaderless_rate == 1.0  # post-step_down propose is rejected (leaderless)
    assert result.reelect_higher_term_rate == 1.0  # the successor's election bumps the term
    assert result.new_leader_commits_rate == 1.0  # the legitimate successor commits


def test_panel_b_authority_and_partition_independence():
    result = run_ed24(ED24Config())
    assert result.nonleader_stepdown_rejected is True  # only the current leader may step down
    assert result.second_stepdown_rejected is True  # a second step_down is a no-op reject
    assert result.minority_leader_steps_down is True  # relinquishing needs no quorum...
    assert result.minority_propose_blocked is True  # ...where committing (propose) is no_quorum


def test_tier_b_reproduces_every_transition():
    result = run_ed24(ED24Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed24(ED24Config())
    out = write_csv(result, tmp_path / "ed24.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("handoff,") for line in lines)
    assert any(line.startswith("partition,") for line in lines)


def test_config_round_trips():
    cfg = ED24Config.from_dict({"cluster_sizes": [3, 5], "v1": "b", "v2": "c"})
    assert cfg.cluster_sizes == (3, 5)
    assert isinstance(run_ed24(cfg), ED24Result)
