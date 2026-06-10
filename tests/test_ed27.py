"""ED27 — membership change: the quorum threshold tracks the voting set (DS0 increment 20).

The smoke instance of the DS0-increment-20 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — shrinking the voting set turns a lone leader's minority into
a majority (and re-growing it re-blocks, Panel A), and removing failed nodes restores availability
while the active leader is fenced from removal (Panel B) — with Tier-B reproducing every transition
bit-for-bit. The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed27 import ED27Config, ED27Result, run_ed27, write_csv


def test_panel_a_quorum_threshold_tracks_membership():
    result = run_ed27(ED27Config())
    assert result.alone_blocked_at_full_rate == 1.0  # a lone leader is a minority of the cluster
    assert result.sole_member_commits_rate == 1.0  # ...but commits once it is the sole member
    assert result.regrow_reblocks_rate == 1.0  # add_replica raises the threshold and re-blocks it


def test_panel_b_restore_availability_and_safety_fence():
    result = run_ed27(ED27Config())
    assert result.stuck_before_removal is True  # 1 live of 3 members cannot commit
    assert result.restored_after_removal is True  # removing the 2 dead restores progress
    assert result.active_leader_remove_blocked is True  # the active leader cannot be removed


def test_tier_b_reproduces_every_transition():
    result = run_ed27(ED27Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed27(ED27Config())
    out = write_csv(result, tmp_path / "ed27.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("threshold,") for line in lines)
    assert any(line.startswith("availability,") for line in lines)


def test_config_round_trips():
    cfg = ED27Config.from_dict({"cluster_sizes": [3, 5], "val": "a"})
    assert cfg.cluster_sizes == (3, 5)
    assert isinstance(run_ed27(cfg), ED27Result)
