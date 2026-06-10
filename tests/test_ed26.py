"""ED26 — Raft log replication: commit-on-majority + log-matching reconciliation (DS0 incr 19).

The smoke instance of the DS0-increment-19 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — a majority-reachable `append` commits while a minority-
stranded one stays uncommitted-but-retained (Panel A), and a deposed leader's uncommitted entry is
never applied to the KV and is overwritten by the higher-term leader's committed entry after heal,
leaving every live log identical (Panel B) — with Tier-B reproducing every transition bit-for-bit.
The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed26 import ED26Config, ED26Result, run_ed26, write_csv


def test_panel_a_commit_requires_a_majority():
    result = run_ed26(ED26Config())
    assert result.majority_commit_rate == 1.0  # a majority-reachable append commits
    assert result.minority_uncommitted_rate == 1.0  # a minority-stranded one stays uncommitted
    assert result.minority_entry_retained_rate == 1.0  # ...but is retained on the leader's log
    assert result.commit_index_monotone_rate == 1.0  # the commit index never moves backward


def test_panel_b_log_matching_reconciliation():
    result = run_ed26(ED26Config())
    assert result.uncommitted_not_applied is True  # uncommitted entries never reach the KV
    assert result.deposed_entry_overwritten is True  # the deposed leader's stale tail is replaced
    assert result.log_matching_after_heal is True  # all live nodes hold an identical log
    assert result.kv_reflects_committed_log is True  # the rejoined node's KV converges


def test_tier_b_reproduces_every_transition():
    result = run_ed26(ED26Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed26(ED26Config())
    out = write_csv(result, tmp_path / "ed26.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("commit,") for line in lines)
    assert any(line.startswith("reconcile,") for line in lines)


def test_config_round_trips():
    cfg = ED26Config.from_dict({"cluster_sizes": [3, 5], "v_a": "a", "v_b": "b"})
    assert cfg.cluster_sizes == (3, 5)
    assert isinstance(run_ed26(cfg), ED26Result)
