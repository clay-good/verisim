"""ED31 — the config push: "will this config push break the cluster?" (DS0 increment 24).

The smoke instance of the DS0-increment-24 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — a `config_push` at the leader commits and reaches every
voting member, while a non-leader/no-leader push is fenced (Panel A); and under a partition a
minority-stranded leader cannot commit (no_quorum), a majority-side push commits but leaves the
partitioned minority with stale config (divergence), and a re-push after heal converges every node
(Panel B) — with Tier-B reproducing every transition bit-for-bit. The committed figure is local-run.
"""

from __future__ import annotations

from verisim.experiments.ed31 import ED31Config, ED31Result, run_ed31, write_csv


def test_panel_a_leader_committed_rollout_and_fence():
    result = run_ed31(ED31Config())
    assert result.commit_full_rate == 1.0  # a leader push reaches every voting member
    assert result.nonleader_fenced is True  # a non-leader push is not_leader
    assert result.noleader_fenced is True  # a push with no leader elected is rejected


def test_panel_b_partition_breaks_the_config():
    result = run_ed31(ED31Config())
    assert result.minority_no_quorum is True  # a minority-stranded leader cannot commit
    assert result.minority_stale_under_partition is True  # majority commits, minority stays stale
    assert result.repush_converges is True  # a re-push after heal converges every node


def test_tier_b_reproduces_every_transition():
    result = run_ed31(ED31Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed31(ED31Config())
    out = write_csv(result, tmp_path / "ed31.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("rollout,") for line in lines)
    assert any(line.startswith("partition,") for line in lines)


def test_config_round_trips():
    cfg = ED31Config.from_dict({"cluster_sizes": [3], "key": "flag", "val": "y"})
    assert cfg.cluster_sizes == (3,) and cfg.key == "flag"
    assert isinstance(run_ed31(cfg), ED31Result)
