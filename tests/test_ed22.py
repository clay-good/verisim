"""ED22 — pairwise gossip: bidirectional anti-entropy + epidemic convergence (DS0 increment 15).

The smoke instance of the DS0-increment-15 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — one pairwise `gossip a b` reconciles BOTH endpoints where
one `anti_entropy a` fixes only the named node (Panel A), and a chain of pairwise gossips converges
the
whole reachable component while a partition blocks it (Panel B) — with Tier-B reproducing the
reconciliation bit-for-bit. The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed22 import ED22Config, ED22Result, run_ed22, write_csv


def test_panel_a_gossip_reconciles_both_anti_entropy_only_one():
    result = run_ed22(ED22Config())
    assert result.gossip_reconciles_both is True  # one pairwise gossip fills both endpoints' gaps
    assert result.anti_entropy_reconciles_one is True  # one anti-entropy fills only the named node


def test_panel_b_epidemic_convergence_bounded_by_reachability():
    result = run_ed22(ED22Config())
    assert result.epidemic_converged_rate == 1.0  # a chain of pairwise gossips converges everyone
    assert result.partition_blocks_epidemic is True  # ...except a node partitioned off the chain


def test_tier_b_reproduces_the_reconciliation():
    result = run_ed22(ED22Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed22(ED22Config())
    out = write_csv(result, tmp_path / "ed22.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("bidirectional,") for line in lines)
    assert any(line.startswith("epidemic,") for line in lines)


def test_config_round_trips():
    cfg = ED22Config.from_dict({"nodes": ["n0", "n1", "n2"], "vx": "b", "vy": "c"})
    assert cfg.nodes == ("n0", "n1", "n2")
    assert isinstance(run_ed22(cfg), ED22Result)
