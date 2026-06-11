"""ED39 — the CRDT LWW-register: deterministic, Lamport-ordered, convergent (DS0 increment 32).

The smoke instance of the DS0-increment-32 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — lwwput reads back the value, a causally-later write wins
regardless of node id, concurrent writes resolve to one value, and lwwput is always available (Panel
A); and the max-by-timestamp join converges every node to the single winner via gossip and
anti_entropy, idempotently, dropping the concurrent loser (Panel B) — with Tier-B reproducing every
step bit-for-bit. The committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed39 import ED39Config, ED39Result, run_ed39, write_csv


def test_panel_a_deterministic_lww():
    result = run_ed39(ED39Config())
    assert result.basic_read_rate == 1.0  # lwwput then lwwget reads back the value
    assert result.causal_lww is True  # a happens-after write wins regardless of node id
    assert result.deterministic_resolve is True  # concurrent writes resolve to one value everywhere
    assert result.always_available is True  # a partitioned-minority lwwput is acknowledged (AP)


def test_panel_b_convergence():
    result = run_ed39(ED39Config())
    assert result.gossip_converges is True  # a gossip chain converges every node to the winner
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the value unchanged
    assert result.loser_dropped is True  # the concurrent loser is dropped (one value, not siblings)


def test_tier_b_reproduces_every_transition():
    result = run_ed39(ED39Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed39(ED39Config())
    out = write_csv(result, tmp_path / "ed39.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("lwwreg,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED39Config.from_dict({"cluster_sizes": [3], "key": "q"})
    assert cfg.cluster_sizes == (3,) and cfg.key == "q"
    assert isinstance(run_ed39(cfg), ED39Result)
