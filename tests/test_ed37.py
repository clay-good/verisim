"""ED37 — the CRDT OR-Set: add-wins, re-addable, convergent (DS0 increment 30).

The smoke instance of the DS0-increment-30 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — sadd reads back all elements, a removed item is re-addable,
a concurrent add survives a concurrent remove (add-wins), and sadd is always available (Panel A);
the CRDT union join over both halves converges every node to the same set via gossip and
anti_entropy, idempotently (Panel B) — with Tier-B reproducing every step bit-for-bit. The committed
figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed37 import ED37Config, ED37Result, run_ed37, write_csv


def test_panel_a_orset_wins():
    result = run_ed37(ED37Config())
    assert result.adds_read_rate == 1.0  # sadd k distinct elems reads back all k
    assert result.re_addable is True  # srem then sadd brings the element back (a 2P-Set cannot)
    assert result.add_wins is True  # a concurrent add survives a concurrent remove (vs 2P-Set)
    assert result.always_available is True  # a partitioned-minority sadd is acknowledged (AP)


def test_panel_b_convergence():
    result = run_ed37(ED37Config())
    assert result.gossip_converges is True  # a gossip chain converges every node to the same set
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the set unchanged


def test_tier_b_reproduces_every_transition():
    result = run_ed37(ED37Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed37(ED37Config())
    out = write_csv(result, tmp_path / "ed37.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("orset,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED37Config.from_dict({"cluster_sizes": [3], "key": "t", "k": 2})
    assert cfg.cluster_sizes == (3,) and cfg.k == 2
    assert isinstance(run_ed37(cfg), ED37Result)
