"""ED35 — the CRDT G-counter: loss-free, always-available, convergent (DS0 increment 28).

The smoke instance of the DS0-increment-28 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — cincr counts to k, is always available (AP), and loses no
concurrent increment across a partition (Panel A); and the CRDT join converges every node to the
full total via gossip and anti_entropy, idempotently (Panel B) — with Tier-B reproducing every step
bit-for-bit. The committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed35 import ED35Config, ED35Result, run_ed35, write_csv


def test_panel_a_loss_free_and_available():
    result = run_ed35(ED35Config())
    assert result.seq_correct_rate == 1.0  # cincr k times reads back k
    assert result.always_available is True  # a partitioned-minority cincr is acknowledged (AP)
    assert result.no_lost_update is True  # three concurrent increments total 3 (ED34's LWW read 2)


def test_panel_b_convergence():
    result = run_ed35(ED35Config())
    assert result.gossip_converges is True  # a gossip chain converges every node
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the count unchanged


def test_tier_b_reproduces_every_transition():
    result = run_ed35(ED35Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed35(ED35Config())
    out = write_csv(result, tmp_path / "ed35.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("crdt,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED35Config.from_dict({"cluster_sizes": [3], "key": "n", "k": 2})
    assert cfg.cluster_sizes == (3,) and cfg.k == 2
    assert isinstance(run_ed35(cfg), ED35Result)
