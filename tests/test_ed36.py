"""ED36 — the CRDT PN-counter: decrementable, loss-free, convergent (DS0 increment 29).

The smoke instance of the DS0-increment-29 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — cdecr nets to k-m, may go negative, is always available,
and loses no concurrent inc/dec across a partition (Panel A); and the CRDT join over both halves
converges every node to the net total via gossip and anti_entropy, idempotently (Panel B) — with
Tier-B reproducing every step bit-for-bit. The committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed36 import ED36Config, ED36Result, run_ed36, write_csv


def test_panel_a_decrement_loss_free_and_negative():
    result = run_ed36(ED36Config())
    assert result.net_correct_rate == 1.0  # k cincrs then m cdecrs net to k-m
    assert result.goes_negative is True  # a fresh cdecr reads -1 (a G-counter cannot go below zero)
    assert result.always_available is True  # a partitioned-minority cdecr is acknowledged (AP)
    assert result.no_lost_update is True  # +2 (majority) - 1 (minority) net 1 — both halves merged


def test_panel_b_convergence():
    result = run_ed36(ED36Config())
    assert result.gossip_converges is True  # a gossip chain converges every node to the net
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the value unchanged


def test_tier_b_reproduces_every_transition():
    result = run_ed36(ED36Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed36(ED36Config())
    out = write_csv(result, tmp_path / "ed36.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("pncounter,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED36Config.from_dict({"cluster_sizes": [3], "key": "n", "k": 4, "m": 2})
    assert cfg.cluster_sizes == (3,) and cfg.k == 4 and cfg.m == 2
    assert isinstance(run_ed36(cfg), ED36Result)
