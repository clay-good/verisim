"""ED41 — the CRDT RGA: the first ordered CRDT, the basis of collaborative text (DS0 increment 34).

The smoke instance of the DS0-increment-34 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — sequential rins builds "abc", a middle insert and a delete
work, concurrent inserts at the same position converge to one deterministic order on every node, and
rins is always available (Panel A); and the union join converges every node to the same sequence via
gossip and anti_entropy, idempotently (Panel B) — with Tier-B reproducing every step. The committed
committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed41 import ED41Config, ED41Result, run_ed41, write_csv


def test_panel_a_sequence_ops_and_concurrent_convergence():
    result = run_ed41(ED41Config())
    assert result.build_rate == 1.0  # sequential rins builds "abc"
    assert result.insert_delete is True  # a middle insert and a delete both work
    assert result.concurrent_converges is True  # concurrent inserts converge to one order
    assert result.always_available is True  # a partitioned-minority rins is acknowledged (AP)


def test_panel_b_convergence():
    result = run_ed41(ED41Config())
    assert result.gossip_converges is True  # a gossip chain converges every node to the same seq
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the sequence unchanged


def test_tier_b_reproduces_every_transition():
    result = run_ed41(ED41Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed41(ED41Config())
    out = write_csv(result, tmp_path / "ed41.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("rga,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED41Config.from_dict({"cluster_sizes": [3], "list_name": "doc"})
    assert cfg.cluster_sizes == (3,) and cfg.list_name == "doc"
    assert isinstance(run_ed41(cfg), ED41Result)
