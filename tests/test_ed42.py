"""ED42 — the nested CRDT counter-map: a CRDT of CRDTs (DS0 increment 35).

The smoke instance of the DS0-increment-35 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — cminc builds per-field totals, cmdel removes a field,
concurrent increments to a field are summed loss-free, a concurrent cminc survives a concurrent del
(add-wins), and cminc is always available (Panel A); and the composed join converges every node to
same fields + totals via gossip and anti_entropy, idempotently (Panel B) — with Tier-B reproducing
every step bit-for-bit. The committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed42 import ED42Config, ED42Result, run_ed42, write_csv


def test_panel_a_map_ops_and_both_guarantees():
    result = run_ed42(ED42Config())
    assert result.basic_read_rate == 1.0  # cminc builds per-field totals that cmget/cmkeys read
    assert result.delete_removes_field is True  # cmdel removes the field (absent from cmkeys/cmget)
    assert result.value_loss_free is True  # concurrent increments to a field summed loss-free
    assert result.add_wins_presence is True  # a concurrent cminc survives a concurrent cmdel
    assert result.always_available is True  # a partitioned-minority cminc is acknowledged (AP)


def test_panel_b_convergence():
    result = run_ed42(ED42Config())
    assert result.gossip_converges is True  # converges every node to the same fields + totals
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the map unchanged


def test_tier_b_reproduces_every_transition():
    result = run_ed42(ED42Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed42(ED42Config())
    out = write_csv(result, tmp_path / "ed42.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("cmap,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED42Config.from_dict({"cluster_sizes": [3], "mapname": "cm"})
    assert cfg.cluster_sizes == (3,) and cfg.mapname == "cm"
    assert isinstance(run_ed42(cfg), ED42Result)
