"""ED40 — the CRDT OR-Map: a CRDT of CRDTs, the compositional capstone (DS0 increment 33).

The smoke instance of the DS0-increment-33 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — mput/mget/mkeys read the field+value, mdel removes a field;
a concurrent value resolves by LWW, a concurrent mput survives a concurrent mdel (add-wins), mput is
always available (Panel A); and the composed join converges every node to the same fields/values via
gossip and anti_entropy, idempotently (Panel B) — with Tier-B reproducing every step. The committed
committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed40 import ED40Config, ED40Result, run_ed40, write_csv


def test_panel_a_map_ops_and_composition():
    result = run_ed40(ED40Config())
    assert result.basic_read_rate == 1.0  # mput then mget/mkeys reads the field+value back
    assert result.delete_removes_field is True  # mdel removes the field (absent from mkeys/mget)
    assert result.value_resolves_lww is True  # a concurrent value update resolves by LWW (1 winner)
    assert result.add_wins_presence is True  # a concurrent mput survives a concurrent mdel
    assert result.always_available is True  # a partitioned-minority op is acknowledged (AP)


def test_panel_b_convergence():
    result = run_ed40(ED40Config())
    assert result.gossip_converges is True  # converges every node to the same fields + values
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the map unchanged


def test_tier_b_reproduces_every_transition():
    result = run_ed40(ED40Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed40(ED40Config())
    out = write_csv(result, tmp_path / "ed40.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("ormap,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED40Config.from_dict({"cluster_sizes": [3], "mapname": "cfg"})
    assert cfg.cluster_sizes == (3,) and cfg.mapname == "cfg"
    assert isinstance(run_ed40(cfg), ED40Result)
