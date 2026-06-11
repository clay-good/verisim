"""ED38 — the CRDT MV-register: conflict-surfacing, convergent, resolvable (DS0 increment 31).

The smoke instance of the DS0-increment-31 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — mvput reads back the value, a sequential overwrite resolves,
two concurrent writes both survive as siblings, and mvput is always available (Panel A); and the
union join converges every node to the same sibling set via gossip and anti_entropy, idempotently,
and a context-aware write resolves it (Panel B) — with Tier-B reproducing every step bit-for-bit.
The committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed38 import ED38Config, ED38Result, run_ed38, write_csv


def test_panel_a_conflict_surfaced():
    result = run_ed38(ED38Config())
    assert result.basic_read_rate == 1.0  # mvput then mvget reads back the single value
    assert result.sequential_resolves is True  # a sequential overwrite collapses to one value
    assert result.siblings_preserved is True  # two concurrent writes both survive (vs LWW loss)
    assert result.always_available is True  # a partitioned-minority mvput is acknowledged (AP)


def test_panel_b_convergence_and_resolution():
    result = run_ed38(ED38Config())
    assert result.gossip_converges is True  # a gossip chain converges every node to the sibling set
    assert result.anti_entropy_converges is True  # anti_entropy on each node converges every node
    assert result.idempotent is True  # a second gossip leaves the sibling set unchanged
    assert result.resolves_conflict is True  # a context-aware write collapses the siblings to one


def test_tier_b_reproduces_every_transition():
    result = run_ed38(ED38Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed38(ED38Config())
    out = write_csv(result, tmp_path / "ed38.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("mvreg,") for line in lines)
    assert any(line.startswith("converge,") for line in lines)


def test_config_round_trips():
    cfg = ED38Config.from_dict({"cluster_sizes": [3], "key": "z"})
    assert cfg.cluster_sizes == (3,) and cfg.key == "z"
    assert isinstance(run_ed38(cfg), ED38Result)
