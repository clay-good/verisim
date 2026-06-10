"""ED33 — the tombstone delete: versioned removal + the resurrection problem (DS0 increment 26).

The smoke instance of the DS0-increment-26 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — delete removes the key on every replica, the tombstone is a
real versioned write, and a newer put legitimately resurrects (Panel A); and under a partition the
minority still reads the deleted item while heal + anti_entropy / gossip converge it to deleted with
no resurrection (Panel B) — with Tier-B reproducing every transition bit-for-bit. The committed
figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed33 import ED33Config, ED33Result, run_ed33, write_csv


def test_panel_a_versioned_tombstone():
    result = run_ed33(ED33Config())
    assert result.delete_removes_rate == 1.0  # deleted on every replica
    assert result.tombstone_outversions_put is True  # the tombstone is a real versioned write
    assert result.newer_put_resurrects is True  # a higher-version put legitimately brings it back


def test_panel_b_resurrection_and_repair():
    result = run_ed33(ED33Config())
    assert result.minority_reads_deleted_item is True  # the danger: minority still reads the item
    assert result.anti_entropy_no_resurrection is True  # anti_entropy converges to deleted
    assert result.gossip_no_resurrection is True  # pairwise gossip converges to deleted


def test_tier_b_reproduces_every_transition():
    result = run_ed33(ED33Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed33(ED33Config())
    out = write_csv(result, tmp_path / "ed33.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("tombstone,") for line in lines)
    assert any(line.startswith("resurrection,") for line in lines)


def test_config_round_trips():
    cfg = ED33Config.from_dict({"cluster_sizes": [3], "key": "k", "val": "a", "val2": "b"})
    assert cfg.cluster_sizes == (3,) and cfg.key == "k"
    assert isinstance(run_ed33(cfg), ED33Result)
