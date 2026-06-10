"""ED34 — the atomic counter: read-modify-write + the lost-update problem (DS0 increment 27).

The smoke instance of the DS0-increment-27 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — incr counts to k and is correct under every model with no
concurrency (Panel A); and under a partition eventual loses a concurrent increment while quorum
makes the minority unavailable and linearizable rejects the write (Panel B) — with Tier-B
reproducing every transition bit-for-bit. The committed figure is from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed34 import ED34Config, ED34Result, run_ed34, write_csv


def test_panel_a_sequential_correctness():
    result = run_ed34(ED34Config())
    assert result.seq_correct_rate == 1.0  # incr k times counts to k
    assert result.seq_correct_all_models is True  # correct under eventual/quorum/linearizable


def test_panel_b_read_modify_write_cap():
    result = run_ed34(ED34Config())
    assert result.eventual_lost_update is True  # two acked incrs, count short by one
    assert result.quorum_no_silent_loss is True  # minority unavailable, no silent loss
    assert result.linearizable_unavailable is True  # CP rejects under partition


def test_tier_b_reproduces_every_transition():
    result = run_ed34(ED34Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed34(ED34Config())
    out = write_csv(result, tmp_path / "ed34.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("sequential,") for line in lines)
    assert any(line.startswith("cap,") for line in lines)


def test_config_round_trips():
    cfg = ED34Config.from_dict({"cluster_sizes": [3], "key": "n", "k": 2})
    assert cfg.cluster_sizes == (3,) and cfg.k == 2
    assert isinstance(run_ed34(cfg), ED34Result)
