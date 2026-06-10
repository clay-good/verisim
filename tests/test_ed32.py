"""ED32 — the quorum-confirmed linearizable read: Raft ReadIndex (DS0 increment 25).

The smoke instance of the DS0-increment-25 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — read_index is served at the leader, fenced for a non-leader,
and `no_quorum` for a minority leader, with the lease/quorum contrast (Panel A); and read_index
reflects the committed value while a deposed leader's read_index is fenced where a plain get serves
stale (Panel B) — with Tier-B reproducing every read verdict bit-for-bit. The committed figure is
from a local run.
"""

from __future__ import annotations

from verisim.experiments.ed32 import ED32Config, ED32Result, run_ed32, write_csv


def test_panel_a_two_reads_opposite_availability():
    result = run_ed32(ED32Config())
    assert result.read_index_ok_rate == 1.0  # served at the leader with quorum
    assert result.nonleader_fenced is True  # a non-leader read_index is not_leader
    assert result.minority_no_quorum is True  # a minority leader cannot confirm leadership
    assert result.lease_serves_where_quorum_refuses is True  # lread ok where read_index no_quorum


def test_panel_b_linearizable_safety():
    result = run_ed32(ED32Config())
    assert result.reflects_committed is True  # read_index returns the committed value
    assert result.deposed_read_index_fenced is True  # a deposed leader's read_index is not_leader
    assert result.stale_get_vs_safe_read_index is True  # get serves stale where read_index refuses


def test_tier_b_reproduces_every_read_verdict():
    result = run_ed32(ED32Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed32(ED32Config())
    out = write_csv(result, tmp_path / "ed32.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("reads,") for line in lines)
    assert any(line.startswith("safety,") for line in lines)


def test_config_round_trips():
    cfg = ED32Config.from_dict({"cluster_sizes": [3], "key": "k", "val": "b"})
    assert cfg.cluster_sizes == (3,) and cfg.key == "k"
    assert isinstance(run_ed32(cfg), ED32Result)
