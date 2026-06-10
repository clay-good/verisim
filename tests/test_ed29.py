"""ED29 — the rolling upgrade: will this deploy break the cluster? (DS0 increment 22).

The smoke instance of the DS0-increment-22 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — a rolling upgrade that stays inside the
version-compatibility window keeps quorum at every step (Panel A), while a deploy creating a split
with no compatible majority loses quorum, and the same shape is safe within the window (Panel B) —
with Tier-B reproducing every transition bit-for-bit. The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed29 import ED29Config, ED29Result, run_ed29, write_csv


def test_panel_a_safe_rolling_upgrade_keeps_quorum():
    result = run_ed29(ED29Config())
    assert result.rolling_commit_rate == 1.0  # propose commits after every single-node bump
    assert result.n_steps > 0


def test_panel_b_break_and_diagnostic():
    result = run_ed29(ED29Config())
    assert result.incompatible_split_breaks is True  # spread 2 > skew 1, no majority -> no_quorum
    assert result.within_window_commits is True  # spread 1 -> compatible majority -> commits
    assert result.wider_window_commits is True  # same over-spread, skew 2 window -> commits


def test_tier_b_reproduces_every_transition():
    result = run_ed29(ED29Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed29(ED29Config())
    out = write_csv(result, tmp_path / "ed29.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("rolling,") for line in lines)
    assert any(line.startswith("break,") for line in lines)


def test_config_round_trips():
    cfg = ED29Config.from_dict({"cluster_sizes": [3, 4], "val": "a"})
    assert cfg.cluster_sizes == (3, 4)
    assert isinstance(run_ed29(cfg), ED29Result)
