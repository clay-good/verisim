"""ED15 — optimistic (OCC) vs pessimistic (2PL) concurrency control, SPEC-7 §3.2 (DS0 incr 8).

The smoke instance of the DS0-increment-8 apparatus (dependency-free, GPU-free): a tiny sweep that
checks the two findings have the right shape — OCC wastes more work per abort than 2PL (it validates
late; 2PL fails fast at the lock), both forbid write skew (serializable), and Tier-B agrees.
The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed15 import ED15Config, ED15Result, run_ed15, write_csv


def _tiny() -> ED15Config:
    return ED15Config(seeds=(0, 1, 2, 3, 4), skew_object_pairs=((0, 1), (1, 2)))


def test_panel_a_occ_wastes_more_work_than_2pl():
    result = run_ed15(_tiny())
    by_cc = {r["cc"]: r for r in result.wasted}
    # OCC validates at commit, so an aborted txn completed all its (3) data ops; 2PL fails earlier.
    assert by_cc["occ"]["wasted_ops_per_abort"] > by_cc["2pl"]["wasted_ops_per_abort"]


def test_panel_b_both_forbid_write_skew():
    result = run_ed15(_tiny())
    by_cc = {r["cc"]: r for r in result.write_skew}
    assert by_cc["occ"]["anomaly_rate"] == 0.0   # OCC-serializable forbids write skew
    assert by_cc["2pl"]["anomaly_rate"] == 0.0    # 2PL forbids it too (read S-locks block writes)


def test_tier_b_reproduces_both_schemes():
    result = run_ed15(_tiny())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed15(_tiny())
    out = write_csv(result, tmp_path / "ed15.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("wasted,") for line in lines)
    assert any(line.startswith("write_skew,") for line in lines)


def test_config_round_trips():
    cfg = ED15Config.from_dict({"n_txns": 6, "n_objects": 3})
    assert cfg.n_txns == 6
    assert cfg.n_objects == 3
    assert isinstance(run_ed15(_tiny()), ED15Result)
