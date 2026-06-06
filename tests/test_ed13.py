"""ED13 — causal consistency: the effect-before-cause anomaly, SPEC-7 §3.4 (DS0 incr 5).

The smoke instance of the DS0-increment-5 apparatus (dependency-free, GPU-free): a tiny battery
checking the two findings have the right shape — eventual admits the effect-before-cause anomaly
while causal forbids it (Panel A), and causal orders only causally-linked writes while still
converging (Panel B). The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed13 import ED13Config, ED13Result, run_ed13, write_csv


def _tiny() -> ED13Config:
    return ED13Config(object_pairs=((0, 1), (1, 2), (0, 2)))


def test_run_ed13_is_well_formed():
    result = run_ed13(_tiny())
    assert {r["model"] for r in result.anomaly} == {"eventual", "causal"}
    assert result.over_sync["scenarios"] == 3
    assert result.convergence["scenarios"] == 3


def test_panel_a_eventual_admits_causal_forbids():
    result = run_ed13(_tiny())
    by_model = {r["model"]: r for r in result.anomaly}
    # eventual delivers greedily → the observer sees the effect before the cause every time
    assert by_model["eventual"]["anomaly_rate"] == 1.0
    # causal holds the dependent message → the anomaly cannot occur
    assert by_model["causal"]["anomaly_rate"] == 0.0


def test_panel_b_orders_only_causal_links_and_converges():
    result = run_ed13(_tiny())
    o = result.over_sync
    # causal holds the dependent message but never the independent one
    assert o["dependent_held_rate"] == 1.0
    assert o["independent_held_rate"] == 0.0
    c = result.convergence
    # convergence preserved: eventual and causal reach the identical durable state, in-flight drains
    assert c["identical_final_state_rate"] == 1.0
    assert c["causal_inflight_after_heal"] == 0


def test_write_csv(tmp_path):
    result = run_ed13(_tiny())
    out = write_csv(result, tmp_path / "ed13.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("anomaly,") for line in lines)
    assert any(line.startswith("convergence,") for line in lines)


def test_config_round_trips():
    cfg = ED13Config.from_dict({"n_objects": 3, "object_pairs": [[0, 1], [1, 2]]})
    assert cfg.n_objects == 3
    assert cfg.object_pairs == ((0, 1), (1, 2))
    assert isinstance(run_ed13(_tiny()), ED13Result)
