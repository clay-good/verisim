"""ED12 — partial observation: the probe-faithful horizon + indistinguishability, SPEC-7 §5.4.

The smoke instance of the DS3-increment-4 apparatus (dependency-free, GPU-free): a tiny seeded
sweep checking the two findings have the right qualitative shape — the ``subtle`` (in-flight) error
class opens a real *observable*-vs-bit horizon gap because no probe can read the replication medium
(Panel A), and a single vantage cannot tell a crash from a partition while a paired one can (Panel
B). The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed12 import ED12Config, ED12Result, run_ed12, write_csv


def _tiny() -> ED12Config:
    return ED12Config(
        eval_seeds=(200, 201, 202, 203),
        n_steps=16,
        battery_seeds=(300, 301, 302, 303),
        battery_prefix=5,
    )


def test_run_ed12_is_well_formed():
    result = run_ed12(_tiny())
    assert {r["mode"] for r in result.horizons} == {"gross", "subtle"}
    for r in result.horizons:
        assert r["bit_h"] <= r["obs_h"] + 1e-9  # observable dominates bit (the structural half)
        assert "cons_h" in r and "gap" in r
    assert result.battery_n == 4


def test_panel_a_subtle_opens_a_probe_gap():
    result = run_ed12(_tiny())
    by_mode = {r["mode"]: r for r in result.horizons}
    # the subtle (in-flight) error is bit-visible but probe-invisible until delivery, so the
    # observable-faithful horizon strictly outlasts the bit-faithful one...
    assert by_mode["subtle"]["obs_h"] > by_mode["subtle"]["bit_h"]
    # ...by materially more than the gross (durable-replica) control, where the probe sees it at
    # once
    assert by_mode["subtle"]["gap"] > by_mode["gross"]["gap"]
    # the structural dominance held on every rollout
    assert result.observable_dominates_bit


def test_panel_b_one_probe_cannot_localize_two_can():
    result = run_ed12(_tiny())
    # from a single external vantage a crash and a partition are indistinguishable...
    assert result.single_vantage_indistinguishable == 1.0
    # ...and a paired vantage that reaches the node's side always separates them
    assert result.paired_vantage_indistinguishable == 0.0


def test_write_csv(tmp_path):
    result = run_ed12(_tiny())
    out = write_csv(result, tmp_path / "ed12.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("horizon,") for line in lines)
    assert any(line.startswith("indist,") for line in lines)


def test_config_round_trips():
    cfg = ED12Config.from_dict({"noise": 0.3, "driver": "uniform", "battery_prefix": 4})
    assert cfg.noise == 0.3
    assert cfg.driver == "uniform"
    assert cfg.battery_prefix == 4
    assert isinstance(run_ed12(_tiny()), ED12Result)
