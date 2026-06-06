"""ED5 — consistency vs bit-faithful horizon (H19) + the competitive ratio (H18), SPEC-7 §12.

The smoke instance of the DS8 apparatus (dependency-free, GPU-free): a tiny seeded sweep checking
the two findings have the right qualitative shape — the ``subtle`` (in-flight) error class opens a
real consistency-vs-bit horizon gap (H19), and the loop's competitive ratio degrades gracefully
with prediction error and recovers the trivial bound for a perfect model (H18). The committed
figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed5 import ED5Config, ED5Result, run_ed5, write_csv


def _tiny() -> ED5Config:
    return ED5Config(
        eval_seeds=(100, 101, 102, 103),
        n_steps=16,
        h18_noises=(0.0, 0.5, 1.0),
        rhos=(0.0, 0.25, 0.5, 1.0),
    )


def test_run_ed5_is_well_formed():
    result = run_ed5(_tiny())
    # H19 has one cell per mode, each with both horizons and a gap CI
    assert {r["mode"] for r in result.h19} == {"gross", "subtle"}
    # H18 has one cell per (noise, ρ)
    assert len(result.h18) == 3 * 4
    assert {c["noise"] for c in result.h18} == {0.0, 0.5, 1.0}
    assert {c["rho"] for c in result.h18} == {0.0, 0.25, 0.5, 1.0}


def test_h19_subtle_opens_a_consistency_gap():
    result = run_ed5(_tiny())
    by_mode = {r["mode"]: r for r in result.h19}
    # the subtle (in-flight) error is bit-visible but consistency-invisible until delivery, so the
    # consistency-faithful horizon strictly outlasts the bit-faithful one...
    assert by_mode["subtle"]["cons_h"] > by_mode["subtle"]["bit_h"]
    # ...by materially more than the gross (durable-replica) control, where both break at once
    assert by_mode["subtle"]["gap"] > by_mode["gross"]["gap"]


def test_h18_competitive_ratio_degrades_gracefully():
    result = run_ed5(_tiny())
    v = result.h18_verdict
    assert v["ceiling"] == 16.0
    # a perfect model (zero prediction error) recovers the trivial bound: ratio 1 at any budget
    assert v["ratio_perfect_model"] == 1.0
    # a useless model buys far less horizon per dollar at the sub-linear quarter budget
    assert v["ratio_useless_model"] < v["ratio_perfect_model"]
    # full consultation reaches the ceiling regardless of prediction error (ratio 1 at ρ=1)
    for c in result.h18:
        if c["rho"] == 1.0:
            assert c["ratio"] == 1.0


def test_write_csv(tmp_path):
    result = run_ed5(_tiny())
    out = write_csv(result, tmp_path / "ed5.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("h19,") for line in lines)
    assert any(line.startswith("h18,") for line in lines)


def test_config_round_trips():
    cfg = ED5Config.from_dict({"h19_noise": 0.3, "driver": "uniform", "rhos": [0.0, 1.0]})
    assert cfg.h19_noise == 0.3
    assert cfg.driver == "uniform"
    assert cfg.rhos == (0.0, 1.0)
    assert isinstance(run_ed5(ED5Config(eval_seeds=(100,), n_steps=8, rhos=(1.0,),
                                        h18_noises=(0.0, 1.0))), ED5Result)
