"""ED18 — message loss: the broken-convergence anomaly (SPEC-7 §3.2, DS0 increment 11).

The smoke instance of the DS0-increment-11 apparatus (dependency-free, GPU-free): a tiny sweep that
checks the two findings have the right shape — drop breaks convergence where partition recovers
(Panel A), and only a newer write heals a dropped write while the lost value is never observed
(Panel B), with Tier-B reproducing the medium decision bit-for-bit. The committed figure comes from
the local run.
"""

from __future__ import annotations

from verisim.experiments.ed18 import ED18Config, ED18Result, run_ed18, write_csv


def test_panel_a_drop_breaks_convergence_partition_does_not():
    result = run_ed18(ED18Config())
    by_regime = {r["regime"]: r for r in result.convergence}
    assert by_regime["partition"]["rate"] == 1.0  # held message delivered on heal
    assert by_regime["drop"]["rate"] == 0.0  # destroyed message — permanently stale


def test_panel_b_only_a_newer_write_heals_a_dropped_write():
    result = run_ed18(ED18Config())
    assert result.drop_heal_only_rate == 0.0  # heal+advance alone never repairs a dropped write
    assert result.drop_overwrite_rate == 1.0  # a newer write (higher version) overwrites the stale
    assert result.lost_value_never_observed is True  # the dropped value is lost, not delayed


def test_tier_b_reproduces_the_medium_decision():
    result = run_ed18(ED18Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed18(ED18Config())
    out = write_csv(result, tmp_path / "ed18.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("convergence,") for line in lines)
    assert any(line.startswith("recovery,") for line in lines)


def test_config_round_trips():
    cfg = ED18Config.from_dict({"nodes": ["n0", "n1", "n2"], "v1": "b", "v2": "c"})
    assert cfg.nodes == ("n0", "n1", "n2")
    assert isinstance(run_ed18(cfg), ED18Result)
