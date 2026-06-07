"""ED19 — anti-entropy / read-repair: convergence restored after message loss (SPEC-7 §5.1, DS0 12).

The smoke instance of the DS0-increment-12 apparatus (dependency-free, GPU-free): a tiny sweep that
checks the two findings have the right shape — anti-entropy repairs a dropped write where advance
cannot (Panel A), and the same repair is bounded by reachability (Panel B) — with Tier-B reproducing
the read-repair bit-for-bit. The committed figure comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed19 import ED19Config, ED19Result, run_ed19, write_csv


def test_panel_a_anti_entropy_repairs_where_advance_cannot():
    result = run_ed19(ED19Config())
    by_regime = {r["regime"]: r for r in result.convergence}
    assert by_regime["advance_only"]["rate"] == 0.0  # no in-flight message remains to deliver
    assert by_regime["anti_entropy"]["rate"] == 1.0  # read-repair pulls the latest directly


def test_panel_b_anti_entropy_is_bounded_by_reachability():
    result = run_ed19(ED19Config())
    assert result.bounded_rate == 0.0    # partitioned away: cannot cross the split
    assert result.reachable_rate == 1.0  # after heal: reaches every replica and repairs


def test_tier_b_reproduces_the_read_repair():
    result = run_ed19(ED19Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed19(ED19Config())
    out = write_csv(result, tmp_path / "ed19.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("convergence,") for line in lines)
    assert any(line.startswith("reachability,") for line in lines)


def test_config_round_trips():
    cfg = ED19Config.from_dict({"nodes": ["n0", "n1", "n2"], "key": "x", "v1": "b"})
    assert cfg.nodes == ("n0", "n1", "n2")
    assert isinstance(run_ed19(cfg), ED19Result)
