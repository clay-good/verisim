"""ED20 — message timing: recoverable delay + reorder-invariant convergence (DS0 increment 13).

The smoke instance of the DS0-increment-13 apparatus (dependency-free, GPU-free): a tiny sweep that
checks the two findings have the right shape — delay is recoverable where drop is not (Panel A), and
reorder flips the in-transit observation while last-writer-wins keeps the converged value invariant
(Panel B), with Tier-B reproducing the medium decision bit-for-bit. The committed figure comes from
the local run.
"""

from __future__ import annotations

from verisim.experiments.ed20 import ED20Config, ED20Result, run_ed20, write_csv


def test_panel_a_delay_is_recoverable_where_drop_is_not():
    result = run_ed20(ED20Config())
    by_regime = {r["regime"]: r for r in result.convergence}
    assert by_regime["delay"]["rate"] == 1.0  # deferred message still arrives — recoverable
    assert by_regime["drop"]["rate"] == 0.0  # destroyed message — permanently stale
    assert result.delay_stale_then_converges is True  # stale before the deferral, converged after


def test_panel_b_reorder_flips_transit_but_not_the_converged_value():
    result = run_ed20(ED20Config())
    assert result.reorder_transit_flip_rate == 1.0  # reorder changes what the peer sees in flight
    assert result.reorder_converged_invariant_rate == 1.0  # ...but never where the cluster lands


def test_tier_b_reproduces_the_medium_decision():
    result = run_ed20(ED20Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed20(ED20Config())
    out = write_csv(result, tmp_path / "ed20.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("convergence,") for line in lines)
    assert any(line.startswith("reorder,") for line in lines)


def test_config_round_trips():
    cfg = ED20Config.from_dict({"nodes": ["n0", "n1", "n2"], "delay_dt": 5, "stagger": 50})
    assert cfg.nodes == ("n0", "n1", "n2")
    assert cfg.stagger == 50
    assert isinstance(run_ed20(cfg), ED20Result)
