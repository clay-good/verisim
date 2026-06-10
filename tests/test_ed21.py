"""ED21 — clock skew: a per-node timing shift convergence is immune to (DS0 increment 14).

The smoke instance of the DS0-increment-14 apparatus (dependency-free, GPU-free): a tiny sweep that
checks the two findings have the right shape — clock skew shifts a node's send timing by exactly its
offset (Panel A) while the converged state is invariant to skew (Panel B, the version-LWW
clock-independence), with Tier-B reproducing the medium decision bit-for-bit. The committed figure
comes from the local run.
"""

from __future__ import annotations

from verisim.experiments.ed21 import ED21Config, ED21Result, run_ed21, write_csv


def test_panel_a_skew_shifts_send_timing_by_exactly_delta():
    result = run_ed21(ED21Config())
    assert result.shift_equals_delta is True  # every send shifted by exactly the skew offset
    by_skew = {t["skew"]: t for t in result.timing}
    # a negative/zero skew lands within a short advance; a large positive skew is deferred past it.
    assert by_skew[-4]["converged_after_short"] is True
    assert by_skew[0]["converged_after_short"] is True
    assert by_skew[4]["converged_after_short"] is False  # ahead-of-clock send deferred


def test_panel_b_convergence_is_clock_independent():
    result = run_ed21(ED21Config())
    assert result.converged_invariant_rate == 1.0  # version-LWW: skew never changes where it lands


def test_tier_b_reproduces_the_medium_decision():
    result = run_ed21(ED21Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed21(ED21Config())
    out = write_csv(result, tmp_path / "ed21.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("timing,") for line in lines)
    assert any(line.startswith("invariance,") for line in lines)


def test_config_round_trips():
    cfg = ED21Config.from_dict({"nodes": ["n0", "n1", "n2"], "skews": [-3, 0, 3]})
    assert cfg.nodes == ("n0", "n1", "n2")
    assert cfg.skews == (-3, 0, 3)
    assert isinstance(run_ed21(cfg), ED21Result)
