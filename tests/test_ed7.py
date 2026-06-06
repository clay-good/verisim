"""ED7 — the Tier-B system-oracle differential (SPEC-7 §5.2, DS8): the distributed W1 retirement.

The smoke instance of the apparatus (dependency-free, GPU-free): a tiny seeded run checking the four
findings have the right shape — Tier-A and Tier-B agree bit-for-bit on the observable cluster across
the exhaustive battery and the driver trajectories (residual zero), the H_ε(ρ) overlay is
oracle-invariant (gap zero at every ρ), the broken-arrival negative control is *caught*, and the
real-OS-thread attestation runs. The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed7 import ED7Config, run_ed7, write_csv


def _tiny() -> ED7Config:
    return ED7Config(
        battery_seeds=(0, 1, 2, 3),
        battery_depth=10,
        traj_seeds=(100, 101, 102),
        traj_steps=24,
        curve_seeds=(200, 201),
        curve_steps=20,
        rhos=(0.0, 0.5, 1.0),
    )


def test_run_ed7_is_well_formed():
    result = run_ed7(_tiny())
    assert result.tier1["n"] > 0
    assert {r["driver"] for r in result.tier2} == {"uniform", "contention", "adversarial"}
    assert {round(p["rho"], 3) for p in result.tier3} == {0.0, 0.5, 1.0}


def test_headline_agreement_is_bit_exact_and_residual_zero():
    result = run_ed7(_tiny())
    assert result.overall_agreement == 1.0
    assert result.residual_fraction == 0.0
    # every command family agrees totally
    for fam, d in result.tier1["by_family"].items():
        assert d["agree"] == d["n"], f"family {fam} disagrees"


def test_tier3_curve_is_oracle_invariant():
    result = run_ed7(_tiny())
    for p in result.tier3:
        assert p["gap"] == 0.0, f"ρ={p['rho']}: Tier-A and Tier-B horizons differ"


def test_negative_control_has_teeth():
    result = run_ed7(_tiny())
    nc = result.negative_control
    assert nc["detects_break"] is True
    assert nc["n_caught"] > 0
    assert nc["n_agree"] < nc["n"]


def test_threaded_attestation_runs():
    result = run_ed7(_tiny())
    th = result.threaded
    # real OS threads are available in the test environment; if not, the skip is disclosed (not a
    # silent pass) and we still assert the field is present.
    assert "available" in th
    if th["available"]:
        assert th["rate"] == 1.0


def test_write_csv_round_trips(tmp_path):
    result = run_ed7(_tiny())
    path = write_csv(result, tmp_path / "ed7.csv")
    text = path.read_text()
    assert "tier1,exhaustive" in text
    assert "tier3,overlay" in text
    assert "negctl,broken_arrival" in text
    assert "threaded,real_os_threads" in text
