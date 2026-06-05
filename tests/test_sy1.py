"""SY1 -- the differential agreement table + the head-to-head curve overlay (SPEC-11 §3, §5; H27).

The smoke instance of the apparatus that retires W1: a tiny seeded sweep asserting the
platform-independent headline holds -- structure-building agreement is bit-exact 1.000, the
residual is 0, and the reference-vs-system H_ε(ρ) curves are indistinguishable.
"""

from __future__ import annotations

import pytest

from verisim.experiments.sy1 import SY1Config, run_sy1, write_csv
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell")


def _tiny() -> SY1Config:
    return SY1Config(
        battery_seeds=(0, 1),
        battery_depth=4,
        traj_seeds=(100, 101),
        traj_steps=20,
        curve_seeds=(200,),
        curve_steps=12,
        rhos=(0.0, 0.5, 1.0),
    )


@requires_shell
def test_headline_structure_agreement_is_one():
    result = run_sy1(_tiny())
    assert result.available
    assert result.modeled_agreement == 1.0  # bit-exact on the modeled regime


@requires_shell
def test_residual_is_zero():
    result = run_sy1(_tiny())
    assert result.residual_fraction == 0.0  # every divergence is a named boundary


@requires_shell
def test_tier3_curve_overlay_is_oracle_invariant():
    result = run_sy1(_tiny())
    # reference-verified and system-verified H_ε(ρ) coincide on the structural grammar
    assert max(p["gap"] for p in result.tier3) == 0.0
    # the curve rises from a floor (ρ=0) toward the ceiling (ρ=1)
    assert result.tier3[0]["h_ref"] <= result.tier3[-1]["h_ref"]


@requires_shell
def test_tier1_modeled_families_all_agree():
    result = run_sy1(_tiny())
    by = result.tier1["by_family"]
    for fam in ("mkdir", "touch", "write", "append", "cat", "ls", "cd", "export"):
        if fam in by and by[fam]["n"]:
            assert by[fam]["agree"] == by[fam]["n"], fam


@requires_shell
def test_write_csv(tmp_path):
    result = run_sy1(_tiny())
    out = write_csv(result, tmp_path / "sy1.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("tier1,") for line in lines)
    assert any(line.startswith("tier2,") for line in lines)
    assert any(line.startswith("tier3,") for line in lines)


def test_config_round_trips():
    cfg = SY1Config.from_dict({"traj_steps": 10, "rhos": [0.0, 1.0]})
    assert cfg.traj_steps == 10
    assert cfg.rhos == (0.0, 1.0)
