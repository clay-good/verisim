"""Smoke + structural-invariant tests for PB-transfer-broad (SPEC-18 §6, the boundary arm).

PB-transfer-broad needs the SPEC-11 system oracle (real shell) and is ``skipif``-guarded — a skip is
never counted as a result (SPEC-11 §2.5). When the shell is present it asserts *structural*
invariants that hold cross-platform (a row per grammar × ρ; the validated structural grammar is
near-lossless; ΔH values are finite; the run is deterministic), not the exact boundary magnitudes —
those are the committed figure on the primary host (BSD coreutils differ from GNU, so broad-grammar
ΔH is host-specific).
"""

from __future__ import annotations

from verisim.experiments.pb_transfer import system_oracle_available
from verisim.experiments.pb_transfer_broad import PBTransferBroadConfig, run_pb_transfer_broad


def _tiny() -> PBTransferBroadConfig:
    return PBTransferBroadConfig(
        grammars=("structural", "weighted", "adversarial"), rhos=(0.0, 0.5), n_steps=16, n_seeds=4,
    )


def test_pb_transfer_broad_maps_the_boundary_or_skips() -> None:
    if not system_oracle_available():
        import pytest

        pytest.skip("system oracle (real shell) unavailable — boundary not counted (§2.5)")
    stats = run_pb_transfer_broad(_tiny())
    assert stats  # measurable when the shell is present
    cells = {(s.grammar, s.rho) for s in stats}
    for g in ("structural", "weighted", "adversarial"):
        for rho in (0.0, 0.5):
            assert (g, rho) in cells
    for s in stats:
        assert s.dh_lo <= s.delta_h <= s.dh_hi
        assert -50.0 < s.delta_h < 50.0  # finite, sane
    # the validated structure grammar is (near-)lossless on any POSIX shell (the PB-transfer anchor)
    structural0 = next(s for s in stats if s.grammar == "structural" and s.rho == 0.0)
    assert abs(structural0.delta_h) <= 1.0


def test_pb_transfer_broad_deterministic() -> None:
    if not system_oracle_available():
        import pytest

        pytest.skip("system oracle (real shell) unavailable — boundary not counted (§2.5)")
    a = [(s.grammar, s.rho, round(s.delta_h, 4)) for s in run_pb_transfer_broad(_tiny())]
    b = [(s.grammar, s.rho, round(s.delta_h, 4)) for s in run_pb_transfer_broad(_tiny())]
    assert a == b


def test_pb_transfer_broad_skips_cleanly_when_unavailable() -> None:
    # Returns a list either way (empty + uncounted when the oracle is absent); never raises.
    stats = run_pb_transfer_broad(PBTransferBroadConfig(rhos=(0.0,), n_steps=8, n_seeds=2))
    assert isinstance(stats, list)
