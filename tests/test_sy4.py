"""SY4 -- the determinism attestation (SPEC-11 §2.4, §5; H30).

The gate that licenses calling SY1 *true* determinism: a fixed (state, action) battery is
bit-reproducible across repeats under the seal, and the determinism report is all-sealed.
"""

from __future__ import annotations

import pytest

from verisim.experiments.sy4 import run_sy4, write_csv
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell")


@requires_shell
def test_repeated_runs_are_bit_identical():
    result = run_sy4(n_repeats=5)
    assert result.available
    assert result.bit_identical, result.first_divergence
    assert result.n_transitions > 0


@requires_shell
def test_seal_is_all_sealed():
    result = run_sy4(n_repeats=3)
    assert result.seal_all_sealed  # v0 grammar reads no clock/RNG/threads -> total seal


@requires_shell
def test_write_csv(tmp_path):
    result = run_sy4(n_repeats=3)
    out = write_csv(result, tmp_path / "sy4.csv")
    text = out.read_text()
    assert "bit_identical,True" in text
    assert "seal_all_sealed,True" in text
