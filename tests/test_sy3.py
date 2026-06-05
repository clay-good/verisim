"""SY3 -- the hermeticity proof (SPEC-11 §2.3, §5; H29).

The safety gate: every prohibited action is denied and the benign positive control is
allowed-then-discarded. These controls are kernel-feature-free, so they pass on the macOS
development host and on Linux CI alike.
"""

from __future__ import annotations

import pytest

from verisim.experiments.sy3 import run_sy3, write_csv
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell")


@requires_shell
def test_every_hermeticity_control_passes():
    result = run_sy3()
    assert result.available
    assert result.all_passed, [c for c in result.controls if not c.passed]


@requires_shell
def test_all_channels_present():
    result = run_sy3()
    channels = {c.channel for c in result.controls}
    expected = {"filesystem", "network", "privilege", "persistence", "resource", "positive-control"}
    assert expected <= channels


@requires_shell
def test_write_csv(tmp_path):
    result = run_sy3()
    out = write_csv(result, tmp_path / "sy3.csv")
    text = out.read_text()
    assert text.startswith("channel,control,kind,passed,detail")
    assert "FAIL" not in text
