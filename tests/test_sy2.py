"""SY2 -- the differential debugger (SPEC-11 §5; H28).

Asserts the two platform-independent guarantees: (1) the teeth control passes -- the harness
detects and localizes a *planted* synthetic divergence -- and (2) every atlas entry is a named
boundary class. The exact atlas membership is platform-stamped (``self_subtree`` appears on
macOS/BSD but agrees on Linux/GNU), so the test checks the subset relation, not equality.
"""

from __future__ import annotations

from verisim.experiments.sy2 import run_sy2, run_teeth_control, write_csv
from verisim.oracle.differential import BOUNDARY_CLASSES
from verisim.oracle.reference import ReferenceOracle


def test_teeth_control_detects_a_planted_divergence():
    passed, detail = run_teeth_control(ReferenceOracle(), corrupt_command="mkdir")
    assert passed, detail


def test_teeth_control_no_false_positive_when_nothing_is_corrupted():
    # corrupting a command the structural driver never emits => no detections, no false positives
    passed, detail = run_teeth_control(ReferenceOracle(), corrupt_command="chmod")
    assert not passed  # nothing detected (the bug is never triggered), but also no false positives
    assert "false positives=False" in detail


def test_atlas_entries_are_named_boundaries():
    result = run_sy2(seeds=(0, 1, 2, 3), steps=30)
    assert result.teeth_passed
    for d in result.atlas:
        assert d.divergence_class in BOUNDARY_CLASSES
        assert d.action_raw  # a concrete minimal reproducer


def test_write_csv(tmp_path):
    result = run_sy2(seeds=(0, 1), steps=20)
    out = write_csv(result, tmp_path / "sy2.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("divergence_class,")
    assert any("teeth_control" in line for line in lines)
