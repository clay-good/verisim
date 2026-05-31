"""Tests for coverage-balanced synthesis analysis (SPEC-2.1 §5 / K1)."""

from verisim.data.coverage import (
    CoverageReport,
    coverage_report,
    missing_commands,
    transition_cell,
)
from verisim.env.action import parse_action
from verisim.env.config import DEFAULT_CONFIG
from verisim.oracle.reference import ReferenceOracle


def test_transition_cell_format():
    assert transition_cell(parse_action("mkdir /a"), 0) == "mkdir:ok"
    assert transition_cell(parse_action("rmdir /a"), 1) == "rmdir:fail"
    assert transition_cell(parse_action("rm -r /a"), 0) == "rm-r:ok"


def test_coverage_spans_commands_and_failures():
    oracle = ReferenceOracle()
    report = coverage_report(
        oracle,
        DEFAULT_CONFIG,
        drivers=("weighted", "adversarial"),
        seeds=(0, 1, 2, 3),
        n_steps=40,
    )
    # every base command shows up across the driver mix.
    required = {
        "mkdir", "rmdir", "touch", "write", "append", "cat", "ls", "rm",
        "mv", "cp", "chmod", "cd", "export",
    }
    assert missing_commands(report, required) == set()
    # the adversarial driver provokes real failures (the K0-flagged coverage gap).
    assert report.n_failures > 0
    # structural creates land at a range of depths (the copy-distribution axis).
    assert report.create_depths
    assert max(report.create_depths) >= 1


def test_coverage_report_deterministic():
    oracle = ReferenceOracle()
    a = coverage_report(oracle, DEFAULT_CONFIG, ("weighted",), (0, 1), 20)
    b = coverage_report(oracle, DEFAULT_CONFIG, ("weighted",), (0, 1), 20)
    assert a.to_dict() == b.to_dict()


def test_empty_report_helpers():
    empty = CoverageReport()
    assert empty.commands == set()
    assert empty.n_failures == 0
    assert missing_commands(empty, {"mkdir"}) == {"mkdir"}
