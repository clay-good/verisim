"""Coverage-balanced synthesis analysis (SPEC-2.1 §5 / K1).

K0 localized the v0 floor to two under-covered regions of the transition space: the
**path-copy distribution** (deep multi-segment paths the model must reproduce exactly) and
**failure/collision cases** (commands on existing/missing/non-empty targets, where the model
wrongly predicts success). This module measures coverage of both so a K1 dataset can be shown
to span them (the K1 gate), and is **dependency-free** (oracle + drivers only, no torch) like
the rest of the data/metric core.

A transition's *coverage cell* is ``"{command}:{ok|fail}"`` — the command crossed with whether
the oracle accepted it — so failure cases are first-class and countable. The *create-depth
histogram* counts the path depth of successful structural creates (mkdir/touch/write), the
copy-distribution axis. Together they are the K1 coverage report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from verisim.env.action import Action
from verisim.env.config import EnvConfig
from verisim.env.state import State
from verisim.oracle.base import Oracle

from .drivers import Driver

_CREATE_COMMANDS = {"mkdir", "touch", "write"}


def transition_cell(action: Action, exit_code: int) -> str:
    """Classify a transition into a coverage cell: ``"{command}:{ok|fail}"``."""
    name = action.name + ("-r" if action.recursive else "")
    return f"{name}:{'ok' if exit_code == 0 else 'fail'}"


def _path_depth(path: str) -> int:
    """Number of segments in an absolute path (``/`` → 0, ``/a`` → 1, ``/a/b`` → 2)."""
    return sum(1 for seg in path.split("/") if seg)


@dataclass
class CoverageReport:
    """Coverage of the transition space (the K1 gate artifact)."""

    cells: dict[str, int] = field(default_factory=dict)  # "command:ok|fail" -> count
    create_depths: dict[int, int] = field(default_factory=dict)  # path depth -> count

    @property
    def commands(self) -> set[str]:
        return {cell.split(":", 1)[0] for cell in self.cells}

    @property
    def n_failures(self) -> int:
        return sum(n for cell, n in self.cells.items() if cell.endswith(":fail"))

    def to_dict(self) -> dict[str, object]:
        return {
            "cells": dict(sorted(self.cells.items())),
            "create_depths": {str(k): v for k, v in sorted(self.create_depths.items())},
        }


def coverage_report(
    oracle: Oracle,
    env: EnvConfig,
    drivers: tuple[str, ...],
    seeds: tuple[int, ...],
    n_steps: int,
) -> CoverageReport:
    """Roll the driver mix forward and tally coverage cells + create-path depths."""
    report = CoverageReport()
    for driver_name in drivers:
        for seed in seeds:
            driver = Driver(name=driver_name, config=env, rng=random.Random(seed))
            state = State.empty()
            for _ in range(n_steps):
                action = driver.sample(state)
                result = oracle.step(state, action)
                cell = transition_cell(action, result.exit_code)
                report.cells[cell] = report.cells.get(cell, 0) + 1
                if result.exit_code == 0 and action.name in _CREATE_COMMANDS:
                    depth = _path_depth(action.args[0])
                    report.create_depths[depth] = report.create_depths.get(depth, 0) + 1
                state = result.state
    report.cells = dict(sorted(report.cells.items()))
    report.create_depths = dict(sorted(report.create_depths.items()))
    return report


def missing_commands(report: CoverageReport, required: set[str], min_count: int = 1) -> set[str]:
    """Required command names whose total count across cells is below ``min_count``."""
    totals: dict[str, int] = {}
    for cell, n in report.cells.items():
        name = cell.split(":", 1)[0]
        totals[name] = totals.get(name, 0) + n
    return {name for name in required if totals.get(name, 0) < min_count}


__all__ = ["CoverageReport", "coverage_report", "missing_commands", "transition_cell"]
