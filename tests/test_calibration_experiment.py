"""Calibration-diagnostic harness tests (SPEC-2 §7.2, §17.2).

Exercises train -> collect pairs -> report on a tiny config, and the teacher-forced
pair collection against a known model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.experiments.calibration import (
    CalibrationConfig,
    collect_pairs,
    read_pairs,
    run_calibration,
    write_pairs,
)
from verisim.experiments.e1 import E1Config, eval_actions
from verisim.loop import OracleBackedModel
from verisim.oracle.reference import ReferenceOracle


def _tiny_config() -> CalibrationConfig:
    return CalibrationConfig(
        base=E1Config(
            train_seeds=(0, 1),
            train_steps_per_traj=16,
            train_iters=60,
            n_layer=1,
            n_embd=32,
            difficulties={"low": "weighted"},
            eval_seeds=(100, 101),
            eval_steps=8,
        ),
        n_bins=5,
    )


def test_run_calibration_reports_over_all_pairs():
    report, pairs = run_calibration(_tiny_config())
    # difficulties(1) x eval_seeds(2) x eval_steps(8)
    assert len(pairs) == 1 * 2 * 8 == report.n
    assert -1.0 <= report.pearson <= 1.0
    assert -1.0 <= report.spearman <= 1.0
    assert sum(b.n for b in report.bins) == report.n


def test_run_calibration_is_deterministic():
    a, _ = run_calibration(_tiny_config())
    b, _ = run_calibration(_tiny_config())
    assert (a.pearson, a.spearman, a.n) == (b.pearson, b.spearman, b.n)


def test_collect_pairs_perfect_model_has_zero_divergence():
    """A perfect model never errs, so every pair's divergence is 0."""
    oracle = ReferenceOracle()
    perfect = OracleBackedModel(oracle)

    # OracleBackedModel exposes only predict_delta; wrap it with a trivial signal.
    class _Perfect:
        def predict_delta_with_uncertainty(self, state, action):
            return perfect.predict_delta(state, action), 0.0

    actions = eval_actions(oracle, DEFAULT_CONFIG, "weighted", 100, 8)
    pairs = collect_pairs(_Perfect(), oracle, State.empty(), actions)
    assert len(pairs) == 8
    assert all(d == 0.0 for _, d in pairs)


def test_pairs_roundtrip(tmp_path):
    _, pairs = run_calibration(_tiny_config())
    path = write_pairs(pairs, tmp_path / "pairs.jsonl")
    restored = read_pairs(path)
    assert len(restored) == len(pairs)
