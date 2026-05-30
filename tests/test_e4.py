"""E4 ablation-harness tests (SPEC-2 §9, §17.5).

Exercises train-per-size -> measure clean faithfulness -> records on a tiny config,
and the teacher-forced accuracy primitive against the b2/b3 baselines.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.experiments.e1 import E1Config, eval_actions
from verisim.experiments.e4 import E4Config, E4Size, run_e4, teacher_forced_accuracy
from verisim.loop import NullModel, OracleBackedModel
from verisim.metrics import aggregate_values
from verisim.oracle.reference import ReferenceOracle


def _tiny_config() -> E4Config:
    return E4Config(
        base=E1Config(
            train_seeds=(0, 1),
            train_steps_per_traj=16,
            train_iters=40,
            difficulties={"low": "weighted"},
            eval_seeds=(100, 101),
            eval_steps=8,
        ),
        sizes=(E4Size("tiny", 1, 16), E4Size("small", 1, 32)),
    )


def test_teacher_forced_accuracy_brackets():
    """A perfect model scores 1.0; the trivial model scores < 1.0."""
    oracle = ReferenceOracle()
    actions = eval_actions(oracle, DEFAULT_CONFIG, "weighted", 100, 12)
    assert teacher_forced_accuracy(OracleBackedModel(oracle), oracle, State.empty(), actions) == 1.0
    assert teacher_forced_accuracy(NullModel(), oracle, State.empty(), actions) < 1.0


def test_run_e4_produces_a_record_per_cell():
    records = run_e4(_tiny_config())
    # sizes(2) x difficulties(1) x eval_seeds(2)
    assert len(records) == 2 * 1 * 2
    for rec in records:
        assert set(rec.config) >= {"size", "difficulty", "step_accuracy", "clean_horizon"}
        assert 0.0 <= rec.config["step_accuracy"] <= 1.0
        assert rec.config["rho"] == 0.0


def test_run_e4_is_deterministic():
    a = run_e4(_tiny_config())
    b = run_e4(_tiny_config())
    assert [r.config["step_accuracy"] for r in a] == [r.config["step_accuracy"] for r in b]


def test_aggregate_values_groups_e4_by_size_and_difficulty():
    stats = aggregate_values(
        run_e4(_tiny_config()), group_keys=["size", "difficulty"], value="step_accuracy"
    )
    assert {s.key for s in stats} == {("tiny", "low"), ("small", "low")}
    for s in stats:
        assert s.n == 2
        assert 0.0 <= s.mean <= 1.0
