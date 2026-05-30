"""E4 objective-axis tests (SPEC-2 §9, §17.4).

Exercises train-supervised -> branch into +RLVR -> measure clean faithfulness per
arm on a tiny config, and that both arms branch from the same Stage-1 init.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.e1 import E1Config
from verisim.experiments.objective import ObjectiveConfig, run_objective
from verisim.metrics import aggregate_values


def _tiny_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        base=E1Config(
            train_seeds=(0, 1),
            train_steps_per_traj=16,
            train_iters=40,
            n_layer=1,
            n_embd=32,
            difficulties={"low": "weighted"},
            eval_seeds=(100, 101),
            eval_steps=6,
        ),
        rlvr_steps=4,
        rlvr_samples_per_env=3,
        rlvr_seeds=(0,),
        rlvr_n_steps=6,
        rlvr_max_edits=8,
        rlvr_max_run=12,
    )


def test_run_objective_produces_a_record_per_arm_and_cell():
    records = run_objective(_tiny_config())
    # objectives(2) x difficulties(1) x eval_seeds(2)
    assert len(records) == 2 * 1 * 2
    assert {r.config["objective"] for r in records} == {"supervised", "rlvr"}
    for rec in records:
        assert set(rec.config) >= {"objective", "difficulty", "step_accuracy", "clean_horizon"}
        assert 0.0 <= rec.config["step_accuracy"] <= 1.0
        assert rec.config["rho"] == 0.0


def test_run_objective_is_deterministic():
    a = run_objective(_tiny_config())
    b = run_objective(_tiny_config())
    assert [r.config["step_accuracy"] for r in a] == [r.config["step_accuracy"] for r in b]
    assert [r.config["clean_horizon"] for r in a] == [r.config["clean_horizon"] for r in b]


def test_aggregate_groups_objective_by_arm_and_difficulty():
    stats = aggregate_values(
        run_objective(_tiny_config()),
        group_keys=["objective", "difficulty"],
        value="step_accuracy",
    )
    assert {s.key for s in stats} == {("supervised", "low"), ("rlvr", "low")}
    for s in stats:
        assert s.n == 2
        assert 0.0 <= s.mean <= 1.0
