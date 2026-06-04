"""HS1.3 joint capacity×data scaling tests (SPEC-10 §5).

The harness runs a ladder of (capacity, data) cells, each with its own coverage set; these cover
that the sweep is well-formed (every metric present, the params x-axis strictly increasing, data
folded into the label, CIs bracket the mean) and deterministic. The smoke instance is tiny (two
cells, two model seeds, few iters) so CI stays fast; the committed figure is from the local sweep.
"""

from __future__ import annotations

import pytest

from verisim.experiments.horizon_scaling import METRICS

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_joint_scaling import run_joint_scaling  # noqa: E402
from verisim.experiments.horizon_scaling import HorizonScalingConfig, ModelScale  # noqa: E402


def _tiny_config() -> HorizonScalingConfig:
    return HorizonScalingConfig(
        base=EN1Config(train_steps_per_traj=12),
        seeds=(0, 1),
        block_size=64,
        batch_size=16,
        num_threads=1,  # bit-deterministic, so the determinism test holds
        eval_seeds=(100, 101),
        eval_steps=12,
        one_step_seeds=(200,),
        one_step_steps=16,
    )


def _tiny_points() -> tuple[tuple[ModelScale, int], ...]:
    return (
        (ModelScale("xs", n_embd=32, n_layer=1, train_steps=60), 2),
        (ModelScale("s", n_embd=64, n_layer=2, train_steps=60), 4),
    )


def test_run_joint_scaling_is_well_formed():
    stats = run_joint_scaling(_tiny_config(), _tiny_points())
    # params (capacity) strictly increases along the ladder; data is folded into the label
    params = sorted({(s.params, s.scale) for s in stats})
    assert params[0][0] < params[-1][0]
    assert params[0][1].startswith("xs@") and params[-1][1].startswith("s@")
    assert {s.metric for s in stats} == set(METRICS)
    for s in stats:
        assert s.n == 2
        assert s.ci_lo <= s.mean <= s.ci_hi or s.n == 1
        if s.metric.startswith("one_step_acc"):
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_run_joint_scaling_is_deterministic():
    a = run_joint_scaling(_tiny_config(), _tiny_points())
    b = run_joint_scaling(_tiny_config(), _tiny_points())
    assert [(s.scale, s.metric, s.mean) for s in a] == [(s.scale, s.metric, s.mean) for s in b]
