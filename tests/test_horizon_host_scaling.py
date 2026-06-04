"""HS2 faithful-horizon scaling-law tests on the HOST world (SPEC-10 §5, H26 universality).

The host analogue of ``test_horizon_scaling``: the capacity-scaling harness re-run on the composed
host world (SPEC-6). Covers that the sweep produces well-formed per-scale stats (every metric, CIs,
params strictly increasing along the capacity axis) and is deterministic. The independence-horizon
math itself is shared with HS1 and tested there, so it is not re-tested here.

The smoke instance is deliberately tiny (two sizes, two seeds, few iters) so CI stays fast; the
committed scaling figure comes from the local sweep, not from CI (the SPEC-9 envelope discipline).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.experiments.horizon_host_scaling import (  # noqa: E402
    HostHorizonScalingConfig,
    run_host_horizon_scaling,
)
from verisim.experiments.horizon_scaling import METRICS, ModelScale  # noqa: E402


def _tiny_config() -> HostHorizonScalingConfig:
    return HostHorizonScalingConfig(
        scales=(
            ModelScale("xs", n_embd=32, n_layer=1, train_steps=40),
            ModelScale("s", n_embd=64, n_layer=2, train_steps=40),
        ),
        seeds=(0, 1),
        train_seeds=(0, 1),
        train_steps_per_traj=12,
        block_size=128,
        batch_size=16,
        num_threads=1,  # bit-deterministic, so the determinism test holds
        eval_seeds=(100, 101),
        eval_steps=12,
        one_step_seeds=(200,),
        one_step_steps=12,
    )


def test_run_host_horizon_scaling_is_well_formed():
    stats = run_host_horizon_scaling(_tiny_config())
    scales = {s.scale for s in stats}
    assert scales == {"xs", "s"}
    assert {s.metric for s in stats} == set(METRICS)
    # params strictly increase along the capacity axis
    params = sorted({(s.params, s.scale) for s in stats})
    assert params[0][1] == "xs" and params[-1][1] == "s"
    assert params[0][0] < params[-1][0]
    for s in stats:
        assert s.ci_lo <= s.mean <= s.ci_hi or s.n == 1
        assert s.n == 2
        if s.metric.startswith("one_step_acc"):
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_run_host_horizon_scaling_is_deterministic():
    a = run_host_horizon_scaling(_tiny_config())
    b = run_host_horizon_scaling(_tiny_config())
    assert [(s.scale, s.metric, s.mean) for s in a] == [(s.scale, s.metric, s.mean) for s in b]
