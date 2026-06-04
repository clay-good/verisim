"""HS1 faithful-horizon scaling-law tests (SPEC-10, H26).

Covers the capacity-scaling harness and the independence-horizon math:

  - ``independence_horizon`` is the geometric `p/(1-p)`, clamped at the eval cap, monotone in `p`;
  - the sweep produces well-formed per-scale stats (every metric, CIs, params strictly increasing
    along the capacity axis) and is deterministic;
  - horizon efficiency is a non-negative ratio of the two horizons.

The smoke instance is deliberately tiny (two sizes, two seeds, few iters) so CI stays fast; the
committed scaling figure comes from the local sweep, not from CI (the SPEC-9 envelope discipline).
"""

from __future__ import annotations

import math

import pytest

from verisim.experiments.horizon_scaling import METRICS, independence_horizon


def test_independence_horizon_is_geometric_and_clamped():
    assert independence_horizon(0.0, cap=100.0) == 0.0
    assert independence_horizon(0.5, cap=100.0) == pytest.approx(1.0)  # p/(1-p) = 1
    assert independence_horizon(0.9, cap=100.0) == pytest.approx(9.0)
    assert independence_horizon(1.0, cap=32.0) == 32.0  # clamped (p=1 would be inf)
    assert independence_horizon(0.999, cap=5.0) == 5.0  # clamp binds below the analytic value
    # strictly increasing in p below the clamp
    from itertools import pairwise

    xs = [independence_horizon(p, cap=1e9) for p in (0.1, 0.3, 0.6, 0.8)]
    assert all(b > a for a, b in pairwise(xs))
    assert all(math.isfinite(x) for x in xs)


# --- torch-gated: the training sweep --------------------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_scaling import (  # noqa: E402
    HorizonScalingConfig,
    ModelScale,
    run_horizon_scaling,
)


def _tiny_config() -> HorizonScalingConfig:
    base = EN1Config(train_seeds=(0, 1), train_steps_per_traj=16)
    return HorizonScalingConfig(
        base=base,
        scales=(
            ModelScale("xs", n_embd=32, n_layer=1, train_steps=60),
            ModelScale("s", n_embd=64, n_layer=2, train_steps=60),
        ),
        seeds=(0, 1),
        block_size=64,
        batch_size=16,
        num_threads=1,  # bit-deterministic, so the determinism test holds
        eval_seeds=(100, 101),
        eval_steps=12,
        one_step_seeds=(200,),
        one_step_steps=16,
    )


def test_run_horizon_scaling_is_well_formed():
    stats = run_horizon_scaling(_tiny_config())
    # every (scale, metric) cell present
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
        if s.metric == "one_step_acc":
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_run_horizon_scaling_is_deterministic():
    a = run_horizon_scaling(_tiny_config())
    b = run_horizon_scaling(_tiny_config())
    assert [(s.scale, s.metric, s.mean) for s in a] == [(s.scale, s.metric, s.mean) for s in b]
