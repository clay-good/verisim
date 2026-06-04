"""HS3 faithful-horizon scaling-law tests for the STRUCTURED (graph) arm (SPEC-10 §5, H26).

The graph-arm analogue of ``test_horizon_scaling``: the capacity-scaling harness re-run with the
GNN+RSSM graph proposer (SPEC-5 §6.1-6.2) instead of the flat transformer. Covers that the sweep
produces well-formed per-scale stats (every metric, CIs, params strictly increasing along the
capacity axis) and is deterministic. The independence-horizon math is shared with HS1 and tested
there.

The smoke instance is deliberately tiny (two sizes, two seeds, few iters) so CI stays fast; the
committed scaling figure comes from the local sweep, not from CI (the SPEC-9 envelope discipline).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_graph_scaling import (  # noqa: E402
    GraphHorizonScalingConfig,
    GraphScale,
    run_graph_horizon_scaling,
)
from verisim.experiments.horizon_scaling import METRICS  # noqa: E402


def _tiny_config() -> GraphHorizonScalingConfig:
    base = EN1Config(train_seeds=(0, 1), train_steps_per_traj=12)
    return GraphHorizonScalingConfig(
        base=base,
        scales=(
            GraphScale("xs", d_model=16, n_layer=1, mp_rounds=1, train_steps=20),
            GraphScale("s", d_model=24, n_layer=2, mp_rounds=1, train_steps=20),
        ),
        seeds=(0, 1),
        eval_seeds=(100,),
        eval_steps=8,
        one_step_seeds=(200,),
        one_step_steps=8,
    )


def test_run_graph_horizon_scaling_is_well_formed():
    stats = run_graph_horizon_scaling(_tiny_config())
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


def test_run_graph_horizon_scaling_is_deterministic():
    a = run_graph_horizon_scaling(_tiny_config())
    b = run_graph_horizon_scaling(_tiny_config())
    assert [(s.scale, s.metric, s.mean) for s in a] == [(s.scale, s.metric, s.mean) for s in b]
