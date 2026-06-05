"""HS3 incr 4 joint capacity×world-size tests for the STRUCTURED (graph) arm (SPEC-10 §5, H26).

The structured compute-optimal ladder: each cell carries its own (graph capacity, world size). It
covers that the ladder produces well-formed per-cell stats (every metric, CIs, params non-decreasing
the ladder) and is deterministic. The smoke instance is tiny so CI stays fast; the committed figure
comes from the local sweep alone (the SPEC-9 envelope discipline).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_graph_joint_scaling import run_graph_joint_scaling  # noqa: E402
from verisim.experiments.horizon_graph_scaling import (  # noqa: E402
    GraphHorizonScalingConfig,
    GraphScale,
)
from verisim.experiments.horizon_scaling import METRICS  # noqa: E402


def _tiny_config() -> GraphHorizonScalingConfig:
    base = EN1Config(train_seeds=(0, 1), train_steps_per_traj=8)
    return GraphHorizonScalingConfig(base=base, seeds=(0, 1), eval_seeds=(100,), eval_steps=6,
                                     one_step_seeds=(200,), one_step_steps=6)


_LADDER = (
    (GraphScale("s", d_model=16, n_layer=1, mp_rounds=1, train_steps=20), 3),
    (GraphScale("m", d_model=24, n_layer=1, mp_rounds=1, train_steps=20), 5),
)


def test_run_graph_joint_scaling_is_well_formed():
    stats = run_graph_joint_scaling(_tiny_config(), _LADDER)
    assert {s.metric for s in stats} == set(METRICS)
    # both ladder cells present, labelled with their world size
    labels = {s.scale for s in stats}
    assert labels == {"s@3h", "m@5h"}
    # params non-decreasing along the ladder
    params = sorted({s.params for s in stats})
    assert params[0] < params[-1]
    for s in stats:
        assert s.ci_lo <= s.mean <= s.ci_hi or s.n == 1
        assert s.n == 2
        if s.metric.startswith("one_step_acc"):
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_run_graph_joint_scaling_is_deterministic():
    a = run_graph_joint_scaling(_tiny_config(), _LADDER)
    b = run_graph_joint_scaling(_tiny_config(), _LADDER)
    assert [(s.scale, s.metric, s.mean) for s in a] == [(s.scale, s.metric, s.mean) for s in b]
