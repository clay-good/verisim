"""HS3 incr 2 data-cross-axis tests for the STRUCTURED (graph) arm (SPEC-10 §5, H26).

The graph-arm analogue of HS1.2: hold the graph capacity fixed and sweep the coverage-set size.
Covers that the sweep produces well-formed per-data-point stats (every metric, CIs, ``n_train``
strictly increasing along the data axis) and is deterministic. The smoke instance is tiny so CI
stays fast; the committed figure comes from the local sweep (the SPEC-9 envelope discipline).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_graph_data_scaling import run_graph_data_scaling  # noqa: E402
from verisim.experiments.horizon_graph_scaling import (  # noqa: E402
    GraphHorizonScalingConfig,
    GraphScale,
)
from verisim.experiments.horizon_scaling import METRICS  # noqa: E402


def _tiny_config() -> GraphHorizonScalingConfig:
    base = EN1Config(train_steps_per_traj=10)
    return GraphHorizonScalingConfig(
        base=base,
        scales=(GraphScale("m", d_model=16, n_layer=1, mp_rounds=1, train_steps=20),),
        seeds=(0, 1),
        eval_seeds=(100,),
        eval_steps=8,
        one_step_seeds=(200,),
        one_step_steps=8,
    )


def test_run_graph_data_scaling_is_well_formed():
    stats = run_graph_data_scaling(_tiny_config(), data_seeds=(2, 4))
    assert {s.metric for s in stats} == set(METRICS)
    # n_train (carried in ``params``) strictly increases along the data axis
    n_trains = sorted({s.params for s in stats})
    assert len(n_trains) == 2 and n_trains[0] < n_trains[1]
    for s in stats:
        assert s.ci_lo <= s.mean <= s.ci_hi or s.n == 1
        assert s.n == 2
        if s.metric.startswith("one_step_acc"):
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_run_graph_data_scaling_rejects_multi_capacity():
    cfg = _tiny_config()
    multi = GraphHorizonScalingConfig(
        base=cfg.base,
        scales=(*cfg.scales, GraphScale("s", d_model=24, n_layer=1, mp_rounds=1, train_steps=20)),
    )
    with pytest.raises(ValueError):
        run_graph_data_scaling(multi, data_seeds=(2,))


def test_run_graph_data_scaling_is_deterministic():
    a = run_graph_data_scaling(_tiny_config(), data_seeds=(2, 4))
    b = run_graph_data_scaling(_tiny_config(), data_seeds=(2, 4))
    assert [(s.params, s.metric, s.mean) for s in a] == [(s.params, s.metric, s.mean) for s in b]
