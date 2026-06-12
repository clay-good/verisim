"""SPEC-21 cross-world network scale-law tests (CS1-net).

The contract: the network keyed extractors read the right dimension, the suite is ordered
structure->content, the faithful predictor is exact on every task (ground-truth labels on the net
world too), and the smoke ladder runs end-to-end so the host harness's reducers compute on a network
ScaleLawResult. Torch-free parts use the oracle; the cross-world gradient comes from training.
"""

from __future__ import annotations

import pytest

from verisim.acd.net_integrity import make_net_workload, oracle_step
from verisim.experiments.net_scale_law import (
    NET_TASK_SUITE,
    _keyed_reward,
    _rollout_keyed,
    flow_set,
    link_set,
    service_set,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


def test_suite_ordered_structure_to_content():
    assert [t.order for t in NET_TASK_SUITE] == [0, 1, 2]
    assert [t.name for t in NET_TASK_SUITE] == [
        "service-control", "link-control", "flow-integrity"
    ]
    assert [t.keyed_dimension for t in NET_TASK_SUITE] == ["services", "links", "flows"]


def test_extractors_read_the_right_dimension():
    oracle = ReferenceNetworkOracle()
    start, actions = make_net_workload(seed=801, n_steps=20, oracle=oracle)
    step = oracle_step(oracle)
    state = start
    for a in actions:
        state = step(state, a)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in service_set(state))  # (host, port)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in link_set(state))  # (a, b)
    assert all(isinstance(x, tuple) and len(x) == 3 for x in flow_set(state))  # (src, dst, port)


def test_faithful_predictor_scores_every_task_perfectly():
    oracle = ReferenceNetworkOracle()
    step = oracle_step(oracle)
    for task in NET_TASK_SUITE:
        for seed in (800, 801, 802):
            start, actions = make_net_workload(seed, 20, oracle=oracle)
            r = _keyed_reward(step, step, start, actions, task.budget, task.key_fn)
            assert r == pytest.approx(1.0)


def test_rollout_keyed_is_cumulative():
    oracle = ReferenceNetworkOracle()
    start, actions = make_net_workload(801, 20, oracle=oracle)
    step = oracle_step(oracle)
    seen = _rollout_keyed(step, start, actions, flow_set)
    running, state = set(flow_set(start)), start
    for a in actions:
        state = step(state, a)
        running |= flow_set(state)
    assert seen == running


# --- torch-gated: the smoke ladder end-to-end (the host reducers on a network result) -------------

torch = pytest.importorskip("torch")

from verisim.experiments.net_scale_law import (  # noqa: E402
    NetScaleLawConfig,
    run_net_scale_law,
)
from verisim.experiments.scale_law import (  # noqa: E402
    forecast_check,
    frontier_verdict,
)


def test_net_smoke_ladder_runs_end_to_end():
    result = run_net_scale_law(NetScaleLawConfig.smoke())
    assert len(result.rungs) == 2
    for rung in result.rungs:
        assert len(rung.gaps) == 3
        for g in rung.gaps:
            assert g.faithful == pytest.approx(1.0)  # ground-truth labels on the network too
        # the structural service task is ~drift-robust; the content flow task carries a real gap
        flow = next(g for g in rung.gaps if g.task == "flow-integrity")
        assert flow.gap > 0.1
    # the host harness's reducers compute on the network ScaleLawResult unchanged
    assert "frontier_recedes_or_flat" in frontier_verdict(result)
    assert "forecastable" in forecast_check(result)
