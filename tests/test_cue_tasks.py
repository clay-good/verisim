"""SPEC-21 CP1-CP3 task-suite tests.

The contract: the keyed-set extractors read the right dimension, the suite is ordered
structure->content, the generic predictive-defense is exact for a faithful predictor and degrades
for a drifted one, the ρ-grounded extremes recover faithful/free, and the cheap keyed-drift forecast
is well-formed. Torch-free via the oracle + small fake M_θs; the real gradient comes from training.
"""

from __future__ import annotations

from typing import cast

import pytest

from verisim.acd.host_integrity import make_workload, oracle_step
from verisim.cue.tasks import (
    TASK_SUITE,
    TaskGapConfig,
    alive_procs,
    file_contents,
    grounded_keyed_reward,
    grounded_keyed_rollout,
    keyed_defense_reward,
    keyed_drift,
    open_fds,
    rollout_keyed,
    task_gap,
    written_files,
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle


class _FaithfulModel:
    def __init__(self, oracle: ReferenceHostOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: HostState, action: HostAction) -> object:
        return self._oracle.step(state, action).delta


class _BlindModel:
    def predict_delta(self, state: HostState, action: HostAction) -> list[object]:
        return []


def test_suite_is_ordered_structure_to_content():
    assert [t.order for t in TASK_SUITE] == [0, 1, 2, 3]
    assert [t.name for t in TASK_SUITE] == [
        "process-control", "fd-control", "file-integrity", "content-value"
    ]
    # the content-value task keys on (path, content) -- strictly finer than file-integrity's paths
    assert TASK_SUITE[2].key_fn is written_files
    assert TASK_SUITE[3].key_fn is file_contents


def test_extractors_read_the_right_dimension():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(seed=701, n_steps=14, oracle=oracle)
    step = oracle_step(oracle)
    state = start
    for a in actions:
        state = step(state, a)
    # procs are (pid, state) pairs; fds are (pid, fd) keys; the content sets relate as a refinement
    assert all(isinstance(x, tuple) and len(x) == 2 for x in alive_procs(state))
    assert all(isinstance(x, tuple) and len(x) == 2 for x in open_fds(state))
    # every written file appears as a (path, content) pair, and the paths match written_files
    contents = cast("set[tuple[str, str]]", file_contents(state))
    paths_from_contents = {p for (p, _c) in contents}
    assert paths_from_contents == written_files(state)


def test_faithful_predictor_scores_every_task_perfectly():
    oracle = ReferenceHostOracle()
    step = oracle_step(oracle)
    for task in TASK_SUITE:
        for seed in (700, 701, 702):
            start, actions = make_workload(seed, 14, oracle=oracle)
            r = keyed_defense_reward(step, step, start, actions, task.budget, task.key_fn)
            assert r == pytest.approx(1.0)


def test_rollout_keyed_is_cumulative():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(701, 14, oracle=oracle)
    step = oracle_step(oracle)
    seen = rollout_keyed(step, start, actions, written_files)
    # the cumulative set is the union over the rollout (monotone growth)
    running, state = set(written_files(start)), start
    for a in actions:
        state = step(state, a)
        running |= written_files(state)
    assert seen == running


def test_grounded_extremes_recover_faithful_and_free():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(701, 14, oracle=oracle)
    blind = _BlindModel()
    pred1, true1, calls1 = grounded_keyed_rollout(blind, oracle, start, actions, 1.0, written_files)
    assert pred1 == true1 and calls1 == len(actions)  # ρ=1 ≡ faithful
    _, _, calls0 = grounded_keyed_rollout(blind, oracle, start, actions, 0.0, written_files)
    assert calls0 == 0  # ρ=0 ≡ free (no oracle calls)


# --- torch-gated: the real structure->content gradient -------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.host_flagship import (  # noqa: E402
    HostFlagshipConfig,
    train_host_flagship,
)


def test_task_gap_gradient_and_keyed_drift():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    cfg = TaskGapConfig(horizon=10, workload_seeds=tuple(range(700, 708)))
    gaps = {t.name: task_gap(t, model, cfg) for t in TASK_SUITE}
    # the faithful predictor is exact on every task; the structural task is drift-robust (~0 gap)
    for g in gaps.values():
        assert g.faithful == pytest.approx(1.0)
        assert 0.0 <= g.free <= 1.0
    assert gaps["process-control"].gap == pytest.approx(0.0, abs=0.05)
    # content tasks carry a material gap (faithfulness load-bearing)
    assert gaps["content-value"].gap > 0.1
    # the cheap keyed drift is well-formed and nonzero where the model drifts
    kd = keyed_drift(TASK_SUITE[3], model, cfg)
    assert 0.0 <= kd <= 1.0 and kd > 0.0


def test_grounded_reward_perfect_model_is_one():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(702, 14, oracle=oracle)
    perfect = _FaithfulModel(oracle)
    for task in TASK_SUITE:
        r, _ = grounded_keyed_reward(perfect, oracle, start, actions, task.budget, 0.0, task.key_fn)
        assert r == pytest.approx(1.0)
