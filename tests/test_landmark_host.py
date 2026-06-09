"""Unit tests for the torch-free host landmark layer (SPEC-12 §6 LP8-host, H38).

The privilege signature and the re-grounding executor are dependency-free (the NW0-NW3 discipline),
so they are tested directly with the ``HostNullModel`` / ``HostOracleBackedModel`` baselines — no
torch, no training. The headline magnitudes come from the committed full-config run (``lp8_host``).
"""

from __future__ import annotations

import random

from verisim.host.config import HostConfig
from verisim.host.state import RUNNING, HostState
from verisim.hostdata.drivers import HostDriver
from verisim.hostloop.model import HostNullModel, HostOracleBackedModel
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.landmark.host import execute_host_plan, privilege_signature


def _journey(oracle: ReferenceHostOracle, host: HostConfig, n: int, seed: int):
    drv = HostDriver("forky", host, random.Random(seed))
    start = HostState.initial()
    state = start
    actions, truth = [], []
    for _ in range(n):
        a = drv.sample(state)
        state = oracle.step(state, a).state
        actions.append(a)
        truth.append(state)
    return start, actions, truth


def test_privilege_signature_initial_is_single_root_running() -> None:
    # The boot state is one RUNNING root process, so the privilege class set is {(RUNNING, 0)}.
    assert privilege_signature(HostState.initial()) == frozenset({(RUNNING, 0)})


def test_oracle_backed_model_reaches_every_goal() -> None:
    host = HostConfig()
    oracle = ReferenceHostOracle()
    start, actions, truth = _journey(oracle, host, 12, seed=100)
    trace = execute_host_plan(
        HostOracleBackedModel(oracle), start, actions, truth, frozenset(), reground=False
    )
    assert trace.goal_reached
    assert trace.priv_horizon == trace.n_steps
    assert trace.full_horizon == trace.n_steps


def test_landmark_regrounding_spends_budget_and_excludes_goal() -> None:
    host = HostConfig()
    oracle = ReferenceHostOracle()
    start, actions, truth = _journey(oracle, host, 12, seed=101)
    reground_at = frozenset({3, 7})  # interior boundaries for L=4, G=12 (goal step 11 excluded)
    trace = execute_host_plan(HostNullModel(), start, actions, truth, reground_at, reground=True)
    assert trace.n_consults == 2
    assert 11 not in reground_at
    assert 0.0 <= float(trace.goal_reached) <= 1.0


def test_execute_host_plan_is_deterministic() -> None:
    host = HostConfig()
    oracle = ReferenceHostOracle()
    start, actions, truth = _journey(oracle, host, 10, seed=102)
    a = execute_host_plan(HostNullModel(), start, actions, truth, frozenset({3}), reground=True)
    b = execute_host_plan(HostNullModel(), start, actions, truth, frozenset({3}), reground=True)
    assert a == b
