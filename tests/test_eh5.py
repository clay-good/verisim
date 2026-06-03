"""EH5 smart which-subsystem policy tests (SPEC-6 §8.2, HC7; H10's host analogue).

Covers the new π_w machinery and the experiment harness:

  - ``UncertaintySubsystem`` picks the argmax-uncertainty subsystem, and falls back to round-robin
    when no uncertainty is supplied (a baseline proposer exposes none) -- a pure unit test.
  - the factored arm exposes a well-formed per-subsystem decode-entropy map (one non-negative float
    per subsystem) -- the smart-π_w signal.
  - EH5 runs the four policies at equal budget and the comparison is well-formed + deterministic.
"""

from __future__ import annotations

import random

import pytest

from verisim.host.state import HostState
from verisim.hostloop import UncertaintySubsystem
from verisim.hostmetrics.divergence import SUBSYSTEMS


def test_uncertainty_policy_picks_argmax_then_falls_back():
    pol = UncertaintySubsystem()
    s = HostState.initial()
    # Picks the most-uncertain subsystem when given a signal.
    assert pol.select(s, {"proc": 0.1, "fd": 0.9, "fs": 0.2, "global": 0.0}) == "fd"
    # With no signal it falls back to a deterministic round-robin sweep (one instance, so the
    # internal cursor advances across calls).
    rr = UncertaintySubsystem()
    seq = [rr.select(s, None) for _ in range(len(SUBSYSTEMS))]
    assert seq == list(SUBSYSTEMS)  # covers every subsystem before repeating
    # An all-zero signal is treated as "no information" -> the same round-robin fallback.
    assert UncertaintySubsystem().select(s, dict.fromkeys(SUBSYSTEMS, 0.0)) == SUBSYSTEMS[0]


# --- torch-gated: the factored arm's signal + the EH5 harness ----------------

torch = pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config  # noqa: E402
from verisim.experiments.eh5 import EH5Config, efficiency_by_policy, run_eh5  # noqa: E402
from verisim.host.config import DEFAULT_HOST_CONFIG  # noqa: E402
from verisim.hostdata import HostDriver  # noqa: E402
from verisim.hostloop import budget_for_rho  # noqa: E402
from verisim.hostmodel import HostVocab, build_host_graph_model  # noqa: E402
from verisim.hostoracle.reference import ReferenceHostOracle  # noqa: E402

CONFIG = DEFAULT_HOST_CONFIG


def test_factored_arm_exposes_per_subsystem_uncertainty():
    model = build_host_graph_model(
        HostVocab(CONFIG, max_pid=32), CONFIG, max_pid=32, d_model=24, mp_rounds=2, seed=1
    )
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(0))
    for _ in range(10):
        action = driver.sample(state)
        delta, unc = model.predict_delta_with_subsystem_uncertainty(state, action)
        assert isinstance(delta, list)
        assert set(unc) == set(SUBSYSTEMS)
        assert all(v >= 0.0 for v in unc.values())
        state = oracle.step(state, action).state


def _tiny_config() -> EH5Config:
    base = EH1Config(
        train_seeds=(0, 1), train_steps_per_traj=16, train_iters=80,
        n_layer=1, n_embd=32, block_size=160, difficulties={"low": "forky"},
        eval_seeds=(100, 101), eval_steps=12, epsilons=(0.0, 0.1),
    )
    return EH5Config(
        base=base, rho=0.5, max_pid=32, graph_iters=80, graph_d_model=24, graph_batch=16
    )


def test_run_eh5_compares_policies_at_equal_budget():
    config = _tiny_config()
    records = run_eh5(config)
    assert {str(r.config["policy"]) for r in records} == set(config.policies)
    budget = budget_for_rho(config.rho, config.base.eval_steps)
    for r in records:
        assert r.oracle_calls == budget  # equal-ρ comparison (the §8.2 invariant)
    eff = efficiency_by_policy(records)
    assert set(eff) == set(config.policies)
    for stats in eff.values():
        assert stats["mean_h"] >= 0.0 and stats["mean_bits"] >= 0.0


def test_run_eh5_is_deterministic():
    a = run_eh5(_tiny_config())
    b = run_eh5(_tiny_config())
    assert [r.divergences for r in a] == [r.divergences for r in b]
