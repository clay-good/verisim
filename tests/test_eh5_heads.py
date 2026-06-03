"""EH5-heads tests: trained per-subsystem heads vs bucketed decode entropy (SPEC-6 §8.2, HC7).

Covers the opt-in per-subsystem decode heads and the confound-free comparison harness:

  - a heads-enabled factored arm exposes **both** π_w signals on the identical proposer -- the
    trained head (via ``predict_delta_with_subsystem_uncertainty``) and the bucketed entropy (via
    ``predict_delta_with_subsystem_entropy``) -- each a well-formed non-negative per-subsystem map;
  - ``_EntropySignalView`` delegates the proposal byte-identically but surfaces the entropy signal
    where the loop reads uncertainty (so the two uncertainty arms differ only in the signal);
  - EH5-heads runs the four policies at equal budget, deterministically, and the §9.4 calibration
    diagnostic is well-formed (a bounded correlation per signal).

A default factored arm (heads off) is unchanged -- pinned here so the opt-in stays opt-in.
"""

from __future__ import annotations

import random

import pytest

from verisim.host.state import HostState
from verisim.hostmetrics.divergence import SUBSYSTEMS

torch = pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config  # noqa: E402
from verisim.experiments.eh5 import efficiency_by_policy  # noqa: E402
from verisim.experiments.eh5_heads import (  # noqa: E402
    EH5HeadsConfig,
    _EntropySignalView,
    _train_heads_arm,
    run_eh5_heads,
    signal_calibration,
)
from verisim.host.config import DEFAULT_HOST_CONFIG  # noqa: E402
from verisim.hostdata import HostDriver  # noqa: E402
from verisim.hostloop import budget_for_rho  # noqa: E402
from verisim.hostmodel import HostVocab, build_host_graph_model  # noqa: E402
from verisim.hostoracle.reference import ReferenceHostOracle  # noqa: E402

CONFIG = DEFAULT_HOST_CONFIG


def test_heads_off_arm_exposes_no_head_signal():
    """The opt-in stays opt-in: a default arm has no head and uses the entropy signal."""
    model = build_host_graph_model(
        HostVocab(CONFIG, max_pid=32), CONFIG, max_pid=32, d_model=24, mp_rounds=2, seed=1
    )
    assert not model.net.per_subsystem_heads
    state = HostState.initial()
    action = HostDriver("forky", CONFIG, random.Random(0)).sample(state)
    _, _, _entropy, head_map = model._decode(state, action, max_edits=64, max_new_tokens=4096)
    assert head_map is None  # no trained head -> the uncertainty method returns the entropy bucket


def test_heads_arm_exposes_both_signals_on_one_proposer():
    model = build_host_graph_model(
        HostVocab(CONFIG, max_pid=32), CONFIG, max_pid=32, d_model=24, mp_rounds=2,
        per_subsystem_heads=True, seed=1,
    )
    assert model.net.per_subsystem_heads
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(0))
    view = _EntropySignalView(model)
    for _ in range(8):
        action = driver.sample(state)
        delta_h, head_sig = model.predict_delta_with_subsystem_uncertainty(state, action)
        delta_e, ent_sig = view.predict_delta_with_subsystem_uncertainty(state, action)
        # Same proposer -> byte-identical deltas; only the reported signal differs.
        assert delta_h == delta_e == view.predict_delta(state, action)
        for sig in (head_sig, ent_sig):
            assert set(sig) == set(SUBSYSTEMS)
            assert all(v >= 0.0 for v in sig.values())
        state = oracle.step(state, action).state


def _tiny_config() -> EH5HeadsConfig:
    base = EH1Config(
        train_seeds=(0, 1), train_steps_per_traj=16, train_iters=80,
        n_layer=1, n_embd=32, block_size=160, difficulties={"low": "forky"},
        eval_seeds=(100, 101), eval_steps=12, epsilons=(0.0, 0.1),
    )
    return EH5HeadsConfig(
        base=base, rho=0.5, max_pid=32, graph_iters=80, graph_d_model=24, graph_batch=16
    )


def test_run_eh5_heads_compares_signals_at_equal_budget():
    config = _tiny_config()
    records = run_eh5_heads(config)
    assert {str(r.config["policy"]) for r in records} == set(config.policies)
    budget = budget_for_rho(config.rho, config.base.eval_steps)
    for r in records:
        assert r.oracle_calls == budget  # equal-ρ comparison (the §8.2 invariant)
    eff = efficiency_by_policy(records)
    assert set(eff) == set(config.policies)
    for stats in eff.values():
        assert stats["mean_h"] >= 0.0 and stats["mean_bits"] >= 0.0


def test_run_eh5_heads_is_deterministic():
    config = _tiny_config()
    oracle = ReferenceHostOracle()
    model = _train_heads_arm(config, HostVocab(CONFIG, max_pid=config.max_pid), oracle)
    a = run_eh5_heads(config, oracle=oracle, model=model)
    b = run_eh5_heads(config, oracle=oracle, model=model)
    assert [r.divergences for r in a] == [r.divergences for r in b]


def test_signal_calibration_is_well_formed():
    config = _tiny_config()
    oracle = ReferenceHostOracle()
    model = _train_heads_arm(config, HostVocab(CONFIG, max_pid=config.max_pid), oracle)
    cal = signal_calibration(model, oracle, config)
    assert set(cal) == {"head", "entropy"}
    for stats in cal.values():
        assert -1.0 <= stats["pearson"] <= 1.0
        assert -1.0 <= stats["spearman"] <= 1.0
        assert stats["n"] > 0
