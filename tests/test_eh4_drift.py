"""EH4-drift §6.3 drift-lever tests (SPEC-6 §6.3, HC7).

  - ``corrupt_host_state`` produces a *valid* one-mutation HostState the oracle can still step
    (so oracle-relabeling is well-defined) -- a torch-free unit test of the noise lever.
  - the noise dataset builder relabels corrupted states (it still yields one example per step).
  - EH4-drift trains the three arms (clean / +noise / +self-forcing) and the comparison is
    well-formed + deterministic.
"""

from __future__ import annotations

import random

import pytest

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle

CONFIG = DEFAULT_HOST_CONFIG


def test_corrupt_host_state_stays_valid_and_steppable():
    """One mutation off-trajectory, but still a state the oracle interprets without error."""
    from verisim.hostdata import HostDriver
    from verisim.hostmodel.graph_train import corrupt_host_state

    oracle = ReferenceHostOracle()
    state = HostState.initial()
    driver = HostDriver("forky", CONFIG, random.Random(0))
    rng = random.Random(1)
    for _ in range(20):
        action = driver.sample(state)
        noisy = corrupt_host_state(state, CONFIG, rng)
        assert isinstance(noisy, HostState)
        assert 1 in noisy.procs  # init is never corrupted away
        # the oracle relabels the corrupted state without raising (it is a valid bundle state)
        result = oracle.step(noisy, action)
        assert isinstance(result.state, HostState)
        state = oracle.step(state, action).state


# --- torch-gated: the dataset levers + the EH4-drift harness -----------------

torch = pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config  # noqa: E402
from verisim.experiments.eh4_drift import EH4DriftConfig, run_eh4_drift  # noqa: E402
from verisim.hostmodel import HostVocab  # noqa: E402
from verisim.hostmodel.graph_train import build_host_graph_dataset  # noqa: E402


def test_noise_dataset_yields_one_example_per_step():
    vocab = HostVocab(CONFIG, max_pid=32)
    oracle = ReferenceHostOracle()
    clean = build_host_graph_dataset(oracle, vocab, CONFIG, seeds=(0,), n_steps=20, noise_prob=0.0)
    noisy = build_host_graph_dataset(
        oracle, vocab, CONFIG, seeds=(0,), n_steps=20, noise_prob=1.0, noise_seed=3
    )
    assert len(clean) == len(noisy) == 20  # one (graph, target) per step regardless of the lever


def _tiny_config() -> EH4DriftConfig:
    base = EH1Config(
        train_seeds=(0, 1), train_steps_per_traj=16, train_iters=80,
        n_layer=1, n_embd=32, block_size=160, difficulties={"low": "forky"},
        eval_seeds=(100, 101), eval_steps=10, epsilons=(0.0, 0.1),
    )
    return EH4DriftConfig(
        base=base, max_pid=32, graph_iters=80, graph_d_model=24, graph_batch=16, sf_rounds=2
    )


def test_run_eh4_drift_scores_three_arms():
    results = run_eh4_drift(_tiny_config())
    assert set(results) == {"clean", "noise", "self_forcing"}
    for r in results.values():
        assert 0.0 <= r["delta_exact"] <= 1.0
        assert r["h@0.0"] >= 0.0 and r["h@0.1"] >= 0.0


def test_run_eh4_drift_is_deterministic():
    a = run_eh4_drift(_tiny_config())
    b = run_eh4_drift(_tiny_config())
    assert a == b
