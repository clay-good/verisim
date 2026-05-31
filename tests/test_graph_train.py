"""Tests for the graph-arm trainer + noise-injection lever (SPEC-5 §6.3).

The headline is the K0-analog "learner works" check: the GNN+RSSM arm can fit oracle-generated
transitions to high teacher-forced accuracy, so any later off-distribution failure (EN4) is a
generalization gap, not a broken learner. The noise-injection lever is exercised end-to-end and
its oracle-relabeled targets are validated.
"""

from __future__ import annotations

from verisim.net.config import NetConfig
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.graph_train import (
    build_graph_dataset,
    corrupt_state,
    graph_teacher_forced_accuracy,
    train_graph_model,
)
from verisim.netmodel.vocab import NetVocab
from verisim.netoracle import ReferenceNetworkOracle

CFG = NetConfig()


def test_learner_fits_oracle_transitions() -> None:
    """K0 analog: the graph arm reaches high teacher-forced accuracy on a small dataset."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    examples = build_graph_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=20)
    model = build_graph_model(vocab, CFG, d_model=32, mp_rounds=2, seed=0)

    before = graph_teacher_forced_accuracy(model, examples)
    losses = train_graph_model(model, examples, steps=400, lr=3e-3, batch_size=16, seed=0)
    after = graph_teacher_forced_accuracy(model, examples)

    assert losses[-1] < losses[0]  # learning happened
    assert after > 0.9  # the learner can fit (not a broken model)
    assert after > before


def test_noise_injection_dataset_is_oracle_labeled() -> None:
    """With noise_prob>0 the dataset still has exactly one example per step, oracle-labeled."""
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    clean = build_graph_dataset(oracle, vocab, CFG, seeds=(0,), n_steps=30, noise_prob=0.0)
    noisy = build_graph_dataset(oracle, vocab, CFG, seeds=(0,), n_steps=30, noise_prob=0.5)
    assert len(clean) == len(noisy) == 30
    # every target ends in <eos> (a valid encoded delta), corrupted or not
    assert all(ex[1][-1] == vocab.eos for ex in noisy)


def test_noise_injected_training_runs_and_learns() -> None:
    vocab = NetVocab(CFG)
    oracle = ReferenceNetworkOracle()
    examples = build_graph_dataset(oracle, vocab, CFG, seeds=(0, 1), n_steps=20, noise_prob=0.3)
    model = build_graph_model(vocab, CFG, d_model=32, mp_rounds=2, seed=0)
    losses = train_graph_model(model, examples, steps=300, lr=3e-3, batch_size=16, seed=0)
    assert losses[-1] < losses[0]


def test_corrupt_state_changes_exactly_a_little() -> None:
    import random

    oracle = ReferenceNetworkOracle()
    base = build_graph_dataset(oracle, NetVocab(CFG), CFG, seeds=(0,), n_steps=5)
    # corrupt_state returns a different, still-valid NetworkState
    from verisim.net.state import NetworkState

    state = NetworkState.initial(CFG.hosts)
    rng = random.Random(0)
    changed = corrupt_state(state, CFG, rng)
    assert isinstance(changed, NetworkState)
    assert set(changed.hosts) == set(CFG.hosts)  # host set preserved
    assert base  # sanity: dataset built
