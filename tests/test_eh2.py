"""EH2 consultation-policy comparison tests (SPEC-6 §8.1, HC7; H9's host analogue).

Exercises the full pipeline on a tiny config: train both arms, compare the three policies at a fixed
interior ρ, and check the equal-budget invariant (every arm spends exactly the budget via the
spend-down backstop) and determinism. The H9 *outcome* is a datum, not asserted -- the test pins the
apparatus.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config
from verisim.experiments.eh2 import EH2Config, run_eh2, unaided_signals
from verisim.hostloop import budget_for_rho


def _tiny_config() -> EH2Config:
    base = EH1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=80,
        n_layer=1,
        n_embd=32,
        block_size=160,
        difficulties={"low": "forky"},
        eval_seeds=(100, 101),
        eval_steps=12,
        epsilons=(0.0, 0.1),
    )
    return EH2Config(
        base=base, rho=0.5, max_pid=32, graph_iters=80, graph_d_model=24, graph_batch=16
    )


def test_run_eh2_covers_arms_and_policies():
    records = run_eh2(_tiny_config())
    labels = {str(r.config["label"]) for r in records}
    assert labels == {
        "flat/fixed", "flat/uncertainty", "flat/drift",
        "factored/fixed", "factored/uncertainty", "factored/drift",
    }


def test_equal_budget_across_policies():
    """The spend-down backstop makes every policy spend exactly floor(ρ·T) consultations."""
    config = _tiny_config()
    records = run_eh2(config)
    budget = budget_for_rho(config.rho, config.base.eval_steps)
    for r in records:
        assert r.oracle_calls == budget  # equal-ρ comparison (the §8.1 invariant)
        assert len(r.divergences) == config.base.eval_steps


def test_signals_are_nonnegative_and_per_step():
    """Both arms expose a per-step uncertainty signal the triggered policies threshold."""
    from verisim.experiments.eh1 import eval_actions
    from verisim.host.config import DEFAULT_HOST_CONFIG
    from verisim.hostmodel import HostVocab, NeuralHostWorldModel
    from verisim.hostoracle.reference import ReferenceHostOracle

    oracle = ReferenceHostOracle()
    vocab = HostVocab(DEFAULT_HOST_CONFIG, max_pid=32)
    from verisim.experiments.eh1 import train_model

    base = _tiny_config().base
    flat = NeuralHostWorldModel(train_model(base, vocab, oracle, DEFAULT_HOST_CONFIG), vocab)
    actions = eval_actions(oracle, DEFAULT_HOST_CONFIG, "forky", 100, 12)
    signals = unaided_signals(flat, actions)
    assert len(signals) == 12
    assert all(s >= 0.0 for s in signals)


def test_run_eh2_is_deterministic():
    a = run_eh2(_tiny_config())
    b = run_eh2(_tiny_config())
    assert [r.divergences for r in a] == [r.divergences for r in b]
