"""E2 policy-comparison tests (SPEC-2 §9, §16, H2, M7).

Exercises the full pipeline on a tiny config: train -> compare policies -> records.
The key invariant is *equal budget*: at one ρ every policy spends the same number
of oracle calls, so the comparison isolates where the budget is spent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.e1 import E1Config
from verisim.experiments.e2 import E2Config, run_e2
from verisim.metrics import aggregate_comparison


def _tiny_base() -> E1Config:
    return E1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=60,
        n_layer=1,
        n_embd=32,
        difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        eval_steps=8,
        epsilons=(0.0, 0.1),
    )


def _tiny_config() -> E2Config:
    return E2Config(base=_tiny_base(), rho=0.5, policies=("fixed", "uncertainty", "drift"))


def test_run_e2_produces_expected_records():
    records = run_e2(_tiny_config())
    # policies(3) x eval_seeds(2) x epsilons(2)
    assert len(records) == 3 * 2 * 2
    for rec in records:
        assert set(rec.config) >= {"experiment", "policy", "difficulty", "rho", "n_steps"}
        assert len(rec.divergences) == rec.config["n_steps"] == 8


def test_policies_compared_at_equal_budget():
    """Every policy spends exactly floor(ρ·T) oracle calls (true equal-ρ)."""
    records = run_e2(_tiny_config())
    budget = int(0.5 * 8)
    for rec in records:
        assert rec.oracle_calls == budget


def test_run_e2_is_deterministic():
    a = run_e2(_tiny_config())
    b = run_e2(_tiny_config())
    assert [r.divergences for r in a] == [r.divergences for r in b]
    assert [r.consultation_schedule for r in a] == [r.consultation_schedule for r in b]


def test_aggregate_comparison_over_policies():
    points = aggregate_comparison(run_e2(_tiny_config()), key="policy", n_resamples=100)
    labels = {p.label for p in points}
    assert labels == {"fixed", "uncertainty", "drift"}
    for p in points:
        assert p.mean_calls == int(0.5 * 8)
