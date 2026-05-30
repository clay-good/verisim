"""E3 operator-comparison tests (SPEC-2 §9, §16, H3, M7).

The honest v0 result: with a full-state oracle truth the three correction
operators snap to the same state, so ``H_ε`` is identical across operators. They
differ only in the diagnostic each records (the residual magnitude / repair cost
that motivate the deferred H3 work). The tests pin both: equal horizons, distinct
diagnostics, equal budget.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.e1 import E1Config
from verisim.experiments.e3 import E3Config, run_e3
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


def _tiny_config() -> E3Config:
    return E3Config(base=_tiny_base(), rho=0.5, policy="fixed")


def test_run_e3_produces_expected_records():
    records = run_e3(_tiny_config())
    # operators(3) x eval_seeds(2) x epsilons(2)
    assert len(records) == 3 * 2 * 2
    for rec in records:
        assert set(rec.config) >= {"experiment", "operator", "policy", "rho", "diagnostic"}


def test_operators_have_identical_faithful_horizon():
    """The v0 theoretical identity: all operators correct to the same truth."""
    records = run_e3(_tiny_config())
    by_op: dict[str, list[tuple[int, float, int]]] = {}
    for rec in records:
        by_op.setdefault(str(rec.config["operator"]), []).append(
            (rec.seed, rec.epsilon, rec.faithful_horizon)
        )
    ops = list(by_op)
    assert len(ops) == 3
    reference = sorted(by_op[ops[0]])
    for op in ops[1:]:
        assert sorted(by_op[op]) == reference


def test_operator_diagnostics_are_distinct():
    records = run_e3(_tiny_config())
    diags = {str(r.config["operator"]): r.config["diagnostic"] for r in records}
    assert diags["hard_reset"] is None
    # residual logs an absolute discrepancy count; projection a [0,1] fraction.
    assert isinstance(diags["residual"], float) and diags["residual"] >= 0.0
    assert isinstance(diags["projection"], float) and 0.0 <= diags["projection"] <= 1.0


def test_operators_compared_at_equal_budget():
    points = aggregate_comparison(run_e3(_tiny_config()), key="operator", n_resamples=100)
    assert {p.label for p in points} == {"hard_reset", "residual", "projection"}
    for p in points:
        assert p.mean_calls == int(0.5 * 8)
