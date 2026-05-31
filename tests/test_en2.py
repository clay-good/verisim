"""EN2 network consultation-policy comparison tests (SPEC-5 §12, H9, NW7).

Mirrors v0's ``test_e2``: train -> compare policies -> records, and the equal-ρ invariant
(every policy spends exactly the budget, so the comparison is at truly equal ρ).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config
from verisim.experiments.en2 import EN2Config, run_en2


def _tiny() -> EN2Config:
    base = EN1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=60,
        n_layer=1,
        n_embd=32,
        block_size=128,
        difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        eval_steps=8,
        rhos=(0.5,),
        epsilons=(0.0, 0.1),
    )
    return EN2Config(base=base, rho=0.5, policies=("fixed", "uncertainty", "drift"))


def test_run_en2_produces_records_and_equal_budget():
    records = run_en2(_tiny())
    # policies(3) x difficulties(1) x eval_seeds(2) x epsilons(2)
    assert len(records) == 3 * 1 * 2 * 2
    budget = int(0.5 * 8)
    for rec in records:
        assert rec.config["policy"] in {"fixed", "uncertainty", "drift"}
        assert rec.oracle_calls == budget  # equal-ρ: every policy spends exactly the budget
        assert rec.config["oracle_bits"] >= 0


def test_run_en2_is_deterministic():
    a = run_en2(_tiny())
    b = run_en2(_tiny())
    assert [r.divergences for r in a] == [r.divergences for r in b]
