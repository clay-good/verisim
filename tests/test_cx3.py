"""Tests for CX3, the matched-coverage cut (SPEC-17 §6, H62).

Two layers. The matched-coverage *sampler* (``causal/coverage.py``) is pure (no torch, no oracle),
so it gets exact unit tests — the bookkeeping that makes the branching-vs-coverage cut honest. The
CX3 *experiment* trains a real distributed `M_θ` (torch), so its smoke test is ``skipif``-guarded
and tiny: it asserts structural invariants (the four arms appear; the matched arms share count *and*
coverage; rates in range; deterministic), not the H62 verdict's magnitude — that is the committed
figure on the primary host.
"""

from __future__ import annotations

import random

import pytest

from verisim.causal.coverage import coverage_rate, feasible_match, match_coverage


def test_coverage_rate() -> None:
    assert coverage_rate([]) == 0.0
    assert coverage_rate([True, True, False, False]) == 0.5
    assert coverage_rate([True, True, True, True]) == 1.0


def test_match_coverage_hits_target_count_and_fraction() -> None:
    examples = list(range(100))
    flags = [i % 2 == 0 for i in examples]  # 50% change
    out = match_coverage(examples, flags, target_coverage=0.25, target_count=40,
                         rng=random.Random(0))
    assert len(out) == 40
    realized = coverage_rate([e % 2 == 0 for e in out])
    assert abs(realized - 0.25) < 1e-9  # 10 changing + 30 non-changing
    assert len(set(out)) == 40  # no replacement


def test_match_coverage_caps_at_bin_capacity() -> None:
    examples = list(range(20))
    flags = [i < 3 for i in examples]  # only 3 changing
    out = match_coverage(examples, flags, target_coverage=0.9, target_count=20,
                         rng=random.Random(1))
    # can't exceed 3 changing examples without replacement
    assert sum(1 for e in out if e < 3) <= 3


def test_match_coverage_aligns_inputs() -> None:
    with pytest.raises(ValueError):
        match_coverage([1, 2, 3], [True, False], target_coverage=0.5, target_count=2,
                       rng=random.Random(0))


def test_feasible_match_is_reachable_by_every_pool() -> None:
    pool_a = [True] * 80 + [False] * 20  # 0.80 coverage
    pool_b = [True] * 30 + [False] * 70  # 0.30 coverage
    cov, count = feasible_match([pool_a, pool_b])
    assert abs(cov - 0.30) < 1e-9  # min natural coverage
    # both pools can supply `count` examples at `cov` without replacement
    for pool in (pool_a, pool_b):
        out = match_coverage(list(range(len(pool))), pool, target_coverage=cov,
                             target_count=count, rng=random.Random(0))
        assert len(out) == count
        assert abs(coverage_rate([pool[e] for e in out]) - cov) < 0.02


def test_match_coverage_deterministic() -> None:
    examples = list(range(50))
    flags = [i % 3 == 0 for i in examples]
    a = match_coverage(examples, flags, target_coverage=0.3, target_count=20, rng=random.Random(7))
    b = match_coverage(examples, flags, target_coverage=0.3, target_count=20, rng=random.Random(7))
    assert a == b


# --- the torch-gated experiment smoke (trains a real distributed M_θ) ---

torch = pytest.importorskip("torch")  # CX3 trains a real GPT; skip cleanly where torch is absent

from verisim.experiments.cx3 import ARMS, CX3Config, run_cx3  # noqa: E402
from verisim.experiments.ed6 import ED6Config  # noqa: E402


def _tiny() -> CX3Config:
    return CX3Config(
        ed6=ED6Config(
            train_seeds=(0, 1), train_steps_per_traj=12, k_counterfactual=3, n_layer=1, n_head=2,
            n_embd=32, block_size=256, train_iters=20, batch_size=32, eval_seeds=(100,),
            eval_steps=6, m_interventions=3,
        ),
        model_seeds=(0,),
    )


def test_cx3_smoke_structural() -> None:
    stats = run_cx3(_tiny())
    by = {(s.arm, s.metric): s for s in stats}
    assert {a for a, _ in by} == set(ARMS)
    for s in stats:
        assert 0.0 <= s.mean <= 1.0
        assert s.ci_lo <= s.mean <= s.ci_hi
        assert s.n_examples > 0
    # the load-bearing invariant: the matched arms share count AND coverage exactly.
    fac = by[("factual-matched", "intervention_exact")]
    cf = by[("+counterfactual-matched", "intervention_exact")]
    assert fac.n_examples == cf.n_examples
    assert abs(fac.coverage - cf.coverage) < 1e-9


def test_cx3_deterministic() -> None:
    def run() -> list[tuple[str, str, float]]:
        return [(s.arm, s.metric, round(s.mean, 4)) for s in run_cx3(_tiny())]
    assert run() == run()
