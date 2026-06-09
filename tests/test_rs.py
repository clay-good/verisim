"""Smoke + structural-invariant test for RS1 free-oracle DAgger (SPEC-16 §5).

RS1 trains the *real* flat ``M_θ`` (torch), so this is ``skipif``-guarded and runs a tiny instance:
it asserts *structural* invariants (the right arms/rounds appear; horizons and rates are in range;
the run is deterministic), not the DAgger lift's magnitude -- that is the committed figure
generated on the primary host (the macOS-first / SPEC-10 scale discipline). The relabel
correctness (the DAgger target equals the oracle's exact delta at the drifted state) is the SCM
property already pinned by ``test_causal.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # RS1 trains a real GPT; skip cleanly where torch is absent

from verisim.experiments.rs1_dagger import RS1Config, run_rs1  # noqa: E402


def _tiny() -> RS1Config:
    return RS1Config(
        n_hosts=4, n_ports=2, n_examples=80, total_steps=40, n_rounds=2,
        n_layer=1, n_head=2, n_embd=32, block_size=96, n_steps=20,
        epsilon=0.05, train_seeds=(0, 1, 2, 3), eval_seeds=(100, 101),
        one_step_seeds=(200,), model_seeds=(0,),
    )


def test_rs1_smoke_structural() -> None:
    stats = run_rs1(_tiny())
    arms = {(s.arm, s.dagger_round) for s in stats}
    assert ("teacher-forced", 0) in arms
    assert ("dagger", 0) in arms and ("dagger", 1) in arms
    for s in stats:
        assert 0.0 <= s.h_free <= 20.0
        assert s.h_lo <= s.h_free <= s.h_hi
        assert 0.0 <= s.p_one_step <= 1.0
        assert s.n == 1


def test_rs1_deterministic() -> None:
    a = [(s.arm, s.dagger_round, round(s.h_free, 4)) for s in run_rs1(_tiny())]
    b = [(s.arm, s.dagger_round, round(s.h_free, 4)) for s in run_rs1(_tiny())]
    assert a == b
