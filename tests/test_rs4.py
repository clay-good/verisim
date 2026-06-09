"""Smoke + structural-invariant test for RS4, the unrolled-loss trainer (SPEC-16 §5).

RS4 trains the *real* structured graph arm (torch), so this is ``skipif``-guarded and runs a tiny
instance: it asserts *structural* invariants (the right arms appear; horizons and rates are in
range; the run is deterministic; the H58 cost multiplier is monotone in depth), not the magnitude
of the unrolled lift — that is the committed figure generated on the primary host (the macOS-first
/ SPEC-10 scale discipline). The load-bearing equivalence — ``unroll_k = 1`` reduces the pushforward
to teacher forcing byte-for-byte — is pinned directly on the example builder, since it is the
property that makes the cost-1.0 anchor honest.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # RS4 trains a real graph arm; skip where torch is absent

from verisim.experiments.rs4_unrolled import RS4Config, cost_mult_for_k, run_rs4  # noqa: E402


def _tiny() -> RS4Config:
    return RS4Config(
        n_hosts=4, n_ports=2, train_seeds=(0, 1), train_steps_per_traj=20,
        graph_d_model=32, graph_mp_rounds=2, train_steps=60, refresh_every=30,
        unroll_ks=(1, 2), model_seeds=(0,), eval_seeds=(100, 101), eval_steps=12,
        one_step_seeds=(200,), one_step_steps=12, epsilons=(0.0, 0.3),
    )


def test_rs4_smoke_structural() -> None:
    stats = run_rs4(_tiny())
    assert {s.arm for s in stats} == {"teacher-forced", "unrolled-k1", "unrolled-k2"}
    for s in stats:
        assert 0.0 <= s.p_one_step <= 1.0
        assert s.p_lo <= s.p_one_step <= s.p_hi
        assert set(s.h_free) == {0.0, 0.3}
        for eps in (0.0, 0.3):
            assert 0.0 <= s.h_free[eps] <= 12.0
            assert s.h_lo[eps] <= s.h_free[eps] <= s.h_hi[eps]
        assert s.n == 1
    by = {s.arm: s for s in stats}
    assert by["teacher-forced"].unroll_k == 0 and by["teacher-forced"].cost_mult == 1.0
    # H58 cost multiplier is monotone in pushforward depth, never below the teacher-forced anchor.
    assert by["unrolled-k1"].cost_mult == 1.0 < by["unrolled-k2"].cost_mult


def test_rs4_cost_mult_law() -> None:
    assert cost_mult_for_k(1) == 1.0
    assert cost_mult_for_k(2) == 1.5
    assert cost_mult_for_k(4) == 2.5
    assert cost_mult_for_k(8) == 4.5


def test_rs4_unroll_k1_is_teacher_forcing() -> None:
    """``unroll_k = 1`` re-anchors every step, so the supervised states never leave the true
    trajectory — the pushforward dataset must equal the teacher-forced dataset byte-for-byte."""
    from verisim.net.config import scaled_net_config
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, build_unrolled_examples
    from verisim.netoracle import ReferenceNetworkOracle

    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(4, 2)
    vocab = NetVocab(net)
    wm = build_graph_model(vocab, net, d_model=32, mp_rounds=2, seed=0)
    seeds = (0, 1)
    tf = build_graph_dataset(oracle, vocab, net, driver="weighted", seeds=seeds, n_steps=16)
    unrolled = build_unrolled_examples(
        wm, oracle, vocab, net, driver="weighted", seeds=seeds, n_steps=16, unroll_k=1
    )
    assert [t for _, t in unrolled] == [t for _, t in tf]  # identical oracle targets


def test_rs4_deterministic() -> None:
    a = [(s.arm, round(s.p_one_step, 4), round(s.h_free[0.3], 4)) for s in run_rs4(_tiny())]
    b = [(s.arm, round(s.p_one_step, 4), round(s.h_free[0.3], 4)) for s in run_rs4(_tiny())]
    assert a == b
