"""Smoke + structural-invariant test for RS7, the cross-world host fork (SPEC-16 §5, H59).

RS7 trains the *real* host factored arm four ways (torch), so this is ``skipif``-guarded and runs a
tiny instance: it asserts *structural* invariants (the four arms appear; horizons/rates are in
range; the run is deterministic), not the transfer verdict's magnitude — that is the committed
figure on the primary host (the SPEC-10 discipline). The load-bearing equivalence — the host
unrolled builder at ``unroll_k = 1`` reduces to the plain host teacher-forced dataset
byte-for-byte — is pinned on the builder, making the unrolled arm's ``k=1`` limit honest.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # RS7 trains a real host graph arm; skip where torch is absent

from verisim.experiments.rs7_host import ARMS, RS7Config, run_rs7  # noqa: E402


def _tiny() -> RS7Config:
    return RS7Config(
        max_pid=32, train_seeds=(0, 1), train_steps_per_traj=16, graph_d_model=32,
        graph_mp_rounds=2, graph_iters=40, sf_rounds=2, unroll_rounds=2, unroll_k=2,
        model_seeds=(0,), eval_seeds=(100, 101), eval_steps=12, one_step_seeds=(200,),
        one_step_steps=12, epsilons=(0.0, 0.3),
    )


def test_rs7_smoke_structural() -> None:
    stats = run_rs7(_tiny())
    assert {s.arm for s in stats} == set(ARMS)
    for s in stats:
        assert 0.0 <= s.p_one_step <= 1.0
        assert s.p_lo <= s.p_one_step <= s.p_hi
        assert set(s.h_free) == {0.0, 0.3}
        for eps in (0.0, 0.3):
            assert 0.0 <= s.h_free[eps] <= 12.0
            assert s.h_lo[eps] <= s.h_free[eps] <= s.h_hi[eps]
        assert s.n == 1


def test_rs7_deterministic() -> None:
    def run() -> list[tuple[str, float, float]]:
        return [(s.arm, round(s.p_one_step, 4), round(s.h_free[0.3], 4)) for s in run_rs7(_tiny())]
    assert run() == run()


def test_rs7_host_unroll_k1_is_teacher_forcing() -> None:
    """``unroll_k = 1`` re-anchors every step, so the supervised host states never leave the true
    trajectory — the pushforward dataset must equal the plain host teacher-forced dataset."""
    from verisim.host.config import DEFAULT_HOST_CONFIG
    from verisim.hostmodel import HostVocab
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, build_unrolled_host_examples
    from verisim.hostoracle.reference import ReferenceHostOracle

    oracle = ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=32)
    wm = build_host_graph_model(vocab, host, max_pid=32, d_model=32, mp_rounds=2, seed=0)
    seeds = (0, 1)
    tf = build_host_graph_dataset(oracle, vocab, host, driver="forky", seeds=seeds, n_steps=16)
    unrolled = build_unrolled_host_examples(
        wm, oracle, vocab, host, driver="forky", seeds=seeds, n_steps=16, unroll_k=1
    )
    assert [t for _, t in unrolled] == [t for _, t in tf]  # identical oracle targets
