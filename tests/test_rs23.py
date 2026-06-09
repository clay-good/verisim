"""Smoke + structural-invariant tests for RS2 (scheduled-sampling sweep) and RS3 (noise grid).

Both train the *real* structured graph arm (torch), so this is ``skipif``-guarded and runs tiny
instances asserting *structural* invariants (the swept axes appear; horizons/rates are in range; the
runs are deterministic), not the magnitude of any tradeoff — that is the committed figure on the
primary host (the SPEC-10 scale discipline). The load-bearing equivalence — ``corrupt_state`` and
``build_graph_dataset`` at ``magnitude=1`` are byte-identical to the pre-RS3 single-mutation
behaviour — is pinned directly, since the new magnitude knob must not perturb any committed caller.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")  # RS2/RS3 train a real graph arm; skip where torch is absent

from verisim.experiments.rs2_scheduled import RS2Config, run_rs2  # noqa: E402
from verisim.experiments.rs3_noise import RS3Config, run_rs3  # noqa: E402


def _rs2_tiny() -> RS2Config:
    return RS2Config(
        n_hosts=4, n_ports=2, train_seeds=(0, 1), train_steps_per_traj=20,
        graph_d_model=32, graph_mp_rounds=2, train_steps=60, refresh_every=30,
        sample_probs=(0.0, 0.5), model_seeds=(0,), eval_seeds=(100, 101), eval_steps=12,
        one_step_seeds=(200,), one_step_steps=12, epsilons=(0.0, 0.3), headline_epsilon=0.3,
    )


def _rs3_tiny() -> RS3Config:
    return RS3Config(
        n_hosts=4, n_ports=2, train_seeds=(0, 1), train_steps_per_traj=20,
        graph_d_model=32, graph_mp_rounds=2, train_steps=60,
        noise_probs=(0.0, 0.3), magnitudes=(1, 2), model_seeds=(0,),
        eval_seeds=(100, 101), eval_steps=12, one_step_seeds=(200,), one_step_steps=12,
        epsilons=(0.0, 0.3), headline_epsilon=0.3,
    )


def test_rs2_smoke_structural() -> None:
    stats = run_rs2(_rs2_tiny())
    assert {s.sample_prob for s in stats} == {0.0, 0.5}
    for s in stats:
        assert 0.0 <= s.p_one_step <= 1.0
        assert s.p_lo <= s.p_one_step <= s.p_hi
        assert set(s.h_free) == {0.0, 0.3}
        for eps in (0.0, 0.3):
            assert 0.0 <= s.h_free[eps] <= 12.0
            assert s.h_lo[eps] <= s.h_free[eps] <= s.h_hi[eps]
        assert s.n == 1


def test_rs2_deterministic() -> None:
    def run() -> list[tuple[float, float, float]]:
        return [(s.sample_prob, round(s.p_one_step, 4), round(s.h_free[0.3], 4))
                for s in run_rs2(_rs2_tiny())]
    assert run() == run()


def test_rs3_smoke_structural() -> None:
    stats = run_rs3(_rs3_tiny())
    # noise_prob=0 collapses to one baseline cell; noise_prob>0 keeps the full magnitude column.
    cells = {(s.noise_prob, s.magnitude) for s in stats}
    assert (0.0, 1) in cells and (0.3, 1) in cells and (0.3, 2) in cells
    assert (0.0, 2) not in cells  # no redundant baseline retrain
    for s in stats:
        assert 0.0 <= s.p_one_step <= 1.0
        assert s.p_lo <= s.p_one_step <= s.p_hi
        assert 0.0 <= s.h_free <= 12.0
        assert s.h_lo <= s.h_free <= s.h_hi
        assert s.n == 1


def test_rs3_deterministic() -> None:
    a = [(s.noise_prob, s.magnitude, round(s.h_free, 4)) for s in run_rs3(_rs3_tiny())]
    b = [(s.noise_prob, s.magnitude, round(s.h_free, 4)) for s in run_rs3(_rs3_tiny())]
    assert a == b


def test_corrupt_state_magnitude_1_is_byte_identical() -> None:
    """The RS3 magnitude knob must not perturb any committed caller: ``magnitude=1`` must reproduce
    the original single-mutation ``corrupt_state`` draw-for-draw (same RNG sequence + result)."""
    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netmodel.graph_train import corrupt_state

    net = scaled_net_config(5, 3)
    state = NetworkState.initial(net.hosts)
    a = corrupt_state(state, net, random.Random(42))  # default magnitude=1
    b = corrupt_state(state, net, random.Random(42), magnitude=1)
    assert a == b
    # magnitude=2 stacks a second mutation, so it must differ from the single-mutation corruption.
    c = corrupt_state(state, net, random.Random(42), magnitude=2)
    assert c != a


def test_build_graph_dataset_magnitude_1_is_byte_identical() -> None:
    """``build_graph_dataset`` with the new ``noise_magnitude=1`` default = the pre-RS3 call."""
    from verisim.net.config import scaled_net_config
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_train import build_graph_dataset
    from verisim.netoracle import ReferenceNetworkOracle

    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(5, 3)
    vocab = NetVocab(net)
    a = build_graph_dataset(  # default noise_magnitude=1
        oracle, vocab, net, driver="weighted", seeds=(0, 1), n_steps=16, noise_prob=0.4,
        noise_seed=7,
    )
    b = build_graph_dataset(
        oracle, vocab, net, driver="weighted", seeds=(0, 1), n_steps=16, noise_prob=0.4,
        noise_seed=7, noise_magnitude=1,
    )
    assert [t for _, t in a] == [t for _, t in b]
