"""Smoke + structural-invariant test for NA0, the SPEC-14 NAR diagnosis (H45).

NA0 trains a real graph arm (torch), so this is ``skipif``-guarded and runs a tiny instance: it
asserts *structural* invariants (every round 0..R appears once; accuracies/lifts are in range; the
run is deterministic), not the H45 verdict's magnitude -- that is the committed figure generated on
the primary host (the macOS-first / SPEC-10 scale discipline). The oracle reachability frontier
itself (``reach_frontiers``) is a pure deterministic read, exercised here without torch.
"""

from __future__ import annotations

import pytest

from verisim.experiments.na0 import reach_frontiers
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState, link_key


def test_reach_frontiers_is_hop_bounded_bfs() -> None:
    # A 4-host line: h0 - h1 - h2 - h3 (all up). <=r-hop frontier grows monotonically with r.
    net = scaled_net_config(4, 2)
    hosts = net.hosts
    state = NetworkState.initial(hosts)
    state.links.update({link_key("h0", "h1"), link_key("h1", "h2"), link_key("h2", "h3")})
    f = reach_frontiers(state, hosts, 3)
    # r=0: self only.
    assert [f[0][i, j] for i in range(4) for j in range(4)] == [
        1.0 if i == j else 0.0 for i in range(4) for j in range(4)
    ]
    # h0 reaches h1 at 1 hop, h2 at 2, h3 at 3 -- exactly the line's geodesics.
    assert f[1][0, 1] == 1.0 and f[1][0, 2] == 0.0
    assert f[2][0, 2] == 1.0 and f[2][0, 3] == 0.0
    assert f[3][0, 3] == 1.0
    # Monotone: every reachable pair at r stays reachable at r+1.
    for r in range(3):
        for i in range(4):
            for j in range(4):
                assert f[r][i, j] <= f[r + 1][i, j]


def test_reach_frontiers_respects_down_hosts() -> None:
    net = scaled_net_config(3, 2)
    hosts = net.hosts
    state = NetworkState.initial(hosts)
    state.links.update({link_key("h0", "h1"), link_key("h1", "h2")})
    state.hosts["h1"] = state.hosts["h1"].with_up(False)  # break the only path h0<->h2
    f = reach_frontiers(state, hosts, 2)
    assert f[2][0, 2] == 0.0  # no path through a down host
    assert f[0][1, 1] == 0.0  # a down host reaches not even itself


pytest.importorskip("torch")  # the rest trains a real graph arm; skip cleanly where torch is absent

from verisim.experiments.na0 import NA0Config, run_na0  # noqa: E402
from verisim.experiments.na5 import NA5Config, run_na5  # noqa: E402


def _tiny() -> NA0Config:
    return NA0Config(
        n_hosts=5, n_ports=2, train_seeds=(0, 1), train_steps_per_traj=20,
        graph_d_model=32, graph_mp_rounds=3, graph_iters=60, model_seeds=(0,),
        eval_seeds=(100, 101), eval_steps=20,
    )


def test_na0_smoke_structural() -> None:
    stats, p_mean, p_lo, p_hi = run_na0(_tiny())
    assert [s.round for s in stats] == [0, 1, 2, 3]  # one row per round 0..R
    assert p_lo <= p_mean <= p_hi and 0.0 <= p_mean <= 1.0
    for s in stats:
        assert 0.0 <= s.probe_acc <= 1.0
        assert 0.0 <= s.base_acc <= 1.0
        assert s.lift_lo <= s.lift <= s.lift_hi
        assert s.n_seeds == 1


def test_na0_deterministic() -> None:
    a = [(s.round, round(s.lift, 4), round(s.control_lift, 4)) for s in run_na0(_tiny())[0]]
    b = [(s.round, round(s.lift, 4), round(s.control_lift, 4)) for s in run_na0(_tiny())[0]]
    assert a == b


def _na5_tiny() -> NA5Config:
    return NA5Config(
        n_hosts=5, n_ports=2, train_seeds=(0, 1), train_steps_per_traj=20,
        graph_d_model=32, graph_mp_rounds=3, graph_iters=60, model_seeds=(0,),
        probe_seeds=(100, 101), probe_steps=20,
        rollout_seeds=(200, 201, 202, 203), rollout_steps=6, depth_buckets=3,
    )


def test_na5_smoke_structural() -> None:
    stats = run_na5(_na5_tiny())
    assert 1 <= len(stats) <= 3  # one row per non-empty depth bucket
    for s in stats:
        assert 0.0 <= s.frozen_own <= 1.0
        assert 0.0 <= s.refit_own <= 1.0
        assert 0.0 <= s.tracks_truth <= 1.0
        assert 0.0 <= s.divergence <= 1.0
        assert s.frozen_lo <= s.frozen_own <= s.frozen_hi
        assert s.n_seeds == 1
    # The refit probe (oracle-relabeled on drifted states) decodes its own state at least as well as
    # the frozen in-distribution probe at the deepest bucket -- the control's defining property.
    assert stats[-1].refit_own >= stats[-1].frozen_own - 1e-6


def test_na5_deterministic() -> None:
    a = [(s.depth, round(s.frozen_own, 4), round(s.refit_own, 4)) for s in run_na5(_na5_tiny())]
    b = [(s.depth, round(s.frozen_own, 4), round(s.refit_own, 4)) for s in run_na5(_na5_tiny())]
    assert a == b
