"""Partial-observation propose-verify-correct loop invariants (SPEC-5 §8, NW5).

Mirrors v0's ``test_loop`` for the full-consultation mode, plus the network-specific
partial-observation invariants that have no v0 analogue (§5.3, §8.2, §8.3):

  - ``ρ = 1`` + full consult reproduces the oracle exactly (``H_ε = T``).
  - a perfect model never drifts even at ``ρ = 0``; a null model drifts.
  - the budget is never exceeded; the spend-down backstop spends it exactly.
  - a *probe* reveals only one host's subgraph, costs fewer oracle-bits than a full
    consult, and corrects strictly less -- so probe-mode horizon is no greater than
    full-mode horizon at equal ``ρ`` (the §8.3 no-identity-collapse property).
"""

from __future__ import annotations

import random

from verisim.loop.policy import FixedInterval, Never, fixed_interval_for_rho
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.net.state import HostState, NetworkState
from verisim.netdata import NetDriver
from verisim.netdelta.apply import apply
from verisim.netloop import (
    NetNullModel,
    NetOracleBackedModel,
    PartialNetOracle,
    RandomProbe,
    RoundRobinProbe,
    belief_filter,
    budget_for_rho,
    ground_truth_rollout,
    observe_host,
    run_net_rollout,
)
from verisim.netloop.observe import full_bits
from verisim.netmetrics.divergence import divergence
from verisim.netoracle import ReferenceNetworkOracle

CONFIG = DEFAULT_NET_CONFIG


def make_net_actions(driver_name: str, seed: int, n: int) -> list[NetAction]:
    """A seeded action sequence by rolling a driver against the oracle."""
    oracle = ReferenceNetworkOracle()
    driver = NetDriver(driver_name, CONFIG, random.Random(seed))
    state = NetworkState.initial(CONFIG.hosts)
    actions: list[NetAction] = []
    for _ in range(n):
        action = driver.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def _partial() -> PartialNetOracle:
    return PartialNetOracle(ReferenceNetworkOracle())


def s0() -> NetworkState:
    return NetworkState.initial(CONFIG.hosts)


# --- full-consultation mode (mirrors v0 test_loop) ---------------------------


def test_rho1_full_consult_reproduces_oracle_exactly():
    oracle = _partial()
    for driver_name in ("uniform", "weighted", "adversarial"):
        actions = make_net_actions(driver_name, seed=1, n=40)
        record = run_net_rollout(
            NetNullModel(), oracle, s0(), actions, FixedInterval(1),
            epsilon=0.0, budget=len(actions),
        )
        assert all(d == 0.0 for d in record.divergences)
        assert record.faithful_horizon == len(actions)


def test_perfect_model_never_drifts_at_rho0():
    oracle = _partial()
    actions = make_net_actions("adversarial", seed=2, n=50)
    record = run_net_rollout(
        NetOracleBackedModel(ReferenceNetworkOracle()),
        oracle, s0(), actions, Never(), epsilon=0.0, budget=0,
    )
    assert record.oracle_calls == 0
    assert all(d == 0.0 for d in record.divergences)
    assert record.faithful_horizon == len(actions)


def test_rho0_matches_unaided_model_rollout():
    oracle = _partial()
    model = NetNullModel()
    actions = make_net_actions("weighted", seed=3, n=30)
    gt = ground_truth_rollout(oracle, s0(), actions)

    state = s0()
    expected: list[float] = []
    for t, action in enumerate(actions):
        state = apply(state, model.predict_delta(state, action))
        expected.append(divergence(gt[t + 1], state))

    record = run_net_rollout(model, oracle, s0(), actions, Never(), epsilon=0.0, budget=0)
    assert record.divergences == expected


def test_null_model_drifts_at_rho0():
    oracle = _partial()
    actions = make_net_actions("weighted", seed=4, n=30)
    record = run_net_rollout(NetNullModel(), oracle, s0(), actions, Never(), epsilon=0.0)
    assert record.faithful_horizon < len(actions)


def test_budget_is_never_exceeded():
    oracle = _partial()
    actions = make_net_actions("uniform", seed=5, n=40)
    for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        budget = budget_for_rho(rho, len(actions))
        record = run_net_rollout(
            NetNullModel(), oracle, s0(), actions, fixed_interval_for_rho(rho),
            epsilon=0.0, budget=budget,
        )
        assert record.oracle_calls <= budget
        assert record.oracle_calls == sum(record.consultation_schedule)
        assert len(record.divergences) == len(actions)


def test_spend_down_backstop_spends_full_budget():
    oracle = _partial()
    actions = make_net_actions("weighted", seed=9, n=5)
    record = run_net_rollout(
        NetNullModel(), oracle, s0(), actions, Never(), epsilon=0.0, budget=2
    )
    assert record.oracle_calls == 2
    assert record.consultation_schedule == [False, False, False, True, True]


# --- partial-observation mode (no v0 analogue, SPEC-5 §5.3 / §8.2 / §8.3) ----


def test_observe_host_returns_true_local_subgraph():
    state = NetworkState(
        hosts={
            "h0": HostState(up=True, services=(80,), fw_deny=("h1",)),
            "h1": HostState(up=False),
            "h2": HostState(),
        },
        links={("h0", "h1"), ("h1", "h2")},
        flows={("h2", "h0", 80), ("h0", "h2", 22)},
    )
    obs = observe_host(state, "h0")
    assert obs.present and obs.up
    assert obs.services == (80,)
    assert obs.fw_deny == ("h1",)
    assert obs.links == frozenset({("h0", "h1")})  # only links incident to h0
    assert obs.flows == frozenset({("h2", "h0", 80), ("h0", "h2", 22)})
    # An absent host yields an empty, not-present observation.
    assert not observe_host(state, "ghost").present


def test_belief_filter_snaps_only_observed_subgraph():
    predicted = NetworkState(
        hosts={"h0": HostState(services=()), "h1": HostState(), "h2": HostState()},
        links={("h1", "h2")},  # a belief about an edge not touching h0 -- must survive
        flows=set(),
    )
    truth = NetworkState(
        hosts={"h0": HostState(services=(80,)), "h1": HostState(), "h2": HostState()},
        links={("h0", "h1"), ("h1", "h2")},
        flows={("h2", "h0", 80)},
    )
    obs = observe_host(truth, "h0")
    corrected = belief_filter(predicted, obs)
    # h0's subgraph is now truth ...
    assert corrected.hosts["h0"].services == (80,)
    assert ("h0", "h1") in corrected.links
    assert ("h2", "h0", 80) in corrected.flows
    # ... but the unobserved belief edge (h1,h2) is retained verbatim.
    assert ("h1", "h2") in corrected.links


def test_probe_costs_fewer_bits_than_full_and_corrects_less():
    """A probe reveals one host; a full consult reveals everything. Probe ⊂ full."""
    oracle = _partial()
    actions = make_net_actions("adversarial", seed=11, n=40)

    full = run_net_rollout(
        NetNullModel(), oracle, s0(), actions, FixedInterval(1), epsilon=0.0, budget=len(actions)
    )
    probe = run_net_rollout(
        NetNullModel(), oracle, s0(), actions, FixedInterval(1), epsilon=0.0,
        budget=len(actions), probe_policy=RoundRobinProbe(CONFIG.hosts),
    )
    # Full consultation at ρ=1 is exact; a one-host probe corrects strictly less, so it
    # cannot match it -- the §8.3 no-identity-collapse property partial observability buys.
    assert full.faithful_horizon == len(actions)
    assert probe.faithful_horizon < len(actions)
    # And a probe is cheaper per consult than a full consult (the §9.4 cost denominator).
    assert probe.config["oracle_bits"] < full.config["oracle_bits"]


def test_probe_policies_select_valid_hosts():
    state = s0()
    rr = RoundRobinProbe(CONFIG.hosts)
    picks = [rr.select(state) for _ in range(len(CONFIG.hosts) + 1)]
    assert picks[: len(CONFIG.hosts)] == list(CONFIG.hosts)  # one full cycle, in order
    assert picks[len(CONFIG.hosts)] == CONFIG.hosts[0]  # then wraps
    rp = RandomProbe(CONFIG.hosts, random.Random(0))
    assert all(rp.select(state) in CONFIG.hosts for _ in range(20))


def test_full_bits_counts_state_facts():
    state = NetworkState(
        hosts={"h0": HostState(services=(80,)), "h1": HostState()},
        links={("h0", "h1")},
        flows=set(),
    )
    # 2 host-up facts + 1 service + 1 link + clock + exit = 6.
    assert full_bits(state) == 6
