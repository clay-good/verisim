"""UA0 containment-env tests (SPEC-20 §2-3, milestone UA0).

The contract is the MDP mechanics and the three-backend seam -- all torch-free (the oracle backend
needs no model), so this runs in CI without the RL stack:

  - the defender action set compiles to valid SPEC-5 NetActions;
  - the scripted adversary only spreads to exposed, connected, uncompromised hosts, and isolating a
    compromised host's neighbours halts the spread (containment actually works);
  - the seeded topology is connected and reproducible;
  - the env runs an episode against the oracle backend, the observation shape is stable, and
    containment is in [0, 1];
  - GroundedBackend at ρ=1 reduces to the oracle, and rejects out-of-range ρ.
"""

from __future__ import annotations

import random

import pytest

from verisim.acd.containment import (
    ContainmentConfig,
    ContainmentEnv,
    DefenderAction,
    GroundedBackend,
    OracleBackend,
    adversary_spread,
    containment_fraction,
    legal_actions,
    seed_topology,
)
from verisim.net.state import connected_hosts
from verisim.netoracle import ReferenceNetworkOracle


def test_defender_actions_compile_to_net_actions():
    assert DefenderAction("isolate", host="h1").to_net_action().name == "host_down"
    assert DefenderAction("patch", host="h1", port=22).to_net_action().name == "svc_down"
    assert DefenderAction("noop").to_net_action().name == "advance"
    with pytest.raises(ValueError):
        DefenderAction("bogus").to_net_action()


def test_seed_topology_is_connected_and_reproducible():
    cfg = ContainmentConfig.smoke()
    net_a, comp_a = seed_topology(cfg, random.Random(7))
    net_b, comp_b = seed_topology(cfg, random.Random(7))
    assert comp_a == comp_b and net_a.links == net_b.links  # reproducible
    assert len(comp_a) == cfg.n_initial_compromised
    # the spanning chain makes every host reachable from h0
    assert connected_hosts(net_a, "h0") == set(net_a.hosts)


def test_adversary_only_spreads_to_exposed_connected_hosts():
    cfg = ContainmentConfig.smoke()
    net, _ = seed_topology(cfg, random.Random(1))
    # a host with no service is never a beachhead even if connected
    bare = {h for h, hs in net.hosts.items() if not hs.services}
    compromised = frozenset({"h0"})
    after = adversary_spread(net, compromised, random.Random(0))
    newly = after - compromised
    for h in newly:
        assert h not in bare  # only exposed (service-bearing) hosts fall


def test_isolation_halts_spread():
    cfg = ContainmentConfig.smoke()
    net, compromised = seed_topology(cfg, random.Random(3))
    # bring every non-compromised host down -> no beachheads -> spread halts
    for h in list(net.hosts):
        if h not in compromised:
            net.hosts[h] = net.hosts[h].with_up(False)
    assert adversary_spread(net, compromised, random.Random(0)) == compromised


def test_containment_fraction_bounds():
    cfg = ContainmentConfig.smoke()
    net, _ = seed_topology(cfg, random.Random(0))
    assert containment_fraction(net, frozenset()) == 1.0
    assert containment_fraction(net, frozenset(net.hosts)) == 0.0


def test_env_episode_runs_against_oracle_backend():
    cfg = ContainmentConfig.smoke()
    env = ContainmentEnv(cfg, OracleBackend(ReferenceNetworkOracle()))
    obs = env.reset(seed=5)
    assert len(obs) == cfg.n_hosts * 4  # four features per host
    steps = 0
    done = False
    while not done:
        actions = legal_actions(env._net, cfg)
        out = env.step(actions[0])  # always noop -> the adversary runs unchecked
        assert 0.0 <= out.info["containment"] <= 1.0
        assert len(out.observation) == len(obs)
        done = out.done
        steps += 1
    assert steps == cfg.episode_steps


def test_grounded_backend_rho1_is_oracle_and_validates():
    oracle = ReferenceNetworkOracle()
    cfg = ContainmentConfig.smoke()
    net, _ = seed_topology(cfg, random.Random(2))
    action = DefenderAction("isolate", host="h1").to_net_action()
    grounded = GroundedBackend(model=_DummyModel(), oracle=oracle, rho=1.0)
    grounded.reset()
    # at ρ=1 every step corrects to the oracle, so the dummy model is never consulted
    assert grounded.step(net, action) == oracle.step(net, action).state
    with pytest.raises(ValueError):
        GroundedBackend(model=_DummyModel(), oracle=oracle, rho=1.5)


class _DummyModel:
    """A model that predicts no change -- proves ρ=1 never consults it (the test above)."""

    def predict_delta(self, state, action):
        return []
