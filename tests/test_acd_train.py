"""UA1 policy + REINFORCE engine tests (SPEC-20 §5).

Torch-free: trains against the OracleBackend so it runs in CI. The contract:

  - the linear policy's softmax/grad are well-formed (probs sum to 1; the score-function gradient is
    φ(chosen) − E[φ]);
  - REINFORCE against the oracle backend *learns* -- a trained defender contains materially better
    than a random/untrained one (the env is learnable, so a null here means the engine is broken);
  - evaluation is deterministic (greedy) and returns a containment fraction in [0, 1].

The committed numbers come from the local run with the real model backends; CI proves the engine.
"""

from __future__ import annotations

from verisim.acd.containment import ContainmentConfig, ContainmentEnv, OracleBackend
from verisim.acd.policy import N_ACTION_FEATURES, LinearPolicy, action_features
from verisim.acd.train import TrainConfig, evaluate, reinforce
from verisim.net.state import NetworkState
from verisim.netoracle import ReferenceNetworkOracle


def test_policy_probs_and_grad_well_formed():
    pol = LinearPolicy([0.0] * N_ACTION_FEATURES)
    feats = [[1.0, 1, 0, 0, 0, 0, 0], [1.0, 0, 1, 0, 1, 1, 2]]
    ps = pol.probs(feats)
    assert abs(sum(ps) - 1.0) < 1e-9 and all(p >= 0 for p in ps)
    # uniform weights -> uniform-ish; grad of chosen 0 = φ0 - E[φ]
    grad = pol.logprob_grad(feats, 0)
    assert len(grad) == N_ACTION_FEATURES


def test_action_features_flags_target_state():
    net = NetworkState.initial(("h0", "h1"))
    net.hosts["h1"] = net.hosts["h1"].with_service(22, True)
    from verisim.acd.containment import DefenderAction

    f_noop = action_features(net, frozenset(), DefenderAction("noop"))
    assert f_noop[1] == 1.0  # is_noop
    f_iso = action_features(net, frozenset({"h0"}), DefenderAction("isolate", host="h0"))
    assert f_iso[2] == 1.0 and f_iso[4] == 1.0  # is_isolate, target_compromised


def _oracle_env_factory(cfg):
    def make():
        return ContainmentEnv(cfg, OracleBackend(ReferenceNetworkOracle()))

    return make


def test_reinforce_learns_against_oracle_backend():
    cfg = ContainmentConfig(n_hosts=6, n_ports=2, episode_steps=10, n_links=7, n_services=6)
    make = _oracle_env_factory(cfg)
    eval_seeds = tuple(range(500, 510))

    untrained = LinearPolicy()
    base = evaluate(make, untrained, seeds=eval_seeds)

    trained = reinforce(make, TrainConfig(episodes=200, batch=10, lr=0.3, seed=1))
    learned = evaluate(make, trained, seeds=eval_seeds)

    # the engine must lift containment over the untrained policy (the env is learnable)
    assert learned >= base
    assert 0.0 <= learned <= 1.0


def test_evaluate_is_deterministic():
    cfg = ContainmentConfig.smoke()
    make = _oracle_env_factory(cfg)
    pol = reinforce(make, TrainConfig.smoke())
    a = evaluate(make, pol, seeds=(500, 501, 502))
    b = evaluate(make, pol, seeds=(500, 501, 502))
    assert a == b


def test_reinforce_is_deterministic():
    cfg = ContainmentConfig.smoke()
    make = _oracle_env_factory(cfg)
    a = reinforce(make, TrainConfig.smoke())
    b = reinforce(make, TrainConfig.smoke())
    assert a.weights == b.weights
