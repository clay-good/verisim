"""UA1 -- the defender policy + REINFORCE engine (SPEC-20 §5, milestone UA1).

The agent SPEC-20 trains is deliberately the *smallest* policy that does the job (SPEC-20 §5): a
linear softmax over hand-built action features, trained with REINFORCE and a moving-average baseline
-- the same minimal-optimizer discipline SPEC-2 §5.3's RLVR kept. The honest reason: a fancy policy
would confound "the agent is clever" with "the world model is a good training environment," and only
the latter is the result (UA2/H74). So the policy is torch-free and tiny; what varies in the
experiment is the *environment backend* it trains against, never the learner.

The action space is state-dependent (you can only isolate up hosts, patch live services), so the
policy scores each *legal* action by a linear function of its features and samples from the softmax.
REINFORCE then nudges the weights toward actions that preceded high containment return.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from verisim.net.state import NetworkState, connected_hosts

from .containment import DefenderAction

# Action-feature layout (the policy's input per candidate action):
#   [bias, is_noop, is_isolate, is_patch, target_compromised, target_exposed, target_n_services]
N_ACTION_FEATURES = 7


def action_features(
    net: NetworkState, compromised: frozenset[str], action: DefenderAction
) -> list[float]:
    """Featurize one candidate defender action against the current (net, compromise) state.

    The features are intentionally legible: the action kind (one-hot) and, for host-targeted
    actions, whether the target is compromised, exposed to the spread, and how many services it runs
    -- everything a sensible containment heuristic would key on, left for the policy to weight.
    """
    is_noop = 1.0 if action.kind == "noop" else 0.0
    is_isolate = 1.0 if action.kind == "isolate" else 0.0
    is_patch = 1.0 if action.kind == "patch" else 0.0

    target_compromised = 0.0
    target_exposed = 0.0
    target_services = 0.0
    if action.host:
        target_compromised = 1.0 if action.host in compromised else 0.0
        exposed_set: set[str] = set()
        for src in compromised:
            exposed_set |= connected_hosts(net, src)
        target_exposed = (
            1.0 if (action.host in exposed_set and action.host not in compromised) else 0.0
        )
        hs = net.hosts.get(action.host)
        target_services = float(len(hs.services)) if hs else 0.0

    return [1.0, is_noop, is_isolate, is_patch, target_compromised, target_exposed, target_services]


def _softmax(scores: list[float]) -> list[float]:
    hi = max(scores)
    exps = [math.exp(s - hi) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class LinearPolicy:
    """A linear softmax policy over action features (the smallest policy that does the job).

    The weight dimension is inferred from ``weights`` -- the default is the basic featurizer's
    ``N_ACTION_FEATURES``, but the UA6 structural featurizer passes a longer vector.
    """

    weights: list[float] = field(default_factory=lambda: [0.0] * N_ACTION_FEATURES)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must be non-empty")

    def _score(self, feats: list[float]) -> float:
        return sum(w * f for w, f in zip(self.weights, feats, strict=True))

    def probs(self, feats_list: list[list[float]]) -> list[float]:
        """The action distribution over a list of candidate-action feature vectors."""
        return _softmax([self._score(f) for f in feats_list])

    def sample(self, feats_list: list[list[float]], rng: random.Random) -> int:
        """Sample an action index from the policy; returns the chosen candidate's index."""
        ps = self.probs(feats_list)
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(ps):
            acc += p
            if r <= acc:
                return i
        return len(ps) - 1

    def greedy(self, feats_list: list[list[float]]) -> int:
        """The argmax action (for deterministic evaluation)."""
        scores = [self._score(f) for f in feats_list]
        return max(range(len(scores)), key=lambda i: scores[i])

    def logprob_grad(self, feats_list: list[list[float]], chosen: int) -> list[float]:
        """∇_w log π(chosen | ·) = φ(chosen) − Σ_a π(a) φ(a) -- the REINFORCE score function."""
        dim = len(self.weights)
        ps = self.probs(feats_list)
        expected = [0.0] * dim
        for p, feats in zip(ps, feats_list, strict=True):
            for k in range(dim):
                expected[k] += p * feats[k]
        chosen_feats = feats_list[chosen]
        return [chosen_feats[k] - expected[k] for k in range(dim)]
