"""UA0 -- the defensive containment task + the three backends (SPEC-20 §2-3, milestone UA0).

The downstream task (SPEC-20 §2): a defender contains a spreading adversary on the SPEC-5 network
world. The world models reachability (hosts, links, services, firewall) but not *compromise*,
so the env tracks the compromise set in its own episode state and computes spread on top of the
shipped reachability functions ([`net.state.connected_hosts`](../net/state.py)). One scripted,
*non-learned* adversary (the §13 ethics commitment -- this trains a defender only); one learned
defender whose actions modify reachability to cut the spread.

  - **adversary (scripted):** each step, compromise spreads to one host that is link-connected to an
    already-compromised host and exposes a listening service (a beachhead). Deterministic given the
    episode seed.
  - **defender (the agent):** `isolate(host)` brings a host down (`host_down`, cutting all its
    connectivity); `patch(host, port)` closes a service (`svc_down`, removing a spread target);
    `noop` (`advance`) spends nothing. Each non-noop action pays an operational cost (isolating
    everything trivially "wins" and is penalized).
  - **reward (task success):** containment -- the fraction of hosts kept uncompromised -- minus the
    per-action operational cost. Orthogonal to faithful horizon (SPEC-20 §2).

The three backends (SPEC-20 §3) differ *only* in how the network state evolves under the defender's
action -- which is the entire experiment:

  - :class:`OracleBackend` -- the exact oracle (`E_oracle`, reality / the expensive baseline);
  - :class:`GroundedBackend` -- the trained `M_θ` with oracle-in-the-loop correction at budget ρ
    (`E_grounded`, cheap and faithful -- the product);
  - :class:`FreeBackend` -- the same `M_θ`, never corrected (`E_free`, the oracle-free ablation that
    isolates what grounding buys).

The adversary always spreads on the *backend's* (possibly model-predicted) reachability, so a
backend hands the adversary -- and any policy trained against it -- a wrong world. That is the
mechanism UA2/H74 measures. CPU-only, deterministic, seeded; no torch import at module load (the
backends that need a model take it as an argument).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol

from verisim.net.action import NetAction
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState, connected_hosts, link_key
from verisim.netdelta.apply import apply
from verisim.netoracle.base import NetOracle

if TYPE_CHECKING:
    from verisim.netloop.model import NetModel


@dataclass(frozen=True)
class ContainmentConfig:
    """The containment MDP instance: topology size, episode length, and the operational cost.

    The default `(n_hosts=5, n_ports=3)` matches `DEFAULT_NET_CONFIG` -- the world the flagship
    `M_θ` is trained on -- so a model backend can predict for it. **Invariant:** with a model
    backend, the env world must be a *sub-world* of the model's training world (its hosts/ports a
    subset), or the model's tokenizer rejects an unseen host (the env is parameterized for the
    oracle backend, which works at any size, but the trained model only knows the world it saw).
    """

    n_hosts: int = 5  # == DEFAULT_NET_CONFIG (the flagship model's world)
    n_ports: int = 3
    episode_steps: int = 12
    action_cost: float = 0.02
    n_links: int = 7  # topology density (edges seeded per episode)
    n_services: int = 6  # listening (host, port) services seeded per episode
    n_initial_compromised: int = 1
    cut_budget: int | None = None  # max isolations per episode (None = unlimited); the UA6 lever

    def net(self) -> NetConfig:
        return scaled_net_config(self.n_hosts, self.n_ports)

    @staticmethod
    def smoke() -> ContainmentConfig:
        return ContainmentConfig(n_hosts=4, n_ports=2, episode_steps=6, n_links=4, n_services=3)


# --- the defender's action set --------------------------------------------------------------------


@dataclass(frozen=True)
class DefenderAction:
    """A defensive operation -- compiled to a SPEC-5 :class:`NetAction` the backend evolves."""

    kind: str  # "isolate" | "patch" | "noop"
    host: str = ""
    port: int = 0

    def to_net_action(self) -> NetAction:
        if self.kind == "isolate":
            return NetAction(raw=f"host_down {self.host}", name="host_down", args=(self.host,))
        if self.kind == "patch":
            return NetAction(
                raw=f"svc_down {self.host} {self.port}", name="svc_down",
                args=(self.host, str(self.port)),
            )
        if self.kind == "noop":
            return NetAction(raw="advance", name="advance", args=())
        raise ValueError(f"unknown defender action kind: {self.kind}")


def legal_actions(net: NetworkState, config: ContainmentConfig) -> list[DefenderAction]:
    """The defender's legal moves: isolate any up host, patch any live service, or noop."""
    actions: list[DefenderAction] = [DefenderAction("noop")]
    for host, hs in sorted(net.hosts.items()):
        if hs.up:
            actions.append(DefenderAction("isolate", host=host))
            for port in hs.services:
                actions.append(DefenderAction("patch", host=host, port=port))
    return actions


# --- the scripted adversary + compromise bookkeeping ----------------------------------------------


def _beachheads(net: NetworkState, compromised: frozenset[str]) -> list[str]:
    """Uncompromised up hosts link-connected to a compromised host that expose a service."""
    reachable: set[str] = set()
    for src in compromised:
        reachable |= connected_hosts(net, src)
    out = [
        h for h in sorted(net.hosts)
        if h not in compromised and net.hosts[h].up and net.hosts[h].services and h in reachable
    ]
    return out


def adversary_spread(
    net: NetworkState, compromised: frozenset[str], rng: random.Random
) -> frozenset[str]:
    """Compromise one new beachhead host (the scripted spreader); no-op if none are exposed."""
    candidates = _beachheads(net, compromised)
    if not candidates:
        return compromised
    victim = rng.choice(candidates)
    return compromised | {victim}


def containment_fraction(net: NetworkState, compromised: frozenset[str]) -> float:
    """The task-success metric: the fraction of hosts kept uncompromised."""
    n = len(net.hosts)
    return (n - len(compromised)) / n if n else 1.0


# --- the topology seeder (a non-trivial network per episode) --------------------------------------


def seed_topology(
    config: ContainmentConfig, rng: random.Random
) -> tuple[NetworkState, frozenset[str]]:
    """A seeded connected-ish network with services + an initial compromise (the episode start).

    Links are drawn to first chain the hosts (guaranteeing the graph is connected, so there is a
    containment problem) and then add random extra edges to ``n_links``; services scatter listening
    ports; the initial compromise seeds the spread. Deterministic given ``rng``.
    """
    net = scaled_net_config(config.n_hosts, config.n_ports)
    hosts = list(net.hosts)
    state = NetworkState.initial(net.hosts)

    links: set[tuple[str, str]] = set()
    for a, b in pairwise(hosts):  # a spanning chain -> connected
        links.add(link_key(a, b))
    while len(links) < config.n_links:  # extra random edges up to the density target
        a, b = rng.sample(hosts, 2)
        links.add(link_key(a, b))
    state.links = links

    ports = list(net.ports)
    for _ in range(config.n_services):
        host = rng.choice(hosts)
        port = rng.choice(ports)
        hs = state.hosts[host]
        state.hosts[host] = hs.with_service(port, True)

    compromised = frozenset(rng.sample(hosts, config.n_initial_compromised))
    return state, compromised


# --- the backends (how the network state evolves under a defender action) -------------------------


class Backend(Protocol):
    """How the network state evolves under one defender :class:`NetAction` (SPEC-20 §3)."""

    name: str

    def reset(self) -> None: ...

    def step(self, state: NetworkState, action: NetAction) -> NetworkState: ...


@dataclass
class OracleBackend:
    """`E_oracle` -- evolve the state with the exact reference oracle (reality / the baseline)."""

    oracle: NetOracle
    name: str = "oracle"

    def reset(self) -> None:
        return None

    def step(self, state: NetworkState, action: NetAction) -> NetworkState:
        return self.oracle.step(state, action).state


@dataclass
class FreeBackend:
    """`E_free` -- evolve the state with the trained `M_θ`, never corrected (the ablation)."""

    model: NetModel
    name: str = "free"

    def reset(self) -> None:
        return None

    def step(self, state: NetworkState, action: NetAction) -> NetworkState:
        return apply(state, self.model.predict_delta(state, action))


@dataclass
class GroundedBackend:
    """`E_grounded` -- the trained `M_θ` corrected by the oracle at budget ρ (the product).

    Every ``round(1/ρ)`` steps the predicted state is snapped back to the oracle's truth (the
    propose-verify-correct loop, hard reset); between corrections the model free-runs. ``ρ=1`` is
    `E_oracle`; ``ρ=0`` is `E_free`; the interior is the faithful-but-cheap regime SPEC-20 measures.
    """

    model: NetModel
    oracle: NetOracle
    rho: float = 0.5
    name: str = "grounded"
    _step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {self.rho}")

    def reset(self) -> None:
        self._step = 0

    def step(self, state: NetworkState, action: NetAction) -> NetworkState:
        self._step += 1
        if self.rho <= 0.0:
            return apply(state, self.model.predict_delta(state, action))
        if self.rho >= 1.0 or self._step % max(1, round(1.0 / self.rho)) == 0:
            return self.oracle.step(state, action).state  # CORRECT (hard reset to truth)
        return apply(state, self.model.predict_delta(state, action))


# --- the environment ------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOutcome:
    """The result of one env step (the verifiers/Gym-shape tuple, kept explicit)."""

    observation: tuple[float, ...]
    reward: float
    done: bool
    info: dict[str, Any]


class ContainmentEnv:
    """The defensive containment env over a plug-swappable backend (SPEC-20 UA0).

    One code path, three backends: the *same* policy trains against `E_grounded` / `E_free` and is
    tested against `E_oracle`, which is what isolates the value of oracle-grounding for transfer
    (UA2/H74). The defender acts; the backend evolves the network state; the scripted adversary
    spreads on the resulting reachability; the reward is containment minus action cost.
    """

    def __init__(self, config: ContainmentConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._net = NetworkState.initial(config.net().hosts)
        self._compromised: frozenset[str] = frozenset()
        self._t = 0
        self._isolations_used = 0
        self._rng = random.Random(0)

    def reset(self, seed: int = 0) -> tuple[float, ...]:
        self._rng = random.Random(seed)
        self._net, self._compromised = seed_topology(self.config, self._rng)
        self._t = 0
        self._isolations_used = 0
        self.backend.reset()
        return self.observe()

    def legal_actions(self) -> list[DefenderAction]:
        """The legal moves, with `isolate` removed once the `cut_budget` is spent (UA6)."""
        actions = legal_actions(self._net, self.config)
        if self.config.cut_budget is not None and self._isolations_used >= self.config.cut_budget:
            actions = [a for a in actions if a.kind != "isolate"]
        return actions

    def observe(self) -> tuple[float, ...]:
        """Per-host features: up, #services (normalized), compromised, connected-to-compromised.

        Fully observable for UA0 (the partial-observation probe variant is SPEC-20 §2 future work).
        Ordered by host id so the vector is stable across episodes.
        """
        reachable_from_compromised: set[str] = set()
        for src in self._compromised:
            reachable_from_compromised |= connected_hosts(self._net, src)
        feats: list[float] = []
        max_ports = max(1, self.config.n_ports)
        for host in sorted(self._net.hosts):
            hs = self._net.hosts[host]
            feats.extend([
                1.0 if hs.up else 0.0,
                len(hs.services) / max_ports,
                1.0 if host in self._compromised else 0.0,
                1.0 if (host in reachable_from_compromised
                        and host not in self._compromised) else 0.0,
            ])
        return tuple(feats)

    def step(self, action: DefenderAction) -> StepOutcome:
        """Apply the defender action, evolve via the backend, spread the adversary, score."""
        if action.kind == "isolate":
            self._isolations_used += 1  # spends one cut-budget slot, whether or not it takes effect
        self._net = self.backend.step(self._net, action.to_net_action())
        self._compromised = adversary_spread(self._net, self._compromised, self._rng)
        self._t += 1

        cost = 0.0 if action.kind == "noop" else self.config.action_cost
        contained = containment_fraction(self._net, self._compromised)
        reward = contained - cost
        done = self._t >= self.config.episode_steps
        info = {
            "containment": contained,
            "compromised": sorted(self._compromised),
            "n_compromised": len(self._compromised),
            "step": self._t,
        }
        return StepOutcome(self.observe(), reward, done, info)

    @property
    def n_compromised(self) -> int:
        return len(self._compromised)

    @property
    def net(self) -> NetworkState:
        """The current network state (read-only access for the policy's action featurizer)."""
        return self._net

    @property
    def compromised(self) -> frozenset[str]:
        """The current compromise set (read-only access for the policy's action featurizer)."""
        return self._compromised
