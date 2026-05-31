"""Network driver policies over the SPEC-5 v0 grammar (SPEC-5 §3.2, NW2).

A seeded stochastic policy that emits actions to roll the oracle forward and generate
trajectories -- not a learned agent (SPEC-5 studies the world model). All randomness lives
here (seeded), preserving oracle purity. Three presets mirror v0:

  - ``uniform``      -- equal weight over commands.
  - ``weighted``     -- build-heavy (links, services, connects) then mutate, producing
                        realistic long-range dependencies (a service opened early is
                        connected to late).
  - ``adversarial``  -- biased toward the drift-inducing ops (``link_down``, ``host_down``,
                        ``fw_deny``, ``advance``) that break reachability and drop flows,
                        stressing compounding error (SPEC-5 §3.4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from verisim.net.action import NetAction, parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState

_COMMANDS = (
    "host_up", "host_down", "link_up", "link_down", "svc_up", "svc_down",
    "fw_deny", "fw_allow", "connect", "close", "advance",
)

_WEIGHTS: dict[str, dict[str, float]] = {
    "uniform": dict.fromkeys(_COMMANDS, 1.0),
    "weighted": {
        "host_up": 0.5, "host_down": 0.5, "link_up": 3.0, "link_down": 1.0,
        "svc_up": 3.0, "svc_down": 1.0, "fw_deny": 1.0, "fw_allow": 0.5,
        "connect": 3.0, "close": 1.0, "advance": 1.5,
    },
    "adversarial": {
        "host_up": 0.5, "host_down": 2.0, "link_up": 1.0, "link_down": 3.0,
        "svc_up": 1.0, "svc_down": 2.0, "fw_deny": 3.0, "fw_allow": 0.5,
        "connect": 1.5, "close": 1.0, "advance": 3.0,
    },
}

NET_DRIVERS = tuple(_WEIGHTS)


@dataclass
class NetDriver:
    """A seeded stochastic policy. ``sample(state)`` returns the next action."""

    name: str
    config: NetConfig
    rng: random.Random

    def __post_init__(self) -> None:
        if self.name not in _WEIGHTS:
            raise ValueError(f"unknown driver {self.name!r}; choose from {NET_DRIVERS}")

    def _two_hosts(self) -> tuple[str, str]:
        a, b = self.rng.sample(self.config.hosts, 2)
        return a, b

    def sample(self, state: NetworkState) -> NetAction:
        weights = _WEIGHTS[self.name]
        cmd = self.rng.choices(_COMMANDS, weights=[weights[c] for c in _COMMANDS])[0]
        return parse_net_action(self._build(cmd, state))

    def _build(self, cmd: str, state: NetworkState) -> str:
        hosts = self.config.hosts
        ports = self.config.ports
        if cmd == "advance":
            return "advance"
        if cmd in {"host_up", "host_down"}:
            return f"{cmd} {self.rng.choice(hosts)}"
        if cmd in {"link_up", "link_down"}:
            a, b = self._two_hosts()
            return f"{cmd} {a} {b}"
        if cmd in {"svc_up", "svc_down"}:
            return f"{cmd} {self.rng.choice(hosts)} {self.rng.choice(ports)}"
        if cmd in {"fw_deny", "fw_allow"}:
            a, b = self._two_hosts()
            return f"{cmd} {a} {b}"
        if cmd == "close":
            # prefer closing an existing flow; otherwise a random (failing) close.
            if state.flows:
                src, dst, port = self.rng.choice(sorted(state.flows))
                return f"close {src} {dst} {port}"
            a, b = self._two_hosts()
            return f"close {a} {b} {self.rng.choice(ports)}"
        # connect
        a, b = self._two_hosts()
        return f"connect {a} {b} {self.rng.choice(ports)}"
