"""Distributed workload + fault drivers over the DS0 grammar (SPEC-7 §3.2, §3.4; DS2).

A seeded stochastic policy that emits actions to roll the distributed oracle forward and generate
trajectories -- not a learned agent (the world model is studied separately). All randomness lives
here (seeded), preserving oracle purity, exactly as v0's :class:`~verisim.data.drivers.Driver`, the
network :class:`~verisim.netdata.drivers.NetDriver`, and the host
:class:`~verisim.hostdata.drivers.HostDriver` do.

The driver **reads the cluster state** to emit mostly-valid actions -- it writes/reads through an
*up* coordinator that holds the key's replica, heals partitions it created, and restarts crashed
nodes -- so trajectories stay productive and build the long-range dependencies that matter
(a value written under partition is read stale until heal+advance). It interleaves three categories
whose mix is the curriculum (SPEC-7 §3.4):

  - **client** ops (``put``/``get``/``cas``) -- the workload;
  - **advance** -- the time engine that actually delivers replication (else nothing converges);
  - **fault** ops (``partition``/``heal``/``crash``/``restart``) -- the medium, the ``BUGGIFY`` of
    deterministic-simulation testing.

The two fault axes H20/H21 will sweep are **explicit dials**: ``fault_prob`` (fault intensity -- the
share of steps that inject a fault) and ``partition_bias`` (partition entropy -- among injected
faults, how often the fault is a network split vs a node crash). Three presets set defaults:

  - ``uniform``     -- balanced client ops, frequent ``advance``, light faults.
  - ``contention``  -- a hot key (writes hit one object) + more ``cas`` (conflicts), light faults.
  - ``adversarial`` -- fault-heavy: frequent ``partition``/``crash`` stressing stale reads,
                       convergence, and compounding error (SPEC-7 §3.4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from verisim.dist.action import DistAction, parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState

# Per-preset (fault_prob, partition_bias, advance_prob, cas_share, hot_key).
#   fault_prob     -- P(a step injects a fault) -- the fault-intensity dial.
#   partition_bias -- among faults, P(partition vs crash) -- the partition-entropy dial.
#   advance_prob   -- P(a non-fault step is ``advance``) vs a client op.
#   cas_share      -- among client ops, the share that are ``cas`` (the rest split put/get).
#   hot_key        -- if True, writes concentrate on the first object (contention).
_PRESETS: dict[str, dict[str, float | bool]] = {
    "uniform": {"fault_prob": 0.10, "partition_bias": 0.5, "advance_prob": 0.35,
                "cas_share": 0.15, "hot_key": False},
    "contention": {"fault_prob": 0.10, "partition_bias": 0.4, "advance_prob": 0.30,
                   "cas_share": 0.45, "hot_key": True},
    "adversarial": {"fault_prob": 0.40, "partition_bias": 0.7, "advance_prob": 0.35,
                    "cas_share": 0.25, "hot_key": False},
}

DIST_DRIVERS = tuple(_PRESETS)


@dataclass
class DistDriver:
    """A seeded stochastic policy over the distributed grammar. ``sample(state)`` -> next action.

    ``fault_prob`` / ``partition_bias`` override the preset's fault-intensity / partition-entropy
    dials (the H20/H21 axes); ``None`` uses the preset default.
    """

    name: str
    config: DistConfig
    rng: random.Random
    fault_prob: float | None = None
    partition_bias: float | None = None

    def __post_init__(self) -> None:
        if self.name not in _PRESETS:
            raise ValueError(f"unknown driver {self.name!r}; choose from {DIST_DRIVERS}")
        preset = _PRESETS[self.name]
        fp = self.fault_prob
        self._fault_prob = fp if fp is not None else float(preset["fault_prob"])
        self._partition_bias = (
            self.partition_bias if self.partition_bias is not None
            else float(preset["partition_bias"])
        )
        self._advance_prob = float(preset["advance_prob"])
        self._cas_share = float(preset["cas_share"])
        self._hot_key = bool(preset["hot_key"])

    def sample(self, state: DistributedState) -> DistAction:
        return parse_dist_action(self._build(state))

    # -- state probes (the driver acts on up nodes that hold the key's replica) --------------------

    def _up_nodes(self, state: DistributedState) -> list[str]:
        return [n for n in self.config.nodes if state.is_up(n)]

    def _coordinator_for(self, state: DistributedState, key: str) -> str | None:
        candidates = [n for n in self.config.replicas_of(key) if state.is_up(n)]
        return self.rng.choice(candidates) if candidates else None

    def _is_partitioned(self, state: DistributedState) -> bool:
        return len(state.partitions) > 1

    def _build(self, state: DistributedState) -> str:
        if self.rng.random() < self._fault_prob:
            fault = self._fault(state)
            if fault is not None:
                return fault
        # non-fault: advance (deliver replication) or a client op
        if self.rng.random() < self._advance_prob:
            return f"advance {self.rng.choice((1, 1, 2))}"
        client = self._client(state)
        return client if client is not None else "advance 1"

    def _fault(self, state: DistributedState) -> str | None:
        """Pick a state-aware fault that keeps the trajectory productive (heal/restart recover)."""
        # recover first: don't let partitions/crashes accumulate forever
        if self._is_partitioned(state) and self.rng.random() < 0.6:
            return "heal"
        if state.down and self.rng.random() < 0.6:
            return f"restart {self.rng.choice(sorted(state.down))}"
        if self.rng.random() < self._partition_bias:
            return self._partition(state)
        return self._crash(state)

    def _partition(self, state: DistributedState) -> str | None:
        nodes = list(self.config.nodes)
        if len(nodes) < 2:
            return None
        # a random non-trivial 2-way split (each side non-empty)
        self.rng.shuffle(nodes)
        cut = self.rng.randint(1, len(nodes) - 1)
        left, right = sorted(nodes[:cut]), sorted(nodes[cut:])
        return f"partition {' '.join(left)} | {' '.join(right)}"

    def _crash(self, state: DistributedState) -> str | None:
        up = self._up_nodes(state)
        if len(up) <= 1:  # keep at least one node up so client ops stay possible
            return None
        return f"crash {self.rng.choice(up)}"

    def _client(self, state: DistributedState) -> str | None:
        key = self.config.objects[0] if self._hot_key else self.rng.choice(self.config.objects)
        node = self._coordinator_for(state, key)
        if node is None:
            return None
        roll = self.rng.random()
        if roll < self._cas_share:
            cur = state.replicas[(key, node)].value
            old = cur if self.rng.random() < 0.6 else self.rng.choice(self.config.values)
            new = self.rng.choice(self.config.values)
            return f"cas {node} {key} {old} {new}"
        if roll < self._cas_share + (1.0 - self._cas_share) / 2:
            return f"put {node} {key} {self.rng.choice(self.config.values)}"
        return f"get {node} {key}"
