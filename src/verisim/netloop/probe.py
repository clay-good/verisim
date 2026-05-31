"""Probe policies ``π_o`` -- *what* to observe (SPEC-5 §8.2). The new axis.

Given that the consultation policy ``π_c`` (when, SPEC-5 §8.1; reused from v0's
:mod:`verisim.loop.policy`) has decided to spend a cheap probe, the probe policy decides
*which* host to look at. This is active sensing / optimal experiment design and has no v0 or
SPEC-3 analogue (H2/H9 schedule *when* to verify; H10 is the first hypothesis about *what* to
look at, EN2).

NW5 ships the dependency-free baselines and the protocol; the uncertainty- and
information-gain-targeted policies that beat them -- the heart of EN2/H10 -- need a model that
localizes its belief uncertainty per host and land at NW7:

  - ``RandomProbe`` -- a uniformly random host (the dumb sensing baseline).
  - ``RoundRobinProbe`` -- cycle through hosts deterministically (uniform coverage).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verisim.net.state import NetworkState


@runtime_checkable
class ProbePolicy(Protocol):
    def select(self, belief: NetworkState) -> str: ...


@dataclass
class RandomProbe:
    """Probe a uniformly random host (the dumb sensing baseline). Seeded for replay."""

    hosts: Sequence[str]
    rng: random.Random

    def select(self, belief: NetworkState) -> str:
        return self.rng.choice(list(self.hosts))


@dataclass
class RoundRobinProbe:
    """Cycle through ``hosts`` deterministically -- uniform coverage, no randomness."""

    hosts: Sequence[str]
    _i: int = field(default=0)

    def select(self, belief: NetworkState) -> str:
        host = self.hosts[self._i % len(self.hosts)]
        self._i += 1
        return host
