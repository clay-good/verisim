"""Host workload drivers over the HC0 syscall grammar (SPEC-6 §3.2, HC2).

A seeded stochastic policy that emits syscalls to roll the host oracle forward and generate
trajectories -- not a learned agent (SPEC-6 studies the world model). All randomness lives here
(seeded), preserving oracle purity, exactly as v0's :class:`~verisim.data.drivers.Driver` and the
network :class:`~verisim.netdata.drivers.NetDriver` do.

The driver **reads the bundle state** to emit mostly-valid syscalls -- it acts on a live
(``RUNNING``) process and, for ``write``/``close``, an actually-open fd -- so trajectories build
long-range dependencies (a process forked early opens a file; a descendant writes it later). Three
presets span the difficulty axis (driver weighting carries difficulty until the scheduler's
interleaving dial lands, SPEC-6 §3.3):

  - ``uniform``      -- equal weight over the six syscalls.
  - ``forky``        -- fork-tree + I/O heavy: deep process trees and lots of file activity, the
                        realistic build-then-use workload (long-range fd dependencies).
  - ``adversarial``  -- privilege- and churn-heavy: biased toward ``setuid`` (escalation attempts,
                        mostly EPERM), ``exit`` (processes die -> compounding state collapse), and
                        ``close`` (fd churn), stressing compounding error (SPEC-6 §3.4 analogue).

``exit`` never targets pid 1 (the init/shell): the workload always retains a live process to act on,
so trajectories stay productive rather than collapsing to an all-dead host.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from verisim.host.action import HostAction, parse_host_action
from verisim.host.config import HostConfig
from verisim.host.state import RUNNING, HostState

_COMMANDS = ("fork", "exit", "setuid", "open", "write", "close")

_WEIGHTS: dict[str, dict[str, float]] = {
    "uniform": dict.fromkeys(_COMMANDS, 1.0),
    "forky": {
        "fork": 3.0, "exit": 0.5, "setuid": 0.5,
        "open": 3.0, "write": 3.0, "close": 1.0,
    },
    "adversarial": {
        "fork": 1.0, "exit": 3.0, "setuid": 3.0,
        "open": 1.0, "write": 1.0, "close": 3.0,
    },
}

HOST_DRIVERS = tuple(_WEIGHTS)


@dataclass
class HostDriver:
    """A seeded stochastic policy over the host grammar. ``sample(state)`` -> the next syscall."""

    name: str
    config: HostConfig
    rng: random.Random

    def __post_init__(self) -> None:
        if self.name not in _WEIGHTS:
            raise ValueError(f"unknown driver {self.name!r}; choose from {HOST_DRIVERS}")

    def sample(self, state: HostState) -> HostAction:
        weights = _WEIGHTS[self.name]
        cmd = self.rng.choices(_COMMANDS, weights=[weights[c] for c in _COMMANDS])[0]
        return parse_host_action(self._build(cmd, state))

    # -- state probes (the driver only ever acts on live processes / open fds) ----

    def _running_pids(self, state: HostState) -> list[int]:
        return sorted(pid for pid, p in state.procs.items() if p.state == RUNNING)

    def _open_fds(self, state: HostState, running: set[int]) -> list[tuple[int, int]]:
        return sorted((pid, fd) for (pid, fd) in state.fds if pid in running)

    def _build(self, cmd: str, state: HostState) -> str:
        running = self._running_pids(state)
        pid = self.rng.choice(running)  # always >= 1 (pid 1 is never exited)
        if cmd == "fork":
            return f"fork {pid}"
        if cmd == "exit":
            survivors = [p for p in running if p != 1]
            if not survivors:  # never exit the init/shell -- keep a live process to act on
                return f"fork {pid}"
            return f"exit {self.rng.choice(survivors)} {self.rng.choice((0, 1))}"
        if cmd == "setuid":
            return f"setuid {pid} {self.rng.choice(self.config.uids)}"
        if cmd == "open":
            return f"open {pid} {self.rng.choice(self.config.paths)}"
        if cmd == "write":
            fds = self._open_fds(state, set(running))
            if not fds:  # nothing open yet -> open instead of a guaranteed EBADF
                return f"open {pid} {self.rng.choice(self.config.paths)}"
            fpid, fd = self.rng.choice(fds)
            return f"write {fpid} {fd} {self.rng.choice(self.config.content_tokens)}"
        # close
        fds = self._open_fds(state, set(running))
        if not fds:
            return f"open {pid} {self.rng.choice(self.config.paths)}"
        fpid, fd = self.rng.choice(fds)
        return f"close {fpid} {fd}"
