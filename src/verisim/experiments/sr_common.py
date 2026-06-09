"""Shared apparatus for the SPEC-13 speculative-rollout (SR) experiments.

Three things the SR experiments share (mirroring :mod:`verisim.experiments.scale_common`):

  - :class:`SRWorld` -- a thin, world-generic bundle ``(name, make_actions, oracle_step, diverge)``
    so the same speculative primitive (:mod:`verisim.loop.speculative`) drives the SPEC-5 network,
    SPEC-6 host, and SPEC-2.1 filesystem worlds. The three constructors below build one each. The
    *gradual-vs-discrete* split SPEC-13 is pre-registered around (H40/H44, K4) lives in the
    divergence metric each world carries: the filesystem state is small, so one missed edit is a
    large fraction of its fact set and exceeds ``ε`` in one step (the K4 cliff); the network/host
    states are large, so one missed edit is a small fraction and drift is gradual. The proposer's
    raw
    error process is held *identical* across worlds (see :class:`StallDrafter`), so what the SR
    figures isolate is the world's contribution, not the proposer's -- the honest core of H44.

  - :class:`StallDrafter` -- the controlled drift proposer. The trained ``M_θ`` arm needs a GPU and
    is deferred/``skipif``-guarded (SPEC-13 §9, the LP7 discipline); the committed CPU core uses a
    *transparent* stand-in, exactly as LP7's myopic walk stands in for the LLM. The drafter predicts
    the oracle's true next state with per-step probability ``alpha`` and otherwise **stalls**
    (predicts no change -- the shipped ``NetNullModel`` behavior for that step). Its raw accuracy is
    a
    stated knob; the experiments vary the *world* and the *schedule* against it.

  - small helpers (``mean``, action replay) the SR drivers reuse.

Everything is CPU-only, deterministic, and seeded -- no torch, no model checkpoint (the NW0-NW3
discipline). Stalls are seeded by ``(seed, step, variant)`` so a draft tree (SR3) has independent
variants (variance) or, with ``systematic=True``, identical ones (bias) -- the H42 fork.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Generic, TypeVar

from verisim.data.drivers import Driver
from verisim.data.generate import Trajectory  # noqa: F401  (kept for type clarity in callers)
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostdata.drivers import HostDriver
from verisim.hostmetrics.divergence import divergence as host_divergence
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.divergence import divergence as fs_divergence
from verisim.net.action import NetAction
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netdata.drivers import NetDriver
from verisim.netmetrics.divergence import divergence as net_divergence
from verisim.netoracle.reference import ReferenceNetworkOracle
from verisim.oracle.reference import ReferenceOracle

S = TypeVar("S")
A = TypeVar("A")


def mean(values: Sequence[float]) -> float:
    """Mean, or ``0.0`` for an empty sequence (the SR reducers' convention)."""
    return fmean(values) if values else 0.0


@dataclass(frozen=True)
class SRWorld(Generic[S, A]):
    """A world-generic bundle the speculative primitive drives (SPEC-13 §6)."""

    name: str
    make_actions: Callable[[int, int], tuple[S, list[A]]]  # (seed, n_steps) -> (s0, actions)
    oracle_step: Callable[[S, A], S]
    diverge: Callable[[S, S], float]


# -- the controlled drift proposer (the committed CPU stand-in, SPEC-13 §9) ------------------------


@dataclass
class StallDrafter(Generic[S, A]):
    """A proposer right with probability ``alpha``, otherwise stalling (predicts no change).

    The transparent stand-in for ``M_θ``'s free-running drift (SPEC-13 §9, the LP7 discipline). The
    raw per-step accuracy ``alpha`` is a stated knob, *identical across worlds*, so the SR figures
    isolate the world's divergence metric (H44) rather than the proposer. Stalls are seeded by
    ``(seed, step, variant)`` -- independent across draft variants by default (variance: a draft
    tree
    helps, H42 positive) or, with ``systematic=True``, keyed by ``(seed, step)`` alone so every
    variant stalls in the same places (bias: a tree cannot help, the H42 null).
    """

    oracle_step: Callable[[S, A], S]
    alpha: float
    seed: int = 0
    systematic: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    def _uniform(self, step: int, variant: int) -> float:
        key = (self.seed, step) if self.systematic else (self.seed, step, variant)
        digest = hashlib.sha256(repr(key).encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2.0**64

    def is_correct(self, step: int, variant: int) -> bool:
        """Whether the draft is faithful at this ``(step, variant)`` -- seeded, deterministic."""
        return self._uniform(step, variant) < self.alpha

    def __call__(self, state: S, action: A, step: int, variant: int) -> S:
        if self.is_correct(step, variant):
            return self.oracle_step(state, action)
        return state  # stall: predict no change


@dataclass
class VaryingDrafter(Generic[S, A]):
    """A drafter whose per-step accuracy *varies by region*, with a tunable confidence signal (SR4).

    SR4's EAGLE-2 link (H41) needs two things the constant-``α`` :class:`StallDrafter` lacks:
    genuine
    *local* heterogeneity in acceptance (so a smarter draft-length choice has something to exploit)
    and
    a *confidence signal* that may or may not predict it. The trajectory alternates ``period``-step
    **easy** regions (accuracy ``alpha_easy``) and **hard** regions (``alpha_hard``); the confidence
    signal blends the true region indicator with seeded noise by ``signal_corr`` -- ``1.0`` is a
    perfect acceptance predictor (calibrated ``k`` should win), ``0.0`` is pure noise (the EH2 null:
    calibrated ``k`` ties fixed ``k`` because the signal carries no acceptance information).
    """

    oracle_step: Callable[[S, A], S]
    alpha_easy: float = 0.95
    alpha_hard: float = 0.5
    period: int = 8
    seed: int = 0
    signal_corr: float = 1.0

    def _easy(self, step: int) -> bool:
        return (step // self.period) % 2 == 0

    def local_alpha(self, step: int) -> float:
        return self.alpha_easy if self._easy(step) else self.alpha_hard

    def _noise(self, step: int, salt: str) -> float:
        digest = hashlib.sha256(repr((self.seed, salt, step)).encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2.0**64

    def confidence(self, step: int) -> float:
        """A signal in ``[0, 1]`` correlated with the upcoming region's accuracy by ``signal_corr``.
        """
        true = 1.0 if self._easy(step) else 0.0
        return self.signal_corr * true + (1.0 - self.signal_corr) * self._noise(step, "conf")

    def is_correct(self, step: int, variant: int) -> bool:
        return self._noise(step, f"stall{variant}") < self.local_alpha(step)

    def __call__(self, state: S, action: A, step: int, variant: int) -> S:
        if self.is_correct(step, variant):
            return self.oracle_step(state, action)
        return state


# -- the three worlds -----------------------------------------------------------------------------


def net_world(
    n_hosts: int = 6, n_ports: int = 3, driver: str = "weighted"
) -> SRWorld[NetworkState, NetAction]:
    """The SPEC-5 network world -- large fact set, *gradual* drift (the H39 win is expected here).
    """
    config: NetConfig = scaled_net_config(n_hosts, n_ports)
    oracle = ReferenceNetworkOracle()

    def make_actions(seed: int, n_steps: int) -> tuple[NetworkState, list[NetAction]]:
        drv = NetDriver(name=driver, config=config, rng=random.Random(seed))
        s0 = NetworkState.initial(config.hosts)
        state = s0
        actions: list[NetAction] = []
        for _ in range(n_steps):
            action = drv.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return s0, actions

    return SRWorld(
        name="network",
        make_actions=make_actions,
        oracle_step=lambda s, a: oracle.step(s, a).state,
        diverge=net_divergence,
    )


def host_world(driver: str = "forky") -> SRWorld[HostState, HostAction]:
    """The SPEC-6 host world -- factored state, gradual drift (the second H39 win is expected here).
    """
    config: HostConfig = DEFAULT_HOST_CONFIG
    oracle = ReferenceHostOracle()

    def make_actions(seed: int, n_steps: int) -> tuple[HostState, list[HostAction]]:
        drv = HostDriver(name=driver, config=config, rng=random.Random(seed))
        s0 = HostState.initial()
        state = s0
        actions: list[HostAction] = []
        for _ in range(n_steps):
            action = drv.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return s0, actions

    return SRWorld(
        name="host",
        make_actions=make_actions,
        oracle_step=lambda s, a: oracle.step(s, a).state,
        diverge=host_divergence,
    )


def fs_world(driver: str = "structural") -> SRWorld[State, Action]:
    """The SPEC-2.1 filesystem world -- small fact set, *discrete* drift (the K4 cliff, H40 split).
    """
    config: EnvConfig = DEFAULT_CONFIG
    oracle = ReferenceOracle()

    def make_actions(seed: int, n_steps: int) -> tuple[State, list[Action]]:
        drv = Driver(name=driver, config=config, rng=random.Random(seed))
        s0 = State.empty()
        state = s0
        actions: list[Action] = []
        for _ in range(n_steps):
            action = drv.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return s0, actions

    return SRWorld(
        name="filesystem",
        make_actions=make_actions,
        oracle_step=lambda s, a: oracle.step(s, a).state,
        diverge=fs_divergence,
    )


# The three worlds in the SR sweep, gradual worlds first (the expected-win order, SPEC-13 §6).
def all_worlds() -> list[SRWorld[Any, Any]]:
    """Network + host (gradual) then filesystem (discrete) -- the H44 contrast in one list."""
    return [net_world(), host_world(), fs_world()]
