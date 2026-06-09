"""Shared apparatus for the SPEC-17 causal / counterfactual (CX) experiments.

:class:`CXWorld` is a world bundle ``(name, make_actions, oracle_step, diverge, alt_action)`` over
which
the SCM machinery (:mod:`verisim.causal.scm`) runs. The first four come from the SPEC-13 bundles
(network/host/filesystem); ``alt_action`` samples an *intervention* action ``a'`` at a visited state
(the ``do(X=x)`` of rung 2/3). The **distributed** world is added here -- the off-policy world where
H5/H61 predicts the counterfactual effect is largest (ED6), with a fault-heavy intervention sampler
(the medium-changing near-misses).

Scope note (SPEC-17 §7, the LP7 rule): the committed CPU tranche is the *pure-oracle* identification
(CX0, H60) and the counterfactual *effect-size* law (CX1, H61) -- both need only the oracle. The
*learned* counterfactual-lift bets (CX2 three-world lift, CX3 matched-coverage, CX4 the CoDA
contrast)
require a parametric/contrastive ``M_θ`` to exploit the paired factual/counterfactual *structure*,
so
they are deferred to the trained arm and never counted on a non-parametric stand-in (which captures
only the coverage channel, not the structure channel). CPU-only, deterministic, seeded.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distmetrics.divergence import divergence as dist_divergence
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.experiments.sr_common import SRWorld, fs_world, host_world, net_world

S = TypeVar("S")
A = TypeVar("A")


@dataclass(frozen=True)
class CXWorld(Generic[S, A]):
    """A world bundle the SCM machinery drives, with an intervention-action sampler (SPEC-17 §3)."""

    name: str
    make_actions: Callable[[int, int], tuple[S, list[A]]]
    oracle_step: Callable[[S, A], S]
    diverge: Callable[[S, S], float]
    alt_action: Callable[[S, int], A]  # sample an intervention action a' at a state, seeded


def _net_cx() -> CXWorld[Any, Any]:
    base = net_world()
    from verisim.net.config import scaled_net_config
    from verisim.netdata.drivers import NetDriver

    config = scaled_net_config(6, 3)

    def alt(state: Any, seed: int) -> Any:
        return NetDriver(name="adversarial", config=config, rng=random.Random(seed)).sample(state)

    return CXWorld("network", base.make_actions, base.oracle_step, base.diverge, alt)


def _host_cx() -> CXWorld[Any, Any]:
    base = host_world()
    from verisim.host.config import DEFAULT_HOST_CONFIG
    from verisim.hostdata.drivers import HostDriver

    def alt(state: Any, seed: int) -> Any:
        return HostDriver(
            name="adversarial", config=DEFAULT_HOST_CONFIG, rng=random.Random(seed)
        ).sample(state)

    return CXWorld("host", base.make_actions, base.oracle_step, base.diverge, alt)


def _fs_cx() -> CXWorld[Any, Any]:
    base = fs_world()
    from verisim.data.drivers import Driver
    from verisim.env.config import DEFAULT_CONFIG

    def alt(state: Any, seed: int) -> Any:
        drv = Driver(name="structural", config=DEFAULT_CONFIG, rng=random.Random(seed))
        return drv.sample(state)

    return CXWorld("filesystem", base.make_actions, base.oracle_step, base.diverge, alt)


def dist_world(driver: str = "uniform") -> SRWorld[Any, Any]:
    """The SPEC-7 distributed world as an SRWorld bundle (light-fault on-policy by default)."""
    config = DistConfig()
    oracle = ReferenceDistOracle()

    def make_actions(seed: int, n_steps: int) -> tuple[DistributedState, list[Any]]:
        drv = DistDriver(name=driver, config=config, rng=random.Random(seed))
        s0 = DistributedState.initial(config)
        state = s0
        actions: list[Any] = []
        for _ in range(n_steps):
            action = drv.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return s0, actions

    return SRWorld(
        "distributed", make_actions, lambda s, a: oracle.step(s, a).state, dist_divergence
    )


def _dist_cx() -> CXWorld[Any, Any]:
    base = dist_world()
    config = DistConfig()

    def alt(state: Any, seed: int) -> Any:
        # Fault-heavy interventions: the medium-changing near-misses where the effect lives (ED6).
        return DistDriver(name="adversarial", config=config, rng=random.Random(seed)).sample(state)

    return CXWorld("distributed", base.make_actions, base.oracle_step, base.diverge, alt)


def all_cx_worlds() -> list[CXWorld[Any, Any]]:
    """The four worlds for the SCM gate (CX0) and the effect-size law (CX1)."""
    return [_net_cx(), _host_cx(), _fs_cx(), _dist_cx()]
