"""Network trajectory generation (SPEC-5 §3, NW2).

Rolls the oracle forward under a seeded driver from the initial (empty) network, recording
``(state, action, next_state, delta)`` per step. Everything is a deterministic function of
``(config, driver, seed)`` so a dataset regenerates identically from its manifest -- the same
reproducibility regime as v0 (SPEC-2 §12).
"""

from __future__ import annotations

import random
from typing import Any

from verisim.net.config import NetConfig
from verisim.net.serialize import to_canonical
from verisim.net.state import NetworkState
from verisim.netdelta.serialize import delta_to_list
from verisim.netoracle.base import NetOracle

from .drivers import NetDriver


def generate_net_trajectory(
    oracle: NetOracle, config: NetConfig, driver_name: str, seed: int, n_steps: int
) -> list[dict[str, Any]]:
    """One seeded rollout as a list of canonical step records."""
    driver = NetDriver(name=driver_name, config=config, rng=random.Random(seed))
    state = NetworkState.initial(config.hosts)
    steps: list[dict[str, Any]] = []
    for _ in range(n_steps):
        action = driver.sample(state)
        result = oracle.step(state, action)
        steps.append(
            {
                "state": to_canonical(state),
                "action": action.raw,
                "next_state": to_canonical(result.state),
                "delta": delta_to_list(result.delta),
                "exit_code": result.exit_code,
            }
        )
        state = result.state
    return steps
