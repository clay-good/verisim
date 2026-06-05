"""Distributed trajectory generation, JSONL records, and versioned manifests (SPEC-7 §3; DS2).

Rolls the distributed oracle forward under a seeded workload+fault driver from the boot cluster
(:meth:`DistributedState.initial`), recording ``(state, action, next_state, delta, result)`` per
step. Everything is a deterministic function of ``(config, driver, seed)`` so a dataset regenerates
identically from its manifest -- the same reproducibility regime as v0 (SPEC-2 §12), the network
(SPEC-5 §3), and the host (SPEC-6 §3). Splits are by *trajectory* with disjoint index sets, so a
trajectory never leaks across splits (SPEC-2 §4). The recorded ``delta`` is the structured
``DistDelta`` (DS1); ``apply(state, delta) == next_state`` by construction.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from verisim.dist.config import DistConfig
from verisim.dist.delta import delta_to_list
from verisim.dist.serialize import to_canonical
from verisim.dist.state import DistributedState
from verisim.distoracle.base import DistOracle

from .drivers import DistDriver


@dataclass
class DistTrajectory:
    """One seeded rollout: the cluster config hash, seed, driver, and the per-step records."""

    dist_config_hash: str
    seed: int
    driver: str
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dist_config_hash": self.dist_config_hash,
            "seed": self.seed,
            "driver": self.driver,
            "steps": self.steps,
        }

    def to_jsonl(self) -> str:
        """One rollout per line (SPEC-2 §4): the trajectory as a single JSON line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


def generate_dist_trajectory(
    oracle: DistOracle,
    config: DistConfig,
    driver_name: str,
    seed: int,
    n_steps: int,
    *,
    fault_prob: float | None = None,
    partition_bias: float | None = None,
) -> DistTrajectory:
    """Roll the distributed oracle ``n_steps`` from the boot cluster under a seeded driver."""
    driver = DistDriver(
        name=driver_name, config=config, rng=random.Random(seed),
        fault_prob=fault_prob, partition_bias=partition_bias,
    )
    state = DistributedState.initial(config)
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
                "result": {"status": result.status, "value": result.value},
            }
        )
        state = result.state
    return DistTrajectory(
        dist_config_hash=config.config_hash(), seed=seed, driver=driver_name, steps=steps
    )


def _split_indices(n: int, fracs: dict[str, float], split_seed: int) -> dict[str, list[int]]:
    """Deterministically partition ``range(n)`` into disjoint, named splits (SPEC-2 §4)."""
    order = list(range(n))
    random.Random(split_seed).shuffle(order)
    out: dict[str, list[int]] = {}
    cursor = 0
    names = list(fracs)
    for i, name in enumerate(names):
        last = i == len(names) - 1
        take = n - cursor if last else round(fracs[name] * n)
        out[name] = sorted(order[cursor : cursor + take])
        cursor += take
    return out


def generate_dataset(
    oracle: DistOracle,
    config: DistConfig,
    *,
    driver: str = "uniform",
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    n_steps: int = 24,
    fracs: dict[str, float] | None = None,
    split_seed: int = 0,
) -> dict[str, Any]:
    """A regenerable manifest: one trajectory per seed + disjoint trajectory-level splits."""
    fracs = fracs or {"train": 0.75, "val": 0.25}
    trajectories = [
        generate_dist_trajectory(oracle, config, driver, seed, n_steps) for seed in seeds
    ]
    splits = _split_indices(len(seeds), fracs, split_seed)
    return {
        "dist_config": config.to_dict(),
        "dist_config_hash": config.config_hash(),
        "driver": driver,
        "seeds": list(seeds),
        "n_steps": n_steps,
        "splits": {name: [seeds[i] for i in idxs] for name, idxs in splits.items()},
        "trajectories": [t.to_dict() for t in trajectories],
    }
