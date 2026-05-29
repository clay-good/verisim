"""Trajectory generation, JSONL records, and versioned manifests (SPEC-2 §4, §12).

A trajectory is produced by rolling the oracle forward under a seeded driver from
the empty state. Everything is a deterministic function of (env config + driver +
seed), so a dataset regenerates identically from its manifest (SPEC-2 §12). Splits
are by *trajectory* with disjoint index sets -- a trajectory never leaks across
splits (§4).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from verisim.delta.serialize import delta_to_list
from verisim.env.config import EnvConfig
from verisim.env.serialize import to_canonical
from verisim.env.state import State
from verisim.oracle.base import Oracle

from .drivers import Driver


@dataclass
class Trajectory:
    env_config_hash: str
    seed: int
    driver: str
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_config_hash": self.env_config_hash,
            "seed": self.seed,
            "driver": self.driver,
            "steps": self.steps,
        }

    def to_jsonl(self) -> str:
        """One rollout per file (SPEC-2 §4): the trajectory as a single JSON line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


def generate_trajectory(
    oracle: Oracle,
    config: EnvConfig,
    driver_name: str,
    seed: int,
    n_steps: int,
) -> Trajectory:
    """Roll the oracle forward ``n_steps`` from the empty state under a seeded driver."""
    driver = Driver(name=driver_name, config=config, rng=random.Random(seed))
    state = State.empty()
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
                "observation": {"exit_code": result.exit_code, "stdout": result.stdout},
            }
        )
        state = result.state
    return Trajectory(
        env_config_hash=config.config_hash(), seed=seed, driver=driver_name, steps=steps
    )


def _split_indices(n: int, fracs: dict[str, float], split_seed: int) -> dict[str, list[int]]:
    """Deterministically partition ``range(n)`` into disjoint, named splits."""
    order = list(range(n))
    random.Random(split_seed).shuffle(order)
    names = list(fracs)
    splits: dict[str, list[int]] = {name: [] for name in names}
    cursor = 0
    for name in names[:-1]:
        count = round(fracs[name] * n)
        for idx in order[cursor : cursor + count]:
            splits[name].append(idx)
        cursor += count
    for idx in order[cursor:]:  # remainder to the last split (no rounding loss)
        splits[names[-1]].append(idx)
    for ids in splits.values():
        ids.sort()
    return splits


@dataclass
class DatasetManifest:
    """A regenerable, integrity-checkable dataset description (SPEC-2 §4, §12)."""

    env_config: dict[str, Any]
    env_config_hash: str
    oracle_version: str
    driver: str
    base_seed: int
    n_trajectories: int
    n_steps: int
    splits: dict[str, list[int]]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> DatasetManifest:
        d = json.loads(text)
        return DatasetManifest(**d)


def build_manifest(
    config: EnvConfig,
    oracle: Oracle,
    driver: str,
    base_seed: int,
    n_trajectories: int,
    n_steps: int,
    split_fracs: dict[str, float] | None = None,
    split_seed: int = 0,
) -> DatasetManifest:
    fracs = split_fracs or {"train": 0.7, "val": 0.15, "test": 0.15}
    return DatasetManifest(
        env_config=config.to_dict(),
        env_config_hash=config.config_hash(),
        oracle_version=getattr(oracle, "version", "unknown"),
        driver=driver,
        base_seed=base_seed,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        splits=_split_indices(n_trajectories, fracs, split_seed),
    )


def generate_dataset(
    manifest: DatasetManifest, oracle: Oracle
) -> dict[str, list[Trajectory]]:
    """Regenerate the full dataset from a manifest. Deterministic given the oracle.

    Trajectory ``i`` uses ``seed = base_seed + i``; splits select which indices
    land in which partition.
    """
    config = EnvConfig(
        name=manifest.env_config["name"],
        content_tokens=tuple(manifest.env_config["content_tokens"]),
        modes=tuple(manifest.env_config["modes"]),
        env_keys=tuple(manifest.env_config["env_keys"]),
        name_pool=tuple(manifest.env_config["name_pool"]),
    )
    out: dict[str, list[Trajectory]] = {name: [] for name in manifest.splits}
    for split, indices in manifest.splits.items():
        for i in indices:
            out[split].append(
                generate_trajectory(
                    oracle, config, manifest.driver, manifest.base_seed + i, manifest.n_steps
                )
            )
    return out
