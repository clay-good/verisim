"""Host trajectory generation, JSONL records, and versioned manifests (SPEC-6 §3, HC2).

Rolls the host oracle forward under a seeded workload driver from the boot state
(:meth:`HostState.initial`), recording ``(state, action, next_state, delta, observation)`` per step.
Everything is a deterministic function of ``(config, driver, seed)`` so a dataset regenerates
identically from its manifest -- the same reproducibility regime as v0 (SPEC-2 §12) and the network
world (SPEC-5 §3). Splits are by *trajectory* with disjoint index sets, so a trajectory never leaks
across splits (SPEC-2 §4). The recorded ``delta`` is the **bundle delta** (HC1); the embedded FS
delta rides inside its ``FsDelta`` verbatim.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from verisim.host.config import HostConfig
from verisim.host.delta import delta_to_list
from verisim.host.state import HostState, to_canonical_host
from verisim.hostoracle.base import HostOracle

from .drivers import HostDriver


@dataclass
class HostTrajectory:
    """One seeded rollout: the workload config hash, seed, driver, and the step records."""

    host_config_hash: str
    seed: int
    driver: str
    steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_config_hash": self.host_config_hash,
            "seed": self.seed,
            "driver": self.driver,
            "steps": self.steps,
        }

    def to_jsonl(self) -> str:
        """One rollout per file (SPEC-2 §4): the trajectory as a single JSON line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


def generate_host_trajectory(
    oracle: HostOracle, config: HostConfig, driver_name: str, seed: int, n_steps: int
) -> HostTrajectory:
    """Roll the host oracle forward ``n_steps`` from the boot state under a seeded driver."""
    driver = HostDriver(name=driver_name, config=config, rng=random.Random(seed))
    state = HostState.initial()
    steps: list[dict[str, Any]] = []
    for _ in range(n_steps):
        action = driver.sample(state)
        result = oracle.step(state, action)
        steps.append(
            {
                "state": to_canonical_host(state),
                "action": action.raw,
                "next_state": to_canonical_host(result.state),
                "delta": delta_to_list(result.delta),
                "observation": {"exit_code": result.exit_code, "stdout": result.stdout},
            }
        )
        state = result.state
    return HostTrajectory(
        host_config_hash=config.config_hash(), seed=seed, driver=driver_name, steps=steps
    )


def _split_indices(n: int, fracs: dict[str, float], split_seed: int) -> dict[str, list[int]]:
    """Deterministically partition ``range(n)`` into disjoint, named splits (SPEC-2 §4)."""
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
class HostDatasetManifest:
    """A regenerable, integrity-checkable host dataset description (SPEC-6 §3, SPEC-2 §4 §12)."""

    host_config: dict[str, Any]
    host_config_hash: str
    oracle_version: str
    driver: str
    base_seed: int
    n_trajectories: int
    n_steps: int
    splits: dict[str, list[int]]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> HostDatasetManifest:
        return HostDatasetManifest(**json.loads(text))


def build_host_manifest(
    config: HostConfig,
    oracle: HostOracle,
    driver: str,
    base_seed: int,
    n_trajectories: int,
    n_steps: int,
    split_fracs: dict[str, float] | None = None,
    split_seed: int = 0,
) -> HostDatasetManifest:
    fracs = split_fracs or {"train": 0.7, "val": 0.15, "test": 0.15}
    return HostDatasetManifest(
        host_config=config.to_dict(),
        host_config_hash=config.config_hash(),
        oracle_version=getattr(oracle, "version", "unknown"),
        driver=driver,
        base_seed=base_seed,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        splits=_split_indices(n_trajectories, fracs, split_seed),
    )


def generate_host_dataset(
    manifest: HostDatasetManifest, oracle: HostOracle
) -> dict[str, list[HostTrajectory]]:
    """Regenerate the full dataset from a manifest. Deterministic given the oracle.

    Trajectory ``i`` uses ``seed = base_seed + i``; splits select which indices land where.
    """
    cfg = manifest.host_config
    config = HostConfig(
        name=cfg["name"],
        paths=tuple(cfg["paths"]),
        content_tokens=tuple(cfg["content_tokens"]),
        uids=tuple(cfg["uids"]),
    )
    out: dict[str, list[HostTrajectory]] = {name: [] for name in manifest.splits}
    for split, indices in manifest.splits.items():
        for i in indices:
            out[split].append(
                generate_host_trajectory(
                    oracle, config, manifest.driver, manifest.base_seed + i, manifest.n_steps
                )
            )
    return out
