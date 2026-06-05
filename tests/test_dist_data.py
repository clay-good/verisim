"""DS2 — the distributed data factory: workload + fault drivers + trajectories (SPEC-7 §3).

Tests the dependency-free, GPU-free data layer (the DS2 gate):

  - every driver emits **valid actions** the oracle accepts, and ``apply == oracle`` holds across a
    full rollout (the data is consistent with the deterministic core);
  - trajectories are **deterministic** in ``(config, driver, seed)`` (regenerable datasets);
  - the **fault-intensity dial** (`fault_prob`) is monotone, and the **partition-entropy dial**
    (`partition_bias`) shifts the fault mix toward partitions — the explicit H20/H21 axes;
  - presets are distinct, and the dataset manifest has disjoint trajectory-level splits.
"""

from __future__ import annotations

import random

from verisim.dist import DistConfig, DistributedState, apply
from verisim.distdata import (
    DIST_DRIVERS,
    DistDriver,
    generate_dataset,
    generate_dist_trajectory,
)
from verisim.distoracle import ReferenceDistOracle

CFG = DistConfig()
ORACLE = ReferenceDistOracle(CFG)
_FAULT = {"partition", "heal", "crash", "restart"}


def _fault_ops(driver: DistDriver, n: int = 300, seed_state: int = 0) -> dict[str, int]:
    state = DistributedState.initial(CFG)
    counts: dict[str, int] = {}
    for _ in range(n):
        action = driver.sample(state)
        result = ORACLE.step(state, action)
        assert apply(state, result.delta) == result.state  # data consistent with the oracle
        counts[action.name] = counts.get(action.name, 0) + 1
        state = result.state
    return counts


def test_every_driver_emits_valid_actions_apply_equals_oracle():
    for name in DIST_DRIVERS:
        counts = _fault_ops(DistDriver(name, CFG, random.Random(7)))
        assert counts["advance"] > 0  # the driver always advances time (so replication happens)
        assert sum(counts.get(c, 0) for c in ("put", "get", "cas")) > 0  # and does real work


def test_trajectories_are_deterministic():
    a = generate_dist_trajectory(ORACLE, CFG, "adversarial", 3, 30)
    b = generate_dist_trajectory(ORACLE, CFG, "adversarial", 3, 30)
    assert a.to_jsonl() == b.to_jsonl()


def test_fault_intensity_dial_is_monotone():
    lo = sum(_fault_ops(DistDriver("uniform", CFG, random.Random(1), fault_prob=0.05)).get(f, 0)
             for f in _FAULT)
    hi = sum(_fault_ops(DistDriver("uniform", CFG, random.Random(1), fault_prob=0.6)).get(f, 0)
             for f in _FAULT)
    assert hi > lo  # more fault_prob -> more fault ops


def test_partition_entropy_dial_shifts_the_mix():
    # high partition_bias -> proportionally more partitions among the faults it injects
    rng_lo = random.Random(2)
    rng_hi = random.Random(2)
    low = _fault_ops(DistDriver("uniform", CFG, rng_lo, fault_prob=0.5, partition_bias=0.1))
    high = _fault_ops(DistDriver("uniform", CFG, rng_hi, fault_prob=0.5, partition_bias=0.9))
    assert high.get("partition", 0) > low.get("partition", 0)


def test_presets_are_distinct():
    uni = _fault_ops(DistDriver("uniform", CFG, random.Random(5)))
    con = _fault_ops(DistDriver("contention", CFG, random.Random(5)))
    adv = _fault_ops(DistDriver("adversarial", CFG, random.Random(5)))
    assert con.get("cas", 0) > uni.get("cas", 0)  # contention is cas-heavy
    assert adv.get("partition", 0) > uni.get("partition", 0)  # adversarial is fault-heavy


def test_dataset_has_disjoint_trajectory_splits():
    ds = generate_dataset(ORACLE, CFG, driver="contention", seeds=(0, 1, 2, 3), n_steps=12)
    assert len(ds["trajectories"]) == 4
    train, val = set(ds["splits"]["train"]), set(ds["splits"]["val"])
    assert train.isdisjoint(val)  # a trajectory never leaks across splits
    assert train | val == {0, 1, 2, 3}
    assert ds["dist_config_hash"] == CFG.config_hash()
