"""Data generation tests (SPEC-2 §16): regenerability, no split leakage, validity."""

from __future__ import annotations

from verisim.data import (
    DRIVERS,
    DatasetManifest,
    build_manifest,
    generate_dataset,
    generate_trajectory,
)
from verisim.delta import apply
from verisim.env import DEFAULT_CONFIG, State, from_canonical, parse_action
from verisim.oracle import ReferenceOracle


def _manifest() -> DatasetManifest:
    return build_manifest(
        config=DEFAULT_CONFIG,
        oracle=ReferenceOracle(),
        driver="weighted",
        base_seed=100,
        n_trajectories=20,
        n_steps=30,
    )


def test_dataset_regenerates_identically_from_manifest():
    manifest = _manifest()
    a = generate_dataset(manifest, ReferenceOracle())
    # Round-trip the manifest through JSON, then regenerate: must be identical.
    b = generate_dataset(DatasetManifest.from_json(manifest.to_json()), ReferenceOracle())
    for split in manifest.splits:
        assert [t.to_jsonl() for t in a[split]] == [t.to_jsonl() for t in b[split]]


def test_no_trajectory_leaks_across_splits():
    splits = _manifest().splits
    seen: set[int] = set()
    total = 0
    for indices in splits.values():
        assert not (set(indices) & seen), "trajectory index appears in two splits"
        seen.update(indices)
        total += len(indices)
    assert total == 20  # every trajectory assigned exactly once
    assert seen == set(range(20))


def test_generated_steps_satisfy_apply_invariant():
    """Every recorded delta reconstructs the recorded next_state."""
    traj = generate_trajectory(ReferenceOracle(), DEFAULT_CONFIG, "adversarial", seed=9, n_steps=50)
    oracle = ReferenceOracle()
    for record_step in traj.steps:
        state = from_canonical(record_step["state"])
        result = oracle.step(state, parse_action(record_step["action"]))
        assert apply(state, result.delta) == from_canonical(record_step["next_state"])


def test_all_drivers_run_end_to_end():
    for driver_name in DRIVERS:
        traj = generate_trajectory(
            ReferenceOracle(), DEFAULT_CONFIG, driver_name, seed=1, n_steps=40
        )
        assert len(traj.steps) == 40
        assert traj.driver == driver_name


def test_trajectory_starts_from_empty_state():
    traj = generate_trajectory(ReferenceOracle(), DEFAULT_CONFIG, "uniform", seed=1, n_steps=5)
    assert from_canonical(traj.steps[0]["state"]) == State.empty()
