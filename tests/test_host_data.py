"""Host data generation tests (SPEC-6 §3, HC2): regenerability, no leakage, validity, integrity.

Mirrors v0's ``test_data.py`` and the network ``test_net_data.py``. The integrity check replays the
recorded **bundle deltas** from the boot state and checks they reproduce each recorded next_state --
the HC1 ``apply == oracle`` invariant carried into the on-disk dataset, with no need for a host
``from_canonical`` (deferred until the HC4 trainer consumes the JSONL).
"""

from __future__ import annotations

from verisim.host import DEFAULT_HOST_CONFIG, HostState, delta_from_list, to_canonical_host
from verisim.host.delta import apply
from verisim.hostdata import (
    HOST_DRIVERS,
    HostDatasetManifest,
    build_host_manifest,
    generate_host_dataset,
    generate_host_trajectory,
)
from verisim.hostoracle import ReferenceHostOracle


def _manifest() -> HostDatasetManifest:
    return build_host_manifest(
        config=DEFAULT_HOST_CONFIG,
        oracle=ReferenceHostOracle(),
        driver="forky",
        base_seed=100,
        n_trajectories=20,
        n_steps=30,
    )


def test_dataset_regenerates_identically_from_manifest() -> None:
    manifest = _manifest()
    a = generate_host_dataset(manifest, ReferenceHostOracle())
    regen = HostDatasetManifest.from_json(manifest.to_json())
    b = generate_host_dataset(regen, ReferenceHostOracle())
    for split in manifest.splits:
        assert [t.to_jsonl() for t in a[split]] == [t.to_jsonl() for t in b[split]]


def test_no_trajectory_leaks_across_splits() -> None:
    splits = _manifest().splits
    seen: set[int] = set()
    total = 0
    for indices in splits.values():
        assert not (set(indices) & seen), "trajectory index appears in two splits"
        seen.update(indices)
        total += len(indices)
    assert total == 20
    assert seen == set(range(20))


def test_recorded_deltas_reconstruct_every_next_state() -> None:
    """Replaying the bundle deltas from boot reproduces each recorded next_state (HC1 in data)."""
    traj = generate_host_trajectory(
        ReferenceHostOracle(), DEFAULT_HOST_CONFIG, "adversarial", seed=9, n_steps=60
    )
    state = HostState.initial()
    for step in traj.steps:
        assert to_canonical_host(state) == step["state"]
        state = apply(state, delta_from_list(step["delta"]))
        assert to_canonical_host(state) == step["next_state"]


def test_all_drivers_run_end_to_end() -> None:
    for driver_name in HOST_DRIVERS:
        traj = generate_host_trajectory(
            ReferenceHostOracle(), DEFAULT_HOST_CONFIG, driver_name, seed=1, n_steps=40
        )
        assert len(traj.steps) == 40
        assert traj.driver == driver_name


def test_trajectory_starts_from_boot_state() -> None:
    traj = generate_host_trajectory(
        ReferenceHostOracle(), DEFAULT_HOST_CONFIG, "uniform", seed=1, n_steps=5
    )
    assert traj.steps[0]["state"] == to_canonical_host(HostState.initial())


def test_forky_driver_builds_a_process_tree() -> None:
    """The fork-heavy workload grows the process table well beyond the single boot process."""
    traj = generate_host_trajectory(
        ReferenceHostOracle(), DEFAULT_HOST_CONFIG, "forky", seed=3, n_steps=80
    )
    final = traj.steps[-1]["next_state"]
    assert len(final["procs"]) > 3  # boot is 1 process; forky builds a real tree


def test_workloads_produce_successful_file_writes() -> None:
    """Top-level paths mean open+write actually mutates the embedded fs (not all-EBADF churn)."""
    traj = generate_host_trajectory(
        ReferenceHostOracle(), DEFAULT_HOST_CONFIG, "forky", seed=7, n_steps=80
    )
    fs_entries = traj.steps[-1]["next_state"]["fs"]["fs"]  # canonical: list of [path, node] pairs
    assert any(node.get("content") for _path, node in fs_entries)
