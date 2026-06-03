"""Host trajectory data: workload drivers, generation, versioned manifests/splits (SPEC-6, HC2)."""

from __future__ import annotations

from .drivers import HOST_DRIVERS, HostDriver
from .generate import (
    HostDatasetManifest,
    HostTrajectory,
    build_host_manifest,
    generate_host_dataset,
    generate_host_trajectory,
)
from .scheduler import HostScheduler, Schedule, Thread, interleaving_entropy, make_workload

__all__ = [
    "HOST_DRIVERS",
    "HostDatasetManifest",
    "HostDriver",
    "HostScheduler",
    "HostTrajectory",
    "Schedule",
    "Thread",
    "build_host_manifest",
    "generate_host_dataset",
    "generate_host_trajectory",
    "interleaving_entropy",
    "make_workload",
]
