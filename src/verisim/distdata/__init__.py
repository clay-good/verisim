"""Distributed data factory (SPEC-7 §3, DS2): seeded workload+fault drivers + trajectories."""

from verisim.distdata.drivers import DIST_DRIVERS, DistDriver
from verisim.distdata.generate import (
    DistTrajectory,
    generate_dataset,
    generate_dist_trajectory,
)

__all__ = [
    "DIST_DRIVERS",
    "DistDriver",
    "DistTrajectory",
    "generate_dataset",
    "generate_dist_trajectory",
]
