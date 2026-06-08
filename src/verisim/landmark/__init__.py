"""The SPEC-12 landmark-planning layer - a planning altitude above the propose-verify-correct loop.

A sparse graph of *landmark* states whose edges are oracle-verified reachability, scattered over the
SPEC-5 network world. It invents no new world and no new oracle (SPEC-12 §0): it reuses the
``ReferenceNetworkOracle`` (data-plane, exact), the ``ControlPlaneOracle`` (reachability, cheap),
the graph arm's ``embed()`` latent, and the shipped ``imagine``/``verify`` loop. This package adds
the graph, the search, and the verification.

Build order (SPEC-12 §7): LP1 (does the latent encode planning geometry? - :mod:`.geometry`) decides
whether landmarks live in the latent or in reachability space; LP2 builds and verifies the graph;
LP3 is the headline (a verified graph converts the structured arm's zero free-running horizon into
long-range goal reach).
"""

from .build import LandmarkSample, LandmarkTransition, sample_landmarks
from .geometry import (
    bfs_geodesics,
    canon_key,
    enumerate_actions,
    pearson,
    ranks,
    spearman,
)
from .graph import LandmarkGraph, ReachSig, reach_signature
from .plan import Hop, RolloutTrace, execute_plan, shortest_landmark_path
from .verify import EdgeVerdict, verify_edge

__all__ = [
    "EdgeVerdict",
    "Hop",
    "LandmarkGraph",
    "LandmarkSample",
    "LandmarkTransition",
    "ReachSig",
    "RolloutTrace",
    "bfs_geodesics",
    "canon_key",
    "enumerate_actions",
    "execute_plan",
    "pearson",
    "ranks",
    "reach_signature",
    "sample_landmarks",
    "shortest_landmark_path",
    "spearman",
    "verify_edge",
]
