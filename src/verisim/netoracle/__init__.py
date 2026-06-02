"""The network oracle: protocol, step result, the Tier-A reference (data-plane) oracle (NW0),
and the Batfish-style control-plane oracle (the second oracle for H12, SPEC-5 §5.1)."""

from .base import NetOracle, NetStepResult
from .control_plane import (
    ControlPlaneOracle,
    ControlPlaneResult,
    control_plane_bits,
    reachability_bits_to_correct,
)
from .reference import ReferenceNetworkOracle

__all__ = [
    "ControlPlaneOracle",
    "ControlPlaneResult",
    "NetOracle",
    "NetStepResult",
    "ReferenceNetworkOracle",
    "control_plane_bits",
    "reachability_bits_to_correct",
]
