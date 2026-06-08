"""Oracle edge verification: the move no continuous-control landmark-planner can make (SPEC-12 §3).

A model proposes an edge ``src --action--> claimed reachability signature``. Verification consults
the oracle on the *true* next state and confirms or rejects the edge. Two prices, both exact + free:

  - **control-plane** cost ``control_plane_bits`` - the verification an edge actually needs (an edge
    is a *reachability* claim, and the control-plane oracle returns exactly the reachability truth);
  - **data-plane** cost ``full_bits`` - the full next-state consult.

H32's claim: the control-plane verification is *sufficient* for edges and *far cheaper* than the
data-plane (the H12 ratio lifted to edges). Because every admitted edge is oracle-confirmed, the
verified graph has **zero false edges by construction** - the SPEC-12 §8 defensive guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.landmark.graph import ReachSig, reach_signature
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netloop.observe import full_bits
from verisim.netoracle.base import NetOracle
from verisim.netoracle.control_plane import control_plane_bits


@dataclass(frozen=True)
class EdgeVerdict:
    """One edge verification: is the claimed reachability real, and what did each oracle cost?"""

    is_real: bool
    control_plane_bits: int
    data_plane_bits: int


def verify_edge(
    oracle: NetOracle, src: NetworkState, action: NetAction, claimed_sig: ReachSig
) -> EdgeVerdict:
    """Confirm a model-proposed edge against the oracle; return the verdict + both consult costs."""
    true_next = oracle.step(src, action).state
    return EdgeVerdict(
        is_real=reach_signature(true_next) == claimed_sig,
        control_plane_bits=control_plane_bits(true_next),
        data_plane_bits=full_bits(true_next),
    )


__all__ = ["EdgeVerdict", "verify_edge"]
