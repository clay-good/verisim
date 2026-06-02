"""Batfish-style control-plane oracle — the second oracle for H12 (SPEC-5 §5.1, §10.2).

The data-plane oracle (:class:`~verisim.netoracle.reference.ReferenceNetworkOracle`) returns the
exact next ``NetworkState`` — *every* fact (hosts, links, services, firewall, flows, clock). The
**control-plane** oracle is a Batfish-style symbolic verifier that returns only the **reachability**
truth — ``R[(src, dst, port)]`` = can ``src`` reach service ``(dst, port)``? — a coarser,
decision-relevant projection of the state, computed directly from the configuration (topology +
host-up + services + firewall), ignoring the data plane (flows, clock, stdout).

It is a genuine *second oracle*: deterministic and free, but a different — and differently priced —
consultation (a full data-plane consult reveals ``full_bits`` facts; a control-plane consult reveals
``control_plane_bits`` reachability entries). **H12** (SPEC-5 §10.2) asks whether it is a
*non-redundant* signal on top of the data-plane oracle. Reachability is a deterministic function of
the state, so the control-plane oracle cannot, by construction, catch a reachability error a
*full-state* data-plane consult misses — but it is a **cheaper, decision-relevant** consultation,
and the model satisfies it more often than the full delta (EN6's change-safety hint). This module is
deterministic, no-GPU, property-tested core; EN10 is the experiment (SPEC-5 §12) that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.net.action import NetAction
from verisim.net.state import NetworkState, reachability_matrix
from verisim.netoracle.base import NetOracle


def control_plane_bits(state: NetworkState) -> int:
    """Description length of a control-plane (reachability) consult: one symbol per ``R`` entry.

    The analogue of :func:`verisim.netloop.observe.full_bits` (one symbol per state fact) for the
    cheaper, coarser control-plane consultation — the cost denominator H12 prices against.
    """
    return len(reachability_matrix(state))


def reachability_bits_to_correct(predicted: NetworkState, true: NetworkState) -> int:
    """Bits to correct ``predicted``'s *reachability* to ``true``'s: the count of differing entries.

    Keyed by ``(src, dst, port)`` over the union of services either state lists (a service the truth
    drops but the prediction keeps, or vice versa, is a reachability error). ``0`` iff the predicted
    state reproduces every reachability fact — the control-plane analogue of
    :func:`verisim.netmetrics.bits.bits_to_correct` (which prices the full data-plane delta).
    """
    true_r = reachability_matrix(true)
    pred_r = reachability_matrix(predicted)
    keys = set(true_r) | set(pred_r)
    return sum(1 for k in keys if true_r.get(k, False) != pred_r.get(k, False))


@dataclass(frozen=True)
class ControlPlaneResult:
    """One control-plane consult: the next state's reachability truth + the consult's bit cost."""

    reachability: dict[tuple[str, str, int], bool]
    bits: int


class ControlPlaneOracle:
    """The Batfish-style second oracle: reachability truth from the config (SPEC-5 §5.1, H12).

    Wraps a data-plane :class:`~verisim.netoracle.base.NetOracle` (the source of the true next
    state) and projects it to the reachability matrix — the control plane's ground truth. Pure and
    deterministic; it never edits the data-plane oracle, the metric, or the gate (DD-AR2).
    """

    def __init__(self, oracle: NetOracle) -> None:
        self._oracle = oracle

    def consult(self, state: NetworkState, action: NetAction) -> ControlPlaneResult:
        """The control-plane truth for the transition: next-state reachability + its bit cost."""
        nxt = self._oracle.step(state, action).state
        return ControlPlaneResult(
            reachability=reachability_matrix(nxt), bits=control_plane_bits(nxt)
        )


__all__ = [
    "ControlPlaneOracle",
    "ControlPlaneResult",
    "control_plane_bits",
    "reachability_bits_to_correct",
]
