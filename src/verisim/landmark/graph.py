"""The faithful landmark graph: nodes are waypoint states, edges verified reachability (SPEC-12 §3).

LP1 refuted H31 (the ``embed()`` latent does not encode planning geometry, Spearman 0.27 < 0.6), so
per SPEC-12 §4's pre-registered fallback the graph is built **directly in reachability space**: a
landmark's identity is its *reachability signature* -- the set of ``(src, dst, port)`` triples that
are reachable in that state (the control-plane projection, SPEC-5 §3.1). Two states with the same
signature are the same landmark; an edge ``i -> j`` means "one action carries landmark ``i`` into
landmark ``j``'s reachability class."

This is the structure the SPEC-12 §8 defensive reading wants: an oracle-verified reachability/attack
graph whose every edge the oracle confirms, so it has **zero false paths by construction** -- the
standing weakness of every static attack-graph tool (MulVAL et al., §2.3). The torch-free graph and
its signature live here; LP2's build + verification (:mod:`verisim.landmark.build`,
:mod:`verisim.landmark.verify`) and the experiment consume them.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.net.state import NetworkState, reachability_matrix

# A landmark's identity: the frozenset of reachable (src, dst, port) triples. Reachability is the
# planning-relevant projection (EN10) and the metric LP1 sent us to (§4 fallback).
ReachSig = frozenset[tuple[str, str, int]]


def reach_signature(state: NetworkState) -> ReachSig:
    """The reachability signature of ``state``: the triples that are reachable (R == True)."""
    return frozenset(k for k, reachable in reachability_matrix(state).items() if reachable)


@dataclass(frozen=True)
class LandmarkGraph:
    """A sparse graph of landmark states with oracle-verified reachability edges (SPEC-12 §3).

    ``nodes[i]`` is a representative state for landmark ``i``; ``signatures[i]`` its reachability
    signature (the node identity); ``edges`` the directed admitted edges as ``(i, j)`` index pairs.
    Built either from the oracle (the *faithful* graph -- every edge true) or from the model (the
    *hoped* graph -- edges the model proposes, some hallucinated); LP2 measures the gap and prunes
    the hoped graph to the faithful one with the control-plane oracle.
    """

    nodes: tuple[NetworkState, ...]
    signatures: tuple[ReachSig, ...]
    edges: frozenset[tuple[int, int]]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def has_edge(self, i: int, j: int) -> bool:
        return (i, j) in self.edges

    def neighbors(self, i: int) -> list[int]:
        """The landmark ids directly reachable from landmark ``i`` (one verified hop)."""
        return sorted(j for (a, j) in self.edges if a == i)


__all__ = ["LandmarkGraph", "ReachSig", "reach_signature"]
