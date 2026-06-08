"""Property tests for the torch-free landmark graph, sampling, and verification (SPEC-12 §3, LP2).

The deterministic half of LP2: the reachability signature is the node identity; sampling collects
reachability-distinct landmarks and true (reachability-changing) candidate edges; verification
confirms or rejects an edge against the oracle and prices both consults. These pin the invariants
(an oracle edge is real; a wrong claimed signature is rejected; control-plane <= data-plane cost)
before LP2's learned-model gap measurement.
"""

from __future__ import annotations

from verisim.landmark.build import sample_landmarks
from verisim.landmark.graph import LandmarkGraph, ReachSig, reach_signature
from verisim.landmark.verify import verify_edge
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState
from verisim.netoracle import ReferenceNetworkOracle


def test_reach_signature_is_state_identity() -> None:
    cfg = scaled_net_config(4, 2)
    oracle = ReferenceNetworkOracle()
    sample = sample_landmarks(oracle, cfg, driver="weighted", seeds=(0,), n_steps=20)
    # Each landmark's representative state reproduces that landmark's signature.
    for node, sig in zip(sample.nodes, sample.signatures, strict=True):
        assert reach_signature(node) == sig
    # Signatures are distinct (reachability-space dedup).
    assert len(set(sample.signatures)) == len(sample.signatures)


def test_sample_oracle_edges_are_real() -> None:
    cfg = scaled_net_config(5, 3)
    oracle = ReferenceNetworkOracle()
    sample = sample_landmarks(oracle, cfg, driver="weighted", seeds=(0, 1), n_steps=40)
    assert sample.transitions  # the weighted driver changes reachability within 40 steps
    # Every candidate edge is a true reachability-changing hop: applying its action to its source
    # lands on the claimed destination landmark's signature.
    for t in sample.transitions:
        true_next = oracle.step(t.src_state, t.action).state
        assert reach_signature(true_next) == sample.signatures[t.true_dst_id]
        assert sample.signatures[t.true_dst_id] != sample.signatures[t.src_id]


def test_verify_edge_confirms_truth_and_rejects_lie() -> None:
    cfg = scaled_net_config(5, 3)
    oracle = ReferenceNetworkOracle()
    sample = sample_landmarks(oracle, cfg, driver="weighted", seeds=(0,), n_steps=40)
    t = sample.transitions[0]
    true_sig = sample.signatures[t.true_dst_id]
    good = verify_edge(oracle, t.src_state, t.action, true_sig)
    assert good.is_real
    # A wrong claimed signature (the source's own reachability) is rejected -- the edge changed it.
    bad = verify_edge(oracle, t.src_state, t.action, sample.signatures[t.src_id])
    assert not bad.is_real
    # The control-plane verification an edge needs is never pricier than the data-plane consult.
    assert 0 < good.control_plane_bits <= good.data_plane_bits


def test_landmark_graph_neighbors() -> None:
    nodes = (NetworkState.initial(("h0", "h1")),) * 3
    sigs: tuple[ReachSig, ...] = (
        frozenset(),
        frozenset({("h0", "h1", 80)}),
        frozenset({("h1", "h0", 22)}),
    )
    g = LandmarkGraph(nodes=nodes, signatures=sigs, edges=frozenset({(0, 1), (0, 2), (1, 2)}))
    assert g.num_nodes == 3
    assert g.num_edges == 3
    assert g.neighbors(0) == [1, 2]
    assert g.has_edge(0, 1) and not g.has_edge(2, 0)
