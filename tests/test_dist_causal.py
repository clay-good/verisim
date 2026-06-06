"""Causal consistency: cross-object delivery ordering (SPEC-7 §3.4, DS0 increment 5).

Pins the third ``CONSISTENCY_MODELS`` end, ``causal`` -- a delivery-order refinement of ``eventual``
that forbids effect-before-cause without over-synchronizing -- and the backward-compatibility the
additive ``Message.deps`` field must preserve (eventual/linearizable serialize to the exact prior
form). Dependency-free, GPU-free.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.delta import MsgSend, delta_from_list, delta_to_list
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distmetrics.divergence import divergence
from verisim.distoracle import ReferenceDistOracle

NODES = ("n0", "n1", "n2")
OBJECTS = ("x", "y")


def _config(model: str) -> DistConfig:
    return DistConfig(name="causal-test", nodes=NODES, objects=OBJECTS, consistency_model=model)


def _run(config: DistConfig, cmds: list[str]) -> DistributedState:
    oracle = ReferenceDistOracle(config)
    state = DistributedState.initial(config)
    for cmd in cmds:
        state = oracle.step(state, parse_dist_action(cmd)).state
    return state


# the scenario that routes the effect (y) to the observer while its cause (x) is still blocked
_ANOMALY = [
    "put n0 x a",
    "partition n0 n1 | n2",
    "advance 1",
    "put n1 y b",
    "partition n0 | n1 n2",
    "advance 1",
]


def test_eventual_admits_effect_before_cause():
    """Under eventual, n2 adopts y=b while x is still in flight — the anomaly causal forbids."""
    state = _run(_config("eventual"), _ANOMALY)
    assert state.replicas[("y", "n2")].value == "b"
    assert state.replicas[("x", "n2")].value == "nil"  # effect without its cause


def test_causal_forbids_effect_before_cause():
    """Under causal, the y@1 message carries deps={x@1}, unmet at n2, so it is held — no anomaly."""
    state = _run(_config("causal"), _ANOMALY)
    assert state.replicas[("y", "n2")].value == "nil"  # the effect is held back
    assert state.replicas[("x", "n2")].value == "nil"
    # the held message is still in flight (waiting for its cause), not lost
    assert any(m.object_id == "y" and m.dst == "n2" for m in state.inflight.values())


def test_causal_message_carries_cross_object_deps():
    """The y write at n1 (which has observed x@1) emits a message depending on x@1."""
    oracle = ReferenceDistOracle(_config("causal"))
    state = DistributedState.initial(_config("causal"))
    for cmd in ["put n0 x a", "partition n0 n1 | n2", "advance 1"]:
        state = oracle.step(state, parse_dist_action(cmd)).state
    result = oracle.step(state, parse_dist_action("put n1 y b"))
    sends = [e for e in result.delta if isinstance(e, MsgSend)]
    assert sends, "the y write must emit replication messages"
    assert all(("x", 1) in s.deps for s in sends)  # every y message depends on x@1


def test_causal_converges_after_heal():
    """Causal is still live: after heal+advance every replica converges, identically to eventual."""
    cmds = [*_ANOMALY, "heal", "advance 5"]
    ev = _run(_config("eventual"), cmds)
    ca = _run(_config("causal"), cmds)
    # the held message is delivered once its cause arrives — n2 ends with both x=a and y=b
    assert ca.replicas[("x", "n2")].value == "a"
    assert ca.replicas[("y", "n2")].value == "b"
    assert not ca.inflight  # everything eventually delivered
    # the durable cluster state matches eventual (only the transient last_result count differs)
    assert divergence(ev, ca) == 0.0


def test_causal_does_not_block_independent_writes():
    """A y written *before* x is observed has no dep, so causal delivers it freely (concurrency)."""
    independent = [
        "put n0 x a",
        "put n1 y b",            # n1 writes y before seeing x@1 → no dependency
        "partition n0 | n1 n2",
        "advance 1",
    ]
    state = _run(_config("causal"), independent)
    assert state.replicas[("y", "n2")].value == "b"  # delivered, not held (no causal link)


# --- backward compatibility: the additive deps field leaves eventual/linearizable untouched -------


def test_eventual_message_serializes_without_deps_key():
    """An eventual in-flight message has no ``deps`` in its canonical form (hashes unchanged)."""
    state = _run(_config("eventual"), ["put n0 x b"])
    canon = to_canonical(state)
    assert canon["inflight"], "the async put leaves replication messages in flight"
    for m in canon["inflight"]:
        assert "deps" not in m


def test_causal_message_serializes_with_deps_and_round_trips():
    """A causal message with deps serializes them and round-trips exactly through canonical form."""
    state = _run(_config("causal"), ["put n0 x a", "advance 1", "put n0 y b"])
    canon = to_canonical(state)
    assert from_canonical(canon) == state  # exact round-trip with deps present
    # at least one in-flight message carries the cross-object dep (y depends on x@1)
    assert any("deps" in m for m in canon["inflight"])


def test_msgsend_delta_round_trips_with_deps():
    """The MsgSend edit serializes/deserializes its deps exactly (delta↔list round-trip)."""
    edit = MsgSend(0, "n0", "n1", "y", 1, "b", 1, deps=(("x", 1),))
    assert delta_from_list(delta_to_list([edit])) == [edit]
    plain = MsgSend(0, "n0", "n1", "x", 1, "a", 1)
    assert delta_to_list([plain])[0].get("deps") is None  # empty deps omitted from the dict
