"""DS3 increment 2 — the tiered oracle, SPEC-7's payload (§5, the H17 premise made concrete).

The structural test of the distributed world: bit-exact global truth is intractable, so the oracle
is a menu of tiers at increasing cost, and the policy spends the **cheapest tier that can refute** a
prediction (DD-D1). These tests pin that each class of error is caught at the right (cheapest) tier,
that the cumulative oracle-dollar cost is correct, and that a subtle error respecting every cheap
invariant is caught only by the expensive bit-exact tier — exactly the non-redundancy question
H17 will measure (DS6).
"""

from __future__ import annotations

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import DistAction
from verisim.dist.state import Message, ReplicaState
from verisim.distoracle import ReferenceDistOracle, TieredOracle
from verisim.distoracle.tiers import TIER_COSTS

CFG = DistConfig()
REF = ReferenceDistOracle(CFG)
TI = TieredOracle(CFG)


def _truth(cmds: list[str]) -> tuple[DistributedState, DistAction]:
    """Return the (pre-state, action) for the last command (its true next-state is REF.step)."""
    state = DistributedState.initial(CFG)
    for cmd in cmds[:-1]:
        state = REF.step(state, parse_dist_action(cmd)).state
    action = parse_dist_action(cmds[-1])
    return state, action


def test_correct_prediction_passes_every_tier():
    state, action = _truth(["put n0 x b"])
    truth = REF.step(state, action).state
    verdict = TI.cheapest_refutation(state, action, truth)
    assert not verdict.refuted
    # paid every tier: 1 + 2 + 4 + 16
    assert verdict.cost == sum(TIER_COSTS.values())


def test_metamorphic_catches_out_of_vocab_cheaply():
    state, action = _truth(["put n0 x b"])
    bad = REF.step(state, action).state.copy()
    bad.replicas[("x", "n0")] = ReplicaState("x", "n0", 1, "ZZZ")  # not a config value
    verdict = TI.cheapest_refutation(state, action, bad)
    assert verdict.refuted and verdict.tier == "metamorphic"
    assert verdict.cost == TIER_COSTS["metamorphic"]  # cheapest possible


def test_metamorphic_catches_backward_clock():
    state, action = _truth(["put n0 x b", "advance 3", "advance 2"])
    bad = REF.step(state, action).state.copy()
    bad.clock = 0  # time ran backward
    assert TI.cheapest_refutation(state, action, bad).tier == "metamorphic"


def test_cycle_catches_a_read_that_mutated_state():
    state, action = _truth(["put n0 x b", "advance 2", "get n0 x"])
    bad = REF.step(state, action).state.copy()
    bad.replicas[("x", "n0")] = ReplicaState("x", "n0", 2, "c")  # a get must not write
    verdict = TI.cheapest_refutation(state, action, bad)
    assert verdict.refuted and verdict.tier == "cycle"
    assert verdict.cost == TIER_COSTS["metamorphic"] + TIER_COSTS["cycle"]


def test_symbolic_catches_a_wrong_coordinator_value():
    state, action = _truth(["put n0 x b"])
    bad = REF.step(state, action).state.copy()
    bad.replicas[("x", "n0")] = ReplicaState("x", "n0", 1, "c")  # in-vocab, version ok, but wrong
    verdict = TI.cheapest_refutation(state, action, bad)
    assert verdict.refuted and verdict.tier == "symbolic"


def test_symbolic_catches_a_changed_peer_on_a_write():
    state, action = _truth(["put n0 x b"])
    bad = REF.step(state, action).state.copy()
    bad.replicas[("x", "n1")] = ReplicaState("x", "n1", 1, "b")  # peers change only via advance
    assert TI.cheapest_refutation(state, action, bad).tier == "symbolic"


def test_bit_exact_alone_catches_a_subtle_inflight_error():
    # correct replicas/partition/clock/down, but a wrong in-flight message set — symbolic (for a
    # write) checks replicas, not the message queue, so only bit-exact refutes this.
    state, action = _truth(["put n0 x b"])
    truth = REF.step(state, action).state
    bad = truth.copy()
    bad.inflight = dict(truth.inflight)
    a_msg = next(iter(bad.inflight))
    m = bad.inflight[a_msg]
    bad.inflight[a_msg] = Message(m.id, m.src, m.dst, m.object_id, m.version, "c", m.deliver_after)
    verdict = TI.cheapest_refutation(state, action, bad)
    assert verdict.refuted and verdict.tier == "bit_exact"
    assert verdict.cost == sum(TIER_COSTS.values())  # paid all tiers to catch it — the H17 cost


def test_cheap_refutations_are_cheaper_than_bit_exact():
    # the whole premise (H17): catching a gross error costs far less than always running bit-exact
    state, action = _truth(["put n0 x b"])
    bad = REF.step(state, action).state.copy()
    bad.replicas[("x", "n0")] = ReplicaState("x", "n0", 1, "ZZZ")
    cheap = TI.cheapest_refutation(state, action, bad).cost
    assert cheap < TIER_COSTS["bit_exact"]


def test_check_rejects_unknown_tier():
    state, action = _truth(["put n0 x b"])
    truth = REF.step(state, action).state
    with pytest.raises(ValueError):
        TI.check("psychic", state, action, truth)
