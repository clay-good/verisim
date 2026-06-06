"""Distributed verified-contribution protocol — trustless by re-execution (SPEC-7 §16).

The §16 guarantee: a contributed distributed trace is accepted iff re-running the deterministic
oracle reproduces it bit-for-bit — *or*, where bit-exact is intractable (W7), iff the cheap
consistency tier admits it. These tests pin both halves plus the structural guarantees the host
:mod:`verisim.contrib` already pins (chaining, content-addressing, hostile-input safety), and the
distributed-specific **tiered acceptance** (a byte-different but legal next-state passes the cheap
tier where bit-exact rejects it).
"""

from __future__ import annotations

import copy
import random
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DEFAULT_DIST_CONFIG
from verisim.dist.delta import delta_to_list
from verisim.dist.serialize import to_canonical
from verisim.dist.state import DistributedState
from verisim.distcontrib import (
    content_address,
    transition_record,
    verify_trajectory,
    verify_transition,
)
from verisim.distdata import DistDriver
from verisim.distoracle import ReferenceDistOracle


def _oracle() -> ReferenceDistOracle:
    return ReferenceDistOracle(DEFAULT_DIST_CONFIG)


def _genuine_transition(seed: int = 1):
    oracle = _oracle()
    drv = DistDriver("adversarial", DEFAULT_DIST_CONFIG, random.Random(seed))
    state = DistributedState.initial(DEFAULT_DIST_CONFIG)
    # walk a few steps so the medium (inflight/partition) is non-trivial, then take one more
    for _ in range(5):
        state = oracle.step(state, drv.sample(state)).state
    action = drv.sample(state)
    result = oracle.step(state, action)
    return state, action, result


def _genuine_trajectory(seed: int, n: int) -> dict[str, Any]:
    oracle = _oracle()
    drv = DistDriver("adversarial", DEFAULT_DIST_CONFIG, random.Random(seed))
    state = DistributedState.initial(DEFAULT_DIST_CONFIG)
    steps = []
    for _ in range(n):
        action = drv.sample(state)
        result = oracle.step(state, action)
        steps.append(transition_record(state, action, result.state))
        state = result.state
    return {"steps": steps}


def test_genuine_transition_is_accepted_bit_exact():
    state, action, result = _genuine_transition()
    rec = transition_record(state, action, result.state)
    report = verify_transition(
        rec["state"], rec["action"], rec["next_state"],
        delta=delta_to_list(result.delta),
        observation={"status": result.status, "value": result.value},
    )
    assert report.accepted
    assert report.tier == "bit_exact"
    assert report.n_reproduced == 1
    assert report.oracle_cost == 16  # the bit-exact tier cost (§5)


def test_forged_next_state_is_rejected():
    state, action, result = _genuine_transition()
    rec = transition_record(state, action, result.state)
    forged = to_canonical(result.state)
    assert forged["replicas"], "expected the walk to have written a replica"
    forged["replicas"][0]["value"] = "Q"  # out-of-vocab corruption
    report = verify_transition(rec["state"], rec["action"], forged)
    assert not report.accepted
    assert "bit_exact_refuted" in report.mismatches


def test_wrong_delta_is_rejected_even_when_next_state_matches():
    # the next-state reproduces but the *claimed delta* is a lie -> rejected at the delta facet.
    state, action, result = _genuine_transition()
    rec = transition_record(state, action, result.state)
    bogus_delta = [{"kind": "ReplicaWrite", "object_id": "x", "node_id": "n0",
                    "version": 999, "value": "Q"}]
    report = verify_transition(rec["state"], rec["action"], rec["next_state"], delta=bogus_delta)
    assert not report.accepted
    assert "delta" in report.mismatches


def test_tiered_acceptance_admits_a_byte_different_but_legal_next_state():
    # the W7 / §16 path: a `get` (read) must only leave replicas intact. A next-state with the same
    # replicas but a different (legal) clock is bit-exact-different yet consistency-admissible: the
    # bit_exact tier rejects it, a cheaper tier admits it.
    oracle = _oracle()
    state = DistributedState.initial(DEFAULT_DIST_CONFIG)
    for raw in ("put n0 x v1", "advance 1"):
        state = oracle.step(state, parse_dist_action(raw)).state
    action = parse_dist_action("get n0 x")
    result = oracle.step(state, action)
    rec = transition_record(state, action, result.state)
    alt = copy.deepcopy(rec["next_state"])
    alt["clock"] = alt["clock"] + 5  # a legal-but-different read outcome

    assert not verify_transition(rec["state"], rec["action"], alt).accepted  # bit_exact rejects
    admitted = verify_transition(rec["state"], rec["action"], alt, tier="symbolic")
    assert admitted.accepted  # the cheap tier admits it (W7)
    assert admitted.oracle_cost == 4  # the symbolic tier cost


def test_cheap_tier_still_catches_genuine_illegality():
    # tiered acceptance is not a loophole: a read that *mutates* a replica is illegal under the
    # symbolic tier, so the cheap tier still rejects it.
    oracle = _oracle()
    state = DistributedState.initial(DEFAULT_DIST_CONFIG)
    state = oracle.step(state, parse_dist_action("put n0 x v1")).state
    action = parse_dist_action("get n0 x")
    result = oracle.step(state, action)
    rec = transition_record(state, action, result.state)
    mutated = copy.deepcopy(rec["next_state"])
    mutated["replicas"][0]["value"] = mutated["replicas"][0]["value"] + "_x"
    assert not verify_transition(rec["state"], rec["action"], mutated, tier="symbolic").accepted


def test_hostile_input_is_rejected_not_raised():
    assert not verify_transition({"bogus": 1}, "garbage action", {"also": 2}).accepted
    assert not verify_transition({"bogus": 1}, "garbage", {"x": 2}, tier="metamorphic").accepted


def test_unknown_tier_raises():
    state, action, result = _genuine_transition()
    rec = transition_record(state, action, result.state)
    try:
        verify_transition(rec["state"], rec["action"], rec["next_state"], tier="psychic")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown tier")


def test_genuine_trajectory_is_accepted_and_chains():
    traj = _genuine_trajectory(seed=2, n=10)
    report = verify_trajectory(traj)
    assert report.accepted
    assert report.n_reproduced == 10
    assert report.oracle_cost == 10 * 16  # ten bit-exact steps


def test_spliced_trajectory_is_rejected():
    traj = _genuine_trajectory(seed=3, n=8)
    spliced = copy.deepcopy(traj)
    spliced["steps"][4]["state"] = to_canonical(DistributedState.initial(DEFAULT_DIST_CONFIG))
    report = verify_trajectory(spliced)
    assert not report.accepted
    assert report.first_failure == 4
    assert "chaining" in report.mismatches


def test_empty_trajectory_is_rejected():
    report = verify_trajectory({"steps": []})
    assert not report.accepted
    assert report.n_transitions == 0


def test_content_address_is_stable_and_order_independent():
    a = {"state": {"x": 1, "y": 2}, "action": "advance 1"}
    b = {"action": "advance 1", "state": {"y": 2, "x": 1}}  # different dict ordering
    assert content_address(a) == content_address(b)
    c = {"action": "advance 2", "state": {"x": 1, "y": 2}}
    assert content_address(a) != content_address(c)
