"""DS0 increment 1 — the distributed deterministic core (SPEC-7 §3-§5).

Property + semantics tests for the dependency-free, GPU-free core (the DS0 gate, SPEC-7 §13):

  - the **M1-analogue invariant** ``apply(state, oracle.delta) == oracle.next_state`` on every
    action type;
  - canonical state + delta **serialization round-trips**;
  - **determinism**: the same action sequence yields the same final state;
  - the **eventual-consistency semantics**: stale reads under partition, convergence after heal;
  - the config curriculum dials + the action grammar.
"""

from __future__ import annotations

import pytest

from verisim.dist import (
    DistConfig,
    DistParseError,
    DistributedState,
    apply,
    parse_dist_action,
    scaled_dist_config,
)
from verisim.dist.delta import delta_from_list, delta_to_list
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistStepResult

CFG = DistConfig()  # 3 nodes n0..n2, objects x,y, full replication, eventual


def _run(actions: list[str], config: DistConfig = CFG) -> list[DistStepResult]:
    """Step a sequence through the oracle; return the step results (asserts M1 each step)."""
    oracle = ReferenceDistOracle(config)
    state = DistributedState.initial(config)
    results = []
    for raw in actions:
        result = oracle.step(state, parse_dist_action(raw))
        assert apply(state, result.delta) == result.state  # the M1-analogue invariant
        results.append(result)
        state = result.state
    return results


# --- the M1 invariant, every action type ---------------------------------------------------------

def test_apply_equals_oracle_every_action():
    actions = [
        "put n0 x b", "get n1 x", "cas n0 x b c", "advance 2",
        "partition n0 n1 | n2", "put n0 x d", "advance 3", "heal", "advance 3",
        "crash n2", "put n0 y a", "advance 2", "restart n2", "advance 2", "get n2 y",
    ]
    results = _run(actions)
    assert len(results) == len(actions)
    # a write enqueues replication; an advance can deliver it
    assert any(r.status == "ok" for r in results)


# --- serialization round-trips -------------------------------------------------------------------

def test_state_serialization_round_trips():
    state = _run(["put n0 x b", "advance 2", "partition n0 | n1 n2", "put n0 x c"])[-1].state
    assert from_canonical(to_canonical(state)) == state


def test_partition_order_is_canonical_so_round_trip_is_exact():
    # a partition declared in *non-sorted* group order must still round-trip bit-for-bit: the state
    # stores partitions in canonical (sorted) order, so to_canonical/from_canonical is an exact
    # inverse regardless of how the oracle/driver happened to name the groups (the §16 prereq).
    state = _run(["partition n1 n2 | n0"])[-1].state
    assert state.partitions == (frozenset({"n0"}), frozenset({"n1", "n2"}))  # sorted, not declared
    assert from_canonical(to_canonical(state)) == state


def test_delta_serialization_round_trips():
    for result in _run(["put n0 x b", "advance 2", "crash n1", "heal", "get n0 x"]):
        assert delta_from_list(delta_to_list(result.delta)) == result.delta


def test_replicas_converge_order_independently():
    # The causal *log* records op order (so full-state hashes differ), but the replicated *data*
    # converges to the same values regardless of the order of independent writes.
    a = _run(["put n0 x b", "put n0 y a", "advance 2"])[-1].state
    b = _run(["put n0 y a", "put n0 x b", "advance 2"])[-1].state
    assert a.replicas == b.replicas


# --- determinism ---------------------------------------------------------------------------------

def test_determinism():
    seq = ["put n0 x b", "advance 1", "partition n0 n1 | n2", "put n0 x c", "advance 2", "heal"]
    assert state_hash(_run(seq)[-1].state) == state_hash(_run(seq)[-1].state)


# --- the eventual-consistency semantics ----------------------------------------------------------

def _value_at(oracle: ReferenceDistOracle, state: DistributedState, node: str, key: str) -> str:
    return oracle.step(state, parse_dist_action(f"get {node} {key}")).value


def test_put_is_local_then_converges():
    oracle = ReferenceDistOracle(CFG)
    state = oracle.step(DistributedState.initial(CFG), parse_dist_action("put n0 x b")).state
    # immediately: coordinator sees b, peers still hold the boot default (async replication)
    assert _value_at(oracle, state, "n0", "x") == "b"
    assert _value_at(oracle, state, "n1", "x") == CFG.default_value
    # after advancing, the replication message delivers and n1 converges
    state = oracle.step(state, parse_dist_action("advance 5")).state
    assert _value_at(oracle, state, "n1", "x") == "b"


def test_partition_yields_stale_reads_then_heal_converges():
    oracle = ReferenceDistOracle(CFG)
    state = DistributedState.initial(CFG)
    for raw in ["put n0 x b", "advance 5", "partition n0 n1 | n2", "put n0 x c", "advance 5"]:
        state = oracle.step(state, parse_dist_action(raw)).state
    assert _value_at(oracle, state, "n1", "x") == "c"  # same side as the writer
    assert _value_at(oracle, state, "n2", "x") == "b"  # isolated: stale
    state = oracle.step(state, parse_dist_action("heal")).state
    state = oracle.step(state, parse_dist_action("advance 5")).state
    assert _value_at(oracle, state, "n2", "x") == "c"  # converged after heal


def test_crashed_node_does_not_receive_until_restart():
    oracle = ReferenceDistOracle(CFG)
    state = DistributedState.initial(CFG)
    for raw in ["crash n1", "put n0 x b", "advance 5"]:
        state = oracle.step(state, parse_dist_action(raw)).state
    # n1 is down; its replica is unchanged and a read is unavailable
    assert oracle.step(state, parse_dist_action("get n1 x")).status == "unavailable"
    state = oracle.step(state, parse_dist_action("restart n1")).state
    state = oracle.step(state, parse_dist_action("advance 5")).state
    assert _value_at(oracle, state, "n1", "x") == "b"  # delivered after restart


def test_cas_conflicts_on_stale_value():
    oracle = ReferenceDistOracle(CFG)
    state = oracle.step(DistributedState.initial(CFG), parse_dist_action("put n0 x b")).state
    ok = oracle.step(state, parse_dist_action("cas n0 x b c"))
    assert ok.status == "ok" and ok.value == "c"
    conflict = oracle.step(state, parse_dist_action("cas n0 x WRONG d"))
    assert conflict.status == "conflict" and conflict.value == "b"


# --- config + grammar ----------------------------------------------------------------------------

def test_scaled_dist_config():
    cfg = scaled_dist_config(5, n_objects=3)
    assert cfg.nodes == ("n0", "n1", "n2", "n3", "n4")
    assert cfg.objects == ("o0", "o1", "o2")
    assert cfg.replication_factor == 3
    assert cfg.replicas_of("o0") == ("n0", "n1", "n2")
    state = DistributedState.initial(cfg)
    assert ("o0", "n0") in state.replicas and ("o0", "n3") not in state.replicas


def test_config_validation():
    with pytest.raises(ValueError):
        DistConfig(consistency_model="bogus")
    with pytest.raises(ValueError):
        DistConfig(replication_factor=99)
    with pytest.raises(ValueError):
        scaled_dist_config(0)


def test_action_grammar_parse_and_errors():
    assert parse_dist_action("put n0 x b").name == "put"
    assert parse_dist_action("partition n0 n1 | n2").groups == (("n0", "n1"), ("n2",))
    with pytest.raises(DistParseError):
        parse_dist_action("")
    with pytest.raises(DistParseError):
        parse_dist_action("frob n0")
    with pytest.raises(DistParseError):
        parse_dist_action("put n0 x")  # wrong arity
    with pytest.raises(DistParseError):
        parse_dist_action("partition n0 n1")  # only one group
    with pytest.raises(DistParseError):
        parse_dist_action("advance -1")  # non-positive dt
