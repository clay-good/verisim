"""Message loss — the ``drop`` fault (SPEC-7 §3.2, DS0 increment 11).

Pins the unreliable-network fault the delta vocabulary already anticipated (``MsgDrop``) but no
action produced: ``drop src dst`` loses every in-flight replication message from ``src`` to ``dst``.
Unlike ``partition`` (which *holds* a message, delivered once the link heals), ``drop`` **destroys**
it, so the destination replica permanently misses that write — the eventual-consistency convergence
guarantee broken until a *newer* write overwrites it. ``drop`` is purely additive (it adds no state
field; existing trajectories never use it), so every prior golden/hash/tokenization is unchanged,
and Tier-B reproduces it bit-for-bit. Dependency-free, GPU-free.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import FAULT_OPS, DistParseError
from verisim.dist.delta import MsgDrop, apply
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle

CONFIG = DistConfig(name="drop-test", nodes=("n0", "n1", "n2"), objects=("x", "y"))


def _run(
    cmds: list[str], cfg: DistConfig = CONFIG
) -> tuple[DistributedState, list[tuple[str, str]]]:
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    results: list[tuple[str, str]] = []
    for cmd in cmds:
        r = oracle.step(state, parse_dist_action(cmd))
        results.append((r.status, r.value))
        state = r.state
    return state, results


def _x(state: DistributedState, node: str) -> tuple[int, str]:
    r = state.replicas[("x", node)]
    return (r.version, r.value)


# --- the grammar ---------------------------------------------------------------------------------


def test_drop_parses_with_two_args_and_is_a_fault_op():
    action = parse_dist_action("drop n0 n1")
    assert action.name == "drop"
    assert action.args == ("n0", "n1")
    assert "drop" in FAULT_OPS


def test_drop_requires_exactly_two_args():
    for bad in ("drop", "drop n0", "drop n0 n1 n2"):
        try:
            parse_dist_action(bad)
        except DistParseError:
            continue
        raise AssertionError(f"expected {bad!r} to fail parsing")


# --- the semantics -------------------------------------------------------------------------------


def test_drop_loses_the_in_flight_message_so_the_peer_stays_stale():
    """put enqueues n0->n1 and n0->n2; drop n0 n1 then advance delivers only n0->n2."""
    state, results = _run(["put n0 x b", "drop n0 n1", "advance 2"])
    assert results[1] == ("dropped", "1")  # exactly one in-flight n0->n1 message lost
    assert results[2] == ("advanced", "1")  # only the surviving n0->n2 message delivered
    assert _x(state, "n0") == (1, "b")  # the writer
    assert _x(state, "n2") == (1, "b")  # received its message
    assert _x(state, "n1") == (0, "nil")  # dropped: permanently stale at the boot value
    assert not state.inflight  # nothing held — the dropped message is gone, not waiting


def test_drop_is_unrecoverable_where_partition_is_recoverable():
    """The medium contrast: a held message is delivered on heal; a dropped one never is."""
    partitioned, _ = _run(
        ["put n0 x b", "partition n0 n2 | n1", "advance 2", "heal", "advance 2"]
    )
    assert _x(partitioned, "n1") == (1, "b")  # the held message was delivered on heal+advance

    dropped, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "heal", "advance 2"])
    assert _x(dropped, "n1") == (0, "nil")  # the dropped message is gone — heal cannot repair it


def test_only_a_newer_write_heals_a_dropped_write_and_the_lost_value_is_never_seen():
    oracle = ReferenceDistOracle(CONFIG)
    state = DistributedState.initial(CONFIG)
    seen_at_n1: set[str] = set()
    for cmd in ["put n0 x b", "drop n0 n1", "advance 2", "put n0 x c", "advance 2"]:
        state = oracle.step(state, parse_dist_action(cmd)).state
        seen_at_n1.add(state.replicas[("x", "n1")].value)
    assert _x(state, "n1") == (2, "c")  # the overwrite reaches n1
    assert "b" not in seen_at_n1  # the dropped value was never observed at n1 (lost, not delayed)


def test_drop_with_no_in_flight_message_is_a_no_op():
    state, results = _run(["drop n0 n1"])
    assert results[0] == ("dropped", "0")
    assert not state.inflight


def test_drop_targets_only_the_named_channel():
    """drop n0 n1 loses the n0->n1 message but leaves the n0->n2 message in flight."""
    state, results = _run(["put n0 x b", "drop n0 n1"])
    assert results[1] == ("dropped", "1")
    survivors = sorted((m.src, m.dst) for m in state.inflight.values())
    assert survivors == [("n0", "n2")]


# --- additive: apply == oracle, no canonical-form change off the drop path, round-trip -----------


def test_apply_equals_oracle_on_a_drop_step():
    oracle = ReferenceDistOracle(CONFIG)
    state = oracle.step(DistributedState.initial(CONFIG), parse_dist_action("put n0 x b")).state
    r = oracle.step(state, parse_dist_action("drop n0 n1"))
    assert apply(state, r.delta) == r.state
    assert any(isinstance(e, MsgDrop) for e in r.delta)


def test_drop_adds_no_state_field_so_canonical_form_round_trips():
    state, _ = _run(["put n0 x b", "drop n0 n1", "advance 2"])
    canon = to_canonical(state)
    assert from_canonical(canon) == state
    # the dropped message left no residue: in-flight is empty and there is no new top-level key
    assert canon["inflight"] == []


# --- Tier-B agreement ----------------------------------------------------------------------------


def test_drop_tier_b_agrees_bit_for_bit():
    """Message loss is a medium change, so Tier-B computes a byte-identical drop and then reproduces
    the broken (and overwrite-repaired) convergence on its own autonomous-actor delivery."""
    ref, sysb = ReferenceDistOracle(CONFIG), SystemDistOracle(CONFIG)
    scripts = [
        ["put n0 x b", "drop n0 n1", "advance 2"],
        ["put n0 x b", "drop n0 n1", "advance 2", "heal", "advance 2"],
        ["put n0 x b", "drop n0 n1", "advance 2", "put n0 x c", "advance 2"],
        ["put n0 x b", "put n0 y c", "drop n0 n2", "advance 2"],  # one channel, multiple objects
        ["put n1 x a", "drop n1 n0", "drop n1 n2", "advance 2"],  # both peers cut off
    ]
    for script in scripts:
        sa = sb = DistributedState.initial(CONFIG)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            assert cluster_view(ra.state) == cluster_view(rb.state), (cmd, script)
            assert ra.status == rb.status and ra.value == rb.value
            sa, sb = ra.state, rb.state
