"""Message-timing faults — ``delay`` and ``reorder`` (SPEC-7 §3.2/§3.4, DS0 increment 13).

Pins the two message-timing faults SPEC-7 §3.4 names as the *medium* ("partition, crash, message
loss, **reorder**, clock skew") but had deferred since increment 1:

  - ``delay src dst dt`` defers every in-flight ``src``->``dst`` message by ``dt`` clock units — a
    *recoverable* delay (the write still arrives, just later), the counterpart to ``drop``'s
    *unrecoverable* loss. So delay extends the stale-read window but convergence is preserved.
  - ``reorder src dst`` reverses the delivery schedule of that channel's messages. Last-writer-wins
    makes the *converged* replica invariant under reordering (a commutative join — the property
    Tier-B's shuffled scheduler already certifies), but it flips which write a peer sees *in
    transit*.

Both only edit the existing ``Message.deliver_after`` field (no new state), so every prior
golden/hash/tokenization is unchanged, and — being pure medium changes — Tier-A and Tier-B compute
byte-identical deltas. Dependency-free, GPU-free.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import FAULT_OPS, DistParseError
from verisim.dist.delta import MsgReschedule, apply
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle

CONFIG = DistConfig(name="timing-test", nodes=("n0", "n1", "n2"), objects=("x", "y"))


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


def _channel_times(state: DistributedState, src: str, dst: str) -> list[int]:
    return sorted(m.deliver_after for m in state.inflight.values() if m.src == src and m.dst == dst)


# --- the grammar ---------------------------------------------------------------------------------


def test_delay_and_reorder_parse_and_are_fault_ops():
    delay = parse_dist_action("delay n0 n1 5")
    assert delay.name == "delay" and delay.args == ("n0", "n1", "5")
    reorder = parse_dist_action("reorder n0 n1")
    assert reorder.name == "reorder" and reorder.args == ("n0", "n1")
    assert {"delay", "reorder"} <= FAULT_OPS


def test_delay_requires_three_args_and_a_positive_dt():
    for bad in ("delay", "delay n0 n1", "delay n0 n1 5 6", "delay n0 n1 0", "delay n0 n1 -3",
                "delay n0 n1 x"):
        try:
            parse_dist_action(bad)
        except DistParseError:
            continue
        raise AssertionError(f"expected {bad!r} to fail parsing")


def test_reorder_requires_exactly_two_args():
    for bad in ("reorder", "reorder n0", "reorder n0 n1 n2"):
        try:
            parse_dist_action(bad)
        except DistParseError:
            continue
        raise AssertionError(f"expected {bad!r} to fail parsing")


# --- delay: a recoverable deferral -------------------------------------------------------------


def test_delay_defers_the_message_but_it_still_arrives():
    """The recoverable-delay contract: a delayed message is not lost, only deferred."""
    # put enqueues n0->n1 (deliver_after=1); delay pushes it to 1+5=6, so advance 2 cannot deliver.
    state, results = _run(["put n0 x b", "delay n0 n1 5", "advance 2"])
    assert results[1] == ("delayed", "1")  # exactly one in-flight n0->n1 message deferred
    assert _x(state, "n1") == (0, "nil")  # not yet delivered at clock 2 (deferred to 6)
    assert _channel_times(state, "n0", "n1") == [6]  # held in flight, due later — not destroyed
    # advancing past the deferral delivers it: convergence is preserved (unlike drop).
    state2, _ = _run(["put n0 x b", "delay n0 n1 5", "advance 2", "advance 10"])
    assert _x(state2, "n1") == (1, "b")  # the delayed write arrived — recoverable


def test_delay_is_recoverable_where_drop_is_not():
    """The medium contrast SPEC-7 frames: a delayed write converges; a dropped one never does."""
    delayed, _ = _run(["put n0 x b", "delay n0 n1 5", "advance 2", "advance 10"])
    assert _x(delayed, "n1") == (1, "b")  # recoverable: arrives once the clock passes the deferral
    dropped, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "advance 10"])
    assert _x(dropped, "n1") == (0, "nil")  # unrecoverable: the dropped write never arrives


def test_delay_targets_only_the_named_channel():
    state, results = _run(["put n0 x b", "delay n0 n1 5"])
    assert results[1] == ("delayed", "1")
    assert _channel_times(state, "n0", "n1") == [6]  # deferred
    assert _channel_times(state, "n0", "n2") == [1]  # the other peer's message is untouched


def test_delay_with_no_in_flight_message_is_a_no_op():
    state, results = _run(["delay n0 n1 5"])
    assert results[0] == ("delayed", "0")
    assert not state.inflight


# --- reorder: flips the transit observation, never the converged value -------------------------


def test_reorder_flips_which_write_a_peer_sees_in_transit_but_not_the_converged_value():
    """Stagger two writes (delay the first), then reorder reverses which lands first; LWW keeps the
    converged value fixed at the newer write — delivery-order independence as a fault."""
    # v1=b then v2=c to n1; delay v1's message far out so v2 is scheduled first.
    setup = ["put n0 x b", "delay n0 n1 100", "put n0 x c"]
    scheduled, _ = _run([*setup, "advance 2"])
    assert _x(scheduled, "n1") == (2, "c")  # in transit: the newer write (v1 deferred) lands first
    reordered, _ = _run([*setup, "reorder n0 n1", "advance 2"])
    assert _x(reordered, "n1") == (1, "b")  # reorder flipped it: the older write lands first now
    # ...yet both converge to the newer write once everything is delivered (LWW commutativity).
    sched_final, _ = _run([*setup, "advance 2", "advance 200"])
    reord_final, _ = _run([*setup, "reorder n0 n1", "advance 2", "advance 200"])
    assert _x(sched_final, "n1") == _x(reord_final, "n1") == (2, "c")


def test_reorder_reverses_the_channel_delivery_schedule():
    # two staggered messages on n0->n1: deliver_after {1, 101}; reorder swaps the assignment.
    state, results = _run(["put n0 x b", "delay n0 n1 100", "put n0 x c", "reorder n0 n1"])
    assert results[3] == ("reordered", "2")  # both messages moved
    assert _channel_times(state, "n0", "n1") == [1, 101]  # the multiset of times is preserved
    # the message that was due first (the v2 write at deliver_after=1) is now due last, and vice
    # versa — verified via the transit observation in the test above.


def test_reorder_of_a_single_message_channel_is_a_no_op():
    state, results = _run(["put n0 x b", "reorder n0 n1"])
    assert results[1] == ("reordered", "0")  # nothing to reorder
    assert _channel_times(state, "n0", "n1") == [1]


def test_reorder_of_equal_time_messages_is_observationally_inert():
    """Two concurrent writes (same deliver_after) are reorder-invariant — nothing to flip."""
    state, results = _run(["put n0 x b", "put n0 x c", "reorder n0 n1"])
    assert results[2] == ("reordered", "0")  # equal times: the reversal changes nothing
    assert _channel_times(state, "n0", "n1") == [1, 1]


# --- additive: apply == oracle, no canonical-form change off the timing path, round-trip --------


def test_apply_equals_oracle_on_delay_and_reorder_steps():
    oracle = ReferenceDistOracle(CONFIG)
    base = oracle.step(DistributedState.initial(CONFIG), parse_dist_action("put n0 x b")).state
    for cmd in ("delay n0 n1 5", "reorder n0 n1"):
        r = oracle.step(base, parse_dist_action(cmd))
        assert apply(base, r.delta) == r.state
    # delay actually emits a reschedule edit (reorder of a 1-msg channel is a no-op, so does not).
    delay_r = oracle.step(base, parse_dist_action("delay n0 n1 5"))
    assert any(isinstance(e, MsgReschedule) for e in delay_r.delta)


def test_timing_faults_add_no_state_field_so_canonical_form_round_trips():
    state, _ = _run(["put n0 x b", "delay n0 n1 5", "put n0 x c", "reorder n0 n1", "advance 200"])
    canon = to_canonical(state)
    assert from_canonical(canon) == state
    # the only residue is the in-flight messages' deliver_after — an existing field, no new key.
    assert set(canon) == set(to_canonical(DistributedState.initial(CONFIG)))


# --- Tier-B agreement ----------------------------------------------------------------------------


def test_timing_faults_tier_b_agree_bit_for_bit():
    """Message timing is a medium change, so Tier-B computes a byte-identical reschedule and then
    reproduces the delayed/reordered delivery on its own shuffled autonomous-actor scheduler."""
    ref, sysb = ReferenceDistOracle(CONFIG), SystemDistOracle(CONFIG)
    scripts = [
        ["put n0 x b", "delay n0 n1 5", "advance 2", "advance 10"],
        ["put n0 x b", "delay n0 n1 100", "put n0 x c", "reorder n0 n1", "advance 2",
         "advance 200"],
        ["put n0 x b", "put n0 y c", "delay n0 n2 3", "advance 2", "advance 5"],
        ["put n1 x a", "reorder n1 n0", "advance 2"],
        ["put n0 x b", "delay n0 n1 4", "delay n0 n2 2", "advance 3", "advance 3"],
    ]
    for script in scripts:
        sa = sb = DistributedState.initial(CONFIG)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            assert cluster_view(ra.state) == cluster_view(rb.state), (cmd, script)
            assert ra.status == rb.status and ra.value == rb.value
            sa, sb = ra.state, rb.state
