"""Clock skew — the `clock_skew` fault (SPEC-7 §3.2/§3.4, DS0 increment 14).

Pins the last of the §3.4 medium faults ("partition, crash, message loss, reorder, **clock skew**").
`clock_skew node delta` offsets a node's local clock by a signed `delta`, which shifts the
`deliver_after` it stamps on the messages it sends (via `DistributedState.sender_clock`). Because
the protocol resolves conflicts by last-writer-wins on `(version, value)` — never on a wall-clock
timestamp — skew shifts *when* a write is delivered but never *which* write wins, so convergence is
clock-independent. It adds one omitted-when-empty `skew` map (no per-message state), so a
synchronized cluster is byte-identical to the pre-increment-14 form, and — being a medium change —
Tier-A and Tier-B compute byte-identical deltas. Dependency-free, GPU-free.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import FAULT_OPS, DistParseError
from verisim.dist.delta import ClockSkewSet, apply
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle

CONFIG = DistConfig(name="skew-test", nodes=("n0", "n1", "n2"), objects=("x", "y"))


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


def _send_times(state: DistributedState, src: str) -> list[int]:
    return sorted(m.deliver_after for m in state.inflight.values() if m.src == src)


# --- the grammar ---------------------------------------------------------------------------------


def test_clock_skew_parses_with_a_signed_delta_and_is_a_fault_op():
    for raw, delta in (("clock_skew n0 3", "3"), ("clock_skew n0 -4", "-4"),
                       ("clock_skew n0 0", "0")):
        action = parse_dist_action(raw)
        assert action.name == "clock_skew"
        assert action.args == ("n0", delta)
    assert "clock_skew" in FAULT_OPS


def test_clock_skew_requires_exactly_two_args_and_an_integer_delta():
    for bad in ("clock_skew", "clock_skew n0", "clock_skew n0 1 2", "clock_skew n0 x"):
        try:
            parse_dist_action(bad)
        except DistParseError:
            continue
        raise AssertionError(f"expected {bad!r} to fail parsing")


# --- the semantics -------------------------------------------------------------------------------


def test_skew_shifts_a_nodes_send_timing_by_exactly_delta():
    """A node skewed by δ stamps deliver_after = clock + δ + 1 on every message it sends."""
    base, _ = _run(["put n0 x b"])
    assert _send_times(base, "n0") == [1, 1]  # un-skewed: clock 0 + 1
    for delta in (-3, 2, 5):
        skewed, results = _run([f"clock_skew n0 {delta}", "put n0 x b"])
        assert results[0] == ("skewed", str(delta))
        assert _send_times(skewed, "n0") == [1 + delta, 1 + delta]  # shifted by exactly δ


def test_positive_skew_defers_delivery_negative_rushes_it():
    # +5 (ahead): deliver_after 6, not due at clock 2 — the peer is still stale.
    ahead, _ = _run(["clock_skew n0 5", "put n0 x b", "advance 2"])
    assert _x(ahead, "n1") == (0, "nil")
    # -3 (behind): deliver_after -2, immediately due — delivered on the first advance.
    behind, _ = _run(["clock_skew n0 -3", "put n0 x b", "advance 1"])
    assert _x(behind, "n1") == (1, "b")


def test_convergence_is_clock_independent_across_a_skew_sweep():
    """The headline: version-LWW makes the converged state invariant to any node's clock skew."""
    baseline = to_canonical(_run(["put n0 x b", "advance 100"])[0])["replicas"]
    for delta in (-7, -2, 0, 3, 9):
        final = to_canonical(_run([f"clock_skew n0 {delta}", "put n0 x b", "advance 100"])[0])
        assert final["replicas"] == baseline  # skew shifts timing, never the converged value


def test_skew_is_persistent_across_multiple_sends():
    # both writes from the skewed node are shifted — a persistent per-node property (unlike delay).
    state, _ = _run(["clock_skew n0 4", "put n0 x b", "advance 5", "put n0 x c"])
    # the second send (at clock 5) is also shifted: deliver_after = 5 + 4 + 1 = 10.
    assert _send_times(state, "n0") == [10, 10]


def test_clock_skew_zero_clears_the_offset():
    state, _ = _run(["clock_skew n0 3", "clock_skew n0 0", "put n0 x b"])
    assert state.skew == {}  # re-synced: no residue
    assert _send_times(state, "n0") == [1, 1]  # back to the un-skewed timing


# --- additive: apply == oracle, no canonical-form change off the skew path, round-trip ----------


def test_apply_equals_oracle_on_a_clock_skew_step():
    oracle = ReferenceDistOracle(CONFIG)
    r = oracle.step(DistributedState.initial(CONFIG), parse_dist_action("clock_skew n0 3"))
    assert apply(DistributedState.initial(CONFIG), r.delta) == r.state
    assert any(isinstance(e, ClockSkewSet) for e in r.delta)


def test_skew_is_omitted_from_canonical_when_empty_round_trips_when_set():
    plain, _ = _run(["put n0 x b", "advance 2"])
    assert "skew" not in to_canonical(plain)  # synchronized cluster: pre-incr-14 normal form
    skewed, _ = _run(["clock_skew n1 -2", "put n0 x b"])
    canon = to_canonical(skewed)
    assert canon["skew"] == {"n1": -2}
    assert from_canonical(canon) == skewed  # exact round-trip with skew present


# --- Tier-B agreement ----------------------------------------------------------------------------


def test_clock_skew_tier_b_agrees_bit_for_bit():
    """Skew is a medium change (a per-node send-timestamp offset), so Tier-B computes a byte-
    identical skew and reproduces the shifted-but-invariant convergence on its own scheduler."""
    ref, sysb = ReferenceDistOracle(CONFIG), SystemDistOracle(CONFIG)
    scripts = [
        ["clock_skew n0 3", "put n0 x b", "advance 2", "advance 5"],
        ["clock_skew n1 -4", "put n1 x a", "put n0 x b", "advance 3"],
        ["clock_skew n0 2", "clock_skew n0 0", "put n0 x b", "advance 2"],
        ["clock_skew n0 5", "put n0 x b", "put n0 y c", "advance 3", "advance 5"],
        ["clock_skew n2 -3", "clock_skew n0 4", "put n0 x b", "put n2 x d", "advance 4"],
    ]
    for script in scripts:
        sa = sb = DistributedState.initial(CONFIG)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            assert cluster_view(ra.state) == cluster_view(rb.state), (cmd, script)
            assert ra.status == rb.status and ra.value == rb.value
            sa, sb = ra.state, rb.state
