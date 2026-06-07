"""Anti-entropy / read-repair — the ``anti_entropy`` protocol op (SPEC-7 §5.1, DS0 increment 12).

Pins the convergence mechanism real eventually-consistent stores use to recover *despite* lost
messages — the SPEC-7 §4 ``ReplicaConverge`` op the spec named but had not implemented.
``anti_entropy node`` pulls each object to the winning ``(version, value)`` among the node's
**reachable** replicas. It is the counterpart to ``drop`` (ED18): where message loss breaks the
eventual-consistency convergence guarantee, anti-entropy restores it without a fresh write — but
only over what is reachable. ``anti_entropy`` reuses the ``ReplicaWrite`` edit and adds no state
field, so every prior golden/hash/tokenization is unchanged, and Tier-B reproduces it bit-for-bit.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import PROTOCOL_OPS, DistParseError
from verisim.dist.delta import ReplicaWrite, apply
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle
from verisim.distoracle.tiers import TIERS, TieredOracle

CONFIG = DistConfig(name="ae-test", nodes=("n0", "n1", "n2"), objects=("x", "y"))
ORACLE = ReferenceDistOracle(CONFIG)


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


def test_anti_entropy_parses_with_one_arg_and_is_a_protocol_op():
    action = parse_dist_action("anti_entropy n1")
    assert action.name == "anti_entropy"
    assert action.args == ("n1",)
    assert "anti_entropy" in PROTOCOL_OPS


def test_anti_entropy_requires_exactly_one_arg():
    for bad in ("anti_entropy", "anti_entropy n0 n1"):
        try:
            parse_dist_action(bad)
        except DistParseError:
            continue
        raise AssertionError(f"expected {bad!r} to fail parsing")


# --- the semantics: it repairs what drop broke, where advance cannot -----------------------------


def test_anti_entropy_repairs_a_dropped_write_where_advance_cannot():
    # drop the write to n1, advance, heal: n1 is permanently stale (ED18).
    stale, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "heal", "advance 2"])
    assert _x(stale, "n1") == (0, "nil")  # advance never recovers it — no message remains

    repaired, results = _run(["put n0 x b", "drop n0 n1", "advance 2", "heal", "anti_entropy n1"])
    assert results[-1] == ("repaired", "1")  # one replica repaired
    assert _x(repaired, "n1") == (1, "b")  # read-repair pulled the latest from a reachable peer


def test_anti_entropy_is_bounded_by_reachability():
    # while n1 is partitioned away, anti_entropy reaches only itself — it cannot pull the value.
    partitioned, results = _run(
        ["put n0 x b", "drop n0 n1", "advance 2", "partition n1 | n0 n2", "anti_entropy n1"]
    )
    assert results[-1] == ("repaired", "0")  # nothing reachable holds the new value
    assert _x(partitioned, "n1") == (0, "nil")
    # the same op after heal reaches every replica and repairs.
    healed, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "heal", "anti_entropy n1"])
    assert _x(healed, "n1") == (1, "b")


def test_anti_entropy_adopts_the_winning_version_skipping_intermediates():
    # n1 misses two writes (b@1 then c@2), then read-repairs straight to the latest.
    state, results = _run([
        "put n0 x b", "drop n0 n1", "advance 2",
        "put n0 x c", "drop n0 n1", "advance 2",
        "anti_entropy n1",
    ])
    assert _x(state, "n1") == (2, "c")  # jumps v0 -> v2, skipping the never-seen v1
    assert results[-1] == ("repaired", "1")


def test_anti_entropy_on_a_crashed_node_is_unavailable():
    state, results = _run(["crash n1", "anti_entropy n1"])
    assert results[-1] == ("unavailable", "")
    assert ("x", "n1") in state.replicas  # state otherwise unchanged


def test_anti_entropy_with_nothing_to_repair_is_a_no_op():
    state, results = _run(["put n0 x b", "advance 2", "anti_entropy n1"])  # already converged
    assert results[-1] == ("repaired", "0")
    assert _x(state, "n1") == (1, "b")


# --- additive: apply == oracle, no canonical-form change, round-trip ------------------------------


def test_apply_equals_oracle_on_an_anti_entropy_step():
    state = DistributedState.initial(CONFIG)
    for cmd in ["put n0 x b", "drop n0 n1", "advance 2", "heal"]:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    r = ORACLE.step(state, parse_dist_action("anti_entropy n1"))
    assert apply(state, r.delta) == r.state
    assert any(isinstance(e, ReplicaWrite) for e in r.delta)  # repair is a ReplicaWrite edit


def test_anti_entropy_adds_no_state_field_so_canonical_form_round_trips():
    state, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "heal", "anti_entropy n1"])
    canon = to_canonical(state)
    assert from_canonical(canon) == state


# --- the tiered oracle accepts the read-repair truth (it jumps versions / writes replicas) --------


def test_tiers_accept_anti_entropy_truth():
    state = DistributedState.initial(CONFIG)
    for cmd in ["put n0 x b", "drop n0 n1", "advance 2", "put n0 x c", "advance 2"]:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    action = parse_dist_action("anti_entropy n1")  # truth jumps n1's x from v0 to v2
    truth = ORACLE.step(state, action).state
    tier = TieredOracle(CONFIG)
    for t in TIERS:
        assert not tier.check(t, state, action, truth).refuted, t


# --- Tier-B agreement ----------------------------------------------------------------------------


def test_anti_entropy_tier_b_agrees_bit_for_bit():
    ref, sysb = ReferenceDistOracle(CONFIG), SystemDistOracle(CONFIG)
    scripts = [
        ["put n0 x b", "drop n0 n1", "advance 2", "heal", "anti_entropy n1"],
        ["put n0 x b", "drop n0 n1", "advance 2", "partition n1 | n0 n2", "anti_entropy n1"],
        ["put n0 x b", "put n0 y c", "drop n0 n2", "advance 2", "heal", "anti_entropy n2"],
        ["put n1 x a", "advance 2", "anti_entropy n0", "anti_entropy n2"],  # already converged
        ["crash n1", "anti_entropy n1"],  # unavailable
    ]
    for script in scripts:
        sa = sb = DistributedState.initial(CONFIG)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            assert cluster_view(ra.state) == cluster_view(rb.state), (cmd, script)
            assert ra.status == rb.status and ra.value == rb.value
            sa, sb = ra.state, rb.state
