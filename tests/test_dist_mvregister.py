"""DS0 increment 31 — the CRDT MV-register: ``mvput``/``mvget`` + conflict-surfacing.

A *state-based* multi-value register (SPEC-7 §3.2), the Dynamo/Riak data type that **surfaces** a
write conflict instead of silently dropping one (the LWW the KV ``put`` and the counters do). It
reuses the OR-Set's dot/union machinery: ``mvput n key val`` tags ``val`` with a fresh dot,
**tombstones every dot it observes** (a write supersedes the values it saw), and adds its own;
``mvget`` reads the surviving (non-tombstoned) sibling values. A *sequential* overwrite collapses to
one value, *concurrent* writes leave **both** as siblings, and a later context-aware write
**resolves** them. The join is **set union** of both halves (commutative/associative/idempotent).
Tier-A ≡ Tier-B, purely additive over increment 30 (two omitted-when-empty ``mvreg_vals``/
``mvreg_tombs`` maps + the ``MVRegWrite``/``MVRegTomb`` edits).
"""

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import CLIENT_OPS, DistParseError
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(n: int = 5) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="mv", nodes=nodes, objects=("r",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


def _siblings(value: str) -> set[str]:
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_mvreg_ops() -> None:
    assert parse_dist_action("mvput n0 r a").args == ("n0", "r", "a")
    assert parse_dist_action("mvget n0 r").args == ("n0", "r")
    assert {"mvput", "mvget"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["mvput n0 r", "mvget n0 r v", "mvput n0 r a b"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local write / read --------------------------------------------------------------------

def test_mvput_then_mvget_reads_the_value() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mvput n0 r a"])
    assert _siblings(oracle.step(s, parse_dist_action("mvget n0 r")).value) == {"a"}


def test_sequential_overwrite_resolves_to_one() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mvput n0 r a", "mvput n0 r b", "mvput n0 r c"])
    # each write observed the prior, so it superseded it — a single resolved value.
    assert _siblings(oracle.step(s, parse_dist_action("mvget n0 r")).value) == {"c"}


def test_mvput_is_always_available_even_partitioned_alone() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("mvput n0 r z"))
    assert (r.status, r.value) == ("ok", "z")


def test_crashed_node_mvput_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("mvput n0 r a")).status == \
        "unavailable"


def test_unique_dots_per_node() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mvput n0 r a", "mvput n0 r b"])
    seqs = sorted(seq for (val, owner, seq) in s.mvreg_vals[("r", "n0")] if owner == "n0")
    assert seqs == [1, 2]  # the overwrite minted a fresh dot, did not reuse the superseded one


# --- conflict surfacing + convergence (the resolution the KV lacks) -----------------------------

def test_concurrent_writes_survive_as_siblings() -> None:
    # The direct contrast with the LWW KV: two concurrent mvputs on opposite sides BOTH survive,
    # where a `put` would keep only one (silent loss).
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "mvput n0 r a", "mvput n3 r b", "heal", "gossip n0 n3"])
    assert _siblings(oracle.step(s, parse_dist_action("mvget n0 r")).value) == {"a", "b"}


def test_context_aware_write_resolves_siblings() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "mvput n0 r a", "mvput n3 r b", "heal", "gossip n0 n3"])
    # n0 now observes both siblings; its write supersedes both → a single resolved value.
    s = oracle.step(s, parse_dist_action("mvput n0 r c")).state
    s = oracle.step(s, parse_dist_action("gossip n0 n3")).state
    assert _siblings(oracle.step(s, parse_dist_action("mvget n0 r")).value) == {"c"}
    assert _siblings(oracle.step(s, parse_dist_action("mvget n3 r")).value) == {"c"}


def test_anti_entropy_converges_all_nodes() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "mvput n0 r a", "mvput n3 r b", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert _siblings(oracle.step(s, parse_dist_action(f"mvget {nd} r")).value) == {"a", "b"}


def test_gossip_union_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 | n2", "mvput n0 r a", "mvput n2 r b", "heal",
              "gossip n0 n2", "gossip n0 n2"])
    assert _siblings(oracle.step(s, parse_dist_action("mvget n0 r")).value) == {"a", "b"}


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_mvreg_until_first_op() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert "mvreg_vals" not in to_canonical(s)  # never-written cluster: purely additive
    assert "mvreg_tombs" not in to_canonical(s)
    written = _run(ReferenceDistOracle(config), config, ["mvput n0 r a"])
    assert to_canonical(written)["mvreg_vals"][0] == {
        "key": "r", "holder": "n0", "value": "a", "owner": "n0", "seq": 1
    }
    overwritten = _run(ReferenceDistOracle(config), config, ["mvput n0 r a", "mvput n0 r b"])
    assert to_canonical(overwritten)["mvreg_tombs"][0] == {
        "key": "r", "holder": "n0", "owner": "n0", "seq": 1
    }
    assert from_canonical(to_canonical(overwritten)) == overwritten
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_mvreg_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["mvput n0 r a"])
    assert "'mvreg_vals'" in cluster_view(s)


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_phantom_dot() -> None:
    from verisim.dist.delta import MVRegWrite, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mvput n0 r a"])
    bogus = apply(s, [MVRegWrite("r", "n0", "a", "ghost", 1)])  # owner is not a cluster node
    verdict = TieredOracle(config).check("metamorphic", s, parse_dist_action("mvput n0 r a"), bogus)
    assert verdict.refuted and "mvreg" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_an_mvreg_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "mvput n0 r a", "mvput n0 r b", "mvget n0 r",
        "partition n0 n1 n2 | n3 n4",
        "mvput n0 r x", "mvput n3 r y",             # concurrent writes, both ack (AP)
        "mvget n0 r", "mvget n3 r",                 # local views diverge, nothing lost
        "heal", "gossip n0 n3", "anti_entropy n1", "mvget n4 r",  # converge to the siblings
        "mvput n0 r z", "gossip n0 n3", "mvget n0 r",             # resolve to one value
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
