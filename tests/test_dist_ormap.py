"""DS0 increment 33 — the CRDT OR-Map: ``mput``/``mget``/``mdel``/``mkeys`` + composition.

A *state-based* add-wins observed-remove map (SPEC-7 §3.2), the **capstone** of the CRDT family — a
CRDT *of* CRDTs. It composes the OR-Set (field presence, add-wins + observed-remove over names)
with the LWW-register (each field's value). ``mput n map field val`` adds a fresh presence dot for
``field`` *and* LWW-writes ``val``; ``mdel`` observed-removes the field; ``mget`` reads a present
field's value; ``mkeys`` enumerates the present fields. The join is the OR-Set union of the presence
halves plus the LWW max of each field's value (sharing the Lamport clock). A concurrent ``mput``
survives a concurrent ``mdel`` (add-wins presence) while a field's value resolves by LWW. Tier-A ≡
Tier-B, purely additive over increment 32 (three omitted-when-empty ``ormap_*`` maps + three edits).
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
    return DistConfig(name="om", nodes=nodes, objects=("m",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


def _keys(value: str) -> set[str]:
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_ormap_ops() -> None:
    assert parse_dist_action("mput n0 m f v").args == ("n0", "m", "f", "v")
    assert parse_dist_action("mget n0 m f").args == ("n0", "m", "f")
    assert parse_dist_action("mdel n0 m f").args == ("n0", "m", "f")
    assert parse_dist_action("mkeys n0 m").args == ("n0", "m")
    assert {"mput", "mget", "mdel", "mkeys"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["mput n0 m f", "mget n0 m", "mdel n0 m f x", "mkeys n0 m f"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local map operations ------------------------------------------------------------------

def test_mput_then_mget_mkeys() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mput n0 m name alice", "mput n0 m age 30"])
    assert oracle.step(s, parse_dist_action("mget n0 m name")).value == "alice"
    assert _keys(oracle.step(s, parse_dist_action("mkeys n0 m")).value) == {"name", "age"}


def test_mput_overwrites_value_lww() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mput n0 m k a", "mput n0 m k b"])
    assert oracle.step(s, parse_dist_action("mget n0 m k")).value == "b"  # later write wins


def test_mdel_removes_field() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mput n0 m a 1", "mput n0 m b 2", "mdel n0 m a"])
    assert _keys(oracle.step(s, parse_dist_action("mkeys n0 m")).value) == {"b"}
    assert oracle.step(s, parse_dist_action("mget n0 m a")).value == ""  # absent after remove


def test_re_mput_after_mdel_restores_field() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mput n0 m a 1", "mdel n0 m a", "mput n0 m a 2"])
    assert _keys(oracle.step(s, parse_dist_action("mkeys n0 m")).value) == {"a"}
    assert oracle.step(s, parse_dist_action("mget n0 m a")).value == "2"


def test_mput_is_always_available_even_partitioned_alone() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("mput n0 m k v"))
    assert (r.status, r.value) == ("ok", "v")


def test_crashed_node_mput_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("mput n0 m k v")).status == \
        "unavailable"


# --- the composition: add-wins presence + LWW value ---------------------------------------------

def test_add_wins_field_survives_concurrent_delete() -> None:
    # field k present cluster-wide; partition; n0 re-mputs (fresh dot) while n3 mdels the seen dot.
    # After heal+gossip the field SURVIVES (add-wins) — a naive map would lose the update.
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["mput n0 m k v0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4", "partition n0 n1 n2 | n3 n4", "mput n0 m k v1", "mdel n3 m k",
              "heal", "gossip n0 n3"])
    assert "k" in _keys(oracle.step(s, parse_dist_action("mkeys n0 m")).value)
    assert "k" in _keys(oracle.step(s, parse_dist_action("mkeys n3 m")).value)
    assert oracle.step(s, parse_dist_action("mget n0 m k")).value == "v1"  # the surviving write


def test_concurrent_value_resolves_by_lww() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "mput n0 m k a", "mput n3 m k b", "heal",
              "gossip n0 n3"])
    assert oracle.step(s, parse_dist_action("mget n0 m k")).value == "b"  # owner n3 wins the tie
    assert oracle.step(s, parse_dist_action("mget n3 m k")).value == "b"


def test_anti_entropy_converges_fields_and_values() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "mput n0 m x 1", "mput n3 m y 2", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert _keys(oracle.step(s, parse_dist_action(f"mkeys {nd} m")).value) == {"x", "y"}
        assert oracle.step(s, parse_dist_action(f"mget {nd} m x")).value == "1"
        assert oracle.step(s, parse_dist_action(f"mget {nd} m y")).value == "2"


def test_composed_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 | n2", "mput n0 m x 1", "mput n2 m y 2", "heal",
              "gossip n0 n2", "gossip n0 n2"])
    assert _keys(oracle.step(s, parse_dist_action("mkeys n0 m")).value) == {"x", "y"}


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_ormap_until_first_op() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert not any(k.startswith("ormap") for k in to_canonical(s))  # purely additive
    written = _run(ReferenceDistOracle(config), config, ["mput n0 m f v"])
    canon = to_canonical(written)
    assert canon["ormap_fields"][0] == {
        "map": "m", "holder": "n0", "field": "f", "owner": "n0", "seq": 1
    }
    assert canon["ormap_vals"][0] == {
        "map": "m", "field": "f", "holder": "n0", "value": "v", "ts": 1, "owner": "n0"
    }
    removed = _run(ReferenceDistOracle(config), config, ["mput n0 m f v", "mdel n0 m f"])
    assert to_canonical(removed)["ormap_tombs"][0] == {
        "map": "m", "holder": "n0", "owner": "n0", "seq": 1
    }
    assert from_canonical(to_canonical(removed)) == removed
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_ormap_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["mput n0 m f v"])
    view = cluster_view(s)
    assert "'ormap_fields'" in view and "'ormap_vals'" in view


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_phantom_value_owner() -> None:
    from verisim.dist.delta import ORMapVal, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["mput n0 m f v"])
    bogus = apply(s, [ORMapVal("m", "f", "n0", "v", 1, "ghost")])  # owner not a cluster node
    act = parse_dist_action("mput n0 m f v")
    verdict = TieredOracle(config).check("metamorphic", s, act, bogus)
    assert verdict.refuted and "ormap" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_an_ormap_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "mput n0 m a 1", "mput n0 m b 2", "mkeys n0 m", "mget n0 m a",
        "anti_entropy n1", "anti_entropy n2", "anti_entropy n3", "anti_entropy n4",
        "partition n0 n1 n2 | n3 n4",
        "mput n0 m a 9", "mdel n3 m a", "mput n3 m c 3",   # concurrent: update + delete + add
        "mkeys n0 m", "mkeys n3 m",                        # local views diverge
        "heal", "gossip n0 n3", "anti_entropy n1", "mkeys n4 m", "mget n4 m a",  # converge
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
