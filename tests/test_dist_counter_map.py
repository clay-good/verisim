"""DS0 increment 35 — the nested CRDT counter-map: ``cminc``/``cmget``/``cmdel``/``cmkeys``.

A *state-based* nested CRDT (SPEC-7 §3.2) — a CRDT whose **values are themselves CRDTs**, the
recursive form of the thesis. It composes the OR-Set (field presence, add-wins + observed-remove)
with the G-counter (each field's value, merging by per-owner **max**, loss-free). ``cminc n m f``
makes the field present *and* increments its counter; ``cmget`` reads the field's total; ``cmdel``
observed-removes the field; ``cmkeys`` enumerates. Both layers' guarantees hold at once: a field
survives a concurrent ``cmdel`` (add-wins) *and* concurrent ``cminc``s to the same field are summed
loss-free (where the OR-Map's LWW value would drop one). Tier-A ≡ Tier-B, purely additive over
increment 34 (three omitted-when-empty ``cmap_*`` maps + three edits).
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
    return DistConfig(name="cm", nodes=nodes, objects=("m",), values=("a",),
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

def test_grammar_parses_counter_map_ops() -> None:
    assert parse_dist_action("cminc n0 m f").args == ("n0", "m", "f")
    assert parse_dist_action("cmget n0 m f").args == ("n0", "m", "f")
    assert parse_dist_action("cmdel n0 m f").args == ("n0", "m", "f")
    assert parse_dist_action("cmkeys n0 m").args == ("n0", "m")
    assert {"cminc", "cmget", "cmdel", "cmkeys"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["cminc n0 m", "cmget n0 m", "cmdel n0 m f x", "cmkeys n0 m f"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local counter-map operations ----------------------------------------------------------

def test_cminc_builds_totals_and_keys() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cminc n0 m visits", "cminc n0 m visits", "cminc n0 m clicks"])
    assert oracle.step(s, parse_dist_action("cmget n0 m visits")).value == "2"
    assert oracle.step(s, parse_dist_action("cmget n0 m clicks")).value == "1"
    assert _keys(oracle.step(s, parse_dist_action("cmkeys n0 m")).value) == {"visits", "clicks"}


def test_cmdel_removes_field() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cminc n0 m a", "cminc n0 m b", "cmdel n0 m a"])
    assert _keys(oracle.step(s, parse_dist_action("cmkeys n0 m")).value) == {"b"}
    assert oracle.step(s, parse_dist_action("cmget n0 m a")).value == ""  # absent after remove


def test_cminc_is_always_available_even_partitioned_alone() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("cminc n0 m k"))
    assert (r.status, r.value) == ("ok", "1")  # reports the field's new total


def test_crashed_node_cminc_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("cminc n0 m k")).status == \
        "unavailable"


# --- the recursion: loss-free counter values + add-wins presence --------------------------------

def test_concurrent_increments_are_loss_free() -> None:
    # The counter recursion: concurrent cminc to the SAME field are summed (where the OR-Map's LWW
    # value would keep one). 'c' present cluster-wide; partition; n0 +2, n3 +1 -> total 3.
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["cminc n0 m c", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4", "partition n0 n1 n2 | n3 n4", "cminc n0 m c", "cminc n3 m c",
              "heal", "gossip n0 n3"])
    assert oracle.step(s, parse_dist_action("cmget n0 m c")).value == "3"  # 1 + 1 + 1, nothing lost


def test_add_wins_field_survives_concurrent_delete() -> None:
    # 'k' present cluster-wide; partition; n0 re-increments (fresh dot) while n3 removes the dot it
    # saw. After heal+gossip the field SURVIVES (add-wins) with its full count.
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["cminc n0 m k", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4", "partition n0 n1 n2 | n3 n4", "cminc n0 m k", "cmdel n3 m k",
              "heal", "gossip n0 n3"])
    assert "k" in _keys(oracle.step(s, parse_dist_action("cmkeys n0 m")).value)
    assert "k" in _keys(oracle.step(s, parse_dist_action("cmkeys n3 m")).value)
    assert oracle.step(s, parse_dist_action("cmget n0 m k")).value == "2"  # both increments counted


def test_anti_entropy_converges_fields_and_totals() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "cminc n0 m x", "cminc n3 m y", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert _keys(oracle.step(s, parse_dist_action(f"cmkeys {nd} m")).value) == {"x", "y"}
        assert oracle.step(s, parse_dist_action(f"cmget {nd} m x")).value == "1"
        assert oracle.step(s, parse_dist_action(f"cmget {nd} m y")).value == "1"


def test_composed_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 | n2", "cminc n0 m x", "cminc n2 m x", "heal",
              "gossip n0 n2", "gossip n0 n2"])
    assert oracle.step(s, parse_dist_action("cmget n0 m x")).value == "2"  # stable under re-merge


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_cmap_until_first_op() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert not any(k.startswith("cmap") for k in to_canonical(s))  # purely additive
    written = _run(ReferenceDistOracle(config), config, ["cminc n0 m f"])
    canon = to_canonical(written)
    assert canon["cmap_fields"][0] == {
        "map": "m", "holder": "n0", "field": "f", "owner": "n0", "seq": 1
    }
    assert canon["cmap_counts"][0] == {
        "map": "m", "field": "f", "holder": "n0", "owner": "n0", "count": 1
    }
    removed = _run(ReferenceDistOracle(config), config, ["cminc n0 m f", "cmdel n0 m f"])
    assert to_canonical(removed)["cmap_tombs"][0] == {
        "map": "m", "holder": "n0", "owner": "n0", "seq": 1
    }
    assert from_canonical(to_canonical(removed)) == removed
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_cmap_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["cminc n0 m f"])
    view = cluster_view(s)
    assert "'cmap_fields'" in view and "'cmap_counts'" in view


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_negative_count() -> None:
    from verisim.dist.delta import CMapCount, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cminc n0 m f"])
    bogus = apply(s, [CMapCount("m", "f", "n0", "n0", -1)])  # a negative counter sub-count
    act = parse_dist_action("cminc n0 m f")
    verdict = TieredOracle(config).check("metamorphic", s, act, bogus)
    assert verdict.refuted and "cmap" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_counter_map_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "cminc n0 m a", "cminc n0 m a", "cmkeys n0 m", "cmget n0 m a",
        "anti_entropy n1", "anti_entropy n2", "anti_entropy n3", "anti_entropy n4",
        "partition n0 n1 n2 | n3 n4",
        "cminc n0 m a", "cmdel n3 m a", "cminc n3 m b",   # concurrent: increment + delete + add
        "cmkeys n0 m", "cmkeys n3 m",                     # local views diverge
        "heal", "gossip n0 n3", "anti_entropy n1", "cmkeys n4 m", "cmget n4 m a",  # converge
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
