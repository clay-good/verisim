"""DS0 increment 27 — the atomic counter: ``incr`` + the lost-update problem under eventual.

`incr node key` is the first **read-modify-write** client op (SPEC-7 §3.2) — `put`/`cas`/`delete`
are blind or compare writes. It reads the coordinator's local counter (a non-numeric/absent value is
`0`) and writes `count + 1`, reusing the `put` replication path. Sequentially it is correct, but
under a partition the consistency model decides whether concurrent increments survive: `eventual`
**silently loses** a concurrent increment (last-writer-wins keeps one of two same-version writes),
where `quorum` makes the minority unavailable (no silent loss) and `linearizable` rejects under any
partition. Tier-A ≡ Tier-B, and the op is purely additive (the counter is a digit-valued replica).
"""

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import CLIENT_OPS, DistParseError
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(n: int = 5, model: str = "eventual") -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="incr", nodes=nodes, objects=("c",), values=("a",),
                      replication_factor=n, consistency_model=model)


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_incr() -> None:
    assert parse_dist_action("incr n0 c").args == ("n0", "c")
    assert "incr" in CLIENT_OPS


@pytest.mark.parametrize("bad", ["incr n0", "incr n0 c v"])
def test_grammar_rejects_bad_incr(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- sequential correctness ---------------------------------------------------------------------

def test_incr_counts_from_zero() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    r = oracle.step(DistributedState.initial(config), parse_dist_action("incr n0 c"))
    assert (r.status, r.value) == ("ok", "1")  # a fresh (nil) counter increments to 1


def test_incr_k_times_counts_to_k() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["incr n0 c", "incr n0 c", "incr n0 c"])
    assert oracle.step(s, parse_dist_action("get n0 c")).value == "3"


@pytest.mark.parametrize("model", ["eventual", "quorum", "linearizable"])
def test_sequential_incr_is_correct_under_every_model(model: str) -> None:
    config = _config(n=3, model=model)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["incr n0 c", "incr n0 c"])
    assert oracle.step(s, parse_dist_action("get n0 c")).value == "2"


# --- the read-modify-write CAP tradeoff under partition -----------------------------------------

def test_eventual_loses_a_concurrent_increment() -> None:
    # Two incrs on opposite sides of a partition are both acknowledged, but the count converges to 2
    # (not 3) — last-writer-wins keeps only one of the two same-version writes: a lost update.
    config = _config(n=5, model="eventual")
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["incr n0 c", "advance 5", "partition n0 n1 n2 | n3 n4"])
    a = oracle.step(s, parse_dist_action("incr n0 c"))
    s = a.state
    b = oracle.step(s, parse_dist_action("incr n3 c"))
    s = b.state
    assert (a.status, b.status) == ("ok", "ok")  # both acknowledged
    s = _run_from(oracle, s, ["heal", "advance 5", "anti_entropy n0"])
    assert oracle.step(s, parse_dist_action("get n0 c")).value == "2"  # expected 3 — one is lost


def test_quorum_minority_incr_is_unavailable_no_silent_loss() -> None:
    config = _config(n=5, model="quorum")
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["incr n0 c", "advance 5", "partition n0 n1 n2 | n3 n4"])
    maj = oracle.step(s, parse_dist_action("incr n0 c"))
    minr = oracle.step(maj.state, parse_dist_action("incr n3 c"))
    assert maj.status == "ok" and minr.status == "unavailable"  # only the accepted incr counts
    assert oracle.step(maj.state, parse_dist_action("get n0 c")).value == "2"


def test_linearizable_incr_rejected_under_any_partition() -> None:
    config = _config(n=5, model="linearizable")
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["incr n0 c", "partition n0 | n1 n2 n3 n4"])
    assert oracle.step(s, parse_dist_action("incr n0 c")).status == "unavailable"


# --- the metamorphic tier admits counter values -------------------------------------------------

def test_metamorphic_tier_admits_a_counter_value() -> None:
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["incr n0 c"])
    act = parse_dist_action("incr n0 c")
    predicted = oracle.step(s, act).state  # counter at "2" — a digit value, legal in any config
    verdict = TieredOracle(config).check("metamorphic", s, act, predicted)
    assert not verdict.refuted


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_an_incr_trajectory() -> None:
    config = _config(n=5, model="eventual")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "incr n0 c", "incr n0 c", "advance 5",          # count 2 everywhere
        "partition n0 n1 n2 | n3 n4",
        "incr n0 c", "incr n3 c",                        # both ack; one will be lost
        "heal", "advance 5", "anti_entropy n0", "get n0 c",  # converges to 3 (one increment lost)
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state


def _run_from(
    oracle: ReferenceDistOracle, s: DistributedState, cmds: list[str]
) -> DistributedState:
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s
