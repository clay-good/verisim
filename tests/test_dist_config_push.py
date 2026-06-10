"""DS0 increment 24 — the config push: ``config_push`` + leader-committed cluster config.

`config_push node key val` is the config-management admin op SPEC-7 §3.2 names ("will this config
push break the cluster?"). Unlike `deploy` (a node-local version label gating consensus
compatibility), it is a **leader-committed, majority-replicated** cluster setting — a Raft-style
config entry — so it shares the leader-fence + majority-reachability rule of `propose`/`append`. A
push under partition commits on the majority and leaves the minority with stale config (config
divergence). Config is omitted from canonical form until the first push (purely additive), and
Tier-A ≡ Tier-B.
"""

import pytest

from verisim.dist import DistConfig, DistributedState, apply, parse_dist_action
from verisim.dist.action import ADMIN_OPS, DistParseError
from verisim.dist.delta import ConfigSet
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle
from verisim.distoracle.tiers import TieredOracle


def _config(n: int = 5) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="config", nodes=nodes, objects=("x",),
                      values=("a", "b", "on", "off", "v2"),
                      replication_factor=n, consistency_model="quorum")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_config_push() -> None:
    assert parse_dist_action("config_push n0 feature on").args == ("n0", "feature", "on")
    assert "config_push" in ADMIN_OPS


@pytest.mark.parametrize("bad", ["config_push n0", "config_push n0 k", "config_push n0 k v extra"])
def test_grammar_rejects_bad_config_push(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- the leader fence ---------------------------------------------------------------------------

def test_push_with_no_leader_is_rejected() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("config_push n0 feature on"))
    assert (r.status, r.value) == ("not_leader", "")
    assert r.state.config == {}  # nothing pushed


def test_push_by_a_non_leader_is_rejected() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("config_push n1 feature on"))
    assert r.status == "not_leader" and r.value == "n0"  # carries the current leader, diagnostic
    assert r.state.config == {}


def test_crashed_leader_push_is_unavailable() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "crash n0"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("config_push n0 feature on"))
    assert r.status == "unavailable"


# --- commit semantics ---------------------------------------------------------------------------

def test_leader_push_commits_and_reaches_every_voting_member() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "config_push n0 feature on"])
    assert s.last_result == ("committed", "on")
    assert all(s.config[(nd, "feature")] == "on" for nd in config.nodes)


def test_re_push_overwrites_the_value() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config,
             ["elect n0", "config_push n0 feature on", "config_push n0 feature off"])
    assert all(s.config[(nd, "feature")] == "off" for nd in config.nodes)


# --- the partition: "will this config push break the cluster?" ----------------------------------

def test_minority_stranded_leader_cannot_commit_and_changes_nothing() -> None:
    # n0 leads, then is partitioned into the 2-of-5 minority {n0,n1}: its push reaches no majority,
    # so no_quorum and not a single node's config changes (all-or-nothing at commit).
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "partition n0 n1 | n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("config_push n0 feature on"))
    assert r.status == "no_quorum"
    assert r.state.config == {}  # nothing installed on the minority side


def test_majority_push_commits_but_minority_keeps_stale_config() -> None:
    # n0 leads on the 3-of-5 majority {n0,n1,n2}; the push commits there but never reaches the
    # partitioned minority {n3,n4} — config divergence, the broken-cluster outcome.
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "partition n0 n1 n2 | n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("config_push n0 feature on"))
    assert r.status == "committed"
    assert all(r.state.config.get((nd, "feature")) == "on" for nd in ("n0", "n1", "n2"))
    assert all((nd, "feature") not in r.state.config for nd in ("n3", "n4"))  # stale (absent)


def test_re_push_after_heal_converges_every_node() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config,
             ["elect n0", "partition n0 n1 n2 | n3 n4", "config_push n0 feature on",
              "heal", "config_push n0 feature on"])
    assert all(s.config.get((nd, "feature")) == "on" for nd in config.nodes)


def test_config_is_distinct_from_deploy_and_does_not_gate_consensus() -> None:
    # config_push installs a config value but, unlike deploy's version, does not affect quorum
    # compatibility: a propose after a config push still commits.
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "config_push n0 feature on"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("propose n0 x a"))
    assert r.status == "ok"
    assert s.versions == {}  # config_push touches no version label


# --- delta + serialization ----------------------------------------------------------------------

def test_config_set_applies_and_round_trips() -> None:
    config = _config()
    s = apply(DistributedState.initial(config), [ConfigSet("n0", "feature", "on")])
    assert s.config[("n0", "feature")] == "on"
    assert from_canonical(to_canonical(s)) == s


def test_canonical_form_omits_config_until_first_push() -> None:
    config = _config()
    s = DistributedState.initial(config)
    assert "config" not in to_canonical(s)  # never-pushed cluster: purely additive
    pushed = _run(ReferenceDistOracle(config), config, ["elect n0", "config_push n0 feature on"])
    assert to_canonical(pushed)["config"][0] == {"node": "n0", "key": "feature", "value": "on"}
    # the never-pushed hash is unchanged by the (omitted) config field
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))


def test_config_in_cluster_view() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "config_push n0 feature on"])
    assert "'config'" in cluster_view(s) and "feature" in cluster_view(s)


# --- the cheap metamorphic tier -----------------------------------------------------------------

def test_metamorphic_tier_refutes_config_on_an_unknown_node() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["elect n0"])
    bogus = apply(s, [ConfigSet("ghost", "feature", "on")])  # a phantom node
    verdict = TieredOracle(config).check(
        "metamorphic", s, parse_dist_action("config_push n0 feature on"), bogus
    )
    assert verdict.refuted and "unknown node" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_config_push_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "elect n0", "config_push n0 feature on",   # commits on all
        "partition n0 n1 | n2 n3 n4",              # strand n0 in the minority
        "config_push n0 feature off",              # no_quorum (nothing changes)
        "heal", "elect n2",                        # n2 leads
        "partition n2 n3 n4 | n0 n1",              # n2 on the majority, n0/n1 minority
        "config_push n2 feature v2",               # commits on majority, minority stale
        "heal", "config_push n2 feature v2",       # re-push converges all
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
