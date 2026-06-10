"""DS0 increment 22 — the rolling upgrade: ``deploy`` + version-compatibility consensus.

`deploy node version` sets a node's running software version; two nodes participate in the same
consensus quorum only if their versions are within `DistConfig.max_version_skew` (SPEC-7 §3.2 — the
"will this deploy break the cluster?" question). A rolling upgrade that stays inside the window
keeps quorum; an incompatible split with no compatible majority loses it. Versions are omitted from
canonical form until the first `deploy` (purely additive), and Tier-A ≡ Tier-B.
"""

import pytest

from verisim.dist import DistConfig, DistributedState, apply, parse_dist_action
from verisim.dist.action import ADMIN_OPS, DistParseError
from verisim.dist.delta import VersionSet
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(skew: int = 1, n: int = 4) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="deploy", nodes=nodes, objects=("x",), values=("a", "b", "c", "d"),
                      replication_factor=n, consistency_model="quorum", max_version_skew=skew)


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_deploy() -> None:
    assert parse_dist_action("deploy n0 2").args == ("n0", "2")
    assert {"deploy", "config_push"} == ADMIN_OPS  # config_push joined the admin family (incr 24)


@pytest.mark.parametrize("bad", ["deploy n0", "deploy n0 2 3", "deploy n0 x", "deploy n0 -1"])
def test_grammar_rejects_bad_deploy(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- deploy semantics ---------------------------------------------------------------------------

def test_deploy_sets_the_node_version() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("deploy n0 2"))
    assert (r.status, r.value) == ("deployed", "2")
    assert r.state.versions == {"n0": 2}


def test_deploy_unknown_node_is_rejected() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("deploy ghost 1"))
    assert r.status == "unknown_node"


def test_deploy_back_to_base_clears_the_version() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["deploy n0 2", "deploy n0 0"])
    assert s.versions == {}  # base version leaves no residue


# --- the version-compatibility quorum -----------------------------------------------------------

def test_safe_rolling_upgrade_keeps_quorum() -> None:
    # Roll every node v0 -> v1 one at a time (spread <= 1, inside the skew-1 window); a propose
    # commits after each bump — the upgrade never breaks the cluster.
    config = _config(skew=1, n=4)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["elect n0"])
    for node in config.nodes:
        s = oracle.step(s, parse_dist_action(f"deploy {node} 1")).state
        r = oracle.step(s, parse_dist_action("propose n0 x a"))
        assert r.status == "ok", f"after deploying {node}"
        s = r.state


def test_incompatible_split_with_no_majority_loses_quorum() -> None:
    # 4 nodes, 2 at v2 and 2 at v0 (spread 2 > skew 1): no compatible cohort is a majority, so the
    # leader's propose is no_quorum — the deploy broke the cluster.
    config = _config(skew=1, n=4)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "deploy n0 2", "deploy n1 2"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("propose n0 x a")).status \
        == "no_quorum"


def test_same_split_within_a_wider_window_commits() -> None:
    # The same v2/v0 over-spread, but max_version_skew = 2 makes the cohorts compatible -> commits.
    config = _config(skew=2, n=4)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "deploy n0 2", "deploy n1 2"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("propose n0 x a")).status == "ok"


def test_incompatible_node_cannot_be_elected_into_a_quorum() -> None:
    # A v2 candidate among v0 peers (skew 1) collects votes only from v2-compatible nodes — here
    # just itself (1 of 4) -> no_quorum; a v0 candidate still has its compatible majority.
    config = _config(skew=1, n=4)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "deploy n0 2"])
    # n0 (v2) re-electing: compatible voters = {n0} of 4 -> no_quorum
    assert ReferenceDistOracle(config).step(s, parse_dist_action("elect n0")).status == "no_quorum"
    # n1 (v0): compatible voters = {n1,n2,n3} of 4 -> elected
    assert ReferenceDistOracle(config).step(s, parse_dist_action("elect n1")).status == "elected"


def test_data_plane_is_version_agnostic() -> None:
    # A plain put/get is best-effort and ignores versions: a v2 node still serves its KV replica.
    config = _config(skew=1, n=4)
    s = _run(ReferenceDistOracle(config), config, ["deploy n0 2", "put n0 x b"])
    assert s.replicas[("x", "n0")].value == "b"  # the data plane does not gate on version


# --- delta + serialization ----------------------------------------------------------------------

def test_version_set_applies_and_round_trips() -> None:
    config = _config()
    s = apply(DistributedState.initial(config), [VersionSet("n0", 3)])
    assert s.versions == {"n0": 3}
    assert from_canonical(to_canonical(s)) == s


def test_canonical_form_omits_versions_until_first_deploy() -> None:
    config = _config()
    s = DistributedState.initial(config)
    assert "versions" not in to_canonical(s)  # all-base cluster: purely additive
    deployed = ReferenceDistOracle(config).step(s, parse_dist_action("deploy n0 2")).state
    assert to_canonical(deployed)["versions"] == {"n0": 2}
    # the all-base hash is unchanged by the (omitted) version field
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))


def test_config_hash_unchanged_at_default_skew() -> None:
    # max_version_skew == 1 (the default) is omitted from the config hash, so a pre-increment-22
    # config hashes identically; a non-default window changes it.
    base = DistConfig(name="c", nodes=("n0", "n1", "n2"))
    assert base.max_version_skew == 1
    assert "max_version_skew" not in base.to_dict()
    wide = DistConfig(name="c", nodes=("n0", "n1", "n2"), max_version_skew=2)
    assert wide.config_hash() != base.config_hash()


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_deploy_trajectory() -> None:
    config = _config(skew=1, n=4)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "elect n0", "propose n0 x a",
        "deploy n0 1", "propose n0 x b",        # spread 1: still commits
        "deploy n0 2", "deploy n1 2",           # spread 2: incompatible split
        "propose n0 x c",                        # no_quorum (the deploy broke it)
        "deploy n2 2", "deploy n3 2",            # finish the upgrade: all v2 -> compatible again
        "propose n0 x d",                        # commits
        "deploy n0 0", "deploy n1 0", "deploy n2 0", "deploy n3 0",  # roll back to base
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
