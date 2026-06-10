"""DS0 increment 16 — the Raft-subset consensus core: ``elect`` + ``propose``.

The third action family (SPEC-7 §3.2): leader election with a monotone term, and a leader-fenced
majority write. The two properties under test are the classic consensus-safety guarantees plain
``quorum`` writes lack:

  - **No split-brain leadership** — a leader is elected only by a partition side holding a strict
    majority of the live cluster, so two disjoint sides can never both elect (``elect_edits``).
  - **Leader completeness / term-fencing** — a leader deposed by a higher-term election cannot
    commit even after the heal, because the global leader has moved on (``propose_edits``).

Everything is pinned bit-exact and Tier-A ≡ Tier-B (the helpers are coordinator-level decisions
shared by both oracles), and the leader/term metadata is omitted from the canonical form until the
first election so every pre-increment-16 golden/hash is unchanged.
"""

import pytest

from verisim.dist import DistConfig, DistributedState, apply, parse_dist_action
from verisim.dist.action import CONSENSUS_OPS, PROTOCOL_OPS, DistParseError
from verisim.dist.delta import CommitIndexSet, LeaseSet, LogSet, MemberSet, ProtocolStep
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.dist.state import LogEntry
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle
from verisim.distoracle.tiers import TieredOracle

CONFIG = DistConfig(name="consensus", nodes=("n0", "n1", "n2"), objects=("x", "y"))


def _ref() -> ReferenceDistOracle:
    return ReferenceDistOracle(CONFIG)


def _run(oracle: ReferenceDistOracle, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(CONFIG)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_elect_and_propose() -> None:
    assert parse_dist_action("elect n0").name == "elect"
    p = parse_dist_action("propose n0 x b")
    assert p.name == "propose" and p.args == ("n0", "x", "b")
    assert parse_dist_action("step_down n0").name == "step_down"
    assert parse_dist_action("lease n0 5").args == ("n0", "5")
    assert parse_dist_action("lread n0 x").args == ("n0", "x")
    assert parse_dist_action("append n0 x b").args == ("n0", "x", "b")
    assert parse_dist_action("add_replica n3").args == ("n3",)
    assert parse_dist_action("remove_replica n1").args == ("n1",)
    assert sorted(CONSENSUS_OPS) == ["add_replica", "append", "elect", "lease", "lread", "propose",
                                     "remove_replica", "step_down"]
    assert CONSENSUS_OPS <= PROTOCOL_OPS


@pytest.mark.parametrize("bad", ["elect", "elect n0 n1", "propose n0 x", "propose n0 x b c",
                                 "step_down", "step_down n0 n1", "lease n0", "lease n0 0",
                                 "lease n0 -1", "lread n0", "lread n0 x y", "append n0 x",
                                 "append n0 x b c", "add_replica", "add_replica n0 n1",
                                 "remove_replica", "remove_replica n0 n1"])
def test_grammar_rejects_bad_arity(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- elect: the majority rule + monotone term ---------------------------------------------------

def test_elect_with_full_connectivity_succeeds_and_bumps_term() -> None:
    oracle = _ref()
    r = oracle.step(DistributedState.initial(CONFIG), parse_dist_action("elect n0"))
    assert (r.status, r.value) == ("elected", "1")
    assert r.state.leader == "n0" and r.state.term == 1


def test_elect_in_minority_partition_is_no_quorum() -> None:
    # n0 alone (1 of 3) cannot win an election: no live majority -> no leader (no split-brain).
    s = _run(_ref(), ["partition n0 | n1 n2"])
    r = _ref().step(s, parse_dist_action("elect n0"))
    assert (r.status, r.value) == ("no_quorum", "")
    assert r.state.leader is None and r.state.term == 0


def test_elect_on_majority_side_of_partition_succeeds() -> None:
    s = _run(_ref(), ["partition n0 | n1 n2"])
    r = _ref().step(s, parse_dist_action("elect n1"))  # {n1,n2} is 2 of 3 = a strict majority
    assert r.status == "elected" and r.state.leader == "n1"


def test_elect_crashed_candidate_is_unavailable() -> None:
    s = _run(_ref(), ["crash n0"])
    r = _ref().step(s, parse_dist_action("elect n0"))
    assert (r.status, r.value) == ("unavailable", "")


def test_elect_requires_live_majority_not_just_group_size() -> None:
    # A 2-node group whose other member is crashed has only one live voter (1 of 3) -> no quorum.
    s = _run(_ref(), ["partition n0 n1 | n2", "crash n1"])
    r = _ref().step(s, parse_dist_action("elect n0"))
    assert r.status == "no_quorum"


def test_no_split_brain_two_sides_cannot_both_elect() -> None:
    # Split 3 nodes into a 1 | 2; the singleton can never elect, only the majority side can —
    # so at most one leader exists across the whole cluster.
    s = _run(_ref(), ["partition n0 | n1 n2"])
    minority = _ref().step(s, parse_dist_action("elect n0"))
    majority = _ref().step(s, parse_dist_action("elect n1"))
    assert minority.status == "no_quorum"
    assert majority.status == "elected"


def test_term_is_monotone_across_reelection() -> None:
    s = _run(_ref(), ["elect n0", "elect n1"])  # two elections in a row
    assert s.term == 2 and s.leader == "n1"


# --- propose: leader-fenced majority write ------------------------------------------------------

def test_propose_by_leader_commits_a_majority_write() -> None:
    s = _run(_ref(), ["elect n0", "propose n0 x b"])
    assert s.last_result == ("ok", "b")
    # full connectivity -> the leader reaches every replica synchronously.
    assert all(s.replicas[("x", n)].value == "b" for n in CONFIG.nodes)
    assert all(s.replicas[("x", n)].version == 1 for n in CONFIG.nodes)


def test_propose_by_non_leader_is_fenced() -> None:
    s = _run(_ref(), ["elect n0"])
    r = _ref().step(s, parse_dist_action("propose n1 x c"))  # n1 is not the leader
    assert (r.status, r.value) == ("not_leader", "n0")
    assert r.state.replicas == s.replicas  # no write happened


def test_propose_before_any_election_is_not_leader() -> None:
    r = _ref().step(DistributedState.initial(CONFIG), parse_dist_action("propose n0 x b"))
    assert (r.status, r.value) == ("not_leader", "")  # no leader yet


def test_deposed_leader_is_fenced_even_after_heal() -> None:
    # The headline safety property: n0 leads, is partitioned into the minority, the majority side
    # elects n1 (higher term); after heal, n0's propose is rejected because the leader moved on.
    s = _run(_ref(), [
        "elect n0",
        "partition n0 | n1 n2",
        "elect n1",          # majority side elects a new leader (term 2)
        "heal",
        "propose n0 x d",    # the old leader tries to write after the heal
    ])
    assert s.last_result == ("not_leader", "n1")
    # and the legitimate new leader commits.
    s2 = _ref().step(s, parse_dist_action("propose n1 x d"))
    assert s2.status == "ok"


def test_propose_without_a_reachable_majority_is_no_quorum() -> None:
    # A leader stranded in the minority cannot commit even though it is still `leader`.
    s = _run(_ref(), ["elect n0", "partition n0 | n1 n2"])
    r = _ref().step(s, parse_dist_action("propose n0 x b"))
    assert r.status == "no_quorum"


def test_propose_in_majority_partition_writes_majority_async_to_minority() -> None:
    # Leader on the majority side commits to the reachable majority and queues async catch-up to the
    # stranded minority replica (delivered on heal+advance) — like a quorum put.
    s = _run(_ref(), ["elect n1", "partition n0 | n1 n2", "propose n1 x b"])
    assert s.last_result == ("ok", "b")
    assert s.replicas[("x", "n1")].value == "b" and s.replicas[("x", "n2")].value == "b"
    assert s.replicas[("x", "n0")].value == "nil"  # minority still stale
    assert any(m.dst == "n0" and m.value == "b" for m in s.inflight.values())  # catch-up queued
    healed = _run(ReferenceDistOracle(CONFIG), [
        "elect n1", "partition n0 | n1 n2", "propose n1 x b", "heal", "advance 5",
    ])
    assert healed.replicas[("x", "n0")].value == "b"  # converges after heal


# --- step_down: voluntary relinquishment (DS0 increment 17) -------------------------------------

def test_step_down_by_leader_leaves_cluster_leaderless_at_same_term() -> None:
    s = _run(_ref(), ["elect n0"])  # term 1, leader n0
    r = _ref().step(s, parse_dist_action("step_down n0"))
    assert (r.status, r.value) == ("stepped_down", "1")
    assert r.state.leader is None and r.state.term == 1  # leaderless, term held


def test_propose_after_step_down_is_not_leader_no_commit_window() -> None:
    # The leaderless gap admits no consensus write — even from the node that just stepped down.
    s = _run(_ref(), ["elect n0", "step_down n0"])
    r = _ref().step(s, parse_dist_action("propose n0 x b"))
    assert (r.status, r.value) == ("not_leader", "")  # no leader to fence the write through


def test_step_down_by_non_leader_is_rejected() -> None:
    s = _run(_ref(), ["elect n0"])
    r = _ref().step(s, parse_dist_action("step_down n1"))  # n1 is not the leader
    assert (r.status, r.value) == ("not_leader", "n0")
    assert r.state.leader == "n0" and r.state.term == 1  # unchanged


def test_second_step_down_is_idempotent_no_op_reject() -> None:
    s = _run(_ref(), ["elect n0", "step_down n0"])  # already leaderless
    r = _ref().step(s, parse_dist_action("step_down n0"))
    assert r.status == "not_leader"  # nothing to relinquish


def test_step_down_crashed_node_is_unavailable() -> None:
    s = _run(_ref(), ["elect n0", "crash n0"])
    r = _ref().step(s, parse_dist_action("step_down n0"))
    assert r.status == "unavailable"


def test_minority_leader_can_step_down_where_propose_is_no_quorum() -> None:
    # Relinquishing needs no quorum (it reads only the node's own leadership); committing does.
    s = _run(_ref(), ["elect n0", "partition n0 | n1 n2"])
    assert _ref().step(s, parse_dist_action("propose n0 x b")).status == "no_quorum"
    r = _ref().step(s, parse_dist_action("step_down n0"))
    assert (r.status, r.value) == ("stepped_down", "1")
    assert r.state.leader is None


def test_handoff_successor_election_bumps_term_strictly() -> None:
    s = _run(_ref(), ["elect n0", "step_down n0", "elect n1"])
    assert s.term == 2 and s.leader == "n1"  # the successor lands at a strictly higher term


def test_step_down_changes_no_replica() -> None:
    s = _run(_ref(), ["elect n0", "propose n0 x b"])
    r = _ref().step(s, parse_dist_action("step_down n0"))
    assert r.state.replicas == s.replicas  # leadership is metadata, not data


def test_step_down_protocol_step_round_trips() -> None:
    s = _run(_ref(), ["elect n0"])
    stepped = apply(s, [ProtocolStep("step_down", s.term, None)])
    assert stepped.leader is None and stepped.term == 1
    assert from_canonical(to_canonical(stepped)) == stepped


# --- lease / lread: the leader lease (DS0 increment 18) -----------------------------------------

def test_lease_by_leader_sets_the_deadline() -> None:
    s = _run(_ref(), ["elect n0", "lease n0 5"])  # clock 0 + 5
    r = _ref().step(_run(_ref(), ["elect n0"]), parse_dist_action("lease n0 5"))
    assert (r.status, r.value) == ("leased", "5")
    assert s.lease_until == 5 and s.replicas == DistributedState.initial(CONFIG).replicas


def test_lease_by_non_leader_is_rejected() -> None:
    s = _run(_ref(), ["elect n0"])
    r = _ref().step(s, parse_dist_action("lease n1 5"))
    assert (r.status, r.value) == ("not_leader", "n0")
    assert r.state.lease_until == 0


def test_lread_under_a_live_lease_serves_the_local_value() -> None:
    s = _run(_ref(), ["elect n0", "propose n0 x b", "lease n0 5"])
    r = _ref().step(s, parse_dist_action("lread n0 x"))
    assert (r.status, r.value) == ("ok", "b")  # local read, no quorum contacted
    assert r.state.replicas == s.replicas  # a read mutates nothing


def test_lread_after_expiry_is_rejected() -> None:
    s = _run(_ref(), ["elect n0", "propose n0 x b", "lease n0 5", "advance 6"])  # clock 6 > 5
    r = _ref().step(s, parse_dist_action("lread n0 x"))
    assert r.status == "lease_expired"


def test_lread_by_non_leader_is_not_leader() -> None:
    s = _run(_ref(), ["elect n0", "lease n0 5"])
    r = _ref().step(s, parse_dist_action("lread n1 x"))
    assert (r.status, r.value) == ("not_leader", "n0")


def test_minority_leader_can_lread_where_propose_is_no_quorum() -> None:
    # The lease's payoff: a leader stranded in the minority serves a local read with no quorum,
    # where its propose (which needs a majority) is no_quorum.
    s = _run(_ref(), ["elect n0", "propose n0 x b", "lease n0 5", "partition n0 | n1 n2"])
    assert _ref().step(s, parse_dist_action("propose n0 x c")).status == "no_quorum"
    r = _ref().step(s, parse_dist_action("lread n0 x"))
    assert (r.status, r.value) == ("ok", "b")  # local linearizable read survives the partition


def test_elect_is_blocked_while_a_lease_is_live() -> None:
    # Leader-lease safety: a successor must wait out the incumbent's unexpired lease.
    s = _run(_ref(), ["elect n0", "lease n0 5"])
    r = _ref().step(s, parse_dist_action("elect n1"))
    assert (r.status, r.value) == ("lease_held", "5")
    assert r.state.leader == "n0" and r.state.term == 1  # leadership unchanged


def test_elect_succeeds_once_the_lease_expires_and_clears_it() -> None:
    s = _run(_ref(), ["elect n0", "lease n0 5", "advance 6"])  # clock 6 > 5
    r = _ref().step(s, parse_dist_action("elect n1"))
    assert r.status == "elected" and r.state.leader == "n1" and r.state.term == 2
    assert r.state.lease_until == 0  # a new term starts with no lease


def test_step_down_releases_the_lease_for_an_immediate_handoff() -> None:
    # The fast-handoff path: a voluntary step_down releases the lease, so a successor elects with
    # no wait (vs a crashed leader, whose lease the cluster must outlast).
    s = _run(_ref(), ["elect n0", "lease n0 5", "step_down n0"])
    assert s.lease_until == 0 and s.leader is None
    r = _ref().step(s, parse_dist_action("elect n1"))  # clock still 0, but lease was released
    assert r.status == "elected" and r.state.leader == "n1"


def test_deposed_leader_cannot_lread_off_a_stale_lease() -> None:
    # After the lease expires, n1 is elected (clearing the lease); the old leader n0 is now fenced,
    # so even if it imagines a lease it gets not_leader, never a stale read.
    s = _run(_ref(), ["elect n0", "lease n0 5", "advance 6", "elect n1"])
    r = _ref().step(s, parse_dist_action("lread n0 x"))
    assert r.status == "not_leader"


def test_lease_set_applies_and_round_trips() -> None:
    s = _run(_ref(), ["elect n0"])
    leased = apply(s, [LeaseSet(7)])
    assert leased.lease_until == 7
    assert from_canonical(to_canonical(leased)) == leased


def test_canonical_form_omits_lease_until_until_first_lease() -> None:
    s = _run(_ref(), ["elect n0"])
    assert "lease_until" not in to_canonical(s)  # no lease yet
    leased = _ref().step(s, parse_dist_action("lease n0 5")).state
    assert to_canonical(leased)["lease_until"] == 5  # appears only after a lease


# --- append: the Raft replicated log (DS0 increment 19) -----------------------------------------

def test_append_by_leader_commits_on_majority_and_grows_the_log() -> None:
    s = _run(_ref(), ["elect n0", "append n0 x a"])
    assert s.last_result == ("appended", "0")
    assert s.commit_index == 1  # the entry committed (a majority holds it)
    assert all(s.logs[n] == (LogEntry(1, 0, "x", "a"),) for n in CONFIG.nodes)  # log-matching
    assert all(s.replicas[("x", n)].value == "a" for n in CONFIG.nodes)  # applied to the KV


def test_append_by_non_leader_is_not_leader() -> None:
    s = _run(_ref(), ["elect n0"])
    r = _ref().step(s, parse_dist_action("append n1 x a"))
    assert (r.status, r.value) == ("not_leader", "n0")


def test_minority_leader_append_is_uncommitted_but_retained_on_its_log() -> None:
    # A leader stranded in the minority appends to its own log, but does not commit (no majority),
    # so it is not applied to the KV and may later be overwritten.
    s = _run(_ref(), ["elect n0", "append n0 x a", "partition n0 | n1 n2", "append n0 x b"])
    assert s.last_result == ("uncommitted", "1")
    assert s.commit_index == 1  # unchanged: b did not commit
    assert s.logs["n0"][1] == LogEntry(1, 1, "x", "b")  # b is on n0's log...
    assert s.replicas[("x", "n0")].value == "a"  # ...but NOT applied to the KV (still committed a)


def test_commit_index_is_monotone_and_log_matching_holds() -> None:
    s = _run(_ref(), ["elect n0", "append n0 x a", "append n0 x b", "append n0 x c"])
    assert s.commit_index == 3
    # all nodes share an identical committed log (the log-matching property)
    expected = (LogEntry(1, 0, "x", "a"), LogEntry(1, 1, "x", "b"), LogEntry(1, 2, "x", "c"))
    assert all(s.logs[n] == expected for n in CONFIG.nodes)
    # the KV folds the committed log: x's version is the count of committed writes (3), value last
    assert all((s.replicas[("x", n)].version, s.replicas[("x", n)].value) == (3, "c")
               for n in CONFIG.nodes)


def test_log_matching_reconciliation_overwrites_a_deposed_leaders_uncommitted_tail() -> None:
    # The headline safety property: n0 appends an uncommitted entry in the minority, is deposed by
    # n1 (higher term) which commits a conflicting entry at the same index; after heal, n0's
    # uncommitted entry is overwritten and every live log is identical.
    s = _run(_ref(), [
        "elect n0", "append n0 x a",          # committed a@0
        "partition n0 | n1 n2", "append n0 x b",  # uncommitted b@1 on n0
        "elect n1", "append n1 x c",          # committed c@1 (term 2) on the majority
        "heal", "append n1 x d",              # n0 reconciles: b@1 overwritten by c@1, then d@2
    ])
    # n0's uncommitted b is gone; index 1 holds the committed c (term 2); logs are identical
    assert all(e.value != "b" for e in s.logs["n0"])
    assert s.logs["n0"][1] == LogEntry(2, 1, "x", "c")
    assert s.logs["n0"] == s.logs["n1"] == s.logs["n2"]
    assert s.commit_index == 3
    assert s.replicas[("x", "n0")].value == "d"  # n0's KV converged to the committed value


def test_append_crashed_leader_is_unavailable() -> None:
    s = _run(_ref(), ["elect n0", "crash n0"])
    assert _ref().step(s, parse_dist_action("append n0 x a")).status == "unavailable"


def test_log_set_and_commit_index_apply_and_round_trip() -> None:
    s = DistributedState.initial(CONFIG)
    s2 = apply(s, [LogSet("n0", (LogEntry(1, 0, "x", "a"),)), CommitIndexSet(1)])
    assert s2.logs["n0"] == (LogEntry(1, 0, "x", "a"),) and s2.commit_index == 1
    assert from_canonical(to_canonical(s2)) == s2


def test_canonical_form_omits_log_and_commit_index_until_first_append() -> None:
    s = _run(_ref(), ["elect n0"])
    canon = to_canonical(s)
    assert "logs" not in canon and "commit_index" not in canon
    appended = _ref().step(s, parse_dist_action("append n0 x a")).state
    canon2 = to_canonical(appended)
    assert canon2["commit_index"] == 1 and len(canon2["logs"]) == len(CONFIG.nodes)


def test_metamorphic_refutes_backward_commit_index() -> None:
    tiers = TieredOracle(CONFIG)
    s = _run(_ref(), ["elect n0", "append n0 x a", "append n0 x b"])  # commit_index 2
    backward = apply(s, [CommitIndexSet(1)])  # un-commits an entry
    v = tiers.check("metamorphic", s, parse_dist_action("append n0 x c"), backward)
    assert v.refuted and "commit index went backward" in v.reason


# --- add_replica / remove_replica: membership change (DS0 increment 20) -------------------------

def test_remove_replica_shrinks_the_majority_threshold() -> None:
    # A lone leader is a minority of 3; after removing the other two, it is the sole member and a
    # majority of 1 — the quorum threshold tracked the voting set down.
    s = _run(_ref(), ["elect n0", "partition n0 | n1 n2"])
    assert _ref().step(s, parse_dist_action("propose n0 x a")).status == "no_quorum"  # 1 of 3
    s = _run(_ref(), ["elect n0", "partition n0 | n1 n2", "remove_replica n1", "remove_replica n2"])
    assert s.members == frozenset({"n0"})
    assert _ref().step(s, parse_dist_action("propose n0 x a")).status == "ok"  # 1 of 1


def test_add_replica_raises_the_threshold_and_round_trips_the_sentinel() -> None:
    # remove then add-back collapses to the empty "all vote" sentinel (clean canonical form).
    s = _run(_ref(), ["elect n0", "remove_replica n1"])
    assert s.members == frozenset({"n0", "n2"})
    s2 = _ref().step(s, parse_dist_action("add_replica n1"))
    assert s2.status == "added" and s2.state.members == frozenset()  # back to "all vote"


def test_remove_replica_rejections() -> None:
    base = _run(_ref(), ["elect n0"])
    # the active leader cannot be removed (step it down first)
    assert _ref().step(base, parse_dist_action("remove_replica n0")).status == "is_leader"
    # an unknown node / a non-member is a no-op reject
    assert _ref().step(base, parse_dist_action("remove_replica ghost")).status == "unknown_node"
    once = _run(_ref(), ["elect n0", "remove_replica n1"])
    assert _ref().step(once, parse_dist_action("remove_replica n1")).status == "not_member"


def test_membership_change_requires_a_leader() -> None:
    s = DistributedState.initial(CONFIG)  # no election yet
    assert _ref().step(s, parse_dist_action("add_replica n1")).status == "no_leader"
    assert _ref().step(s, parse_dist_action("remove_replica n1")).status == "no_leader"


def test_remove_replica_refuses_to_drop_the_last_member() -> None:
    # Craft a one-member voting set whose sole member is NOT the leader (so `is_leader` does not
    # fire first), and confirm the final member is protected.
    s = apply(DistributedState.initial(CONFIG),
              [MemberSet(frozenset({"n1"})), ProtocolStep("elect", 1, "n0")])
    assert _ref().step(s, parse_dist_action("remove_replica n1")).status == "last_member"


def test_elect_requires_membership_and_uses_the_member_majority() -> None:
    # After removing n2, the voting set is {n0, n1} (majority 2). n2 (a non-member) cannot be
    # elected, and a re-election needs a majority of the *members*, not the full cluster.
    s = _run(_ref(), ["elect n0", "remove_replica n2"])
    assert _ref().step(s, parse_dist_action("elect n2")).status == "not_member"
    # {n0, n1} both live → a 2-of-2 majority elects n1
    assert _ref().step(s, parse_dist_action("elect n1")).status == "elected"


def test_member_set_applies_and_round_trips() -> None:
    s = _run(_ref(), ["elect n0"])
    reduced = apply(s, [MemberSet(frozenset({"n0", "n1"}))])
    assert reduced.members == frozenset({"n0", "n1"})
    assert from_canonical(to_canonical(reduced)) == reduced


def test_canonical_form_omits_members_until_first_change() -> None:
    s = _run(_ref(), ["elect n0"])
    assert "members" not in to_canonical(s)  # the "all vote" sentinel is omitted
    reduced = _ref().step(s, parse_dist_action("remove_replica n1")).state
    assert sorted(to_canonical(reduced)["members"]) == ["n0", "n2"]


def test_metamorphic_refutes_unknown_voting_member() -> None:
    tiers = TieredOracle(CONFIG)
    s = _run(_ref(), ["elect n0"])
    bogus = apply(s, [MemberSet(frozenset({"n0", "ghost"}))])
    v = tiers.check("metamorphic", s, parse_dist_action("add_replica n1"), bogus)
    assert v.refuted and "unknown node" in v.reason


# --- delta + serialization ----------------------------------------------------------------------

def test_protocol_step_applies_and_round_trips() -> None:
    s = DistributedState.initial(CONFIG)
    s2 = apply(s, [ProtocolStep("elect", 3, "n2")])
    assert s2.term == 3 and s2.leader == "n2"
    assert from_canonical(to_canonical(s2)) == s2


def test_canonical_form_omits_leader_term_until_first_election() -> None:
    s = DistributedState.initial(CONFIG)
    canon = to_canonical(s)
    assert "term" not in canon and "leader" not in canon
    # an election makes the metadata appear (and only then).
    elected = _ref().step(s, parse_dist_action("elect n0")).state
    canon2 = to_canonical(elected)
    assert canon2["term"] == 1 and canon2["leader"] == "n0"


def test_pre_increment_hash_unchanged_by_the_new_fields() -> None:
    # A cluster that never runs consensus serializes to the exact pre-increment-16 form, so a
    # put/advance trajectory's content hash is identical to what it was before leader/term existed.
    s = _run(_ref(), ["put n0 x b", "advance 2", "put n1 y c", "advance 2"])
    # the metadata is at its boot defaults, so it contributes nothing to the canonical form.
    assert s.term == 0 and s.leader is None
    assert "term" not in to_canonical(s)
    # (the value of the hash itself is pinned by test_dist_goldens; here we assert the omission.)
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))


# --- the tiered oracle --------------------------------------------------------------------------

def test_metamorphic_refutes_backward_term_and_bogus_leader() -> None:
    tiers = TieredOracle(CONFIG)
    s = _run(_ref(), ["elect n0", "elect n1"])  # term 2, leader n1
    backward = apply(s, [ProtocolStep("elect", 1, "n0")])  # term went backward
    v = tiers.check("metamorphic", s, parse_dist_action("elect n0"), backward)
    assert v.refuted and "term went backward" in v.reason
    bogus = apply(s, [ProtocolStep("elect", 3, "ghost")])  # leader is not a cluster node
    v2 = tiers.check("metamorphic", s, parse_dist_action("elect n0"), bogus)
    assert v2.refuted and "not a cluster node" in v2.reason


def test_elect_propose_defer_to_bit_exact_in_cheap_tiers() -> None:
    tiers = TieredOracle(CONFIG)
    s = DistributedState.initial(CONFIG)
    # elect: a correct election passes every tier (its leader/term correctness is bit-exact's job).
    a_elect = parse_dist_action("elect n0")
    truth = _ref().step(s, a_elect).state
    assert not tiers.check("symbolic", s, a_elect, truth).refuted
    assert not tiers.check("cycle", s, a_elect, truth).refuted
    # propose: a correct majority write passes the cheap tiers (defers to bit-exact).
    s2 = _ref().step(s, a_elect).state
    a_prop = parse_dist_action("propose n0 x b")
    truth2 = _ref().step(s2, a_prop).state
    assert not tiers.check("symbolic", s2, a_prop, truth2).refuted
    assert not tiers.check("cycle", s2, a_prop, truth2).refuted
    assert not tiers.cheapest_refutation(s2, a_prop, truth2).refuted


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_consensus_trajectory() -> None:
    ref, sysb = _ref(), SystemDistOracle(CONFIG)
    sa = sb = DistributedState.initial(CONFIG)
    script = [
        "elect n0", "propose n0 x b", "advance 2",
        "propose n1 x c",                       # not_leader
        "partition n0 | n1 n2", "elect n1",     # depose n0 on the majority side
        "propose n0 x d",                       # no_quorum (n0 in minority)
        "heal", "advance 2",
        "propose n0 y a",                       # not_leader (fenced)
        "propose n1 y a", "advance 5",          # the new leader commits
        "step_down n1",                         # the leader voluntarily relinquishes -> leaderless
        "propose n1 y b",                       # not_leader (no leader to commit through)
        "elect n0", "propose n0 y b", "advance 5",  # a clean handoff: re-elect, then commit
        "lease n0 5", "lread n0 y",             # the leader takes a lease + a local read
        "elect n1",                             # lease_held (n0's lease is live)
        "partition n0 | n1 n2", "lread n0 y",   # minority leader still reads locally (no quorum)
        "heal", "advance 6", "lread n0 y",      # lease expired -> lease_expired
        "elect n1", "advance 2",                # lease expired, so the election now succeeds
        "append n1 x a",                        # replicated-log append, committed on the majority
        "partition n0 n1 | n2", "append n1 x b",  # committed on {n0,n1}; n2 stranded
        "elect n2",                             # n2 cannot elect (minority) -> stays n1's term
        "heal", "append n1 x c",                # n2 reconciles + backfills on the next append
        "remove_replica n2",                    # shrink the voting set: members {n0,n1}, majority 2
        "partition n0 | n1 n2", "append n1 x m",  # n1 reaches {n1} of members {n0,n1} -> no_quorum
        "heal", "append n1 x p",                # reachable again -> commits on the 2-member quorum
        "add_replica n2",                       # restore membership (back to the all-vote sentinel)
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
