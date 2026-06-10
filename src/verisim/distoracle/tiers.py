"""The tiered oracle -- SPEC-7's payload (§5, the heart of the distributed world; DS3 increment 2).

The structural novelty of the distributed world is that **bit-exact global truth is intractable**
(§1.1, §2.2), so the oracle is a *menu at four price points* and the consultation policy chooses
*which tier* to spend, not just *when* (the `π_w` axis generalized from "which sub-oracle" to "which
tier", §8.2):

    tier         cost  verifies                       refutes (in the DS0-incr-1 KV world)
    -----------  ----  -----------------------------  -------------------------------------------
    metamorphic     1  a reference-free invariant     out-of-vocab value; version went backward;
                                                      illegal partition; clock ran backward
    cycle           2  the history is admissible      a read mutated state; a version jumped by >1
                       under the consistency model
    symbolic        4  this transition is *legal*     the coordinator replica is wrong; a
                       under the next-state relation  non-coordinator replica changed on a write
    bit-exact      16  the full next state, exactly   anything the cheaper tiers missed (a subtle
                       (the Tier-A reference DES)     invariant-respecting error elsewhere)

> **Design decision (DD-D1):** *consult the cheapest tier that can refute the current prediction.* A
> prediction that violates a metamorphic invariant is rejected by the very-low-cost tier; one that
> violates the consistency model is caught by the low-cost cycle tier; only predictions passing both
> and still need ground-truth correction spend the high-cost bit-exact tier. Whether the cheap tiers
> are a *non-redundant, sufficient* signal over bit-exact -- whether they buy more horizon per
> oracle-dollar -- is **H17**, measured later (DS6) using exactly this interface.

The cost numbers are relative/nominal (any monotone schedule preserves the policy); they make the
oracle-dollar of H17 a concrete, swept quantity. Pure, dependency-free (Tier-B real-DST is later).
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.reference import ReferenceDistOracle

# Relative tier costs (nominal; any monotone schedule preserves the cheapest-refutation policy).
TIER_COSTS: dict[str, int] = {
    "metamorphic": 1,
    "cycle": 2,
    "symbolic": 4,
    "bit_exact": 16,
}
TIERS: tuple[str, ...] = ("metamorphic", "cycle", "symbolic", "bit_exact")


@dataclass(frozen=True)
class TierVerdict:
    """One tier's verdict on a predicted next-state: ``refuted`` (caught) + why + the cost paid."""

    tier: str
    refuted: bool
    reason: str
    cost: int


class TieredOracle:
    """A menu of verification tiers over the Tier-A reference DES (SPEC-7 §5).

    ``cheapest_refutation`` runs the tiers cheapest-first and returns the first that refutes the
    prediction (the §5 / DD-D1 policy); if all pass, the verdict is a non-refutation at bit-exact
    cost (you paid the full price to be sure). ``check`` exposes a single named tier.
    """

    def __init__(self, config: DistConfig = DEFAULT_DIST_CONFIG) -> None:
        self.config = config
        self.reference = ReferenceDistOracle(config)

    def check(
        self,
        tier: str,
        state: DistributedState,
        action: DistAction,
        predicted: DistributedState,
    ) -> TierVerdict:
        """Run a single named tier; return its verdict (``refuted`` + reason + cost)."""
        if tier not in TIER_COSTS:
            raise ValueError(f"unknown tier {tier!r}; choose from {TIERS}")
        refuted, reason = getattr(self, f"_{tier}")(state, action, predicted)
        return TierVerdict(tier, refuted, reason, TIER_COSTS[tier])

    def cheapest_refutation(
        self, state: DistributedState, action: DistAction, predicted: DistributedState
    ) -> TierVerdict:
        """The cheapest tier that refutes ``predicted`` (DD-D1); a passing bit-exact verdict else.

        ``cost`` is the cumulative oracle-dollar spent: the sum of every tier consulted up to and
        including the one that returned a verdict -- the quantity H17 minimizes per faithful step.
        """
        spent = 0
        for tier in TIERS:
            verdict = self.check(tier, state, action, predicted)
            spent += verdict.cost
            if verdict.refuted:
                return TierVerdict(tier, True, verdict.reason, spent)
        return TierVerdict("bit_exact", False, "passes every tier", spent)

    # --- the tiers (each returns (refuted, reason)) ----------------------------------------------

    def _metamorphic(
        self, state: DistributedState, action: DistAction, predicted: DistributedState
    ) -> tuple[bool, str]:
        """Reference-free invariants any legal state satisfies (cheapest)."""
        vocab = set(self.config.values) | {self.config.default_value}
        for (obj, node), r in predicted.replicas.items():
            if r.value not in vocab:
                return True, f"replica ({obj},{node}) has out-of-vocab value {r.value!r}"
            prior = state.replicas.get((obj, node))
            if prior is not None and r.version < prior.version:
                msg = f"replica ({obj},{node}) version went backward"
                return True, f"{msg} {prior.version}->{r.version}"
        covered = {n for group in predicted.partitions for n in group}
        if covered != set(self.config.nodes):
            return True, "partition groups do not cover exactly the cluster nodes"
        if predicted.clock < state.clock:
            return True, f"clock ran backward {state.clock}->{predicted.clock}"
        if not predicted.down <= set(self.config.nodes):
            return True, "down set contains an unknown node"
        # Consensus metadata (DS0 incr 16): the election term is monotone (a term can never go
        # backward) and the leader is always a known cluster node or unset — reference-free
        # invariants any legal protocol step satisfies, so a bogus-leader or backward-term
        # prediction is refuted at the cheapest tier.
        if predicted.term < state.term:
            return True, f"election term went backward {state.term}->{predicted.term}"
        if predicted.leader is not None and predicted.leader not in set(self.config.nodes):
            return True, f"leader {predicted.leader!r} is not a cluster node"
        # The Raft commit index (DS0 incr 19) is monotone — a committed entry is permanent, so the
        # committed-prefix length can never shrink. A reference-free safety invariant any legal log
        # step satisfies, so a prediction that "un-commits" an entry is refuted at the cheap tier.
        if predicted.commit_index < state.commit_index:
            return True, (f"commit index went backward "
                          f"{state.commit_index}->{predicted.commit_index}")
        # The voting membership (DS0 incr 20) is always a subset of the configured cluster — a
        # reference-free invariant any legal reconfiguration satisfies (empty = the "all vote"
        # sentinel), so a membership naming an unknown node is refuted at the cheapest tier.
        if predicted.members and not predicted.members <= set(self.config.nodes):
            return True, "voting membership contains an unknown node"
        # Node versions (DS0 incr 22) are non-negative and belong to known cluster nodes — a
        # reference-free invariant any legal `deploy` satisfies.
        for node, ver in predicted.versions.items():
            if node not in set(self.config.nodes) or ver < 0:
                return True, f"node {node!r} has an invalid version {ver}"
        return False, ""

    def _cycle(
        self, state: DistributedState, action: DistAction, predicted: DistributedState
    ) -> tuple[bool, str]:
        """History admissibility: a read never mutates state; versions jump by <=1 per step."""
        if action.name == "get" and predicted.replicas != state.replicas:
            return True, "a read (get) mutated a replica -- inadmissible history"
        if action.name in ("anti_entropy", "gossip", "append"):
            # read-repair (DS0 incr 12), pairwise gossip (incr 15), and the committed-log fold of
            # ``append`` (incr 19 — a rejoined follower backfills several missed committed entries
            # at once) legitimately jump a stale replica *several* versions in one step, so the
            # "jump by <=1" rule does not apply -- defer to bit-exact.
            return False, ""
        for (obj, node), r in predicted.replicas.items():
            prior = state.replicas.get((obj, node))
            base = prior.version if prior is not None else 0
            if r.version > base + 1:
                return True, f"replica ({obj},{node}) version jumped by >1 in one step"
        return False, ""

    def _symbolic(
        self, state: DistributedState, action: DistAction, predicted: DistributedState
    ) -> tuple[bool, str]:
        """Is *this* transition legal under the protocol's next-state relation (per action type)?

        Checks the structural relation the action must satisfy -- cheaper than recomputing the full
        canonical next-state, but catches the action-specific structural errors bit-exact would.
        """
        name = action.name
        no_write = ("get", "partition", "heal", "crash", "restart", "drop", "delay", "reorder",
                    "clock_skew", "begin", "tget", "tput", "abort", "elect", "step_down",
                    "lease", "lread", "add_replica", "remove_replica", "enqueue", "dequeue",
                    "deploy", "host")
        if name in no_write:
            # none of these write replicas; the replica map must be unchanged. ``drop`` (DS0 inc 11)
            # and ``delay``/``reorder`` (DS0 inc 13) only touch the in-flight set; the txn ops
            # begin/tget/tput/abort only touch the
            # (consistency-invisible) txn buffer — a committed write reaches replicas only via
            # ``commit`` (handled below); ``elect`` (inc 16) / ``step_down`` (inc 17) / ``lease`` /
            # ``lread`` (inc 18) write only leader/term/lease metadata, never a replica (`lread` is
            # a read); ``enqueue``/``dequeue`` (inc 21) write only the separate queue data plane —
            # all defer to bit-exact. A no-write op that mutated a *replica* is an inadmissible
            # transition the symbolic tier refutes.
            if predicted.replicas != state.replicas:
                return True, f"{name} must not change any replica"
            return False, ""
        if name in ("anti_entropy", "gossip"):
            # read-repair (DS0 incr 12) reconciles a node to the winning ``(version, value)`` among
            # its reachable replicas, and pairwise ``gossip`` (incr 15) reconciles two nodes to
            # their mutual winner; the exact post-state depends on which peers are reachable (the
            # medium), which the cheap symbolic tier does not recompute — defer to bit-exact.
            return False, ""
        if name == "commit":
            # ``commit`` applies the txn's buffered writes (an MVCC bump per key) or aborts on a
            # read-set conflict; the exact post-state depends on the buffered writes the symbolic
            # tier does not track, so it defers to bit-exact (no cheap-tier refutation here).
            return False, ""
        if name in ("propose", "append"):
            # ``propose`` (DS0 incr 16) is a leader-fenced write and ``append`` (incr 19) a
            # replicated-log append: whether either commits depends on leadership + the reachable
            # majority, and on commit each writes *several* replicas (and ``append`` also rewrites
            # logs + the commit index) — the exact post-state depends on the medium the symbolic
            # does not recompute, so both defer to bit-exact (like the quorum ``put``).
            return False, ""
        if name in ("put", "cas"):
            node, key = action.args[0], action.args[1]
            prior = state.replicas.get((key, node))
            if prior is None or not state.is_up(node):
                # the unavailable/no-replica path leaves replicas unchanged; bit-exact judges it
                return False, ""
            # the coordinator replica is fully determined by the relation
            if name == "cas" and action.args[2] != prior.value:
                # conflict: no write -> coordinator unchanged
                if predicted.replicas.get((key, node)) != prior:
                    return True, "cas conflict must leave the coordinator replica unchanged"
                return False, ""
            want_val = action.args[2] if name == "put" else action.args[3]
            got = predicted.replicas.get((key, node))
            if got is None or got.version != prior.version + 1 or got.value != want_val:
                return True, f"coordinator replica ({key},{node}) is not the legal post-write value"
            # peers must not change on the write step (they change only via advance)
            for (o, n), r in predicted.replicas.items():
                if (o, n) != (key, node) and r != state.replicas.get((o, n)):
                    return True, f"non-coordinator replica ({o},{n}) changed on a write"
            return False, ""
        # advance: every changed replica must match an in-flight message that was deliverable
        changed = {
            (o, n): r for (o, n), r in predicted.replicas.items() if r != state.replicas.get((o, n))
        }
        deliverable = {
            (m.object_id, m.dst): (m.version, m.value)
            for m in state.inflight.values()
        }
        for (o, n), r in changed.items():
            if deliverable.get((o, n)) != (r.version, r.value):
                msg = f"advance delivered ({o},{n})={r.value!r} with no matching in-flight msg"
                return True, msg
        return False, ""

    def _bit_exact(
        self, state: DistributedState, action: DistAction, predicted: DistributedState
    ) -> tuple[bool, str]:
        """The full Tier-A reference DES: recompute truth and compare bit-for-bit (priciest)."""
        truth = self.reference.step(state, action).state
        if predicted != truth:
            return True, "predicted next-state differs from the bit-exact reference"
        return False, ""
