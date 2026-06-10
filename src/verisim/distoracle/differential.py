"""The distributed differential-validation harness (SPEC-7 §5.2): run Tier-A (the analytic DES) and
Tier-B (the autonomous-actor system oracle) on the same ``(state, action)`` and return an exact
agreement record.

This is the distributed echo of :mod:`verisim.oracle.differential` (SPEC-11 §3). It calls *both*
oracles on the identical transition and compares them on the **observable-cluster channel** -- the
state a real cluster actually exposes:

  - **replicas** -- each ``(object, node)`` replica's ``(version, value)``;
  - **in-flight** -- the set of replication messages sent but not delivered, compared
    *id-independently* as ``(src, dst, object, version, value, deliver_after)`` tuples (the monotone
    message id is internal bookkeeping, not observable behavior);
  - **medium** -- the partition groups, the crashed-node set, and the clock;
  - **result** -- the client-visible ``(status, value)`` of the step.

The **causal log and the monotone id counters are deliberately excluded** from the channel: they are
bookkeeping of *our representation*, reconstructed identically by construction (exactly as the host
differential excludes the ``last`` observation to keep the world channel orthogonal). The headline
relation is whether the two oracles agree on the *observable cluster*.

When they disagree, :func:`classify_dist_divergence` localizes the cause into a named boundary --
the only one the DS0-increment-1 KV semantics admits is ``delivery_order`` (a replica whose value
depends on the order messages were delivered, i.e. a *non-commutative* convergence, which a correct
LWW actor never produces and the broken-arrival negative control always does) -- or flags it
``residual`` for inspection. A divergence is never silent.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.dist.action import DistAction
from verisim.dist.state import DistributedState
from verisim.distoracle.base import DistOracle, DistStepResult
from verisim.host.state import to_canonical_host

AGREE = "agree"
# The one named modeling boundary the KV semantics admits: a converged replica whose value depends
# on delivery order (a non-commutative join). A correct LWW actor is order-independent, so this is
# only ever produced by a faithfulness break (the broken-arrival negative control).
C_DELIVERY_ORDER = "delivery_order"
RESIDUAL = "residual"  # an unexplained disagreement -- a first-class finding

BOUNDARY_CLASSES = (C_DELIVERY_ORDER,)


def cluster_view(state: DistributedState) -> str:
    """The observable-cluster channel: replicas + in-flight + medium + result, id-independent.

    Excludes the causal log and the monotone ``next_event_id``/``next_msg_id`` counters (internal
    bookkeeping), so the channel is implementation-independent and orthogonal to representation.
    """
    replicas = sorted(
        (r.object_id, r.node_id, r.version, r.value) for r in state.replicas.values()
    )
    inflight = sorted(
        # ``deps`` (the causal context, empty under eventual/linearizable) is part of the observable
        # message, so it is compared too — validating that Tier-B attaches the same causal deps as
        # Tier-A. It folds in id-independently and is a no-op where the model does not order
        # delivery (deps is () under eventual / linearizable, so the channel is unchanged there).
        (m.src, m.dst, m.object_id, m.version, m.value, m.deliver_after, m.deps)
        for m in state.inflight.values()
    )
    partitions = sorted(sorted(g) for g in state.partitions)
    # Queue replicas (DS0 incr 21) are part of the observable cluster — each (queue, node) replica's
    # ordered contents, sorted id-independently. Empty for a KV-only cluster, so the channel is
    # unchanged where queues are unused.
    queues = sorted(
        (q, n, list(items)) for (q, n), items in state.queues.items() if items
    )
    # Per-node running versions (DS0 incr 22) are observable cluster metadata; empty (all base) for
    # a cluster that never deploys, so the channel is unchanged there.
    versions = sorted((n, v) for n, v in state.versions.items() if v != 0)
    # The cluster config (DS0 incr 24, `config_push`) is observable cluster metadata — each pushed
    # (node, key) value, sorted id-independently. Empty for a cluster that never pushes config, so
    # the channel is unchanged there (and the config-divergence-under-partition is compared too).
    config = sorted((n, k, v) for (n, k), v in state.config.items())
    # CRDT G-counter copies (DS0 incr 28) are observable cluster state — each (key, holder, owner)
    # sub-count, sorted id-independently. Empty for a cluster with no CRDT counter (channel same).
    gcounters = sorted((k, h, o, c) for (k, h, o), c in state.gcounters.items() if c != 0)
    # The PN-counter decrement half (DS0 incr 29) is observable cluster state too — same shape.
    ncounters = sorted((k, h, o, c) for (k, h, o), c in state.ncounters.items() if c != 0)
    # The embedded per-node hosts (DS0 incr 23) are observable cluster state — each node's host
    # canonical form, sorted by node. Empty for a host-free cluster, so the channel is unchanged.
    hosts = sorted((n, to_canonical_host(h)) for n, h in state.hosts.items())
    return repr({
        "replicas": replicas,
        "inflight": inflight,
        "partitions": partitions,
        "down": sorted(state.down),
        "clock": state.clock,
        "last_result": state.last_result,
        "queues": queues,
        "versions": versions,
        "config": config,
        "gcounters": gcounters,
        "ncounters": ncounters,
        "hosts": hosts,
    })


@dataclass(frozen=True)
class DistDiffRecord:
    """An exact agreement record for one distributed ``(state, action)`` transition (§5.2)."""

    action_raw: str
    command: str
    agree_cluster: bool
    divergence_class: str

    @property
    def agree(self) -> bool:
        return self.agree_cluster


def dist_differential_step(
    state: DistributedState, action: DistAction, ref: DistOracle, sys: DistOracle
) -> DistDiffRecord:
    """Run ``ref`` (Tier-A) and ``sys`` (Tier-B) on one transition; return the agreement record."""
    r_ref: DistStepResult = ref.step(state, action)
    r_sys: DistStepResult = sys.step(state, action)
    agree = cluster_view(r_ref.state) == cluster_view(r_sys.state)
    cls = AGREE if agree else classify_dist_divergence(state, action, r_ref.state, r_sys.state)
    return DistDiffRecord(
        action_raw=action.raw,
        command=action.name,
        agree_cluster=agree,
        divergence_class=cls,
    )


def classify_dist_divergence(
    state: DistributedState,
    action: DistAction,
    ref_next: DistributedState,
    sys_next: DistributedState,
) -> str:
    """Localize an observable-cluster disagreement to a named boundary (or ``residual``).

    ``advance`` is the only action that delivers messages, so it is the only one whose result can
    depend on delivery order. A replica-value disagreement on an ``advance`` step is therefore the
    ``delivery_order`` boundary -- the non-commutative convergence a correct LWW actor never
    produces. Any other disagreement (or a replica disagreement on a non-``advance`` step) is a
    first-class ``residual`` finding.
    """
    if action.name == "advance":
        ref_repl = {k: (r.version, r.value) for k, r in ref_next.replicas.items()}
        sys_repl = {k: (r.version, r.value) for k, r in sys_next.replicas.items()}
        if ref_repl != sys_repl:
            return C_DELIVERY_ORDER
    return RESIDUAL
