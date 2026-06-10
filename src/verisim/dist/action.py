"""The distributed action grammar (SPEC-7 §3.2, DS0 increment 1).

Actions in a constrained grammar paired with the reference oracle, exactly as every prior world
pairs a grammar with its oracle. Two of SPEC-7's three families ship in increment 1 -- the client
workload and the fault/time medium (the consensus/admin family arrives with the Raft-subset):

    put <node> <key> <val>          # client: write <val> to <key> at <node> (async repl)
    get <node> <key>                # client: read <key>'s local replica at <node> (may be stale)
    cas <node> <key> <old> <new>    # client: set <key> to <new> iff <node>'s local value == <old>
    advance <dt>                    # time: +<dt> clock; deliver in-flight msgs now due & reachable
    partition <nodes> | <nodes>     # fault: split the network into groups (| separates groups)
    heal                            # fault: remove all partitions (one fully-connected group)
    crash <node>                    # fault: <node> goes down (stops delivering/applying)
    restart <node>                  # fault: <node> comes back up
    drop <src> <dst>                # fault: lose every in-flight message from <src> to <dst>
    delay <src> <dst> <dt>          # fault: defer every in-flight <src>-><dst> message by <dt>
    reorder <src> <dst>             # fault: reverse the delivery schedule of <src>-><dst> messages
    clock_skew <node> <delta>       # fault: offset <node>'s clock by <delta> (signed; 0 clears it)
    anti_entropy <node>             # protocol: read-repair <node> to the latest reachable replica
    gossip <a> <b>                  # protocol: pairwise sync — a and b both adopt the per-obj winner
    elect <node>                    # consensus: <node> becomes leader iff its side holds a majority
    propose <node> <key> <val>      # consensus: leader-only write to a majority (term-fenced)
    step_down <node>                # consensus: the current leader relinquishes (leaderless, same term)
    lease <node> <dt>               # consensus: the leader takes a read lease until clock+<dt>
    lread <node> <key>              # consensus: a leader-lease local linearizable read (no quorum)
    append <node> <key> <val>       # consensus: append <key>=<val> to the replicated log (Raft log)
    add_replica <node>              # consensus: add <node> to the voting membership (quorum grows)
    remove_replica <node>           # consensus: remove <node> from the voting membership (quorum shrinks)

The fault/time family is the source of all interesting dynamics (stale reads under partition,
convergence after heal+advance) -- the distributed analogue of SPEC-5's ``advance Δt`` and SPEC-6's
scheduler, and the ``BUGGIFY`` of deterministic simulation testing (SPEC-7 §2.1). ``drop`` (DS0
increment 11) is the unreliable-network fault: where ``partition`` *holds* a message (delivered once
the link ``heal``s), ``drop`` **destroys** it, so the destination replica permanently misses that
write -- the lost-message anomaly that breaks the eventual-consistency convergence guarantee until a
*newer* write overwrites it (ED18). ``anti_entropy`` (DS0 increment 12) is the first **protocol**
op: the **read-repair / anti-entropy** mechanism real eventually-consistent stores (Dynamo,
Cassandra) use to converge *despite* lost messages -- a node pulls each object to the latest
``(version, value)`` among its reachable replicas, so it repairs a dropped write that ``advance``
never can, bounded only by what is currently reachable (ED19). ``gossip`` (DS0 increment 15) is its
**pairwise, bidirectional** sibling: ``gossip a b`` reconciles *both* ``a`` and ``b`` to the
per-object winner of their two replicas in one step (the Merkle-tree pairwise anti-entropy of
Dynamo/Cassandra, vs ``anti_entropy``'s one-directional pull-to-one-node), so a chain of pairwise
gossips spreads a write across the whole reachable component epidemically (ED22). ``delay`` and
``reorder`` (DS0
increment 13) are the message-timing faults: ``delay src dst dt`` defers every in-flight
``src``->``dst`` message by ``dt`` (a *recoverable* delay -- the counterpart to ``drop``'s
unrecoverable loss), and ``reorder src dst`` reverses the delivery schedule of that channel's
messages (the multiset of times preserved, the order flipped). Both only edit the existing
``Message.deliver_after`` field, so they add no state and compose with every consistency model;
they make the §3.4 "reorder/skew is the medium" axis a controllable input (ED20). ``clock_skew``
(DS0 increment 14) is the last of the §3.4 medium faults: ``clock_skew node delta`` offsets a node's
local clock by a signed ``delta``, which shifts the ``deliver_after`` it stamps on the messages it
sends (ahead = its sends are deferred, behind = rushed). Because the protocol resolves conflicts by
last-writer-wins on ``(version, value)`` -- never on a wall-clock timestamp -- skew shifts *when* a
write is delivered but never *which* write wins, so the converged state is clock-independent (ED21,
the property deterministic-simulation testing injects skew to verify). ``elect``/``propose`` (DS0
increment 16) are the **consensus** family — the third action family (SPEC-7 §3.2), a Raft-subset
leader-election core. ``elect node`` makes ``node`` the cluster leader **iff its partition side holds
a strict majority of the live nodes** (so two majorities — hence two leaders — can never coexist: no
split-brain), bumping the monotone ``term``. ``propose node key val`` is a **leader-fenced** write:
it commits a synchronous majority write (the consensus quorum, regardless of the KV
``consistency_model``) only if ``node`` is the *current* leader, so a leader deposed by a higher-term
election cannot commit even after the partition heals — the Raft leader-completeness safety property
plain ``quorum`` writes lack (ED23). ``step_down`` (DS0 increment 17) completes the leadership
lifecycle: ``step_down node`` lets the *current* leader **voluntarily relinquish** power, leaving the
cluster **leaderless at the same term** — the graceful counterpart to ``elect``'s involuntary,
higher-term deposition. Until a fresh ``elect`` installs a successor every ``propose`` is rejected
``not_leader``, so a clean handoff is ``step_down`` then ``elect <successor>`` and no leaderless
window ever commits. Relinquishing needs no quorum (it reads only the node's own leadership), so a
leader stranded in a minority can step down where its ``propose`` is ``no_quorum`` — giving up
authority is always safe, committing under it is not (ED24). ``lease``/``lread`` (DS0 increment 18)
are the **leader lease** — the Raft read optimization. ``lease node dt`` lets the *current* leader
take a read lease through global clock ``+ dt``; while that lease holds, ``lread node key`` serves a
**local linearizable read without a quorum round-trip** (safe because the lease guarantees the
leader's term is uncontested — a new ``elect`` is *blocked* until the lease expires). So a leader
partitioned into the minority can still ``lread`` (local, no quorum) where its ``propose`` is
``no_quorum`` — the lease's whole purpose. The safety tension it resolves: a new ``elect`` is
rejected ``lease_held`` until the incumbent's lease expires (a successor waits it out), so leadership
cannot change hands under a live lease — but ``step_down`` *releases* the lease immediately, so a
graceful handoff needs no wait where a crashed leader forces the cluster to wait the lease out (ED25).
``append`` (DS0 increment 19) is the **replicated-log** path the spec named since increment 1:
``append node key val`` appends a ``(term, index, key, value)`` entry to the leader's log and
replicates it to the reachable followers (who adopt the leader's prefix, **overwriting any divergent
uncommitted tail** — the log-matching reconciliation), committing it (and applying it to the KV)
**iff a majority holds it**. A minority-stranded leader still appends locally (``uncommitted``) but
does not commit, so its entry can be overwritten by a higher-term leader — the Raft log-matching
safety the one-shot ``propose`` could not express (ED26). ``add_replica``/``remove_replica`` (DS0
increment 20) are the **membership-change** admin ops the §3.2 grammar named: they reconfigure the
*consensus voting set* (the nodes that count toward an election/commit quorum), a leader-committed
change, so the **majority threshold tracks the membership** — ``remove_replica`` shrinks the cluster
(a smaller majority suffices, the standard way to restore availability after nodes fail) and
``add_replica`` grows it. All config nodes still physically store replicas; membership is the voting
overlay (the empty set is the "all nodes vote" default). The active leader cannot be removed
(``is_leader`` — step it down first), and the last member cannot be removed (ED27).
"""

from __future__ import annotations

from dataclasses import dataclass

# name -> fixed arity (number of whitespace tokens after the name), or None for variable arity
# (``partition`` carries a ``|``-separated group list; ``heal`` takes none).
_ARITY: dict[str, int | None] = {
    "put": 3,  # node key val
    "get": 2,  # node key
    "cas": 4,  # node key old new
    "advance": 1,  # dt
    "partition": None,  # <nodes> | <nodes> [| ...]
    "heal": 0,
    "crash": 1,  # node
    "restart": 1,  # node
    "drop": 2,  # src dst  -- lose every in-flight message from <src> to <dst> (DS0 incr 11)
    "delay": 3,  # src dst dt  -- defer every in-flight <src>-><dst> message by <dt> (DS0 incr 13)
    "reorder": 2,  # src dst  -- reverse the delivery schedule of <src>-><dst> messages (DS0 incr 13)
    "clock_skew": 2,  # node delta  -- set <node>'s clock offset (signed; DS0 incr 14)
    "anti_entropy": 1,  # node  -- read-repair <node> to the latest reachable replica (DS0 incr 12)
    "gossip": 2,  # a b  -- pairwise bidirectional anti-entropy: both adopt the winner (DS0 incr 15)
    "elect": 1,  # node  -- <node> becomes leader iff its side holds a majority (DS0 incr 16)
    "propose": 3,  # node key val  -- leader-only term-fenced majority write (DS0 incr 16)
    "step_down": 1,  # node  -- the current leader relinquishes; leaderless at the same term (incr 17)
    "lease": 2,  # node dt  -- the leader takes a read lease until clock+dt (DS0 incr 18)
    "lread": 2,  # node key  -- a leader-lease local linearizable read, no quorum (DS0 incr 18)
    "append": 3,  # node key val  -- append to the Raft replicated log; commit on majority (incr 19)
    "add_replica": 1,  # node  -- add <node> to the voting membership (DS0 incr 20)
    "remove_replica": 1,  # node  -- remove <node> from the voting membership (DS0 incr 20)
    # Transaction family (DS0 increment 2): a multi-key OCC transaction at a coordinator node.
    "begin": 2,  # node txn  -- open a transaction
    "tget": 3,  # node txn key  -- read <key> within the txn (pins the read version for validation)
    "tput": 4,  # node txn key val  -- buffer a write to <key> within the txn
    "commit": 2,  # node txn  -- validate the read-set + apply buffered writes atomically, or abort
    "abort": 2,  # node txn  -- discard the txn
}

# ``begin``/``commit``/``abort`` + the txn-scoped ``tget``/``tput`` are client ops (the workload).
CLIENT_OPS = frozenset({"put", "get", "cas", "begin", "tget", "tput", "commit", "abort"})
TXN_OPS = frozenset({"begin", "tget", "tput", "commit", "abort"})
FAULT_OPS = frozenset(
    {"advance", "partition", "heal", "crash", "restart", "drop", "delay", "reorder", "clock_skew"}
)
# Protocol/admin ops (SPEC-7 §3.2, the third action family). ``anti_entropy`` (the read-repair
# convergence mechanism, DS0 increment 12) and ``gossip`` (pairwise anti-entropy, incr 15) are the
# convergence ops; ``elect``/``propose`` (the Raft-subset consensus core, incr 16) are the consensus
# ops. ``CONSENSUS_OPS`` is the subset that reads/writes the leader/term metadata.
CONSENSUS_OPS = frozenset({
    "elect", "propose", "step_down", "lease", "lread", "append", "add_replica", "remove_replica",
})
PROTOCOL_OPS = frozenset({"anti_entropy", "gossip"}) | CONSENSUS_OPS


class DistParseError(ValueError):
    """Raised when a string is not a valid distributed action in the DS0 grammar."""


@dataclass(frozen=True)
class DistAction:
    """A parsed distributed action. ``args`` are the post-name tokens; ``groups`` holds the parsed
    ``partition`` node groups (empty for every other op)."""

    raw: str
    name: str
    args: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] = ()


def parse_dist_action(raw: str) -> DistAction:
    """Parse an action string into a :class:`DistAction`, validating name + arity + structure."""
    parts = raw.split()
    if not parts:
        raise DistParseError("empty action")
    name = parts[0]
    if name not in _ARITY:
        raise DistParseError(f"unknown action {name!r}; choose from {sorted(_ARITY)}")
    rest = parts[1:]

    if name == "partition":
        groups = _parse_partition_groups(rest, raw)
        return DistAction(raw=raw, name=name, args=tuple(rest), groups=groups)

    arity = _ARITY[name]
    assert arity is not None
    if len(rest) != arity:
        raise DistParseError(f"{name} expects {arity} args, got {len(rest)}: {raw!r}")
    if name == "advance":
        _parse_positive_int(rest[0], raw)
    if name == "delay":
        _parse_positive_int(rest[2], raw)  # dt: a positive deferral, like advance's
    if name == "lease":
        _parse_positive_int(rest[1], raw)  # dt: a positive lease duration, like advance's
    if name == "clock_skew":
        _parse_signed_int(rest[1], raw)  # delta: a signed clock offset (may be negative or 0)
    return DistAction(raw=raw, name=name, args=tuple(rest))


def _parse_partition_groups(rest: list[str], raw: str) -> tuple[tuple[str, ...], ...]:
    """Parse ``n0 n1 | n2`` into ``(("n0","n1"), ("n2",))``; require >= 2 non-empty groups."""
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    for tok in rest:
        if tok == "|":
            if not current:
                raise DistParseError(f"empty partition group in {raw!r}")
            groups.append(tuple(current))
            current = []
        else:
            current.append(tok)
    if current:
        groups.append(tuple(current))
    if len(groups) < 2:
        raise DistParseError(f"partition needs >= 2 groups separated by '|': {raw!r}")
    seen: set[str] = set()
    for group in groups:
        for node in group:
            if node in seen:
                raise DistParseError(f"node {node!r} in more than one partition group: {raw!r}")
            seen.add(node)
    return tuple(groups)


def _parse_positive_int(tok: str, raw: str) -> int:
    try:
        value = int(tok)
    except ValueError as exc:
        raise DistParseError(f"expected an integer, got {tok!r} in {raw!r}") from exc
    if value <= 0:
        raise DistParseError(f"dt must be positive, got {value} in {raw!r}")
    return value


def _parse_signed_int(tok: str, raw: str) -> int:
    """Parse a signed integer (the ``clock_skew`` offset may be negative or zero)."""
    try:
        return int(tok)
    except ValueError as exc:
        raise DistParseError(f"expected an integer, got {tok!r} in {raw!r}") from exc
