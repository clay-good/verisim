"""Elle-style black-box serializability checking (SPEC-7 §5, §9.1; DS3 increment 2).

The per-step ``cycle`` tier (:mod:`verisim.distoracle.tiers`) is the *eventual-consistency* form of
history admissibility: it checks one transition in isolation (a read did not mutate state, a version
did not jump). This module is its **stronger-consistency sibling** — the over-a-history checker the
DS3 milestone deferred ("Elle-style cross-object cycle detection for stronger consistency models").
It is the distributed analogue of Jepsen's Elle (Kingsbury & Alvaro, VLDB 2020): given only the
**observable transaction history** — what each committed transaction read and wrote — it
reconstructs Adya's Direct Serialization Graph (the DSG) and reports a violation iff it has a cycle.

Why this matters for the program. Every other oracle tier consults the *reference DES* (the analytic
truth) to judge a prediction. Elle consults **nothing**: it is a pure function of the client-visible
history, exactly the signal a real operator (or a defender watching a cluster they do not control)
actually has. So it answers a different question than bit-exact truth — *was the observed schedule
serializable?* — and answers it for free. ED10 (SPEC-7 §3.2) shows it **recovers the ED9 write-skew
anomaly black-box** (a G2 anti-dependency cycle) at exactly the rate the oracle's commit-count sees,
and **certifies** the ``serializable`` isolation level (zero cycles) the oracle enforces — a cheap,
reference-free verifier that agrees with the expensive one on the question it is built to answer.

The theory (Adya 1999; Elle, VLDB 2020). A history over a multi-version store is serializable iff
its DSG is acyclic. The DSG has one node per committed transaction and three edge kinds between
them, all read off the per-object **version order** (the MVCC version sequence the store assigns):

  - **ww** (write-depends): ``Ti`` installs version ``v`` of key ``k`` and ``Tj`` installs the next
    installed version of ``k`` — ``Ti → Tj`` (the write order).
  - **wr** (read-depends): ``Ti`` installs version ``v`` of ``k`` and ``Tj`` reads exactly ``v`` —
    ``Ti → Tj`` (``Tj`` saw ``Ti``'s write).
  - **rw** (anti-dependency): ``Ti`` reads version ``v`` of ``k`` and ``Tj`` installs the version
    *immediately after* ``v`` — ``Ti → Tj`` (``Tj`` overwrote what ``Ti`` read).

A cycle classifies (Adya's G-hierarchy, the standard isolation-anomaly taxonomy):

  - **G0** (dirty write): a cycle of ``ww`` edges only.
  - **G1c** (circular information flow): a cycle of ``ww``/``wr`` edges only.
  - **G2** (anti-dependency cycle): any cycle containing at least one ``rw`` edge — the general
    non-serializable form, and the canonical shape of **write skew** (two transactions each read a
    pair and write disjoint halves: ``A →_rw B →_rw A``).

Version ``0`` is the boot version of every replica; it has no writer (a virtual initial txn), so no
real ``ww``/``wr`` edge originates at it, but a transaction that *reads* version ``0`` still gets an
``rw`` edge to whoever installed version ``1``. Pure standard library, dependency-free, GPU-free;
deterministic (edges and the cycle search are built in sorted order).

The version oracle (DS3 increment 3 — list-append / value-recoverable histories).
:class:`TxnObservation` above is the *version-supplied* mode: the store hands Elle the integer MVCC
version each read pinned and each write installed. That is one cooperation Jepsen's Elle removes,
and the reason it works against a true black box. Over a **list-append** register — every write
*appends* a globally-unique value to a key's list, every read returns the **whole list** — the
per-key version order is **recoverable from the read values themselves**: a read returning
``[x, y, z]`` is direct testimony that the append of ``x`` preceded ``y`` preceded ``z``, with no
question put to the store (Kingsbury & Alvaro 2020, the "version oracle"). :func:`recover_versions`
is that recovery — it merges every observed list-read for a key (each is a *prefix* of the one
growing append log) into a single total order, then :func:`check_serializable_appends` assigns each
appended value its recovered version and reuses the exact DSG/cycle machinery above. Two anomaly
classes only the value-recovery path can even *represent* (the integer-version mode receives a
single non-contradictory version sequence and so cannot express them) surface during recovery,
before any cycle search:

  - **incompatible-order** — two reads of one key disagree on append order (neither list is a prefix
    of the other: a *fork*). This is the black-box signature of split-brain — a partition let two
    sides extend the same key divergently — the distributed anomaly the §9.1 consistency view exists
    to catch, now caught from the client-visible history alone.
  - **dirty-read** (Adya G1a, aborted read) / **duplicate-write** — a read observed a value no
    committed transaction appended, or a value was appended/observed twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TxnObservation:
    """One committed transaction's observable footprint: what it read and what it installed.

    ``reads`` maps each key the transaction read to the **version** it observed; ``writes`` maps
    each key it committed to the **version it installed** (the new MVCC version). Versions are the
    store's monotone per-object counters — the order Elle builds the DSG from. Only *committed*
    transactions enter the history (an aborted transaction has no effect on serializability).
    """

    txn_id: str
    reads: tuple[tuple[str, int], ...] = ()
    writes: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Edge:
    """A directed DSG edge ``src → dst`` of kind ``ww`` / ``wr`` / ``rw`` over key ``key``."""

    src: str
    dst: str
    kind: str  # "ww" | "wr" | "rw"
    key: str


@dataclass
class ElleReport:
    """The verdict over a transaction history: serializable iff the DSG has no cycle."""

    serializable: bool
    anomaly: str  # "" | "G0" | "G1c" | "G2"
    cycle: tuple[str, ...] = ()  # the transaction ids on the witnessing cycle (empty if none)
    cycle_kinds: tuple[str, ...] = ()  # the edge kinds along the cycle, in order
    n_txns: int = 0
    edges: list[Edge] = field(default_factory=list)

    @property
    def detail(self) -> str:
        if self.serializable:
            return f"serializable ({self.n_txns} txns, {len(self.edges)} DSG edges)"
        if not self.cycle:  # a recovery anomaly (incompatible-order / dirty-read / duplicate-write)
            return f"{self.anomaly} (recovery anomaly, no DSG cycle)"
        return f"{self.anomaly} cycle {'->'.join(self.cycle)} via {','.join(self.cycle_kinds)}"


def build_dsg(history: list[TxnObservation]) -> list[Edge]:
    """Build the Direct Serialization Graph edges for a committed-transaction ``history``.

    Reads the per-key version order off the installed versions (plus the boot version ``0``) and
    emits ``ww`` / ``wr`` / ``rw`` edges per Adya's definitions. Deterministic: keys, versions, and
    transactions are visited in sorted order, so the returned edge list is canonical.
    """
    # writer[key][version] = txn_id that installed that version (a version has at most one writer:
    # first-committer-wins makes each MVCC bump a distinct version owned by one transaction).
    writer: dict[str, dict[int, str]] = {}
    readers: dict[str, dict[int, list[str]]] = {}
    keys: set[str] = set()
    for t in sorted(history, key=lambda t: t.txn_id):
        for k, v in t.writes:
            keys.add(k)
            writer.setdefault(k, {})[v] = t.txn_id
        for k, v in t.reads:
            keys.add(k)
            readers.setdefault(k, {}).setdefault(v, []).append(t.txn_id)

    edges: list[Edge] = []
    for key in sorted(keys):
        kw = writer.get(key, {})
        kr = readers.get(key, {})
        # The installed-version order for this key: version 0 (boot, no writer) then every written
        # version, ascending. ``succ`` maps a version to the next installed version after it.
        installed = sorted({0, *kw.keys()})
        succ = {installed[i]: installed[i + 1] for i in range(len(installed) - 1)}

        # ww: consecutive installed versions (both with real writers) — writer(v) -> writer(next).
        for v, nxt in succ.items():
            a, b = kw.get(v), kw.get(nxt)
            if a is not None and b is not None and a != b:
                edges.append(Edge(a, b, "ww", key))

        # wr: writer(v) -> every transaction that read exactly v.
        for v, w in kw.items():
            for r in sorted(kr.get(v, [])):
                if r != w:
                    edges.append(Edge(w, r, "wr", key))

        # rw: a reader of version v -> the writer of the version immediately after v.
        for rv, reader_list in kr.items():
            succ_v = succ.get(rv)
            if succ_v is None:
                continue
            succ_writer = kw.get(succ_v)
            if succ_writer is None:
                continue
            for r in sorted(reader_list):
                if r != succ_writer:
                    edges.append(Edge(r, succ_writer, "rw", key))
    return edges


def _find_cycle(
    nodes: list[str], adj: dict[str, list[Edge]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (cycle txn ids, edge kinds along it) for the first cycle found, or ((), ())."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)
    # parent edge used to reach each node, so a back-edge reconstructs the witnessing cycle.
    parent_edge: dict[str, Edge] = {}

    def walk(u: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        color[u] = GRAY
        for e in adj.get(u, []):
            v = e.dst
            if color[v] == WHITE:
                parent_edge[v] = e
                got = walk(v)
                if got[0]:
                    return got
            elif color[v] == GRAY:
                # back-edge u->v closes a cycle; walk parents from u back to v.
                path_nodes = [v, u]
                path_edges = [e]
                cur = u
                while cur != v:
                    pe = parent_edge[cur]
                    path_edges.append(pe)
                    cur = pe.src
                    path_nodes.append(cur)
                nodes_ordered = tuple(reversed(path_nodes[:-1]))
                kinds = tuple(reversed([pe.kind for pe in path_edges]))
                return nodes_ordered, kinds
        color[u] = BLACK
        return (), ()

    for n in nodes:
        if color[n] == WHITE:
            got = walk(n)
            if got[0]:
                return got
    return (), ()


def _classify(kinds: tuple[str, ...]) -> str:
    """Adya's G-hierarchy from the edge kinds on a cycle (G2 > G1c > G0, most-severe label)."""
    if "rw" in kinds:
        return "G2"
    if "wr" in kinds:
        return "G1c"
    return "G0"


def check_serializable(history: list[TxnObservation]) -> ElleReport:
    """Elle's verdict: build the DSG and report the first cycle (with its anomaly class), if any.

    Black-box — it consults no oracle and no cluster state, only the observable read/write history.
    Serializable iff the DSG is acyclic (Adya 1999). The reported cycle is a concrete witness.
    """
    edges = build_dsg(history)
    nodes = sorted({t.txn_id for t in history})
    adj: dict[str, list[Edge]] = {n: [] for n in nodes}
    for e in edges:
        adj.setdefault(e.src, []).append(e)
    # Deterministic adjacency order so the witnessing cycle is reproducible.
    for n in adj:
        adj[n].sort(key=lambda e: (e.dst, e.kind, e.key))
    cycle, kinds = _find_cycle(nodes, adj)
    if cycle:
        return ElleReport(
            serializable=False,
            anomaly=_classify(kinds),
            cycle=cycle,
            cycle_kinds=kinds,
            n_txns=len(nodes),
            edges=edges,
        )
    return ElleReport(serializable=True, anomaly="", n_txns=len(nodes), edges=edges)


# --- the version oracle: list-append / value-recoverable histories (DS3 increment 3) --------------


@dataclass(frozen=True)
class AppendObservation:
    """One committed transaction over a **list-append** register: what it appended and what it read.

    ``appends`` maps each key the transaction wrote to the **value it appended** (each value
    globally unique per key — the list-append datatype's contract). ``list_reads`` maps each key it
    read to the **whole observed list** of values, in list order. Unlike :class:`TxnObservation` no
    integer version appears: the version order is *recovered* from the read lists
    (:func:`recover_versions`), which lets Elle check a true black box that never exposes versions.
    """

    txn_id: str
    appends: tuple[tuple[str, str], ...] = ()
    list_reads: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class RecoveredOrder:
    """The version oracle's output: the recovered per-key append order, or the blocking anomaly.

    ``order`` maps each key to its recovered total append order (``order[k][i]`` is the value at
    version ``i+1``; version ``0`` is the empty boot list). ``anomaly`` is empty when recovery
    succeeded; otherwise it names the class (``incompatible-order`` / ``dirty-read`` /
    ``duplicate-write``) and ``detail`` is a one-line witness. ``ok`` is the success predicate.
    """

    order: dict[str, list[str]]
    anomaly: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.anomaly


def _merge_prefixes(reads: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    """Merge list-append reads of one key into the maximal list they are all prefixes of.

    Every read of a list-append register returns a *prefix* of the single growing append log, so the
    reads of one key must form a chain under the prefix relation. Returns the longest such list, or
    ``None`` if any two reads diverge (neither is a prefix of the other — a fork / split-brain).
    """
    longest: tuple[str, ...] = ()
    for lst in reads:
        if len(lst) > len(longest):
            if longest != lst[: len(longest)]:  # the shorter incumbent must be a prefix of the new
                return None
            longest = lst
        elif lst != longest[: len(lst)]:  # the new (shorter-or-equal) must be a prefix of longest
            return None
    return longest


def recover_versions(history: list[AppendObservation]) -> RecoveredOrder:
    """Recover each key's append (version) order from the observed list-reads alone — no store.

    The version oracle (Kingsbury & Alvaro 2020): per key, merge every observed read-list into the
    one total order they are all prefixes of, then place any appended-but-never-read value after
    that prefix (deterministic value-sorted tiebreak — its position relative to the read prefix is
    fixed, relative to its unread siblings is unconstrained). Surfaces the recovery anomalies the
    integer-version mode cannot represent: a **fork** (``incompatible-order``), a read of an
    uncommitted value (``dirty-read``), or a duplicated value (``duplicate-write``). Deterministic.
    """
    appends_by_key: dict[str, list[str]] = {}
    reads_by_key: dict[str, list[tuple[str, ...]]] = {}
    for t in sorted(history, key=lambda t: t.txn_id):
        for k, v in t.appends:
            appends_by_key.setdefault(k, []).append(v)
        for k, lst in t.list_reads:
            reads_by_key.setdefault(k, []).append(tuple(lst))

    order: dict[str, list[str]] = {}
    for key in sorted(set(appends_by_key) | set(reads_by_key)):
        appended = appends_by_key.get(key, [])
        appended_set = set(appended)
        if len(appended) != len(appended_set):
            return RecoveredOrder(order, "duplicate-write", f"key {key}: a value appended twice")
        reads = reads_by_key.get(key, [])
        for lst in reads:
            if len(set(lst)) != len(lst):
                return RecoveredOrder(order, "duplicate-write", f"key {key}: value read twice")
            for v in lst:
                if v not in appended_set:
                    return RecoveredOrder(order, "dirty-read", f"key {key}: read uncommitted {v!r}")
        merged = _merge_prefixes(reads)
        if merged is None:
            return RecoveredOrder(order, "incompatible-order", f"key {key}: reads fork on order")
        merged_set = set(merged)
        unread = sorted(v for v in appended if v not in merged_set)
        order[key] = list(merged) + unread
    return RecoveredOrder(order)


def appends_to_version_history(
    history: list[AppendObservation], recovered: RecoveredOrder
) -> list[TxnObservation]:
    """Map a list-append history onto the integer-version :class:`TxnObservation` form.

    Using the recovered order, every appended value becomes ``(key, version)`` (its 1-based
    position) and every read of a length-``L`` list becomes ``(key, L)`` (it observed the
    version-``L`` prefix). The result feeds the unchanged :func:`build_dsg` /
    :func:`check_serializable` machinery, so value-recovery and version-supplied agree by build.
    """
    index = {k: {v: i + 1 for i, v in enumerate(seq)} for k, seq in recovered.order.items()}
    out: list[TxnObservation] = []
    for t in history:
        writes = tuple(sorted((k, index[k][v]) for k, v in t.appends))
        reads = tuple(sorted((k, len(lst)) for k, lst in t.list_reads))
        out.append(TxnObservation(t.txn_id, reads=reads, writes=writes))
    return out


def check_serializable_appends(history: list[AppendObservation]) -> ElleReport:
    """Elle's verdict over a **black-box list-append history** — values only, no supplied versions.

    Recovers the per-key version order from the read values (:func:`recover_versions`); a recovery
    anomaly (fork / dirty-read / duplicate-write) is itself a non-serializable verdict reported
    before any cycle search. Otherwise the recovered version history is checked for a DSG cycle
    exactly as :func:`check_serializable` does — clean schedules make the two modes agree by build.
    """
    recovered = recover_versions(history)
    if not recovered.ok:
        return ElleReport(
            serializable=False, anomaly=recovered.anomaly, n_txns=len({t.txn_id for t in history})
        )
    return check_serializable(appends_to_version_history(history, recovered))
