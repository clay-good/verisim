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
