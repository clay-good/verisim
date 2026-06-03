"""Torch-free featurization of a ``HostState`` into a typed interaction graph (SPEC-6 §6.1, HC4).

The factored, interaction-graph arm of ``M_θ`` (the DD-H1 alternative the flat arm is the floor for)
operates on the host bundle *as a graph of its objects* rather than on a serialized token stream.
This module is the deterministic, dependency-free bridge: it turns a
:class:`~verisim.host.state.HostState` and the conditioning :class:`~verisim.host.action.HostAction`
into a :class:`HostGraph` of plain Python numbers -- process nodes with feature vectors, the two
**interaction edge sets** the composition law (H13) is about, and graph-level action features --
which the torch GNN then consumes.

The factorization SPEC-6 §2.3 / DD-H1 prescribes is *the process table is the spine and the other
subsystems hang off it through references*, so the graph is **process-indexed** (node ``i`` == pid
``i``, with a validity mask) and carries two interaction edge sets:

  - **lineage** -- a directed ``ppid -> pid`` edge per process (the fork tree), so a syscall on a
    descendant can attend to its ancestor;
  - **shared-file** -- an undirected edge between two processes that hold an fd to the *same* path
    (the cross-subsystem coupling the flat serializer flattens away -- exactly the structure H13
    found load-bearing).

Per-process node features fold in the fd subsystem (which paths the process has open) and the
credential/exit state, so the model sees the bundle's coupling, not four independent subsystems.
Keeping the featurization torch-free and here (not in the model) follows the HC0-HC3 discipline: the
deterministic core is testable with no GPU before any weights touch it. ``symlog`` (DreamerV3, §6.1)
compresses the unbounded counters (open-fd count, the pid allocator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from verisim.host.action import HostAction
from verisim.host.config import HostConfig
from verisim.host.state import RUNNING, ZOMBIE, HostState

# The §3.2 syscall grammar, in a fixed canonical order. The action-type one-hot indexes here.
ACTION_NAMES: tuple[str, ...] = ("fork", "exit", "setuid", "open", "write", "close")
_ACTION_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}
_EXITS: tuple[int, ...] = (0, 1)  # the oracle's exit/return-code pool (EXIT_OK / EXIT_ERR)


def symlog(x: float) -> float:
    """Symmetric log (DreamerV3): ``sign(x)*log(1+|x|)``. Identity-ish near 0, log-ish far."""
    return math.copysign(math.log1p(abs(x)), x)


@dataclass(frozen=True)
class FeatureDims:
    """The exact widths of each feature block, so the torch model can size itself (§6.1)."""

    n_pids: int  # node count == max_pid + 1
    n_uids: int
    n_paths: int
    n_content: int

    @property
    def node(self) -> int:
        # exists(1) running(1) zombie(1) uid-onehot(U) exit-onehot(2) n_fds-symlog(1)
        #   is-acting(1) open-paths-multihot(P)
        return 1 + 1 + 1 + self.n_uids + len(_EXITS) + 1 + 1 + self.n_paths

    @property
    def graph(self) -> int:
        # action-onehot(A) arg-path-onehot(P) arg-uid-onehot(U) arg-content-onehot(C)
        #   arg-fd-symlog(1) arg-exit-onehot(2) last-exit-onehot(2) next-pid-symlog(1)
        return (
            len(ACTION_NAMES) + self.n_paths + self.n_uids + self.n_content
            + 1 + len(_EXITS) + len(_EXITS) + 1
        )


def feature_dims(config: HostConfig, max_pid: int) -> FeatureDims:
    return FeatureDims(
        n_pids=max_pid + 1,
        n_uids=len(config.uids),
        n_paths=len(config.paths),
        n_content=len(config.content_tokens),
    )


@dataclass(frozen=True)
class HostGraph:
    """A host bundle state featurized as a process-interaction graph (§6.1). Plain floats/ints.

    - ``node_features[i]``: feature vector for pid ``i`` (width ``dims.node``); invalid pids are the
      zero vector (``exists`` == 0). ``node_mask[i]`` is ``1.0`` iff pid ``i`` is a live entry.
    - ``lineage_edges``: directed ``(ppid, pid)`` index pairs (the fork tree).
    - ``share_edges``: undirected ``(i, j)`` (``i < j``) pairs of processes sharing an open path.
    - ``acting_pid``: the syscall's acting process index (for the decoder's per-object condition).
    - ``graph_features``: action + args + last-exit + allocator conditioning (width ``dims.graph``).
    """

    n_pids: int
    node_features: tuple[tuple[float, ...], ...]
    node_mask: tuple[float, ...]
    lineage_edges: tuple[tuple[int, int], ...]
    share_edges: tuple[tuple[int, int], ...]
    acting_pid: int
    graph_features: tuple[float, ...]
    dims: FeatureDims


def _onehot(index: int | None, width: int) -> list[float]:
    vec = [0.0] * width
    if index is not None and 0 <= index < width:
        vec[index] = 1.0
    return vec


def build_host_graph(
    state: HostState, action: HostAction | None, config: HostConfig, max_pid: int
) -> HostGraph:
    """Featurize ``(state, action)`` into a :class:`HostGraph` over a ``max_pid``-bounded world.

    ``action`` may be ``None`` (encode a state with no pending syscall); then the action one-hot and
    every arg feature are zero and ``acting_pid`` is ``0``.
    """
    dims = feature_dims(config, max_pid)
    n = dims.n_pids
    uid_index = {u: i for i, u in enumerate(config.uids)}
    path_index = {p: i for i, p in enumerate(config.paths)}
    content_index = {t: i for i, t in enumerate(config.content_tokens)}
    exit_index = {e: i for i, e in enumerate(_EXITS)}

    acting_pid = action.pid if action is not None and 0 <= action.pid < n else 0

    # Which paths each process has open (the fd subsystem folded into the process node).
    open_paths: dict[int, set[int]] = {}
    for (pid, _fd), entry in state.fds.items():
        if 0 <= pid < n and entry.path in path_index:
            open_paths.setdefault(pid, set()).add(path_index[entry.path])
    n_fds: dict[int, int] = {}
    for (pid, _fd) in state.fds:
        if 0 <= pid < n:
            n_fds[pid] = n_fds.get(pid, 0) + 1

    # --- per-process node features ----------------------------------------------
    node_features: list[tuple[float, ...]] = []
    node_mask: list[float] = []
    for pid in range(n):
        proc = state.procs.get(pid)
        feats: list[float] = []
        feats.append(1.0 if proc is not None else 0.0)
        feats.append(1.0 if (proc is not None and proc.state == RUNNING) else 0.0)
        feats.append(1.0 if (proc is not None and proc.state == ZOMBIE) else 0.0)
        feats.extend(_onehot(uid_index.get(proc.uid) if proc is not None else None, dims.n_uids))
        exit_idx: int | None = None
        if proc is not None and proc.exit_code is not None:
            exit_idx = exit_index.get(proc.exit_code)
        feats.extend(_onehot(exit_idx, len(_EXITS)))
        feats.append(symlog(float(n_fds.get(pid, 0))))
        feats.append(1.0 if (action is not None and pid == acting_pid) else 0.0)
        opened = open_paths.get(pid, set())
        feats.extend(1.0 if p in opened else 0.0 for p in range(dims.n_paths))
        assert len(feats) == dims.node
        node_features.append(tuple(feats))
        node_mask.append(1.0 if proc is not None else 0.0)

    # --- interaction edges ------------------------------------------------------
    lineage_edges = tuple(
        sorted(
            (proc.ppid, proc.pid)
            for proc in state.procs.values()
            if 0 <= proc.ppid < n and 0 <= proc.pid < n
        )
    )
    by_path: dict[int, list[int]] = {}
    for pid, paths in open_paths.items():
        for p in paths:
            by_path.setdefault(p, []).append(pid)
    share: set[tuple[int, int]] = set()
    for pids in by_path.values():
        sp = sorted(set(pids))
        for a_i in range(len(sp)):
            for b_i in range(a_i + 1, len(sp)):
                share.add((sp[a_i], sp[b_i]))
    share_edges = tuple(sorted(share))

    # --- graph-level action / arg / allocator features --------------------------
    g: list[float] = []
    act_idx = _ACTION_INDEX.get(action.name) if action is not None else None
    g.extend(_onehot(act_idx, len(ACTION_NAMES)))
    arg_path: int | None = None
    arg_uid: int | None = None
    arg_content: int | None = None
    arg_exit: int | None = None
    arg_fd_val = 0.0
    if action is not None:
        if action.name == "open" and action.args:
            arg_path = path_index.get(action.args[0])
        elif action.name == "setuid" and action.args:
            try:
                arg_uid = uid_index.get(int(action.args[0]))
            except ValueError:
                arg_uid = None
        elif action.name == "write" and len(action.args) >= 2:
            try:
                arg_fd_val = float(int(action.args[0]))
            except ValueError:
                arg_fd_val = 0.0
            arg_content = content_index.get(action.args[1])
        elif action.name == "close" and action.args:
            try:
                arg_fd_val = float(int(action.args[0]))
            except ValueError:
                arg_fd_val = 0.0
        elif action.name == "exit" and action.args:
            try:
                arg_exit = exit_index.get(int(action.args[0]))
            except ValueError:
                arg_exit = None
    g.extend(_onehot(arg_path, dims.n_paths))
    g.extend(_onehot(arg_uid, dims.n_uids))
    g.extend(_onehot(arg_content, dims.n_content))
    g.append(symlog(arg_fd_val))
    g.extend(_onehot(arg_exit, len(_EXITS)))
    g.extend(_onehot(exit_index.get(state.last_exit), len(_EXITS)))
    g.append(symlog(float(state.next_pid)))
    assert len(g) == dims.graph

    return HostGraph(
        n_pids=n,
        node_features=tuple(node_features),
        node_mask=tuple(node_mask),
        lineage_edges=lineage_edges,
        share_edges=share_edges,
        acting_pid=acting_pid,
        graph_features=tuple(g),
        dims=dims,
    )
