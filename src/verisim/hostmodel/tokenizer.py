"""Serialize host bundle states, syscalls, and bundle deltas to/from the closed token vocab.

The model maps ``<bos> serialize(state) serialize(action) <gen>`` -> ``serialize(Δ) <eos>``
(SPEC-6 §6.1, the flat HC4 arm), mirroring v0's :mod:`verisim.model.tokenizer` and the network
:mod:`verisim.netmodel.tokenizer`.

Only the **delta target** needs a decode grammar (see :mod:`grammar`); the prompt is the model
*input* we encode ourselves, so its serialization is free to be any deterministic scheme. The
embedded v0 filesystem is serialized as its *files* (path + content token), the only fs shape a
top-level write produces (§5.1); ``next_pid`` **is** serialized -- a ``fork`` predicts
``ProcSpawn(pid=next_pid)``, so the model must be able to copy it (the analogue of why the network
prompt omits the unbounded clock but the host prompt keeps the bounded pid allocator).

The embedded FS write delta is flattened, not nested: an :class:`~verisim.host.delta.FsDelta` over a
top-level path is always ``[Create|Modify(path, content), SetResult(exit, "")]`` (§5.1), so it
round-trips exactly from ``(op, path, content, exit)`` -- the create node's defaults and the empty
stdout hash are reconstructed verbatim. Nesting the *whole* v0 delta vocabulary is the DD-H1
*factored* arm (a later HC4 increment); this flat arm is the baseline it must beat.
"""

from __future__ import annotations

from verisim.delta.edits import Create, Modify, SetResult
from verisim.env.state import EMPTY_HASH, File
from verisim.host.action import HostAction
from verisim.host.delta import (
    CredChange,
    FdClose,
    FdOpen,
    FsDelta,
    HostDelta,
    HostEdit,
    ProcExit,
    ProcSpawn,
    SetExit,
)
from verisim.host.state import RUNNING, HostState

from .vocab import HostVocab


class HostTokenizeError(ValueError):
    """Raised when a value or token sequence is not encodable/parseable."""


# --- leaf encoders (each raises if a value falls outside the closed pool) ----


def _pid_id(pid: int, vocab: HostVocab) -> int:
    if pid not in vocab.pid_to_id:
        raise HostTokenizeError(f"pid {pid} exceeds the vocab pool (max_pid={vocab.max_pid})")
    return vocab.pid_to_id[pid]


def _fd_id(fd: int, vocab: HostVocab) -> int:
    if fd not in vocab.fd_to_id:
        raise HostTokenizeError(f"fd {fd} exceeds the vocab pool (max_fd={vocab.max_fd})")
    return vocab.fd_to_id[fd]


def _uid_id(uid: int, vocab: HostVocab) -> int:
    if uid not in vocab.uid_to_id:
        raise HostTokenizeError(f"uid {uid} is not in the config uid pool")
    return vocab.uid_to_id[uid]


def _path_id(path: str, vocab: HostVocab) -> int:
    if path not in vocab.path_to_id:
        raise HostTokenizeError(f"path {path!r} is not in the config path pool")
    return vocab.path_to_id[path]


def _content_id(token: str, vocab: HostVocab) -> int:
    if token not in vocab.content_to_id:
        raise HostTokenizeError(f"content token {token!r} is not in the config pool")
    return vocab.content_to_id[token]


def _exit_id(code: int, vocab: HostVocab) -> int:
    if code not in vocab.exit_to_id:
        raise HostTokenizeError(f"exit/return code {code} is not in {{0, 1}}")
    return vocab.exit_to_id[code]


# --- state / action encoders ------------------------------------------------


def encode_state(state: HostState, vocab: HostVocab) -> list[int]:
    """The bundle state: process table, fd table, embedded fs files, allocator, last result."""
    ids = [vocab.id("<state>")]
    for pid in sorted(state.procs):
        p = state.procs[pid]
        ids += [vocab.id("<proc>"), _pid_id(p.pid, vocab), _pid_id(p.ppid, vocab)]
        if p.state == RUNNING:
            ids.append(vocab.id("<running>"))
        else:
            ids += [vocab.id("<zombie>"), _exit_id(p.exit_code or 0, vocab)]
        ids.append(_uid_id(p.uid, vocab))
    for (pid, fd) in sorted(state.fds):
        ids += [vocab.id("<fd>"), _pid_id(pid, vocab), _fd_id(fd, vocab)]
        ids.append(_path_id(state.fds[(pid, fd)].path, vocab))
    for path in sorted(state.fs.fs):
        node = state.fs.fs[path]
        if isinstance(node, File):
            ids += [vocab.id("<fsnode>"), _path_id(path, vocab), _content_id(node.content, vocab)]
    ids += [vocab.id("<next>"), _pid_id(state.next_pid, vocab)]
    ids += [vocab.id("<last>"), _exit_id(state.last_exit, vocab)]
    return ids


def encode_action(action: HostAction, vocab: HostVocab) -> list[int]:
    """A syscall: the command token, the acting pid, then the per-command argument leaves."""
    ids = [vocab.id("<action>"), vocab.command_token_id(action.name), _pid_id(action.pid, vocab)]
    name = action.name
    if name == "fork":
        return ids
    if name == "exit":
        ids.append(_exit_id(int(action.args[0]), vocab))
    elif name == "setuid":
        ids.append(_uid_id(int(action.args[0]), vocab))
    elif name == "open":
        ids.append(_path_id(action.args[0], vocab))
    elif name == "write":
        ids += [_fd_id(int(action.args[0]), vocab), _content_id(action.args[1], vocab)]
    elif name == "close":
        ids.append(_fd_id(int(action.args[0]), vocab))
    else:  # pragma: no cover - parser already rejects unknown syscalls
        raise HostTokenizeError(f"cannot encode action {name!r}")
    return ids


def encode_prompt(state: HostState, action: HostAction, vocab: HostVocab) -> list[int]:
    """The model input: ``<bos> state action <gen>``."""
    return [vocab.bos, *encode_state(state, vocab), *encode_action(action, vocab), vocab.gen]


# --- delta (target) encoder -------------------------------------------------


def _fs_edit_ids(edit: FsDelta, vocab: HostVocab) -> list[int]:
    """Flatten the embedded FS write delta to ``<fs_create>|<fs_modify> path content exit``."""
    create_or_modify: Create | Modify | None = None
    exit_code = 0
    for sub in edit.edits:
        if isinstance(sub, (Create, Modify)):
            create_or_modify = sub
        elif isinstance(sub, SetResult):
            exit_code = sub.exit_code
    if create_or_modify is None:  # pragma: no cover - a write always creates or modifies
        raise HostTokenizeError("FsDelta has no Create/Modify edit to encode")
    if isinstance(create_or_modify, Create):
        node = create_or_modify.node
        if not isinstance(node, File):  # pragma: no cover - writes only create files
            raise HostTokenizeError("FsDelta create target is not a file")
        op, path, content = "<fs_create>", create_or_modify.path, node.content
    else:
        op, path, content = "<fs_modify>", create_or_modify.path, create_or_modify.content
    return [
        vocab.id(op), _path_id(path, vocab), _content_id(content, vocab), _exit_id(exit_code, vocab)
    ]


def _edit_ids(edit: HostEdit, vocab: HostVocab) -> list[int]:
    if isinstance(edit, ProcSpawn):
        return [
            vocab.id("<proc_spawn>"),
            _pid_id(edit.pid, vocab), _pid_id(edit.ppid, vocab), _uid_id(edit.uid, vocab),
        ]
    if isinstance(edit, ProcExit):
        return [vocab.id("<proc_exit>"), _pid_id(edit.pid, vocab), _exit_id(edit.code, vocab)]
    if isinstance(edit, FdOpen):
        return [
            vocab.id("<fd_open>"),
            _pid_id(edit.pid, vocab), _fd_id(edit.fd, vocab), _path_id(edit.path, vocab),
        ]
    if isinstance(edit, FdClose):
        return [vocab.id("<fd_close>"), _pid_id(edit.pid, vocab), _fd_id(edit.fd, vocab)]
    if isinstance(edit, CredChange):
        return [vocab.id("<cred_change>"), _pid_id(edit.pid, vocab), _uid_id(edit.uid, vocab)]
    if isinstance(edit, FsDelta):
        return _fs_edit_ids(edit, vocab)
    if isinstance(edit, SetExit):
        return [vocab.id("<set_exit>"), _exit_id(edit.exit_code, vocab)]
    # ``ProcReap`` (``wait``) and ``CwdChange`` (``chdir``) have no token here yet: the
    # learned-model coverage of the post-HC0-increment-1 syscalls is the deferred GPU arm (exactly
    # as the dist model covers only the base KV+fault ops), so the data factory never produces them.
    # Raise rather than silently mis-encoding them as ``<set_exit>``.
    raise ValueError(f"host edit {type(edit).__name__} has no model token (coverage deferred)")


def encode_target(delta: HostDelta, vocab: HostVocab) -> list[int]:
    """Encode a bundle delta as the model target (edits, then ``<eos>``)."""
    ids: list[int] = []
    for edit in delta:
        ids += _edit_ids(edit, vocab)
    ids.append(vocab.eos)
    return ids


# --- parser (inverse of encode_target) --------------------------------------


class _Cursor:
    def __init__(self, ids: list[int], vocab: HostVocab) -> None:
        self.ids = ids
        self.vocab = vocab
        self.i = 0

    def peek(self) -> int:
        if self.i >= len(self.ids):
            raise HostTokenizeError("unexpected end of token stream")
        return self.ids[self.i]

    def take(self) -> int:
        tok = self.peek()
        self.i += 1
        return tok


def _take_pid(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_pid:
        raise HostTokenizeError(f"expected a pid token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_pid[tok]


def _take_fd(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_fd:
        raise HostTokenizeError(f"expected an fd token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_fd[tok]


def _take_uid(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_uid:
        raise HostTokenizeError(f"expected a uid token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_uid[tok]


def _take_path(cur: _Cursor) -> str:
    tok = cur.take()
    if tok not in cur.vocab.id_to_path:
        raise HostTokenizeError(f"expected a path token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_path[tok]


def _take_content(cur: _Cursor) -> str:
    tok = cur.take()
    if tok not in cur.vocab.id_to_content:
        raise HostTokenizeError(f"expected a content token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_content[tok]


def _take_exit(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_exit:
        raise HostTokenizeError(f"expected an exit token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_exit[tok]


def _fs_delta(path: str, content: str, exit_code: int, *, create: bool) -> FsDelta:
    """Reconstruct the exact embedded v0 write delta (§5.1: a write's only two shapes)."""
    head: Create | Modify = (
        Create(path, File(content=content)) if create else Modify(path, content)
    )
    return FsDelta(edits=[head, SetResult(exit_code, EMPTY_HASH)])


def parse_target(ids: list[int], vocab: HostVocab) -> HostDelta:
    """Parse a target token sequence into a :class:`HostDelta`; raise if malformed."""
    cur = _Cursor(ids, vocab)
    delta: HostDelta = []
    while cur.peek() != vocab.eos:
        op = cur.take()
        if op == vocab.id("<proc_spawn>"):
            delta.append(ProcSpawn(pid=_take_pid(cur), ppid=_take_pid(cur), uid=_take_uid(cur)))
        elif op == vocab.id("<proc_exit>"):
            delta.append(ProcExit(pid=_take_pid(cur), code=_take_exit(cur)))
        elif op == vocab.id("<fd_open>"):
            delta.append(FdOpen(pid=_take_pid(cur), fd=_take_fd(cur), path=_take_path(cur)))
        elif op == vocab.id("<fd_close>"):
            delta.append(FdClose(pid=_take_pid(cur), fd=_take_fd(cur)))
        elif op == vocab.id("<cred_change>"):
            delta.append(CredChange(pid=_take_pid(cur), uid=_take_uid(cur)))
        elif op == vocab.id("<fs_create>"):
            path, content, code = _take_path(cur), _take_content(cur), _take_exit(cur)
            delta.append(_fs_delta(path, content, code, create=True))
        elif op == vocab.id("<fs_modify>"):
            path, content, code = _take_path(cur), _take_content(cur), _take_exit(cur)
            delta.append(_fs_delta(path, content, code, create=False))
        elif op == vocab.id("<set_exit>"):
            delta.append(SetExit(exit_code=_take_exit(cur)))
        else:
            raise HostTokenizeError(f"expected an edit op, got {vocab.token(op)!r}")
    return delta


__all__ = [
    "HostTokenizeError",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "parse_target",
]
