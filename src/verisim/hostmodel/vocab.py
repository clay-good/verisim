"""The closed token vocabulary for the host world model `M_θ` (SPEC-6 §6.1, HC4).

The host world (`docs/host-semantics.md`) draws its syscall arguments from a finite
:class:`~verisim.host.config.HostConfig` (paths, content tokens, uids), exactly as v0 fixes
content/mode/name pools and the network world fixes host-id/port pools. The two *unbounded*
identifier families -- process ids and file descriptors -- are bounded here by fixed pools
(``max_pid``/``max_fd``) sized past what a finite-length rollout can allocate (pids grow by one
per ``fork``, fds are the smallest-free index per process), so the entire serialization DSL --
bundle states, syscalls, and bundle deltas -- maps to a small *closed* set of tokens built
deterministically from a config. The flat HC4 arm (the DD-H1 baseline `M_θ` must beat) trains
over exactly these ids; the constrained decoder masks to them.

Token families:
  - specials: ``<pad> <bos> <eos>``;
  - section/structure markers: ``<state> <proc> <fd> <fsnode> <next> <last> <action> <gen>``
    plus the process-state markers ``<running> <zombie>``;
  - delta ops (one per :mod:`verisim.host.delta` edit, the embedded FS write split into
    ``<fs_create>``/``<fs_modify>`` -- the only two shapes a top-level write produces, §5.1);
  - command tokens (one per §3.2 syscall) and the five leaf vocabularies: pids, fds, uids,
    paths, content tokens, and exit/return codes ``0/1`` (the oracle's ``EXIT_OK``/``EXIT_ERR``).
"""

from __future__ import annotations

from verisim.host.config import HostConfig

_SPECIALS = ("<pad>", "<bos>", "<eos>")
_MARKERS = (
    "<state>", "<proc>", "<fd>", "<fsnode>", "<next>", "<last>", "<action>", "<gen>",
    "<running>", "<zombie>",
)
_OPS = (
    "<proc_spawn>", "<proc_exit>", "<fd_open>", "<fd_close>", "<cred_change>",
    "<fs_create>", "<fs_modify>", "<set_exit>",
)
# Syscalls (SPEC-6 §3.2). Each mirrors the action grammar's argument shape.
_COMMANDS = ("fork", "exit", "setuid", "open", "write", "close")
# Exit/return codes: the oracle's convention (0 == EXIT_OK, 1 == EXIT_ERR); process ``exit``
# codes the workload drivers emit are drawn from the same {0, 1} pool (SPEC-6 §3.2).
_EXITS = (0, 1)


class HostVocab:
    """A closed, deterministic token<->id mapping for one :class:`HostConfig`.

    ``max_pid``/``max_fd`` bound the otherwise-unbounded pid/fd id families; the defaults are
    sized for the HC2 trajectory lengths (a rollout forks at most once per step). ``__init__``
    raises if a config value would collide with the structural tokens.
    """

    def __init__(self, config: HostConfig, *, max_pid: int = 64, max_fd: int = 16) -> None:
        self.config = config
        self.max_pid = max_pid
        self.max_fd = max_fd
        tokens: list[str] = []
        tokens += _SPECIALS
        tokens += _MARKERS
        tokens += _OPS
        tokens += [f"<cmd:{c}>" for c in _COMMANDS]
        tokens += [f"<exit:{e}>" for e in _EXITS]
        tokens += [f"<pid:{i}>" for i in range(max_pid + 1)]  # 0 == ppid of the init process
        tokens += [f"<fd:{i}>" for i in range(max_fd + 1)]
        tokens += [f"<uid:{u}>" for u in config.uids]
        tokens += [f"<path:{p}>" for p in config.paths]
        tokens += [f"<c:{t}>" for t in config.content_tokens]

        self._tokens = tuple(tokens)
        self._id = {tok: i for i, tok in enumerate(self._tokens)}

        # Leaf lookup tables (token id <-> domain value).
        self.pid_to_id = {i: self._id[f"<pid:{i}>"] for i in range(max_pid + 1)}
        self.id_to_pid = {v: k for k, v in self.pid_to_id.items()}
        self.fd_to_id = {i: self._id[f"<fd:{i}>"] for i in range(max_fd + 1)}
        self.id_to_fd = {v: k for k, v in self.fd_to_id.items()}
        self.uid_to_id = {u: self._id[f"<uid:{u}>"] for u in config.uids}
        self.id_to_uid = {v: k for k, v in self.uid_to_id.items()}
        self.path_to_id = {p: self._id[f"<path:{p}>"] for p in config.paths}
        self.id_to_path = {v: k for k, v in self.path_to_id.items()}
        self.content_to_id = {t: self._id[f"<c:{t}>"] for t in config.content_tokens}
        self.id_to_content = {v: k for k, v in self.content_to_id.items()}
        self.exit_to_id = {e: self._id[f"<exit:{e}>"] for e in _EXITS}
        self.id_to_exit = {v: k for k, v in self.exit_to_id.items()}

    # -- core mapping --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tokens)

    def id(self, token: str) -> int:
        return self._id[token]

    def token(self, token_id: int) -> str:
        return self._tokens[token_id]

    def command_token_id(self, name: str) -> int:
        return self._id[f"<cmd:{name}>"]

    # -- frequently used ids (named for readability) -------------------------

    @property
    def pad(self) -> int:
        return self._id["<pad>"]

    @property
    def bos(self) -> int:
        return self._id["<bos>"]

    @property
    def eos(self) -> int:
        return self._id["<eos>"]

    @property
    def gen(self) -> int:
        return self._id["<gen>"]

    # -- token-class sets (for the grammar / constrained decoder) ------------

    @property
    def op_ids(self) -> frozenset[int]:
        return frozenset(self._id[op] for op in _OPS)

    @property
    def pid_ids(self) -> frozenset[int]:
        return frozenset(self.pid_to_id.values())

    @property
    def fd_ids(self) -> frozenset[int]:
        return frozenset(self.fd_to_id.values())

    @property
    def uid_ids(self) -> frozenset[int]:
        return frozenset(self.uid_to_id.values())

    @property
    def path_ids(self) -> frozenset[int]:
        return frozenset(self.path_to_id.values())

    @property
    def content_ids(self) -> frozenset[int]:
        return frozenset(self.content_to_id.values())

    @property
    def exit_ids(self) -> frozenset[int]:
        return frozenset(self.exit_to_id.values())
