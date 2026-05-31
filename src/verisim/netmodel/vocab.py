"""The closed token vocabulary for the network world model `M_θ` (SPEC-5 §6.1).

The network world (`docs/network-semantics.md`) fixes finite host-id and port pools in
:class:`~verisim.net.config.NetConfig`, exactly as v0 fixes content/mode/name pools, so the
entire serialization DSL -- states, actions, and graph deltas -- maps to a small *closed*
set of tokens built deterministically from a config. A from-scratch tiny transformer (the
flat NW4 arm) trains over exactly these ids; the constrained decoder masks to them.

Token families:
  - specials: ``<pad> <bos> <eos>``;
  - section markers: ``<state> <host> <up> <down> <svc> <deny> <links> <link> <flows>
    <flow> <last> <action> <gen>``;
  - delta ops (one per :mod:`verisim.netdelta.edits` type);
  - command tokens (one per §3.2 command), exit codes ``0/1/2``, and the two leaf
    vocabularies (host ids, ports).
"""

from __future__ import annotations

from verisim.net.config import NetConfig

_SPECIALS = ("<pad>", "<bos>", "<eos>")
_MARKERS = (
    "<state>", "<host>", "<up>", "<down>", "<svc>", "<deny>",
    "<links>", "<link>", "<flows>", "<flow>", "<last>", "<action>", "<gen>",
)
_OPS = (
    "<host_up>", "<host_down>", "<link_add>", "<link_del>",
    "<svc_up>", "<svc_down>", "<fw_deny>", "<fw_allow>",
    "<flow_open>", "<flow_close>", "<clock_advance>", "<set_result>",
)
# Commands (SPEC-5 §3.2). ``advance`` has no args; the rest mirror the action grammar.
_COMMANDS = (
    "host_up", "host_down", "link_up", "link_down", "svc_up", "svc_down",
    "fw_deny", "fw_allow", "connect", "close", "advance",
)
_EXITS = (0, 1, 2)  # the oracle's exit convention (0 ok, 1 runtime fail, 2 bad args)


class NetVocab:
    """A closed, deterministic token<->id mapping for one :class:`NetConfig`."""

    def __init__(self, config: NetConfig) -> None:
        self.config = config
        tokens: list[str] = []
        tokens += _SPECIALS
        tokens += _MARKERS
        tokens += _OPS
        tokens += [f"<cmd:{c}>" for c in _COMMANDS]
        tokens += [f"<exit:{e}>" for e in _EXITS]
        tokens += [f"<h:{h}>" for h in config.hosts]
        tokens += [f"<port:{p}>" for p in config.ports]

        self._tokens = tuple(tokens)
        self._id = {tok: i for i, tok in enumerate(self._tokens)}

        # Leaf lookup tables (token id <-> domain value).
        self.host_to_id = {h: self._id[f"<h:{h}>"] for h in config.hosts}
        self.id_to_host = {v: k for k, v in self.host_to_id.items()}
        self.port_to_id = {p: self._id[f"<port:{p}>"] for p in config.ports}
        self.id_to_port = {v: k for k, v in self.port_to_id.items()}
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
    def host_ids(self) -> frozenset[int]:
        return frozenset(self.host_to_id.values())

    @property
    def port_ids(self) -> frozenset[int]:
        return frozenset(self.port_to_id.values())

    @property
    def exit_ids(self) -> frozenset[int]:
        return frozenset(self.exit_to_id.values())
