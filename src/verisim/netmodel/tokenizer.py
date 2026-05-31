"""Serialize network states, actions, and graph deltas to/from the closed token vocab.

The model maps ``<bos> serialize(state) serialize(action) <gen>`` -> ``serialize(Δ) <eos>``
(SPEC-5 §6.1, the flat NW4 arm), mirroring v0's :mod:`verisim.model.tokenizer`.

Only the **delta target** needs a decode grammar (see :mod:`grammar`); the prompt is the
model *input* we encode ourselves, so its serialization is free to be any deterministic
scheme. The global ``clock`` is deliberately omitted from the prompt: it grows unbounded
under ``advance`` (which would break the closed vocabulary) and no transition depends on its
value -- ``advance`` always emits ``ClockAdvance(1)``. Flows *are* serialized, since the
reachability re-validation at ``advance`` reads them.
"""

from __future__ import annotations

from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.edits import (
    ClockAdvance,
    FlowClose,
    FlowOpen,
    FwAllow,
    FwDeny,
    HostDown,
    HostUp,
    LinkAdd,
    LinkDel,
    NetDelta,
    NetEdit,
    SetResult,
    SvcDown,
    SvcUp,
)

from .vocab import NetVocab


class NetTokenizeError(ValueError):
    """Raised when a value or token sequence is not encodable/parseable."""


# --- encoders ---------------------------------------------------------------


def _host_id(host: str, vocab: NetVocab) -> int:
    if host not in vocab.host_to_id:
        raise NetTokenizeError(f"host {host!r} is not in the config host pool")
    return vocab.host_to_id[host]


def _port_id(port: int, vocab: NetVocab) -> int:
    if port not in vocab.port_to_id:
        raise NetTokenizeError(f"port {port!r} is not in the config port pool")
    return vocab.port_to_id[port]


def encode_state(state: NetworkState, vocab: NetVocab) -> list[int]:
    ids = [vocab.id("<state>")]
    for host in sorted(state.hosts):
        hs = state.hosts[host]
        ids += [vocab.id("<host>"), _host_id(host, vocab)]
        ids.append(vocab.id("<up>") if hs.up else vocab.id("<down>"))
        for port in hs.services:
            ids += [vocab.id("<svc>"), _port_id(port, vocab)]
        for src in hs.fw_deny:
            ids += [vocab.id("<deny>"), _host_id(src, vocab)]
    ids.append(vocab.id("<links>"))
    for a, b in sorted(state.links):
        ids += [vocab.id("<link>"), _host_id(a, vocab), _host_id(b, vocab)]
    ids.append(vocab.id("<flows>"))
    for src, dst, port in sorted(state.flows):
        ids.append(vocab.id("<flow>"))
        ids += [_host_id(src, vocab), _host_id(dst, vocab), _port_id(port, vocab)]
    ids += [vocab.id("<last>"), vocab.exit_to_id[state.last_exit]]
    return ids


def encode_action(action: NetAction, vocab: NetVocab) -> list[int]:
    ids = [vocab.id("<action>"), vocab.command_token_id(action.name)]
    name = action.name
    if name == "advance":
        return ids
    if name in {"host_up", "host_down"}:
        ids.append(_host_id(action.args[0], vocab))
    elif name in {"link_up", "link_down", "fw_deny", "fw_allow"}:
        ids += [_host_id(action.args[0], vocab), _host_id(action.args[1], vocab)]
    elif name in {"svc_up", "svc_down"}:
        ids += [_host_id(action.args[0], vocab), _port_id(action.port, vocab)]
    elif name in {"connect", "close"}:
        ids += [
            _host_id(action.args[0], vocab),
            _host_id(action.args[1], vocab),
            _port_id(action.port, vocab),
        ]
    else:  # pragma: no cover - parser already rejects unknown commands
        raise NetTokenizeError(f"cannot encode action {name!r}")
    return ids


def _edit_ids(edit: NetEdit, vocab: NetVocab) -> list[int]:
    if isinstance(edit, HostUp):
        return [vocab.id("<host_up>"), _host_id(edit.host, vocab)]
    if isinstance(edit, HostDown):
        return [vocab.id("<host_down>"), _host_id(edit.host, vocab)]
    if isinstance(edit, LinkAdd):
        return [vocab.id("<link_add>"), _host_id(edit.a, vocab), _host_id(edit.b, vocab)]
    if isinstance(edit, LinkDel):
        return [vocab.id("<link_del>"), _host_id(edit.a, vocab), _host_id(edit.b, vocab)]
    if isinstance(edit, SvcUp):
        return [vocab.id("<svc_up>"), _host_id(edit.host, vocab), _port_id(edit.port, vocab)]
    if isinstance(edit, SvcDown):
        return [vocab.id("<svc_down>"), _host_id(edit.host, vocab), _port_id(edit.port, vocab)]
    if isinstance(edit, FwDeny):
        return [vocab.id("<fw_deny>"), _host_id(edit.host, vocab), _host_id(edit.src, vocab)]
    if isinstance(edit, FwAllow):
        return [vocab.id("<fw_allow>"), _host_id(edit.host, vocab), _host_id(edit.src, vocab)]
    if isinstance(edit, FlowOpen):
        return [
            vocab.id("<flow_open>"),
            _host_id(edit.src, vocab),
            _host_id(edit.dst, vocab),
            _port_id(edit.port, vocab),
        ]
    if isinstance(edit, FlowClose):
        return [
            vocab.id("<flow_close>"),
            _host_id(edit.src, vocab),
            _host_id(edit.dst, vocab),
            _port_id(edit.port, vocab),
        ]
    if isinstance(edit, ClockAdvance):
        return [vocab.id("<clock_advance>")]  # amount is always 1 in this world
    return [vocab.id("<set_result>"), vocab.exit_to_id[edit.exit_code]]


def encode_target(delta: NetDelta, vocab: NetVocab) -> list[int]:
    """Encode a graph delta as the model target (edits, then ``<eos>``)."""
    ids: list[int] = []
    for edit in delta:
        ids += _edit_ids(edit, vocab)
    ids.append(vocab.eos)
    return ids


def encode_prompt(state: NetworkState, action: NetAction, vocab: NetVocab) -> list[int]:
    """The model input: ``<bos> state action <gen>``."""
    return [vocab.bos, *encode_state(state, vocab), *encode_action(action, vocab), vocab.gen]


# --- parser (inverse of encode_target) --------------------------------------


class _Cursor:
    def __init__(self, ids: list[int], vocab: NetVocab) -> None:
        self.ids = ids
        self.vocab = vocab
        self.i = 0

    def peek(self) -> int:
        if self.i >= len(self.ids):
            raise NetTokenizeError("unexpected end of token stream")
        return self.ids[self.i]

    def take(self) -> int:
        tok = self.peek()
        self.i += 1
        return tok


def _take_host(cur: _Cursor) -> str:
    tok = cur.take()
    if tok not in cur.vocab.id_to_host:
        raise NetTokenizeError(f"expected a host token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_host[tok]


def _take_port(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_port:
        raise NetTokenizeError(f"expected a port token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_port[tok]


def parse_target(ids: list[int], vocab: NetVocab) -> NetDelta:
    """Parse a target token sequence into a :class:`NetDelta`; raise if malformed."""
    cur = _Cursor(ids, vocab)
    delta: NetDelta = []
    while cur.peek() != vocab.eos:
        op = cur.take()
        if op == vocab.id("<host_up>"):
            delta.append(HostUp(_take_host(cur)))
        elif op == vocab.id("<host_down>"):
            delta.append(HostDown(_take_host(cur)))
        elif op == vocab.id("<link_add>"):
            delta.append(LinkAdd(_take_host(cur), _take_host(cur)))
        elif op == vocab.id("<link_del>"):
            delta.append(LinkDel(_take_host(cur), _take_host(cur)))
        elif op == vocab.id("<svc_up>"):
            delta.append(SvcUp(_take_host(cur), _take_port(cur)))
        elif op == vocab.id("<svc_down>"):
            delta.append(SvcDown(_take_host(cur), _take_port(cur)))
        elif op == vocab.id("<fw_deny>"):
            delta.append(FwDeny(_take_host(cur), _take_host(cur)))
        elif op == vocab.id("<fw_allow>"):
            delta.append(FwAllow(_take_host(cur), _take_host(cur)))
        elif op == vocab.id("<flow_open>"):
            delta.append(FlowOpen(_take_host(cur), _take_host(cur), _take_port(cur)))
        elif op == vocab.id("<flow_close>"):
            delta.append(FlowClose(_take_host(cur), _take_host(cur), _take_port(cur)))
        elif op == vocab.id("<clock_advance>"):
            delta.append(ClockAdvance(1))
        elif op == vocab.id("<set_result>"):
            exit_tok = cur.take()
            if exit_tok not in vocab.id_to_exit:
                raise NetTokenizeError("malformed set_result exit code")
            delta.append(SetResult(vocab.id_to_exit[exit_tok]))
        else:
            raise NetTokenizeError(f"expected an edit op, got {vocab.token(op)!r}")
    return delta


__all__ = [
    "NetTokenizeError",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "parse_target",
]
