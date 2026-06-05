"""Serialize distributed states, actions, and log/replica deltas to/from the closed token vocab.

The model maps ``<bos> serialize(state) serialize(action) <gen>`` -> ``serialize(Δ) <eos>``
(SPEC-7 §6.1, the flat DS4 arm), mirroring v0's :mod:`verisim.model.tokenizer` and the network /
host tokenizers.

Only the **delta target** needs a decode grammar (see :mod:`grammar`, DS4 increment 2); the prompt
is the model *input* we encode ourselves, so its serialization is any deterministic scheme. The
prompt carries everything a legal next-delta depends on: the per-replica ``(version, value)`` (a
write bumps the coordinator's version, ``advance`` adopts an in-flight version under
last-writer-wins), the partition groups and ``down`` set (they gate message delivery), the in-flight
messages (``advance`` delivers the due+reachable ones), the ``clock`` (``advance`` sets it forward)
and ``next_msg_id`` (a write's outgoing messages are numbered from it). The causal log is
**omitted**: the only delta field it feeds -- ``happens_before`` -- is
rebuilt deterministically by :func:`parse_target` (handed the state), never predicted, so the model
never needs the log to produce a correct target.

The ``EventAppend`` edit is encoded as its bare ``<event_append>`` marker; :func:`parse_target`
rebuilds the whole edit from ``(state, action)`` (``id = next_event_id``, ``node`` the coordinator,
``op`` = ``action.raw``, ``clock`` = the current clock, ``happens_before`` = the prior same-node log
events) -- the distributed analogue of the network tokenizer reconstructing the always-1
``ClockAdvance``. Every other edit's arguments are real predictions and are encoded as tokens.
"""

from __future__ import annotations

from verisim.dist.action import DistAction
from verisim.dist.delta import (
    ClockSet,
    DistDelta,
    DistEdit,
    EventAppend,
    MsgDeliver,
    MsgDrop,
    MsgSend,
    NodeDown,
    NodeUp,
    PartitionSet,
    ReplicaWrite,
    SetResult,
)
from verisim.dist.state import DistributedState

from .vocab import DistVocab


class DistTokenizeError(ValueError):
    """Raised when a value or token sequence is not encodable/parseable."""


# --- leaf encoders ----------------------------------------------------------


def _node_id(node: str, vocab: DistVocab) -> int:
    if node not in vocab.node_to_id:
        raise DistTokenizeError(f"node {node!r} is not in the config node pool")
    return vocab.node_to_id[node]


def _obj_id(obj: str, vocab: DistVocab) -> int:
    if obj not in vocab.obj_to_id:
        raise DistTokenizeError(f"object {obj!r} is not in the config object pool")
    return vocab.obj_to_id[obj]


def _val_id(value: str, vocab: DistVocab) -> int:
    if value not in vocab.val_to_id:
        raise DistTokenizeError(f"value {value!r} is not in the config value pool")
    return vocab.val_to_id[value]


def _int_id(n: int, vocab: DistVocab) -> int:
    if not 0 <= n < vocab.max_int:
        raise DistTokenizeError(
            f"integer {n} out of the vocab pool [0, {vocab.max_int}); raise max_int"
        )
    return vocab.int_to_id[n]


def _result_value_ids(status: str, value: str, vocab: DistVocab) -> list[int]:
    """Encode a ``SetResult`` value, dispatched by status (``advanced`` carries an int count)."""
    if status == "advanced":
        return [_int_id(int(value), vocab)]
    return [_val_id(value, vocab)]


# --- prompt (model input; free-form, no grammar) ----------------------------


def encode_state(state: DistributedState, vocab: DistVocab) -> list[int]:
    ids = [vocab.id("<state>")]
    for (obj, node) in sorted(state.replicas):
        r = state.replicas[(obj, node)]
        ids += [vocab.id("<replica>"), _obj_id(obj, vocab), _node_id(node, vocab),
                _int_id(r.version, vocab), _val_id(r.value, vocab)]
    ids.append(vocab.id("<parts>"))
    for group in state.partitions:
        ids.append(vocab.id("<pgroup>"))
        ids += [_node_id(n, vocab) for n in sorted(group)]
    ids.append(vocab.id("<down>"))
    ids += [_node_id(n, vocab) for n in sorted(state.down)]
    ids.append(vocab.id("<inflight>"))
    for msg_id in sorted(state.inflight):
        m = state.inflight[msg_id]
        ids += [vocab.id("<msg>"), _int_id(m.id, vocab), _node_id(m.src, vocab),
                _node_id(m.dst, vocab), _obj_id(m.object_id, vocab), _int_id(m.version, vocab),
                _val_id(m.value, vocab), _int_id(m.deliver_after, vocab)]
    ids += [vocab.id("<clock>"), _int_id(state.clock, vocab)]
    ids += [vocab.id("<nextids>"), _int_id(state.next_event_id, vocab),
            _int_id(state.next_msg_id, vocab)]
    ids.append(vocab.id("<last>"))
    if state.last_result is None:
        ids.append(vocab.id("<none>"))
    else:
        status, value = state.last_result
        ids += [vocab.status_to_id[status], *_result_value_ids(status, value, vocab)]
    return ids


def encode_action(action: DistAction, vocab: DistVocab) -> list[int]:
    ids = [vocab.id("<action>"), vocab.command_token_id(action.name)]
    name = action.name
    if name == "heal":
        return ids
    if name == "advance":
        ids.append(_int_id(int(action.args[0]), vocab))
    elif name in ("crash", "restart"):
        ids.append(_node_id(action.args[0], vocab))
    elif name == "get":
        ids += [_node_id(action.args[0], vocab), _obj_id(action.args[1], vocab)]
    elif name == "put":
        ids += [_node_id(action.args[0], vocab), _obj_id(action.args[1], vocab),
                _val_id(action.args[2], vocab)]
    elif name == "cas":
        ids += [_node_id(action.args[0], vocab), _obj_id(action.args[1], vocab),
                _val_id(action.args[2], vocab), _val_id(action.args[3], vocab)]
    elif name == "partition":
        for group in action.groups:
            ids.append(vocab.id("<pgroup>"))
            ids += [_node_id(n, vocab) for n in group]
        ids.append(vocab.id("<pgroups_end>"))
    else:  # pragma: no cover - parser already rejects unknown commands
        raise DistTokenizeError(f"cannot encode action {name!r}")
    return ids


def encode_prompt(state: DistributedState, action: DistAction, vocab: DistVocab) -> list[int]:
    """The model input: ``<bos> state action <gen>``."""
    return [vocab.bos, *encode_state(state, vocab), *encode_action(action, vocab), vocab.gen]


# --- target (the graph/log delta; the only part with a decode grammar) ------


def _edit_ids(edit: DistEdit, vocab: DistVocab) -> list[int]:
    if isinstance(edit, ReplicaWrite):
        return [vocab.id("<replica_write>"), _obj_id(edit.object_id, vocab),
                _node_id(edit.node_id, vocab), _int_id(edit.version, vocab),
                _val_id(edit.value, vocab)]
    if isinstance(edit, MsgSend):
        return [vocab.id("<msg_send>"), _int_id(edit.msg_id, vocab), _node_id(edit.src, vocab),
                _node_id(edit.dst, vocab), _obj_id(edit.object_id, vocab),
                _int_id(edit.version, vocab), _val_id(edit.value, vocab),
                _int_id(edit.deliver_after, vocab)]
    if isinstance(edit, MsgDeliver):
        return [vocab.id("<msg_deliver>"), _int_id(edit.msg_id, vocab)]
    if isinstance(edit, MsgDrop):
        return [vocab.id("<msg_drop>"), _int_id(edit.msg_id, vocab)]
    if isinstance(edit, EventAppend):
        return [vocab.id("<event_append>")]  # content reconstructed from (state, action) on parse
    if isinstance(edit, PartitionSet):
        ids = [vocab.id("<partition_set>")]
        for group in edit.groups:
            ids.append(vocab.id("<pgroup>"))
            ids += [_node_id(n, vocab) for n in group]
        ids.append(vocab.id("<pgroups_end>"))
        return ids
    if isinstance(edit, NodeDown):
        return [vocab.id("<node_down>"), _node_id(edit.node, vocab)]
    if isinstance(edit, NodeUp):
        return [vocab.id("<node_up>"), _node_id(edit.node, vocab)]
    if isinstance(edit, ClockSet):
        return [vocab.id("<clock_set>"), _int_id(edit.clock, vocab)]
    assert isinstance(edit, SetResult)
    return [vocab.id("<set_result>"), vocab.status_to_id[edit.status],
            *_result_value_ids(edit.status, edit.value, vocab)]


def encode_target(delta: DistDelta, vocab: DistVocab) -> list[int]:
    """Encode a log/replica delta as the model target (edits, then ``<eos>``)."""
    ids: list[int] = []
    for edit in delta:
        ids += _edit_ids(edit, vocab)
    ids.append(vocab.eos)
    return ids


# --- parser (inverse of encode_target) --------------------------------------


class _Cursor:
    def __init__(self, ids: list[int], vocab: DistVocab) -> None:
        self.ids = ids
        self.vocab = vocab
        self.i = 0

    def peek(self) -> int:
        if self.i >= len(self.ids):
            raise DistTokenizeError("unexpected end of token stream")
        return self.ids[self.i]

    def take(self) -> int:
        tok = self.peek()
        self.i += 1
        return tok


def _take_node(cur: _Cursor) -> str:
    tok = cur.take()
    if tok not in cur.vocab.id_to_node:
        raise DistTokenizeError(f"expected a node token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_node[tok]


def _take_obj(cur: _Cursor) -> str:
    tok = cur.take()
    if tok not in cur.vocab.id_to_obj:
        raise DistTokenizeError(f"expected an object token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_obj[tok]


def _take_val(cur: _Cursor) -> str:
    tok = cur.take()
    if tok not in cur.vocab.id_to_val:
        raise DistTokenizeError(f"expected a value token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_val[tok]


def _take_int(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_int:
        raise DistTokenizeError(f"expected an int token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_int[tok]


def _reconstruct_event(state: DistributedState, action: DistAction) -> EventAppend:
    """Rebuild the ``EventAppend`` for the current step from ``(state, action)`` (it is derivable).

    Mirrors :meth:`verisim.distoracle.reference.ReferenceDistOracle._event`: a client op appends one
    causal-log event at the coordinator node, with program-order ``happens_before`` over the prior
    same-node events.
    """
    node = action.args[0]
    prior = tuple(e.id for e in state.log if e.node == node)
    return EventAppend(state.next_event_id, node, action.raw, state.clock, prior)


def parse_target(
    ids: list[int], vocab: DistVocab, state: DistributedState, action: DistAction
) -> DistDelta:
    """Parse a target token sequence into a :class:`DistDelta`; raise if malformed.

    ``state`` and ``action`` are the step context: the bare ``<event_append>`` marker is expanded
    into the full (derivable) :class:`EventAppend` from them, exactly as the encoder dropped it.
    """
    v = vocab
    cur = _Cursor(ids, v)
    delta: DistDelta = []
    while cur.peek() != v.eos:
        op = cur.take()
        if op == v.id("<replica_write>"):
            obj, node = _take_obj(cur), _take_node(cur)
            delta.append(ReplicaWrite(obj, node, _take_int(cur), _take_val(cur)))
        elif op == v.id("<msg_send>"):
            msg_id = _take_int(cur)
            src, dst, obj = _take_node(cur), _take_node(cur), _take_obj(cur)
            version, value, deliver_after = _take_int(cur), _take_val(cur), _take_int(cur)
            delta.append(MsgSend(msg_id, src, dst, obj, version, value, deliver_after))
        elif op == v.id("<msg_deliver>"):
            delta.append(MsgDeliver(_take_int(cur)))
        elif op == v.id("<msg_drop>"):
            delta.append(MsgDrop(_take_int(cur)))
        elif op == v.id("<event_append>"):
            delta.append(_reconstruct_event(state, action))
        elif op == v.id("<partition_set>"):
            delta.append(PartitionSet(_parse_groups(cur)))
        elif op == v.id("<node_down>"):
            delta.append(NodeDown(_take_node(cur)))
        elif op == v.id("<node_up>"):
            delta.append(NodeUp(_take_node(cur)))
        elif op == v.id("<clock_set>"):
            delta.append(ClockSet(_take_int(cur)))
        elif op == v.id("<set_result>"):
            delta.append(_parse_result(cur))
        else:
            raise DistTokenizeError(f"expected an edit op, got {v.token(op)!r}")
    return delta


def _parse_groups(cur: _Cursor) -> tuple[tuple[str, ...], ...]:
    """Parse ``<pgroup> node+ ( <pgroup> node+ )* <pgroups_end>`` into a tuple of node tuples."""
    v = cur.vocab
    groups: list[tuple[str, ...]] = []
    if cur.peek() != v.id("<pgroup>"):
        raise DistTokenizeError("partition_set must begin with a <pgroup>")
    while cur.peek() != v.id("<pgroups_end>"):
        cur.take()  # consume <pgroup>
        current: list[str] = []
        while cur.peek() not in (v.id("<pgroup>"), v.id("<pgroups_end>")):
            current.append(_take_node(cur))
        if not current:
            raise DistTokenizeError("empty partition group")
        groups.append(tuple(current))
    cur.take()  # consume <pgroups_end>
    return tuple(groups)


def _parse_result(cur: _Cursor) -> SetResult:
    v = cur.vocab
    status_tok = cur.take()
    if status_tok not in v.id_to_status:
        raise DistTokenizeError(f"expected a status token, got {v.token(status_tok)!r}")
    status = v.id_to_status[status_tok]
    value = str(_take_int(cur)) if status == "advanced" else _take_val(cur)
    return SetResult(status, value)


__all__ = [
    "DistTokenizeError",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "parse_target",
]
