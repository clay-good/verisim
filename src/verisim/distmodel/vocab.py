"""The closed token vocabulary for the distributed world model `M_θ` (SPEC-7 §6.1, DS4).

The distributed world (`docs/distributed-semantics.md`) draws every argument from a finite
:class:`~verisim.dist.config.DistConfig` -- the cluster **nodes**, the replicated **objects**, and
the **value tokens** a write may store -- exactly as v0 fixes content/mode/name pools, the network
world fixes host-id/port pools, and the host world fixes path/content/uid pools. The one *unbounded*
identifier family the serialization DSL carries -- the monotone bookkeeping counters (replica
``version``, message ``msg_id`` / ``deliver_after``, the simulation ``clock``, and the ``advanced``
delivery count) -- is bounded here by a single fixed integer pool ``<int:0..max_int>`` sized past
what a finite-length rollout can reach (the host's ``max_pid`` / ``max_fd`` trick), so the entire
``(state, action) -> Δ`` DSL maps to a small *closed* set of tokens built deterministically from a
config. The flat DS4 arm (the DD-D-analogue flat-serializer baseline) trains over exactly these ids;
the constrained decoder (DS4 increment 2) masks to them.

The causal-log ``EventAppend`` edit is intentionally **not** given argument tokens: its content
(``id`` = ``next_event_id``, ``node`` = the coordinator, ``op`` = the action's raw string, ``clock``
= the current clock, ``happens_before`` = the prior same-node events) is a pure function of
``(state, action)``, so the target encodes only its one-token *marker* and the tokenizer rebuilds
the edit from context -- as the network tokenizer omits the always-1 ``ClockAdvance`` amount.
This keeps the one genuinely variable-length field (``happens_before``) out of the token grammar.

Token families:
  - specials: ``<pad> <bos> <eos>``;
  - prompt/structure markers (``<state> <replica> <parts> <pgroup> <down> <inflight> <msg> <clock>
    <nextids> <last> <none> <action> <gen>``) plus the target's partition-groups terminator
    ``<pgroups_end>``;
  - delta ops (one per :mod:`verisim.dist.delta` edit type);
  - command tokens (one per §3.2 action), status tokens (the five ``SetResult`` statuses), and the
    four leaf vocabularies: nodes, objects, values (config values + the boot ``default_value`` + the
    empty token an unavailable/no-replica result carries), and the bounded integer pool.
"""

from __future__ import annotations

from verisim.dist.config import DistConfig

_SPECIALS = ("<pad>", "<bos>", "<eos>")
_MARKERS = (
    "<state>", "<replica>", "<parts>", "<pgroup>", "<down>", "<inflight>", "<msg>",
    "<clock>", "<nextids>", "<last>", "<none>", "<action>", "<gen>", "<pgroups_end>",
)
_OPS = (
    "<replica_write>", "<msg_send>", "<msg_deliver>", "<msg_drop>", "<event_append>",
    "<partition_set>", "<node_down>", "<node_up>", "<clock_set>", "<set_result>",
)
# Commands (SPEC-7 §3.2). ``heal`` takes no args; the rest mirror the action grammar.
_COMMANDS = ("put", "get", "cas", "advance", "partition", "heal", "crash", "restart")
# The five client-visible result statuses the reference oracle emits (SPEC-7 §5.1).
_STATUSES = ("ok", "conflict", "unavailable", "no_replica", "advanced")

#: Default integer-pool size. Sized past the monotone counters a curriculum rollout (n_steps <= ~40,
#: full replication) can reach (msg_id grows ~2 per write); the tokenizer raises on overflow.
DEFAULT_MAX_INT = 256


class DistVocab:
    """A closed, deterministic token<->id mapping for one :class:`DistConfig` (+ integer pool)."""

    def __init__(self, config: DistConfig, *, max_int: int = DEFAULT_MAX_INT) -> None:
        if max_int < 1:
            raise ValueError(f"max_int must be >= 1, got {max_int}")
        self.config = config
        self.max_int = max_int
        #: the value leaf vocab: config values + the boot default + the empty token (unavailable /
        #: no_replica results carry ``""``), so every ``ReplicaWrite`` / ``MsgSend`` / ``SetResult``
        #: value is encodable.
        self._values = (*config.values, config.default_value, "")

        tokens: list[str] = []
        tokens += _SPECIALS
        tokens += _MARKERS
        tokens += _OPS
        tokens += [f"<cmd:{c}>" for c in _COMMANDS]
        tokens += [f"<st:{s}>" for s in _STATUSES]
        tokens += [f"<nd:{n}>" for n in config.nodes]
        tokens += [f"<ob:{o}>" for o in config.objects]
        tokens += [f"<val:{v}>" for v in self._values]
        tokens += [f"<int:{i}>" for i in range(max_int)]

        self._tokens = tuple(tokens)
        self._id = {tok: i for i, tok in enumerate(self._tokens)}

        # Leaf lookup tables (token id <-> domain value).
        self.node_to_id = {n: self._id[f"<nd:{n}>"] for n in config.nodes}
        self.id_to_node = {v: k for k, v in self.node_to_id.items()}
        self.obj_to_id = {o: self._id[f"<ob:{o}>"] for o in config.objects}
        self.id_to_obj = {v: k for k, v in self.obj_to_id.items()}
        self.val_to_id = {v: self._id[f"<val:{v}>"] for v in self._values}
        self.id_to_val = {v: k for k, v in self.val_to_id.items()}
        self.status_to_id = {s: self._id[f"<st:{s}>"] for s in _STATUSES}
        self.id_to_status = {v: k for k, v in self.status_to_id.items()}
        self.int_to_id = {i: self._id[f"<int:{i}>"] for i in range(max_int)}
        self.id_to_int = {v: k for k, v in self.int_to_id.items()}

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

    # -- token-class sets (for the grammar / constrained decoder, DS4 incr 2) -

    @property
    def op_ids(self) -> frozenset[int]:
        return frozenset(self._id[op] for op in _OPS)

    @property
    def node_ids(self) -> frozenset[int]:
        return frozenset(self.node_to_id.values())

    @property
    def obj_ids(self) -> frozenset[int]:
        return frozenset(self.obj_to_id.values())

    @property
    def value_ids(self) -> frozenset[int]:
        return frozenset(self.val_to_id.values())

    @property
    def status_ids(self) -> frozenset[int]:
        return frozenset(self.status_to_id.values())

    @property
    def int_ids(self) -> frozenset[int]:
        return frozenset(self.int_to_id.values())
