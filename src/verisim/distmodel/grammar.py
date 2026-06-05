"""The log/replica-delta grammar automaton for constrained decoding (SPEC-7 §6.1, DS4 incr 2).

The distributed analogue of v0's :class:`~verisim.model.grammar.DeltaGrammar` and the network /
host :class:`~verisim.netmodel.grammar.NetDeltaGrammar`: an LL(1) pushdown automaton over the
delta token grammar :mod:`tokenizer` encodes. At each step it exposes the grammar-valid next token
ids so the decoder masks the model's logits to that set -- making every prediction a *valid*
distributed delta by construction, independent of the model's weights (the standing split:
grammar-validity is the decoder's job, semantic faithfulness is the oracle's).

Distributed deltas are richer than the net/host worlds' fixed-body deltas in two ways, so this
automaton carries two extra nonterminals beyond a flat per-op body:

  - ``<partition_set>`` expands to a *nested run* -- one or more groups, each one or more nodes,
    closed by ``<pgroups_end>`` (the only world whose delta has a variable-length argument, since
    the causal-log run is reconstructed off-band, never decoded). The ``PGROUP`` / ``PGROUP_TAIL``
    loop mirrors v0's ``PATH_SEG`` / ``CONTENT_TOK`` runs: a leaf set unioned with the closing
    marker, LL(1) because node ids are disjoint from ``<pgroup>`` / ``<pgroups_end>``.
  - ``<set_result>`` carries a status token whose payload *type* depends on the status:
    ``advanced`` is followed by an integer count, every other status by a value token. The
    ``RESULT_STATUS`` symbol branches on the consumed status exactly as v0's ``NODE`` branches on
    ``<file>`` vs ``<dir>``.

The ``<event_append>`` op has an empty body: its content is reconstructed from ``(state, action)``
by :func:`~verisim.distmodel.tokenizer.parse_target`, so only its bare marker is decoded. A
top-level edit cap forces ``<eos>`` (always grammar-valid at ``DELTA``); see :mod:`decode`.
"""

from __future__ import annotations

from .vocab import DistVocab

# Each delta op expands to the sequence of argument nonterminals that follow it. ``<event_append>``
# has an empty body (reconstructed off-band); ``<partition_set>`` and ``<set_result>`` expand to the
# two structured nonterminals below rather than a flat leaf list.
_OP_BODY: dict[str, list[str]] = {
    "<replica_write>": ["OBJ", "NODE", "INT", "VALUE"],
    "<msg_send>": ["INT", "NODE", "NODE", "OBJ", "INT", "VALUE", "INT"],
    "<msg_deliver>": ["INT"],
    "<msg_drop>": ["INT"],
    "<event_append>": [],
    "<partition_set>": ["PGROUPS"],
    "<node_down>": ["NODE"],
    "<node_up>": ["NODE"],
    "<clock_set>": ["INT"],
    "<set_result>": ["RESULT_STATUS"],
}


class DistDeltaGrammar:
    """LL(1) recognizer/constrainer for the log/replica-delta token grammar."""

    def __init__(self, vocab: DistVocab) -> None:
        self.v = vocab
        self._op_ids = {vocab.id(op): op for op in _OP_BODY}

    def start(self) -> list[str]:
        return ["DELTA"]

    @staticmethod
    def is_accept(stack: list[str]) -> bool:
        return not stack

    def allowed(self, stack: list[str]) -> frozenset[int]:
        if not stack:
            return frozenset()
        v = self.v
        top = stack[0]
        if top == "DELTA":
            return frozenset({*v.op_ids, v.eos})
        if top == "NODE":
            return v.node_ids
        if top == "OBJ":
            return v.obj_ids
        if top == "VALUE":
            return v.value_ids
        if top == "INT":
            return v.int_ids
        if top == "RESULT_STATUS":
            return v.status_ids
        if top == "PGROUPS":  # a partition's groups must open with a <pgroup>
            return frozenset({v.id("<pgroup>")})
        if top == "PGROUP_TAIL":  # within/between groups: a node, a new group, or the terminator
            return frozenset({*v.node_ids, v.id("<pgroup>"), v.id("<pgroups_end>")})
        raise AssertionError(f"unknown grammar symbol {top!r}")  # pragma: no cover

    def advance(self, stack: list[str], token: int) -> list[str]:
        if token not in self.allowed(stack):
            raise ValueError(f"token {self.v.token(token)!r} not allowed in state {stack}")
        v = self.v
        rest = stack[1:]
        top = stack[0]

        if top == "DELTA":
            if token == v.eos:
                return rest  # complete delta
            return [*_OP_BODY[self._op_ids[token]], "DELTA", *rest]
        if top == "RESULT_STATUS":
            # advanced carries an int count; every other status carries a value token.
            leaf = "INT" if token == v.status_to_id["advanced"] else "VALUE"
            return [leaf, *rest]
        if top == "PGROUPS":
            # consumed the opening <pgroup>: require >=1 node, then the loop.
            return ["NODE", "PGROUP_TAIL", *rest]
        if top == "PGROUP_TAIL":
            if token == v.id("<pgroups_end>"):
                return rest  # close the whole partition
            if token == v.id("<pgroup>"):
                return ["NODE", "PGROUP_TAIL", *rest]  # a new group: require >=1 node
            return stack  # a node id: stay in the current group's run
        # NODE / OBJ / VALUE / INT: a single leaf terminal, then pop.
        return rest
