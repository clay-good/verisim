"""The graph-delta-grammar automaton for constrained decoding (SPEC-5 §6.1).

The network analogue of v0's :class:`~verisim.model.grammar.DeltaGrammar`: an LL(1)
pushdown automaton over the delta token grammar :mod:`tokenizer` encodes. At each step it
exposes the grammar-valid next token ids so the decoder masks the model's logits to that
set -- making every prediction a *valid* graph delta by construction, independent of the
model's weights (the spec's standing split: grammar-validity is the decoder's job, semantic
faithfulness is the oracle's).

Net deltas are simpler than v0's filesystem deltas: every op body is a *fixed* sequence of
single-leaf terminals (host ids, ports, exit codes) with **no repeating runs** (no path
segments or content loops), so the automaton needs no per-run cap -- only the top-level
edit cap forces ``<eos>`` (see :mod:`decode`).
"""

from __future__ import annotations

from .vocab import NetVocab

# Each delta op expands to the sequence of argument nonterminals that follow it.
_OP_BODY: dict[str, list[str]] = {
    "<host_up>": ["HOST"],
    "<host_down>": ["HOST"],
    "<link_add>": ["HOST", "HOST"],
    "<link_del>": ["HOST", "HOST"],
    "<svc_up>": ["HOST", "PORT"],
    "<svc_down>": ["HOST", "PORT"],
    "<fw_deny>": ["HOST", "HOST"],
    "<fw_allow>": ["HOST", "HOST"],
    "<flow_open>": ["HOST", "HOST", "PORT"],
    "<flow_close>": ["HOST", "HOST", "PORT"],
    "<clock_advance>": [],
    "<set_result>": ["EXIT"],
}


class NetDeltaGrammar:
    """LL(1) recognizer/constrainer for the graph-delta token grammar."""

    def __init__(self, vocab: NetVocab) -> None:
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
        if top == "HOST":
            return v.host_ids
        if top == "PORT":
            return v.port_ids
        if top == "EXIT":
            return v.exit_ids
        raise AssertionError(f"unknown grammar symbol {top!r}")  # pragma: no cover

    def advance(self, stack: list[str], token: int) -> list[str]:
        if token not in self.allowed(stack):
            raise ValueError(f"token {self.v.token(token)!r} not allowed in state {stack}")
        rest = stack[1:]
        top = stack[0]
        if top == "DELTA":
            if token == self.v.eos:
                return rest  # complete delta
            return [*_OP_BODY[self._op_ids[token]], "DELTA", *rest]
        # HOST / PORT / EXIT: a single leaf terminal, then pop.
        return rest
