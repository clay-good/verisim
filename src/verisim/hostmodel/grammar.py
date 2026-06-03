"""The bundle-delta grammar automaton for constrained decoding (SPEC-6 §6.1, HC4).

The host analogue of v0's :class:`~verisim.model.grammar.DeltaGrammar` and the network
:class:`~verisim.netmodel.grammar.NetDeltaGrammar`: an LL(1) pushdown automaton over the delta
token grammar :mod:`tokenizer` encodes. At each step it exposes the grammar-valid next token ids
so the decoder masks the model's logits to that set -- making every prediction a *valid* bundle
delta by construction, independent of the model's weights (the standing split: grammar-validity is
the decoder's job, semantic faithfulness is the oracle's).

Host bundle deltas, like net deltas, expand each op to a *fixed* sequence of single-leaf terminals
(pids, fds, uids, paths, content tokens, exit codes) with **no repeating runs**, so the automaton
needs no per-run cap -- only the top-level edit cap forces ``<eos>`` (see :mod:`decode`).
"""

from __future__ import annotations

from .vocab import HostVocab

# Each delta op expands to the sequence of argument nonterminals that follow it.
_OP_BODY: dict[str, list[str]] = {
    "<proc_spawn>": ["PID", "PID", "UID"],
    "<proc_exit>": ["PID", "EXIT"],
    "<fd_open>": ["PID", "FD", "PATH"],
    "<fd_close>": ["PID", "FD"],
    "<cred_change>": ["PID", "UID"],
    "<fs_create>": ["PATH", "CONTENT", "EXIT"],
    "<fs_modify>": ["PATH", "CONTENT", "EXIT"],
    "<set_exit>": ["EXIT"],
}


class HostDeltaGrammar:
    """LL(1) recognizer/constrainer for the bundle-delta token grammar."""

    def __init__(self, vocab: HostVocab) -> None:
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
        if top == "PID":
            return v.pid_ids
        if top == "FD":
            return v.fd_ids
        if top == "UID":
            return v.uid_ids
        if top == "PATH":
            return v.path_ids
        if top == "CONTENT":
            return v.content_ids
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
        # PID / FD / UID / PATH / CONTENT / EXIT: a single leaf terminal, then pop.
        return rest
