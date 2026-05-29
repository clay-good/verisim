"""The delta-grammar automaton for constrained decoding (SPEC-2 §5.2).

The decoder must be unable to emit a syntactically invalid edit. This is an LL(1)
pushdown automaton over the same grammar :mod:`tokenizer` encodes: at each step it
exposes the set of grammar-valid next token ids, so the decoder masks the model's
logits to that set. Validity is then guaranteed *by construction*, independent of
the model's weights -- which is exactly the point the spec keeps crisp:
grammar-validity is the decoder's job, semantic faithfulness is the oracle's.

The stack holds nonterminal symbols (top = index 0). ``allowed`` returns the
terminals legal next; ``advance`` consumes one token and returns the new stack; an
empty stack means a complete, accepted delta.
"""

from __future__ import annotations

from .vocab import Vocab

# Each delta op expands to the sequence of argument nonterminals that follow it.
_OP_BODY: dict[str, list[str]] = {
    "<create>": ["PATH", "NODE"],
    "<delete>": ["PATH"],
    "<modify>": ["PATH", "CONTENT"],
    "<move>": ["PATH", "PATH"],
    "<chmod>": ["PATH", "MODE"],
    "<setcwd>": ["PATH"],
    "<setenv>": ["ENVKEY", "CONTENTTOK"],
    "<setresult>": ["EXIT", "STDOUT"],
}


class DeltaGrammar:
    """LL(1) recognizer/constrainer for the delta token grammar."""

    def __init__(self, vocab: Vocab) -> None:
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
        if top == "PATH":
            return frozenset({v.id("<p>")})
        if top == "PATH_SEG":
            return frozenset({*v.name_ids, v.id("</p>")})
        if top == "NODE":
            return frozenset({v.id("<file>"), v.id("<dir>")})
        if top == "CONTENT":
            return frozenset({v.id("<c>")})
        if top == "CONTENT_TOK":
            return frozenset({*v.content_ids, v.id("</c>")})
        if top == "MODE":
            return v.mode_ids
        if top == "EXIT":
            return v.exit_ids
        if top == "ENVKEY":
            return v.envkey_ids
        if top == "CONTENTTOK":
            return v.content_ids
        if top == "STDOUT":
            return frozenset({v.id("<o>")})
        if top == "STDOUT_TOK":
            return frozenset({*v.content_ids, *v.name_ids, v.id("<nl>"), v.id("</o>")})
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
        if top == "PATH":
            return ["PATH_SEG", *rest]  # consumed <p>
        if top == "PATH_SEG":
            return rest if token == v.id("</p>") else stack  # name -> stay; </p> -> pop
        if top == "NODE":
            if token == v.id("<file>"):
                return ["CONTENT", "MODE", *rest]
            return ["MODE", *rest]  # <dir>
        if top == "CONTENT":
            return ["CONTENT_TOK", *rest]  # consumed <c>
        if top == "CONTENT_TOK":
            return rest if token == v.id("</c>") else stack
        if top == "STDOUT":
            return ["STDOUT_TOK", *rest]  # consumed <o>
        if top == "STDOUT_TOK":
            return rest if token == v.id("</o>") else stack
        # MODE / EXIT / ENVKEY / CONTENTTOK: a single leaf terminal, then pop.
        return rest
