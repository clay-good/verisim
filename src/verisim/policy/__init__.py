"""SPEC-23 Direction B -- a declarative policy language for the un-dominated triad (relational,
cumulative, context-dependent harms), each compiling to a (:class:`Monitor`, :class:`Oracle`,
proposer) the SPEC-23 auditor audits with the *same* loop. See docs/specs/SPEC-23.md §3.2.
"""

from __future__ import annotations

from .language import (
    ContextPolicy,
    CumulativePolicy,
    PolicyProposer,
    RelationalPolicy,
    compile_policy,
)

__all__ = [
    "ContextPolicy",
    "CumulativePolicy",
    "PolicyProposer",
    "RelationalPolicy",
    "compile_policy",
]
