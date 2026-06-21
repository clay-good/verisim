"""SPEC-23 -- the Monitor Auditor: certified-complete effect-monitoring as a tool you point at
*someone else's* guardrail.

``audit(monitor, oracle, proposer, budget) -> Certificate`` runs the RA22-RA25 discover->fix->
re-verify loop against any covering surface that satisfies the :class:`Monitor` protocol and any
exact harm oracle that satisfies the :class:`Oracle` protocol, with no literal path baked in. See
docs/specs/SPEC-23.md.
"""

from __future__ import annotations

from .auditor import audit
from .monitors import DenylistMonitor, ResolverMonitor, SyntacticPathMonitor
from .oracles import ContainerDiffOracle, ShellPathOracle
from .proposers import CorpusProposer, GrammarProposer, NeuralGrammarProposer
from .protocols import Action, Certificate, Hole, Monitor, Oracle, Proposer

__all__ = [
    "Action",
    "Certificate",
    "ContainerDiffOracle",
    "CorpusProposer",
    "DenylistMonitor",
    "GrammarProposer",
    "Hole",
    "Monitor",
    "NeuralGrammarProposer",
    "Oracle",
    "Proposer",
    "ResolverMonitor",
    "ShellPathOracle",
    "SyntacticPathMonitor",
    "audit",
]
