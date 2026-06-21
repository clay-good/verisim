"""SPEC-23/SPEC-24 -- the Monitor Auditor: certified-complete effect-monitoring as a tool you point
at *someone else's* guardrail, with a graded guarantee.

``audit(monitor, oracle, proposer, budget) -> Certificate`` runs the RA22-RA25 discover->fix->
re-verify loop against any covering surface that satisfies the :class:`Monitor` protocol and any
exact harm oracle that satisfies the :class:`Oracle` protocol, with no literal path baked in.
``certify(...)`` (SPEC-24) adds the graded guarantee: an exhaustive theorem to depth k plus a
statistical residual bound over the sampled tail; ``differential(...)`` certifies a monitor patch is
a monotone improvement. See docs/specs/SPEC-23.md and docs/specs/SPEC-24.md.
"""

from __future__ import annotations

from .auditor import DiffCertificate, audit, audit_diff, certify, differential
from .guarantee import Guarantee
from .llm_guardrail import (
    ClaudeCliJudge,
    LLMCertificate,
    LLMGuardrailMonitor,
    RelationalClaudeJudge,
    StubJudge,
    certify_llm,
)
from .monitors import (
    DenylistMonitor,
    ResolverMonitor,
    SubprocessMonitor,
    SyntacticPathMonitor,
)
from .oracles import ContainerDiffOracle, ShellPathOracle
from .proposers import (
    CorpusProposer,
    DirectedNeuralProposer,
    ExhaustiveDepthProposer,
    GrammarProposer,
    NeuralGrammarProposer,
)
from .protocols import Action, Certificate, Hole, Monitor, Oracle, Proposer

__all__ = [
    "Action",
    "Certificate",
    "ClaudeCliJudge",
    "ContainerDiffOracle",
    "CorpusProposer",
    "DenylistMonitor",
    "DiffCertificate",
    "DirectedNeuralProposer",
    "ExhaustiveDepthProposer",
    "GrammarProposer",
    "Guarantee",
    "Hole",
    "LLMCertificate",
    "LLMGuardrailMonitor",
    "Monitor",
    "NeuralGrammarProposer",
    "Oracle",
    "Proposer",
    "RelationalClaudeJudge",
    "ResolverMonitor",
    "ShellPathOracle",
    "StubJudge",
    "SubprocessMonitor",
    "SyntacticPathMonitor",
    "audit",
    "audit_diff",
    "certify",
    "certify_llm",
    "differential",
]
