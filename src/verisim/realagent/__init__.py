"""SPEC-22 RA -- the real-agent arc: the oracle safety gate around a *real LLM* doing real work.

The CU1-CU39 arc proved the safety gate against a synthetic adversary on hand-specified danger
grammars. The RA arc breaks that assumption: a real Claude agent does real shell tasks, its proposed
actions route through the CU21 covering-target gate against the exact oracle, and the missed-danger
rate is measured on dangers and actions *nobody hand-authored*. The real-LLM arm is gated behind an
env var (the LP7 discipline, generalized to the model provider); CI and the committed figure run a
recorded transcript so the result is hermetic and regenerable.
"""

from verisim.realagent.harness import (
    RA1Config,
    RA1Result,
    ScriptedAgent,
    Task,
    cu_ra1_verdict,
    run_ra1,
    write_csv,
)

__all__ = [
    "RA1Config",
    "RA1Result",
    "ScriptedAgent",
    "Task",
    "cu_ra1_verdict",
    "run_ra1",
    "write_csv",
]
