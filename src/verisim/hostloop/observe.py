"""The partial-observation host oracle (SPEC-6 §5.3, §8.2): full + per-subsystem modes.

The host world's enrichment of ``ρ`` is not just *when* to consult but *which subsystem's truth to
buy* (the ``π_w`` axis, §8.2): a consultation may reveal the whole bundle, or just one subsystem
(the process table, the fd table, the embedded filesystem, or the global last-exit). This is the
host analogue of the network's host-probe (SPEC-5 §5.3), generalized from "which host" to "which
truth-source" -- and it is what makes the composition-law budgeting (H13) measurable.

  - **full** (expensive): the complete true next state, as in v0. Reveals every subsystem's facts.
  - **probe** (cheap): a localized observation of one subsystem -- only that subsystem's facts of
    the true next state are revealed and corrected, and the bit-cost counts only that subsystem.

Both consult the same underlying deterministic :class:`~verisim.hostoracle.base.HostOracle`, so the
truth is identical; they differ only in *how much* is returned and therefore in cost. The bit-cost
(``probe.bits`` vs :func:`full_bits`) is the denominator of per-subsystem consultation efficiency --
faithful horizon per oracle-bit (SPEC-6 §9.4). The subsystem partition is reused verbatim from the
HC3 metric (:func:`~verisim.hostmetrics.divergence.facts_by_subsystem`) so the observation, the
correction, and the divergence all cut the bundle the same way. Pure and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostmetrics.divergence import SUBSYSTEMS, facts_by_subsystem, host_facts
from verisim.hostoracle.base import HostOracle, HostStepResult


@dataclass(frozen=True)
class SubsystemObservation:
    """The true next state, of which only ``subsystem`` is revealed/corrected (§5.3, §8.2).

    The probe computes the full true next state (the oracle is deterministic) but the contract is
    *partial*: a :class:`~verisim.hostloop.operator.SubsystemFilter` snaps only ``subsystem`` to
    this truth and keeps the model's belief for every other subsystem, and ``bits`` counts only
    ``subsystem``'s facts. So a per-subsystem consult corrects strictly less than a full consult.
    """

    subsystem: str
    truth: HostState
    bits: int


def full_bits(state: HostState) -> int:
    """Description length of a *full* consultation in symbols (one per bundle fact)."""
    return len(host_facts(state))


def subsystem_bits(state: HostState, subsystem: str) -> int:
    """Description length of a single-subsystem observation (one per revealed fact, §9.4)."""
    return len(facts_by_subsystem(state)[subsystem])


class PartialHostOracle:
    """A :class:`~verisim.hostoracle.base.HostOracle` exposed in two modes (SPEC-6 §5.3)."""

    def __init__(self, oracle: HostOracle) -> None:
        self._oracle = oracle

    def full(self, state: HostState, action: HostAction) -> HostStepResult:
        """The complete one-step truth from ``state`` under ``action`` (expensive)."""
        return self._oracle.step(state, action)

    def probe(
        self, state: HostState, action: HostAction, subsystem: str
    ) -> SubsystemObservation:
        """A localized observation of one ``subsystem`` of the true next state (cheap)."""
        if subsystem not in SUBSYSTEMS:
            raise ValueError(f"unknown subsystem {subsystem!r}; choose from {SUBSYSTEMS}")
        truth = self._oracle.step(state, action).state
        return SubsystemObservation(
            subsystem=subsystem, truth=truth, bits=subsystem_bits(truth, subsystem)
        )
