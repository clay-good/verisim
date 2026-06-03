"""Correction and belief operators ``C`` (SPEC-6 §8.3).

Two families, because the host oracle has two consultation modes (§5.3):

  - **Full-consult operators** -- given the complete one-step truth ``s'`` and the model's
    prediction ``ŝ'``, decide the post-consultation coupled state. ``HardReset``, ``Residual``, and
    ``Projection`` mirror v0's §6.2 and NW5's §8.3 operators exactly. With a full-truth consultation
    all three snap the coupled state to ``s'`` and so are identical on faithful horizon; they differ
    only in what they *record*.
  - **The subsystem filter** -- given a *partial* observation of one subsystem (the cheap probe) and
    the prediction ``ŝ'``, snap only that subsystem (the process table, the fd table, the embedded
    filesystem, or the global last-exit) to truth and keep the model's belief for everything else.
    This is the host's version of NW5's belief filter, cutting the bundle by *subsystem* instead of
    by host. It is native here (no v0 identity collapse): per-subsystem correction fixes strictly
    less than a full consult, so the two genuinely differ on horizon -- the EH3 lever, §8.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from verisim.host.state import HostState
from verisim.hostmetrics.divergence import divergence, host_facts

from .observe import SubsystemObservation


@runtime_checkable
class FullCorrection(Protocol):
    def correct(self, predicted: HostState, truth: HostState) -> HostState: ...


class HardReset:
    """``s ← s'``: snap the coupled state to the full truth (the §8.3 baseline)."""

    def correct(self, predicted: HostState, truth: HostState) -> HostState:
        return truth


@dataclass
class Residual:
    """Corrected state is truth; records the discrepancy ``|s' △ ŝ'|`` per correction.

    The discrepancy (the symmetric-difference fact count) is the online-learning signal a
    self-healing loop would step on (SPEC-6 §8.4); recorded here as a diagnostic.
    """

    discrepancies: list[int] = field(default_factory=list)

    def correct(self, predicted: HostState, truth: HostState) -> HostState:
        self.discrepancies.append(len(host_facts(predicted) ^ host_facts(truth)))
        return truth


@dataclass
class Projection:
    """Corrected state is truth; records the repaired *fraction* per correction.

    The fraction is the pre-correction composed divergence ``d(ŝ', s')`` -- how much of the bundle
    the consultation had to fix (SPEC-6 §8.3 cheapest-faithful-horizon-per-correction lens).
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: HostState, truth: HostState) -> HostState:
        self.repaired_fractions.append(divergence(predicted, truth))
        return truth


def subsystem_filter(predicted: HostState, obs: SubsystemObservation) -> HostState:
    """Snap the observed subsystem to truth; keep the prediction for the rest (§8.3).

    Corrects exactly what the probe reveals -- one of ``proc`` (the process table, credentials and
    exit codes included), ``fd`` (the fd table), ``fs`` (the embedded filesystem), or ``global``
    (the last-exit) -- and nothing else. Returns a fresh state (inputs are not mutated), so every
    unobserved subsystem is the model's belief verbatim.
    """
    truth = obs.truth
    if obs.subsystem == "proc":
        return replace(predicted, procs=dict(truth.procs), next_pid=truth.next_pid)
    if obs.subsystem == "fd":
        return replace(predicted, fds=dict(truth.fds))
    if obs.subsystem == "fs":
        return replace(predicted, fs=truth.fs.copy())
    return replace(predicted, last_exit=truth.last_exit)  # "global"


@dataclass
class SubsystemFilter:
    """The probe-mode operator: correct only the observed subsystem, recording cost.

    ``repaired_fractions`` logs the pre-correction composed divergence each consultation removes
    (the EH3 diagnostic, the per-subsystem analogue of :class:`Projection`).
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: HostState, obs: SubsystemObservation) -> HostState:
        corrected = subsystem_filter(predicted, obs)
        self.repaired_fractions.append(divergence(predicted, corrected))
        return corrected
