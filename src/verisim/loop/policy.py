"""Consultation policies ``π_c`` (SPEC-2 §6.1).

A policy decides, per step, whether to *propose* spending an oracle consultation,
given a :class:`StepContext` (the step index plus the model's per-step uncertainty
signal and the uncertainty accumulated since the last consultation). The runner
enforces the budget ``ρ`` on top of the policy's proposals -- both the hard cap
*and* a spend-down backstop -- so every policy spends *exactly* the budget and
comparisons in §10 are at truly equal ``ρ`` (SPEC-2 §16 loop invariant).

v0 ships:

  - ``fixed`` (every-``k``-steps) -- the dumb RQ2 baseline;
  - ``uncertainty_triggered(τ)`` -- consult when the model's *instantaneous*
    predictive uncertainty over the delta exceeds ``τ`` (SPEC-2 §6.1, §7.2);
  - ``drift_triggered(τ)`` -- consult when *accumulated* uncertainty since the last
    consultation (a cheap, oracle-free proxy for compounding drift) exceeds ``τ``.

The uncertainty signal itself comes from the model (the neural ``M_θ`` exposes the
mean entropy of its constrained decode; baselines report ``0`` and so reduce these
policies to "never trigger, spend the budget at the tail"). The learned-policy
variant of §6.1 is a later stretch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StepContext:
    """What a policy sees at step ``step`` before deciding whether to consult.

    ``signal`` is the model's uncertainty for *this* step's prediction (``0`` for a
    model that exposes none). ``cumulative_signal`` is the sum of signals since the
    last consultation, including this step -- the runner resets it on every consult.
    """

    step: int
    signal: float = 0.0
    cumulative_signal: float = 0.0


@runtime_checkable
class ConsultationPolicy(Protocol):
    def should_consult(self, ctx: StepContext) -> bool: ...


@dataclass(frozen=True)
class FixedInterval:
    """Consult every ``k`` steps (``k = 1`` is every step). The §6.1 baseline."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"FixedInterval k must be >= 1, got {self.k}")

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.step % self.k == 0


@dataclass(frozen=True)
class Never:
    """Never consult -- the ``ρ = 0`` pure-model policy (baseline b0)."""

    def should_consult(self, ctx: StepContext) -> bool:
        return False


@dataclass(frozen=True)
class UncertaintyTriggered:
    """Consult when the model's instantaneous uncertainty exceeds ``τ`` (§6.1).

    Spends the budget on the steps the model is *least* sure about. Needs a model
    that exposes a per-step uncertainty signal (the neural ``M_θ``); calibration of
    that signal vs. actual divergence is the §7.2 diagnostic.
    """

    tau: float

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.signal > self.tau


@dataclass(frozen=True)
class DriftTriggered:
    """Consult when uncertainty *accumulated* since the last consult exceeds ``τ``.

    The running sum of per-step uncertainty is a cheap, oracle-free estimate of
    accumulated drift (drift compounds, so summing the per-step signal tracks it).
    Distinct from :class:`UncertaintyTriggered`, which triggers on the instantaneous
    value: this one tolerates a sequence of small uncertainties and fires once they
    add up.
    """

    tau: float

    def should_consult(self, ctx: StepContext) -> bool:
        return ctx.cumulative_signal > self.tau


def fixed_interval_for_rho(rho: float) -> ConsultationPolicy:
    """A ``fixed`` policy whose natural consultation rate is ``rho``.

    ``rho = 0`` -> never consult; ``rho = 1`` -> every step; otherwise consult
    every ``round(1/rho)`` steps. The runner still caps total calls at the budget
    ``floor(rho * T)`` (see :func:`verisim.loop.runner.budget_for_rho`), so the
    policy's rate and the hard budget agree.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if rho == 0.0:
        return Never()
    return FixedInterval(k=max(1, round(1.0 / rho)))
