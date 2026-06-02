"""The composition-faithfulness diagnostic -- the headline-new metric (SPEC-6 §9.2, H13).

This is the object the prime directive (§0) is about: **is whole-machine faithfulness predictable
from the faithfulness of its parts?** Given the per-step, per-subsystem faithfulness booleans (a
step is composed-faithful iff *every* subsystem is faithful), the diagnostic measures the
**composed per-step acceptance** ``a`` against two candidate laws over the per-subsystem acceptance
rates ``a_i`` (SPEC-6 §9.2, §10.2 H13):

- **Multiplicative** ``a ≈ ∏_i a_i`` -- holds iff subsystem failures are *independent* (the
  speculative-decoding-acceptance view, SPEC-3 §8). If so, the optimal which-oracle policy ``π_w``
  (spend the oracle on the subsystem with the lowest ``a_i``) is *derivable* (§8.2).
- **Weakest-link** ``a ≈ min_i a_i`` -- holds iff failures *coincide* (one subsystem dominates;
  when it fails the others tend to fail too).

Because a step fails if *any* subsystem fails, the composed acceptance is always bounded
``∏_i a_i ≤ a ≤ min_i a_i`` (independent ↔ perfectly-correlated failures). The verdict reports which
end ``a`` sits at, or **coupled** if it falls below the multiplicative floor (anti-correlated
failures spread across more steps) -- the honest negative where "model subsystems independently" is
the wrong bet and coupling is the load-bearing structure (§10.2). Pure and dependency-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

#: Tolerance band (in acceptance units) for calling a law a fit rather than "coupled".
DEFAULT_TOL = 0.05


@dataclass(frozen=True)
class CompositionLaw:
    """The result of the composition-faithfulness diagnostic (SPEC-6 §9.2)."""

    subsystem_acceptance: dict[str, float]
    composed_acceptance: float
    multiplicative_prediction: float
    weakest_link_prediction: float
    multiplicative_residual: float
    weakest_link_residual: float
    verdict: str  # "multiplicative" | "weakest_link" | "coupled"

    def to_dict(self) -> dict[str, object]:
        return {
            "subsystem_acceptance": self.subsystem_acceptance,
            "composed_acceptance": self.composed_acceptance,
            "multiplicative_prediction": self.multiplicative_prediction,
            "weakest_link_prediction": self.weakest_link_prediction,
            "multiplicative_residual": self.multiplicative_residual,
            "weakest_link_residual": self.weakest_link_residual,
            "verdict": self.verdict,
        }


def composition_law(
    per_step_faithful: Sequence[Mapping[str, bool]], tol: float = DEFAULT_TOL
) -> CompositionLaw:
    """Diagnose whether composed acceptance follows the multiplicative or weakest-link law.

    ``per_step_faithful`` is one mapping per step from subsystem name to whether it was faithful
    (e.g. from :func:`~verisim.hostmetrics.divergence.step_faithful_by_subsystem`). Empty input or
    no subsystems yields a degenerate fully-faithful result (verdict ``"multiplicative"``).
    """
    n = len(per_step_faithful)
    subsystems = sorted({sub for step in per_step_faithful for sub in step})
    if n == 0 or not subsystems:
        return CompositionLaw({}, 1.0, 1.0, 1.0, 0.0, 0.0, "multiplicative")

    acceptance = {
        sub: sum(1 for step in per_step_faithful if step.get(sub, True)) / n for sub in subsystems
    }
    composed = sum(1 for step in per_step_faithful if all(step.values())) / n
    mult = prod(acceptance.values())
    weakest = min(acceptance.values())
    r_mult = abs(composed - mult)
    r_weak = abs(composed - weakest)

    if min(r_mult, r_weak) > tol:
        verdict = "coupled"
    elif r_mult <= r_weak:
        verdict = "multiplicative"
    else:
        verdict = "weakest_link"

    return CompositionLaw(
        subsystem_acceptance=acceptance,
        composed_acceptance=composed,
        multiplicative_prediction=mult,
        weakest_link_prediction=weakest,
        multiplicative_residual=r_mult,
        weakest_link_residual=r_weak,
        verdict=verdict,
    )


__all__ = ["DEFAULT_TOL", "CompositionLaw", "composition_law"]
