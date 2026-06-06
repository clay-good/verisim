"""Correction operators ``C`` for the distributed loop (SPEC-7 §8.3; DS5/ED3).

Given a consultation that returns the truth ``s'`` (the tiered oracle's transition from the current
coupled state) and the model's prediction ``ŝ'``, a correction operator decides the
post-consultation coupled state. v0/net/host each ship a ``loop/operator.py`` with this same
``correct(predicted, truth)`` shape; the distributed loop gained it here (DS5 had only the hardcoded
full snap).

The distributed world is where the operator choice stops being a v0-style identity. In v0 a consult
returns the *full* one-step truth, so ``hard_reset``/``residual``/``projection`` all snap the
coupled state to the same ``s'`` and are behaviorally identical on faithful horizon (SPEC-2 §6.2 --
they differ only in the diagnostics they expose). The distributed state has a part a *partial*
correction can decline to fix: the **in-flight replication messages** (``inflight``), the source of
stale reads under partition (SPEC-7 §3.1) and exactly the ``subtle`` error class the cheap tiers
also miss (§5). So a correction operator that snaps only the durable replicas and *trusts the
model's predicted in-flight* corrects strictly less -- and the §8.3 question (ED3) is whether that
correction recovers as much horizon, which it does for replica (``gross``) errors and does **not**
for in-flight (``subtle``) ones. That mode-dependent break of the v0 identity is the ED3 result.

Operators (mirroring v0's three, plus the distributed partial one):

  - ``HardReset`` -- ``s ← s'``: snap the whole cluster to truth. The §8.3 baseline, default.
  - ``Residual`` -- corrected state is truth; records the discrepancy ``|facts(s') △ facts(ŝ')|``
    (the Stage-2 online-learning signal, SPEC-2 §6.2 / the distributed bits-to-correct lens).
  - ``Projection`` -- corrected state is truth; records the repaired *fraction* (pre-correction
    divergence ``d(ŝ', s')``) -- the per-correction cost lens.
  - ``ReplicasOnlyCorrection`` -- snap the durable ``replicas`` to truth but **keep the model's
    predicted ``inflight``** (and clock/log/partition): a partial correction that fixes the
    consistency view but trusts the model on the in-flight medium. Corrects strictly less than
    ``HardReset``; records the repaired fraction. The operator that breaks the v0 identity (ED3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from verisim.dist.state import DistributedState
from verisim.distmetrics.divergence import dist_facts, divergence


@runtime_checkable
class CorrectionOperator(Protocol):
    def correct(self, predicted: DistributedState, truth: DistributedState) -> DistributedState: ...


class HardReset:
    """``s ← s'``: snap the whole cluster to truth, discarding the prediction."""

    def correct(self, predicted: DistributedState, truth: DistributedState) -> DistributedState:
        return truth


@dataclass
class Residual:
    """Corrected state is truth; records the discrepancy ``|facts(s') △ facts(ŝ')|`` per correction.

    The discrepancy is the Stage-2 online-learning signal (SPEC-2 §6.2 / SPEC-7 §8.3). The loop logs
    it as a diagnostic (``discrepancies``) without learning from it -- the distributed
    bits-to-correct a correction injects.
    """

    discrepancies: list[int] = field(default_factory=list)

    def correct(self, predicted: DistributedState, truth: DistributedState) -> DistributedState:
        self.discrepancies.append(len(dist_facts(predicted) ^ dist_facts(truth)))
        return truth


@dataclass
class Projection:
    """Corrected state is truth; records the repaired *fraction* per correction.

    The fraction is the pre-correction divergence ``d(ŝ', s')`` -- how much of the cluster the
    consultation had to fix. Lower means the model was nearly right (SPEC-2 §6.2 /
    "cheapest-faithful-horizon-per-correction").
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: DistributedState, truth: DistributedState) -> DistributedState:
        self.repaired_fractions.append(divergence(predicted, truth))
        return truth


@dataclass
class ReplicasOnlyCorrection:
    """Partial correction: snap durable ``replicas`` to truth, **keep the model's ``inflight``**.

    The distributed operator that breaks the v0 identity (SPEC-7 §8.3, ED3). A consult that corrects
    only the consistency view leaves the in-flight replication medium as the model predicted it --
    so when the model mispredicts an in-flight message (the ``subtle`` error class, §5) the error
    survives the correction and the coupled state keeps drifting. It corrects strictly less than
    ``HardReset``, so its faithful horizon is bounded above by it; the gap is exactly the horizon
    the in-flight medium would have bought. Records the same repaired-fraction diagnostic as
    ``Projection`` for the cost lens.
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: DistributedState, truth: DistributedState) -> DistributedState:
        self.repaired_fractions.append(divergence(predicted, truth))
        # durable replicas from truth; in-flight medium (and the metadata riding with it) trusted
        # from the model -- the partial correction that declines to fix the stale-read source.
        return replace(truth, inflight=dict(predicted.inflight))
