"""UA12 -- the operational detection characteristic: drift costs precision, not just recall (H92).

SPEC-20 UA8 (file-integrity) proved faithfulness is load-bearing for content-keyed control by the
**catch rate** (recall): a faithful predictor catches every corruption (1.000) while a drifted `M_θ`
catches 0.50-0.73, the gap widening with horizon. But UA8 scored a *budget-limited recall only* --
``|protected ∩ truly-corrupted| / min(budget, |truly-corrupted|)`` -- which is **structurally blind
to false alarms**: it caps the defender's flags at ``budget`` and counts only the hits, so a model
that flags the *wrong* files pays nothing for it. A real detector does not get a free budget; it
flags every file it predicts will be corrupted, and a SOC gates on its **false-alarm rate**, not
its recall. A drifting world model mis-predicts *which* files the workload writes, so its predicted
set differs from the truth in *both* directions -- it misses real corruptions (false negatives)
**and** flags files that are never touched (false positives). UA8 only measured the first.

This module scores the **full confusion matrix** over the (uncapped) predicted-vs-true written-file
set -- precision, recall, and F1 -- so the operational cost of drift is visible. The prediction
(H92): the faithful predictor holds precision = recall = 1.000 at every horizon, while the free
predictor's **precision collapses alongside its recall**, so the F1 gap is *wider* than the recall
gap UA8 reported -- and the deployability gap (a SOC needs high precision *and* recall) is the real
operational stake. The ρ-grounded predictor (re-anchor every ``round(1/ρ)`` steps) buys back *both*
at the cheap UA9 knee: the operating characteristic, not just the catch rate, restored sub-linearly.

The detector here flags **which files get corrupted** (the UA8 keyed dimension, ``written_files``);
keyed on the deeper ``(path, content)`` set the story only sharpens (a drifted model names the right
file yet predicts wrong content), but which-file is the direct operational completion of UA8. No
training in the core (the metric is a pure function of two rollouts); CPU-only, seeded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verisim.acd.host_integrity import (
    HostStepFn,
    grounded_rollout_writes,
    rollout_writes,
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.base import HostOracle


@dataclass(frozen=True)
class Detection:
    """One detector's confusion matrix over the written-file set + the derived rates."""

    tp: int  # files correctly flagged as corrupted
    fp: int  # files flagged but never corrupted (the false alarms UA8 could not see)
    fn: int  # corruptions the detector missed
    n_flagged: int  # tp + fp -- the detector's total alarm volume (its SOC load)

    @property
    def precision(self) -> float:
        """Of the files flagged, the fraction truly corrupted (1.0 if it flagged none)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        """Of the true corruptions, the fraction the detector caught (1.0 if there were none)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        """The harmonic mean of precision and recall -- the single deployability number."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def confusion(predicted: set[str], true: set[str]) -> Detection:
    """The confusion matrix of a predicted written-file set against the truth (no budget cap)."""
    return Detection(
        tp=len(predicted & true),
        fp=len(predicted - true),
        fn=len(true - predicted),
        n_flagged=len(predicted),
    )


def detection_scores(
    predictor: HostStepFn, true_step: HostStepFn, start: HostState,
    actions: Sequence[HostAction],
) -> Detection:
    """The full-confusion detection score: flag every file the predictor expects corrupted.

    Unlike UA8's :func:`~verisim.acd.host_integrity.predictive_defense_reward`, the detector flags
    its *entire* predicted set (no ``budget`` cap), so false positives are counted -- the
    operational metric. A faithful predictor flags exactly the truth (precision = recall = 1.0); a
    drifted one flags a set with symmetric difference from the truth in both directions.
    """
    predicted = rollout_writes(predictor, start, actions)
    true = rollout_writes(true_step, start, actions)
    return confusion(predicted, true)


def grounded_detection(
    model: object, oracle: HostOracle, start: HostState, actions: Sequence[HostAction], rho: float,
) -> tuple[Detection, int]:
    """The ρ-grounded detector (UA9's predictor): re-anchor every round(1/ρ) steps; score P/R/F1.

    ``ρ=1`` recovers the faithful detector (precision = recall = 1.0, ``|actions|`` calls); ``ρ=0``
    the free one. The interior is the cheap-but-faithful regime -- here measured on the *operating
    characteristic*, so the question is whether grounding buys back deployable precision *and*
    recall sub-linearly, not just the catch rate. Returns ``(Detection, oracle_calls)``.
    """
    predicted, true, calls = grounded_rollout_writes(model, oracle, start, actions, rho)
    return confusion(predicted, true), calls


def mean_detection(scores: Sequence[Detection]) -> dict[str, float]:
    """Mean precision / recall / F1 over a battery of workloads (the reported operating point)."""
    if not scores:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_flagged": 0.0}
    n = len(scores)
    return {
        "precision": sum(s.precision for s in scores) / n,
        "recall": sum(s.recall for s in scores) / n,
        "f1": sum(s.f1 for s in scores) / n,
        "n_flagged": sum(s.n_flagged for s in scores) / n,
    }
