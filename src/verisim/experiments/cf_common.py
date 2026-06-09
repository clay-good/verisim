"""Shared apparatus for the SPEC-15 conformal-consultation (CF) experiments.

The conformal calibrator is torch-free -- it operates on recorded ``(score, oracle-divergence)``
pairs
(SPEC-15 §8). The one dependency is a *score*: the trained ``M_θ``'s ``belief_var`` / decode-entropy
(GPU, deferred and ``skipif``-guarded, the LP7 rule). The committed CPU core supplies that score
with a
transparent **controlled signal** -- a noisy predictor of the step's true divergence whose
correlation
with it is a stated knob -- so the two arms SPEC-15 must contrast are reproduced honestly:

  - a **calibrated** signal (``corr ≈ 1``) stands in for ``belief_var`` -- high score ⇒ high d,
    so it *conformalizes* (a threshold that buys ρ);
  - an **uncalibrated** signal (``corr ≈ 0``) stands in for decode-entropy -- the score does not
    separate within-ε from breach steps, so any threshold is no better than fixed (the ED2-smart
    null).

The *drift and divergence are real* (the SPEC-13
:class:`~verisim.experiments.sr_common.StallDrafter`
free-running against the real oracle and the real divergence metric); only the score is a stand-in.
Two regimes: an **exchangeable pool** (teacher-forced single-step divergences -- the CF1/CF4 i.i.d.
calibration/test data) and **drift rollouts** (free-running, divergence compounding with depth --
the
CF2 non-exchangeable test). Everything is CPU-only, deterministic, seeded.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from verisim.experiments.sr_common import SRWorld, StallDrafter, net_world
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class ScoredStep:
    """One step's conformal record: the model score, the oracle divergence, and the rollout depth.
    """

    score: float
    divergence: float
    depth: int


def mean(values: Sequence[float]) -> float:
    return fmean(values) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    """The ``q``-quantile (linear interpolation) -- used to pick ``ε`` for a target breach rate."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(xs):
        return xs[-1]
    return xs[lo] * (1 - frac) + xs[lo + 1] * frac


def _noise(seed: int, *parts: int) -> float:
    digest = hashlib.sha256(repr((seed, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def make_score(divergence: float, ref: float, corr: float, seed: int, *parts: int) -> float:
    """A controlled score in ``[0, 1]`` correlated with ``divergence`` by ``corr`` (SPEC-15 §9).

    ``corr = 1`` -> the score is the normalized divergence (a perfect breach predictor, the
    calibrated
    ``belief_var`` stand-in); ``corr = 0`` -> the score is pure seeded noise (the uncalibrated
    decode-entropy stand-in). ``ref`` normalizes the divergence to a comparable scale.
    """
    d_norm = min(1.0, divergence / ref) if ref > 0 else 0.0
    return corr * d_norm + (1.0 - corr) * _noise(seed, *parts)


def exchangeable_pool(
    world: SRWorld[Any, Any],
    *,
    drafter_alpha: float,
    corr: float,
    n_rollouts: int,
    steps: int,
    seed: int,
    max_window: int = 12,
) -> list[ScoredStep]:
    """Free-run-from-a-fresh-anchor (score, divergence) pairs -- the exchangeable CF1/CF4 data.

    Each record free-runs a *random* number of steps ``L ∈ [1, max_window]`` from an independent
    anchor
    and records the divergence there, so divergences spread continuously (a single-step
    teacher-force
    is nearly binary). The draws are i.i.d. across records -- exchangeable, the conformal
    precondition.
    The score is the controlled signal at correlation ``corr``.
    """
    rng = random.Random(seed)
    raw: list[tuple[float, int]] = []  # (divergence, draw-id) before scoring
    draw = 0
    for r in range(n_rollouts):
        s0, actions = world.make_actions(seed + r, steps)
        drafter = StallDrafter(world.oracle_step, drafter_alpha, seed=seed + r)
        i = 0
        while i < len(actions):
            length = rng.randint(1, max_window)
            window = actions[i : i + length]
            coupled = s0_anchor = _advance_truth(world, s0, actions[:i])
            truth = s0_anchor
            for t, action in enumerate(window):
                coupled = drafter(coupled, action, i + t, 0)
                truth = world.oracle_step(truth, action)
            raw.append((world.diverge(truth, coupled), draw))
            draw += 1
            i += length
    ref = quantile([d for d, _ in raw], 0.95) or 1.0
    pool = [ScoredStep(make_score(d, ref, corr, seed, idx), d, depth=0) for d, idx in raw]
    rng.shuffle(pool)
    return pool


def _advance_truth(world: SRWorld[Any, Any], s0: Any, actions: Sequence[Any]) -> Any:
    """The true state after applying ``actions`` from ``s0`` (the fresh anchor for a window)."""
    state = s0
    for action in actions:
        state = world.oracle_step(state, action)
    return state


def drift_rollouts(
    world: SRWorld[Any, Any],
    *,
    drafter_alpha: float,
    corr: float,
    n_rollouts: int,
    steps: int,
    seed: int,
    ref: float,
    overconfidence: float = 0.6,
) -> list[list[ScoredStep]]:
    """Free-running rollouts -- divergence compounds with depth, breaking exchangeability (CF2).

    Each rollout is ordered by depth; ``ref`` is the score-normalizer from the calibration pool so
    the
    two regimes share a score scale. No correction: the coupled state drifts from truth, so later
    steps
    breach systematically more often. The exchangeability violation is *signal de-calibration*: as
    the
    state goes out of the calibration distribution the model gets **overconfident** (``belief_var``
    shrinks), modeled as a depth-proportional downward bias ``overconfidence·(t/steps)`` on the
    score.
    So a static threshold calibrated in-distribution increasingly *misses* deep breaches (high
    divergence, but the de-calibrated score has fallen below ``τ``) -- the H52 drift ACI must
    recover.
    """
    out: list[list[ScoredStep]] = []
    for r in range(n_rollouts):
        s0, actions = world.make_actions(10_000 + seed + r, steps)
        drafter = StallDrafter(world.oracle_step, drafter_alpha, seed=10_000 + seed + r)
        coupled = s0
        truth = s0
        rollout: list[ScoredStep] = []
        for t, action in enumerate(actions):
            coupled = drafter(coupled, action, t, 0)
            truth = world.oracle_step(truth, action)
            d = world.diverge(truth, coupled)
            base = make_score(d, ref, corr, seed, 10_000 + r, t)
            score = max(0.0, base - overconfidence * (t / steps))  # overconfidence under shift
            rollout.append(ScoredStep(score, d, depth=t))
        out.append(rollout)
    return out


def breaches(steps: Sequence[ScoredStep], epsilon: float) -> list[int]:
    """The breach indicator (``divergence > ε``) per step."""
    return [1 if s.divergence > epsilon else 0 for s in steps]


def default_world() -> SRWorld[Any, Any]:
    """The SPEC-5 network world -- the CF headline world (SPEC-15 §6)."""
    return net_world()


def faithful_depth(rollout: Sequence[ScoredStep], epsilon: float) -> int:
    """The free-running faithful horizon of a drift rollout (first depth past ``ε``)."""
    return faithful_horizon([s.divergence for s in rollout], epsilon)
