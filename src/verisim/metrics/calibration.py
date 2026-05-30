"""Uncertainty calibration (SPEC-2 §7.2, §17.2).

The ``uncertainty``/``drift``-triggered consultation policies (§6.1) spend the
oracle budget on the steps the model is *least sure* about, keyed off the mean
entropy of the constrained decode (the model's confidence). That only helps if the
confidence is **calibrated** -- if higher entropy actually predicts higher per-step
divergence. E2 (M7) found the triggered policies losing to even-spacing, which is
exactly the symptom of an *un*calibrated signal; this module is the diagnostic that
measures it directly.

Given a set of per-step ``(signal, divergence)`` pairs (collected teacher-forced, so
each pair is one step's confidence vs. that step's actual error, uncompounded), it
reports:

  - **Pearson** and **Spearman** correlation between signal and divergence -- the
    headline calibration numbers. A useful signal has a strong positive correlation;
    ``≈ 0`` means the entropy carries no information about where the model errs, so no
    threshold policy built on it can beat a fixed schedule.
  - A **reliability table**: divergence binned by signal, so the relationship can be
    plotted (a well-calibrated signal gives a monotonically rising curve).

Dependency-free (pure Python): no numpy, consistent with the rest of ``metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

Pair = tuple[float, float]


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of ``xs`` and ``ys``; ``0.0`` if undefined (n<2 or no spread)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    return float(sxy / (sxx**0.5 * syy**0.5))


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks (ties share the average rank), 0-indexed."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation -- robust to the signal's arbitrary scale."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(_ranks(xs), _ranks(ys))


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    mean_signal: float
    mean_divergence: float
    n: int


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    pearson: float
    spearman: float
    mean_signal: float
    mean_divergence: float
    bins: list[ReliabilityBin]


def calibration_report(pairs: list[Pair], *, n_bins: int = 10) -> CalibrationReport:
    """Summarize per-step ``(signal, divergence)`` pairs (see the module docstring).

    Bins are equal-width over the observed signal range; empty bins are dropped. With
    fewer than two pairs (or no signal spread) the correlations are ``0.0`` and the
    single bin holds all the data.
    """
    if not pairs:
        return CalibrationReport(0, 0.0, 0.0, 0.0, 0.0, [])
    signals = [s for s, _ in pairs]
    divergences = [d for _, d in pairs]
    lo, hi = min(signals), max(signals)

    buckets: list[list[Pair]] = [[] for _ in range(n_bins)]
    width = (hi - lo) / n_bins if hi > lo else 0.0
    for s, d in pairs:
        idx = 0 if width == 0.0 else min(n_bins - 1, int((s - lo) / width))
        buckets[idx].append((s, d))

    bins: list[ReliabilityBin] = []
    for b, bucket in enumerate(buckets):
        if not bucket:
            continue
        bins.append(
            ReliabilityBin(
                lo=lo + b * width,
                hi=lo + (b + 1) * width if width else hi,
                mean_signal=fmean(s for s, _ in bucket),
                mean_divergence=fmean(d for _, d in bucket),
                n=len(bucket),
            )
        )

    return CalibrationReport(
        n=len(pairs),
        pearson=pearson(signals, divergences),
        spearman=spearman(signals, divergences),
        mean_signal=fmean(signals),
        mean_divergence=fmean(divergences),
        bins=bins,
    )
