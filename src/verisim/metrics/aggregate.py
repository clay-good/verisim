"""Aggregate run-records into ``H_ε(ρ)`` curve points with bootstrap CIs (SPEC-2 §9).

The headline result is the faithful horizon as a function of the oracle-consultation
budget, reported with confidence intervals over seeds (SPEC.md §7, SPEC-2 §9). This
module turns a flat list of :class:`RunRecord` (one per rollout) into one
:class:`CurvePoint` per ``(difficulty, ε, ρ)`` cell -- the data a figure is drawn
from. Figures are produced *only* from these records (SPEC-2 §7.3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean

from .record import RunRecord


def bootstrap_ci(
    values: list[float], *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Deterministic given ``seed``. With a single value the CI collapses to it; with
    none it is ``(nan, nan)``. Resamples the mean ``n_resamples`` times and returns
    the ``[alpha/2, 1-alpha/2]`` percentiles.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    lo = means[max(0, int((alpha / 2) * n_resamples))]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


@dataclass(frozen=True)
class GroupStat:
    """Bootstrapped mean of one numeric ``config`` field over a record group."""

    key: tuple[str, ...]
    mean: float
    ci_low: float
    ci_high: float
    n: int


def aggregate_values(
    records: list[RunRecord],
    *,
    group_keys: list[str],
    value: str,
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[GroupStat]:
    """Group records by ``config[group_keys]`` and bootstrap the mean of ``config[value]``.

    A generic companion to :func:`aggregate_curve` for ablations (E4): the grouped
    quantity is an arbitrary numeric ``config`` field (e.g. ``"step_accuracy"``),
    not the faithful horizon. Points are returned sorted by their key tuple.
    """
    groups: dict[tuple[str, ...], list[float]] = {}
    for record in records:
        key = tuple(str(record.config[k]) for k in group_keys)
        groups.setdefault(key, []).append(float(record.config[value]))

    stats: list[GroupStat] = []
    for key, values in sorted(groups.items()):
        lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        stats.append(GroupStat(key=key, mean=fmean(values), ci_low=lo, ci_high=hi, n=len(values)))
    return stats


@dataclass(frozen=True)
class CurvePoint:
    difficulty: str
    epsilon: float
    rho: float
    mean_h: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "epsilon": self.epsilon,
            "rho": self.rho,
            "mean_h": self.mean_h,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
        }


@dataclass(frozen=True)
class ComparisonPoint:
    """A single arm of an equal-budget comparison (E2 policies / E3 operators).

    ``label`` is the compared variant (policy or operator name); ``mean_calls`` is
    the mean oracle consultations spent -- the budget the arms are compared *at*
    (SPEC-2 §9, "at equal ρ").
    """

    label: str
    epsilon: float
    mean_h: float
    ci_low: float
    ci_high: float
    mean_calls: float
    n: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "epsilon": self.epsilon,
            "mean_h": self.mean_h,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "mean_calls": self.mean_calls,
            "n": self.n,
        }


def aggregate_comparison(
    records: list[RunRecord], *, key: str, n_resamples: int = 1000, seed: int = 0
) -> list[ComparisonPoint]:
    """Group records by ``(config[key], ε)`` and summarize ``H_ε`` with a CI.

    Used for the equal-budget comparisons (E2 over consultation policies, E3 over
    correction operators): ``key`` is the config field naming the compared variant
    (``"policy"`` or ``"operator"``). Points are sorted by ``(label, ε)``.
    """
    horizons: dict[tuple[str, float], list[float]] = {}
    calls: dict[tuple[str, float], list[float]] = {}
    for record in records:
        group = (str(record.config[key]), record.epsilon)
        horizons.setdefault(group, []).append(float(record.faithful_horizon))
        calls.setdefault(group, []).append(float(record.oracle_calls))

    points: list[ComparisonPoint] = []
    for (label, epsilon), values in sorted(horizons.items()):
        lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        points.append(
            ComparisonPoint(
                label=label,
                epsilon=epsilon,
                mean_h=fmean(values),
                ci_low=lo,
                ci_high=hi,
                mean_calls=fmean(calls[(label, epsilon)]),
                n=len(values),
            )
        )
    return points


def aggregate_curve(
    records: list[RunRecord], *, n_resamples: int = 1000, seed: int = 0
) -> list[CurvePoint]:
    """Group records by ``(difficulty, ε, ρ)`` and summarize ``H_ε`` with a CI.

    Each record's ``config`` must carry ``difficulty`` and ``rho``; ``ε`` is the
    record's ``epsilon``. Points are returned sorted by ``(difficulty, ε, ρ)``.
    """
    groups: dict[tuple[str, float, float], list[float]] = {}
    for record in records:
        key = (str(record.config["difficulty"]), record.epsilon, float(record.config["rho"]))
        groups.setdefault(key, []).append(float(record.faithful_horizon))

    points: list[CurvePoint] = []
    for (difficulty, epsilon, rho), values in sorted(groups.items()):
        lo, hi = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        points.append(
            CurvePoint(
                difficulty=difficulty,
                epsilon=epsilon,
                rho=rho,
                mean_h=fmean(values),
                ci_low=lo,
                ci_high=hi,
                n=len(values),
            )
        )
    return points
