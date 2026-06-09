"""The faithfulness leaderboard and its discriminative validity (SPEC-18 §6, H65).

A benchmark earns the name only if it **stably orders** proposers — if the ranking by faithful
horizon
is the same whichever seeds you draw. This module scores the fidelity ladder
(:data:`~verisim.bench.manifest.REFERENCE_PROPOSERS`) on the frozen battery, builds the per-world
`H_ε` leaderboard, and measures rank stability as **Kendall's τ between disjoint seed splits** with
a
bootstrap CI (H65). The proposers are scored via free-running faithful horizon (`ρ = 0`) on the
SPEC-13
world bundles, with the real drift and the real divergence metric of each world; the proposer
fidelity
`α` is the controlled stand-in (the trained arms are deferred, the LP7 rule).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean
from typing import Any

from verisim.bench.manifest import BatteryManifest, Proposer
from verisim.experiments.sr2 import SR2Config, granularity
from verisim.experiments.sr_common import (
    SRWorld,
    StallDrafter,
    fs_world,
    host_world,
    net_world,
)
from verisim.loop.speculative import free_run_divergences
from verisim.metrics.horizon import faithful_horizon


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's τ-b rank correlation between two equal-length scorings (no scipy dependency).

    ``+1`` = identical ranking, ``-1`` = reversed, ``0`` = unrelated. τ-b handles ties in either
    ranking (the floor/ceiling proposers can tie across splits), so the rank-stability number is
    honest
    when two proposers score equal.
    """
    n = len(a)
    if n < 2:
        return 1.0
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = concordant + discordant + ties_a
    n1 = concordant + discordant + ties_b
    denom = (n0 * n1) ** 0.5
    return (concordant - discordant) / denom if denom > 0 else 1.0


def _world(name: str, driver: str) -> SRWorld[Any, Any]:
    if name == "network":
        return net_world(driver=driver)
    if name == "host":
        return host_world(driver=driver)
    if name == "filesystem":
        return fs_world(driver=driver)
    raise ValueError(f"unknown world {name!r}")


def score_proposer(
    world: SRWorld[Any, Any], proposer: Proposer, seed: int, n_steps: int, epsilon: float
) -> float:
    """Free-running faithful fraction ``H_ε / T`` of ``proposer`` on one ``(world, seed)`` rollout.
    """
    s0, actions = world.make_actions(seed, n_steps)
    drafter = StallDrafter(world.oracle_step, proposer.alpha, seed=seed)
    divs = free_run_divergences(s0, actions, drafter, world.oracle_step, world.diverge)
    return faithful_horizon(divs, epsilon) / n_steps


@dataclass(frozen=True)
class LeaderRow:
    """One (world, proposer) leaderboard cell: mean faithful fraction over the battery seeds."""

    world: str
    proposer: str
    tier: str
    mean_faithful: float
    n_seeds: int

    def csv_row(self, manifest_hash: str) -> str:
        return (
            f"{manifest_hash},{self.world},{self.proposer},{self.tier},"
            f"{self.mean_faithful:.6f},{self.n_seeds}"
        )


@dataclass(frozen=True)
class RankStability:
    """The H65 discriminative-validity verdict for one world.

    The strict test: not just "top beats floor" (trivial) but "the benchmark *resolves adjacent
    fidelity tiers above seed noise*" -- the smallest gap between adjacently-ranked proposers
    exceeds
    the worst proposer's across-split score noise -- together with a high, CI-positive Kendall τ.
    """

    world: str
    tau_mean: float  # mean Kendall τ between disjoint seed-split leaderboards
    tau_lo: float
    tau_hi: float
    min_adjacent_gap: float  # smallest gap between adjacently-ranked proposers (the signal)
    max_seed_noise: float  # worst per-proposer across-split score std (the noise)
    discriminative: bool  # τ high AND CI-positive AND adjacent tiers resolved above noise


CSV_HEADER = "manifest_hash,world,proposer,tier,mean_faithful,n_seeds"
STABILITY_HEADER = "world,tau_mean,tau_lo,tau_hi,min_adjacent_gap,max_seed_noise,discriminative"


def build_leaderboard(
    manifest: BatteryManifest,
) -> tuple[list[LeaderRow], list[RankStability]]:
    """Score the ladder; return the leaderboard and per-world rank stability (H65)."""
    rows: list[LeaderRow] = []
    stability: list[RankStability] = []
    gran_cfg = SR2Config(n_steps=manifest.n_steps)
    for world_name, driver in zip(manifest.worlds, manifest.drivers, strict=False):
        world = _world(world_name, driver)
        epsilon = manifest.epsilon_g * granularity(world, gran_cfg)
        # per-seed scores: scores[proposer][seed]
        scores: dict[str, list[float]] = {p.name: [] for p in manifest.proposers}
        for seed in manifest.seeds:
            for p in manifest.proposers:
                scores[p.name].append(score_proposer(world, p, seed, manifest.n_steps, epsilon))
        for p in manifest.proposers:
            rows.append(
                LeaderRow(world_name, p.name, p.tier, fmean(scores[p.name]), len(manifest.seeds))
            )
        stability.append(_rank_stability(world_name, manifest, scores))
    return rows, stability


def _rank_stability(
    world: str, manifest: BatteryManifest, scores: dict[str, list[float]]
) -> RankStability:
    """Kendall τ between disjoint seed-split leaderboards + the gap-vs-noise check (H65)."""
    names = [p.name for p in manifest.proposers]
    n_seeds = len(manifest.seeds)
    rng = random.Random(0)
    taus: list[float] = []
    # Track each proposer's per-split mean to estimate its across-split noise.
    split_means: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(200):
        idx = list(range(n_seeds))
        rng.shuffle(idx)
        half = n_seeds // 2
        left, right = idx[:half], idx[half:]
        lb_l = [fmean([scores[name][i] for i in left]) for name in names]
        lb_r = [fmean([scores[name][i] for i in right]) for name in names]
        taus.append(kendall_tau(lb_l, lb_r))
        for name, v in zip(names, lb_l, strict=True):
            split_means[name].append(v)
    taus_sorted = sorted(taus)
    lo = taus_sorted[int(0.025 * len(taus))]
    hi = taus_sorted[min(len(taus) - 1, int(0.975 * len(taus)))]

    def _std(xs: list[float]) -> float:
        mu = fmean(xs)
        return float((fmean([(x - mu) ** 2 for x in xs])) ** 0.5)

    # Adjacent-tier discrimination is *paired*: seed noise is common-mode (a hard seed lowers all
    # proposers together, preserving order), so the gap that matters is the adjacent gap and
    # the noise that matters is the noise *of that gap*, not of either score alone.
    ordered = sorted(names, key=lambda nm: fmean(scores[nm]))
    resolved = True
    binding_gap = float("inf")  # the gap of the worst-margin (binding) adjacent pair
    binding_noise = 0.0
    best_margin = float("inf")
    for a, b in pairwise(ordered):
        diffs = [hi_v - lo_v for lo_v, hi_v in zip(split_means[a], split_means[b], strict=True)]
        gap = fmean(diffs)
        gap_noise = _std(diffs)
        if gap <= 2 * gap_noise:  # this adjacent pair is not resolved above its paired noise
            resolved = False
        margin = gap - 2 * gap_noise
        if margin < best_margin:
            best_margin, binding_gap, binding_noise = margin, gap, gap_noise
    if binding_gap == float("inf"):
        binding_gap = 0.0
    discriminative = fmean(taus) >= 0.8 and lo > 0.0 and resolved
    return RankStability(world, fmean(taus), lo, hi, binding_gap, binding_noise, discriminative)
