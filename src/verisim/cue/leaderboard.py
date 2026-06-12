"""The verisim-cue leaderboard and its discriminative validity (SPEC-21 §8.2, H91 / CL1).

[`scorecard`](./scorecard.py) answers *for one model, which tasks still need the oracle*. This
module answers the question an **adopter** asks before trusting any of it: *does the benchmark
discriminate -- does it stably rank computer-use world-models by faithfulness, or is the ranking
seed noise?* It is the computer-use vertical's parallel of the SPEC-18 ``verisim-bench`` headline
([`bench.leaderboard`](../bench/leaderboard.py), H65), which SPEC-21 §8.2 positions verisim-cue to
have: the benchmark "packaged on the SPEC-18 verisim-bench line" earns the name only if its
scorecard *resolves adjacent fidelity tiers above seed noise*.

The measurement (CPU, torch-free, seconds):

  - **A fidelity ladder** of controlled host stand-ins (:data:`REFERENCE_CUE_PROPOSERS`): a floor
    (drifts on every content step) -> graded learned tiers -> the oracle ceiling. The fidelity
    ``alpha`` is the controlled per-step content faithfulness -- the trained ``M_θ`` arm is deferred
    (the repo's LP7 rule), exactly as ``verisim-bench``'s ladder defers it. ``alpha`` is faithful on
    *structure* (the process tree, the fd table -- the model never drifts there) and drifts on
    *content* (which files / what content) with probability ``1 - alpha``, the SPEC-20/21 model of
    where a host world-model's error lives.
  - **Score each proposer** through the ordered cue suite (:data:`~verisim.cue.tasks.TASK_SUITE`):
    its mean catch rate over tasks x seeds is its leaderboard cell, with the per-task breakdown.
  - **The discriminative-validity verdict (H91):** the strict SPEC-18 test, not the trivial
    "top beats floor". Kendall's tau between disjoint seed-split leaderboards (rank stability)
    **and** the binding adjacent-tier gap resolved above its *paired* across-split noise. The
    verdict is positive iff tau is high, its CI is positive, and every adjacent tier clears noise.

The bankable negative is first-class (SPEC.md §10.1): if the cue scorecard does *not* stably rank --
adjacent fidelity tiers collapse into seed noise -- then verisim-cue is not trustworthy as a frozen
eval, and that is a result that redirects the artifact, not a number to bury. Deterministic, seeded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from statistics import fmean

from verisim.acd.host_integrity import make_workload, oracle_step
from verisim.bench.leaderboard import kendall_tau  # the proven tau-b (no scipy), reused
from verisim.cue.tasks import TASK_SUITE, CueTask
from verisim.cue.tasks import keyed_defense_reward as cue_keyed_defense_reward
from verisim.env.state import File
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle

# --- the controlled fidelity ladder (the stand-in for M_θ; the trained arm deferred, LP7) ---------


@dataclass(frozen=True)
class CueProposer:
    """One rung of the fidelity ladder: a name, a tier, and the controlled content fidelity."""

    name: str
    tier: str  # "floor" | "learned-lo" | "learned-mid" | "learned-hi" | "ceiling"
    alpha: float  # controlled per-step *content* fidelity (alpha->1 = larger, less-drifting model)


#: The reference fidelity ladder (mirrors :data:`verisim.bench.manifest.REFERENCE_PROPOSERS`):
#: floor -> graded learned tiers -> the oracle ceiling. Faithful on structure at every rung.
REFERENCE_CUE_PROPOSERS: tuple[CueProposer, ...] = (
    CueProposer("null", "floor", 0.0),
    CueProposer("learned-lo", "learned-lo", 0.45),
    CueProposer("learned-mid", "learned-mid", 0.65),
    CueProposer("learned-hi", "learned-hi", 0.85),
    CueProposer("oracle-ceiling", "ceiling", 1.0),
)


class HostFidelityProposer:
    """A seeded, torch-free host ``M_θ`` stand-in: faithful on structure, drifts on content.

    A ``HostStepFn`` (``state, action -> state``) for the cue predictive-defense. On every
    ``write``/``append`` it is content-faithful with probability ``alpha`` and drifts otherwise --
    the SPEC-20/21 model: a more capable model (``alpha->1``) names the right file and writes the
    right content more often. On a drift it keeps *structure* (the process tree, the fd table, the
    file's *path*) but corrupts the *content* the action wrote: with probability ``blank_frac`` it
    blanks the file (so the *which-file* keyed set misses it too -- file-integrity drift), otherwise
    it writes a sentinel the v0 grammar can never produce (so only the *(path, content)* keyed set
    misses -- the deeper content-value drift). Content-value drift therefore dominates which-file
    drift at every ``alpha`` -- the structure->content gradient, by construction.

    Deterministic in ``(seed, step, action)`` via SHA-256 coins, so a proposer is identical across
    the workload seeds it is scored on; the only across-seed variation is the workload itself.
    """

    _SENTINEL = "\x00drift"  # content the v0 grammar (lowercase words) can never emit

    def __init__(self, alpha: float, seed: int, *, blank_frac: float = 0.4) -> None:
        self._oracle = ReferenceHostOracle()
        self._alpha = alpha
        self._seed = seed
        self._blank_frac = blank_frac
        self._step = 0

    def _coins(self, action: HostAction) -> tuple[float, float]:
        digest = hashlib.sha256(repr((self._seed, self._step, action.raw)).encode()).digest()
        capacity = int.from_bytes(digest[:8], "big") / 2.0**64
        blank = int.from_bytes(digest[8:16], "big") / 2.0**64
        return capacity, blank

    def step(self, state: HostState, action: HostAction) -> HostState:
        nxt = self._oracle.step(state, action).state
        capacity, blank = self._coins(action)
        self._step += 1
        if capacity < self._alpha or action.name not in ("write", "append"):
            return nxt  # content-faithful here (or a non-content action -- structure is exact)
        out = nxt.copy()
        before = state.fs.fs
        for path, node in nxt.fs.fs.items():
            content = getattr(node, "content", "")
            if content == "":
                continue
            prev = before.get(path)
            if getattr(prev, "content", "") == content:
                continue  # unchanged by this action -- leave it faithful
            mode = getattr(node, "mode", 0o644)
            corrupted = "" if blank < self._blank_frac else self._SENTINEL
            out.fs.fs[path] = File(content=corrupted, mode=mode)
        return out


# --- scoring the ladder through the cue suite -----------------------------------------------------


# The leaderboard scores *recall* over each task's keyed set -- a defense budget covering the whole
# true set, so catch = |predicted ∩ true| / |true|, a smooth fidelity score (unlike the scale-law's
# small fixed defense budget, which saturates to {0, 0.5, 1} per seed and collapses adjacent tiers).
# Recall is the natural "what fraction of the true dynamics did this model recover" number.
RECALL_BUDGET = 64


def _recall_suite(tasks: tuple[CueTask, ...]) -> tuple[CueTask, ...]:
    """The cue tasks re-budgeted to recall (budget >= |keyed set|) -- the leaderboard metric."""
    return tuple(dataclasses.replace(t, budget=RECALL_BUDGET) for t in tasks)


@dataclass(frozen=True)
class CueLeaderboardConfig:
    """The leaderboard sweep: the ladder, the battery seeds, and the workload regime."""

    proposers: tuple[CueProposer, ...] = REFERENCE_CUE_PROPOSERS
    seeds: tuple[int, ...] = tuple(range(700, 724))
    horizon: int = 20
    driver: str = "forky"
    threshold: float = 0.05  # gap above which the oracle is load-bearing (the scorecard contour)
    tasks: tuple[CueTask, ...] = field(default_factory=lambda: _recall_suite(TASK_SUITE))

    @staticmethod
    def smoke() -> CueLeaderboardConfig:
        return CueLeaderboardConfig(seeds=tuple(range(700, 720)), horizon=16)


@dataclass(frozen=True)
class CueLeaderRow:
    """One proposer's leaderboard cell: mean catch over the suite + the per-task breakdown."""

    proposer: str
    tier: str
    alpha: float
    mean_catch: float
    per_task: dict[str, float]
    n_seeds: int

    def csv_row(self, manifest_tag: str) -> str:
        tasks = ";".join(f"{k}={v:.4f}" for k, v in sorted(self.per_task.items()))
        return (
            f"{manifest_tag},{self.proposer},{self.tier},{self.alpha:.4f},"
            f"{self.mean_catch:.6f},{self.n_seeds},{tasks}"
        )


@dataclass(frozen=True)
class CueRankStability:
    """The H91 discriminative-validity verdict (the SPEC-18 ``RankStability`` parallel).

    Not the trivial "top beats floor" but the strict test: a high Kendall tau between disjoint
    seed-split leaderboards (rank stability), with a positive bootstrap-CI lower bound, *and* every
    adjacent fidelity tier resolved above its paired across-split noise (``min_adjacent_gap`` is the
    binding -- worst-margin -- adjacent gap; ``max_seed_noise`` is that gap's noise).
    """

    tau_mean: float
    tau_lo: float
    tau_hi: float
    min_adjacent_gap: float
    max_seed_noise: float
    discriminative: bool


def score_cue_proposer(
    proposer: CueProposer, seed: int, config: CueLeaderboardConfig,
    oracle: ReferenceHostOracle,
) -> dict[str, float]:
    """The per-task catch rate of one proposer on one workload seed (the scorecard, one seed)."""
    true_step = oracle_step(oracle)
    start, actions = make_workload(seed, config.horizon, driver=config.driver, oracle=oracle)
    out: dict[str, float] = {}
    for task in config.tasks:
        stand_in = HostFidelityProposer(proposer.alpha, seed)
        out[task.name] = cue_keyed_defense_reward(
            stand_in.step, true_step, start, actions, task.budget, task.key_fn
        )
    return out


def build_cue_leaderboard(
    config: CueLeaderboardConfig | None = None,
) -> tuple[list[CueLeaderRow], CueRankStability]:
    """Score the fidelity ladder through the cue suite; return the leaderboard + the H91 verdict."""
    config = config or CueLeaderboardConfig()
    oracle = ReferenceHostOracle()
    # scores[proposer][seed] = mean-over-tasks catch (the ranking metric); per_task accumulates too
    scores: dict[str, list[float]] = {p.name: [] for p in config.proposers}
    per_task: dict[str, dict[str, list[float]]] = {
        p.name: {t.name: [] for t in config.tasks} for p in config.proposers
    }
    for seed in config.seeds:
        for p in config.proposers:
            task_catch = score_cue_proposer(p, seed, config, oracle)
            scores[p.name].append(fmean(task_catch.values()))
            for tname, v in task_catch.items():
                per_task[p.name][tname].append(v)
    rows = [
        CueLeaderRow(
            proposer=p.name, tier=p.tier, alpha=p.alpha,
            mean_catch=fmean(scores[p.name]),
            per_task={t: fmean(vs) for t, vs in per_task[p.name].items()},
            n_seeds=len(config.seeds),
        )
        for p in config.proposers
    ]
    rows.sort(key=lambda r: r.mean_catch, reverse=True)
    stability = _rank_stability(config, scores)
    return rows, stability


def _rank_stability(
    config: CueLeaderboardConfig, scores: dict[str, list[float]]
) -> CueRankStability:
    """Kendall tau between disjoint seed-split leaderboards + the paired gap-vs-noise check (H91).

    Mirrors :func:`verisim.bench.leaderboard._rank_stability`: bootstrap disjoint seed halves, take
    Kendall tau between the two half-leaderboards, and -- because seed noise is common-mode (a hard
    workload lowers every proposer together, preserving order) -- resolve each *adjacent* tier by
    its *paired* gap (the mean of per-split gap differences) against that gap's own std, not either
    score's std alone.
    """
    names = [p.name for p in config.proposers]
    n_seeds = len(config.seeds)
    rng = random.Random(0)
    taus: list[float] = []
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

    ordered = sorted(names, key=lambda nm: fmean(scores[nm]))
    resolved = True
    binding_gap = float("inf")
    binding_noise = 0.0
    best_margin = float("inf")
    for a, b in pairwise(ordered):
        diffs = [hi_v - lo_v for lo_v, hi_v in zip(split_means[a], split_means[b], strict=True)]
        gap = fmean(diffs)
        gap_noise = _std(diffs)
        if gap <= 2 * gap_noise:
            resolved = False
        margin = gap - 2 * gap_noise
        if margin < best_margin:
            best_margin, binding_gap, binding_noise = margin, gap, gap_noise
    if binding_gap == float("inf"):
        binding_gap = 0.0
    discriminative = fmean(taus) >= 0.8 and lo > 0.0 and resolved
    return CueRankStability(fmean(taus), lo, hi, binding_gap, binding_noise, discriminative)


# --- output --------------------------------------------------------------------------------------

CSV_HEADER = "manifest,proposer,tier,alpha,mean_catch,n_seeds,per_task"


def write_csv(rows: list[CueLeaderRow], path: str | Path, manifest_tag: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER] + [r.csv_row(manifest_tag) for r in rows]
    out.write_text("\n".join(lines) + "\n")
    return out
