"""Speculative rollout: draft-verify-accept-prefix as the consultation policy (SPEC-13 §3).

The program's propose-verify-correct loop, read at the right altitude, *is* speculative decoding
lifted from tokens to world-states (SPEC-13 §0). The cheap learned ``M_θ`` drafts ``k`` steps ahead;
the oracle verifies the draft and **accepts the longest faithful prefix** -- the same
accept-longest-correct-prefix rule that makes LLM speculative decoding exact-by-construction -- then
re-anchors at the first divergence. The one verisim twist: the verifier is an **exact deterministic
oracle**, so "longest correct prefix" is correct against ground truth, not against a larger model's
distribution (SPEC-13 §0, §10).

This module is world-generic: it speaks only in callables (``draft``, ``oracle_step``, ``diverge``),
so the same primitive drives the SPEC-5 network, SPEC-6 host, and SPEC-2.1 filesystem worlds via the
thin per-world bundles in :mod:`verisim.experiments.sr_common`. It adds almost no new primitive over
the shipped loop: the accepted-prefix length is exactly
:func:`verisim.metrics.horizon.faithful_horizon`
of the draft window (SPEC-13 §7).

Two honest cost accountings ride on every rollout (SPEC-13 §8):

  - **oracle calls** -- one *verify event* (decision point) per window. The speedup figure of merit
    (SPEC-13 §3 commitment 2) is *accepted faithful steps per oracle call*, ``E[a]``.
  - **verify steps vs corrections** -- the *true* compute. A window verify walks the oracle forward
    only to the first divergence (``a + 1`` steps), never the rejected suffix; a *correction* (an
    expensive full snap) happens once per window that rejects. The speculative win over fixed-``ρ``
    is real precisely when a **cheap verify** is cheaper than a **full produce** -- the
    control-plane
    reachability projection vs the data-plane oracle (SPEC-13 §2, §8) -- which the experiments
    measure
    as a cost-ratio sweep rather than assume.

The verify uses ``oracle_step`` for both detection and the snap; the demonstration's cheap
projection
is assumed *faithful* (cheap-divergence ≤ ε ⟹ full-divergence ≤ ε, the SPEC-12 H32 coarse-projection
property), so a single ``oracle_step`` suffices and the cost ratio is applied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from verisim.metrics.horizon import faithful_horizon

S = TypeVar("S")  # a world state
A = TypeVar("A")  # a world action

# A drafter free-runs the model one step: (state, action, step_index, variant) -> predicted state.
# ``variant`` lets a tree of independent drafts (SR3) be sampled from one anchor; single-draft
# rollouts pass variant 0 throughout.
Drafter = Callable[[S, A, int, int], S]
OracleStep = Callable[[S, A], S]
Diverge = Callable[[S, S], float]


def accepted_prefix_law(alpha: float, k: int) -> float:
    """The speculative-speedup law ``E[a] = α(1 - α^k) / (1 - α) = Σ α^i`` (SPEC-13 §3, H40).

    Expected accepted-prefix length of a length-``k`` draft window when each step is faithful i.i.d.
    with probability ``alpha`` (so the prefix is geometric, truncated at ``k``). This counts the
    *accepted free-run steps* (``a`` in :func:`speculative_rollout`), bounded by ``k`` -- not the
    Leviathan ``(1-α^{k+1})/(1-α)`` form, which adds the +1 bonus/corrected step and ranges to
    ``k+1``. ``alpha = 1`` accepts the whole window (``k``); ``alpha = 0`` accepts nothing (``0``).
    This is the i.i.d. baseline SR2's *empirical* prefix is measured against -- the
    position-dependence
    of acceptance is the residual, not assumed away (SPEC-13 §8).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if alpha >= 1.0:
        return float(k)
    return alpha * (1.0 - alpha**k) / (1.0 - alpha)


def free_run_divergences(
    s0: S,
    actions: Sequence[A],
    draft: Drafter[S, A],
    oracle_step: OracleStep[S, A],
    diverge: Diverge[S],
    *,
    variant: int = 0,
    start: int = 0,
) -> list[float]:
    """Per-step divergence of a free-running draft vs the oracle truth from ``s0`` (SR2's α(t)).

    No correction: the draft compounds on its own predicted states while the oracle compounds on the
    truth, so divergence ``[d_0, d_1, ...]`` is the unguided drift profile whose first ε-exceedance
    is
    the free-running faithful horizon. ``start`` offsets the step index handed to ``draft`` (for
    deterministic seeding when this is one window of a longer rollout).
    """
    drafted = s0
    truth = s0
    divergences: list[float] = []
    for j, action in enumerate(actions):
        drafted = draft(drafted, action, start + j, variant)
        truth = oracle_step(truth, action)
        divergences.append(diverge(truth, drafted))
    return divergences


@dataclass(frozen=True)
class SpeculativeRecord(Generic[S]):
    """One speculative rollout's accounting (SPEC-13 §3).

    With an unbounded correction budget every produced step is faithful **by construction**
    (accepted
    steps are within ``ε``; corrected steps are snapped to truth). Under a finite
    ``max_corrections``
    budget, once the budget is spent the rollout free-runs and drift accumulates -- so
    ``faithful_steps`` (steps within ``ε``) becomes the proper ``H_ε(ρ)`` quantity to compare
    against
    fixed-``ρ`` at equal expensive budget (SPEC-13 §6, SR1). ``divergences`` is the whole-rollout
    per-step trajectory those counts are taken from.
    """

    accepted_prefixes: tuple[int, ...]  # accepted free-run length ``a`` per window
    window_lengths: tuple[int, ...]  # drafted length ``k`` per window (shorter near the tail)
    oracle_calls: int  # verify events (decision points) -- the headline cost denominator
    verify_steps: int  # cheap oracle steps spent detecting divergence (Σ min(a+1, k))
    corrections: int  # expensive full snaps (one per window that rejected, capped at budget)
    total_steps: int  # steps advanced over the whole rollout (== len(actions))
    faithful_free_steps: int  # Σ accepted_prefixes -- free-run steps that held (the win)
    faithful_steps: int  # steps within ε over the whole rollout (== total_steps when unbounded)
    divergences: tuple[float, ...]  # per-step divergence over the whole rollout

    @property
    def steps_per_call(self) -> float:
        """Accepted free-run faithful steps per oracle call -- the speculative speedup ``E[a]``."""
        return self.faithful_free_steps / self.oracle_calls if self.oracle_calls else 0.0

    def cost(self, cheap_ratio: float) -> float:
        """Two-tier oracle cost: ``cheap_ratio * verify_steps + corrections`` (SPEC-13 §8).

        ``cheap_ratio = 1`` is the conservative case (verify is as expensive as produce -- no cheap
        projection, the speculative win collapses); ``cheap_ratio -> 0`` is the cheap-reachability-
        projection regime where the win lives. The crossover is the *measured condition*, not an
        assumption.
        """
        return cheap_ratio * self.verify_steps + self.corrections


def speculative_rollout(
    s0: S,
    actions: Sequence[A],
    draft: Drafter[S, A],
    oracle_step: OracleStep[S, A],
    diverge: Diverge[S],
    *,
    k: int,
    epsilon: float,
    n_drafts: int = 1,
    max_corrections: int | None = None,
    k_of: Callable[[S, int], int] | None = None,
) -> SpeculativeRecord[S]:
    """Draft ``k``, verify against the oracle, accept the longest faithful prefix, re-anchor.

    One window from anchor ``s`` over the next ``k`` actions:

      1. **draft** -- free-run the model ``k`` steps (``n_drafts`` independent variants for the SR3
         tree; the longest-faithful branch wins, the SpecInfer move with the oracle as the single
         verifier);
      2. **verify** -- walk the oracle forward, comparing each draft step to truth, stopping at the
         first divergence (so the rejected suffix costs no oracle work, ``verify_steps = a + 1``);
      3. **accept + re-anchor** -- credit the accepted prefix ``a``; if short of the window, snap
         to the oracle's true state at step ``a`` (one ``correction``) and re-anchor there; advance.

    With ``max_corrections is None`` every produced step is faithful by construction. With a finite
    budget, once it is exhausted the rollout *free-runs* the rest (no more snaps) and divergence
    accumulates honestly -- this is what makes the ``H_ε(ρ)`` comparison against fixed-``ρ`` an
    equal-expensive-budget one (SR1).

    ``k_of(anchor, step)`` overrides the draft length per window -- the calibrated-``k`` policy of
    SR4
    (draft longer where the model is confident, the EAGLE-2 link). When ``None`` every window is the
    fixed ``k``. Returns the :class:`SpeculativeRecord`.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_drafts < 1:
        raise ValueError(f"n_drafts must be >= 1, got {n_drafts}")

    anchor = s0
    i = 0
    n = len(actions)
    accepted: list[int] = []
    windows: list[int] = []
    verify_steps = 0
    corrections = 0
    divergences: list[float] = []
    # ``anchor`` is the coupled state; ``truth`` is the parallel ground-truth rollout used to score
    # divergence over the *whole* trajectory (including any post-budget free-run tail).
    truth = s0

    while i < n:
        budget_left = max_corrections is None or corrections < max_corrections

        if not budget_left:
            # Budget spent: free-run a single step with no oracle snap; record honest drift.
            action = actions[i]
            anchor = draft(anchor, action, i, 0)
            truth = oracle_step(truth, action)
            divergences.append(diverge(truth, anchor))
            i += 1
            continue

        kk = max(1, k_of(anchor, i)) if k_of is not None else k
        window = actions[i : i + kk]
        w = len(window)
        windows.append(w)

        # Draft every variant from the anchor; keep the longest faithful prefix (best-of-m, SR3).
        best_a = -1
        best_divs: list[float] = []
        for v in range(n_drafts):
            divs = free_run_divergences(
                anchor, window, draft, oracle_step, diverge, variant=v, start=i
            )
            a = faithful_horizon(divs, epsilon)
            if a > best_a:
                best_a = a
                best_divs = divs
        a = best_a
        accepted.append(a)

        # Verify walks truth only to the first divergence (a accepted + 1 to detect the break).
        verify_steps += min(a + 1, w)

        if a < w:  # the window rejected: snap to truth at the break, advance one past it
            corrections += 1
            advance = a + 1
        else:  # whole window accepted: no correction, advance the full window
            advance = w

        # The accepted steps held within ε; the corrected step (if any) snaps to truth (divergence
        # 0).
        for j in range(advance):
            divergences.append(best_divs[j] if j < a else 0.0)

        # Re-anchor at the true state after ``advance`` actions (computed during verify), and keep
        # the
        # ground-truth rollout in lockstep so a later free-run tail is scored correctly.
        for action in window[:advance]:
            anchor = oracle_step(anchor, action)
            truth = oracle_step(truth, action)
        i += advance

    faithful_steps = sum(1 for d in divergences if d <= epsilon)
    return SpeculativeRecord(
        accepted_prefixes=tuple(accepted),
        window_lengths=tuple(windows),
        oracle_calls=len(windows),
        verify_steps=verify_steps,
        corrections=corrections,
        total_steps=n,
        faithful_free_steps=sum(accepted),
        faithful_steps=faithful_steps,
        divergences=tuple(divergences),
    )


@dataclass(frozen=True)
class FixedSegmentRecord:
    """A fixed-interval consultation rollout's accounting, the SR1 baseline (SPEC-13 §6).

    Consult every ``j`` steps; between consults the model free-runs and drift accumulates. Unlike
    speculative, the produced trajectory is *not* fully faithful -- steps past a segment's first
    ε-exceedance are unfaithful-but-uncorrected (the clock has not ticked yet). The contrast SR1
    measures is exactly this: consult-on-a-clock wastes calls on still-faithful steps and rides out
    drift past ε mid-segment, while consult-at-the-break does neither.
    """

    faithful_steps: int  # steps within ε over the whole rollout
    oracle_calls: int  # one per consult
    verify_steps: int  # full produce per consult (no cheap detection -- fixed cannot stop early)
    corrections: int  # == oracle_calls (every consult is a full snap)
    total_steps: int

    @property
    def steps_per_call(self) -> float:
        return self.faithful_steps / self.oracle_calls if self.oracle_calls else 0.0

    def cost(self, cheap_ratio: float) -> float:
        """Fixed pays a full produce at every consult -- no cheap verify to discount (SPEC-13 §8).
        """
        return cheap_ratio * self.verify_steps + self.corrections


def fixed_interval_rollout(
    s0: S,
    actions: Sequence[A],
    draft: Drafter[S, A],
    oracle_step: OracleStep[S, A],
    diverge: Diverge[S],
    *,
    interval: int,
    epsilon: float,
) -> FixedSegmentRecord:
    """Free-run, consulting (full snap) every ``interval`` steps -- the SR1 fixed-``ρ`` baseline.

    Counts a step as faithful iff the coupled state is within ``ε`` of truth *after* that step. A
    consult snaps the coupled state to the oracle's true state and resets drift. The model still
    drafts every step (its prediction is what drift is measured on between consults).
    """
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")

    coupled = s0
    truth = s0
    faithful = 0
    calls = 0
    for t, action in enumerate(actions):
        coupled = draft(coupled, action, t, 0)
        truth = oracle_step(truth, action)
        if (t + 1) % interval == 0:  # consult: full snap to truth, reset drift
            coupled = truth
            calls += 1
        if diverge(truth, coupled) <= epsilon:
            faithful += 1
    return FixedSegmentRecord(
        faithful_steps=faithful,
        oracle_calls=calls,
        verify_steps=calls,  # each consult is one full produce
        corrections=calls,
        total_steps=len(actions),
    )
