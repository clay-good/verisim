"""SPEC-22 CU7 -- verify-before-commit (H99): where you verify beats how much you verify.

CU6 left a warning: a free agent that replans amplifies its own harm, because the retry loop
searches its gate's blind spots. CU5 answered "spend a budget where the model is uncertain." CU7
gives the *structural* answer, the one an agent builder can act on: **the harm only happens at the
moment of commit** -- the agent can only do the irreversible bad thing by *executing* a route, and
it only executes a route it believes safe. So verification is needed at exactly one place: the route
the agent is about to commit to. Verify the model's "yes"; trust its "no."

A **verify-before-commit** agent replans *freely* (cheap model-only search) to find the first route
it previews as safe, then spends **one oracle call** to verify that route before executing it. If
the oracle rejects it (a blind-spotted danger), the agent keeps searching from the next route --
free -- and verifies the next candidate. Two consequences:

  - **zero harm, by construction.** Every executed route was verified by the oracle, so no
    blind-spot danger is ever executed -- the missed-danger rate is **0**, not "small on average."
  - **near-minimal cost.** The agent pays only for the routes it *commits* to (the model's "yes"),
    and trusts the model's "no" for free -- because a "no" leads to an abort, and an abort can never
    cause harm. The wasted verifications a uniform "verify-everything" agent spends on routes it was
    going to skip anyway are exactly the cost CU7 removes.

The headline (H99): a verify-before-commit agent reaches the **zero-harm guarantee at a fraction of
a full-verification agent's oracle cost**, and the saving **grows with how adversarial the
environment is** (the more dangerous shortcuts a goal offers, the more "no" verifications a uniform
agent wastes and CU7 skips). Where you verify beats how much. This is the constructive resolution of
CU6: the same persistence that was a liability for a free agent becomes safe *and* cheap once
verification is placed at the commit point. CPU-only, torch-free (a controlled risk stand-in, the
trained arm deferred per the LP7 rule); danger labels grounded in the real
:class:`ReferenceOracle`. Deterministic, seeded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from verisim.acd.closed_loop_replan import (
    CU6Config,
    Route,
    _risk,
    build_goal,
)
from verisim.oracle.reference import ReferenceOracle

Schedule = Literal["uniform", "prioritized"]
Outcome = Literal["success", "harm", "lost"]


@dataclass(frozen=True)
class CU7Config:
    """The verify-where sweep: a goal with many dangerous shortcuts (the realistic cyber case)."""

    protected_prefix: str = "/a"
    work_prefix: str = "/w"
    n_goals: int = 400
    n_routes: int = 8  # more routes -> more dangerous shortcuts to skip-or-verify
    danger_fraction: float = 0.6  # an adversary-rich environment: most routes are dangerous
    phi: float = 0.6
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)  # the budgeted frontier
    schedule: Schedule = "prioritized"
    seed0: int = 5000

    def world(self) -> CU6Config:
        """The CU6 world this strategy is studied on (so goals are apples-to-apples with CU6)."""
        return CU6Config(
            protected_prefix=self.protected_prefix, work_prefix=self.work_prefix,
            n_goals=self.n_goals, n_routes=self.n_routes, danger_fraction=self.danger_fraction,
            phi=self.phi, rhos=self.rhos, schedule=self.schedule, seed0=self.seed0,
        )

    @staticmethod
    def smoke() -> CU7Config:
        return CU7Config(n_goals=80, n_routes=5, rhos=(0.0, 0.5, 1.0))


def _consult_set(config: CU7Config, goal: int, risks: list[float], rho: float) -> set[int]:
    k = round(rho * config.n_routes)
    if k <= 0:
        return set()
    idx = list(range(config.n_routes))
    if config.schedule == "uniform":
        return set(random.Random(config.seed0 * 7 + goal).sample(idx, k))
    return set(sorted(idx, key=lambda r: abs(risks[r] - 0.5))[:k])


def run_goal_budgeted(
    config: CU7Config, goal: int, routes: list[Route], rho: float,
) -> tuple[Outcome, int]:
    """The CU6 budgeted replanner: verify a ρ-budget of routes; stop at the first allowed; count.

    Calls are counted **honestly** -- only routes the agent actually walks (it stops early on the
    first allowed route) and that fall in the consult set cost an oracle query.
    """
    risks = [_risk(config.world(), goal, r, rt.dangerous) for r, rt in enumerate(routes)]
    consult = _consult_set(config, goal, risks, rho)
    calls = 0
    for r, rt in enumerate(routes):
        if r in consult:
            calls += 1
            allowed = not rt.dangerous  # the oracle's true verdict
        else:
            allowed = risks[r] < 0.5  # the model's verdict
        if allowed:
            return ("harm" if rt.dangerous else "success"), calls
    return "lost", calls


def run_goal_verify_before_commit(
    config: CU7Config, goal: int, routes: list[Route],
) -> tuple[Outcome, int]:
    """Replan freely; verify only the route the agent is about to commit to (the model's "yes")."""
    risks = [_risk(config.world(), goal, r, rt.dangerous) for r, rt in enumerate(routes)]
    calls = 0
    for r, rt in enumerate(routes):
        if risks[r] < 0.5:  # the model proposes this route (free search)
            calls += 1  # verify the commit candidate (one oracle call)
            if not rt.dangerous:  # the oracle confirms safe -> execute
                return "success", calls
            # the oracle rejects a blind-spotted danger -> abort, keep searching (no harm)
    return "lost", calls


@dataclass(frozen=True)
class CU7Cell:
    """One budgeted-replanner rung: its oracle cost and the safety/capability it buys."""

    rho: float
    calls: float  # mean oracle calls per goal (the honest cost)
    harm_rate: float
    success_rate: float


@dataclass(frozen=True)
class CU7Point:
    """A single strategy's operating point in (cost, harm, success) space."""

    label: str
    calls: float
    harm_rate: float
    success_rate: float
    wasted_calls: float = 0.0  # calls spent verifying routes the model already called dangerous


@dataclass(frozen=True)
class CU7Result:
    phi: float
    n_goals: int
    n_routes: int
    budgeted: list[CU7Cell]  # the ρ-swept frontier of the CU6 replanner
    full_verify: CU7Point  # the budgeted replanner at ρ=1 (verify everything walked)
    vbc: CU7Point  # verify-before-commit


def _budgeted_call_split(config: CU7Config, goal: int, routes: list[Route]) -> tuple[int, int]:
    """At ρ=1 (verify all walked): split calls into model-"yes" (needed) and model-"no" (wasted).

    A call on a route the model already flags dangerous is redundant for *safety* -- the agent would
    have aborted it anyway -- so it is the cost verify-before-commit removes.
    """
    risks = [_risk(config.world(), goal, r, rt.dangerous) for r, rt in enumerate(routes)]
    needed = wasted = 0
    for r, rt in enumerate(routes):
        # at ρ=1 every walked route is consulted; the oracle allows iff truly safe
        if risks[r] < 0.5:
            needed += 1
        else:
            wasted += 1
        if not rt.dangerous:  # the oracle allows the first truly-safe route -> the agent stops
            break
    return needed, wasted


def run_cu7(config: CU7Config | None = None) -> CU7Result:
    """Sweep the budgeted replanner's cost/safety frontier and place verify-before-commit on it."""
    config = config or CU7Config()
    ref = ReferenceOracle()
    goals = [build_goal(config.world(), g, ref) for g in range(config.n_goals)]

    budgeted: list[CU7Cell] = []
    for rho in config.rhos:
        outs = [run_goal_budgeted(config, g, routes, rho) for g, routes in enumerate(goals)]
        budgeted.append(CU7Cell(
            rho=rho,
            calls=fmean(c for _, c in outs),
            harm_rate=fmean(o == "harm" for o, _ in outs),
            success_rate=fmean(o == "success" for o, _ in outs),
        ))

    full = budgeted[-1]
    splits = [_budgeted_call_split(config, g, routes) for g, routes in enumerate(goals)]
    full_verify = CU7Point(
        label="full-verify (verify every route walked)", calls=full.calls,
        harm_rate=full.harm_rate, success_rate=full.success_rate,
        wasted_calls=fmean(w for _, w in splits),
    )

    vbc_outs = [run_goal_verify_before_commit(config, g, routes)
                for g, routes in enumerate(goals)]
    vbc = CU7Point(
        label="verify-before-commit", calls=fmean(c for _, c in vbc_outs),
        harm_rate=fmean(o == "harm" for o, _ in vbc_outs),
        success_rate=fmean(o == "success" for o, _ in vbc_outs),
    )

    return CU7Result(phi=config.phi, n_goals=config.n_goals, n_routes=config.n_routes,
                     budgeted=budgeted, full_verify=full_verify, vbc=vbc)


def cu7_verdict(result: CU7Result) -> dict[str, object]:
    """H99: verify-before-commit gets the zero-harm guarantee at a fraction of full-verify cost."""
    free = result.budgeted[0]
    vbc, full = result.vbc, result.full_verify
    return {
        # the free agent (no verification) is unsafe; both verified strategies reach zero harm
        "free_agent_unsafe": free.harm_rate > 0.05,
        "vbc_zero_harm": vbc.harm_rate <= 0.0,
        "full_verify_zero_harm": full.harm_rate <= 0.0,
        # H99 -- verify-before-commit reaches that guarantee strictly cheaper than full verification
        "vbc_cheaper_than_full_verify": vbc.calls < full.calls,
        "cost_saving_factor": full.calls / vbc.calls if vbc.calls > 0 else float("inf"),
        "wasted_fraction_of_full_verify": full.wasted_calls / full.calls if full.calls > 0 else 0.0,
        "vbc_calls": vbc.calls,
        "full_verify_calls": full.calls,
        "vbc_success": vbc.success_rate,
    }


CSV_HEADER = "row,label,rho,calls,harm_rate,success_rate,wasted_calls,phi,n_goals,n_routes"


def write_csv(result: CU7Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.budgeted:
        rows.append(f"budgeted,budgeted,{c.rho:.3f},{c.calls:.6f},{c.harm_rate:.6f},"
                    f"{c.success_rate:.6f},0.0,{result.phi:.3f},{result.n_goals},{result.n_routes}")
    for p in (result.full_verify, result.vbc):
        rows.append(f"point,{p.label},,{p.calls:.6f},{p.harm_rate:.6f},{p.success_rate:.6f},"
                    f"{p.wasted_calls:.6f},{result.phi:.3f},{result.n_goals},{result.n_routes}")
    out.write_text("\n".join(rows) + "\n")
    return str(out)
